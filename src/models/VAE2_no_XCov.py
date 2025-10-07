# Import general packages
import logging
from typing import Dict, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import itertools
import torch
import torch.nn.functional as F
from ray import tune
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

# Import helper functions for data handling, logging and plotting
from utils.helper_utils import loss_proportions, update_performance_metrics
from utils.config_utils import Config_XCov_CV
from utils.data_utils import (
    combine_latent_spaces,
    process_latent_space,
    save_model,
    save_model_metrics,
)
from utils.logging_utils import (
    end_logging,
    log_and_print,
    log_checkpoint,
    log_early_stopping,
    log_early_stopping_th,
    log_extracting_latent_space,
    log_model_metrics,
    log_training_start,
    log_title
)
from utils.plotting_utils import (
    latent_space_batch_plot,
    latent_space_details_plot,
    latent_space_plot,
    metrics_plot,
)


# ====================================================================================================
#
# **VAE^2 model**
#
# The VAE^2 model takes as input a modular amount of atlas segementations for each patients. Several 
# "sub-VAEs" handle this data and compress each atlas into a single latent space. The latent spaces
# are aggregated to allow for a single classifier to predict the diagnostic label. The aggregated
# latent space is reconstructed into the input. After training the latent spaces are extracted and 
# plotted for all samples. Which plots are generated can be adjusted in VAE2_training.py  
#
# This is the version of the model WITH a cross-covariance penalty that penalizes cross covariance 
# between the latent space and a predicted covariate label.
#
# ====================================================================================================


# Initialize an encoder
def init_encoder(input_dim, hidden_dim, dropout_prob, latent_dim):
    encoder = nn.Sequential(
        #Layer 1 
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(1e-2),
        nn.Dropout1d(p=dropout_prob),
        #Layer 2 
        nn.Linear(hidden_dim, int(hidden_dim/2)),
        # nn.LeakyReLU(1e-2),
        # nn.Dropout1d(p=dropout_prob),
        # # Layer 3 
        # nn.Linear(hidden_dim, int(hidden_dim/2)),
        nn.LeakyReLU(1e-2),
        nn.Dropout1d(p=dropout_prob),
        nn.Linear(int(hidden_dim/2), latent_dim),
        #nn.Linear(hidden_dim, latent_dim)
        )
    return encoder 

# Initialize a decoder
def init_decoder(input_dim, hidden_dim, dropout_prob, latent_dim):
    decoder = nn.Sequential(
        nn.Linear(latent_dim, int(hidden_dim/2)),  # For cross covariance, pass the age information back into the model! 
        nn.LeakyReLU(1e-2),
        nn.Dropout1d(p=dropout_prob*2),
        # nn.Linear(hidden_dim, hidden_dim),
        # Layer 2
        nn.Linear(int(hidden_dim/2), hidden_dim),  # For cross covariance, pass the age information back into the model! 
        nn.LeakyReLU(1e-2),
        nn.Dropout1d(p=dropout_prob*2),
        nn.Linear(hidden_dim, input_dim),
        )
    return decoder


class VAE2_no_XCov(nn.Module):

    def __init__(
        self,
        # The number of classes (diagnostic groups) the model should predict
        num_classes: int,
        # The learning rate for the optimizer
        learning_rate: float,
        # The weight decay for the optimizer
        weight_decay: float,
        # The weight of the reconstruction loss
        recon_loss_weight: float,
        # The weight of the KL Divergence loss
        kldiv_loss_weight: float,
        # The weight of the diagnosis classifier loss
        class_loss_weight: float,
        # The weight of the cross covariance loss
        xcov_weight: float,
        # The weight of the age classifier
        xcov_cat_weight: float, 
        # The weight of the orthogonality loss
        ortho_weight: float,
        # The gradient scaler for mixed precision training
        scaler: GradScaler,
        # NEW! Size of input data
        atlas_input_dims: Dict,
        # Batch size
        batch_size: int = 64,
        # The dimension of the hidden layers in encoder and decoder
        hidden_dim: int = 100, 
        # The latent dimension of the model
        latent_dim: int = 20,
        # The gpu to train the model on
        device: torch.device = None,
        # Should the model use SSIM for the reconstruction loss
        # use_ssim: bool = False,
        # Should the learning rate be scheduled based on the validation loss (True) or training loss (False)
        schedule_on_validation: bool = True,
        # The patience for the learning rate scheduler
        scheduler_patience: int = 10,
        # The factor learning rate is reduced by when the scheduler is triggered
        scheduler_factor: float = 0.5,
        # Number of age cohorts (10 years) present in the data.
        num_xcov_groups: int = 6,
        # Is the model being used for fine tuning?
        fine_tuning: bool = False,
        # What is the weight of each of the classes in cross entropy?
        ce_class_weights: list = [1.0, 1.0, 1.0],
        dropout_prob: float = 0.3,
        xcov_covar: str = "Dataset"
    ):
        super(VAE2_no_XCov, self).__init__()  # should be possible to change this to super().__init__()
        
        # This model combines several VAEs which each take their own unique input, akin to multi-modal VAEs.
        # Each VAE produces a latent space. These latent spaces are concatenated to form a aggregated latent vector.
        # A classifier is attached to this aggregated latent space and predicts the diagnostic label.
        # Each VAE is expanded by a XCov module that reduces confounding batch effects.
        
        self.whole_model = nn.ModuleList()
        self.input_dims = [0]
        
        for atlas, atlas_input_dim in atlas_input_dims.items():

            self.input_dims.append(atlas_input_dim + self.input_dims[-1])

            latent_dim = int(atlas_input_dim**0.5)  # Scale latent dimension with the size / complexity of the input! 

            encoder = init_encoder(input_dim=atlas_input_dim, 
                                   hidden_dim=hidden_dim,
                                   dropout_prob=dropout_prob,
                                   latent_dim=latent_dim
                                   )
            
            decoder = init_decoder(input_dim=atlas_input_dim, 
                                   hidden_dim=hidden_dim,
                                   dropout_prob=dropout_prob,
                                   latent_dim=latent_dim,
                                   )
            

            fc_mu = nn.Sequential(nn.Linear(latent_dim, latent_dim),)
            
            fc_var = nn.Sequential(nn.Linear(latent_dim, latent_dim),)
            
            # For each input atlas we initialize one VAE model with an attached XCov classifier.
            sub_vae = nn.ModuleDict({"encoder": encoder, 
                                     "decoder": decoder, 
                                     "fc_mu": fc_mu,
                                     "fc_var": fc_var,}
                                     )
            
            sub_vae = sub_vae.float()

            self.whole_model.append(sub_vae)

        self.concat_dim = int(sum([int(input_dim**0.5) for atlas, input_dim in atlas_input_dims.items()]))
        
        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(self.concat_dim, int(20)),
            nn.LeakyReLU(1e-2),
            nn.Dropout1d(p=dropout_prob),
            # Output
            nn.Linear(int(20), num_classes),
        )

        # Set training parameters
        # Set optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Set learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )
        # Set gradient scaler
        self.scaler = scaler

        # Set loss weights
        self.recon_loss_weight = recon_loss_weight
        self.kldiv_loss_weight = kldiv_loss_weight
        self.class_loss_weight = class_loss_weight
        self.xcov_loss_weight = xcov_weight
        self.xcov_cat_weight = xcov_cat_weight
        self.ortho_loss_weight = ortho_weight
        self.batch_size = batch_size

        # set number of classes
        self.num_classes = num_classes
        self.num_xcov_groups = num_xcov_groups
        self.xcov_covar = xcov_covar

        # Record some values for logging
        self.schedule_on_validation = schedule_on_validation
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.atlas_input_dims = atlas_input_dims

        self.fine_tuning = fine_tuning
        
        # Send to Device
        if device is not None:
            self.device = device
            self.to(device)
        
        if ce_class_weights:
            self.ce_class_weights = torch.tensor(ce_class_weights).to(self.device)
        else:
            self.ce_class_weights = None

        # Initialize weights
        self.apply(self.weights_init)

    # The base forward function for the model
    def forward(
        self,
        # The input sample
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        vae_reconstructions = []

        vae_latent_spaces = []
        vae_mus = []
        vae_vars = []

        for idx, atlas_dim in enumerate(self.atlas_input_dims.values()):
            start_idx = self.input_dims[idx]
            end_idx = self.input_dims[idx+1]
            sub_x = x[:, start_idx:end_idx]

            sub_x = sub_x.float()

            current_vae = self.whole_model[idx]

            sub_x = current_vae["encoder"](sub_x)

            mu = current_vae["fc_mu"](sub_x)

            logvar = current_vae["fc_var"](sub_x)
            vae_mus.append(mu)
            vae_vars.append(logvar)

            z = self.reparameterize(mu, logvar)
            vae_latent_spaces.append(z)

            decoder_input = z  # In the Xcov counterpart, here the latent space would be concatenated with the predicted covariate

            sub_x = current_vae["decoder"](decoder_input)

            vae_reconstructions.append(sub_x)

        concat_latent_vector = torch.cat(vae_mus, dim=1)
        concat_latent_spaces_z = torch.cat(vae_latent_spaces, dim=1)

        class_logits = self.classifier(concat_latent_vector)

        return vae_reconstructions, vae_mus, vae_vars, class_logits,  concat_latent_spaces_z

    # This function is used to convert an image to the latent space.
    # This is useful for investigating the latent space the mode produces.
    def to_latent(self, x: torch.Tensor) -> torch.Tensor:

        vae_reconstructions = []
        vae_latent_spaces = []
        vae_mus = []
        vae_vars = []

        for idx, atlas_dim in enumerate(self.atlas_input_dims.values()):
            start_idx = self.input_dims[idx]
            end_idx = self.input_dims[idx+1]
            sub_x = x[:, start_idx:end_idx]

            current_vae = self.whole_model[idx]

            sub_x = current_vae["encoder"](sub_x)

            mu = current_vae["fc_mu"](sub_x)
            vae_mus.append(mu)

        concat_latent_vector = torch.cat(vae_mus, dim=1)
        class_logits = self.classifier(concat_latent_vector)

        # Return the the latent space and the classifier logits
        return concat_latent_vector, class_logits

    # This function performs reparameterization, which is used to sample from the latent space before decoding.
    def reparameterize(
        self,
        # The latent space produced by the encoder
        mu: torch.Tensor,
        # The log variance produced by the encoder
        logvar: torch.Tensor,
    ) -> torch.Tensor:

        #logvar = torch.clamp(logvar, min=-5, max=16) 
        # standard deviations
        std = torch.exp(0.5 * logvar)

        # random element between 0 and 1
        eps = torch.randn_like(std)

        # return a random sample from the latent space distribution
        return mu + eps * std

    # This function trains the model for one epoch.
    def train_one_epoch(
        self,
        # the training data loader
        train_loader: DataLoader,
        # the epoch number (used for logging)
        epoch: int,
    ) -> Dict[str, float]:

        # set model to train mode (important for batch normalization and dropout)
        self.train()

        # Initialize loss values, total loss is the sum of the class, reconstruction, and KL Divergence losses
        total_loss, class_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Go through each batch in the training dataset
        for batch_idx, (measurements, labels, names) in enumerate(train_loader):

            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)  # Shape = [batch_size, input_features]
            
            nas = torch.sum(torch.isnan(batch_measurements))
            if nas != 0:
                logging.info(f"Train Batch {batch_idx} contains NaNs")
            
            batch_labels = torch.stack([label for label in labels["Diagnosis"]]).to(self.device)
            #batch_xcov = torch.transpose(torch.stack([label for label in labels[self.xcov_covar]]), 0, 1).to(self.device)  # Shape = [64, 6]

            if self.fine_tuning:
                batch_labels = torch.roll(batch_labels, shifts=-1, dims=0)
                
            # Forward pass through model
            recon_measurements, mus, logvars, class_logits, latent_spaces = self(batch_measurements)

            batch_recon_measurements = torch.cat(recon_measurements, dim=1)

            # Calculate loss
            (
                b_total_loss,
                b_class_loss,
                b_recon_loss,
                b_kldiv_loss,
            ) = self.combined_loss_function(
                recon_measurements=batch_recon_measurements,
                target_measurements=batch_measurements,
                mus=mus,
                log_vars=logvars,
                class_logits=class_logits,
                labels=batch_labels,
                batch_latent_sample=mus  # Use mu as the latent space sample, as this is conceptually closest to the implementation from Cheung et al.
            )

            # Backward pass with gradient scaling
            self.scaler.scale(b_total_loss).backward()

            # Remove NaNs from gradients
            for param in self.parameters():
                if param.grad is not None and torch.any(torch.isnan(param.grad)):
                    param.grad.nan_to_num_(0.0)

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

            # Update weights w.r.t. loss
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # save total data loss stats
            total_loss += b_total_loss.item()
            class_loss += b_class_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

        # calculate loss averaged by the number of samples
        epoch_metrics = {
            "train_loss": total_loss / len(train_loader.dataset),
            "t_class_loss": class_loss / len(train_loader.dataset),
            "t_recon_loss": recon_loss / len(train_loader.dataset),
            "t_kldiv_loss": kldiv_loss / len(train_loader.dataset),
        }

        # loss proportions are a bit more interpretable for printing and logging
        epoch_props = loss_proportions("train_loss", epoch_metrics)

        # log and print loss proportions
        log_model_metrics(
            epoch,
            epoch_props,
            type="Training Metrics:",
        )

        # if we're not scheduling on validation, adjust learning rate based on training loss
        if not self.schedule_on_validation:
            # adjust learning rate based on training loss (may lead to overfitting)
            self.scheduler.step(total_loss / len(train_loader))

            # get current learning rate, log it
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)

            # save learning rate as a metric
            epoch_metrics["learning_rate"] = current_lr

        # return epoch losses (not proportions!)
        return epoch_metrics

    # This function validates the model on the validation dataset.
    # It produces important metrics for evaluating the model.
    # No gradients are calculated during validation.
    @torch.no_grad()
    def validate(
        self,
        # the validation data loader
        valid_loader: DataLoader,
        # the epoch number (used for logging)
        epoch: int,
    ) -> Dict[str, float]:

        # set model to evaluation mode, important for batch normalization and dropout
        self.eval()

        # Initialize loss values, total loss is the sum of the class, reconstruction, and KL Divergence losses
        total_loss, class_loss, recon_loss, kldiv_loss = 0.0, 0.0, 0.0, 0.0

        # Initialize lists to store classifier predictions and diagnostic labels
        all_preds = []
        all_labels = []

        # Go through each batch in the validation dataset
        for batch_idx, (measurements, labels, names) in enumerate(valid_loader):

            # Move samples to device
            batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)  # Shape = [batch_size, input_features]
            
            nas = torch.sum(torch.isnan(batch_measurements))
            if nas != 0:
                logging.info(f"Valid Batch {batch_idx} contains NaNs")
            
            batch_labels = torch.stack([label for label in labels["Diagnosis"]]).to(self.device)  # Shape = [num_classes, batch_size] ? 
            #batch_xcov = torch.transpose(torch.stack([label for label in labels[self.xcov_covar]]), 0, 1).to(self.device)  # Shape = [64, 6]
            batch_names = names

            if self.fine_tuning:
                batch_labels = torch.roll(batch_labels, shifts=-1, dims=0)

            # Forward pass
            recon_measurements, mus, logvars, class_logits, latent_spaces = self(batch_measurements)
            batch_recon_measurements = torch.cat(recon_measurements, dim=1)  # Shape = [64, 680]
            assert batch_recon_measurements.shape == batch_measurements.shape
            
            # Calculate loss
            (
                b_total_loss,
                b_class_loss,
                b_recon_loss,
                b_kldiv_loss,
            ) = self.combined_loss_function(
                recon_measurements=batch_recon_measurements,
                target_measurements=batch_measurements,
                mus=mus,
                log_vars=logvars,
                class_logits=class_logits,
                labels=batch_labels,
                batch_latent_sample=mus  # Use mu as the latent space sample, as this is conceptually closest to the implementation from Cheung et al.
            )

            # sum loss stats for this batch
            total_loss += b_total_loss.item()
            class_loss += b_class_loss.item()
            recon_loss += b_recon_loss.item()
            kldiv_loss += b_kldiv_loss.item()

            # Calculate Model Predictions and format diagnostic labels
            preds = torch.argmax(class_logits, dim=1)
            labels = torch.argmax(batch_labels, dim=0)


            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Caclulate confusion matrix metrics based on all predictions and labels
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=np.nan
        )

        # Store confusion matrix metrics
        epoch_confusion = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }

        # Calculate loss averaged by the number of samples
        epoch_losses = {
            "valid_loss": total_loss / len(valid_loader.dataset),
            "v_class_loss": class_loss / len(valid_loader.dataset),
            "v_recon_loss": recon_loss / len(valid_loader.dataset),
            "v_kldiv_loss": kldiv_loss / len(valid_loader.dataset),
        }

        # loss proportions are a bit more interpretable for printing and logging
        epoch_props = loss_proportions("valid_loss", epoch_losses)

        # log and print loss proportions as well as confusion matrix metrics
        log_model_metrics(
            epoch,
            {**epoch_confusion, **epoch_props},
            type="Validation Metrics:",
        )

        # collect all metrics (but not loss proportions)
        epoch_metrics = {**epoch_losses, **epoch_confusion}

        # if we're scheduling on validation, adjust learning rate based on validation loss
        if self.schedule_on_validation:
            # adjust learning rate based on validation loss (this is a form of regularization called early stopping, we stop training before overfitting occurs)
            self.scheduler.step(total_loss / len(valid_loader))

            # get current learning rate, log it
            current_lr = self.optimizer.param_groups[0]["lr"]
            logging.info("Current Learning Rate: %f", current_lr)

            # save learning rate as a metric
            epoch_metrics["learning_rate"] = current_lr

        # return all metrics
        return epoch_metrics

    # Function to extract latent embedding of data fed to the model.
    # This is useful for investigating the latent space the model produces.
    # Returns an AnnData object with the latent space and classifier predictions.
    # No gradients are calculated during this function.
    @torch.no_grad()
    def extract_latent_space(
        self,
        # The data loader to get the laten representation of
        data_loader: DataLoader,
        # A string to describe the data type (for logging)
        data_type: str = "Data",
    ) -> ad.AnnData:

        # log that we are extracting latent space
        log_extracting_latent_space(data_type)

        # any run in self.train will have dropout etc, will not be a real representation of the latent space
        self.eval()

        # Initialize lists to store latent spaces, sample names, and classifier predictions
        # Any other metadata can be added by the process_latent_space function instead.
        latent_spaces = []
        sample_names = []
        predictions = []

        # Go through each batch in the data loader
        for batch_idx, (measurements, labels, names) in enumerate(data_loader):
            # collect batch info
            batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

            # forward to latent space
            mu, class_logits = self.to_latent(batch_measurements)

            
            # collect running output info
            latent_spaces.append(mu.cpu().numpy())
            sample_names.extend(names)
            predictions.append(class_logits.argmax(dim=1).cpu().numpy())

        # create anndata object of latent space
        adata = ad.AnnData(np.concatenate(latent_spaces))

        # add metadata to anndata object, namely sample names and classifier predictions
        adata.obs_names = sample_names
        #adata.obs["Diagnosis_pred"] = pd.Categorical(np.concatenate(predictions))

        mapping = {0: "HC",
                   1: "MDD",
                   2: "SCHZ"}
        
        adata.obs["Diagnosis_pred"] = pd.Categorical(np.concatenate(predictions))
        adata.obs["Diagnosis_pred"] = adata.obs["Diagnosis_pred"].cat.rename_categories(mapping)

        # return anndata object
        return adata

    # Function to initialize weights of the model, using the Kaiming Uniform and Xavier Uniform initializations.
    # This is important for starting the model with good weights, which can help with training.
    def weights_init(self, param: nn.Module):

        # if the parameter is a convolutional layer, use Kaiming Uniform initialization
        if isinstance(param, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_uniform_(param.weight, nonlinearity="relu")
            if param.bias is not None:
                nn.init.constant_(param.bias, 0)

        # if the parameter is a linear layer, use Xavier Uniform initialization
        elif isinstance(param, nn.Linear):
            nn.init.xavier_uniform_(param.weight)
            nn.init.constant_(param.bias, 0)

    # Loss function to force orthogonality of latent space vectors. This should reduce colinearity!
    def ortho_loss_fn(self, latent_tensors):
        if len(latent_tensors) < 2:
            return torch.tensor(0.0, device=latent_tensors[0].device, requires_grad=True)
    
        total_ortho_loss = 0.0
        num_pairs = 0
        
        # Iterate through all unique pairs of latent spaces
        for i, j in itertools.combinations(range(len(latent_tensors)), 2):
            latent_i = latent_tensors[i]  # [batch_size, latent_dim_i]
            latent_j = latent_tensors[j]  # [batch_size, latent_dim_j]
            
            # Normalize the latent vectors to unit length for each sample
            latent_i_norm = F.normalize(latent_i)  # [batch_size, latent_dim_i]
            latent_j_norm = F.normalize(latent_j)  # [batch_size, latent_dim_j]
            
            # Use matrix multiplication to compute similarities between all pairs
            # This creates a [latent_dim_i, latent_dim_j] similarity matrix
            similarity_matrix = torch.mm(latent_i_norm.T, latent_j_norm)  # [latent_dim_i, latent_dim_j]
            
            # Orthogonality loss: minimize all pairwise similarities between dimensions
            # We want the similarity matrix to be close to zero everywhere, which is the case if two latent features are prefectly orthogonal!
            pair_ortho_loss = torch.mean(similarity_matrix ** 2)
            
            total_ortho_loss += pair_ortho_loss
            num_pairs += 1
        
        # Average across all pairs
        ortho_loss = total_ortho_loss / num_pairs
        
        return ortho_loss

    # Function to calculate the combined loss of the model.
    # This is the main loss function of the model, which is used to train the model.
    # The total loss is the sum of the classifier, reconstruction, KL Divergence, XCov and Orthogonality losses.
    # The total loss is used to update the model weights.
    def combined_loss_function(
        self,
        # The reconstructed measurements, after passing through the decoder
        recon_measurements: list[torch.Tensor],
        # The original 3D MRI image
        target_measurements: list[torch.Tensor],
        # The latent space mean, produced by the encoder
        mus: torch.Tensor,
        # The latent space log variance, produced by the encoder
        log_vars: torch.Tensor,
        # The classifier logits (1 per diagnostic group), produced by the classifier
        class_logits: torch.Tensor,
        # The diagnostic group labels
        labels: torch.Tensor,   
        # Latent space
        batch_latent_sample: torch.Tensor
    ):
        # The classification and reconstruction errors are calculated outside of the loop.

        vae_specific_recon_loss = 0
        vae_specific_kldiv_loss = 0

        # Calculate diagnosis classification loss and weight it
        target_for_loss = labels.argmax(dim=0).flatten() # flatten to bring from [N,1] to [N,] for cross entropy calculation. 

        if self.ce_class_weights:
            class_loss = F.cross_entropy(
                class_logits.float(), target_for_loss, reduction="mean", weight=self.ce_class_weights,
            )
        else:
            class_loss = F.cross_entropy(
                class_logits.float(), target_for_loss, reduction="mean",
            )
        class_loss = class_loss * self.class_loss_weight

        ortho_loss = self.ortho_loss_weight * self.ortho_loss_fn(mus)

        total_loss = class_loss + ortho_loss

        for i in range(len(self.atlas_input_dims.values())): 
            # Get the values specific to one of the sub-VAEs
            mu = mus[i]
            log_var = log_vars[i]

            start_idx = self.input_dims[i]
            end_idx = self.input_dims[i+1]

            recon_measurement = recon_measurements[:,start_idx:end_idx]
            target_measurement = target_measurements[:,start_idx:end_idx]

            recon_loss = F.mse_loss(recon_measurement, target_measurement, reduction="mean")
            recon_loss = recon_loss * self.recon_loss_weight

            # Calculate KLD loss and weight it
            kldiv_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            kldiv_loss = kldiv_loss * self.kldiv_loss_weight

            # Calculate the xcov classifiers loss and weight it
            vae_specific_recon_loss += recon_loss
            vae_specific_kldiv_loss += kldiv_loss

        total_loss += vae_specific_kldiv_loss + vae_specific_recon_loss

        # Return all losses
        return total_loss, class_loss, vae_specific_recon_loss, vae_specific_kldiv_loss


# This function trains the entire VAE2_no_XCov model.
def train_VAE2_no_XCov(
    # the ClassVAE model you want to train
    model: VAE2_no_XCov,
    # the training data loader
    train_loader: DataLoader,
    # the validation data loader
    valid_loader: DataLoader,
    # the annotations for the training and validation data
    annotations: pd.DataFrame,
    # the model metrics to store the performance metrics of the model
    # If you're loading a model, you can load the model metrics from the previous training run
    model_metrics: pd.DataFrame,
    # the configuration for the training run
    config: Config_XCov_CV,
    # How many epochs to wait for improvement before early stopping.
    early_stopping_threshold: int = 50,
    # whether to report the accuracy to Ray Tune for hyperparameter tuning
    accuracy_tune: bool = False,
    # whether to skip plotting of UMAPs. Plotting can be unstable with Ray Tune
    no_plotting: bool = True,
    # wether to skip plotting of training metrics. Plotting can be unstable with Ray Tune.
    no_val_plotting: bool = False,
    # wether to save at every checkpoint the respective model
    no_saving: bool = False
):
    # Print / log training start message
    log_training_start()
    best_model = model  # Save the best model to then use for latent space plotting at the end of training.
    early_stopping_count = 0
    best_accuracy = 0
    lowest_loss = 1000000

    # Train and Validate once per epoch
    for epoch in tqdm(
        range(config.START_EPOCH, config.FINAL_EPOCH),
        initial=config.START_EPOCH,
        ncols=100,
        desc="\nEpochs",
    ):
        # Separate log messages from different epochs in the terminal
        print("\n")

        # determine if it is a checkpoint epoch
        is_checkpoint = (
            epoch % config.CHECKPOINT_INTERVAL == 0 or epoch == config.FINAL_EPOCH - 1
        )

        # Train and Validate. We validate first because validation doesn't alter the model, training does.
        # So by doing it in this order, the loss generated in valid and train is based on the same model
        # Otherwise the comparison is off by one epoch, valid would be after the first correction.
        # Validate the model
        valid_metrics = model.validate(valid_loader=valid_loader, epoch=epoch)

        if epoch != config.START_EPOCH:
            #if valid_metrics["accuracy"] > best_accuracy:
            if valid_metrics["valid_loss"] < lowest_loss and epoch >= 25:
                best_accuracy = valid_metrics["accuracy"]
                lowest_loss = valid_metrics["valid_loss"]
                early_stopping_count = 0
                save_model(
                        model=model,
                        save_path=config.MODEL_DIR,
                        timestamp=config.TIMESTAMP,
                        descriptor=config.RUN_NAME,
                        epoch='best',
                    )
                best_model = copy.deepcopy(model)

            else:
                early_stopping_count += 1

        # train the model for one epoch
        train_metrics = model.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
        ) 

        # export information about the model
        # update model performance metrics
        model_metrics = update_performance_metrics(
            model_metrics, [train_metrics, valid_metrics]
        )

        # report accuracy to Ray Tune for hyperparameter tuning
        if accuracy_tune:
            tune.report({"accuracy": model_metrics["accuracy"].max()})

        # If we're going to stop training early, we need to do a checkpoint
        if config.EARLY_STOPPING:
            if early_stopping_count > early_stopping_threshold:
                is_checkpoint = True

        if config.EARLY_STOPPING:
            if model_metrics["learning_rate"].iloc[-1] <= config.STOP_LEARNING_RATE:
                is_checkpoint = True

        # save more detailed information at checkpoints
        if is_checkpoint:

            # log that we've done the checkpoint plots
            log_checkpoint(figure_path=config.FIGURES_DIR)

            # save model and performance metrics, but not if it's the first epoch (either useless or already saved)
            if epoch != config.START_EPOCH and not no_saving:
                save_model(
                    model=model,
                    save_path=config.MODEL_DIR,
                    timestamp=config.TIMESTAMP,
                    descriptor=config.RUN_NAME,
                    epoch=epoch,
                )

            # stop training early if learning rate is too low or if there hasn't been an improvement in a number of epochs.
            if config.EARLY_STOPPING:
                if model_metrics["learning_rate"].iloc[-1] <= config.STOP_LEARNING_RATE:

                    # log that we're stopping early
                    log_early_stopping(
                        model_metrics["learning_rate"].iloc[-1],
                        config.STOP_LEARNING_RATE,
                        epoch,
                    )

                    # break the training loop
                    break
            
            if config.EARLY_STOPPING:
                if early_stopping_count > early_stopping_threshold: 
                    # log that we're stopping early
                    log_early_stopping_th(
                        early_stopping_threshold,
                        epoch,
                    )

                    # break the training loop
                    break  

    if not no_val_plotting:
        logging.info(log_title("LATENT AND METRIC PLOTTING"))

        if not no_plotting:   
            # extract latent space and process it
            valid_latent = best_model.extract_latent_space(valid_loader, "Validation Data")
            train_latent = best_model.extract_latent_space(train_loader, "Training Data")

            # processing latent space means adding aligned annotations
            # as well as calculating PCA, Neighbors, and UMAP
            train_latent = process_latent_space(
                adata=train_latent,
                annotations=annotations,
                umap_neighbors=config.UMAP_NEIGHBORS,
                seed=config.SEED,
                save_data=True,
                save_path=config.DATA_DIR,
                timestamp=config.TIMESTAMP,
                epoch=epoch,
                data_type="train",
            )
            valid_latent = process_latent_space(
                adata=valid_latent,
                annotations=annotations,
                umap_neighbors=config.UMAP_NEIGHBORS,
                seed=config.SEED,
                save_data=True,
                save_path=config.DATA_DIR,
                timestamp=config.TIMESTAMP,
                epoch=epoch,
                data_type="valid",
            )
            # combine the validation and training latent spaces
            # this is useful for plotting the overall latent space and to see if
            # training is generalizing well (i.e. validation is similar to training)
            combi_latent = combine_latent_spaces(
                tdata=train_latent,
                vdata=valid_latent,
                umap_neighbors=config.UMAP_NEIGHBORS,
                seed=config.SEED,
                save_data=True,
                save_path=config.DATA_DIR,
                timestamp=config.TIMESTAMP,
                epoch=epoch,
                data_type="combined",
            )

            # plot latent space umap
            latent_space_plot(
                data=combi_latent,
                epoch=epoch,
                plot_types=["umap"],
                plot_by=[
                    "Diagnosis",
                    "Diagnosis_pred",
                    "Dataset",
                    "Sex",
                    "Age",
                ],
                data_type="combined",
                save_path=config.FIGURES_DIR,
                timestamp=config.TIMESTAMP,
                diagnosis_colors=config.PLOTTING_COLORS,
                save=True,
                show=False,
            )

            # plot latent space umap with detailed annotations
            # we plot the combined latent space, as it contains both training and validation data
            # but we also plot validation data solo, to see how well it generalizes
            # If there's only validation data, we use bigger dots as there's less data
            for data, data_type, descriptor, size in zip(
                [combi_latent, combi_latent, valid_latent],
                ["train", "valid", "valid"],
                ["", "", "_solo"],
                [
                    config.UMAP_DOT_SIZE,
                    config.UMAP_DOT_SIZE,
                    config.UMAP_DOT_SIZE * 3,
                ],
            ):
                latent_space_details_plot(
                    data=data,
                    epoch=epoch,
                    plot_by=[
                        "Diagnosis",
                        "Diagnosis_pred",
                        "Sex",
                        "Age",
                    ],  # Data Type is set, Dataset makes no sense
                    data_type=data_type,
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    save=True,
                    show=False,
                    descriptor=descriptor,
                    diagnosis_colors=config.PLOTTING_COLORS,
                    size=size,
                )

            # plot latent space batch plot. This shows off batch effect in individual diagnoses well
            # since it has less data, we use bigger dots
            for data, data_type, size in zip(
                [train_latent, valid_latent],
                ["train", "valid"],
                [config.UMAP_DOT_SIZE * 3, config.UMAP_DOT_SIZE * 9],
            ):
                latent_space_batch_plot(
                    data=data,
                    epoch=epoch,
                    data_type=data_type,
                    save_path=config.FIGURES_DIR,
                    timestamp=config.TIMESTAMP,
                    save=True,
                    show=False,
                    size=size,
                )

        metrics_plot(
            metrics=model_metrics,
            save_path=config.FIGURES_DIR,
            timestamp=config.TIMESTAMP,
            skip_first_n_epochs=config.DONT_PLOT_N_EPOCHS,
        )

        # save model performance metrics
        save_model_metrics(
            model_metrics=model_metrics,
            save_path=config.MODEL_DIR,
            timestamp=config.TIMESTAMP,
            descriptor=config.RUN_NAME,
        )

    # log the best accuracy and at which epoch it was achieved
    max_accuracy = model_metrics['accuracy'].max()
    best_epoch = model_metrics['accuracy'].idxmax()
    
    logging.info(f"Best Overall Validation Accuracy: {max_accuracy} at epoch {best_epoch}")
    logging.info(f"Validation Accuracy of Best Model: {best_accuracy}")

    # once the training loop is done, end logging
    #end_logging(config)
    return max_accuracy, best_model