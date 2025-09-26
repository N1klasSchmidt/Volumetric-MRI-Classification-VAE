# Import general packages
import torch
import numpy as np
import anndata as ad
import pandas as pd
from typing import Dict
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sklearn as sk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.cuda.amp import GradScaler
import logging

from models.aggr_VAE import NewVAE

from utils.data_utils import (process_subjects_XCov, 
                                      load_mri_data_NewVAE,
                                      split_annotations,
                                      split_subjects,
                                      process_latent_space_2D
                                      )

from utils.plotting_utils import (
    latent_space_plot_2D,
    prediction_confidence_plot
)

"""
This is the central code to run predictions on single models. For this a SinglePredictor class is initialized that handles prediction and evaluation with different metrics.

The EnsemblePredictor is a class that aggregates the predictions of several models and learns the optimal linear weighting of each model's predictions. 
This weighting is applied to the testing dataset. The optimal weights are computed through a Linear Least Squares (LLS) approach. 
"""

# ------------------------------------------------------------ Single Atlas Predictor ------------------------------------------------------------
class SinglePredictor():
    def __init__(
            self,
            # Device which should be used to run the models.
            device: torch.device,
            # Name of the atlas that was originally used to train the model
            atlas_name: str,
            # Dimension of the latent space encoding
            latent_dim: int,
            # Output for logging
            logging_config: Dict,
            # Parameters to make code adaptable
            model_config: Dict = None,
            # Path to where the desired model is saved.
            model_path: str = None,
            # Trained model
            model = None,
            # What model should be used?
            model_type: str = "xcov",
            # How are the diagnostic labels mapped.
            label_mapping: Dict = None,
            ):
        
        self.model_path = model_path
        self.model = model
        self.device = device
        self.dataloader = None
        self.atlas_name = atlas_name
        self.test_annotations = None
        self.latent_dim = latent_dim
        self.logging_config = logging_config
        self.model_config = model_config
        self.metadata_pred = None
        self.nss_df = None
        self.whitecat_df = None
        self.model_type = model_type
        self.label_mapping = label_mapping
        self.atlas_lengths = None

    # Initiliaze testing data on which the model will predict.
    def init_test_data(self, csv_paths, data_paths, scalers):
        annotations = pd.read_csv(csv_paths[0], header=[0])
        self.test_annotations = annotations[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]

        assert self.model_config.PURPOSE != "train"  # Ensuring that we scale the testing data! 

        subjects, _ , atlas_lengths = load_mri_data_NewVAE(
                csv_paths=csv_paths,
                atlas_data_paths=data_paths,
                scalers_per_atlas=scalers,
                diagnoses=self.model_config.DIAGNOSES,
                hdf5=False,
                train_or_test=self.model_config.PURPOSE,
                covars=self.model_config.COVARS,
                # How to scale and subset the data
                scaling_method="z",
                scale_per_dataset=True,
                vol_to_use=self.model_config.VOLUMES,
                all_labels=self.model_config.ALL_LABELS,
                batch_corrected=self.model_config.BATCH_CORRECTION
            )
        self.atlas_lengths = atlas_lengths
        self.dataloader = process_subjects_XCov(subjects=subjects, batch_size=64, shuffle_data=False, covars=[self.model_config.COVARS[0]])
        
        # Save the extended metadata of the NSS and WhiteCAT datasets to correct the true labels based on codiagnoses.
        nss_path = "/workspace/data/metadata/meta_data_NSS_all_variables.csv"
        whitecat_path = "/workspace/data/metadata/meta_data_whiteCAT_all_variables_bpfree.csv"
        new_whitecat_path = "/workspace/data/new_whitecat_features/new_whitecat_metadata/only_new_whiteCat_metadata_clean.csv"

        self.nss_df = pd.read_csv(nss_path)
        #self.whitecat_df = pd.concat([pd.read_csv(whitecat_path), pd.read_csv(new_whitecat_path)])
        self.whitecat_df = pd.read_csv(whitecat_path)

        return

    # Initialize models if they aren't being passed directly from memory
    def init_model(self):
        if self.model is None and self.model_path:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)

            single_model = NewVAE(
                                    # The number of classes (diagnostic groups) the model should predict
                                    num_classes=3,
                                    # The learning rate for the optimizer
                                    learning_rate=4e-5,
                                    # The weight decay for the optimizer
                                    weight_decay=4e-3,
                                    # The weight of the reconstruction loss
                                    recon_loss_weight=40.0,
                                    # The weight of the KL Divergence loss
                                    kldiv_loss_weight=4.0,
                                    # The weight of the classifier loss
                                    class_loss_weight=4.0,
                                    # XCov specific losses
                                    xcov_cat_weight=10.0,
                                    xcov_weight=10.0,
                                    # The gradient scaler for mixed precision training
                                    scaler=GradScaler,
                                    # The dimension of the input feature vector
                                    atlas_input_dims=self.atlas_lengths, 
                                    latent_dim=self.latent_dim,
                                    device=self.device,
                                    num_xcov_groups=6,
                                    hidden_dim=self.model_config.HIDDEN_DIM,
                                    xcov_covar=self.model_config.COVARS[0]
                                    )
                
            single_model.load_state_dict(state_dict=checkpoint)
            single_model.eval()
            single_model.to(self.device)
            self.model = single_model

        #log_pred_model_ready(model=self.model, atlas_name=self.atlas_name, model_path=self.model_path)
        return
    
    # This function extracts, processes, plots and saves the latent space of this model
    def extract_latent_space(self, dataloader):
            
        self.model.to(self.device)
        self.model.eval()

        model_preds = []
        model_probs = []
        latent_spaces = []
        sample_names = []
        xcov_predicted_labels = []

        # Go through each batch in the data loader
        with torch.no_grad():
            for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                # collect batch info
                batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

                # forward to latent space
                mu, class_logits, xcov_preds = self.model.to_latent(batch_measurements)

                # We select the majortiy predicted XCov label across each of the single VAEs
                xcov_pred_labels = torch.stack([xcov_pred.argmax(dim=1) for xcov_pred in xcov_preds], dim=0)  # Shape = [num_atlases, batch_size]
                xcov_majority_labels, _ = torch.mode(xcov_pred_labels, dim=0)  # Shape = [batch_size]
                
                # For alternative model structure with single latent space and xcov classifier
                # xcov_majority_labels = torch.argmax(xcov_preds, dim=1)

                # collect running output info
                latent_spaces.append(mu.cpu().numpy())
                sample_names.extend(names)
                model_preds.append(class_logits.argmax(dim=1).cpu().numpy())
                model_probs.append(torch.max(torch.softmax(class_logits, dim=1), dim=1).values.tolist())
                xcov_predicted_labels.append(xcov_majority_labels.cpu().numpy())

        adata = ad.AnnData(np.concatenate(latent_spaces))

        # add metadata to anndata object, namely sample names and classifier predictions
        adata.obs_names = sample_names
        adata.obs["Diagnosis_pred"] = pd.Categorical(np.concatenate(model_preds))
        adata.obs["XCov_pred"] = pd.Categorical(np.concatenate(xcov_predicted_labels))
        adata.obs["Confidence"] = np.concatenate(model_probs)

        mapping = {0: "HC",
                   1: "MDD",
                   2: "SCHZ"}
        
        adata.obs["Diagnosis_pred"] = adata.obs["Diagnosis_pred"].cat.rename_categories(mapping)

        self.latent = adata

        latent_df = self.latent.to_df()
        latent_df.to_csv(f"{self.logging_config.LOGGING_DIR}/latent_space_data.csv")

        latent = process_latent_space_2D(
            adata=self.latent,
            annotations=self.test_annotations,
            umap_neighbors=self.logging_config.UMAP_NEIGHBORS,
            seed=self.logging_config.SEED,
            atlas_name=self.atlas_name,
            save_data=True,
            save_path=self.logging_config.LOGGING_DIR,
            timestamp=self.logging_config.TIMESTAMP,
            epoch=999,  # Just a placeholder
            data_type="train",
        )

        latent_space_plot_2D(
            data=latent,
            epoch=999, # Just a placeholder
            plot_types=["umap", "pca"],
            plot_by=[
                "Diagnosis",
                "Diagnosis_pred",
                "Confidence",
                "Dataset",
                "Sex",
                "Age",
            ],
            data_type="combined",
            atlas_name=self.atlas_name,
            save_path=self.logging_config.LOGGING_DIR,
            timestamp=self.logging_config.TIMESTAMP,
            diagnosis_colors=self.model_config.PLOTTING_COLORS,
            save=True,
            show=False,
            size=80
        )
        return 


    # CTT patients should be counted as correclty diagnosed if the codiagnosis (either MDD or SCHZ) is correctly predicted.
    # This function "corrects" the ground truth lables, converting CTT diagnoses to the respective codiagnosis per patient.
    def correct_labels(self, labels, names):
        corrected_labels = torch.zeros(len(labels), 3)
        #corrected_labels = torch.zeros(labels.shape[0], 3)

        label_indices = labels #torch.argmax(labels, dim=1).cpu().tolist()

        for idx, label in enumerate(label_indices):
            
            if self.label_mapping[label] == "CTT": 
                sample_name = names[idx]

                # Check if this sample is from the NSS or the WhiteCAT dataset and extract the codiagnosis.
                if sample_name in self.nss_df["Filename"].tolist():
                    if "F2" in str(self.nss_df.loc[self.nss_df["Filename"] == sample_name, "Diagnosis_y"].iloc[0]): #.startswith("F2"): 
                        corrected_labels[idx] = torch.tensor([0, 0, 1])
                    if "F3" in str(self.nss_df.loc[self.nss_df["Filename"] == sample_name, "Diagnosis_y"].iloc[0]): #.startswith("F3"): 
                        corrected_labels[idx] = torch.tensor([0, 1, 0])

                elif sample_name in self.whitecat_df["Filename"].tolist():
                    if "F2" in str(self.whitecat_df.loc[self.whitecat_df["Filename"] == sample_name, "ana_icd10"].iloc[0]): # .startswith("F2"): 
                        corrected_labels[idx] = torch.tensor([0, 0, 1])
                    if "F3" in str(self.whitecat_df.loc[self.whitecat_df["Filename"] == sample_name, "ana_icd10"].iloc[0]): #.startswith("F3"): 
                        corrected_labels[idx] = torch.tensor([0, 1, 0])
            
            elif self.label_mapping[label] == "SCHZ":
                corrected_labels[idx] = torch.tensor([0, 0, 1])

            elif self.label_mapping[label] == "MDD":
                corrected_labels[idx] = torch.tensor([0, 1, 0])
            
            elif self.label_mapping[label] == "HC":
                corrected_labels[idx] = torch.tensor([1, 0, 0])  

        return corrected_labels


    # Let model predict on the loaded testing data
    def make_predictions(self, dataloader): 

        self.model.to(self.device)
        self.model.eval()

        all_preds = []
        all_labels = []
        all_names = []

        with torch.no_grad():
            for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                # Move samples to device
                batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)  # Shape: (batch_size, len_atlas)
                batch_labels = torch.stack([label for label in labels["Diagnosis"]]).to(self.device)  # Shape: (batch_size, num_classes)
                recon_measurements, mus, logvars, class_logits, xcov_preds, latent_spaces = self.model(batch_measurements)

                all_preds.append(class_logits.cpu())  # Shape: (num_batches, batch_size, num_classes)
                all_labels.extend(torch.argmax(batch_labels, dim=0).cpu().numpy())
                all_names.extend(names)  # Shape: (num_samples)
                
        all_preds = torch.cat(all_preds, dim=0)  # Shape: (num_samples, num_classes)
        #all_labels = torch.stack(all_labels, dim=0)   # Shape: (num_samples, num_classes)
                
        all_probs = torch.softmax(all_preds, dim=1).cpu().numpy()  # Shape: (num_samples, num_classes)
        all_prediction_labels = torch.argmax(torch.softmax(all_preds, dim=1), dim=1).cpu().numpy()

        all_labels_corr = self.correct_labels(labels=all_labels, names=all_names)
        
        probs_df = pd.DataFrame(all_probs, columns=["HC", "MDD", "SCHZ"])
        probs_df["Filename"] = all_names
        probs_df["Pred_labels"] = all_prediction_labels

        label_map = {0: "HC", 1: "MDD", 2: "SCHZ"}
        probs_df["Pred_class"] = probs_df["Pred_labels"].map(label_map)

        # Merge with test annotations to maintain same order
        metadata_pred = pd.merge(self.test_annotations, probs_df, on="Filename")
        metadata_pred.set_index("Filename", inplace=True, drop=True)

        # Save to CSV
        metadata_pred.to_csv(f"{self.logging_config.LOGGING_DIR}/testing_metadata_preds_{self.atlas_name}.csv")
        self.metadata_pred = metadata_pred

        return all_preds, all_labels, all_labels_corr, all_names
    

    # The proportion of incorrect diagnoses per variable can be investigated. The variables "Dataset" and "Diagnosis" are particularly interesting.
    def compute_incorrect_proportion(self, prediction_labels, target_labels, names, var="Dataset"):

        target_labels = torch.tensor(target_labels)

        non_matching_indices = torch.nonzero(prediction_labels != target_labels, as_tuple=True)[0]

        wrong_prediction_filenames = [names[i] for i in non_matching_indices]
        
        wrong_predictions_metadata = self.test_annotations[self.test_annotations['Filename'].isin(wrong_prediction_filenames)]
        wrong_predictions_per_var = wrong_predictions_metadata[var].value_counts()
        total_predictions_per_var = self.test_annotations[self.test_annotations['Filename'].isin(names)][var].value_counts()
        
        proportion_wrong_per_var = wrong_predictions_per_var / total_predictions_per_var

        logging.info(f"Proportion of wrong predictions per {var}: {proportion_wrong_per_var}")

        return 
    
    #Create a stacked bar plot with categorical data markers at the top and samples sorted by predicted class.
    def make_stacked_barplot(self):

        # Correct for possible fine tuning
        filenames_pred = set(self.metadata_pred[self.metadata_pred["Dataset"].isin(["whiteCAT", "NSS"])].index)
        subset_nss_df = self.nss_df[self.nss_df['Filename'].isin(filenames_pred)]
        subset_whitecat_df = self.whitecat_df[self.whitecat_df['Filename'].isin(filenames_pred)]

        subset_nss_df = subset_nss_df[["Filename", "Age", "Diagnosis", "Sex", "Dataset"]]
        subset_wc_df = subset_whitecat_df[["Filename", "Age", "Diagnosis", "Sex", "Dataset"]]

        merged_df = pd.concat([subset_nss_df, subset_wc_df], axis=0)  # Concatenate along the index = rows

        # Some of the diagnosis labels are incorrect in the NSS and whiteCAT metadata. We correct this!
        all_meta = pd.read_csv("/workspace/data/relevant_metadata/full_metadata_filtered.csv")
        diagnosis_map = all_meta[["Filename", "Diagnosis"]].copy()
        diagnosis_map.rename(columns={"Diagnosis": "correct_diagnosis"}, inplace=True)
        merged_df = merged_df.merge(diagnosis_map, on="Filename", how="left")
        merged_df["Diagnosis"] = merged_df["correct_diagnosis"].fillna(merged_df["Diagnosis"])
        merged_df.drop(columns=["correct_diagnosis"], inplace=True)

        # Initialize categorical columns with NaN values
        for var_name in ["Duration", "Codiagnosis", "Pred_labels", "Pred_class", "HC", "SCHZ", "MDD", "Male", "Female", "F20-F29", "F30-F39", "Confidence", "Uncertainty"]:
            merged_df[var_name] = np.nan

        merged_df['Codiagnosis'] = merged_df['Codiagnosis'].astype('string')
        merged_df['Pred_class'] = merged_df['Pred_class'].astype('string')
        #merged_df["Codiagnosis"] = merged_df["Codiagnosis"].astype(object)
        
        merged_df.set_index("Filename", drop=False, inplace=True)
        metadata_map = {}
        
        # Process NSS and whiteCAT data from data frame into per-patient dictionary
        for _, row in self.nss_df.iterrows():
            if "Filename" in self.nss_df.columns and row["Filename"] not in metadata_map:
                metadata_map[row["Filename"]] = {}
                
                # Add relevant fields
                if "Sex" in self.nss_df.columns:
                    metadata_map[row["Filename"]]["Sex"] = row["Sex"]
                if "Age" in self.nss_df.columns:
                    metadata_map[row["Filename"]]["Age"] = row["Age"]
                if "Diagnosis" in self.nss_df.columns:
                    metadata_map[row["Filename"]]["Diagnosis"] = row["Diagnosis"]
                if "Diagnosis_y" in self.nss_df.columns:
                    metadata_map[row["Filename"]]["Codiagnosis"] = str(row["Diagnosis_y"])
                if "DOI" in self.nss_df.columns:
                    metadata_map[row["Filename"]]["Duration"] = float(row["DOI"]) # DOI stands here for Duration of Illness, not Date of Illness
                    
        for _, row in self.whitecat_df.iterrows():
            if "Filename" in self.whitecat_df.columns and row["Filename"] not in metadata_map:
                metadata_map[row["Filename"]] = {}
                
                if "Sex" in self.whitecat_df.columns:
                    metadata_map[row["Filename"]]["Sex"] = row["Sex"]
                if "Age" in self.whitecat_df.columns:
                    metadata_map[row["Filename"]]["Age"] = row["Age"]
                if "Diagnosis" in self.whitecat_df.columns:
                    metadata_map[row["Filename"]]["Diagnosis"] = row["Diagnosis"]
                if "ana_icd10" in self.whitecat_df.columns:
                    metadata_map[row["Filename"]]["Codiagnosis"] = str(row["ana_icd10"])
                if "duration_illness" in self.whitecat_df.columns:
                    metadata_map[row["Filename"]]["Duration"] = row["duration_illness"]
        
        # Fill in metadata for each row in our merged dataframe
        for idx, row in merged_df.iterrows():
            filename = row["Filename"]
            
            if filename in metadata_map:
                meta = metadata_map[filename]

                # Set continuous variables
                if "Duration" in meta:
                    merged_df.loc[idx, "Duration"] = meta["Duration"]
                if "Codiagnosis" in meta:
                    merged_df.loc[idx, "Codiagnosis"] = meta["Codiagnosis"]

                merged_df.loc[idx, "Pred_labels"] = self.metadata_pred.loc[filename, "Pred_labels"]
                merged_df.loc[idx, "Pred_class"] = self.metadata_pred.loc[filename, "Pred_class"]
                merged_df.loc[idx, "SCHZ"] = self.metadata_pred.loc[filename, "SCHZ"]
                merged_df.loc[idx, "HC"] = self.metadata_pred.loc[filename, "HC"]
                merged_df.loc[idx, "MDD"] = self.metadata_pred.loc[filename, "MDD"]

                merged_df.loc[idx, "Male"] = 1 if merged_df.loc[idx, "Sex"] == "Male" else 0
                merged_df.loc[idx, "Female"] = 1 if merged_df.loc[idx, "Sex"] == "Female" else 0
                merged_df.loc[idx, "F20-F29"] = 1 if "F2" in str(merged_df.loc[idx, "Codiagnosis"]) or "f2" in str(merged_df.loc[idx, "Codiagnosis"]) else 0
                merged_df.loc[idx, "F30-F39"] = 1 if "F3" in str(merged_df.loc[idx, "Codiagnosis"]) else 0

                actual_row_class = None
                if merged_df.loc[idx, "F20-F29"] == 1:
                    actual_row_class = "SCHZ"
                if merged_df.loc[idx, "F30-F39"] == 1:
                    actual_row_class = "MDD"

                if merged_df.loc[idx, "Pred_class"] == actual_row_class:
                    merged_df.loc[idx, "Misdiagnosis"] = 0
                else: 
                    merged_df.loc[idx, "Misdiagnosis"] = 1

        for idx, row in merged_df.iterrows():
            pred_class = row["Pred_class"] 
            merged_df.loc[idx, "Confidence"] = row[pred_class]
            merged_df.loc[idx, "Uncertainty"] = 1-row[pred_class]
        
        diagnosis_order = ["CTT", "SCHZ", "MDD", "HC"]

        # Convert to categorical with custom order
        merged_df["Diagnosis"] = pd.Categorical(merged_df["Diagnosis"], 
                                                categories=diagnosis_order, 
                                                ordered=True)

        # Sort by predicted class (ascending) and then by confidence (descending)
        merged_df = merged_df.sort_values(by=["Diagnosis","Pred_labels", "Confidence"], ascending=[True, True, False])
        merged_df_reset = merged_df.reset_index(drop=True)

        merged_df_reset.to_csv(f"{self.logging_config.LOGGING_DIR}/testing_metadata_extended.csv")

        cat_variables = {
            "F20-F29": {"pos": 1.10, "color": "red"},
            "F30-F39": {"pos": 1.10, "color": "blue"},
            "Duration": {"pos": 1.15, "color": "black"},
            "Male": {"pos": 1.20, "color": "cyan"},
            "Female": {"pos": 1.20, "color": "magenta"},
            "Age": {"pos": 1.25, "color": "gray"},
            "Diagnosis": {"pos": 1.30, "color": "royalblue"},
            "Misdiagnosis": {"pos": 1.05, "color": "deeppink"}
        }

        subset_max_size = 50
        num_plots = math.ceil(merged_df_reset.shape[0]/subset_max_size)
        
        for i in range(num_plots):

            start_idx = subset_max_size * i
            end_idx = min(subset_max_size * (i + 1), merged_df_reset.shape[0])
            subset_merged_df = merged_df_reset.iloc[start_idx:end_idx].copy()
            fig_width = max(int(15/50*(end_idx-start_idx)), 10)

            # Create the figure with appropriate size
            fig, ax = plt.subplots(figsize=(fig_width, 10))
            
            # Plot the stacked bars
            # Use only probability columns for the stacked bar plot
            bar_data = subset_merged_df[["HC", "MDD", "SCHZ"]]
            bar_data.plot(kind="bar", stacked=True, ax=ax, width=0.8, 
                        color=["green", "orange", "purple"])
            
            
            # Add markers for each categorical variable
            for var_name, settings in cat_variables.items():
                # Get the normalized values for plotting
                y_pos = settings["pos"]
                x_indices = np.arange(len(subset_merged_df))
                
                if var_name == "Age":
                    # For age, use the actual values but normalize to [0, 1] range
                    # Skip NaN values
                    age_data = subset_merged_df["Age"].dropna()
                    if not age_data.empty:
                        age_indices = subset_merged_df.index[subset_merged_df["Age"].notna()]
                        x_pos_age = [list(subset_merged_df.index).index(i) for i in age_indices]
                        
                        age_min = age_data.min()
                        age_max = age_data.max()
                        if age_max > age_min:  # Avoid division by zero
                            age_norm = (age_data - age_min) / (age_max - age_min)
                            size_norm = 40 + 200 * age_norm  # Scale dot size
                            ax.scatter(x_pos_age, [y_pos] * len(age_data), s=size_norm, color=settings["color"], alpha=1)
                
                elif var_name == "Duration":
                    # For duration, use the actual values but normalize to [0, 1] range
                    # Skip NaN values
                    dur_data = subset_merged_df["Duration"].dropna()
                    if not dur_data.empty:
                        dur_indices = subset_merged_df.index[subset_merged_df["Duration"].notna()]
                        x_pos_dur = [list(subset_merged_df.index).index(i) for i in dur_indices]
                        
                        dur_min = dur_data.min()
                        dur_max = dur_data.max()
                        if dur_max > dur_min:  # Avoid division by zero
                            dur_norm = (dur_data - dur_min) / (dur_max - dur_min)
                            size_norm = 40 + 200 * dur_norm  # Scale dot size
                            ax.scatter(x_pos_dur, [y_pos] * len(dur_data), s=size_norm, color=settings["color"], alpha=1)
                
                elif var_name == "Diagnosis":
                    diag_data = subset_merged_df["Diagnosis"]

                    colors = {"CTT": "royalblue",
                            "HC": "green",
                            "MDD": "orange",
                            "SCHZ": "purple",
                            }
                    
                    for diag in colors.keys():
                        if not diag_data.empty:
                            diag_indices = subset_merged_df.index[subset_merged_df["Diagnosis"] == diag]
                            x_pos_diag = [list(subset_merged_df.index).index(i) for i in diag_indices]
                            ax.scatter(x_pos_diag, [y_pos] * len(diag_indices), color=colors[diag], alpha=1, s=250)

                else:
                    binary_indices = subset_merged_df.index[subset_merged_df[var_name] == 1]
                    binary_x_pos = [list(subset_merged_df.index).index(i) for i in binary_indices]
                    if len(binary_x_pos) > 0:
                        ax.scatter(binary_x_pos, [y_pos] * len(binary_indices), color=settings["color"], alpha=1, s=90)
            
            # Add a horizontal line at y=1 to separate the bars from the categorical markers
            ax.axhline(y=1.0, color="black", linestyle="-", linewidth=0.5)
            ax.set_title(f"Stacked Probability Bars by Diagnosis", fontsize=14)
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_ylim(0, 1.40)  # Extend y-axis to fit the categorical markers

            bar_legend = ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1.0))

            
            # Legend for categorical markers
            cat_legend_elements = []
            for var_name, settings in cat_variables.items():
                # Add specific entries for binary variables like 'Male', 'Female', 'F20-F29', 'F30-F39'
                if var_name in ["F20-F29", "F30-F39", "Male", "Female"]:
                    # Create a marker for the "present" state (value 1)
                    cat_legend_elements.append(
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=settings["color"],
                               markersize=8, label=f"{var_name}")
                    )
                # Add Age and Duration as continuous markers (size variation)
                elif var_name in ["Age", "Duration"]:
                     cat_legend_elements.append(
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=settings["color"],
                               markersize=8, label=f"{var_name} (Size indicates value)")
                    )
                # Add Diagnosis
                elif var_name == "Diagnosis":
                    # You might want to list all diagnoses, or just the ones present in the data
                    for diag_key, diag_color in colors.items():
                        cat_legend_elements.append(
                            Line2D([0], [0], marker="o", color="w", markerfacecolor=diag_color,
                                   markersize=8, label=f"Diagnosis: {diag_key}")
                        )

                elif var_name == "Misdiagnosis":
                    cat_legend_elements.append(
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=settings["color"],
                                markersize=8, label=f"{var_name}")
                    )

            ax.legend(handles=cat_legend_elements, loc="upper left", bbox_to_anchor=(1.02, 0.7), fontsize=10)
            ax.add_artist(bar_legend)
            
            # Make x-axis labels more readable
            ax.set_xticks(range(len(subset_merged_df)))
            ax.set_xticklabels(subset_merged_df["Filename"], rotation=90, ha="center", fontsize=4)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(f"{self.logging_config.LOGGING_DIR}/testing_metadata_preds_{self.atlas_name}_plot_{i+1}.png", 
                        dpi=300, bbox_inches="tight")
            plt.close()
    
        return
    

    # Calculate the ROC curve and the AUROC value for each of the diagnoses.
    def compute_auc(self, probs, labels, group=""): 
        num_classes = 3

        if self.model_config.PURPOSE == "hold_out":
            iterator = range(1, num_classes)  # skip HC
        else:
            iterator = range(num_classes)

        cls_names = {0: "HC", 1: "MDD", 2: "SCHZ"}
        colors = ['darkorange', 'blue', 'red'] # Define colors for each class

        # fig, axs = plt.subplots(nrows=1, ncols=num_classes, figsize=(25, 5))

        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()

        for cls in iterator:
            cls_name = cls_names[cls] 
            cls_probs = probs[:, cls]

            cls_labels = [1 if i == cls else 0 for i in labels]

            fpr, tpr, thresholds = sk.metrics.roc_curve(y_true=cls_labels, y_score=cls_probs, pos_label=1)

            fpr = np.array(fpr)
            tpr = np.array(tpr)

            # Calculate the Area Under the Curve (AUC) and plot for this diagnosis
            roc_auc = sk.metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colors[cls], label=f"{cls_name}, AUC = {roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic (ROC) Curves', fontsize=12)
        ax.grid(True)

        # Place the legend outside the plot
        ax.legend(loc='lower right', borderaxespad=0.2)

        plt.tight_layout() # Adjust plot to prevent labels from being cut off

        fig.savefig(f"{self.logging_config.LOGGING_DIR}/{self.atlas_name}_cls_AUROC_{group}.png")
        fig.savefig(f"{self.logging_config.LOGGING_DIR}/{self.atlas_name}_cls_AUROC_{group}.svg")

        plt.close()

        # In the case of hold-out testing (NSS and whiteCAT) the multi-class AUROC computation becomes a binary AUROC computation, requiring different inputs. 
        # See sk-learn documentation for details
        if self.model_config.PURPOSE == "hold_out":
            probs_corr = probs[:,2] # Becuase now binary case!

            labels_corr = []
            for i in range(len(labels)):
                labels_corr.append(labels[i]-1)

            multiclass_roc_auc = sk.metrics.roc_auc_score(y_true=labels_corr, y_score=probs_corr, average="weighted", multi_class="ovo")
        else:
            multiclass_roc_auc = sk.metrics.roc_auc_score(y_true=labels, y_score=probs, average="weighted", multi_class="ovo")

        return multiclass_roc_auc


    # Central function to perform predictions and evaluation. Contrary the ensemble method above, here the latent spaces are computed for the current atlas model.
    def compute_testing_performance(self):
       
        test_predictions, test_labels, test_labels_corr, test_names = self.make_predictions(dataloader=self.dataloader)

        test_labels_indices = torch.argmax(test_labels_corr, dim=1).tolist()  # Shape = [num_samples]

        test_probs = torch.softmax(test_predictions, dim=1)
        test_pred_labels = torch.argmax(test_probs, dim=1)

        multiclass_auroc = self.compute_auc(test_probs, test_labels_indices)

        # From the weighted probabilities / one hot encoded labels, get the correct labels. One-hot encoding is sorted alphabetically! 
        class_logits = torch.argmax(test_probs, dim=1).tolist()  # Shape = [704]
        
        confidence_scores_list = torch.max(test_probs, dim=1).values.tolist()

        self.compute_incorrect_proportion(prediction_labels=test_pred_labels, 
                                          target_labels=test_labels_indices, 
                                          names=test_names, 
                                          var="Diagnosis")

        accuracy = accuracy_score(y_true=test_labels_indices, y_pred=class_logits)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels_indices, 
                                                                   class_logits,
                                                                   average="weighted", 
                                                                   zero_division=np.nan
                                                                   )
        
        prediction_eval = {
            "model_name": self.atlas_name,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
            "auroc": round(multiclass_auroc, 3),
        }

        if self.model_config.PURPOSE == "hold_out":

            ctt_filenames = self.metadata_pred[self.metadata_pred["Diagnosis"] == "CTT"].index.to_list()
            non_ctt_filenames = self.metadata_pred[self.metadata_pred["Diagnosis"] != "CTT"].index.to_list()
            ctt_indices = [test_names.index(name) for name in ctt_filenames]
            non_ctt_indices = [test_names.index(name) for name in non_ctt_filenames]

            test_probs_non_ctt = test_probs[torch.tensor(non_ctt_indices)]
            test_labels_indices_non_ctt = [test_labels_indices[i] for i in non_ctt_indices]
            test_probs_ctt = test_probs[torch.tensor(ctt_indices)]
            test_labels_indices_ctt = [test_labels_indices[i] for i in ctt_indices]
            
            multiclass_auroc_non_ctt = self.compute_auc(test_probs_non_ctt, test_labels_indices_non_ctt, "non_CTT")
            multiclass_auroc_ctt = self.compute_auc(test_probs_ctt, test_labels_indices_ctt, "CTT")

            class_logits_non_ctt = torch.argmax(test_probs_non_ctt, dim=1).tolist()
            class_logits_ctt = torch.argmax(test_probs_ctt, dim=1).tolist()

            accuracy_ctt = accuracy_score(y_true=test_labels_indices_ctt, y_pred=class_logits_ctt)
            precision_ctt, recall_ctt, f1_ctt, _ = precision_recall_fscore_support(test_labels_indices_ctt, 
                                                                    class_logits_ctt,
                                                                    average="weighted", 
                                                                    zero_division=np.nan
                                                                    )
            
            accuracy_non_ctt = accuracy_score(y_true=test_labels_indices_non_ctt, y_pred=class_logits_non_ctt)
            precision_non_ctt, recall_non_ctt, f1_non_ctt, _ = precision_recall_fscore_support(test_labels_indices_non_ctt, 
                                                                    class_logits_non_ctt,
                                                                    average="weighted", 
                                                                    zero_division=np.nan
                                                                    )

            prediction_eval_ctt = {
                "model_name": self.atlas_name,
                "accuracy": round(accuracy_ctt, 3),
                "precision": round(precision_ctt, 3),
                "recall": round(recall_ctt, 3),
                "f1-score": round(f1_ctt, 3),
                "auroc": round(multiclass_auroc_ctt, 3),
            }
            logging.info(f"CTT Subset: {prediction_eval_ctt}")

            prediction_eval_non_ctt = {
                "model_name": self.atlas_name,
                "accuracy": round(accuracy_non_ctt, 3),
                "precision": round(precision_non_ctt, 3),
                "recall": round(recall_non_ctt, 3),
                "f1-score": round(f1_non_ctt, 3),
                "auroc": round(multiclass_auroc_non_ctt, 3),
            }
            logging.info(f"Non-CTT Subset: {prediction_eval_non_ctt}")
        

        correct_confs, incorrect_confs, ctt_confs = prediction_confidence_plot(
            true_labels=test_labels_indices,
            prediction_logits=class_logits,
            prediction_probs=confidence_scores_list,
            epoch=999,
            save_path=self.logging_config.LOGGING_DIR,
            timestamp=self.logging_config.TIMESTAMP,
            atlas_name=self.atlas_name
        )

        self.extract_latent_space(dataloader=self.dataloader)

        return prediction_eval