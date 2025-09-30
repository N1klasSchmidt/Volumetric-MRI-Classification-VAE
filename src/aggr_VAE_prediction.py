# Import general packages
import torch
import numpy as np
import anndata as ad
import pandas as pd
from typing import Dict, List
import math
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sklearn as sk
import matplotlib.pyplot as plt
import logging

from utils.data_utils import (process_subjects, 
                                      load_mri_data,
                                      process_latent_space_2D
                                      )

from utils.plotting_utils import latent_space_plot_2D

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
            run_id: str,
            # Output for logging
            logging_config: Dict,
            # Is the XCov or Base model being used? 
            xcov: bool = True,
            # Parameters to make code adaptable
            model_config: Dict = None,
            # Path to where the desired model is saved.
            model_path: str = None,
            # Trained model
            model = None,
            # How are the diagnostic labels mapped.
            label_mapping: Dict = None,
            # Extended metadata for further testing
            extended_data_path: str = None,
            # List of the original SORTED diagnostic labels, used for model training = corresponding with model output
            original_diagnoses: List = None,
            ):
        
        self.model_path = model_path
        self.model = model
        self.device = device
        self.dataloader = None
        self.run_id = run_id
        self.test_annotations = None
        self.logging_config = logging_config
        self.model_config = model_config
        self.metadata_pred = None
        self.label_mapping = label_mapping
        self.atlas_lengths = None
        self.extended_data_path = extended_data_path
        self.original_diagnoses = original_diagnoses
        self.xcov = xcov

    # Initiliaze testing data on which the model will predict.
    def init_test_data(self, csv_paths, data_paths, scalers):
        annotations = pd.read_csv(csv_paths[0], header=[0])
        self.test_annotations = annotations[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]

        assert self.model_config.PURPOSE != "train"  # Ensuring that we scale the testing data! 

        subjects, _ , atlas_lengths = load_mri_data(
                csv_paths=csv_paths,
                atlas_data_paths=data_paths,
                scalers_per_atlas=scalers,
                diagnoses=self.model_config.DIAGNOSES,
                train_or_test=self.model_config.PURPOSE,
                covars=self.model_config.COVARS,
                # How to scale and subset the data
                vol_to_use=self.model_config.VOLUMES,
                all_labels=self.model_config.ALL_LABELS,
                batch_corrected=self.model_config.BATCH_CORRECTION
            )
        self.atlas_lengths = atlas_lengths
        self.dataloader = process_subjects(subjects=subjects, batch_size=64, shuffle_data=False, covars=[self.model_config.COVARS[0]])

        if self.extended_data_path != None:
            self.extended_data = pd.read_csv(self.extended_data_path, header=0, index_col=0)

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
        if self.xcov:
            with torch.no_grad():
                for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                    # collect batch info
                    batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

                    # forward to latent space
                    mu, class_logits, xcov_preds = self.model.to_latent(batch_measurements)

                    # We select the majortiy predicted XCov label across each of the single VAEs
                    xcov_pred_labels = torch.stack([xcov_pred.argmax(dim=1) for xcov_pred in xcov_preds], dim=0)  # Shape = [num_atlases, batch_size]
                    xcov_majority_labels, _ = torch.mode(xcov_pred_labels, dim=0)  # Shape = [batch_size]

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
        
        else: 
            with torch.no_grad():
                for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                    batch_measurements = torch.stack([measurement.to(self.device) for measurement in measurements])

                    mu, class_logits = self.model.to_latent(batch_measurements)

                    latent_spaces.append(mu.cpu().numpy())
                    sample_names.extend(names)
                    model_preds.append(class_logits.argmax(dim=1).cpu().numpy())
                    model_probs.append(torch.max(torch.softmax(class_logits, dim=1), dim=1).values.tolist())

            adata = ad.AnnData(np.concatenate(latent_spaces))

            # add metadata to anndata object, namely sample names and classifier predictions
            adata.obs_names = sample_names
            adata.obs["Diagnosis_pred"] = pd.Categorical(np.concatenate(model_preds))
            adata.obs["Confidence"] = np.concatenate(model_probs)

        label_map = {}
        for idx, diag in enumerate(self.original_diagnoses):
            label_map[idx] = diag
        
        adata.obs["Diagnosis_pred"] = adata.obs["Diagnosis_pred"].cat.rename_categories(label_map)

        self.latent = adata

        latent_df = self.latent.to_df()
        latent_df.to_csv(f"{self.logging_config.LOGGING_DIR}/latent_space_data.csv")

        latent = process_latent_space_2D(
            adata=self.latent,
            annotations=self.test_annotations,
            umap_neighbors=self.logging_config.UMAP_NEIGHBORS,
            seed=self.logging_config.SEED,
            run_id=self.run_id,
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
            run_id=self.run_id,
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

        label_indices = labels

        for idx, label in enumerate(label_indices):
            empty_list = [0 for i in range(len(self.original_diagnoses))]
            current_label = self.label_mapping[label]
            
            if current_label not in self.original_diagnoses: 
                assert self.extended_data is not None, "If using non-training diagnoses during testing, ensure to pass extended data with 'Diagnosis_extra' column!"
                sample_name = names[idx]
                current_label = str(self.extended_data[sample_name]["Diagnosis_extra"])
            
            current_label_idx = self.original_diagnoses.index(current_label) 
            empty_list[current_label_idx] = 1
            
            corrected_labels[idx] = torch.tensor(empty_list)
        return corrected_labels


    # Let model predict on the loaded testing data
    def make_predictions(self, dataloader): 

        self.model.to(self.device)
        self.model.eval()

        all_preds = []
        all_labels = []
        all_names = []

        if self.xcov:
            with torch.no_grad():
                for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                    # Move samples to device
                    batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)  # Shape: (batch_size, len_atlas)
                    batch_labels = torch.stack([label for label in labels["Diagnosis"]]).to(self.device)  # Shape: (batch_size, num_classes)
                    recon_measurements, mus, logvars, class_logits, xcov_preds, latent_spaces = self.model(batch_measurements)

                    all_preds.append(class_logits.cpu())  # Shape: (num_batches, batch_size, num_classes)
                    all_labels.extend(torch.argmax(batch_labels, dim=0).cpu().numpy())
                    all_names.extend(names)  # Shape: (num_samples)
        else:
            with torch.no_grad():
                for batch_idx, (measurements, labels, names) in enumerate(dataloader):
                    # Move samples to device
                    batch_measurements = torch.stack([measurement for measurement in measurements]).to(self.device)  # Shape: (batch_size, len_atlas)
                    batch_labels = torch.stack([label for label in labels["Diagnosis"]]).to(self.device)  # Shape: (batch_size, num_classes)
                    recon_measurements, mus, logvars, class_logits, latent_spaces = self.model(batch_measurements)

                    all_preds.append(class_logits.cpu())  # Shape: (num_batches, batch_size, num_classes)
                    all_labels.extend(torch.argmax(batch_labels, dim=0).cpu().numpy())
                    all_names.extend(names)  # Shape: (num_samples)
                
        all_preds = torch.cat(all_preds, dim=0)  # Shape: (num_samples, num_classes)
        #all_labels = torch.stack(all_labels, dim=0)   # Shape: (num_samples, num_classes)
                
        all_probs = torch.softmax(all_preds, dim=1).cpu().numpy()  # Shape: (num_samples, num_classes)
        all_prediction_labels = torch.argmax(torch.softmax(all_preds, dim=1), dim=1).cpu().numpy()

        all_labels_corr = self.correct_labels(labels=all_labels, names=all_names)
        
        probs_df = pd.DataFrame(all_probs, columns=self.original_diagnoses)
        probs_df["Filename"] = all_names
        probs_df["Pred_labels"] = all_prediction_labels

        label_map = {}
        for idx, diag in enumerate(self.original_diagnoses):
            label_map[idx] = diag

        probs_df["Pred_class"] = probs_df["Pred_labels"].map(label_map)

        # Merge with test annotations to maintain same order
        metadata_pred = pd.merge(self.test_annotations, probs_df, on="Filename")
        metadata_pred.set_index("Filename", inplace=True, drop=True)

        # Save to CSV
        metadata_pred.to_csv(f"{self.logging_config.LOGGING_DIR}/testing_metadata_preds_{self.run_id}.csv")
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
    

    # Calculate the ROC curve and the AUROC value for each of the diagnoses.
    def compute_auc(self, probs, labels, group="", colors={"HC":'darkorange', "MDD":'blue', "SCHZ":'red'}): 
        num_classes = len(self.original_diagnoses)

        assert len(colors.keys()) == len(self.original_diagnoses), "Ensure that number of colors is equal to number of diagnoses!"

        cls_names = {}
        for idx, diag in enumerate(self.original_diagnoses):
            cls_names[idx] = diag

        iterator_list = []
        for idx in range(num_classes):
            if all(probs[:, idx] == 0): 
                iterator_list.append(idx)

        fig = plt.figure(figsize=(6,6))
        ax = plt.gca()

        for cls in iterator_list:
            cls_name = cls_names[cls] 
            cls_probs = probs[:, cls]

            cls_labels = [1 if i == cls else 0 for i in labels]

            fpr, tpr, thresholds = sk.metrics.roc_curve(y_true=cls_labels, y_score=cls_probs, pos_label=1)

            fpr = np.array(fpr)
            tpr = np.array(tpr)

            # Calculate the Area Under the Curve (AUC) and plot for this diagnosis
            roc_auc = sk.metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colors[cls_name], label=f"{cls_name}, AUC = {roc_auc:.2f}")

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

        fig.savefig(f"{self.logging_config.LOGGING_DIR}/{self.run_id}_cls_AUROC_{group}.png")
        fig.savefig(f"{self.logging_config.LOGGING_DIR}/{self.run_id}_cls_AUROC_{group}.svg")

        plt.close()

        # In the case of hold-out testing (NSS and whiteCAT) the multi-class AUROC computation becomes a binary AUROC computation, requiring different inputs. 
        # See sk-learn documentation for details
        if len(iterator_list) == 2:
            probs_corr = probs[:, max(iterator_list)] # Becuase now binary case! y_score should contain estimates of the class with the greater label (see documentation).

            labels_corr = []
            for i in range(len(labels)):
                labels_corr.append(labels[i]-min(iterator_list))  # Floor all lables to 0,1

            multiclass_roc_auc = sk.metrics.roc_auc_score(y_true=labels_corr, y_score=probs_corr, average="weighted", multi_class="ovo")
        elif len(iterator_list) <= 1: 
            raise ValueError(f"Cannot compute AUROC value using only {len(iterator_list)} valid columns!")
        else:
            multiclass_roc_auc = sk.metrics.roc_auc_score(y_true=labels, y_score=probs, average="weighted", multi_class="ovo")

        return multiclass_roc_auc


    # Central function to perform predictions and evaluation. Contrary the ensemble method above, here the latent spaces are computed for the current atlas model.
    def compute_testing_performance(self, colors):
       
        test_predictions, test_labels, test_labels_corr, test_names = self.make_predictions(dataloader=self.dataloader)

        test_labels_indices = torch.argmax(test_labels_corr, dim=1).tolist()  # Shape = [num_samples]

        test_probs = torch.softmax(test_predictions, dim=1)
        test_pred_labels = torch.argmax(test_probs, dim=1)

        multiclass_auroc = self.compute_auc(test_probs, test_labels_indices, colors=colors)

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
            "model_name": self.run_id,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
            "auroc": round(multiclass_auroc, 3),
        }

        for diagnosis in self.metadata_pred["Diagnosis"].unique().to_list():

            diag_filenames = self.metadata_pred[self.metadata_pred["Diagnosis"] == diagnosis].index.to_list()
            non_diag_filenames = self.metadata_pred[self.metadata_pred["Diagnosis"] != diagnosis].index.to_list()
            diag_indices = [test_names.index(name) for name in diag_filenames]
            non_diag_indices = [test_names.index(name) for name in non_diag_filenames]

            test_probs_non_diag = test_probs[torch.tensor(non_diag_indices)]
            test_labels_indices_non_diag = [test_labels_indices[i] for i in non_diag_indices]
            test_probs_diag = test_probs[torch.tensor(diag_indices)]
            test_labels_indices_diag = [test_labels_indices[i] for i in diag_indices]
            
            multiclass_auroc_non_diag = self.compute_auc(test_probs_non_diag, test_labels_indices_non_diag, f"non_{diagnosis}")
            multiclass_auroc_diag = self.compute_auc(test_probs_diag, test_labels_indices_diag, f"{diagnosis}")

            class_logits_non_ctt = torch.argmax(test_probs_non_diag, dim=1).tolist()
            class_logits_ctt = torch.argmax(test_probs_diag, dim=1).tolist()

            accuracy_ctt = accuracy_score(y_true=test_labels_indices_diag, y_pred=class_logits_ctt)
            precision_ctt, recall_ctt, f1_ctt, _ = precision_recall_fscore_support(test_labels_indices_diag, 
                                                                    class_logits_ctt,
                                                                    average="weighted", 
                                                                    zero_division=np.nan
                                                                    )
            
            accuracy_non_ctt = accuracy_score(y_true=test_labels_indices_non_diag, y_pred=class_logits_non_ctt)
            precision_non_ctt, recall_non_ctt, f1_non_ctt, _ = precision_recall_fscore_support(test_labels_indices_non_diag, 
                                                                    class_logits_non_ctt,
                                                                    average="weighted", 
                                                                    zero_division=np.nan
                                                                    )

            prediction_eval_ctt = {
                "model_name": self.run_id,
                "accuracy": round(accuracy_ctt, 3),
                "precision": round(precision_ctt, 3),
                "recall": round(recall_ctt, 3),
                "f1-score": round(f1_ctt, 3),
                "auroc": round(multiclass_auroc_diag, 3),
            }
            logging.info(f"{diagnosis} Subset: {prediction_eval_ctt}")

            prediction_eval_non_ctt = {
                "model_name": self.run_id,
                "accuracy": round(accuracy_non_ctt, 3),
                "precision": round(precision_non_ctt, 3),
                "recall": round(recall_non_ctt, 3),
                "f1-score": round(f1_non_ctt, 3),
                "auroc": round(multiclass_auroc_non_diag, 3),
            }
            logging.info(f"Non-{diagnosis} Subset: {prediction_eval_non_ctt}")

        self.extract_latent_space(dataloader=self.dataloader)

        return prediction_eval