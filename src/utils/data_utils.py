# Import general functions
import os
from typing import List, Tuple
import anndata as ad
import pandas as pd
import regex as re
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import logging

from utils.logging_utils import log_and_print, log_checkpoint, log_title
from utils.helper_utils import (
    flatten_array, 
    scale_data_transform, 
    combine_dfs, 
    )



# This function splits a list of subjects into training and validation lists according to two dataframes containing training and validation annotations.
# While called train_ann, and val_ann, any dataframes can be used to guide the splitting of subjects.
def split_subjects(
    # The list of patients that should be split according to the prior train_val_split
    subjects: List[dict], 
    # Training annotations. 
    train_ann: pd.DataFrame, 
    # Validation annotations.
    val_ann: pd.DataFrame
) -> Tuple[List, List]:
    
    train_files = list(train_ann["Filename"])
    valid_files = list(val_ann["Filename"])

    train_subjects = []
    valid_subjects = []

    for subject in subjects:

        if subject["name"] in train_files:
            train_subjects.append(subject)

        elif subject["name"] in valid_files:
            valid_subjects.append(subject)
            
        else: 
            print(f"Filename {subject['name']} not found in annotations.")

    return train_subjects, valid_subjects


# The following two classes convert the lists of subjects into PyTorch dataloaders.

class CustomDataset():  
    # Create Datasets that can then be converted into DataLoader objects
    def __init__(self, subjects, covars):
        self.subjects = subjects
        self.covars = covars

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        measurements = self.subjects[idx]["measurements"]
        labels = self.subjects[idx]["labels"]


        all_output_labels = {}

        for covar in self.covars:
            label = labels[covar]
            all_output_labels[covar] = torch.as_tensor(label, dtype=torch.float32)  # float32 required for linear operations!

        names = self.subjects[idx]["name"]

        measurements = torch.as_tensor(measurements, dtype=torch.float32)  # Ensure this is float32, as the weights of the model are initialized.
        
        return measurements, labels, names


# This function loads all the mri data from .h5 files and outputs them as a list of subjects alongside all covariables including the true label. 
# The data for each patient is saved and importantly the length of each flattened input feature to split this again when passing the tensor to the model.
def load_mri_data(
    # The path to the directory where the MRI data is stored (.csv file formats)
    atlas_data_paths: dict,
    # Per dataset scalers that should be used for scaling the validation and testing data.
    scalers_per_atlas: dict = None, 
    # The path to the CSV file that contains the filenames of the MRI data and the diagnoses and covariates
    csv_paths: str = None,
    # The annotations DataFrame that contains the filenames of the MRI data and the diagnoses and covariates
    annotations = None,
    # The diagnoses that you want to include in the data loading, defaults to all
    diagnoses = None,
    covars = [],
    train_or_test: str = "train",
    # Should only a specific volume measurement be used? Choose one from Vgm, Vwm, Vcsf.
    vol_to_use: str = None,
    # Has the data already been batch corrected? In this case no scaling!
    batch_corrected: bool = False, 
    ) -> Tuple:

    if csv_paths is not None:
        for csv_path in csv_paths: 
            assert os.path.isfile(csv_path), f"CSV file '{csv_path}' not found"
            assert annotations is None, "Both CSV and annotations provided"
        
        # Initialize the data overview DataFrame
        data_overview = combine_dfs(csv_paths)

    # If the annotations are provided, make sure that they are a pandas DataFrame, and that the CSV path is not provided
    if annotations is not None:
        assert isinstance(
            annotations, pd.DataFrame
        ), "Annotations must be a pandas DataFrame"
        assert csv_paths is None, "Both CSV and annotations provided"
        # Initialize the data overview DataFrame
        data_overview = annotations

    # If no diagnoses are provided, use all diagnoses in the data overview
    if diagnoses is None:
        diagnoses = data_overview["Diagnosis"].unique().tolist()

    # If the covariates are not a list, make them a list
    if not isinstance(covars, list):
        covars = [covars]

    # If the diagnoses are not a list, make them a list
    if not isinstance(diagnoses, list):
        diagnoses = [diagnoses]
    
    # Set all the variables that will be one-hot encoded
    variables = ["Diagnosis"] + covars

    # Filter unwanted diagnoses
    data_overview = data_overview[data_overview["Diagnosis"].isin(diagnoses)]
    data_overview = data_overview[list(set(["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]+variables))]

    # produce one hot coded labels for each variable
    one_hot_labels = {}
    for var in variables:
        # check that the variables is in the data overview
        if var not in data_overview.columns:
            raise ValueError(f"Column '{var}' not found in CSV file or annotations")

        # one hot encode the variable
        one_hot_labels[var] = pd.get_dummies(data_overview[var], dtype=float)

    # For each subject, collect MRI data and variable data in the Subject object
    subjects = {}
    atlas_lengths = {}
    counter = 0

    for atlas, data_path in atlas_data_paths.items():

        atlas_lengths[atlas] = None
        data = pd.read_csv(data_path, header=0, index_col=0)

        # The oversampled trianing data is already normalized and scaled.
        if train_or_test != "train" and scalers_per_atlas is not None and not batch_corrected:
            logging.info(f"Scaling {atlas} testing data!")
            scalers = scalers_per_atlas[atlas] 
            mtypes = data.index.str.split('^').str.get(-1).unique()
            data, _ = scale_data_transform(data, mtypes, training_scalers=scalers)
        
        data.sort_index(inplace=True)  # Crucial step! Oversampling of the input data causes an alphabetical sorting of the features! 

        all_file_names = data.columns

        for index, row in data_overview.iterrows():

            if not row["Filename"] in all_file_names:
                data_overview.drop([index], inplace=True)
                counter+=1
                continue

            file_name = row["Filename"]
            patient_data = data[file_name]
            flat_patient_data = patient_data.values.tolist()

            if atlas_lengths[atlas] is None:
                atlas_lengths[atlas] = len(flat_patient_data)

            if not file_name in subjects: 
                subjects[file_name] = {"name": file_name,
                                    "measurements": flat_patient_data,
                                    "labels": {}
                                    }
            else: 
                subjects[file_name]["measurements"] += flat_patient_data

            for var in variables:
                subjects[file_name]["labels"][var] = one_hot_labels[var].loc[index].to_numpy().tolist()

    logging.info(f"{counter} files skipped because of lacking metadata!")
    print(f"{counter} files skipped because of lacking metadata!")
    logging.info(f"Altas input dimensions: {atlas_lengths}")

    return list(subjects.values()), data_overview, atlas_lengths


def process_subjects(
    # The list of tio.Subject objects that you want to process
    subjects: List,
    # The batch size for the DataLoader (the larger, the more memory is needed)
    batch_size: int,
    # Whether the data should be shuffled or not. Shuffling is important for training, but not for validation.
    shuffle_data: bool,
    # Covars used during training
    covars: List
) -> DataLoader:

    # Apply transformations
    dataset = CustomDataset(subjects=subjects, covars=covars)
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)

    # Return the DataLoader object
    return data_loader


# This function processes a latent space representation of mri data in AnnData format. Given a set of annotations, it
# adds the annotations to the AnnData object, performs PCA, UMAP and neighbors calculations, and saves the data if
# specified.
def process_latent_space_2D(
    # The AnnData object that contains the latent space representation of the MRI data.
    adata: ad.AnnData,
    # The annotations DataFrame that contains the meta data annotations for the MRI data.
    annotations: pd.DataFrame,
    # The number of neighbors for the UMAP calculation.
    umap_neighbors: int,
    # The random seed for reproducibility of the UMAP calculation.
    seed: int,
    # Name of the run, for which the latent space should be processed. 
    run_id,
    # Whether the data should be saved or not. If True, save_path, timestamp, epoch and data_type must be provided.
    save_data: bool = False,
    # The path where the data should be saved.
    save_path: str = None,
    # The timestamp of the current run, used for the filename.
    timestamp: str = None,
    # The current epoch, used for the filename.
    epoch: int = None,
    # The data type of adata (typically "training" or "validation"), used for the filename.
    data_type: str = None,
) -> ad.AnnData:

    # align annotation metadata with anndata.obs
    aligned_ann = annotations.set_index("Filename").reindex(adata.obs_names)

    # add annotation metadata to anndata.obs
    for col in aligned_ann.columns:
        adata.obs[col] = aligned_ann[col]

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, umap_neighbors, use_rep="X")
    sc.tl.umap(adata, random_state=seed)
    sc.tl.tsne(adata, random_state=seed)

    # save the data if specified
    if save_data:
        adata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}_atlas-{run_id}.h5ad")
        )

    # return the processed data
    return adata


# This function combines two latent spaces in AnnData format into a single AnnData object. It concatenates the two
# datasets, performs PCA, UMAP and tSNE, and saves the data if specified. It expects one dataset to be
# the training dataset and the other to be the validation dataset.
def combine_latent_spaces(
    # The training AnnData object that contains the latent space representation of the MRI data.
    tdata: ad.AnnData,
    # The validation AnnData object that contains the latent space representation of the MRI data.
    vdata: ad.AnnData,
    # The number of neighbors for the UMAP calculation.
    umap_neighbors: int,
    # The random seed for reproducibility of the UMAP calculation.
    seed: int,
    # Whether the data should be saved or not. If True, save_path, timestamp, epoch and data_type must be provided.
    save_data: bool = False,
    # The path where the data should be saved.
    save_path: str = None,
    # The timestamp of the current run, used for the filename.
    timestamp: str = None,
    # The current epoch, used for the filename.
    epoch: int = None,
    # The data type of adata (typically "training" or "validation"), used for the filename.
    data_type: str = None,
) -> ad.AnnData:

    # concatenate the training and validation data
    cdata = ad.concat({"train": tdata, "valid": vdata}, join="outer", label="Data_Type")

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(cdata)
    sc.pp.neighbors(cdata, umap_neighbors, use_rep="X")
    sc.tl.umap(cdata, random_state=seed)

    # save the data if specified
    if save_data:
        cdata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}.h5ad")
        )

    # return the combined data
    return cdata


# This function loads a model from a checkpoint file if it exists, otherwise it will raise an error.
# This function needs an overhaul. It should:
# - load model weights
# - load an optimizer state
# - load a scheduler state
# - load model metrics
# - and probably load model hyperparamaters
# Ideally all of this would be loaded from a single file.
# Also this function probably belongs in the base model file.
def load_checkpoint_model(model, model_filename: str):

    if os.path.isfile(model_filename):
        log_and_print(f"Loading checkpoint from '{model_filename}'")

        model_state = torch.load(model_filename)
        model.load_state_dict(model_state)

        log_and_print(f"Checkpoint loaded successfully")

    else:
        log_and_print(
            f"No checkpoint found at '{model_filename}', starting training from scratch"
        )

        raise FileNotFoundError(f"No checkpoint found at '{model_filename}'")

    return model


# This function saves the model to a file.
# This function needs an overhaul to:
# - save model weights
# - save optimizer state
# - save scheduler state
# - save model metrics
# - save model hyperparameters
# Ideally all of this would be saved in a single file.
# Also this function probably belongs in the base model file.
def save_model(model, save_path: str, timestamp: str, descriptor: str, epoch: int):
    model_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_e{epoch}_model.pth",
    )

    torch.save(model.state_dict(), model_save_path)

    log_checkpoint(model_path=model_save_path)


# This function saves the model metrics to a csv file.
# It should probably be combined with the save_model function.
def save_model_metrics(model_metrics, save_path: str, timestamp: str, descriptor: str):
    metrics_save_path = os.path.join(
        save_path,
        f"{timestamp}_{descriptor}_model_performance.csv",
    )

    pd.DataFrame(model_metrics).to_csv(metrics_save_path, index=False)

    log_checkpoint(
        metrics_path=metrics_save_path,
    )

# Performs the same operations as process_latent_space_2D, but is atlas agnostic.
def process_latent_space(
    # The AnnData object that contains the latent space representation of the MRI data.
    adata: ad.AnnData,
    # The annotations DataFrame that contains the meta data annotations for the MRI data.
    annotations: pd.DataFrame,
    # The number of neighbors for the UMAP calculation.
    umap_neighbors: int,
    # The random seed for reproducibility of the UMAP calculation.
    seed: int,
    # Whether the data should be saved or not. If True, save_path, timestamp, epoch and data_type must be provided.
    save_data: bool = False,
    # The path where the data should be saved.
    save_path: str = None,
    # The timestamp of the current run, used for the filename.
    timestamp: str = None,
    # The current epoch, used for the filename.
    epoch: int = None,
    # The data type of adata (typically "training" or "validation"), used for the filename.
    data_type: str = None,
) -> ad.AnnData:

    # align annotation metadata with anndata.obs
    aligned_ann = annotations.set_index("Filename").reindex(adata.obs_names)

    # add annotation metadata to anndata.obs
    for col in aligned_ann.columns:
        adata.obs[col] = aligned_ann[col]

    # perform PCA, UMAP and neighbors calculations
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, umap_neighbors, use_rep="X")
    sc.tl.umap(adata, random_state=seed)

    # save the data if specified
    if save_data:
        adata.write_h5ad(
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}.h5ad")
        )

    # return the processed data
    return adata
