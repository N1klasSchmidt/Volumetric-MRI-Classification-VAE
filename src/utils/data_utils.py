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
    resplit_data,
    )



"""
These are crucial functions for data processing that are used in all other scripts that handle the patients data. 
They handle any kind of data loading, preprocessing and saving.
"""


# This function perfroms a k-fold cross validation split of the entire metadata. Contrary to the function split_annotations, each split does not contain the
# same proportion of e.g. Sex or Diagnoses.
def cross_validation_split(
        # Path to the original metadata file
        path_original: str,
        # Path to the directory to which the new TESTING data should be saved.
        path_to_dir: str,
        # Number of folds into which the TRAINING data should be split. 
        k: int
        ):
    
    df = pd.read_csv(path_original, header=[0]) # 2892 patients, entire metadata, all samples
    df = df.drop(columns=["Unnamed: 0"])

    # Separate the catatonia patients from the rest to ensure that they are all in the testing dataset.
    df_ctt = df[df["Diagnosis"] == "CTT"]
    df_non_ctt = df[df["Diagnosis"] != "CTT"]

    # Split the remaining data into training and testing data.
    train_df, test_df = split_annotations(annotations=df_non_ctt, train_proportion=0.8) # This split should be 2253 and 560 ish
    
    # Combine with CTT patients again to form final testing dataset.
    test_df_ctt = pd.concat([test_df.copy(), df_ctt], axis=0)

    resplit_data(training_csv=train_df, 
                 testing_csv=test_df_ctt)  

    path_test = f"{path_to_dir}/testing_metadata.csv"
    test_df_ctt.to_csv(path_test)
    
    path_train = f"{path_to_dir}/training_metadata.csv"
    train_df.to_csv(path_train)

    # Perfrom k-fold splitting of training data into training and validation datasets.
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    valid_indices = []
    train_indices = []
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_df)):
        # print(len(train_index)) 1802
        # print(len(valid_index)) 451
        train_indices.append(train_index)
        valid_indices.append(valid_index)

    return train_indices, valid_indices


# This function splits an annotations dataframe into a test set and train set.
# Both sets have as equal a distribution of diagnoses, datasets and sex as possible.
def split_annotations(
    # The annotations dataframe that you want to split into a train and validation part
    annotations: pd.DataFrame,
    # The proportion of the data that should be in the first of the two outputted data frames
    proportion: float = 0.7,
    # The diagnoses you want to include in the split, defaults to all
    diagnoses: List[str] = None,
    # The datasets you want to include in the split, defaults to all
    datasets: List[str] = None,
    # The random seed for reproducibility
    seed: int = 123,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Initialize empty dataframes
    first = pd.DataFrame()
    second = pd.DataFrame()

    # If no diagnoses are specified, use all diagnoses in the annotations
    if diagnoses is None:
        diagnoses = annotations["Diagnosis"].unique()

    # If no datasets are specified, use all datasets in the annotations
    if datasets is None:
        datasets = annotations["Dataset"].unique()

    # For each diagnosis, dataset and sex, take a random sample of the data and split it into train and test
    for diagnosis in diagnoses:
        for dataset in datasets:
            #for sex in ["Female", "Male"]:
                # subset to the current cohort
            dataset_annotations = annotations[
                (annotations["Diagnosis"] == diagnosis)
                & (annotations["Dataset"] == dataset)
                #& (annotations["Sex"] == sex)
            ]
            # shuffle the data
            dataset_annotations = dataset_annotations.sample(
                frac=1, random_state=seed
            )
            # split the data train_proportion of the way through (usually 80%)
            split = round(len(dataset_annotations) * proportion)

            # add the split data to the train and test dataframes
            first = pd.concat(
                [first, dataset_annotations[:split]], ignore_index=True
            )
            second = pd.concat([second, dataset_annotations[split:]], ignore_index=True)

    return first, second



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
    # Should fine tuning be considered? This changes the path of the extracted data.
    fine_tuning: bool = False,
    # Should all labels be allowed to be loaded?
    all_labels: bool = False,
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

    if fine_tuning:  # The fine-tuning data has to be scaled with the rest to ensure similar performance!
        fine_tuning_df = data_overview[data_overview["Diagnosis"] == "CTT"]
        path_new_tune = f"/workspace/data/relevant_metadata/fine_tuning_metadata_xcov.csv"
        fine_tuning_df.to_csv(path_new_tune)

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
        data = pd.read_csv(data_path, header=[0,1], index_col=0)

        # The oversampled trianing data is already normalized and scaled.
        if train_or_test != "train" and scalers_per_atlas is not None and not batch_corrected:
            logging.info(f"Scaling {atlas} testing data!")
            scalers = scalers_per_atlas[atlas] 
            mtypes = data.index.str.split('^').str.get(-1).unique()
            data, _ = scale_data_transform(data, mtypes, training_scalers=scalers)
        
            
        data.sort_index(inplace=True)  # Crucial step! Oversampling of the input data causes an alphabetical sorting of the features! 
        #print(f"Index: {data.index.to_list()[:10]}")

        all_file_names = data.columns

        for index, row in data_overview.iterrows():
            
            if not fine_tuning and not all_labels:
                if row["Diagnosis"] == "CTT":
                    continue

            if not row["Filename"] in all_file_names:
                data_overview.drop([index], inplace=True)
                counter+=1
                continue

            file_name = row["Filename"]

            # if file_name in subjects: 
            #     if data_path in subjects[file_name]["measurements"]:
            #         continue

            patient_data = data[file_name]
            flat_patient_data = flatten_array(patient_data).tolist()

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
    # Name of the atlas / model whose latent space should be processed. 
    atlas_name,
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
            os.path.join(save_path, f"{timestamp}_e{epoch}_latent_{data_type}_atlas-{atlas_name}.h5ad")
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
