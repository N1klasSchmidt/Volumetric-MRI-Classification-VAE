import pathlib
import pandas as pd
import regex as re
import os
import numpy as np
import logging
from typing import Dict, List, Tuple
from utils.logging_utils import log_and_print, log_title
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC, RandomOverSampler
from imblearn.under_sampling import AllKNN, TomekLinks, RandomUnderSampler, NearMiss
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from combat.pycombat import pycombat
import matplotlib.pyplot as plt
import umap
os.environ["SCIPY_ARRAY_API"] = "1"
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import json
import hashlib



# ====================================================================================================
# **HELPER FUNCTIONS**
#   
# While data_processing_2D.py handles all the processing tasks after the input .h5 files have been 
# loaded, the functions here revolve around handling the compressed .h5 files.Especially the resplit_data 
# is essential, as it reshuffles the entire dataset from the h5 files into more managable csv files. 
# This is performed before every training run, which ensures that always the correct data is being
# passed to the data_processing_*.py functions. 
#
# Here you can also find functions used for plotting e.g. UMAPs or Saliency maps, as well as normaliz-
# ation and scaling functions. A crucial part are also the three resampling functions: oversample(),
# undersample() and hybrid_sample().
# ====================================================================================================



# Get a hash of the current PyTorch model. This is useful to validate that the model saved at one point is the same at a later stage.
def get_model_hash(model):
    """Get hash of model parameters"""
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()

# Converts e.g. GM, WM and CSF columns into a single 1D feature vector.
def flatten_array(df: pd.DataFrame) -> np.ndarray:
    # Converts data frame to flattened array. 
    array = df.to_numpy()
    flat_array = np.nan_to_num(array.flatten())
    return flat_array

def get_atlas_map(atlases: List) -> Dict:
    atlas_map = {}
    for atlas in sorted(atlases): 
        caps = atlas.title()

        if f"[{caps[0]}]" in atlas_map.values():
            atlas_map[atlas] = f"[{caps[:1]}]"
        else: 
            atlas_map[atlas] = f"[{caps[0]}]"
    return atlas_map

# Normalization and scaling of input data. Scaling is performed on a per datatype basis. There are two options for scaling
# - "z" for z-scaling along the ROIs
# - "iqr" for inter quantile range scaling
def normalize_and_scale_df(
        df: pd.DataFrame, 
        df_meta: pd.DataFrame, 
        scaling_method: str = "z", 
        scale_per_dataset: bool = False,
        vol_to_use: str = None,
        ) -> pd.DataFrame:
    
    # Normalizes the columns (patient volumes) by Log Scaling and scales the rows (ROIs) with Z-transformation.

    df_copy = df.copy()
    column_sums = df_copy.sum()
    
    # Apply the formula: ln((10000*value)/sum_values + 1) "Log transformation"
    transformed_df = np.log((10000 * df_copy) / column_sums + 1)

    norm_copy = transformed_df.copy()
  
    cols = norm_copy.columns.get_level_values(-1).tolist() # Select lowest level of Multiindex (Measurements: Vgm, Vwm, Vcsf)
    unique_cols = list(set(cols))

    filename_to_dataset_map = df_meta.set_index("Filename")["Dataset"]
    filenames_in_data = norm_copy.columns.get_level_values(0).unique()
    datasets = df_meta["Dataset"].unique().tolist()

    for col_type in unique_cols:  # scale each of the measurement types separately
        vols_to_scale = [col for col in norm_copy.columns if col[-1] == col_type]
            
        # Scale the selected columns per row
        z_scaled = norm_copy[vols_to_scale]
        iqr_scaled = df_copy[vols_to_scale]
        
        if scale_per_dataset:  # Scale each of the datasets separately if desired
            for dataset in datasets: 
                filenames_for_target_dataset = filenames_in_data[
                        filename_to_dataset_map.reindex(filenames_in_data) == dataset
                        ].tolist()

                sets_to_scale = [col_tuple for col_tuple in z_scaled.columns 
                                if col_tuple[0] in filenames_for_target_dataset]
                
                if not sets_to_scale: # Skip if no columns for this dataset and col_type combination
                    continue

                if scaling_method == "z":  # For z-scaling
                    z_scaled_dataset_sub = z_scaled[sets_to_scale]
                    
                    z_scaled_dataset_sub = z_scaled_dataset_sub.apply(
                        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
                        axis=1
                        )

                    z_scaled.loc[:, sets_to_scale] = z_scaled_dataset_sub
                
                if scaling_method == "iqr":  # For inter-quantile-range scaling
                    iqr_scaled_dataset_sub = iqr_scaled[sets_to_scale]
                    
                    iqr_scaled_dataset_sub = iqr_scaled_dataset_sub.apply(
                        lambda row: (row - row.median()) / (row.quantile(0.75)-row.quantile(0.25)) if (row.quantile(0.75)-row.quantile(0.25)) > 0 else pd.Series(0.0, index=row.index),
                        axis=1
                        )

                    iqr_scaled.loc[:, sets_to_scale] = iqr_scaled_dataset_sub

                    # Optionally perform on the rows a z-scaling to prevent possible NaNs during training. 
                    # z_scaled_dataset_sub = iqr_scaled[sets_to_scale].apply(
                    #     lambda row: row / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
                    #     axis=1
                    #     )
                    
                    # iqr_scaled.loc[:, sets_to_scale] = z_scaled_dataset_sub
        
        else:  # Otherwise simply scale across all datasets
            if scaling_method == "z":  # For z-scaling
                z_scaled = z_scaled.apply(
                    lambda row: (row - row.mean()) / row.std() if row.std() > 0 else pd.Series(0.0, index=row.index),
                    axis=1
                )

            if scaling_method == "iqr":  # For inter-quantile-range scaling
                iqr_scaled = iqr_scaled.apply(
                    lambda row: (row - row.median()) / (row.quantile(0.75)-row.quantile(0.25)) if (row.quantile(0.75)-row.quantile(0.25)) > 0 else pd.Series(0.0, index=row.index),
                    axis=1
                )

        # Output for z-scaled data
        if scaling_method == "z":
            norm_copy.loc[:, vols_to_scale] = z_scaled
        
        # Output for IQR scaled data
        if scaling_method == "iqr":
            df_copy.loc[:, vols_to_scale] = iqr_scaled
    
    if scaling_method == "z":
        if vol_to_use is not None:  # Subset only a certain type of measurement (Vgm, Vwm, Vcsf)
            subset = [col for col in norm_copy.columns if col[-1] == vol_to_use]
            return norm_copy[subset]
        else: 
            return norm_copy
    
    if scaling_method == "iqr":
        if vol_to_use is not None and scaling_method == "iqr":  # Subset only a certain type of measurement (Vgm, Vwm, Vcsf)
            subset = [col for col in norm_copy.columns if col[-1] == vol_to_use]
            return df_copy[subset]
        else: 
            return df_copy
        
# Primary standardization function for aggregated VAE model. If no scalers are passed, new one are fit to the assumed training data. Otherwise, existing scalers are applied to assumed validation or training data. 
def scale_data_transform(
    df: pd.DataFrame,
    mtypes: List[str], 
    training_scalers: Dict[str, StandardScaler] = None,  # During scaling of validation and testing data we use the scaler from the training.
    log_transform: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Normalizes and scales volumetric data using log transformation and StandardScaler.

    The function first applies a log transformation to normalize ROI volumes relative
    to the total volume of each scan. It then applies Z-score scaling (StandardScaler)
    to the ROIs (rows). Scaling is performed across all datasets combined. Ideally the
    scaling would be per-datasets, but they differ between training and testing. 

    Args:
        df: DataFrame with ROIs as rows and multi-index columns (Filename, MeasurementType).
        training_scalers: Dictionary mapping measurement types to fitted StandardScaler objects.
                         If None, new scalers will be fitted. If provided, these scalers will be used
                         for transformation (for validation/test data).

    Returns:
        Tuple of (scaled_dataframe, scalers_dict) where scalers_dict has structure {m_type: StandardScaler}.
    """
    
    get_training_scalers = {}

    if log_transform:        
        for suffix in mtypes:
            mask = df.index.str.endswith(suffix)
            subset = df[mask].copy()
            remainder = df[~mask]
            column_sums = subset.sum()
            transformed_subset = np.log((10000 * subset) / column_sums + 1)
            df = pd.concat([remainder, transformed_subset])#.sort_index()
    
    scaled_df = df.copy()

    # Remove possible rows where all values = 0! 
    rows_with_all_zeros = (df.sum(axis=1) == 0)
    scaled_df = scaled_df[~rows_with_all_zeros]

    # Scale across all datasets together
    for mtype in mtypes:
        # Identify all columns for the current measurement type
        rows_to_scale = scaled_df.index.str.endswith(mtype)

        # Select the data, transpose it so ROIs are features (columns)
        data_subset = scaled_df[rows_to_scale]
        data_transposed = data_subset.T
        
        # Apply StandardScaler
        if training_scalers is None:
            scaler = StandardScaler()
            scaled_transposed = scaler.fit_transform(data_transposed)
            get_training_scalers[mtype] = scaler
        else: 
            scaler = training_scalers[mtype]
            scaled_transposed = scaler.transform(data_transposed)
        
        # Transpose back and place into the final DataFrame
        scaled_data = pd.DataFrame(
            scaled_transposed.T, 
            index=data_subset.index, 
            columns=data_subset.columns
        )
        scaled_df[rows_to_scale] = scaled_data

    return scaled_df, get_training_scalers

# Normalizes the columns of a data frame by log-transformation
def normalize_columns(df: pd.DataFrame):
    col_sums = df.sum()
    log_transformed_df = np.log((10000 * df) / col_sums + 1)
    return log_transformed_df

# Extract all data paths for aggregation of all atlases
def get_all_data(directory: str, ext: str = ".h5") -> list:
    data_paths = list(pathlib.Path(directory).rglob(f"*{ext}"))
    return data_paths

# Find the path to the best model of a specific training run
def get_model_path(directory: str, ext: str = "ebest_model.pth"):
    model_paths = list(pathlib.Path(directory).rglob(f"*{ext}"))  # Recursively search all subdirectories
    
    if len(model_paths) > 1: 
        logging.info(f"Multiple models selected! Defaulting to first.")
    
    model_path = model_paths[0]
    return model_path

# Combines any number of csv files to a single pandas DataFrame, keeping only shared column indices. 
def combine_dfs(paths: list):
    if len(paths) > 1: 
        for i in range(1,len(paths)):
            if i == 1: 
                joined_df = pd.read_csv(paths[i-1], header=[0])
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")  # Parameter "inner" keeps only the shared column indices.
            else:
                next_df = pd.read_csv(paths[i], header=[0])
                joined_df = pd.concat([joined_df, next_df], join="inner")
    else: 
        joined_df = pd.read_csv(paths[0], header=[0])
    return joined_df


# Reads the original hdf5 files with all the input data and outputs them as a data frame
def read_hdf5_to_df(filepath: str):
    if not os.path.exists(filepath):
        logging.info(f"File {filepath} does not exist")
        return None
    try:
        df = pd.read_hdf(filepath, key='atlas_data')
        return df 
    except Exception as e:
        logging.info(f"Error reading {filepath}: {e}")
        return None
    
    
def split_df(
        path_original: str, 
        path_to_dir: str
        ):

    df = pd.read_csv(path_original, header=[0])
    df = df.drop(columns=["Unnamed: 0"])

    for option in ["training", "testing"]:
        if option == "training":
            path = f"{path_to_dir}/{option}_metadata.csv"
            subset_df = df[df["Usage_original"] != "testing"]
            subset_df.to_csv(path)

        elif option == "testing":
            path = f"{path_to_dir}/{option}_metadata.csv"
            subset_df = df[df["Usage_original"] == option]
            subset_df.to_csv(path)


# Improved function to resplit the data that takes any number of splits, applying them to the input data and outputting corresponding csv files
# with the respective splits.
def resplit_data(df_dict: Dict, output_dir: str, data_type: str = "csv"):

    atlases = ["cobra", "lpba40", "neuromorphometrics", "suit", "thalamic_nuclei", "thalamus"]

    for purpose, df in df_dict.items():
        message = log_title(f"DATA RESPLIT {purpose}")
        logging.info(message)

        metadata_path = f"{output_dir}/{purpose}_metadata.csv"
        df.to_csv(metadata_path)

        if not os.path.exists(f"/workspace/data/raw_data_{purpose}/"):
            os.mkdir(f"/workspace/data/raw_data_{purpose}/")

        df_filenames = df["Filename"].tolist()

        for atlas in atlases:
            logging.info(f"Reshuffling the extracted xml data for atlas {atlas}")
            atlas_data_path = f"/workspace/data/raw_data_{data_type}/{atlas}_data.{data_type}"

            if data_type == "h5":
                atlas_data = read_hdf5_to_df(atlas_data_path)
            elif data_type == "csv":
                atlas_data = pd.read_csv(atlas_data_path, index_col=0, header=0)
            else: 
                raise ValueError(f"Unsupported data type: '{data_type}'. Must be 'h5' or 'csv'.")

            total_raw_filenames = atlas_data.columns.unique().to_list()
            filtered_filenames = [filename for filename in df_filenames 
                                   if filename in total_raw_filenames] 

            filtered_atlas_data = atlas_data[filtered_filenames]

            new_path = f"/workspace/data/raw_data_{purpose}/{atlas}_data.csv"

            filtered_atlas_data.to_csv(new_path)

    return 

# Function that splits a passed meta data file into several parts. A training/validation split, a testing split and optionally a
# held out testing set are returned. This is all adjusted through a central config file and is performed on a metadata file 
# that contains all the data. 
def determine_split(
        split_config: Dict, 
        meta: pd.DataFrame
        ) -> Dict:

    meta_split = {}

    if len(split_config["remove_within_variable"]) == 0:

        train_valid_df, test_df = split_annotations(annotations=meta, proportion=split_config["train_valid_proportion"])
        meta_split =  {"train_valid": train_valid_df, "test_1": test_df,}

    else: 
        removed_subsets = []
        
        for idx, var in enumerate(split_config["remove_within_variable"]): 
            remove_subset = meta[meta[var].isin(split_config["remove_variable_values"][idx])]
            removed_subsets.append(remove_subset)

        removed_data = pd.concat(removed_subsets).drop_duplicates(keep="first")
        removed_samples = removed_data["Filename"].to_list()

        filtered_df = meta[~meta["Filename"].isin(removed_samples)]
        
        train_valid_df, test_df = split_annotations(annotations=filtered_df, proportion=split_config["train_valid_proportion"])
        meta_split =  {"train_valid": train_valid_df, "test_1": test_df,}

        if split_config["test_on_subset"] == "all":
            meta_split["test_2"] = removed_data
        elif split_config["test_on_subset"] != None: 
            if split_config["test_on_subset"] in split_config["remove_within_variable"]:
                sub_idx = split_config["remove_within_variable"].index(split_config["test_on_subset"])
                meta_split["test_2"] = removed_subsets[sub_idx]
            else: 
                for sub_idx, var_vals in enumerate(split_config["remove_variable_values"]): 
                    if split_config["test_on_subset"] in var_vals:
                        _hold_out = removed_subsets[sub_idx]

                        hold_out = _hold_out[_hold_out[split_config["remove_within_variable"][sub_idx]].isin([split_config["test_on_subset"]])]
                       
                        meta_split["test_2"] = hold_out
        else: 
            logging.info("Not testing on removed data subset.")

    for purpose, df in meta_split.items(): 
        print(f"\n Shuffled metadata for {purpose}")
        print(f"Shape: {df.shape}")
        print(f"{df.head(5)}")

    return meta_split


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


# In the case there are more metadata entries than data points, the metadata has to be filtered to prevent shape mismatches. 
def filter_all_metadata(df: pd.DataFrame, data_path: str, atlas_name: str):
    data_train_path = data_path+f"/train_xml_data/Aggregated_{atlas_name}.csv"
    data_test_path = data_path+f"/test_xml_data/Aggregated_{atlas_name}.csv"
    data_hold_out_path = data_path+f"/hold_out_xml_data/Aggregated_{atlas_name}.csv"

    data_train = pd.read_csv(data_train_path, header = [0,1], index_col=0)
    data_test = pd.read_csv(data_test_path, header = [0,1], index_col=0)
    data_hold_out = pd.read_csv(data_hold_out_path, header = [0,1], index_col=0)

    data = pd.concat([data_train, data_test, data_hold_out], axis=0)

    all_file_names = data.columns.get_level_values(0).to_list()
    logging.info(f"Original metadata shape: {df.shape}")

    for index, row in df.iterrows():
        if not row["Filename"] in all_file_names:
            df.drop([index], inplace=True)
            continue

    logging.info(f"New metadata shape: {df.shape}")
    df.to_csv("/workspace/data/relevant_metadata/full_metadata_filtered.csv")
    return df


# This function takes the existing model metrics as a DataFrame and the metrics of the current epoch as a list of dictionaries.
# Each dictionary can contain metrics for multiple losses, e.g. {"valid_loss": 0.5, "v_recon_loss": 0.3, "v_kldiv_loss": 0.2}.
# It then updates the data frame with the new metrics and returns it.
def update_performance_metrics(
    # The existing model metrics as a DataFrame
    model_metrics: pd.DataFrame,
    # The metrics of the current epoch as a list of dictionaries
    epoch_metrics: List[Dict[str, float]],
) -> pd.DataFrame:

    # Combine metrics dicts into one
    epoch_losses_comb = epoch_metrics[0]

    for i in range(1, len(epoch_metrics)):
        epoch_losses_comb.update(epoch_metrics[i])

    # Convert combined metrics to DataFrame
    epoch_losses_df = pd.DataFrame(epoch_losses_comb, index=[0])

    # Append to model_metrics
    model_metrics = pd.concat([model_metrics, epoch_losses_df], axis=0)
    model_metrics.reset_index(drop=True, inplace=True)

    # Return updated model_metrics
    return model_metrics


# This function takes a dictionary of losses and returns a dictionary of loss proportions. By specifying the total_loss_key,
# the function will calculate the proportion of each loss relative to the total loss. If the total loss is 0, the function will
# return the input dictionary unchanged. The function will also add the total loss to the output dictionary. If there are metrics
# that are not losses in the input dictionary, they will be added to the output dictionary unchanged.
def loss_proportions(
        # The dict key of the total loss
        total_loss_key: str, 
        # The dictionary of losses
        losses_dict: Dict[str, float]) -> Dict[str, float]:

    # Get the total loss
    total_loss = losses_dict[total_loss_key]

    # If the total loss is 0, we cannot calculate proportions
    if total_loss == 0:
        log_and_print("Total loss is 0, cannot calculate proportions.")
        return losses_dict

    # Create a dictionary to store the loss proportions
    prop_dict = {}
    # Add the total loss to the dictionary to start
    prop_dict[total_loss_key] = total_loss

    # Calculate the proportion of each loss relative to the total loss
    for key, value in losses_dict.items():
        # ignore the total loss key
        if key == total_loss_key:
            continue

        # Calculate the proportion of the loss
        # only calculate proportions for losses, not for other metrics
        if "loss" in key:
            # replace "_loss" with "_prop"
            key = key.split("_")[-2] + "_prop"
            # store the proportion of the loss	
            prop_dict[key] = value / total_loss
        # for other metrics...
        else:
            # ... store the metric unchanged
            prop_dict[key] = value
    
    # Return the dictionary of loss proportions
    return prop_dict

# This function preprocesses the data using batch correction. Categorical variables are binarized with the LabelEncoders, then, using PyCombat 
# the batch correction is carried out. Finally, the batch corrected data is scaled and saved accordingly. Optionally plot a UMAP of the new
# features. In the past the batch correction was too strong, eliminating biological variance.

def batch_corrected_preprocessing(
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame, 
        hold_out_df: pd.DataFrame,
        metadata: pd.DataFrame) -> None:

    atlas_map = {"neuromorphometrics": "[N]",
            "lpba40": "[L]",
            "suit": "[S]",
            "cobra": "[C]",}

    # Load all data, across all atlases and modalities (train, test, hold_out)
    all_unscaled_uncorrected_data = join_all_h5(metadata, atlas_map)

    log_transformed_uncorrected_data = normalize_columns(all_unscaled_uncorrected_data)

    logging.info(log_title("BATCH CORRECTION"))
    # Perform batch correction of data. We add as covatiates the Diagnosis, Age and Sex to try and avoid the variance being removed here!
    le = LabelEncoder()
    le.fit(metadata["Dataset"].unique())
    datasets = le.transform(metadata["Dataset"])
    metadata["Dataset_int"] = datasets

    le2 = LabelEncoder()
    le2.fit(metadata["Diagnosis"].unique())
    diagnoses = le2.transform(metadata["Diagnosis"])
    metadata["Diagnosis_int"] = diagnoses

    metadata["Age_binned"] = pd.cut(metadata["Age"].to_numpy(), 3)
    le3 = LabelEncoder()
    le3.fit(metadata["Age_binned"].unique())
    ages_binned = le3.transform(metadata["Age_binned"])
    metadata["Age_binned_int"] = ages_binned

    all_batch_uncorrected_data = pd.merge(log_transformed_uncorrected_data.T, metadata, how="inner", left_index=True, right_index=True).drop(columns=["Age", "Sex", "Age_binned"]).dropna(how="any")
    all_uncorrected_data = all_batch_uncorrected_data.drop(columns=["Dataset_int", "Dataset", "Diagnosis", "Diagnosis_int", "Age_binned_int", "Sex_int"]).T
    batch = all_batch_uncorrected_data["Dataset_int"].to_list()

    filtered_rows_uncorrected_data = all_uncorrected_data[all_uncorrected_data.std(axis=1) != 0]

    diagnosis_mod = all_batch_uncorrected_data["Diagnosis_int"].to_list()
    sex_mod = all_batch_uncorrected_data["Sex_int"].to_list()
    age_mod = all_batch_uncorrected_data["Age_binned_int"].to_list()

    mod_list = [diagnosis_mod, sex_mod, age_mod]

    logging.info("Performning batch correction with PyCombat")
    all_unscaled_corrected_data = pycombat(filtered_rows_uncorrected_data, batch, mod=mod_list)

    logging.info(log_title("SCALING"))
    train_data, test_data = scale_and_save_corrected_data(all_unscaled_corrected_data, 
                                  atlas_map=atlas_map,
                                  train_df=train_df,
                                  test_df=test_df,
                                  hold_out_df=hold_out_df)
    
    #logging.info(log_title("COMPUTING UMAPs"))
    #umap_correction(all_unscaled_corrected_data, metadata, "all_data")
    #umap_correction(test_data, metadata, "test_data")

    return 


# This function essentially performs together with scale_and_save_corrected_data the function of resplit_v2, converting -h5 files to separate csv files
def join_all_h5(
        metadata: pd.DataFrame,
        atlas_map: Dict,
        ) -> pd.DataFrame:
    
    joined_df = None

    for atlas in atlas_map.keys():
        atlas_abbrev = atlas_map[atlas]

        path_train = f"/workspace/data/raw_extracted_xml_data/train_xml_data/Aggregated_{atlas}.h5"
        path_test = f"/workspace/data/raw_extracted_xml_data/test_xml_data/Aggregated_{atlas}.h5"
        #path_hold_out = f"/workspace/data/raw_extracted_xml_data/hold_out_xml_data/Aggregated_{atlas}.h5"
        
        train_data = pd.read_hdf(path_train,key='atlas_data')
        test_data = pd.read_hdf(path_test, key='atlas_data')
        #hold_out_data = pd.read_hdf(path_hold_out, key='atlas_data')

        atlas_data_df = pd.concat([train_data, test_data], axis=1)

        filter_files = metadata.index.to_list()
        atlas_data_df = atlas_data_df.loc[:, filter_files]
        
        # We convert the mutli-index to a single index adding the volume type to the row name instead
        df_stacked = atlas_data_df.stack(level='Volume', future_stack=True)
        df_transformed_index = df_stacked.index.map(lambda x: f"{atlas_abbrev}^{x[0]}^{x[1]}")  # By adding an atlas-abbreviation we can easily trace back the ROIs to the atlas.
        df_final = df_stacked.set_axis(df_transformed_index, axis=0)
        df_final.index.name = 'Feature_Combined'

        if joined_df is None:
            joined_df = df_final
        else:
            joined_df = pd.concat([joined_df, df_final])
    return joined_df

# This function scales and saves the batch corrected data. It is integrated with the resampling functions, as not to perform scaling twice. 
# All data is saved to their original file paths again, allowing seamless integration with the other processing functions. 
def scale_and_save_corrected_data(
        unscaled_data: pd.DataFrame, 
        atlas_map: Dict,
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        hold_out_df: pd.DataFrame,
        ):

    for atlas, abbrev in atlas_map.items():
        logging.info(f"Saving {atlas} batch corrected and scaled training, testing and hold-out data")
        atlas_df = unscaled_data[unscaled_data.index.str.startswith(abbrev)]
        index_parts = np.array(atlas_df.index.str.split('^', n=2, expand=True).to_list())

        original_multiindex = pd.MultiIndex.from_arrays([
            index_parts[:,1], 
            index_parts[:,2]
        ], names=[None, 'Volume'])
        
        df_restored = atlas_df.copy()
        df_restored.index = original_multiindex
        df_original = df_restored.unstack('Volume')

        df_original_train = df_original[train_df["Filename"]]
        df_original_test = df_original[test_df["Filename"]]
        df_original_hold_out = df_original[hold_out_df["Filename"]]

        # Scale the training data and transform the testing data with the fitted scalers
        df_scaled_train, training_scalers = scale_data_transform(df_original_train, log_transform=False)
        df_scaled_test, _ = scale_data_transform(df_original_test, training_scalers=training_scalers, log_transform=False)
        df_scaled_hold_out, _ = scale_data_transform(df_original_hold_out, training_scalers=training_scalers, log_transform=False)

        path_train = f"/workspace/data/raw_extracted_xml_data/train_xml_data/Aggregated_{atlas}_bat_corr.csv"
        df_scaled_train.to_csv(path_train)
        path_test = f"/workspace/data/raw_extracted_xml_data/test_xml_data/Aggregated_{atlas}_bat_corr.csv"
        df_scaled_test.to_csv(path_test)
        path_hold_out = f"/workspace/data/raw_extracted_xml_data/hold_out_xml_data/Aggregated_{atlas}_bat_corr.csv"
        df_scaled_hold_out.to_csv(path_hold_out)

    return df_scaled_train, df_scaled_test 


# This function is integrated with the below resampling functions. It loads and scales all the data from the specified atlases and metadata. 
# The Scaler is fit to the training data and applied to the validation and testing data. 
def load_all_data(atlas_map: Dict,
                  metadata_df: pd.DataFrame,
                  save_dir: str,
                  train_scalers_atlases: Dict = None,
                  batch_corr_ext: str = "",
                  data_type: str = "csv"
                  ):
    
    joined_df = None

    get_train_scalers_atlases = {}

    for atlas, atlas_abbrev in atlas_map.items():
        
        path = f"{save_dir}/{atlas}_data{batch_corr_ext}.{data_type}"
        atlas_data_df = pd.read_csv(path, header=0, index_col=0)

        filter_files = metadata_df["Filename"].to_list()
        atlas_data_df = atlas_data_df.loc[:, filter_files]

        mtypes = atlas_data_df.index.str.split('^').str.get(-1).unique()

        # We scale each of the atlases separately. We perform the scaling BEFORE the over sampling to ensure that each of the features has an equal effect on the generation of new samples.
        if len(batch_corr_ext) == 0: # Batch corrected data has already been scaled! 
            # The training data gets scaled normally. We save the scaler. The validation data should be scaled with the values from the train scaling! 
            if train_scalers_atlases is None:
                logging.info(f"Step: Fitting and transforming with scaler on {atlas} data")
                atlas_data_df, train_scalers = scale_data_transform(atlas_data_df, mtypes, training_scalers=None)
                get_train_scalers_atlases[atlas] = train_scalers  # train_scalers is a dictionary with measurement_type : StandardScaler
            
            else:
                logging.info(f"Step: Transforming with scaler on {atlas} data")
                train_scalers = train_scalers_atlases[atlas]
                atlas_data_df, _ = scale_data_transform(atlas_data_df, mtypes, training_scalers=train_scalers)

        prefix = f"{atlas_abbrev}^"

        # We convert the mutli-index to a single index adding the volume type to the row name instead
        atlas_data_df.index = atlas_data_df.index.map(lambda x: f"{prefix}{x}")
        #atlas_data_df.index.name = 'Feature_Combined'

        #atlas_data_df.index = atlas_data_df.index.str.replace('_', '^')

        if joined_df is None:
            joined_df = atlas_data_df
        else:
            joined_df = pd.concat([joined_df, atlas_data_df])
    return joined_df, get_train_scalers_atlases


# This function performs oversampling of the training data. By specifying which files should and shoulnd't be oversampled the train-validation split can be considered. 
# The oversampled data is saved to the training data directory. Three methods of oversampling have been integrated: SMOTE, ADASYN and Random. Further information can be 
# found under: https://imbalanced-learn.org/stable/over_sampling.html 
def oversample(files_to_oversample: pd.DataFrame,
               files_not_to_oversample: pd.DataFrame = None, 
               method: str = "ADASYN",
               save_dir: str = "/workspace/data/raw_extracted_xml_data/train_xml_data",
               batch_corr_ext: str = "",
               atlas_map = {"cobra": "[C]",
                            "lpba40": "[L]",
                            "neuromorphometrics": "[N]",
                            "suit": "[S]",
                            },
                data_type: str = "csv",
               ) -> pd.DataFrame:
    
    # Step 1: Create a merged dataframe of the stacked volume measurements across all atlases. Normalize and Scale
    logging.info("Step: Scaling and merging all data")
    
    files_to_oversample = files_to_oversample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
    joined_df, scalers = load_all_data(atlas_map, files_to_oversample, save_dir, batch_corr_ext=batch_corr_ext, data_type=data_type)  # scalers is a dictionary with the structure: {atlas: {measurment_type:StandardScaler}}
    rois = joined_df.index.to_list()

    if files_not_to_oversample is not None:
        files_not_to_oversample = files_not_to_oversample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
        joined_df_rest, _ = load_all_data(atlas_map, files_not_to_oversample, save_dir, train_scalers_atlases=scalers, batch_corr_ext=batch_corr_ext, data_type=data_type)
        original_rest_data = pd.merge(joined_df_rest.T, files_not_to_oversample, on="Filename", how="left")
        original_rest_data = original_rest_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")

    # Step 2: One-Hot encode the dataset variable, to allow assigning after oversampling
    logging.info("Step: One-hot encoding the Dataset variable")
    original_data = pd.merge(joined_df.T, files_to_oversample, on="Filename", how="left")  # Transpose as we need the columns to contain the features (ROIs)
    
    if method != "SMOTENC":
        original_data_onehot = pd.get_dummies(original_data, columns=["Dataset"], prefix=None).drop(columns=["Age", "Sex", "Sex_int"])

        # Step 3: Perform oversampling with method of choice
        logging.info("Step: Oversampling of files")
        X = original_data_onehot.drop(columns=["Diagnosis", "Filename"])
        y = original_data_onehot["Diagnosis"]

        original_filenames = original_data_onehot["Filename"].to_list()

        # We perform the over sampling of the minority groups MDD and SCHZ. Considering we have 1661 HC patients, we will have a total of 1661*3=4983 patients.
        if method == "SMOTE":
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=6)
            X_resampled, y_resampled = smote.fit_resample(X, y)

        elif method == "ADASYN":
            ada = ADASYN(sampling_strategy="auto", random_state=42, n_neighbors=6)
            X_resampled, y_resampled = ada.fit_resample(X, y)
        
        elif method == "random":
            ros = RandomOverSampler(sampling_strategy="auto", random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
        
        else:
            print("Please specify a valid oversampling method!")

        logging.info("Step: Preparing oversampled data frame")
        one_hot_cols = [col for col in X_resampled if col.startswith("Dataset_")]
        new_column_data = X_resampled[one_hot_cols].idxmax(axis=1)
        X_resampled = pd.concat([X_resampled, new_column_data.rename("Dataset")], axis=1)

        #X_resampled["Dataset"] = X_resampled.copy()[one_hot_cols].idxmax(axis=1)
        X_resampled["Dataset"] = X_resampled["Dataset"].str.replace("Dataset_", "")
        X_resampled.drop(columns=one_hot_cols, inplace=True)

        new_filenames = original_filenames + [f"Oversampled_File_{i}" for i in range(1,(X_resampled.shape[0]-len(original_filenames)+1))]
        sample_status = len(original_filenames)*["original"] + (len(new_filenames)-len(original_filenames))*["sampled"]
        X_resampled["Filename"] = new_filenames
        X_resampled["Sample_status"] = sample_status
        final_oversampled_df = X_resampled.set_index("Filename")
        final_oversampled_df["Diagnosis"] = y_resampled.to_list()

        metadata_to_keep = ["Dataset", "Diagnosis"]
    else:
        # Step 3: Perform oversampling with method of choice
        logging.info("Oversampling of files ...")
        original_data = pd.merge(joined_df.T, files_to_oversample, on="Filename", how="left")  # Transpose as we need the columns to contain the features (ROIs)
        X = original_data.drop(columns=["Diagnosis", "Filename", "Sex_int"])
        y = original_data["Diagnosis"]

        original_filenames = original_data["Filename"].to_list()

        # We perform the over sampling of the minority groups MDD and SCHZ. Considering we have 1661 HC patients, we will have a total of 1661*3=4983 patients.
        smotenc = SMOTENC(categorical_features=[-1, -2, -3], sampling_strategy="auto", random_state=42, k_neighbors=6)
        X_resampled, y_resampled = smotenc.fit_resample(X, y)

        logging.info("Adjusting data frame for further use ...")
        X_resampled["Filename"] = None
        new_filenames = original_filenames + [f"Oversampled_File_{i}" for i in range(1,(X_resampled.shape[0]-len(original_filenames)+1))]
        sample_status = len(original_filenames)*["original"] + (len(new_filenames)-len(original_filenames))*["sampled"]
        X_resampled["Filename"] = new_filenames
        X_resampled["Sample_status"] = sample_status
        final_oversampled_df = X_resampled.set_index("Filename")
        final_oversampled_df["Diagnosis"] = y_resampled.to_list()

        metadata_to_keep = ["Dataset", "Diagnosis", "Age", "Sex"]  # SMOTENC also inherently handles the categorical variables, opposed to ADASYN and SMOTE

    # Step 4: Create the new metadata file with oversampled file information
    logging.info("Step: Creating new metadata dataframe with oversampled file information")
    
    # Filter for newly sampled data and select relevant columns
    new_samples_metadata = final_oversampled_df.query("Sample_status == 'sampled'")[metadata_to_keep].reset_index()
    oversampled_metadata = pd.concat([files_to_oversample, new_samples_metadata], ignore_index=True)

    # Step 5: Convert the oversampled data back into a per-atlas format to be seamlessly integrated with the rest of the pipeline
    logging.info("Step: Returning data to original per-atlas format")

    total_df = pd.concat([final_oversampled_df, original_rest_data])
    intermediate_df = total_df.drop(columns=metadata_to_keep+["Sample_status"]).T

    for atlas, abbrev in atlas_map.items():
        atlas_df = intermediate_df[intermediate_df.index.str.startswith(abbrev)]
        index_parts = np.array(atlas_df.index.str.split('_', n=2, expand=True).to_list())

        original_multiindex = pd.MultiIndex.from_arrays([
            index_parts[:,1],  # Original level 0 (e.g., region names)
            index_parts[:,2]   # Original level 1 (e.g., Volume)
        ], names=[None, 'Volume'])
        
        df_restored = atlas_df.copy()
        df_restored.index = original_multiindex
        df_original = df_restored.unstack('Volume')

        df_original.sort_index(inplace=True)

        path = f"{save_dir}/Aggregated_{atlas}{batch_corr_ext}_os.csv"
        logging.info(f"Step: Saving oversampled {atlas} data to: {path}")
        df_original.to_csv(path)

    return oversampled_metadata, rois, scalers


# This function performs undersampling of the training data. By specifying which files should and shoulnd't be undersampled the train-validation split can be considered. 
# The undersampled data is saved to the training data directory. Four methods of undersampling have been integrated: All-KNN, NearMiss2, Tomek Links and Random. Further information can be 
# found under: https://imbalanced-learn.org/stable/under_sampling.html
def undersample(files_to_undersample: pd.DataFrame,
               files_not_to_undersample: pd.DataFrame = None, 
               method: str = "random",
               save_dir: str = "/workspace/data/raw_extracted_xml_data/train_xml_data",
               batch_corr_ext: str = "",
               atlas_map = {"cobra": "[C]",
                            "lpba40": "[L]",
                            "neuromorphometrics": "[N]",
                            "suit": "[S]",
                            },
                data_type: str = "csv",
               ) -> pd.DataFrame:
    
    # Step 1: Create a merged dataframe of the stacked volume measurements across all atlases. Normalize and Scale
    logging.info("Step: Merging and scaling all data")
    
    files_to_undersample = files_to_undersample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
    joined_df, scalers = load_all_data(atlas_map, files_to_undersample, save_dir, batch_corr_ext=batch_corr_ext, data_type=data_type)
    rois = joined_df.index.to_list()

    if files_not_to_undersample is not None:
        files_not_to_undersample = files_not_to_undersample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
        joined_df_rest, _ = load_all_data(atlas_map, files_not_to_undersample, save_dir, train_scalers_atlases=scalers, batch_corr_ext=batch_corr_ext, data_type=data_type)
        original_rest_data = pd.merge(joined_df_rest.T, files_not_to_undersample, on="Filename", how="left")
        original_rest_data = original_rest_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")

    to_be_undersampled_data = pd.merge(joined_df.T, files_to_undersample, on="Filename", how="inner")  # Transpose as we need the columns to contain the features (ROIs)
    non_undersampled_data = pd.merge(joined_df_rest.T, files_not_to_undersample, on="Filename", how="left")
    
    X = to_be_undersampled_data.drop(columns=["Age", "Sex", "Sex_int", "Diagnosis", "Dataset"]).set_index("Filename")
    y = to_be_undersampled_data["Diagnosis"]

    logging.info("Step: Undersampling of files")

    if method == "allknn":
        sampler = AllKNN(sampling_strategy="auto", n_neighbors=4)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif method == "nm2":
        sampler = NearMiss(sampling_strategy="auto", n_neighbors=3, version=2)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    elif method == "tomek":
        sampler = TomekLinks(sampling_strategy="auto")
        X_resampled, y_resampled = sampler.fit_resample(X, y)

    elif method == "random": 
        sampler = RandomUnderSampler(sampling_strategy="auto", random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    else: 
        print("Please specify a valid undersampling method!")

    chosen_samples = sampler.sample_indices_

    undersampled_data = to_be_undersampled_data.iloc[chosen_samples]

    non_undersampled_data["Sample_status"] = "valid"
    undersampled_data["Sample_status"] = "sampled"

    non_undersampled_data = non_undersampled_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")
    undersampled_data = undersampled_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")

    total_data = pd.concat([undersampled_data, non_undersampled_data])

    # Step 5: Convert the oversampled data back into a per-atlas format to be seamlessly integrated with the rest of the pipeline
    logging.info("Step: Returning data to original per-atlas format")

    intermediate_df = total_data.drop(columns=["Sample_status", "Dataset", "Diagnosis"]).T

    atlas_dfs = {}

    for atlas, abbrev in atlas_map.items():
        atlas_df = intermediate_df[intermediate_df.index.str.startswith(abbrev)]
        index_parts = np.array(atlas_df.index.str.split('^', n=2, expand=True).to_list())

        original_multiindex = pd.MultiIndex.from_arrays([
            index_parts[:,1], 
            index_parts[:,2]
        ], names=[None, 'Volume'])
        
        df_restored = atlas_df.copy()
        df_restored.index = original_multiindex
        df_original = df_restored.unstack('Volume')

        atlas_dfs[atlas] = df_original
        path = f"{save_dir}/Aggregated_{atlas}{batch_corr_ext}_us.csv"
        logging.info(f"Step: Saving oversampled {atlas} data to: {path}")
        df_original.to_csv(path)

    return files_to_undersample.iloc[chosen_samples], rois, scalers


# This function performs a hybrid resampling = combination of over- and under sampling. The majority class is under sampled, while the minority class is
# over sampled to the level of the target class (here SCHZ). There exist also pre-implemented version from sklearn (https://imbalanced-learn.org/stable/combine.html),
# yet this combination of Random under sampling and ADASYN over sampling has proven to be most effective.
def hybrid_sample(files_to_resample: pd.DataFrame,
               files_not_to_resample: pd.DataFrame = None,
               class_props: Dict = {"majority": "HC",
                                    "target": "SCHZ",
                                    "minority": "MDD"},
               save_dir: str = "/workspace/data/raw_data_train_valid",
               batch_corr_ext: str = "",
               atlas_map = {"cobra": "[C]",
                            "lpba40": "[L]",
                            "neuromorphometrics": "[N]",
                            "suit": "[S]",
                            },
                data_type: str = "csv",
               ) -> pd.DataFrame:
    
    # Step 1: Create a merged dataframe of the stacked volume measurements across all atlases. Normalize and Scale
    logging.info("Step: Merging and scaling all data (unless batch corrected)")

    print(files_to_resample)

    files_to_resample = files_to_resample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
    joined_df, scalers = load_all_data(atlas_map, files_to_resample, save_dir, batch_corr_ext=batch_corr_ext, data_type=data_type)
    rois = joined_df.index.to_list()

    joined_df_t = joined_df.T
    joined_df_t.index.name = "Filename"
    joined_df_t = joined_df_t.reset_index()

    non_resampled_data = None
    if files_not_to_resample is not None:
        files_not_to_resample = files_not_to_resample[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
        joined_df_rest, _ = load_all_data(atlas_map, files_not_to_resample, save_dir, train_scalers_atlases=scalers, batch_corr_ext=batch_corr_ext, data_type=data_type)
        
        joined_df_rest_t = joined_df_rest.T
        joined_df_rest_t.index.name = "Filename"
        joined_df_rest_t = joined_df_rest_t.reset_index()
        
        non_resampled_data = pd.merge(joined_df_rest_t, files_not_to_resample, on="Filename", how="left")
        non_resampled_data = non_resampled_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")

    to_be_resampled_data = pd.merge(joined_df_t, files_to_resample, on="Filename", how="left")#.set_index("Filename")  # Transpose as we need the columns to contain the features (ROIs)

    #print(to_be_resampled_data)
    #non_resampled_data = pd.merge(joined_df_rest.T, files_not_to_resample, on="Filename", how="left")

    # We subset only the HC and SCHZ patients, so that under sample to the level of SCHZ
    to_be_undersampled_data = to_be_resampled_data[to_be_resampled_data["Diagnosis"] != class_props["minority"]]
    undersampling_counts = to_be_undersampled_data["Diagnosis"].value_counts()
    logging.info(f"Step: Undersampling input data class counts: {undersampling_counts}")

    X = to_be_undersampled_data.drop(columns=["Dataset", "Diagnosis", "Filename", "Age", "Sex", "Sex_int"])
    y = to_be_undersampled_data["Diagnosis"]

    rus = RandomUnderSampler(sampling_strategy="majority", random_state=42)
    #rus = AllKNN(sampling_strategy="auto", n_neighbors=6)
    X_under_resampled, y_under_resampled = rus.fit_resample(X, y)

    # This is now the new input data for the over sampling
    undersampled_data = to_be_undersampled_data.iloc[rus.sample_indices_]

    to_be_oversampled_majority = undersampled_data[undersampled_data["Diagnosis"] == class_props["target"]]
    to_be_oversampled_minority = to_be_resampled_data[to_be_resampled_data["Diagnosis"] == class_props["minority"]]

    to_be_oversampled = pd.concat([to_be_oversampled_majority, to_be_oversampled_minority])
    to_be_oversampled_one_hot = pd.get_dummies(to_be_oversampled, columns=["Dataset"], prefix=None).drop(columns=["Age", "Sex", "Sex_int"])
    oversampling_counts = to_be_oversampled["Diagnosis"].value_counts()
    logging.info(f"Step: Oversampling input data class counts: {oversampling_counts}")

    X_new = to_be_oversampled_one_hot.drop(columns=["Diagnosis", "Filename"])
    y_new = to_be_oversampled_one_hot["Diagnosis"]

    original_filenames = to_be_oversampled_one_hot["Filename"].to_list()

    ada = ADASYN(sampling_strategy="minority", random_state=42, n_neighbors=6)  #n_neighbors originally 6
    X_over_resampled, y_over_resampled = ada.fit_resample(X_new, y_new)

    # We convert the one-hot encoding back into a single column with strings for the row values
    one_hot_cols = [col for col in X_over_resampled if col.startswith("Dataset_")]
    X_over_resampled["Dataset"] = X_over_resampled[one_hot_cols].idxmax(axis=1)
    X_over_resampled["Dataset"] = X_over_resampled["Dataset"].str.replace("Dataset_", "")
    X_over_resampled = X_over_resampled.drop(columns=one_hot_cols)
    X_over_resampled["Filename"] = None

    new_filenames = original_filenames + [f"Oversampled_File_{i}" for i in range(1,(X_over_resampled.shape[0]-len(original_filenames)+1))]
    sample_status = len(original_filenames)*["original"] + (len(new_filenames)-len(original_filenames))*["over_sampled"]

    X_over_resampled["Filename"] = new_filenames
    X_over_resampled["Sample_status"] = sample_status
    oversampled_data = X_over_resampled.set_index("Filename")
    oversampled_data["Diagnosis"] = y_over_resampled.to_list()

    # We also adjust the undersample_train data to allow for merging
    undersampled_data["Sample_status"] = "under_sampled"
    undersampled_data = undersampled_data.set_index("Filename")

    balanced_data = pd.concat([undersampled_data, oversampled_data[oversampled_data["Diagnosis"] == class_props["minority"]]])
    balanced_counts = balanced_data["Diagnosis"].value_counts()
    logging.info(f"Step: Balanced data class counts: {balanced_counts}")

    # We create a new dataframe that contains the new metadata for the resampled training (non-validation) data
    #oversampled_data_metadata = oversampled_data.query("Diagnosis == 'MDD'")[["Dataset", "Diagnosis"]].reset_index()
    #undersampled_data_metadata = files_to_resample.set_index("Filename").loc[undersampled_data.index.to_list()].reset_index()
    #new_metadata = pd.concat([oversampled_data_metadata, undersampled_data_metadata, None], ignore_index=True)

    train_metadata = files_to_resample.set_index("Filename")
    original_minority_samples = oversampled_data.query("Sample_status == 'original' & Diagnosis == 'MDD'").index.to_list()
    original_minority_metadata = train_metadata.loc[original_minority_samples].reset_index()

    oversampled_minority_metadata = oversampled_data.query("Sample_status == 'over_sampled'")[["Dataset", "Diagnosis"]].reset_index()

    undersampled_train_metadata = files_to_resample.set_index("Filename").loc[undersampled_data.index.to_list()].reset_index()

    new_metadata = pd.concat([undersampled_train_metadata, original_minority_metadata, oversampled_minority_metadata, None], ignore_index=True)

    # Finally we save the new training data (training + validation) in the same location, to be used for training.
    # non_resample_subset = non_resampled_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")
    balanced_data_subset = balanced_data.drop(columns=["Age", "Sex", "Sex_int"])

    logging.info("Step: Returning data to original per-atlas format")
    if non_resampled_data is not None:
        total_df = pd.concat([balanced_data_subset, non_resampled_data])
    else: 
        total_df = balanced_data_subset

    intermediate_df = total_df.drop(columns=["Dataset", "Diagnosis", "Sample_status"]).T

    atlas_dfs = {}

    for atlas, abbrev in atlas_map.items():
        atlas_df = intermediate_df[intermediate_df.index.str.startswith(abbrev)]
        index_parts = np.array(atlas_df.index.str.split('^', n=2, expand=True).to_list())

        original_multiindex = pd.MultiIndex.from_arrays([
            index_parts[:,1], 
            index_parts[:,2]
        ], names=[None, 'Volume'])
        
        df_restored = atlas_df.copy()
        df_restored.index = original_multiindex
        df_original = df_restored.unstack('Volume')

        df_original = df_original.astype(float)

        atlas_dfs[atlas] = df_original
        path = f"{save_dir}/{atlas}_data{batch_corr_ext}_rs.csv"
        logging.info(f"Step: Saving resampled {atlas} data to: {path}")
        df_original.to_csv(path)

    return new_metadata, rois, scalers


# No resampling is performed, only data scaling
def no_resample(
        training_files: pd.DataFrame,
        validation_files: pd.DataFrame,
        save_dir: str = "/workspace/data/raw_data_train_valid",
        batch_corr_ext: str = "",
        atlas_map: Dict = {"cobra": "[C]",
                    "lpba40": "[L]",
                    "neuromorphometrics": "[N]",
                    "suit": "[S]",
                    },
        ) -> pd.DataFrame:
    
    # Step 1: Create a merged dataframe of the stacked volume measurements across all atlases. Normalize and Scale
    logging.info("Step: Merging and scaling all data (unless batch corrected)")

    training_files = training_files[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
    joined_training_df, scalers = load_all_data(atlas_map, training_files, save_dir, batch_corr_ext=batch_corr_ext)
    rois = joined_training_df.index.to_list()

    joined_training_df_t = joined_training_df.T
    joined_training_df_t.index.name = "Filename"
    joined_training_df_t = joined_training_df_t.reset_index()

    validation_data = None
    if validation_files is not None:
        validation_files = validation_files[["Filename", "Dataset", "Diagnosis" , "Age", "Sex", "Sex_int"]]
        joined_validation_df, _ = load_all_data(atlas_map, validation_files, save_dir, train_scalers_atlases=scalers, batch_corr_ext=batch_corr_ext)
        
        joined_validation_df_t = joined_validation_df.T
        joined_validation_df_t.index.name = "Filename"
        joined_validation_df_t = joined_validation_df_t.reset_index()
        
        validation_data = pd.merge(joined_validation_df_t, validation_files, on="Filename", how="left")
        validation_data = validation_data.drop(columns=["Age", "Sex", "Sex_int"]).set_index("Filename")

    training_data = pd.merge(joined_training_df_t, training_files, on="Filename", how="left") # Transpose as we need the columns to contain the features (ROIs)

    training_data = training_data.drop(columns=["Age", "Sex", "Sex_int"])

    logging.info("Step: Returning data to original per-atlas format")
    
    if validation_data is not None:
        total_df = pd.concat([training_data, validation_data]).drop(columns=["Dataset", "Diagnosis", "Sample_status"]).T
    else: 
        total_df = training_data.drop(columns=["Dataset", "Diagnosis", "Sample_status"]).T

    atlas_dfs = {}

    for atlas, abbrev in atlas_map.items():
        atlas_df = total_df[total_df.index.str.startswith(abbrev)]
        index_parts = np.array(atlas_df.index.str.split('^', n=2, expand=True).to_list())

        original_multiindex = pd.MultiIndex.from_arrays([
            index_parts[:,1], 
            index_parts[:,2]
        ], names=[None, 'Volume'])
        
        df_restored = atlas_df.copy()
        df_restored.index = original_multiindex
        df_original = df_restored.unstack('Volume')

        df_original = df_original.astype(float)

        atlas_dfs[atlas] = df_original
        path = f"{save_dir}/{atlas}_data{batch_corr_ext}_nors.csv"
        logging.info(f"Step: Saving unsampled / original {atlas} data to: {path}")
        df_original.to_csv(path)

    return rois, scalers


# This function is tailored to plot the batch corrected results after transforming the data with batch_corrected_preprocessing()
def umap_correction(
        data: pd.DataFrame, 
        metadata: pd.DataFrame,
        data_subset: str,
        ): 
    # Combine the data with the metadata
    corrected_data_meta = pd.merge(data.T, metadata, left_index=True, right_index=True)#.drop(columns=["Unnamed: 0", "Usage_original"])
    
    # Compute the embeddings
    logging.info("Computing UMAP embeddings")
    raw_data = corrected_data_meta.drop(columns=["Age", "Sex", "Sex_int", "Dataset", "Dataset_int", "Diagnosis", "Diagnosis_int", "Age_binned", "Age_binned_int"]).values
    reducer = umap.UMAP(random_state=42, n_neighbors=100)
    embedding = reducer.fit_transform(raw_data)
    
    diagnosis_colors = {
        'CTT-MDD': 'navy',
        'CTT-SCHZ': 'navy',
        "CTT": "black",
        'HC': 'green',
        'MDD': 'fuchsia',
        'SCHZ': 'red',
        'Unknown': 'gray',
    }

    dataset_colors = {
        "NSS": "mediumpurple",
        "whiteCAT": "gray", 
        "COBRE": "dodgerblue",
        "SRBPS": "violet",
        "MCIC": "red",
        "EPSY": "darkorange",
        "NU": "sienna",
        "IXI": "limegreen",
        # SRBPS subsets
        "SWA": "thistle",
        "HUH": "plum",
        "HRC": "purple",
        "HKH": "darkmagenta",
        "COI": "fuchsia",
        "KUT": "orchid",
        "KTT": "hotpink",
        "UTO": "pink",
        "ATT": "hotpink",
        "ATV": "deeppink",
        "CIN": "violet",
        # SRBPS aggr subsets
        "HUHRC": "cyan",
        "HCS": "olive",
        "KITKAT": "peru",
    }

    sex_colors = {
        "Male": "red",
        "Female": "blue"
    }

    color_by_list = ["Diagnosis", "Dataset", "Sex"]
    use_colors_list = [diagnosis_colors, dataset_colors, sex_colors]

    for color_by, use_colors in zip(color_by_list, use_colors_list):

        color_data = corrected_data_meta[color_by]#.to_list()

        point_colors = [use_colors[str(i)] for i in color_data]

        plt.figure(figsize=(7, 5))
        ax1 = plt.gca()

        ax1.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_colors,  # Fix 1: Use the list of colors generated above
            s=2,
            alpha=1,
            edgecolors="none",
        )

        ax1.set_aspect('equal', 'datalim')
        ax1.set_title(color_by, fontsize=12)
        ax1.set_xticks([])
        ax1.tick_params(length=0)
        ax1.set_yticks([])

        for spine in ax1.spines.values():
            spine.set_visible(False)

        unique_diagnoses = color_data.unique()
        handles_color = []
        for diag in unique_diagnoses:
            handles_color.append(plt.Line2D([0], [0], marker='o', color='w', label=diag,
                                            markerfacecolor=use_colors.get(diag, 'gray'),
                                            markersize=10))

        legend1 = ax1.legend(handles=handles_color, loc="upper left", bbox_to_anchor=(1.01, 0.65), frameon=False)
        plt.savefig(f"UMAP_{color_by}_{data_subset}.png")
        plt.savefig(f"UMAP_{color_by}_{data_subset}.svg")
    return


# The SRBPS dataset can be split into a total of eleven smaller datasets according to the scanner location. Some of these show strong confounding effects. 
# The grouping is based on a prior plotting of the original scanner locations and clustering them manually. 
def update_metadata_srbps(metadata_original: pd.DataFrame, srbps_data: pd.DataFrame):

    groups = {"HUHRC": ["HUH", "HRC"], "UTO": ["UTO"], "HCS": ["HKH", "COI", "SWA"], "KITKAT": ["KTT", "CIN", "ATV", "ATT", "KUT"]}

    metadata_original_new_index = metadata_original.set_index("Filename")
    srbps_files = metadata_original_new_index[metadata_original_new_index["Dataset"] == "SRBPS"].index.to_list()
    srbps_data_subset = srbps_data.loc[srbps_files,:]

    # Create a copy of metadata to avoid modifying the original
    metadata = metadata_original_new_index.copy()

    # Map the site information to the Diagnosis column for SRBPS entries
    # This will only update rows where Dataset == "SRBPS"
    srbps_mask = metadata_original_new_index["Dataset"] == "SRBPS"
    metadata.loc[srbps_mask, "Dataset"] = metadata.loc[srbps_mask].index.map(srbps_data_subset["site"])

    for new_dataset_name, old_datasets in groups.items():
        metadata["Dataset"].replace(old_datasets, new_dataset_name, inplace=True)

    return metadata.reset_index(drop=False)


# More general function to plot UMAPs with several different annotations. Primarily used to compute a joined plot across the non-NSS/WC and NSS/WC testing data.
# Too change the annotations, simply adjust the color_by_list and use_colors_list. 
def plot_umap(
        data: pd.DataFrame, 
        drop_cat_cols: List,
        save_path: str, 
        ):
    
    raw_data = data.drop(columns=drop_cat_cols).values
    reducer = umap.UMAP(random_state=42, n_neighbors=30)
    embedding = reducer.fit_transform(raw_data)
    
    diagnosis_colors = {
        'CTT-MDD': 'black',
        'CTT-SCHZ': 'black',
        "CTT": "black",
        'HC': 'darkorange',
        'MDD': 'blue',
        'SCHZ': 'red',
        'Unknown': 'gray',
    }

    diagnosis_colors_ctt = {
        'CTT-MDD': '#645DD7',
        'CTT-SCHZ': '#FF4242',
        "CTT": "lightgrey",
        'HC': 'lightgrey',
        'MDD': 'lightgrey',
        'SCHZ': 'lightgrey',
        'Unknown': 'lightgrey',
    }

    dataset_colors = {
        "NSS": "mediumpurple",
        "whiteCAT": "gray", 
        "COBRE": "dodgerblue",
        "SRBPS": "violet",
        "MCIC": "red",
        "EPSY": "darkorange",
        "NU": "sienna",
        "IXI": "limegreen",
        # SRBPS subsets
        "SWA": "darkmagenta",
        "HUH": "plum",
        "HRC": "plum",
        "HKH": "darkmagenta",
        "COI": "darkmagenta",
        "KUT": "black",
        "KTT": "black",
        "UTO": "pink",
        "ATT": "black",
        "ATV": "black",
        "CIN": "black",
        # SRBPS aggr subsets
        "HUHRC": "cyan",
        "HCS": "olive",
        "KITKAT": "peru",
    }

    sample_colors = {
        "valid": "blue",
        "sampled": "darkorange",
        "testing": "red",
    }

    sex_colors = {
        "Male": "red",
        "Female": "blue"
    }

    use_colors = {
        "Internal": "#F18805",
        "External": "#202C59",
    }

    color_by_list = ["Diagnosis", "Diagnosis", "Pred_class", "Dataset", "Use", "Sex"]
    use_colors_list = [diagnosis_colors, diagnosis_colors_ctt, diagnosis_colors, dataset_colors, use_colors, sex_colors]

    fig, axes = plt.subplots(1, len(color_by_list), figsize=(21, 5))

    # Handle case where there's only one plot
    if len(color_by_list) == 1:
        axes = [axes]

    for idx, (color_by, use_colors) in enumerate(zip(color_by_list, use_colors_list)):
        color_data = data[color_by]
        point_colors = [use_colors[str(i)] for i in color_data]

        ax = axes[idx]
        
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=point_colors,
            s=14,
            alpha=0.7,
            edgecolors="none",
        )

        ax.set_aspect('equal', 'datalim')
        ax.set_title(color_by, fontsize=12)
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        unique_diagnoses = color_data.unique()
        handles_color = []
        for diag in unique_diagnoses:
            handles_color.append(plt.Line2D([0], [0], marker='o', color='w', label=diag,
                                            markerfacecolor=use_colors.get(diag, 'gray'),
                                            markersize=10))

        legend1 = ax.legend(handles=handles_color, loc="upper left", bbox_to_anchor=(1.01, 1.00), frameon=False)

    plt.title("Internal and External Testing Data")
    plt.tight_layout()
    plt.savefig(f"{save_path}/UMAP_comparison_internal_external.png")
    plt.savefig(f"{save_path}/UMAP_comparison_internal_external.svg")

    return


# Used to plot saliency results. Currently only implemented for the neuromorphometrics atlas, as the other label maps couldn't be found.
# In principle this function isn't limited to saliency results. Any data in the form of ROI name : Data within a data frame can be plotted accordingly.
# Uses the neuroimaging specific packages nibabel (https://nipy.org/nibabel/) and nilearn (https://nilearn.github.io/stable/index.html).
def plot_brain_with_saliency(
    atlas_nifti_path,
    label_map_xml_path,
    saliency_df,
    roi_name_col_cleaned,
    data_col,
    save_path,
    vmax=None,
    cmap='Reds',
    ): 

    def normalize_spaces(text):
    #spaces and Unicode-spaces and other non-alphabetical letters in the names of the nifti or xml files should be normalized
    #original problem: ROI names in the xml label files had several spaces/Unicode signs between words instead of one
        if isinstance(text, str):
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            normalized_text = text.replace('\xa0', ' ')
            normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            return normalized_text.strip()
        return text

    # 1. load the atlas (e.g. neuromorphometrics)
    atlas_img = nib.load(atlas_nifti_path)
    atlas_data = atlas_img.get_fdata()

    # 2. Label-Map XML-file parsen and normalize spaces
    tree = ET.parse(label_map_xml_path)
    root = tree.getroot()

    label_map = {}
    for label in root.findall('Label'):
        name = normalize_spaces(label.find('Name').text)
        number = int(label.find('Number').text)
        label_map[name] = number
    print("ROI-Namen aus Label-Map normalisiert.")

    effect_map_data = np.zeros_like(atlas_data, dtype=float)

    found_rois = []
    missing_rois = []

    for index, row in saliency_df.iterrows():
        roi_name = str(row[roi_name_col_cleaned])
        effect_size = row[data_col]

        if roi_name in label_map:
            roi_number = label_map[roi_name]
            if np.any(atlas_data == roi_number):
                effect_map_data[atlas_data == roi_number] = effect_size
                found_rois.append(roi_name)
            else:
                missing_rois.append(f"{roi_name} (ROI-Nummer {roi_number} nicht im Atlas gefunden)")
        else:
            missing_rois.append(f"{roi_name} (Nicht in Label-Map gefunden)")

    effect_map_img = nib.Nifti1Image(effect_map_data, atlas_img.affine, atlas_img.header)

    # 6. Adjust vmax calculation for positive-only values
    if vmax is None:
        max_val = np.max(np.abs(effect_map_data))
        vmax = max_val
        min_val = np.min(np.abs(effect_map_data))
        vmin = min_val

    # 7. Create static orthogonal visualization
    static_title = f""
    
    fig_ortho = plotting.plot_stat_map(
        effect_map_img,
        bg_img=atlas_img,
        black_bg=False,
        display_mode='ortho',
        colorbar=True,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        title=static_title,
        cut_coords=(-155, 168, -124),
        output_file=f"{save_path}_static_visual.png",
        threshold=0.3*vmax, 
    )

    interactive_title = f""

    interactive_plot = plotting.view_img(
        effect_map_img,
        bg_img=atlas_img,
        black_bg=False,
        colorbar=True,
        symmetric_cmap=False,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        title=interactive_title,
        threshold=0.3*vmax, 
    )
    
    return interactive_plot