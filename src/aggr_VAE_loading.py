# Import general packages
import sys
sys.path.append("/workspace/src")
import matplotlib
import numpy as np
import pandas as pd
import os
import torch
import logging

from utils.data_utils import (
    load_mri_data,
    process_subjects,
    )

from utils.logging_utils import (
    log_and_print,
    log_data_loading,
    log_atlas_mode,
    )

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")

def load_data_for_model(config, train_df, valid_df, resampling_ext, atlas_map):

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    all_training_data_paths = {}
    for atlas in sorted(atlas_map.keys()):
        all_training_data_paths[atlas] = os.path.join(config.TRAIN_DATA_PATH, f"{atlas}_data_{resampling_ext}")

    logging.info("Loading training and validation data")

    train_subjects, train_annotations, atlas_lengths = load_mri_data(
        annotations=train_df,
        atlas_data_paths=all_training_data_paths,
        covars=config.COVARS,
        diagnoses=config.DIAGNOSES,
        train_or_test="train",
        # How to scale and subset the data
        vol_to_use=None,
        batch_corrected=config.BATCH_CORRECTION,
    )

    valid_subjects, valid_annotations, _ = load_mri_data(
        annotations=valid_df,
        atlas_data_paths=all_training_data_paths,
        covars=config.COVARS,
        diagnoses=config.DIAGNOSES,
        train_or_test="train",
        # How to scale and subset the data
        vol_to_use=None,
        batch_corrected=config.BATCH_CORRECTION,
    )

    len_atlas = len(train_subjects[0]["measurements"])

    train_annotations.insert(1, "Data_Type", "train")  # Inserts a new column called "Data_Type" with "train" for values at the second position.
    valid_annotations.insert(1, "Data_Type", "valid")  # Inserts a new column called "Data_Type" with "valid" for values at the second position.

    annotations = pd.concat([train_annotations, valid_annotations])  # Stacks the two data frames again after splitting.
    annotations.sort_values(by=["Data_Type", "Filename"], inplace=True)
    annotations.reset_index(drop=True, inplace=True)

    annotations = annotations.astype(
        {
            "Age": "float",
            "Dataset": "category",
            "Diagnosis": "category",
            "Sex": "category",
            "Data_Type": "category",
            "Filename": "category",
        }
    )

    log_and_print(annotations)

    train_loader = process_subjects(
        subjects=train_subjects,
        batch_size=config.BATCH_SIZE,
        shuffle_data=config.SHUFFLE_DATA,
        covars=config.COVARS
    )

    valid_loader = process_subjects(
        subjects=valid_subjects,
        batch_size=config.BATCH_SIZE,
        shuffle_data=False,
        covars=config.COVARS,
    )

    # Log the used atlas and the number of ROIs
    log_atlas_mode(atlas_name=config.ATLAS_NAME, 
                num_rois=len_atlas
                )

    # Log data setup
    log_data_loading(
        datasets={
            "Training Data": len(train_subjects),
            "Validation Data": len(valid_subjects),
        }
    )
    return train_loader, valid_loader, atlas_lengths, annotations
