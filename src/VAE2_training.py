# Import general packages
import sys
sys.path.append("/workspace/src")
import matplotlib
import numpy as np
import pandas as pd
import os
import torch
from torch.cuda.amp import GradScaler
import logging
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from models.VAE2 import VAE2, train_VAE2
from models.VAE2_no_XCov import VAE2_no_XCov, train_VAE2_no_XCov
from utils.data_utils import load_checkpoint_model

from utils.logging_utils import (
    log_model_ready,
    log_model_setup,
    log_title,
    )

# Use non-interactive plotting to avoid tmux crashes
matplotlib.use("Agg")


# ====================================================================================================
#
# **TRAIN model**
#
# This script handles the initialization of a new, untrained model and training it with the specified 
# parameters. It is called during the RUN_aggr_VAE.py script. The output is the best model (determined
# by lowest validation loss, after 25 epochs) and the accuarcy at this epoch. To determine which model
# is selcted as the best, please adjust the train_ function in the /models/ scripts. 
#
# ====================================================================================================


def train_model(
        # Config dictionary with model parameters
        config: Dict, 
        # PyTorch dataloader that handles training data
        train_loader: DataLoader, 
        # PyTorch dataloader that handles validation data 
        valid_loader: DataLoader, 
        # Metadata data frame with all training and validation annotations
        annotations: pd.DataFrame, 
        # Dictionary specifying the input lengths of each atlas' data e.g. Neuromorphometrics: 408
        atlas_lengths: Dict, 
        # Should the trained model use XCov or not?
        xcov: bool = True,
        ):

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    #Initialize Model
    log_model_setup()
    logging.info(log_title("INITIALIZE MODEL"))
    if xcov:
        model = VAE2(
            num_classes=len(config.DIAGNOSES),
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            device=config.DEVICE,
            scaler=GradScaler(),
            kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
            recon_loss_weight=config.RECON_LOSS_WEIGHT,
            class_loss_weight=config.CLASS_LOSS_WEIGHT,
            xcov_weight=config.XCOV_LOSS_WEIGHT,
            xcov_cat_weight=config.XCOV_CAT_LOSS_WEIGHT,
            ortho_weight=config.ORTHO_LOSS_WEIGHT,
            latent_dim=config.LATENT_DIM,
            atlas_input_dims=atlas_lengths,
            schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
            scheduler_patience=config.SCHEDULER_PATIENCE,
            scheduler_factor=config.SCHEDULER_FACTOR,
            num_xcov_groups=config.NUM_XCOV_GROUPS,
            ce_class_weights=config.CE_CLASS_WEIGHTS,
            hidden_dim=config.HIDDEN_DIM,
            dropout_prob=config.DROPOUT_PROB,
            xcov_covar=config.COVARS[0]
        )
    else: 
        model = VAE2_no_XCov(
            num_classes=len(config.DIAGNOSES),
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            device=config.DEVICE,
            scaler=GradScaler(),
            kldiv_loss_weight=config.KLDIV_LOSS_WEIGHT,
            recon_loss_weight=config.RECON_LOSS_WEIGHT,
            class_loss_weight=config.CLASS_LOSS_WEIGHT,
            xcov_weight=config.XCOV_LOSS_WEIGHT,
            xcov_cat_weight=config.XCOV_CAT_LOSS_WEIGHT,
            ortho_weight=config.ORTHO_LOSS_WEIGHT,
            latent_dim=config.LATENT_DIM,
            atlas_input_dims=atlas_lengths,
            schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
            scheduler_patience=config.SCHEDULER_PATIENCE,
            scheduler_factor=config.SCHEDULER_FACTOR,
            num_xcov_groups=config.NUM_XCOV_GROUPS,
            ce_class_weights=config.CE_CLASS_WEIGHTS,
            hidden_dim=config.HIDDEN_DIM,
            dropout_prob=config.DROPOUT_PROB,
            xcov_covar=config.COVARS[0]
        )

    if config.LOAD_MODEL:
        model = load_checkpoint_model(model, config.PRETRAIN_MODEL_PATH)
        model_metrics = pd.read_csv(config.PRETRAIN_METRICS_PATH)
    else:
        model_metrics = pd.DataFrame(
            {
                "train_loss": [],
                "t_class_loss": [],
                "t_recon_loss": [],
                "t_kldiv_loss": [],
                "t_xcov_loss": [],
                "valid_loss": [],
                "v_class_loss": [],
                "v_recon_loss": [],
                "v_kldiv_loss": [],
                "v_xcov_loss": [],
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1-score": [],
                "learning_rate": [],
            }
        )

    log_model_ready(model)

    logging.info(log_title("START TRAINING CYCLE"))

    if xcov:
        max_accuracy, trained_model = train_VAE2(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            annotations=annotations,
            model_metrics=model_metrics,
            config=config,
            no_plotting=True,
            no_val_plotting=True,
            no_saving=True
        )

    else:
        max_accuracy, trained_model = train_VAE2_no_XCov(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            annotations=annotations,
            model_metrics=model_metrics,
            config=config,
            no_plotting=True,
            no_val_plotting=True,
            no_saving=True
        )

    print(f"Maximum accuracy of model {config.ATLAS_NAME}: {max_accuracy}")
    return trained_model, max_accuracy
