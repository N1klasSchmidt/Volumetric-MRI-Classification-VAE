import sys
import json
sys.path.append("/workspace/src")

# Import general packages 
import numpy as np
import pandas as pd
import os
from typing import List
import torch
from torch.cuda.amp import GradScaler
import logging
from typing import Dict
import pathlib
import copy 

# Import utils for processing of data, logging and configuration
from utils.config_utils import Config_XCov_CV, TestingConfig

from utils.helper_utils import (
    resplit_data,
    oversample,
    undersample,
    hybrid_sample,
    no_resample, 
    batch_correct,
    plot_umap
    )

from utils.data_utils import load_checkpoint_model
from utils.helper_utils import split_annotations, determine_split, get_atlas_map
from utils.logging_utils import setup_logging, log_title
from utils.logging_utils import setup_prediction_logging, log_prediction_metrics
                                      
# Import from other scripts the necessary functions to train the models
from VAE2_training import train_model
from VAE2_loading import load_data_for_model
from VAE2_prediction import Predictor
from models.VAE2 import VAE2
from models.VAE2_no_XCov import VAE2_no_XCov

from utils.downstream_utils import (
    VAE2_Wrapper,
    VAE2_no_XCov_Wrapper,
    VAE2_Wrapper_latent,
    VAE2_no_XCov_Wrapper_latent,
    run_shap_analysis,
    get_all_correlations,
    predict_unseen_risk_from_latent,
    predict_clinical_from_latent_updt,
    trace_latent_to_roi,
    saliency_from_latent,
    get_latent_feature_correlation_plot,
    visualize_saliency,
    aggregate_testing_data
)

# ====================================================================================================
#
# **RUN Aggregated VAE model with Expeiment Config Parameters**
#
# This is the central script for running different experiments. It takes the experiment and model 
# configs and handles them accordingly. All output data is saved to a single experiment under the 
# 'experiments' folder. The split data is saved to the data folder.
#
# ====================================================================================================


def run_VAE_experiment(
        # Config dictionary with experimental parameters
        experiment_config: Dict, 
        # Config dictionary with model parameters
        model_config: Dict
        ):
    
    # ------------------------------ STEP 1: Prepare the parameters of the run ------------------------------
    atlas_map = get_atlas_map(experiment_config["data"]["input_atlases"])
    metadata = pd.read_csv(experiment_config["data"]["metadata_path"], header=[0])
    
    for data_col in ["Filename", "Sex", "Sex_int", "Dataset", "Diagnosis", "Age"]:
        assert data_col in metadata.columns.to_list(), f"{data_col} not found in metadata. Abborting run."
    
    metadata = metadata[["Filename", "Sex", "Sex_int", "Dataset", "Diagnosis", "Age"]]

    for covar in model_config["covariate_names"]:
        assert covar in metadata.columns.to_list(), f"Covariate '{covar}' not found. Please use covariates for which there is a column in the provided metadata!"

    if experiment_config["experiment"]["run_name"] is None:
        atlas_abbrevs = [i[1] for i in atlas_map.values()]
        run_id = ""
        for i in atlas_abbrevs:
            run_id += i
        run_name = f"experiment_{run_id}"
    else:
        run_name = experiment_config["experiment"]["run_name"]

    vol_to_use = None

    if experiment_config["experiment"]["train_model"] == False: 
        assert experiment_config["experiment"]["base_experiment_to_load"] is not None, "If training is set to false, specify an experiment to emulate."
        
        if experiment_config["experiment"]["base_experiment_to_load"].endswith(".pth"): 
            trained_model_path = experiment_config["experiment"]["base_experiment_to_load"]
        else: 
            training_model_dir = os.path.join(experiment_config["experiment"]["base_experiment_to_load"], "training", "models")
            trained_model_paths = list(pathlib.Path(training_model_dir).rglob(f"*.pth"))

            assert len(trained_model_paths) == 1, "Multiple or no models found at destination. Please specify exact path or choose a different experiment to load."

            trained_model_path = trained_model_paths[-1]

    diagnosis_colors_plotting = {}
    for diag, color in zip(experiment_config["data"]["diagnoses"], experiment_config["data"]["diagnosis_colors"]):
        diagnosis_colors_plotting[diag] = color 

    # ------------------------------ STEP 3: Prepare the config of the run ------------------------------
    config = Config_XCov_CV(
            # General Parameters
            RUN_NAME=run_name,
            # Input / Output Paths 
            ATLAS_NAME="training",
            OUTPUT_DIR=experiment_config["experiment"]["output_root_dir"],
            # Loading Model
            LOAD_MODEL=False,
            PRETRAIN_MODEL_PATH=None,
            PRETRAIN_METRICS_PATH=None,
            CONTINUE_FROM_EPOCH=0,
            # Loss Parameters
            RECON_LOSS_WEIGHT=model_config["model_parameters"]["reconstruction_loss"],
            KLDIV_LOSS_WEIGHT=model_config["model_parameters"]["kldiv_loss"],
            CLASS_LOSS_WEIGHT=model_config["model_parameters"]["classification_loss"],
            XCOV_LOSS_WEIGHT=model_config["model_parameters"]["xcov_loss"],
            XCOV_CAT_LOSS_WEIGHT=model_config["model_parameters"]["xcov_class_loss"],
            ORTHO_LOSS_WEIGHT=model_config["model_parameters"]["orthogonality_loss"],
            # Learning and Regularization
            TOTAL_EPOCHS=model_config["model_parameters"]["epochs"],
            LEARNING_RATE=model_config["model_parameters"]["learning_rate"],
            WEIGHT_DECAY=model_config["model_parameters"]["weight_decay"],
            EARLY_STOPPING=model_config["model_parameters"]["early_stopping"],
            STOP_LEARNING_RATE=4e-8,
            SCHEDULE_ON_VALIDATION=True,
            SCHEDULER_PATIENCE=6,
            SCHEDULER_FACTOR=0.5,
            # Visualization
            CHECKPOINT_INTERVAL=5,
            DONT_PLOT_N_EPOCHS=0,
            UMAP_NEIGHBORS=30,
            UMAP_DOT_SIZE=20,
            METRICS_ROLLING_WINDOW=10,
            PLOTTING_COLORS=diagnosis_colors_plotting,
            # Data Parameters
            BATCH_SIZE=model_config["model_parameters"]["batch_size"],
            DIAGNOSES=experiment_config["data"]["diagnoses"], #This is just a placeholder. It is overwritten for training and testing.
            COVARS=model_config["covariate_names"],
            # Misc.
            LATENT_DIM=10, ##################################################### PLACEHOLDER, REMOVE
            SHUFFLE_DATA=False,
            CE_CLASS_WEIGHTS=None,
            ALL_LABELS=False,
            PURPOSE="train",
            VOLUMES=vol_to_use,
            DROPOUT_PROB=model_config["model_parameters"]["dropout_probability"],
            HIDDEN_DIM=model_config["model_parameters"]["hidden_dimension_size"],
        )
    config.update_working_dir()
    config.update_dirs()

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    setup_logging(config)

    # ------------------------------ STEP 3: Split, batch correct and resample data ------------------------------
    logging.info(log_title("TRAIN-TEST SPLITTING" ))
    metadata_dict = determine_split(experiment_config["data"]["train_test_split"], metadata)
    resplit_data(df_dict=metadata_dict, output_dir=config.OUTPUT_DIR, data_type=experiment_config["data"]["data_format"])

    bat_corr_ext = ""
    if experiment_config["data"]["batch_correction"]:
        batch_correct(atlas_map, metadata, metadata_dict, experiment_config["data"]["data_dir"], experiment_config["data"]["data_format"])
        bat_corr_ext = "_bat_corr_"

    config.TRAIN_DATA_PATH = f"/workspace/data/raw_data_train_valid"

    train_annotations_split, valid_annotations_split = split_annotations(annotations=metadata_dict["train_valid"], proportion=experiment_config["data"]["train_proportion"])

    # Oversampling of the minority class labels
    section_title = "OVER- / UNDERSAMPLING" 
    logging.info(log_title(section_title))

    resample_method = experiment_config["data"]["resampling_method"]

    print(experiment_config["data"]["resampling_method"])
    if experiment_config["data"]["resampling"] == "over":
        logging.info(f"Performing oversampling of data with method: {resample_method}")
        train_df_sampled, rois, scalers = oversample(files_to_oversample=train_annotations_split,
                                    files_not_to_oversample=valid_annotations_split,
                                    method=experiment_config["data"]["resampling_method"],
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map,
                                    batch_corr_ext=bat_corr_ext)
        resampling_ext = f"{bat_corr_ext}os.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    # Undersampling of the majority class labels
    elif experiment_config["data"]["resampling"] == "under":
        logging.info(f"Performing undersampling of data with method: {resample_method}") 
        train_df_sampled, rois, scalers = undersample(files_not_to_undersample=valid_annotations_split,
                                    files_to_undersample=train_annotations_split,
                                    method=experiment_config["data"]["resampling_method"],
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map,
                                    batch_corr_ext=bat_corr_ext)
        resampling_ext = f"{bat_corr_ext}us.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    # Undersampling of the majority class label and oversampling of the minority class label to target label
    elif experiment_config["data"]["resampling"] == "hybrid":
        logging.info(f"Performing hybrid sampling of data with method: {resample_method}")

        assert len(experiment_config["data"]["hybrid_ranking"]) == 3, f"Hybrid sampling only available for three diagnoses. {len(experiment_config['data']['hybrid_ranking'])} is an invalid number of diagnoses!"
        assert len(train_annotations_split["Diagnosis"].unique()) == 3, f"Training annotations don't contain exactly 3 unique diagnoses. Instead {len(train_annotations_split['Diagnosis'].unique())}."

        diag_ranks = experiment_config["data"]["hybrid_ranking"]

        train_df_sampled, rois, scalers = hybrid_sample(files_not_to_resample=valid_annotations_split,
                                        files_to_resample=train_annotations_split,
                                        class_props= {"majority": diag_ranks[2], "target": diag_ranks[1], "minority": diag_ranks[0]},
                                        save_dir="/workspace/data/raw_data_train_valid",
                                        atlas_map=atlas_map,
                                        batch_corr_ext=bat_corr_ext)
        resampling_ext = f"{bat_corr_ext}rs.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    elif experiment_config["data"]["resampling"]  == None:
        logging.info(f"Scaling and transforming training data.")
        rois, scalers = no_resample(validation_files=valid_annotations_split,
                                    training_files=train_annotations_split,
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map)
        train_df_sampled = train_annotations_split
        resampling_ext = f"{bat_corr_ext}nors.csv"
        train_valid_df_sampled = pd.concat([train_annotations_split, valid_annotations_split])

    else: 
        logging.info("Please specify a valid resampling method. This can be either 'over', 'under', 'hybrid', or None")

    train_valid_df_sampled.to_csv(f"{config.OUTPUT_DIR}/train_valid_metadata.csv")

    xcov_groups_train = train_annotations_split["Dataset"].unique().tolist()
    xcov_groups_valid = valid_annotations_split["Dataset"].unique().tolist()

    training_diagnoses = train_annotations_split["Diagnosis"].unique().tolist()
    config.DIAGNOSES = training_diagnoses

    assert len(xcov_groups_train) == len(xcov_groups_valid)

    num_xcov_groups = len(xcov_groups_train)
    config.NUM_XCOV_GROUPS = num_xcov_groups

    # ------------------------------ STEP 4: Load data and train model ------------------------------

    train_dataloader, valid_dataloader, atlas_lengths, annotations = load_data_for_model(config=config, 
                                                                                        train_df=train_df_sampled,
                                                                                        valid_df=valid_annotations_split,
                                                                                        resampling_ext=resampling_ext,
                                                                                        atlas_map=atlas_map)

    if experiment_config["experiment"]["train_model"]:
        section_title = "START MODEL TRAINING" 
        logging.info(log_title(section_title))

        trained_model, _ = train_model(config=config,
                                                train_loader=train_dataloader,
                                                valid_loader=valid_dataloader,
                                                annotations=annotations,
                                                atlas_lengths=atlas_lengths,
                                                xcov=model_config["xcov"])
    else: 
        logging.info(log_title("LOAD MODEL"))

        if model_config["xcov"]:
            model_architecture = VAE2(
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
            )
        else: 
            model_architecture = VAE2_no_XCov(
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
            )

        trained_model = load_checkpoint_model(model=model_architecture, model_filename=trained_model_path)

    # ------------------------------ STEP 5: Perform the testing with the trained model ------------------------------
    test_dirs = [item for item in metadata_dict.keys() if item.startswith("test")]
    assert sorted(experiment_config["analysis"].keys()) == sorted(test_dirs), "Ensure that the testing runs under 'experiment' correspond with the setup data split"

    for testing_run, testing_data in metadata_dict.items():
        if not testing_run.startswith("test"):
            continue 

        testing_run_diagnoses = testing_data["Diagnosis"].unique().tolist()
        testing_run_experimental_details = experiment_config["analysis"][testing_run]

        trained_model_config = copy.deepcopy(config)
        
        trained_model_config.DIAGNOSES = testing_run_diagnoses
        trained_model_config.PURPOSE = testing_run

        testing_config = TestingConfig(
            UMAP_NEIGHBORS=100, 
            SEED=123,
            TIMESTAMP=None,
            DATA_DIR=trained_model_config.OUTPUT_DIR, # We save the testing data in the same place as training!
            XCov=model_config["xcov"],
            mode=str(testing_run),
        )

        testing_config.update_logging_dir_vae2()

        setup_prediction_logging(config=testing_config)

        logging.info(log_title(f"TESTING RUN {testing_run}"))

        label_mapping = {}
        for idx, diagnosis in enumerate(sorted(trained_model_config.DIAGNOSES)): 
            label_mapping[idx] = diagnosis

        # if model_config["xcov"]:
        predictor = Predictor(logging_config=testing_config, 
                              model_config=trained_model_config,
                              model=trained_model, 
                              run_id=testing_run, 
                              device=config.DEVICE,
                              label_mapping=label_mapping,
                              original_diagnoses=config.DIAGNOSES,
                              extended_data_path=experiment_config["analysis"][testing_run]["extra_metadata_path"], 
                              xcov=model_config["xcov"]
                              )

        all_data_paths_filtered = {}
        for atlas in sorted(atlas_map.keys()):
            all_data_paths_filtered[atlas] = os.path.join(experiment_config["data"]["data_dir"], f"raw_data_{testing_run}", f"{atlas}_data.csv")

        logging.info("Step: Initialize testing data")
        predictor.init_test_data(test_annotations=metadata_dict[testing_run],
                                 data_paths=all_data_paths_filtered,
                                 scalers=scalers
                                 )
        
        logging.info("Step: Computing performance")
        model_pred_metrics = predictor.compute_testing_performance(colors=diagnosis_colors_plotting)
        logging.info(f"Performance of model on testing data: {model_pred_metrics}")

        log_prediction_metrics(metrics=model_pred_metrics, epoch="best")

        # Get plot for internal latent correlations
        get_latent_feature_correlation_plot(testing_config.LOGGING_DIR, atlas_map, atlas_lengths)


        # ------------------------------ STEP 7: Downstream analysis of model outputs on testing data ------------------------------

        if experiment_config["analysis"][testing_run]["downstream"]: 
            logging.info(log_title("DOWNSTREAM ANALYSIS"))
            downstream_config = experiment_config["analysis"][testing_run]
            assert downstream_config["extra_metadata_path"] is not None, "To perform downstream analysis, pass additional metadata!"

            extended_metadata_raw = pd.read_csv(downstream_config["extra_metadata_path"], header=[0], index_col=0)
            assert extended_metadata_raw["Filename"].isin(metadata["Filename"]).all(), "Not all extended metadata entries found also in full metadata!"
            extended_metadata_raw.set_index("Filename", inplace=True, drop=True)

            extended_metadata = extended_metadata_raw.loc[testing_data["Filename"].to_list(), :]

            extended_metadata_only_new = extended_metadata.copy()
            for var in ["Age", "Diagnosis", "Sex", "Dataset", "Codiagnosis"]: 
                if var in extended_metadata_only_new.columns.to_list():
                    extended_metadata_only_new.drop(columns=[var], inplace=True)

            latent_data = pd.read_csv(f"{testing_config.LOGGING_DIR}/latent_space_data.csv", index_col=0)
            model_predictions = pd.read_csv(f"{testing_config.LOGGING_DIR}/testing_metadata_extended.csv", index_col=0)

            unseen_diagnoses = [diag for diag in testing_run_diagnoses if not diag in config.DIAGNOSES]

            if downstream_config["correlation_analysis"]:
                logging.info(log_title("CORRELATION ANALYSIS"))

                save_path = os.path.join(testing_config.LOGGING_DIR, "CORR")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Compute correlations for each training-unseen diagnosis separately
                get_all_correlations(clinical_data=extended_metadata_only_new,
                                    model_data=model_predictions,
                                    metadata=metadata,
                                    diagnoses=unseen_diagnoses,
                                    save_path=save_path,
                                    original_model_diagnoses=config.DIAGNOSES,
                                    ) 
                
                # Compute correlations across all diagnoses
                get_all_correlations(clinical_data=extended_metadata_only_new,
                                    model_data=model_predictions,
                                    metadata=metadata,
                                    diagnoses=None,
                                    save_path=save_path,
                                    original_model_diagnoses=config.DIAGNOSES,
                                    )
                
            
            if downstream_config["latent_feature_analysis"]:
                logging.info(log_title("LATENT FEATURE ANALYSIS"))

                save_path = os.path.join(testing_config.LOGGING_DIR, "LATENT")
                os.makedirs(save_path)

                # We predict two things from the latent vectors: 1. A "Unseen-Risk" and 2. Clinical variables like Age or nss_motor
                # Besides simply knowing which latent features are involved in this prediction we can also determine which ROIs these features correspond with.
                # For this there are again two approaches:
                # 1. Use (Gradient)SHAP to compute which ROIs correspond with which latent feature. This is also just handy as an overview.
                # 2. Use a "chained Saliency" where we use the chain rule to propagate the gradients (=coeff.) of a linear model through the model encoder to get the ROI gradients. 

                for unseen_diagnosis in unseen_diagnoses:
                    
                    extended_metadata_unseen_diag = extended_metadata.copy()
                    extended_metadata_unseen_diag[unseen_diagnosis] = (extended_metadata_unseen_diag["Diagnosis"] == unseen_diagnosis).astype(int)

                    ########## Unseen-Diagnosis-Risk Prediction ##########
                    unseen_risk_latent_importance, unseen_risk_coeff = predict_unseen_risk_from_latent(
                        target=unseen_diagnosis,
                        clinical=extended_metadata_unseen_diag,
                        latent=latent_data,
                        save_path=save_path,
                        colors=diagnosis_colors_plotting)

                    unseen_risk_filename = saliency_from_latent(unseen_risk_coeff, predictor, config, rois, save_path, config.DEVICE)

                    if "neuromorphometrics" in experiment_config["data"]["input_atlases"]: 
                        logging.info(f"Plotting saliency results on neuromorphometrics.")
                        visualize_saliency(unseen_risk_filename, "neuromorphometrics", ["Vgm", "Vwm", "Vcsf"], method="weighted")
                    else: 
                        logging.info(f"Unable to plot saliency results, as 'neuromorphometrics' not included as atlas input.")

                ########## Clinical Variable Prediction ##########
                clinical_results, clinical_latent_importance, clinical_coeff = predict_clinical_from_latent_updt(clinical=extended_metadata,
                                                                        latent=latent_data,
                                                                        save_path=save_path)
                
                with open(f"{save_path}/significant_latent_feaures.json", "w") as json_file:
                    json.dump(clinical_latent_importance, json_file, indent=4)

                clinical_results.to_csv(f"{save_path}/clinical_pred_results.csv")
                logging.info(f"Results saved to: {save_path}/clinical_pred_results.csv")
                
                ########## Analysis of ROIs ##########
                logging.info(log_title("MAP ROI - LATENT"))

                latent_dim = len(latent_data.columns.to_list())

                if model_config["xcov"]:
                    wrapped_model_for_latent = VAE2_Wrapper_latent(trained_model, config.DEVICE)
                else:
                    wrapped_model_for_latent = VAE2_no_XCov_Wrapper_latent(trained_model, config.DEVICE)

                mapping_roi_to_latent = trace_latent_to_roi(config.DEVICE, wrapped_model_for_latent, valid_dataloader, rois, 3, latent_dim)
                mapping_roi_to_latent.to_csv(f"{save_path}/map_latent_to_roi.csv")

                for pred_var, features in clinical_latent_importance.items():
                    print(f"\n")
                    logging.info(f"########## Mapping for: {pred_var} ##########")
                    for feature in features: 
                        feature_rois = mapping_roi_to_latent.loc[f"Feature_{feature}"].to_list()
                        logging.info(f"\t{feature_rois}")

                if len(clinical_coeff.keys()) > 0:
                    _ = saliency_from_latent(clinical_coeff, predictor, config, rois, save_path, config.DEVICE)
            

            if downstream_config["SHAP_analysis"]: 
                logging.info(log_title("FEATURE IMPORTANCE"))

                save_path = os.path.join(testing_config.LOGGING_DIR, "SHAP")
                os.makedirs(save_path)

                if model_config["xcov"]: 
                    wrapped_model = VAE2_Wrapper(trained_model, config.DEVICE)
                else: 
                    wrapped_model = VAE2_no_XCov_Wrapper(trained_model, config.DEVICE)

                model_outputs = sorted(train_annotations_split["Diagnosis"].unique().tolist())

                for idx, label in enumerate(model_outputs): 

                    logging.info(f"Running SHAP for {label}")
                    run_shap_analysis(wrapped_model, 
                                    dataloader=valid_dataloader,
                                    target_class_idx=idx,
                                    class_label=label,
                                    device=config.DEVICE, 
                                    feature_names=rois,
                                    dir_name=save_path)
    
    # If all testings have been performed, we can plot the joint results in a summarized umap. To do this, please adjust the color parameters in the plot_umap function individually!
    # if experiment_config["analysis"]["joint_umap"]:
    #     logging.info(log_title("PLOTTING JOINT UMAP"))
    #     load_path = config.LOGGING_DIR
    #     total_testing_data = aggregate_testing_data(load_path, config.DIAGNOSES)
    #     plot_umap(total_testing_data, ["Age", "Sex", "Dataset", "Diagnosis", "Pred_class"], load_path)

    logging.info("Experiment concluded successfully!")
    return
            
        

        
