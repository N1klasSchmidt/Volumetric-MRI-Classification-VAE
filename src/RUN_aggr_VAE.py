# ------------------------------ STEP 0: Import necessary packages and scripts ------------------------------
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

# Import utils for processing of data, logging and configuration
from utils.config_utils import Config_XCov_CV, TestingConfig

from utils.helper_utils import (
    resplit_data,
    oversample,
    undersample,
    hybrid_sample,
    no_resample, 
    plot_umap
    )

from utils.data_utils import load_checkpoint_model
from utils.helper_utils import split_annotations, determine_split, get_atlas_map
from utils.logging_utils import setup_logging, log_title
from utils.logging_utils import setup_prediction_logging, log_prediction_metrics
                                      
# Import from other scripts the necessary functions to train the models
from aggr_VAE_training import train_model
from aggr_VAE_loading import load_data_for_model
from aggr_VAE_prediction import SinglePredictor
from aggr_VAE_no_XCov_prediction import SinglePredictor_no_XCov
from models.aggr_VAE import NewVAE

from utils.downstream_utils import (
    NewVAE_Wrapper,
    NewVAE_no_XCov_Wrapper,
    NewVAE_Wrapper_latent,
    NewVAE_no_XCov_Wrapper_latent,
    run_shap_analysis,
    get_all_correlations,
    rename_nss_wc,
    run_interaction_term_analysis,
    predict_ctt_risk_from_latent,
    predict_clinical_from_latent,
    trace_latent_to_roi,
    saliency_from_latent,
    get_latent_feature_correlation_plot,
    visualize_saliency
)

# ====================================================================================================
# **RUN Aggregated VAE model with Expeiment Config Parameters**
#
# This is the central script for running different training iterations. Through the parameters at the
# beginning of the script the run parameters can be adjusted  
# ====================================================================================================


def run_VAE_experiment(experiment_config: Dict):
    
    atlas_map = get_atlas_map(experiment_config["data"]["input_atlases"])
    metadata = pd.read_csv(experiment_config["data"]["metadata_path"], header=[0])
    
    for data_col in ["Filename", "Sex", "Sex_int", "Dataset", "Diagnosis"]:
        assert data_col in metadata.columns.to_list(), f"{data_col} not found in metadata. Abborting run."
    
    metadata = metadata[["Filename", "Sex", "Sex_int", "Dataset", "Diagnosis"]]

    for covar in experiment_config["model"]["covariate_names"]:
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

    if experiment_config["experiment"]["train"] == False: 
        assert experiment_config["experiment"]["base_experiment_to_load"] is not None, "If training is set to false, specify an experiment to emulate."
        
        if experiment_config["experiment"]["base_experiment_to_load"].endswith(".pth"): 
            trained_model_path = experiment_config["experiment"]["base_experiment_to_load"]
        else: 
            training_model_dir = os.path.join(experiment_config["experiment"]["base_experiment_to_load"], "training", "models")
            trained_model_paths = list(pathlib.Path(training_model_dir).rglob(f"*.pth"))

            assert len(trained_model_paths) == 1, "Multiple models found at destination. Please specify exact path."

            trained_model_path = trained_model_paths[-1]

    diagnosis_colors_plotting = {}
    for diag, color in zip(experiment_config["data"]["diagnoses"], experiment_config["data"]["diagnosis_colors"]):
        diagnosis_colors_plotting[diag] = color 

    # ------------------------------ STEP 3: Prepare the config file and parameters of the run ------------------------------

    params = pd.read_csv(experiment_config["model"]["params"]).set_index("Unnamed: 0")
    params.index.name = None

    atlas_name = "aggr_VAE"

    weight_decay = params.loc["weight_decay", :][atlas_name]
    learning_rate = params.loc["learning_rate", :][atlas_name]
    latent_dim = int(params.loc["latent_dim", :][atlas_name])
    num_epochs = int(params.loc["num_epochs", :][atlas_name])
    kldiv_weight = params.loc["kl_div_weight", :][atlas_name]
    class_weight = params.loc["class_weight", :][atlas_name]
    recon_weight = params.loc["recon_weight", :][atlas_name]
    xcov_weight = params.loc["xcov_weight", :][atlas_name]
    xcov_cls_weight = params.loc["xcov_cls_weight", :][atlas_name]

    # Initialize the config file for this run. 

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
            RECON_LOSS_WEIGHT=recon_weight,
            KLDIV_LOSS_WEIGHT=kldiv_weight,
            CLASS_LOSS_WEIGHT=class_weight,
            XCOV_LOSS_WEIGHT=xcov_weight,
            XCOV_CAT_LOSS_WEIGHT=xcov_cls_weight,
            # Learning and Regularization
            TOTAL_EPOCHS=num_epochs,
            LEARNING_RATE=learning_rate,
            WEIGHT_DECAY=weight_decay,
            EARLY_STOPPING=True,
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
            BATCH_SIZE=128,
            DIAGNOSES=experiment_config["data"]["diagnoses"],
            COVARS=experiment_config["model"]["covariate_names"],
            # Misc.
            LATENT_DIM=latent_dim,
            SHUFFLE_DATA=False,
            FINE_TUNING=False,
            CE_CLASS_WEIGHTS=None, #[1.0, 20.0, 1.0],
            ALL_LABELS=False,
            PURPOSE="train",
            VOLUMES=vol_to_use,
            DROPOUT_PROB=0.2,
            HIDDEN_DIM=100,
        )
    config.update_working_dir()
    config.update_dirs()

    # Set seed for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    setup_logging(config)

    logging.info(log_title("TRAIN-TEST SPLITTING" ))
    metadata_dict = determine_split(experiment_config["data"]["train_test_split"], metadata)
    resplit_data(df_dict=metadata_dict, output_dir=config.OUTPUT_DIR)

    config.TRAIN_DATA_PATH = f"/workspace/data/raw_data_train_valid"

    train_annotations_split, valid_annotations_split = split_annotations(annotations=metadata_dict["train_valid"], proportion=experiment_config["data"]["train_proportion"])

    # Oversampling of the minority class labels (MDD and SCHZ)
    section_title = "OVER- / UNDERSAMPLING" 
    logging.info(log_title(section_title))

    if experiment_config["data"]["resampling"] == "over":
        logging.info(f"Performing oversampling of data with method: {experiment_config["data"]["resampling_method"]}")
        train_df_sampled, rois, scalers = oversample(files_to_oversample=train_annotations_split,
                                    files_not_to_oversample=valid_annotations_split,
                                    method=experiment_config["data"]["resampling_method"],
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map)
        resampling_ext = "os.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    # Undersampling of the majority class labels (HC and SCHZ)
    elif experiment_config["data"]["resampling"] == "under":
        logging.info(f"Performing undersampling of data with method: {experiment_config["data"]["resampling_method"]}") 
        train_df_sampled, rois, scalers = undersample(files_not_to_undersample=valid_annotations_split,
                                    files_to_undersample=train_annotations_split,
                                    method=experiment_config["data"]["resampling_method"],
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map)
        resampling_ext = "us.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    # Undersampling of the majority class label (HC) and oversampling of the minority class label (MDD) to target label (SCHZ)
    elif experiment_config["data"]["resampling"] == "hybrid":
        logging.info(f"Performing hybrid sampling of data with method: {experiment_config["data"]["resampling_method"]}")
        train_df_sampled, rois, scalers = hybrid_sample(files_not_to_resample=valid_annotations_split,
                                        files_to_resample=train_annotations_split,
                                        class_props= {"majority": "HC", "target": "SCHZ", "minority": "MDD"},
                                        save_dir="/workspace/data/raw_data_train_valid",
                                        atlas_map=atlas_map)
        resampling_ext = "rs.csv"
        train_valid_df_sampled = pd.concat([train_df_sampled, valid_annotations_split])

    elif experiment_config["data"]["resampling"]  == None:
        logging.info(f"Scaling and transforming training data.")
        rois, scalers = no_resample(validation_files=valid_annotations_split,
                                    training_files=train_annotations_split,
                                    save_dir="/workspace/data/raw_data_train_valid",
                                    atlas_map=atlas_map)
        resampling_ext = "nors.csv"
        train_valid_df_sampled = pd.concat([train_annotations_split, valid_annotations_split])

    else: 
        logging.info("Please specify a valid resampling method. This can be either 'over', 'under', 'hybrid', or None")

    train_valid_df_sampled.to_csv(f"{config.OUTPUT_DIR}/train_valid_metadata.csv")

    xcov_groups_train = train_annotations_split["Dataset"].unique().tolist()
    xcov_groups_valid = valid_annotations_split["Dataset"].unique().tolist()

    assert len(xcov_groups_train) == len(xcov_groups_valid)

    num_xcov_groups = len(xcov_groups_train)
    config.NUM_XCOV_GROUPS = num_xcov_groups

    # ------------------------------ STEP 5: Load data and train model ------------------------------

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
                                                xcov=experiment_config["model"]["xcov"])
    else: 
        logging.info(log_title("LOAD MODEL"))
        # If the model has already been trained and we only want to perform testing
        model_architecture = NewVAE(
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
            latent_dim=config.LATENT_DIM,
            atlas_input_dims={'cobra': 104, 'lpba40': 112, 'neuromorphometrics': 408, 'suit': 56},
            schedule_on_validation=config.SCHEDULE_ON_VALIDATION,
            scheduler_patience=config.SCHEDULER_PATIENCE,
            scheduler_factor=config.SCHEDULER_FACTOR,
            num_xcov_groups=config.NUM_XCOV_GROUPS,
            ce_class_weights=config.CE_CLASS_WEIGHTS,
            hidden_dim=config.HIDDEN_DIM,
            dropout_prob=config.DROPOUT_PROB,
        )

        trained_model = load_checkpoint_model(model=model_architecture, model_filename=trained_model_path)


    # ------------------------------ STEP 6: Perform the testing with the trained model ------------------------------

    assert sorted(experiment_config["analysis"].keys()) == sorted(metadata_dict.keys()), "Ensure that the testing runs under 'experiment' correspond with the setup data split"

    for testing_run, testing_data in metadata_dict: 

        testing_run_diagnoses = testing_data["Diagnosis"].unique().to_list()
        testing_run_experimental_details = experiment_config["analysis"][testing_run]

        model_config = config
        if sorted(testing_run_diagnoses) == sorted(experiment_config["data"]["diagnoses"]):
            model_config.ALL_LABELS = True
        else:
            model_config.ALL_LABELS = False
        
        model_config.DIAGNOSES = testing_run_diagnoses
        model_config.PURPOSE = testing_run

        testing_config = TestingConfig(
            UMAP_NEIGHBORS=100, 
            SEED=123,
            TIMESTAMP=None,
            DATA_DIR=model_config.OUTPUT_DIR, # We save the testing data in the same place as training!
            XCov=experiment_config["model"]["xcov"],
        )

        testing_config.update_logging_dir_newvae()

        setup_prediction_logging(config=testing_config)

        logging.info(log_title(f"TESTING RUN {testing_run}"))

        label_mapping = {}
        for idx, diagnosis in sorted(model_config.DIAGNOSES): 
            label_mapping[idx] = diagnosis

        if experiment_config["model"]["xcov"]:
            predictor = SinglePredictor(logging_config=testing_config, 
                                        model_config=model_config,
                                        model=trained_model, 
                                        run_id=testing_run, 
                                        latent_dim=latent_dim,
                                        device=config.DEVICE,
                                        label_mapping=label_mapping)
        else: 
            predictor = SinglePredictor_no_XCov(logging_config=testing_config, 
                                                model_config=model_config,
                                                model=trained_model, 
                                                run_id=testing_run, 
                                                latent_dim=latent_dim,
                                                device=config.DEVICE,
                                                label_mapping=label_mapping)

        all_data_paths_filtered = {}
        for atlas in sorted(atlas_map.keys()):
            all_data_paths_filtered[atlas] = os.path.join(experiment_config["data"]["data_dir"], f"raw_data_{testing_run}", f"{atlas}_data.csv")

        logging.info("Step: Initialize internal testing data")
        predictor.init_test_data(csv_paths=metadata_dict["test"],
                                    data_paths=all_data_paths_filtered,
                                    scalers=scalers)
        logging.info("Step: Initialize model")
        predictor.init_model()
        logging.info("Step: Computing performance")
        model_pred_metrics = predictor.compute_testing_performance()
        logging.info(f"Performance of model on internal testing data: {model_pred_metrics}")

        log_prediction_metrics(metrics=model_pred_metrics, epoch="best")

        # Get plot for internal latent correlations
        get_latent_feature_correlation_plot(testing_config.LOGGING_DIR, atlas_map, atlas_lengths)


        # If both testings have been performed, we can plot the joint results in a summarized umap.
        # if internal_test and external_test: 
        #     logging.info(log_title("PLOTTING JOINT UMAP"))
        #     load_path = config.LOGGING_DIR
        #     total_testing_data = aggregate_testing_data(load_path)
        #     plot_umap(total_testing_data, ["Age", "Sex", "Dataset", "Diagnosis", "Use", "Pred_class"], load_path)


        # ------------------------------ STEP 7: Downstream analysis of model outputs on testing data ------------------------------

        logging.info(log_title("DOWNSTREAM ANALYSIS"))

        total_metadata = pd.read_csv("data/relevant_metadata/full_metadata_filtered_bpfree.csv")
        nss_df = pd.read_csv("data/metadata/meta_data_NSS_all_variables.csv")
        wc_df = pd.read_csv("data/metadata/meta_data_whiteCAT_all_variables_bpfree.csv")

        latent_data = pd.read_csv(f"{testing_config.LOGGING_DIR}/latent_space_data.csv", index_col=0)
        model_predictions = pd.read_csv(f"{testing_config.LOGGING_DIR}/testing_metadata_extended.csv", index_col=0)

        nsswc_df = rename_nss_wc(nss_df, wc_df)
                
                

        logging.info(log_title("CORRELATION ANALYSIS"))

        save_path = os.path.join(testing_config.LOGGING_DIR, "CORR")
        os.makedirs(save_path)
        diagnoses = ["SCHZ", "MDD", "CTT"]

        nsswc_df_dropped = nsswc_df.drop(columns=["Age", "Diagnosis", "Sex", "Dataset", "Codiagnosis"])

        # Compute correlations for each diagnosis separately
        get_all_correlations(clinical_data=nsswc_df_dropped,
                            model_data=model_predictions,
                            metadata=total_metadata,
                            diagnoses=diagnoses,
                            save_path=save_path,
                            ) 
        
        # Compute correlations across all diagnoses
        get_all_correlations(clinical_data=nsswc_df_dropped,
                            model_data=model_predictions,
                            metadata=total_metadata,
                            diagnoses=None,
                            save_path=save_path,
                            )
        
        diagnoses = ["SCHZ", "MDD", "HC"]
        # Determine statsitically significant interaction terms and save the plotted results
        run_interaction_term_analysis(model_data=model_predictions,
                                    clinical_data=nsswc_df_dropped,
                                    metadata=total_metadata,
                                    save_path=save_path,
                                    diagnoses=diagnoses)
        
        logging.info(log_title("LATENT FEATURE ANALYSIS"))

        save_path = os.path.join(testing_config.LOGGING_DIR, "LATENT")
        os.makedirs(save_path)

        # We predict two things from the latent vectors: 1. A "CTT-Risk" and 2. Clinical variables like Age or nss_motor
        # Besides simply knowing which latent features are involved in this prediction we can also determine which ROIs these features correspond with.
        # For this there are again two approaches:
        # 1. Use (Gradient)SHAP to compute which ROIs correspond with which latent feature. This is also just handy as an overview.
        # 2. Use a "chained Saliency" where we use the chain rule to propagate the gradients (=coeff.) of a linear model through the model encoder to get the ROI gradients. 

        ########## CTT-Risk Prediction ##########
        ctt_risk_results, ctt_risk_latent_importance, ctt_risk_coeff = predict_ctt_risk_from_latent(clinical=nsswc_df,
                                                        latent=latent_data,
                                                        save_path=save_path,
                                                        split="random",)
        
        # with open(f"{save_path}/significant_latent_feaures.json", "w") as json_file:
        #     json.dump(ctt_risk_latent_importance, json_file, indent=4)
        
        ctt_risk_results.save(f"{save_path}/ctt_risk_summary.pkl")
        logging.info(f"Results saved to: {save_path}/ctt_risk_summary.pkl")

        ########## Clinical Variable Prediction ##########
        clinical_results, clinical_latent_importance, clinical_coeff = predict_clinical_from_latent(clinical=nsswc_df,
                                                                latent=latent_data,
                                                                save_path=save_path)
        
        with open(f"{save_path}/significant_latent_feaures.json", "w") as json_file:
            json.dump(clinical_latent_importance, json_file, indent=4)

        clinical_results.to_csv(f"{save_path}/clinical_pred_results.csv")
        logging.info(f"Results saved to: {save_path}/clinical_pred_results.csv")
        
        ########## Analysis of ROIs ##########
        logging.info(log_title("MAP ROI - LATENT"))
        # 1. GradientSHAP for all latent features
        logging.info(log_title("Indirect with SHAP"))
        latent_dim = len(latent_data.columns.to_list())

        if experiment_config["model"]["xcov"]:
            wrapped_model_for_latent = NewVAE_Wrapper_latent(trained_model, config.DEVICE)
        else:
            wrapped_model_for_latent = NewVAE_no_XCov_Wrapper_latent(trained_model, config.DEVICE)

        mapping_roi_to_latent = trace_latent_to_roi(config.DEVICE, wrapped_model_for_latent, valid_dataloader, rois, 3, latent_dim)
        mapping_roi_to_latent.to_csv(f"{save_path}/map_latent_to_roi.csv")

        merge_dicts = ctt_risk_latent_importance | clinical_latent_importance

        for pred_var, features in merge_dicts.items():
            print(f"\n")
            logging.info(f"########## Mapping for: {pred_var} ##########")
            for feature in features: 
                feature_rois = mapping_roi_to_latent.loc[f"Feature_{feature}"].to_list()
                logging.info(f"\t{feature_rois}")

        # 2. Chained Saliency
        logging.info(log_title("Direct with Saliency"))

        ctt_risk_filename = saliency_from_latent(ctt_risk_coeff, predictor, config, rois, save_path, config.DEVICE)

        if len(clinical_coeff.keys()) > 0:
            _ = saliency_from_latent(clinical_coeff, predictor, config, rois, save_path, config.DEVICE)

        # ctt_risk_filename = saliency_from_latent_integrated_gradients(ctt_risk_coeff, hold_out_predictor, config, rois, save_path, config.DEVICE)
        # _ = saliency_from_latent_integrated_gradients(clinical_coeff, hold_out_predictor, config, rois, save_path, config.DEVICE)

        # We can visualize these results
        visualize_saliency(ctt_risk_filename, "neuromorphometrics", ["Vgm", "Vwm", "Vcsf"], method="weighted")
            

        logging.info(log_title("FEATURE IMPORTANCE"))

        save_path = os.path.join(testing_config.LOGGING_DIR, "SHAP")
        os.makedirs(save_path)

        if experiment_config["model"]["xcov"]: 
            wrapped_model = NewVAE_Wrapper(trained_model, config.DEVICE)
        else: 
            wrapped_model = NewVAE_no_XCov_Wrapper(trained_model, config.DEVICE)

        idx_mapping = {0: "HC", 1: "MDD", 2: "SCHZ"}

        for idx, label in idx_mapping.items(): # Iterate through HC, MDD and SCHZ

            logging.info(f"Running SHAP for {label}")
            run_shap_analysis(wrapped_model, 
                            dataloader=valid_dataloader,
                            target_class_idx=idx,
                            class_label=label,
                            device=config.DEVICE, 
                            feature_names=rois,
                            dir_name=save_path)

    logging.info("Experiment concluded successfully!")
    return
            
        

        
