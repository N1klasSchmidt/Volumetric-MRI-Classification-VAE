import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict
from utils.config_utils import Config_XCov_CV


# ====================================================================================================
#
# **LOGGING util functions**
#
# Here all functions used to log training and testing can be found. These are outputted to a central
# logging file, making it easy to follow training. All outputs from the logging functions are also 
# printed to the terminal. The functions are generally split into training and prediction logging. 
#
# ====================================================================================================


_logging_configured = False

# Sets up logging to output to both a file and the console. This function should be called ONLY ONCE from main script.
def setup_logging(config):
    global _logging_configured
    if _logging_configured:
        return

    # Define the log file path
    log_filename = os.path.join(
        config.LOGGING_DIR, f"{config.TIMESTAMP}_{config.RUN_NAME}_log_.txt"
    )
    os.makedirs(config.LOGGING_DIR, exist_ok=True) # Ensure the directory exists

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest level to handle

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Create a file handler to write to the log file
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Create a stream handler to write to the console (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter) # You could use a simpler formatter for the console if you wish
    logger.addHandler(stream_handler)

    _logging_configured = True
    
    # Log the initial configuration settings
    logging.info("=" * 50)
    logging.info("STARTING NEW TRAINING SESSION")
    logging.info("=" * 50)
    logging.info("Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"  - {parameter}: {value}")
    logging.info(f"Log file available at: {log_filename}")

# Used to log a section title
def log_title(title: str):
    title_len = len(title)
    border = "#"*((100-title_len)//2)
    return f"{border} {title} {border}"

# THis function is incorporated in a lot of the other logging functions, simply for convenience.
# Takes a message and ouputs either 'info', 'warning', 'error', 'debug' depending on level. 
def log_and_print(message: str, level: str = "info"):
    # Get a logger instance named after the calling module
    # This helps in tracing where the log message originated from
    logger = logging.getLogger(__name__)
    
    # Get the logging function based on the level string
    log_func = getattr(logger, level.lower(), logger.info)
    
    log_func(message)

# Log which atlases are being used and the number of inputs the model is taking
def log_atlas_mode(atlas_name: str, num_rois: int):
    log_and_print(f"Using atlas: {atlas_name} - Total number of ROIs: {num_rois}")

# Log the datasets
def log_data_loading(datasets: Dict[str, int]):
    colon = ":"
    for dataset, n_subjects in datasets.items():
        log_and_print(
            f"{dataset+colon:<20} {n_subjects:>4} subjects loaded"
        )


def log_model_setup():
    logging.info("Creating Model")

# Log before the start of model training 
def log_model_ready(model):

    log_and_print(f"Model Archtecture: \n{model}")
    print()

    colon = ":"

    for parameter, value in model.__dict__.items():
        if parameter in [
            "optimizer",
            "scheduler",
            "scaler",
            "contr_loss_weight",
            "kldiv_loss_weight",
            "recon_loss_weight",
            "class_loss_weight",	
            "adver_loss_weight",
            "contrast_temperature",
            "num_epochs",
            "device",
            "learning_rate",
            "weight_decay",
            "kernel_size",
            "stride",
            "padding",
            "image_size",
            "out_channels",
            "encoded_image_dim",
            "encoder_output_dim",
            "latent_dim",
            "schedule_on_validation",
            "scheduler_patience",
            "scheduler_factor",
            "dropout_prob",
            "use_ssim",
        ]:
            log_and_print(f"    {parameter+colon:<20} {str(value).splitlines()[0]}")

    # Logging total number of parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_and_print(f"    Total Parameters:    {total_params:,}")
    log_and_print(f"    Trainable Params:    {trainable_params:,}")

# Log the start of the training and validation loop 
def log_training_start():
    message = "Starting Training / Validation Loop"
    log_and_print(message)

# Log training or validation metrics.
def log_model_metrics(
    epoch: int,
    metrics: Dict[str, float],
    type: str,
    n_digits: int = 3,
):

    metrics_log = ""

    for i, (metric, value) in enumerate(metrics.items()):
        metric += ":"
        significance = max(n_digits - len(str(int(value))), 0)

        if value < 10000:
            combo = f"{metric:<13} {value:>4.{significance}f},"
        else:
            combo = f"{metric:<13} {value:>7.1e},"

        metrics_log += f"{combo:<20}"

        # Print and reset every 4 metrics
        if i % 4 == 3:
            metrics_log = metrics_log[: metrics_log.rfind(",")]

            logging.info(f"{type} - Epoch {epoch:>3} - {metrics_log}")
            # print(
            #     f"{prefix}{datetime.now()+timedelta(hours=2):%H:%M} - {type:<21}{metrics_log}"
            # )

            metrics_log = ""

# Log that currently the latent space is being extracted
def log_extracting_latent_space(data_type: str):
    log_and_print(
        f"{datetime.now()+timedelta(hours=2):%H:%M} - Extracting Latent Space of {data_type}."
    )

# Log the metrics of a certain checkpoint. 
def log_checkpoint(
    figure_path: str = None, model_path: str = None, metrics_path: str = None
):
    if figure_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Figures saved to {figure_path}"
        )
    if model_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Model saved to {model_path}"
        )
    if metrics_path is not None:
        log_and_print(
            f"{datetime.now()+timedelta(hours=2):%H:%M} - Checkpoint Save:     Metrics saved to {metrics_path}"
        )

# Log that training was terminated because the learning rate went below a certain threshold
def log_early_stopping(
    current_lr: float,
    min_lr: float,
    current_epoch: int,
):
    message = f"Stopping Training at Epoch {current_epoch} because current learning rate {current_lr} is below minimum learning rate {min_lr}."
    log_and_print(message)

# Log that training was terminated because the goal metric (e.g. valid_loss or accuracy) hasn't improved in a certain number of epochs
def log_early_stopping_th(
    threshold: int, 
    current_epoch: int,
):
    message = f"Stopping Training at Epoch {current_epoch} because goal metric hasn't improved in {threshold} Epochs."
    log_and_print(message)

# Log the ent of logging
def end_logging(config: Config_XCov_CV):
    logging.info(log_title("TRAINING COMPLETED"))
    
    end_messsage = f"Training session completed at {datetime.now()+timedelta(hours=2)}."

    logging.info(end_messsage)
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")
    logging.info(end_messsage)
    logging.info("Thanks for flying catatonia VAE air! ;)")
    
    airplane_art = r"""
            _
          -=\`\
      |\ ____\_\__
    -=\c`````````"`)
       `~~~~~/ /~~`
         -==/ /
           '-'
    """

    logging.info(airplane_art)


# ================================================== Prediction Logging ==================================================

# Log the start of prediction
def setup_prediction_logging(config: Config_XCov_CV):
    """
    Set up logging for the training session.

    """
    filename = os.path.join(
        config.LOGGING_DIR, f"{config.TIMESTAMP}_{config.RUN_NAME}_testing_log_.txt"
    )
    # Set the logging configuration
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Log the initial configuration settings
    logging.info("Starting new training session")
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")

    # Print the start time
    title = f"Starting testing / prediction session {config.RUN_NAME} at {config.TIMESTAMP}"
    print(title)

# Log performance metrics of a testing / prediction run
def log_prediction_metrics(
    epoch: int,
    metrics: Dict[str, float],
):

    metrics_log = ""

    for i, (metric, value) in enumerate(metrics.items()):
        metric += ":"
        
        combo = f"{metric:<13} {str(value)},"

        metrics_log += f"{combo:<20}"

        # Print and reset every 4 metrics
        if i % 4 == 3:
            metrics_log = metrics_log[: metrics_log.rfind(",")]

            logging.info(f"Epoch {epoch:>3} - {metrics_log}")
            metrics_log = ""


# Log the end of a specific testing / prediction run
def end_prediction_logging(config: Config_XCov_CV):
    end_messsage = f"Training session completed at {datetime.now()+timedelta(hours=2)}."

    logging.info(end_messsage)
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")
    logging.info("End of training session")
