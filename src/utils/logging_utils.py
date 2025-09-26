import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict
from utils.config_utils import Config_XCov_CV

"""
Logging functions used to track training progress.
"""
_logging_configured = False

# def log_and_print(message: str):
#     """
#     Log and print a message.
#     """
#     logging.info(message)
#     print(message)


def setup_logging(config):
    """
    Sets up logging to output to both a file and the console.
    This function should be called ONLY ONCE from your main script.
    """
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
    #logging.info("-" * 50)
    logging.info(f"Log file available at: {log_filename}")
    #logging.info("-" * 50)

def log_title(title: str):
    title_len = len(title)
    border = "#"*((100-title_len)//2)
    return f"{border} {title} {border}"


def log_and_print(message: str, level: str = "info"):
    """
    A convenience function to log a message.
    It will appear in both the console and the log file.

    Args:
        message (str): The message to log.
        level (str): The logging level ('info', 'warning', 'error', 'debug').
    """
    # Get a logger instance named after the calling module
    # This helps in tracing where the log message originated from
    logger = logging.getLogger(__name__)
    
    # Get the logging function based on the level string
    log_func = getattr(logger, level.lower(), logger.info)
    
    log_func(message)


# def setup_logging_old(config: Config_XCov_CV):
#     """
#     Set up logging for the training session.

#     """
#     filename = os.path.join(
#         config.LOGGING_DIR, f"{config.TIMESTAMP}_{config.RUN_NAME}_training_log_.txt"
#     )
#     # Set the logging configuration
#     logging.basicConfig(
#         filename=filename,
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     # Log the initial configuration settings
#     logging.info("Starting new training session")
#     logging.info(f"Configuration:")
#     for parameter, value in config.__dict__.items():
#         logging.info(f"{parameter}: {value}")

#     # Print the start time
#     title = f"Starting Catatonia CPA training session {config.RUN_NAME} at {config.TIMESTAMP}"
#     print("")
#     print("#" * 100)
#     print("")
#     print(title)
#     print("-" * len(title))
#     print("")
#     print(f"Using Device:        {config.DEVICE}")
#     print(f"Queued Epochs:       {config.TOTAL_EPOCHS}")
#     print(f"Batch Size:          {config.BATCH_SIZE}")
#     print(f"MRI Data Directory:  {config.MRI_DATA_PATH}")
#     print(f"Loading Model:       {config.LOAD_MODEL}")
#     print(f"Output Directory:    {config.OUTPUT_DIR}")
#     print("")
#     print(f"More details in log file: {filename}")
#     print("")
#     print("#" * 100)
#     print("")


def log_atlas_mode(atlas_name: str, num_rois: int):
    log_and_print(f"Using atlas: {atlas_name} - Total number of ROIs: {num_rois}")

def log_data_loading(datasets: Dict[str, int]):
    colon = ":"
    for dataset, n_subjects in datasets.items():
        log_and_print(
            f"{dataset+colon:<20} {n_subjects:>4} subjects loaded"
        )

def log_model_setup():
    logging.info("Creating Model")


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


def log_training_start():
    message = "Starting Training / Validation Loop"
    log_and_print(message)

def log_model_metrics(
    epoch: int,
    metrics: Dict[str, float],
    type: str,
    prefix: str = "",
    n_digits: int = 3,
):
    """
    Log training or validation metrics.
    """

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


def log_extracting_latent_space(data_type: str):
    log_and_print(
        f"{datetime.now()+timedelta(hours=2):%H:%M} - Extracting Latent Space of {data_type}."
    )


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


def log_early_stopping(
    current_lr: float,
    min_lr: float,
    current_epoch: int,
):
    message = f"Stopping Training at Epoch {current_epoch} because current learning rate {current_lr} is below minimum learning rate {min_lr}."
    log_and_print(message)


def log_early_stopping_th(
    threshold: int, 
    current_epoch: int,
):
    message = f"Stopping Training at Epoch {current_epoch} because accuracy hasn't improved in {threshold} Epochs."
    log_and_print(message)

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


def log_pred_model_ready(model, atlas_name, model_path = None):

    if model_path is not None:
        log_and_print(f"Model saved at: {model_path}")
    #log_and_print(f"Model Archtecture {atlas_name} used for prediction: \n{model}")
    print()

    colon = ":"

    for parameter, value in model.__dict__.items():
        if parameter in [
            "optimizer",
            "scheduler",
            "scaler",
            "kldiv_loss_weight",
            "recon_loss_weight",
            "class_loss_weight",	
            "contrast_temperature",
            "num_epochs",
            "device",
            "learning_rate",
            "weight_decay",
            "image_size",
            "out_channels",
            "encoded_image_dim",
            "encoder_output_dim",
            "latent_dim",
            "schedule_on_validation",
            "scheduler_patience",
            "scheduler_factor",
        ]:
            log_and_print(f"    {parameter+colon:<20} {str(value).splitlines()[0]}")


def log_prediction_metrics(
    epoch: int,
    metrics: Dict[str, float],
    n_digits: int = 3,
):
    """
    Log prediction metrics.
    """

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



def end_prediction_logging(config: Config_XCov_CV):
    end_messsage = f"Training session completed at {datetime.now()+timedelta(hours=2)}."

    logging.info(end_messsage)
    logging.info(f"Configuration:")
    for parameter, value in config.__dict__.items():
        logging.info(f"{parameter}: {value}")
    logging.info("End of training session")
