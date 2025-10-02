import sys
sys.path.append("/workspace/src")
import yaml
from RUN_aggr_VAE import run_VAE_experiment

# ====================================================================================================
# **RUN Your Experiment Here**
#
# This is the central script for running a new experiment. Adjust the experiment and model parameters
# in the experiment_config.yaml and model_config.yaml files respectively. Ensure that all data is 
# passed in the right format. This might require individual pre-processing of your data.  
# ====================================================================================================

with open("/workspace/experiment_config.yaml", 'r') as file:
    experiment_config = yaml.safe_load(file)

with open("/workspace/model_config.yaml", 'r') as file:
    model_config = yaml.safe_load(file)

run_VAE_experiment(experiment_config, model_config)
