import sys
sys.path.append("/workspace/src")
import yaml
from RUN_aggr_VAE import run_VAE_experiment

# Specify: 
# - Which groups should be used for testing? 
# - Where is the data and metadata located 
# - Which model should be used? If Xcov, which covariate in the metadata -> check that possible! 
# - What atlases does the input data contain -> Determine abbreviation automatically! 
# - Is the input data either csv or h5? 
# - Should a model be trained or simply a model loaded? Loaded from which other experiment? 

with open("/workspace/experiment_config.yaml", 'r') as file:
    experiment_config = yaml.safe_load(file)

with open("/workspace/model_config.yaml", 'r') as file:
    model_config = yaml.safe_load(file)

run_VAE_experiment(experiment_config, model_config)

