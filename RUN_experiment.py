import yaml
from RUN_aggr_VAE import run_VAE_experiment

# Specify: 
# - Which groups should be used for testing? 
# - Where is the data and metadata located 
# - Which model should be used? If Xcov, which covariate in the metadata -> check that possible! 
# - What atlases does the input data contain -> Determine abbreviation automatically! 
# - Is the input data either csv or h5? 
# - Should a model be trained or simply a model loaded? Loaded from which other experiment? 

experiment_config = yaml.safe_load("experiment_config.yaml")
model_config = yaml.safe_load("model_config.yaml")

run_VAE_experiment(experiment_config, model_config)

