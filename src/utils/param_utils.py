import pandas as pd

run_parameters = {}

run_parameters["aggr_VAE"] = {
    "num_epochs": 250,
    "learning_rate": 0.002,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.0006699932339627386,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

param_df = pd.DataFrame(run_parameters)
param_df.to_csv("/workspace/run_parameters.csv")