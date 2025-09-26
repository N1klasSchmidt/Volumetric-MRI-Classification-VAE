import pandas as pd

run_parameters = {}

run_parameters["neuromorphometrics"] = {
    "num_epochs": 250,
    "learning_rate": 0.003759793310967523,#0.003914688476998507, # 0.00021663101933987368,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 3.385949026421857e-05, #0.0001915955879777888, #0.0001700732909263692,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 2,
}

run_parameters["cobra"] = {
    "num_epochs": 250,
    "learning_rate": 0.0012340002438247426,#0.0022476531990841,#0.00044112275017128957,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.004485669543432305,#7.323674848645112e-05,#0.00020439805747132097,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

run_parameters["lpba40"] = {
    "num_epochs": 250,
    "learning_rate": 0.0015944158495775778,#0.0027740777008349047,#0.00014250375469981316,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 6.504543080149891e-05,#1.1139525464827086e-05,#0.0006603263377571053,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}
run_parameters["suit"] = {
    "num_epochs": 250,
    "learning_rate": 0.003611757440798006,#0.0023538551456407553,#0.00036439861701331657,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.0008449799418737108,#0.00012568947242207095,#0.000186350775046269,
    "class_weight": 7,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

run_parameters["thalamic_nuclei"] = {
    "num_epochs": 250,
    "learning_rate": 0.00042219041294478814,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.0006346915465540319,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

run_parameters["thalamus"] = {
    "num_epochs": 250,
    "learning_rate": 0.00009162570600632493,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.009190689273414694,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

run_parameters["all"] = {
    "num_epochs": 250,
    "learning_rate": 0.004612097503532612,#0.0021985671458473036,#0.00048269258144649155,
    "kl_div_weight": 1,
    "latent_dim": 20,
    "weight_decay": 0.0006699932339627386,#0.0006561720023619554,#0.001789003021472561,
    "class_weight": 20,
    "recon_weight": 10,
    "xcov_weight": 5,
    "xcov_cls_weight": 5,
}

param_df = pd.DataFrame(run_parameters)
param_df.to_csv("/workspace/run_parameters.csv")