import numpy as np 
import random
import pandas as pd
import json
import torch
import shap
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import anndata as ad
import os 
import datetime as dt
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import itertools
import copy

from models.aggr_VAE import NewVAE
from models.aggr_VAE_no_XCov import NewVAE_no_XCov
from utils.logging_utils import log_title
from utils.helper_utils import plot_brain_with_saliency

from sklearn.linear_model import LogisticRegression, ridge_regression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, confusion_matrix

from captum.attr import GradientShap

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap
import matplotlib.ticker as mtick
from statannotations.Annotator import Annotator

from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr, f, linregress, t, ttest_ind, f_oneway
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from itertools import chain
from matplotlib.patches import Rectangle

# ====================================================================================================
# **DOWNSTREAM ANALYSIS**
#
# This script contains all the necessary scripts to perform the entire downstream analysis. It is 
# generally split into _ parts: 
# 
# 1. Determining Feature Importance with SHAP
# 2. Trace to ROI: Matching certain latent features used for prediction to specific ROIs
# 3. Predict with Latent: Use the extracted latent features of the model to predict e.g. CTT-Risk
# 4. Correlation Analysis: Compute correlation between model output and clinical variables
# 5. Saliency Plotting: Project saliency results onto Neuromorph. brain plot for visualization
# 6. Miscellaneous 
# 
# These functions are called during the "RUN" scripts and are the downstream equivalent of the helper
# functions that are used during preprocessing of the data.
# ====================================================================================================




################################################## 1. Feature Importance ##################################################

# For SHAP we disable the global logger, as this clashes with the internal logging of the shap functions
logging.getLogger('shap').setLevel(logging.WARNING) # turns off the "shap INFO" logs
#logging.getLogger('matplotlib').setLevel(logging.WARNING)

# We need to different wrappers for our models, as the XCov and non-XCov models have different outputs during the forawrd pass.
class NewVAE_Wrapper:
    def __init__(self, model: NewVAE, device):
        self.model = model.to(device).eval()

    def forward(self, image):
        _, class_logits, _ = self.model.to_latent(image)
        self.model.zero_grad()
        return class_logits
    
class NewVAE_no_XCov_Wrapper:
    def __init__(self, model: NewVAE_no_XCov, device):
        self.model = model.to(device).eval()

    def forward(self, image):
        _, class_logits = self.model.to_latent(image)
        self.model.zero_grad()
        return class_logits

# This is the central function to run the SHAP analysis for SCHZ, MDD and HC patients within the validation data set. 
# It calls the different functions of this section
def run_shap_analysis(
        # Appropriate model wrapped with one of the above wrappers. Allows direct passing of class logits.
        wrapped_model, 
        # Dataloader with validation data
        dataloader,
        # Classes are sorted in alphabetical order 
        target_class_idx,
        # Class label as a string
        class_label,
        # Same device as which the model has been initiliazed on. 
        device,
        # Names of the regions of interest that should be used in place of e.g. Feature 1, 2, ... 
        feature_names,
        # Name of directory where analysis results should be saved.
        dir_name, 
        # Number of samples used by KernelSHAP to re-evlauate the explanation of a single prediction.
        nsamples=200
        ):
    
    # To make results reprobducible
    random.seed(42)
    np.random.seed(42)

    logging.info(log_title("SHAP explanations"))
    shap_values, explanations, input_data, names = conceptual_shap_dataset(
            model_wrapper=wrapped_model,
            dataloader=dataloader,
            target_class_idx=target_class_idx,
            background_data_np=None,  # Will create random background
            feature_names=feature_names,
            nsamples=nsamples,
            device=device
        )
    
    logging.info(log_title("Analyzing SHAP values"))
    analyze_shap_results(shap_values, explanations, feature_names, top_k=25)

    shap_values_for_plot = np.array([exp.values for exp in explanations])

    fig, ax = plt.subplots(figsize=(18, 20))
    shap_results_plot = shap.summary_plot(shap_values_for_plot, 
                np.vstack(input_data), 
                feature_names=feature_names, 
                plot_size=(18, 20),
                #plot_type="bar",
                max_display=25,
                show=False)

    for label in ax.get_yticklabels():
        label.set_fontsize(20) # Adjust as needed

    for label in ax.get_xticklabels():
        label.set_fontsize(20)

    plt.title("Feature importance prediction by SHAP values", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{dir_name}/SHAP_{class_label}.svg")
    plt.savefig(f"{dir_name}/SHAP_{class_label}.png")
    
    return shap_values, explanations, input_data, names


# Wrapper function to make predictions with ClassVAE_2D_Wrapper using numpy inputs.
# Returns numpy array of per-class probabilities.
def predict_fn(model_wrapper, x_numpy, device):
    x_tensor = torch.from_numpy(x_numpy).float()
    
    # Move to same device as model
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        # Use the wrapper's forward method
        logits = model_wrapper.forward(x_tensor)
        probabilities = F.softmax(logits, dim=1)
        
    return probabilities.cpu().numpy()

# This function is responsible for the main computation of the SHAP data using the KernelExplainer
def conceptual_shap_dataset(model_wrapper, dataloader, target_class_idx, background_data_np=None, 
                           feature_names=None, nsamples=100, device='cpu'):
    
    # Create prediction function with model_wrapper bound
    def model_predict_fn(x_numpy):
        return predict_fn(model_wrapper, x_numpy, device)
    
    # If no background data provided, create a small random background
    if background_data_np is None:

        first_batch = next(iter(dataloader))
        input_shape = first_batch[0].shape[1:]

        background_size = max(50, input_shape[0] * 2) if len(input_shape) > 0 else 50
        background_data_np = torch.randn(background_size, *input_shape).cpu().numpy()

    background_data_np = shap.kmeans(background_data_np, 100)
    
    # Initialize KernelExplainer with L1 regularization for better stability
    explainer = shap.KernelExplainer(model_predict_fn, background_data_np, link="identity") 
    
    all_shap_values = []
    all_explanations = []
    all_input_data = []
    all_names = []
    
    for batch_idx, batch_data in enumerate(dataloader):
        inputs, labels, names = batch_data
        
        # Convert to numpy and CPU
        inputs_np = inputs.detach().cpu().numpy()
        
        logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)} - batch size: {len(inputs_np)}")
        
        shap_values_raw = explainer.shap_values(inputs_np, nsamples=nsamples, progress_message=False)
        expected_value = explainer.expected_value
        
        shap_values_target = shap_values_raw[:, :, target_class_idx]
        expected_value_target = expected_value[target_class_idx]
        
        # Create explanation objects for each instance in the batch
        batch_explanations = []
        for i in range(len(inputs_np)):
            explanation = shap.Explanation(
                values=shap_values_target[i],
                base_values=expected_value_target,
                data=inputs_np[i],
                feature_names=feature_names,
            )
            batch_explanations.append(explanation)
            
        # Store results
        all_shap_values.append(torch.from_numpy(shap_values_target))
        all_explanations.extend(batch_explanations)
        all_input_data.append(inputs_np)
        all_names.extend(names)
    
    logging.info(f"Completed SHAP analysis for {len(all_explanations)} instances")
    
    # Concatenate all SHAP values
    if all_shap_values:
        all_shap_values_tensor = torch.cat(all_shap_values, dim=0)
    else:
        all_shap_values_tensor = torch.empty(0)
    
    return all_shap_values_tensor, all_explanations, all_input_data, all_names


# This function only outputs the results of the previous functions to the terminal.
# The other results are all saved to the spcified directories above.
def analyze_shap_results(shap_values_tensor, explanations, feature_names=None, top_k=25):
    
    if len(shap_values_tensor) == 0:
        print("No SHAP values to analyze")
        return
    
    # Calculate mean absolute SHAP values across all instances
    mean_abs_shap = torch.abs(shap_values_tensor).mean(dim=0)

    # Get top features by importance
    top_indices = torch.argsort(mean_abs_shap, descending=True)[:top_k].tolist()
    logging.info(f"Top {top_k} most important features (by mean |SHAP value|):")
    
    for i, idx in enumerate(top_indices):
        feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
        importance = mean_abs_shap[idx].item()
        logging.info(f"{i+1:2d}. {feature_name:<20} | Importance: {importance:.4f}")
    
    # # Summary statistics
    # summary_message = f"""
    #     SHAP Values Summary:
    #     Shape: {shap_values_tensor.shape}
    #     Mean absolute value: {torch.abs(shap_values_tensor).mean().item():.4f}
    #     Standard deviation: {shap_values_tensor.std().item():.4f}
    #     Min value: {shap_values_tensor.min().item():.4f}
    #     Max value: {shap_values_tensor.max().item():.4f}
    # """
    # logging.info(summary_message)


################################################## 2. Trace to ROI ##################################################

# We define two more wrappers that instead of the class logits, output the concatenated latent vector
class NewVAE_Wrapper_latent:
    def __init__(self, model: NewVAE, device):
        self.model = model.to(device).eval()

    def forward(self, image):
        latent_vector, _, _ = self.model.to_latent(image)
        self.model.zero_grad()
        return latent_vector

class NewVAE_no_XCov_Wrapper_latent:
    def __init__(self, model: NewVAE, device):
        self.model = model.to(device).eval()

    def forward(self, image):
        latent_vector, _ = self.model.to_latent(image)
        self.model.zero_grad()
        return latent_vector


# This function traces or matches ROIs to specific latent features by computing a gradientSHAP score for each model input 
# variable and tracing it to the wrapped (!) model's outputs (which are the latent vectors).
def trace_latent_to_roi(
        # Device to run computation on. This should be the same as the device the data is on
        device, 
        # Model wrapped to output the concatenated latent vector
        wrapped_model, 
        # PyTorch dataloader object with input data
        input_subjects_loader, 
        # List of region of interest names in the same order as they appear in the model input
        roi_list, 
        # Number of ROIs that should be listed per latent feature 
        top_k: int = 3, 
        # Size of the concatenated latent dimension of the model. Varies with the number of input atlases.
        latent_dim: int = 46
        ):

    def norm_attribution(saliency_map):
        mean = torch.mean(saliency_map, dim=0)
        std = torch.std(saliency_map, dim=0) + 1e-8  # Epsilon for stability
        norm_saliency_map = mean / std
        return norm_saliency_map

    def conceptual_gradientshap(model, dataloader, target_idx):

        gs = GradientShap(model.forward)
        all_attributions = []

        for batch_idx, (inputs, labels, names) in enumerate(dataloader):
            inputs = inputs.to(device)
            baseline = torch.zeros_like(inputs)

            attributions_gs = gs.attribute(inputs,
                                            baselines=baseline,
                                            target=target_idx,
                                            n_samples=50) # Number of steps for approximation of the integral
            
            all_attributions.append(attributions_gs.detach().cpu())

        return torch.cat(all_attributions, dim=0)

    def analyze_feature_importance(saliency_maps, feature_names=None, top_k=10, target_idx=0):
        """
        Analyze which features are most important across all samples
        """
        # Average saliency across all samples
        mean_saliency = norm_attribution(saliency_maps)
        atrribution_data = {}


        # Get top-k most important features
        top_values, top_indices = torch.topk(mean_saliency.abs(), top_k)
        top_features = []

        for i, (idx, val) in enumerate(zip(top_indices, top_values)):
            feature_name = feature_names[idx] if feature_names else f"Feature_{idx}"
            top_features.append(feature_name)


        atrribution_data["Total"] = {"Values": top_values, "Indices": top_indices, "Features": top_features}
        
        return top_features
    
    # This is where the actual computation happens
    feature_data = {}

    for i in range(latent_dim):
        captum_gs = conceptual_gradientshap(wrapped_model, input_subjects_loader, target_idx=i)
        top_features = analyze_feature_importance(captum_gs, top_k=top_k, feature_names=roi_list, target_idx=i)

        feature_data[f"Feature_{i}"] = top_features

    columns_renaming = {}
    for i in range(top_k):
        if i == 0:
            columns_renaming[i] = f"1st ROI"
        elif i == 1: 
            columns_renaming[i] = f"2nd ROI"
        elif i == 2: 
            columns_renaming[i] = f"3rd ROI"
        else:
            count = int(i+1)
            columns_renaming[i] = f"{count}th ROI"

    feature_data_df = pd.DataFrame.from_dict(feature_data).T.rename(columns=columns_renaming)

    return feature_data_df


################################################## 3. Predict with Latent ##################################################

# Computes Spearman correlation and Benjamini-Hochberg corrected p-values for a target column against all other columns in a DataFrame.
# Returns a data frame with the columns 'column', 'spearman_rho', 'p_value', and 'bh_corrected_p_value'
def spearman_corr_bh_correction(
        # Input dataframe
        df: pd.DataFrame, 
        # Name of the target column
        target_column: str
        ) -> pd.DataFrame:

    # Exclude non-numeric columns and the target column itself
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    columns_to_correlate = [col for col in numeric_columns if col != target_column]

    if target_column not in numeric_columns:
        print(f"Warning: Target column '{target_column}' is not numeric and will not be included.")
        return pd.DataFrame(columns=['column', 'spearman_rho', 'p_value', 'bh_corrected_p_value'])

    results = []
    p_values = []

    # Calculate Spearman correlation and p-value for each column
    for col in columns_to_correlate:
        spearman_rho, p_value = spearmanr(df[target_column], df[col], nan_policy='omit')
        results.append({
            'column': col,
            'spearman_rho': spearman_rho,
            'p_value': p_value
        })
        p_values.append(p_value)

    results_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg correction
    if not results_df.empty:
        bh_corrected_p_values = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df['bh_corrected_p_value'] = bh_corrected_p_values
        results_df = results_df.sort_values(by='bh_corrected_p_value')

    return results_df

# Fit a simple linear model to predict a target variable from the other variables
def fit_lin_model(
        # Target column
        target: str, 
        # Input data 
        df: pd.DataFrame,
        ) -> Tuple:

    num_df = df.select_dtypes(include=['number'])

    X = num_df.drop(columns=[target])
    y = num_df[target]

    formula = f"{target} ~ 1"

    for term in X.columns.to_list():
        formula += f" + {term}"

    data_all = pd.concat([X, y], axis = 1)

    final_model = smf.ols(formula=formula, data=data_all).fit()
    return final_model, final_model.summary()

# Compute the Variance Inflation Factor. It measures how much of the variance of an estimated regression coefficient
# is increased due to collinearity between the different independent variables. Specifically this is used to filter
# highly correlated variables e.g. NSS motor and NSS total.
def calculate_vif(
        # Input data frame with ideally only numeric columns
        numeric_df: pd.DataFrame, 
        # Target column that is being used for prediction
        target_column: str = None,
        ):
    
    # Select only numeric predictor variables
    if target_column is not None:
        numeric_df = numeric_df.select_dtypes(include=['number']).drop(columns=[target_column])
    
    # Add a constant term to the dataframe
    X = sm.add_constant(numeric_df)
    
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    
    return vif_data.sort_values(by='VIF', ascending=False)

# Use logistic regression on the latent features to predict a "CTT-risk".
# Finally try to link this CTT-risk with clinical variables through fitting a simple linear model
def predict_unseen_risk_from_latent(
        # Target diagnosis
        target: str, 
        # Input metadata with clinical scores of the NSS and whiteCAT patients
        clinical: pd.DataFrame, 
        # Input data with the extracted latent features of the NSS and whiteCAT patients
        latent: pd.DataFrame,
        # Path to save the produced figures and data
        save_path: str,
        # Color mapping for diagnoses
        colors: Dict,
        ): 

    # Sort data frames to ensure that the indices match!
    sorted_clinical = clinical.sort_index()
    sorted_latent = latent.sort_index()

    results_coeff = {}

    # Splitting into training and testing randomly
    X_train, X_test, y_train, y_test = train_test_split(sorted_latent, sorted_clinical[target].to_numpy(), test_size=0.3, random_state=42)

    # Perform training of CTT-risk prediction using latent features
    logging.info(f"Using logisitc regression to predict {target} risk from latent feature vectors")
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    probabilities = log_reg.predict_proba(X_test)  # testing
    score = log_reg.score(X_test, y_test)

    grad_y_wrt_z = log_reg.coef_

    results_coeff[target] = {"grads": grad_y_wrt_z,
                             "train_samples": X_train.index.to_list()}

    # Return the Feature indices of the most relevant features for CTT-risk prediction (determined by absolute coefficient value)
    params = abs(log_reg.coef_[0]).tolist()
    indexed_list = [(value, index) for index, value in enumerate(params)]
    indexed_list.sort(key=lambda x: x[0], reverse=True)
    sorted_indices = [index for value, index in indexed_list]
    logging.info(f"Most relevant latent features: {sorted_indices[:10]}")

    unseen_risk_important_features = {target: sorted_indices[:10]}

    # Evaluate the performance of the logistic regression model using a confusion matrix and comparing CTT-risk prediction between CTT and non-CTT SCHZ patients. 
    logging.info("Evaluating the performance of logistic regression model")
    compute_logreg_performance(probabilities, X_test, y_test, sorted_clinical, save_path, target=target)

    logging.info("Evaluating the performance of Random Forest model")
    predict_unseen_risk_from_latent_rf(clinical_data=sorted_clinical, latent_data=sorted_latent, save_path=save_path, target=target, color_map=colors)

    return unseen_risk_important_features, results_coeff


# Evaluate the performance of the logistic regression model used to predict the CTT label from latent features.
# Generates a confusion matrix and an array of boxplots showing the per class predictions. 
# A "subset_index" can be specified to filter the results of only a spcific diagnosis subset e.g. SCHZ
def compute_logreg_performance(probabilities, X_test, y_test, sorted_clinical, save_path, target):
    
    # Make a confusion matrix
    ctt_predictions = np.argmax(probabilities, axis=1)
    cf_matrix = confusion_matrix(y_test, ctt_predictions)

    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    plt.figure()
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
    plt.title(f"Confusion Matrix of {target}-risk prediction")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{save_path}/{target}_risk_confusion_matrix.png")
    plt.savefig(f"{save_path}/{target}_risk_confusion_matrix.svg")
    plt.close()

    probabilities_df = pd.DataFrame(probabilities, columns=['probability_class_0', 'probability_class_1'])

    # Assign the index of the X_test DataFrame to the new probabilities DataFrame
    probabilities_df.index = X_test.index

    # Join the original sorted_nss_wc_df with the probabilities DataFrame on the index
    # We use .loc[X_test.index] to ensure we only join for the patients in the test set and maintain the correct order
    patients_with_probabilities = sorted_clinical.loc[X_test.index].join(probabilities_df)
    patients_with_probabilities["Prob_percent"] = patients_with_probabilities["probability_class_1"]*100

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.boxplot(patients_with_probabilities, x="Diagnosis", y="Prob_percent", hue="Diagnosis", width=.5)

    diag_combinations = list(itertools.combinations(patients_with_probabilities["Diagnosis"].unique().tolist(), 2))

    annotator = Annotator(ax, diag_combinations, data=patients_with_probabilities, x="Diagnosis", y="Prob_percent")

    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
    annotator.apply_and_annotate()

    plt.xlabel("Diagnostic Label")
    plt.ylabel(f"{target}-risk prediction")

    plt.ylim(0.0, 100.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.title(f"{target}-risk prediction stratified by diagnosis (Mann-Whitney U)", pad=30)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{target}_risk_comparison.png")
    plt.savefig(f"{save_path}/{target}_risk_comparison.svg")
    return

# Use a random forest approach to classify MDD, SCHZ and CTT from the extracted latent space of the NSS and whiteCAT patients. 
# Results are then directly plotted as boxplots, showing the per class distributions.
def predict_unseen_risk_from_latent_rf(
    # Input metadata
    clinical_data: pd.DataFrame, 
    # Input latent data
    latent_data: pd.DataFrame, 
    # Path where the results should be saved
    save_path: str,
    # What is the target diagnosis
    target: str,
    # What is the color mapping? 
    color_map: Dict, 
    ):
    
    current_diagnoses = clinical_data["Diagnosis"].unique().tolist()

    total_data = pd.concat([latent_data, clinical_data["Diagnosis"]], axis=1)

    X = total_data.drop('Diagnosis', axis=1)
    y = total_data['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    probabilities_df = pd.DataFrame(probabilities, columns=current_diagnoses)
    probabilities_df.index = X_test.index
    patients_with_probabilities = pd.concat([total_data.loc[X_test.index], probabilities_df], axis=1)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    diag_combinations = list(itertools.combinations(patients_with_probabilities["Diagnosis"].unique().tolist(), 2))

    for i, prob_col in enumerate(current_diagnoses):
        # Create boxplot showing this probability across all true diagnostic labels
        sns.boxplot(data=patients_with_probabilities, 
                    x="Diagnosis", 
                    y=prob_col, 
                    hue="Diagnosis",
                    palette=color_map,
                    width=0.6,
                    ax=axes[i])
        
        annotator = Annotator(axes[i], diag_combinations, data=patients_with_probabilities, x="Diagnosis", y=prob_col)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
        annotator.apply_and_annotate()
        
        # Customize subplot
        axes[i].set_title(f'{prob_col} Probability', pad=20)
        axes[i].set_xlabel('True Diagnostic Label')
        axes[i].set_ylabel('Probability' if i == 0 else '')  # Only show ylabel on first subplot
        axes[i].set_ylim(0.0, 1.15)
        axes[i].legend().remove()  # Remove legend since colors match the x-axis labels

    # Add overall title
    fig.suptitle('Predicted Class Probabilities by True Diagnostic Label', fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{target}_risk_comparison_RF.png")
    plt.savefig(f"{save_path}/{target}_risk_comparison_RF.svg")


# Predict clinical variables using the latent space features. This is the approach used by Yamaguchi et al.
# In a five fold CV process a ridge regression / LASSO model is fit to the within bag data and prediction on the OOB. 
# An f statistic of the predictions is determined to evaluate if this metric can be significantly predicted 
# using the latent features. Significant predictions are plotted and saved. 
def predict_clinical_from_latent(
    # Input clinical metadata with patients along the index and clinical variables as columns
    clinical: pd.DataFrame, 
    # Input data frame with latent data for each of the samples in clinical
    latent: pd.DataFrame,
    # Path to which the data should be saved
    save_path: str, 
    # k-Fold validation 
    k: int = 4,
    # Type of regression that should be used to determine relevant clinical features ("lasso" or "ridge")
    method: str = "lasso",
    ) -> pd.DataFrame:

    logging.info("Predicting clinical variables from latent feature vectors (CV Ridge Regression)")
    
    results = {}
    resuls_selected_features = {}
    results_coeff = {}

    sorted_clinical = clinical.sort_index().drop(columns=['Dataset', 'Diagnosis', 'Sex',])
    sorted_clinical_num = sorted_clinical.select_dtypes(include=np.number)
    sorted_latent = latent.sort_index()

    for idx, target in enumerate(sorted_clinical_num.columns.to_list()):
        logging.info(f"Step: [{idx+1}/{len(sorted_clinical_num.columns.to_list())}] Ridge Regression on {target} clinical variable")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        y_preds = []
        y_tests = []
        selected_features = []

        for i, (train_indices, test_indices) in enumerate(kf.split(sorted_latent)):

            X_train = sorted_latent.iloc[train_indices, :]
            y_train = sorted_clinical_num.iloc[train_indices, :][target]

            X_test = sorted_latent.iloc[test_indices, :]
            y_test = sorted_clinical_num.iloc[test_indices, :][target]

            # Drop rows with NaN values from both training and testing sets
            train_df = pd.concat([X_train, y_train], axis=1).dropna()
            X_train = train_df.iloc[:, :-1]
            y_train = train_df.iloc[:, -1]

            test_df = pd.concat([X_test, y_test], axis=1).dropna()
            X_test = test_df.iloc[:, :-1]
            y_test = test_df.iloc[:, -1]

            y_tests.extend(y_test)

            if method == "ridge":
                coef, intercept = ridge_regression(X_train, y_train, alpha=1, return_intercept=True, random_state=42)

                y_pred = X_test @ coef + intercept

            if method == "lasso":
                lasso = Lasso(alpha=0.05, random_state=42, max_iter=1000)
                lasso.fit(X_train, y_train)
                y_pred = lasso.predict(X_test)

                # Get features with non-zero coefficients
                non_zero_mask = lasso.coef_ != 0
                selected_features.append(lasso.coef_)

            y_preds.extend(y_pred)

        y_preds_np = np.array(y_preds)
        y_tests_np = np.array(y_tests)

        rmse = np.sqrt(np.mean((y_preds_np - y_tests_np)**2))

        y_preds_null = np.full_like(y_tests, sorted_clinical_num[target].mean())
        rmse_null = np.sqrt(np.mean((y_preds_null - y_tests_np)**2))

        r2 = r2_score(y_tests_np, y_preds_np)

        n = len(y_tests_np)
        k = sorted_latent.shape[1]
        f_stat = (r2/(1-r2))*((n-k-1)/k)

        dfN = k  # degrees of freedom in the numerator
        dfD = n - k - 1 # degrees of freedom in the denominator

        pVal = f.sf(f_stat, dfN, dfD)

        results[target] = {"f_stat": f_stat, "p_val": pVal}

        if pVal <= 0.05:
            logging.info(f"P-value for {target}: {pVal}")

            selection_df = pd.DataFrame(selected_features).T.abs()  # We make a df of the coefficients and transpose to have (n_coeff, n_folds)
            rows_with_a_zero = (selection_df ==0).any(axis=1)
            df_intersect = selection_df.loc[~rows_with_a_zero]  # Remove any rows that contain a zero = not all folds share this feature! 
            df_intersect["avg"] = df_intersect.mean(axis=1)
            df_intersect.sort_values(by="avg", ascending=False, inplace=True)
            intersecting_features = df_intersect.index.to_list()  # Intersecting features sorted by average value of coefficients

            resuls_selected_features[target] = intersecting_features
            logging.info(f"Intersectiong latent features across all folds: {intersecting_features}")

            grad_y_wrt_z = lasso.coef_
            train_samples = sorted_clinical_num.iloc[train_indices].index.to_list()
            
            results_coeff[target] = {"grads": grad_y_wrt_z,
                                     "train_samples": train_samples}
            
            plt.figure(figsize=(8, 8))
            plt.grid(True, alpha=0.3)
            plt.title(f"Predicted vs. True Values for {target}")
            plt.xlabel(f"True {target} Values")
            plt.ylabel(f"Predicted  {target} Values")
            plt.xlim(y_tests_np.min() - 5, y_tests_np.max() + 5)
            plt.ylim(y_preds_np.min() - 5, y_preds_np.max() + 5)
            plt.axline((0, 0), (1, 1), color='red', linestyle='--', label='y=x')
            plt.scatter(y_tests_np, y_preds_np, alpha=0.5, c="green", s=100)
            plt.legend()
            plt.savefig(f"{save_path}/predict_{target}_from_latent.png")
            plt.savefig(f"{save_path}/predict_{target}_from_latent.svg")


    results_df = pd.DataFrame.from_dict(results)
    return results_df, resuls_selected_features, results_coeff


# Use with a trained hold_out_model predictor to determine via Saliency which ROIs are linked to which latent features
# This is different from Section 2 "Trace to Latent", which simply does an indirect assignment. This function propagates
# the gradients of a regression model for specific latent features through the model to determine saliency.
def saliency_from_latent(
        # Coefficients = Gradients that should be propagated through model
        coeff: Dict,
        # Model through which the gradients should be propagated
        predictor,
        # Config file
        config: Dict, 
        # List of ROIs that are sorted in the same order as the data that is passed to the model during training 
        rois: List, 
        # Path to which the data should be saved
        save_path: str,
        # Device to be used for propagation
        device,
        ):
    
    for target in coeff.keys():

        grad_y_wrt_z = coeff[target]["grads"]
        subset_names = coeff[target]["train_samples"]

        hold_out_data = predictor.dataloader
        hold_out_data_subset = hold_out_data #torch.utils.data.Subset(hold_out_data, subset_indices)

        model = copy.deepcopy(predictor.model)

        first_batch_measurements, _, _ = next(iter(hold_out_data_subset))
        total_abs_saliency = torch.zeros_like(first_batch_measurements[0], device=device)

        for batch_idx, (measurements, labels, names) in enumerate(hold_out_data_subset):
            
            keep_indices = []
            for idx, name in enumerate(names):
                if name in subset_names:
                    keep_indices.append(idx)

            filtered_measurements = measurements[keep_indices]

            batch_measurements = torch.stack([measurement for measurement in filtered_measurements]).to(device)
            batch_measurements.requires_grad_(True)

            if total_abs_saliency is None:
                total_abs_saliency = torch.zeros_like(batch_measurements, requires_grad=False)

            model.eval()
            if predictor.xcov:
                with torch.enable_grad():
                # Forward pass to get latent space representation z
                    _, _, _, _, _, z = model.forward(batch_measurements)
                    batch_grad_y_wrt_z = torch.from_numpy(grad_y_wrt_z).to(device, dtype=z.dtype).repeat(z.shape[0], 1)
                    z.backward(gradient=batch_grad_y_wrt_z, retain_graph=True)
            else:
                with torch.enable_grad():
                # Forward pass to get latent space representation z
                    _, _, _, _, z = model.forward(batch_measurements)
                    batch_grad_y_wrt_z = torch.from_numpy(grad_y_wrt_z).to(device, dtype=z.dtype).repeat(z.shape[0], 1)
                    z.backward(gradient=batch_grad_y_wrt_z, retain_graph=True)

            # Accumulate the absolute saliency map
            total_abs_saliency += batch_measurements.grad.abs().sum(dim=0)

        mean_abs_saliency = total_abs_saliency / len(hold_out_data_subset.dataset)

        assert len(rois) == mean_abs_saliency.shape[0], "Mismatch between number of ROIs and saliency features."

        saliency_results = {roi: score.item() for roi, score in zip(rois, mean_abs_saliency)}

        # Save the dictionary to a 
        output_filename = f"{save_path}/saliency_results_{target}_from_latent.json"
        with open(output_filename, "w") as f:
            json.dump(saliency_results, f, indent=4)

        top_n_rois = dict(sorted(saliency_results.items(), key = lambda x: x[1], reverse=True)[:10])
        logging.info(f"\nSaliency results for {target} saved to {output_filename}")
        logging.info(f"Top 10 regions for prediction of {target} from latent features:")
        for k, v in top_n_rois.items():
            logging.info(f"\t{k}: {v}")
    
    return output_filename


# Use with a trained hold_out_model predictor to determine via Integrated Gradients which ROIs are linked to which latent features.
# This is in principle the same as the function saliency_from_latent(), but uses integrated gradients. From experience this yields 
# very similar results.
def saliency_from_latent_integrated_gradients(
        # Coefficients = Gradients of regression model used
        coeff: Dict,
        # Model through which the gradients should be propagated
        predictor,
        # Config data
        config: Dict, 
        # List of ROIs that are sorted in the same order as the data that is passed to the model during training  
        rois: List, 
        # Path to which the data should be saved
        save_path: str, 
        # Device on which the computation should be run
        device,
        # Parameters for integrated gradients computation
        n_steps: int = 50,
        baseline: str = 'zeros'  # 'zeros', 'mean', or 'random'
        ):
    
    def get_baseline(measurements, baseline_type):
        """Generate baseline for integrated gradients"""
        if baseline_type == 'zeros':
            return torch.zeros_like(measurements)
        elif baseline_type == 'mean':
            return torch.mean(measurements, dim=0, keepdim=True).expand_as(measurements)
        elif baseline_type == 'random':
            return torch.randn_like(measurements) * 0.1  # Small random noise
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def compute_integrated_gradients(model, measurements, grad_y_wrt_z, baseline, n_steps, xcov):
        """Compute integrated gradients for a batch"""
        integrated_grads = torch.zeros_like(measurements)
        
        # Create interpolated inputs
        for step in range(n_steps):
            alpha = step / (n_steps - 1)  # From 0 to 1
            interpolated_input = baseline + alpha * (measurements - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            if xcov:
                _, _, _, _, _, z = model.forward(interpolated_input)
            else:
                _, _, _, _, z = model.forward(interpolated_input)
            
            # Backward pass
            batch_grad_y_wrt_z = torch.from_numpy(grad_y_wrt_z).to(device, dtype=z.dtype).repeat(z.shape[0], 1)
            z.backward(gradient=batch_grad_y_wrt_z, retain_graph=True)
            
            # Accumulate gradients
            integrated_grads += interpolated_input.grad
            
            # Clear gradients for next iteration
            if interpolated_input.grad is not None:
                interpolated_input.grad.zero_()
        
        # Average over steps and multiply by (input - baseline)
        integrated_grads = integrated_grads / n_steps
        integrated_grads = integrated_grads * (measurements - baseline)
        
        return integrated_grads
    
    for target in coeff.keys():
        grad_y_wrt_z = coeff[target]["grads"]
        subset_names = coeff[target]["train_samples"]

        hold_out_data = predictor.dataloader
        hold_out_data_subset = hold_out_data

        model = copy.deepcopy(predictor.model)

        first_batch_measurements, _, _ = next(iter(hold_out_data_subset))
        total_abs_saliency = torch.zeros_like(first_batch_measurements[0], device=device)
        n_samples = 0

        for batch_idx, (measurements, labels, names) in enumerate(hold_out_data_subset):
            
            keep_indices = []
            for idx, name in enumerate(names):
                if name in subset_names:
                    keep_indices.append(idx)

            if not keep_indices:  # Skip if no samples to keep
                continue
                
            filtered_measurements = measurements[keep_indices]
            batch_measurements = torch.stack([measurement for measurement in filtered_measurements]).to(device)
            
            # Generate baseline
            batch_baseline = get_baseline(batch_measurements, baseline)
            
            model.eval()
            with torch.enable_grad():
                # Compute integrated gradients
                integrated_grads = compute_integrated_gradients(
                    model, 
                    batch_measurements, 
                    grad_y_wrt_z, 
                    batch_baseline, 
                    n_steps,
                    xcov=predictor.xcov,
                )

            # Accumulate the absolute saliency map
            total_abs_saliency += integrated_grads.abs().sum(dim=0)
            n_samples += len(keep_indices)

        mean_abs_saliency = total_abs_saliency / n_samples

        assert len(rois) == mean_abs_saliency.shape[0], "Mismatch between number of ROIs and saliency features."

        saliency_results = {roi: score.item() for roi, score in zip(rois, mean_abs_saliency)}

        # Save the dictionary
        output_filename = f"{save_path}/integrated_gradients_saliency_results_{target}_from_latent.json"
        with open(output_filename, "w") as f:
            json.dump(saliency_results, f, indent=4)

        top_n_rois = dict(sorted(saliency_results.items(), key=lambda x: x[1], reverse=True)[:10])
        logging.info(f"\nIntegrated Gradients saliency results for {target} saved to {output_filename}")
        logging.info(f"Top 10 regions for prediction of {target} from latent features:")
        for k, v in top_n_rois.items():
            logging.info(f"\t{k}: {v}")
    
    return output_filename



################################################## 4. Correlation Analysis ##################################################

# Computes the correlations between the model's testing outputs and the clinical variables of NSS and whiteCAT. Significance 
# is evaluated and only clinical variables across one of the five model outputs (SCHZ prob, MDD prob, HC prob, Misdiagnosis, 
# Uncertainty) are plotted in a heatmap. This function isn't called directly in the RUN scripts, instead over the 
# get_all_correlations() function.
def compute_correlations(
        # Data frame with extended prediction data from the model. Columns as features, rows as samples. 
        model_pred_df: pd.DataFrame, 
        # Data frame with metadata for all patients in model_pred_df. Columns as features, rows as samples.
        metadata_df: pd.DataFrame, 
        # Path to save the outputted plots
        save_path: str, 
        # Which diagnosis subset is being used? Ususally either "all" for no subset, or "CTT" for catatonia patients
        diagnosis: str, 
        # Which dataset subset is being used? Usually either NSS_WC for all data, or either NSS or whiteCAT
        datasets: str, 
        ):
    
    # Create matrices for correlations and p-values
    correlation_results = pd.DataFrame(
        index=model_pred_df.columns,
        columns=metadata_df.columns,
        dtype=float
    )

    pvalue_results = pd.DataFrame(
        index=model_pred_df.columns,
        columns=metadata_df.columns,
        dtype=float
    )

    # Compute correlations and p-values
    logging.info("Computing Spearman Correlation between model predictions and metadata features")
    for model_col in model_pred_df.columns: # Changed outer loop for consistency with inner
        for meta_col in metadata_df.columns:
            meta_data = metadata_df[meta_col].dropna()
            model_data = model_pred_df[model_col].dropna()
            
            common_idx = meta_data.index.intersection(model_data.index)
            
            if len(common_idx) > 2:  # Need at least 3 points for meaningful p-value
                meta_common = meta_data.loc[common_idx]
                model_common = model_data.loc[common_idx]
                
                # Compute Spearman correlation and p-value
                corr_val, p_val = spearmanr(meta_common, model_common)
                
                correlation_results.loc[model_col, meta_col] = corr_val
                pvalue_results.loc[model_col, meta_col] = p_val

    # --- Apply Multiple Testing Correction (Benjamini-Hochberg FDR) ---
    # Flatten the p-values, remove NaNs for correction, and get original shape
    original_shape = pvalue_results.shape
    p_values_flat = pvalue_results.values.flatten()
    
    # Store indices of NaN values to put them back later
    nan_mask = pd.isna(p_values_flat)
    p_values_for_correction = p_values_flat[~nan_mask]

    logging.info("Correction of p-values using Benjamini-Hochberg FDR")
    if len(p_values_for_correction) > 0:
        # Apply Benjamini-Hochberg FDR correction
        reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
            p_values_for_correction, 
            alpha=0.05, 
            method='fdr_bh' # Benjamini-Hochberg False Discovery Rate
        )
        
        # Create a new flattened array for corrected p-values, filling NaNs back
        corrected_p_values_flat = np.full(p_values_flat.shape, np.nan)
        corrected_p_values_flat[~nan_mask] = pvals_corrected
        
        # Reshape corrected p-values back to the original DataFrame structure
        pvalue_results_corrected = pd.DataFrame(
            corrected_p_values_flat.reshape(original_shape),
            index=pvalue_results.index,
            columns=pvalue_results.columns
        )
        pvalue_results = pvalue_results_corrected # Use corrected p-values from now on
    else:
        print("No valid p-values to correct for multiple testing.")

    # Create significance mask (True for significant correlations after correction)
    alpha = 0.05  # significance level for corrected p-values
    significance_mask = pvalue_results < alpha

    # Create annotation matrix combining correlation values and significance into a single string!
    annotations = correlation_results.copy().astype(str)
    for i in range(len(correlation_results.index)):
        for j in range(len(correlation_results.columns)):
            corr_val = correlation_results.iloc[i, j]
            p_val = pvalue_results.iloc[i, j] # Now using corrected p-values
            
            if pd.notna(corr_val) and pd.notna(p_val):
                if p_val < 0.001:
                    annotations.iloc[i, j] = f"{corr_val:.2f}***"
                elif p_val < 0.01:
                    annotations.iloc[i, j] = f"{corr_val:.2f}**"
                elif p_val < 0.05:
                    annotations.iloc[i, j] = f"{corr_val:.2f}*"
                else:
                    annotations.iloc[i, j] = f"{corr_val:.2f}"
            else:
                annotations.iloc[i, j] = ""

    # Filter to only show columns/rows with at least one significant correlation
    # This filter also uses the corrected p-values
    # significant_rows = (pvalue_results < alpha).any(axis=1)
    significant_cols = (pvalue_results < alpha).any(axis=0)

    # Filter all matrices to only include significant columns
    correlation_filtered = correlation_results.loc[:, significant_cols]
    pvalue_filtered = pvalue_results.loc[:, significant_cols] # Corrected p-values
    annotations_filtered = annotations.loc[:, significant_cols]

    # Collect summary information for CSV
    summary_info = []
    summary_info.append({
        'Metric': 'Original matrix shape',
        'Value': f"{correlation_results.shape[0]} x {correlation_results.shape[1]}"
    })
    summary_info.append({
        'Metric': 'Filtered matrix shape',
        'Value': f"{correlation_filtered.shape[0]} x {correlation_filtered.shape[1]}"
    })
    # summary_info.append({
    #     'Metric': 'Model features with significant correlations',
    #     'Value': significant_rows.sum()
    # })
    summary_info.append({
        'Metric': 'Metadata features with significant correlations',
        'Value': significant_cols.sum()
    })

    # Plot filtered heatmap with significance annotations
    colors1 = ['navy', 'white', 'firebrick']
    cust_cmap = LinearSegmentedColormap.from_list("custom_palette", colors1)

    plt.figure(figsize=(max(10, len(correlation_filtered.columns) * 0.8), 
                        max(6, len(correlation_filtered.index) * 0.5)))

    ax = sns.heatmap(correlation_filtered,
                    cmap=cust_cmap,
                    vmin=-1,
                    vmax=1,
                    annot=annotations_filtered,
                    fmt='',  # Use empty format since we're providing strings
                    linewidths=.0,
                    linecolor='white',
                    cbar_kws={"shrink": .7, "label": "Spearman Correlation Coefficient"})

    plt.title("Significant Correlations: Metadata Features vs Model Features\n(*p<0.05, **p<0.01, ***p<0.001 - BH-FDR Corrected)", fontsize=14)
    plt.xlabel("Metadata Features", fontsize=14)
    plt.ylabel("Model Features", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'correlation_heatmap_{diagnosis}_{datasets}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'correlation_heatmap_{diagnosis}_{datasets}.svg'), dpi=300, bbox_inches='tight')

    plt.close()

    # Create a separate filtered heatmap showing only p-values (corrected)
    plt.figure(figsize=(max(10, len(pvalue_filtered.columns) * 0.8), 
                        max(6, len(pvalue_filtered.index) * 0.5)))

    sns.heatmap(pvalue_filtered, # This now contains corrected p-values
                cmap='Reds_r',  # Reverse red colormap (darker = more significant)
                vmin=0,
                vmax=0.05,  # Focus on p-values up to 0.05 (corrected alpha)
                annot=True,
                fmt=".3f",
                linewidths=.0,
                linecolor='white',
                cbar_kws={"shrink": .7, "label": "Corrected P-value (BH-FDR)"})

    plt.title("P-values of Significant Correlations (BH-FDR Corrected)", fontsize=14)
    plt.xlabel("Metadata Features", fontsize=14)
    plt.ylabel("Model Features", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'pvalue_heatmap_{diagnosis}_{datasets}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'pvalue_heatmap_{diagnosis}_{datasets}.svg'), dpi=300, bbox_inches='tight')

    plt.close()

    # Add summary statistics to summary_info (now based on corrected p-values)
    summary_info.append({
        'Metric': 'Total correlations computed',
        'Value': correlation_results.notna().sum().sum()
    })
    summary_info.append({
        'Metric': 'Significant at p<0.05 (BH-FDR corrected)',
        'Value': (pvalue_results < 0.05).sum().sum()
    })
    summary_info.append({
        'Metric': 'Significant at p<0.01 (BH-FDR corrected)',
        'Value': (pvalue_results < 0.01).sum().sum()
    })
    summary_info.append({
        'Metric': 'Significant at p<0.001 (BH-FDR corrected)',
        'Value': (pvalue_results < 0.001).sum().sum()
    })

    # Show the most significant correlations (now based on corrected p-values)
    significant_correlations = []
    for i in range(len(correlation_results.index)):
        for j in range(len(correlation_results.columns)):
            corr_val = correlation_results.iloc[i, j]
            p_val = pvalue_results.iloc[i, j] # Now using corrected p-values
            
            if pd.notna(corr_val) and pd.notna(p_val) and p_val < 0.05:
                significant_correlations.append({
                    'Model_Feature': correlation_results.index[i],
                    'Metdata_Feature': correlation_results.columns[j],
                    'Correlation': corr_val,
                    'P_value': p_val
                })

    # Create summary DataFrame and save
    summary_df = pd.DataFrame(summary_info)
    summary_df.to_csv(os.path.join(save_path, f'correlation_summary_{diagnosis}_{datasets}.csv'), index=False)

    # Save significant correlations if any exist
    if significant_correlations:
        sig_df = pd.DataFrame(significant_correlations)
        sig_df = sig_df.sort_values('P_value')
        sig_df.to_csv(os.path.join(save_path, f'significant_correlations_{diagnosis}_{datasets}.csv'), index=False)
    else:
        # Save empty DataFrame with message
        no_sig_df = pd.DataFrame({'Message': ['No significant correlations found at p<0.05 (BH-FDR corrected)']})
        no_sig_df.to_csv(os.path.join(save_path, f'significant_correlations_{diagnosis}_{datasets}.csv'), index=False)


# Main function to compute the correlations of the desired data subsets. Currently only set up to do a CTT specific run and an all-diagnosis run. 
def get_all_correlations(
        # Clinical metadata e.g. from NSS and whiteCAT
        clinical_data: pd.DataFrame,
        # Data frame with model's prediction data e.g. class probabilities
        model_data: pd.DataFrame, 
        # General file with all the basic metadata of ALL files. This is needed to correct the NSS and whiteCAT files.
        metadata: pd.DataFrame,
        # Save path for outputted correlation data
        save_path: str,
        diagnoses: List,
        original_model_diagnoses: List, 
        datasets="no_dataset_subset"
        ):

    raw_model_pred_df = model_data.copy().set_index("Filename", drop=True)
    joined_df = raw_model_pred_df.join(clinical_data).drop(columns=["Confidence", "Pred_labels"])

    if diagnoses is not None:
        for unseen_diagnosis in diagnoses:
            logging.info(f"Computing correlations between model outputs and clinical variables for {unseen_diagnosis} patients")
            joined_df_diagnosis_subset = joined_df[joined_df["Diagnosis"] == unseen_diagnosis]
            num_joined_df_diagnosis_subset = joined_df_diagnosis_subset.select_dtypes(include=np.number)

            numeric_model = num_joined_df_diagnosis_subset[original_model_diagnoses+["Misdiagnosis", "Uncertainty"]]
            metadata_cols = ~num_joined_df_diagnosis_subset.columns.isin(original_model_diagnoses+["Misdiagnosis", "Uncertainty"])
            numeric_meta = num_joined_df_diagnosis_subset.iloc[:, metadata_cols]

            compute_correlations(model_pred_df=numeric_model, metadata_df=numeric_meta, save_path=save_path, diagnosis=unseen_diagnosis, datasets=datasets)
    else: 
        logging.info("Computing correlations between model outputs and clinical variables across ALL diagnoses")
        num_joined_df_diagnosis_subset = joined_df.select_dtypes(include=np.number)

        numeric_model = num_joined_df_diagnosis_subset[original_model_diagnoses+["Misdiagnosis", "Uncertainty"]]
        metadata_cols = ~num_joined_df_diagnosis_subset.columns.isin(original_model_diagnoses+["Misdiagnosis", "Uncertainty"])
        numeric_meta = num_joined_df_diagnosis_subset.iloc[:, metadata_cols]

        compute_correlations(model_pred_df=numeric_model, metadata_df=numeric_meta, save_path=save_path, diagnosis="all", datasets=datasets)
    
    return


################################################## 5. Saliency Plotting ##################################################


# Extract the saliency information from the saved json file and format into the correct shape. To aggregate the saliency
# across different measurement types, the mean is taken for each ROI (if volumes list more than one entry!).
def get_saliency_df(
        # Path pointing to .json file with the saliency data we want to plot.
        saliency_path: str,
        # Mapping of dictionaries with their abbreviations
        atlas_mapping: Dict,
        # Which atlas are we plotting? Currently only Neuromorphometrics is implemented
        atlas: str = "neuromorphometrics",
        # Which measurement types are we analyzing
        volumes: list = ["Vgm"],
        ) -> pd.DataFrame:

    with open(saliency_path) as f:
        saliency_data = json.load(f)

    saliency_df = pd.DataFrame.from_dict(saliency_data, orient="index")
    saliency_index = saliency_df.index

    subset_saliency_dfs = []

    for volume in volumes: 
        keep_sal_index_vol = [i for i in saliency_index if i.startswith(atlas_mapping[atlas]["abbrev"]) and i.endswith(volume)]
        saliency_n_df_vol= saliency_df.loc[keep_sal_index_vol,:]
        subset_saliency_dfs.append(saliency_n_df_vol)

    saliency_n_df = pd.concat(subset_saliency_dfs, axis=1)

    saliency_n_df["Total_Saliency"] = saliency_n_df.sum(axis=1)

    saliency_n_df = saliency_n_df.copy()[["Total_Saliency"]]

    saliency_n_df_index = saliency_n_df.index.to_list()
    saliency_n_df_index_clean = [i.split("^")[1] for i in saliency_n_df_index]
    saliency_n_df.index = saliency_n_df_index_clean
    saliency_n_df.index.rename("ROI_Name", inplace=True)
    saliency_n_df.reset_index(inplace=True)

    return saliency_n_df

# An improved version of get_saliency_df(). To aggregate several measurement types each type is ranked by a cumulative rank sum. A weighted
# mean is taken, using the relative weights of each measurement type's rank sum total. 
def get_saliency_df_weighted(
        # Path pointing to .json file with the saliency data we want to plot.
        saliency_path: str,
        # Mapping of dictionaries with their abbreviations
        atlas_mapping: Dict,
        # Which atlas are we plotting? Currently only Neuromorphometrics is implemented
        atlas: str = "neuromorphometrics",
        # Which measurement types are we analyzing
        volumes: list = ["Vgm", "Vwm", "Vcsf"],
        ) -> pd.DataFrame:
    
    if len(volumes) < 2: 
        print("Please specify at least two volume types to perform a weighted plotting! Cancelling!")
        return 

    with open(saliency_path) as f:
        saliency_data = json.load(f)

    # Load data and subset the atlas we want to plot
    saliency_df = pd.DataFrame.from_dict(saliency_data, orient="index")
    keep_sal_index = [i for i in saliency_df.index if i.startswith(atlas_mapping[atlas]["abbrev"])]
    saliency_df_atlas = saliency_df.loc[keep_sal_index,:]

    # Sort our data by the volume type and assign points based on ranking
    vol_type = [i.split("^")[-1] for i in saliency_df_atlas.index]
    saliency_df_atlas["vol_type"] = vol_type
    saliency_df_atlas.sort_values(by=0, axis=0, ascending=False, inplace=True)
    pts = list(range(saliency_df_atlas.shape[0]))
    saliency_df_atlas["points"] = pts

    total_vol_points = {}

    # Compute the scaling factor for each volume type
    for volume in volumes: 
        saliency_vol_pts = saliency_df_atlas[saliency_df_atlas["vol_type"] == volume]["points"].sum()
        total_vol_points[volume] = saliency_vol_pts
    
    total_pts = sum(total_vol_points.values())
    scalers = {vol_type: pts / (total_pts - pts) for vol_type, pts in total_vol_points.items()}

    # Apply scaling to saliency values (assuming column 0 contains saliency)
    saliency_df_atlas["scaled_saliency"] = saliency_df_atlas[0] * saliency_df_atlas["vol_type"].map(scalers)
    saliency_df_scaled = saliency_df_atlas.drop(columns=[0, "vol_type", "points"])

    # Finally sum over the weighted vol type saliencies
    subset_saliency_dfs = []

    for volume in volumes: 
        keep_sal_index_vol = [i for i in saliency_df_scaled.index if i.endswith(volume)]
        saliency_df_scaled_vol= saliency_df_scaled.loc[keep_sal_index_vol,:]
        subset_saliency_dfs.append(saliency_df_scaled_vol)

    saliency_df = pd.concat(subset_saliency_dfs, axis=1)
    saliency_df["Total_Saliency"] = saliency_df.sum(axis=1)
    saliency_df = saliency_df.copy()[["Total_Saliency"]]

    saliency_df_index_clean = [i.split("^")[1] for i in saliency_df.index.to_list()]
    saliency_df.index = saliency_df_index_clean
    saliency_df.index.rename("ROI_Name", inplace=True)
    saliency_df.reset_index(inplace=True)
    return saliency_df


# Function to visualize the saliency results once as a static image and once as an interactive html-file
def visualize_saliency(
        # Path pointing to saliency information saved in .json file
        saliency_path, 
        # Which atlas label mapping should be used. Currently only Neuromorphometrics is implemented
        atlas, 
        # Which measurement types are we analyzing
        volumes, 
        # Which method should be used to aggregate the saliency data 
        method="weighted"):

    # Currently only neuromorphometrics is available
    atlas_mapping = {"neuromorphometrics": {
        "abbrev": "[N]",
        "atlas_nifti_path": "/workspace/data/nifti/neuromorphometrics/neuromorphometrics.nii",
        "atlas_labels_path": "/workspace/data/nifti/neuromorphometrics/1103_3_glm_LabelMap.xml",}
                     }

    if method == "weighted": 
        saliency_df = get_saliency_df_weighted(saliency_path, atlas_mapping, atlas, volumes)
    else:
        saliency_df = get_saliency_df(saliency_path, atlas_mapping, atlas, volumes)

    save_path = str(Path(saliency_path).with_suffix(""))
    
    logging.info("Making saliency plots!")
    interactive_viewer = plot_brain_with_saliency(
        atlas_nifti_path=atlas_mapping[atlas]["atlas_nifti_path"],
        label_map_xml_path=atlas_mapping[atlas]["atlas_labels_path"],
        saliency_df=saliency_df,
        roi_name_col_cleaned='ROI_Name',
        data_col="Total_Saliency",
        save_path=save_path,
        cmap='Reds',  
    )

    interactive_viewer.save_as_html(f"{save_path}_interactive.html")
    logging.info("Saliency plotting successful!")
    return 


################################################## 6. Miscellaneous ##################################################


# To be used after internal and external testing to then plot the joined latent spaces and compare 
# the distributions of diagnoses. This data is then passed to the umap helper function.
def aggregate_testing_data(load_path: str, diagnoses: List) -> pd.DataFrame:
    latent_data_found = list(Path(load_path).rglob(f"*latent_space_data.csv"))
    
    logging.info(f"Found the following latent testing data: {latent_data_found}")

    all_data = []

    for path in latent_data_found:
        latent_data = pd.read_csv(path, header=0, index_col=0).reset_index().rename(columns={"index": "Filename"}).set_index("Filename")
        testing_meta = pd.read_csv(f"{path.parents[0]}/internal_testing/testing_metadata_preds_testing.csv", header=0, index_col=0)
        concat_data = pd.concat([latent_data, testing_meta], axis=1)
        all_data.append(concat_data)
    
    total_data = pd.concat(all_data).drop(columns=diagnoses+["Pred_labels", "Sex_int"]).reset_index()
    total_data.set_index("Filename", inplace=True)

    return total_data


# Extract a list assigning each latent feature in the concatenated latent space the atlas which lies upstream 
def get_group_labels(
        # Mapping of atlas abbreviation : atlas name
        atlas_map: Dict, 
        # Mapping of atlas name : int(input dimension of atlas)
        atlas_input_dims: Dict, 
        # Factor used for downsampling of the latent space from the input dimension of each sub-VAE in training
        downsampling_factor: float, 
        ) -> List:
    
    assert len(atlas_map) == len(atlas_input_dims), "Please ensure that both the atlas mapping and the input dimensions have the same number!"

    atlas_latent_dims = [int(v**downsampling_factor) for v in atlas_input_dims.values()]

    map_conversion = {"cobra": "COBRA", "neuromorphometrics": "Neuromorphometrics", "lpba40": "LPBA40", "suit": "SUIT"}

    atlas_plot_labels = [[map_conversion[atlas]] for atlas in atlas_map.keys()]
    group_order = [map_conversion[atlas] for atlas in atlas_map.keys()]

    group_labels = []

    for idx, group in enumerate(group_order): 
        group_labels.extend([atlas_plot_labels[idx]*atlas_latent_dims[idx]])

    return list(chain.from_iterable(group_labels)), group_order

# Generate and save a plot showing correlations among the latent features. Correlations within one atlas's latent space 
# are highlighted by a black box each. 
def get_latent_feature_correlation_plot(
        path, 
        atlas_map, 
        # Mapping of atlas name : int(input dimension of atlas)
        atlas_input_dims: Dict, 
        # Factor used for downsampling of the latent space from the input dimension of each sub-VAE in training
        downsampling_factor: float = 0.5, 
        ):
    
    data = pd.read_csv(f"{path}/latent_space_data.csv", index_col=0)
    
    corr = data.corr(method="spearman")
    corr = pd.DataFrame(np.flipud(corr.values), 
                        index=corr.index[::-1], 
                        columns=corr.columns)

    group_labels, group_order = get_group_labels(atlas_map, atlas_input_dims, downsampling_factor)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = sns.diverging_palette(250, 15, as_cmap=True)

    # Create the heatmap
    sns.heatmap(corr,  
                cmap=cmap, 
                vmax=1, 
                vmin=-1,
                center=0,
                square=True, 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.5, "aspect": 30},
                annot=False,  # Set to True if you want correlation values shown
                fmt='.2f',
                ax=ax)

    current_pos = 0
    group_colors = ["black", "black", "black", "black"]

    for i, label in enumerate(group_order):
        group_size = group_labels.count(label)
        
        rect_x = Rectangle((current_pos, len(corr) - current_pos - group_size), 
                        group_size, group_size, 
                        linewidth=2, edgecolor=group_colors[i], 
                        facecolor='none', alpha=1)
        ax.add_patch(rect_x)
        
        # Add group labels on the side
        ax.text(-0.5, len(corr) - current_pos - group_size/2, label, 
                rotation=90, ha='right', va='center', 
                fontsize=12, fontweight='bold', color=group_colors[i])
        
        current_pos += group_size

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    ax.set_xlim(-0.5, len(corr) + 0.5)
    ax.set_ylim(-0.5, len(corr) + 0.5)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(f"{path}/latent_space_correlations.png")
    plt.savefig(f"{path}/latent_space_correlations.svg")
    return