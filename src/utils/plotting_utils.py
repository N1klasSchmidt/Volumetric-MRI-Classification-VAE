import math
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Literal
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns

from utils.logging_utils import log_and_print


# ====================================================================================================
#
# **PLOTTING util functions**
#
# Here are all the functions to plot training results. Downstream plotting is usually handled directly
# in the downstream_utils.py functions. The functions here revolve around plotting metrics per training
# epoch and the latent spaces.  
#
# ====================================================================================================


# This function plots the metrics of a model over epochs. It can plot each metric in a separate subplot, but it can also
# combine all the training and validation losses in two subplots, accompanied by an overall train/validation comparison.
# The function can also plot a rolling average of the metrics, and can skip the first N epochs of plotting (the firs
# epochs of training can mess with the plot scales). The function can save the plots to a specified path, and can also
# show the plots in the console.
# The number of plots is variable and can be extended as desired by adjusting the metric_annotations here and in VAE2.py
def metrics_plot(
    # The metrics dataframe, containing the metrics of the model over epochs. Index is epochs, columns are names of metrics.
    # The expected column names are detailed in the metric_annotations dictionary.
    metrics: pd.DataFrame,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The number of epochs to skip in the beginning of the plot. Useful when the first epochs have very differen train/val losses.
    skip_first_n_epochs: int = 0,
    # Do you want to plot the metrics separately or combined?
    plot_types: List[Literal["separate", "combined"]] = ["separate", "combined"],
    # Should the rolling average of metrics be plotted instead? If so, what is the window size of the rolling average?
    rolling_window: int = None,
    # What path should the plots be saved to?
    save_path: str = None,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = True,
    # Should the plots be shown in the console?
    show: bool = False,
):

    # make sure not to modify the original dataframe
    metrics_copy = metrics.copy()

    # remove columns where every value is 0, setting weights for certain metrics to 0 is a common way to ignore that metric
    metrics_copy = metrics_copy.loc[:, (metrics_copy != 0).any(axis=0)]

    # These addendums are used to add information to the title and file name of the plots based on if certain epochs are
    # skipped or rolling averages are used.
    title_addendum = ""
    file_addendum = ""

    # Sometimes the first N epochs of a plot frame the rest of the data poorly
    if skip_first_n_epochs > 0:
        # If there are enough epochs to skip
        if len(metrics_copy) > skip_first_n_epochs:
            # Remove the first N epochs
            metrics_copy = metrics_copy.iloc[skip_first_n_epochs:, :]
            # Add information to the plot title
            title_addendum = f" (first {skip_first_n_epochs} epochs not shown)"

        # If there are not enough epochs to skip
        else:
            # Log that you couldn't skip epochs. Don't change the metrics dataframe.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Cannot skip first {skip_first_n_epochs} epochs in plot, not enough epochs to skip."

            log_and_print(prefix + message)

    # If a rolling average is used
    if rolling_window is not None:
        # If the rolling window is too large, set it to the maximum possible value
        if rolling_window > (len(metrics_copy) - 1):
            # Log that the rolling window is being set to the maximum possible value.
            prefix = (
                f"{datetime.now()+timedelta(hours=2):%H:%M} - Metrics Plot:        "
            )
            message = f"Rolling window of {rolling_window} too large, setting window to {max(len(metrics_copy) - 1, 1)} instead."

            log_and_print(prefix + message)

            # Set the rolling window to the maximum possible value
            rolling_window = max(len(metrics_copy) - 1, 1)

        # If the rolling window is valid, calculate the rolling average
        for metric in metrics_copy.columns:
            # Calculate the rolling average
            metrics_copy[metric] = (
                metrics_copy[metric].rolling(window=rolling_window).mean()
            )

        # Add information to the plot title and file name if a rolling average was used.
        title_addendum += f" (rolling average over {rolling_window} epochs)"
        file_addendum = f"_rolling"

    # Titles and colors for all the metrics we expect to see in the metrics dataframe
    metric_annotations = {
        # annotations for loss components
        "class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        "xcov_loss": {"title": "Cross Covariance Loss", "color": "green"},
        # annotations for training loss components
        "t_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "t_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "t_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "t_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        "t_xcov_loss": {"title": "Cross Covariance Loss", "color": "green"},
        # annotations for validation loss components
        "v_class_loss": {"title": "Diagnostic Classifier Loss", "color": "brown"},
        "v_contr_loss": {"title": "Supervised Contrastive Loss", "color": "brown"},
        "v_recon_loss": {"title": "Reconstruction Loss", "color": "red"},
        "v_kldiv_loss": {"title": "KL-Divergence Loss", "color": "darkorange"},
        "v_xcov_loss": {"title": "Cross Covariance Loss", "color": "green"},
        # annotations for general losses
        "conf_loss": {"title": "Confounder Adversarial Loss", "color": "blue"},
        "train_loss": {"title": "Total Training Loss", "color": "green"},
        "valid_loss": {"title": "Total Validation Loss", "color": "purple"},
        # annotations for confusion metrics
        "accuracy": {"title": "Accuracy", "color": "blue"},
        "precision": {"title": "Precision", "color": "orange"},
        "recall": {"title": "Recall", "color": "green"},
        "f1-score": {"title": "f1-Score", "color": "red"},
        # annotation for learning_rate
        "learning_rate": {"title": "Learning Rate", "color": "black"},
        "VAE_learning_rate": {"title": "VAE Learning Rate", "color": "black"},
        "adv_class_lr": {
            "title": "Adversarial Encoder Learning Rate",
            "color": "black",
        },
        "adv_encod_lr": {
            "title": "Adversarial Classifier Learning Rate",
            "color": "black",
        },
    }

    # If the "separate" plot was desired, plot all metrics separately
    if "separate" in plot_types:
        # collect relevant colors and title names
        titles = [
            metric_annotations[metric]["title"] for metric in metrics_copy.columns
        ]
        colors = [
            metric_annotations[metric]["color"] for metric in metrics_copy.columns
        ]

        # Each row of the plot will contain 4 subplots, length is adjustable! 
        nrows = math.ceil(len(metrics_copy.columns) / 4)

        # Plot each metric in a separate subplot using pandas plot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig, ax = plt.subplots(figsize=(24, 7 * nrows))
            metrics_copy.plot(
                kind="line",
                subplots=True,
                xlabel="Epoch",
                ylabel="Loss",
                layout=(nrows, 4),
                grid=True,
                title=titles,
                color=colors,
                ax=ax,
                use_index=True,
            )

        # Set plot title and layout
        plt.suptitle("Loss Components Over Epochs" + title_addendum, fontsize=30)
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_metrics_separate{file_addendum}.png"
                )
            )
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_metrics_separate{file_addendum}.svg"
                )
            )
        if show:
            plt.show()

        # Close the plot (for stability)
        plt.close(fig)

    # If the "combined" plot was desired, plot all metrics in three subplots:
    # - one for the total training and validation loss together
    # - one for training loss components (KLD loss, recon loss, class loss, etc)
    # - one for validation loss components (KLD loss, recon loss, class loss, etc)
    if "combined" in plot_types:
        # Plot losses in one figure
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Get the toal valid and total train loss
        main_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("valid_")
            ]
        ]

        # Get the components of the train and valid losses
        train_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("train_")
                | metrics_copy.columns.str.startswith("t_")
            ]
        ]
        valid_losses = metrics_copy[
            metrics_copy.columns[
                metrics_copy.columns.str.startswith("valid_")
                | metrics_copy.columns.str.startswith("v_")
            ]
        ]

        # Plot the losses in the three subplots
        for i, model_losses, title in zip(
            range(3),
            [main_losses, train_losses, valid_losses],
            ["Combined", "Train", "Validation"],
        ):
            # Get the relevant colors and titles for the metrics
            colors = [
                metric_annotations[metric]["color"] for metric in model_losses.columns
            ]
            titles = [
                metric_annotations[metric]["title"] for metric in model_losses.columns
            ]

            # Plot the losses in the chosen subplot
            model_losses.plot(
                kind="line",
                xlabel="Epochs",
                ylabel="Losses",
                ax=ax[i],
                color=colors,
                label=titles,
                title=f"{title} Loss Over Epochs" + title_addendum,
                use_index=True,
            )

        # Set the plot title and layout
        plt.tight_layout()

        # Save if specified, show if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_losses_combined{file_addendum}.png"
                )
            )
            plt.savefig(
                os.path.join(
                    save_path, f"{timestamp}_losses_combined{file_addendum}.svg"
                )
            )
        if show:
            plt.show()

        plt.close(fig)


# Latent space plots are generated on model checkpoints to visualize the latent space of the model. The function can
# plot the latent space in PCA and UMAP, and can plot the latent space for different variables. The function can save
# the plots to a specified path, and can also show the plots in the console.
def latent_space_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    # These variables should be present in the data.obs dataframe, and should be categorical or continuous.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The type of data being plotted. Used for the plot title and file name.
    data_type: str,
    # What colors should be used for plotting the diagnoses? 
    diagnosis_colors: Dict,
    # The types of plots to generate. The function can generate PCA and UMAP plots. This functione expects the anndata to already
    plot_types: List[str] = ["umap", "pca"],
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
):

    # If PCA was specified, plot the latent space in PCA
    if "Diagnosis" in data.obs:
        data.uns["Diagnosis_colors"] = [diagnosis_colors[cat] for cat in data.obs["Diagnosis"].cat.categories if cat in diagnosis_colors]
    
    if "Diagnosis_pred" in data.obs:
        data.uns["Diagnosis_pred_colors"] = [diagnosis_colors[cat] for cat in data.obs["Diagnosis_pred"].cat.categories if cat in diagnosis_colors]

    # If PCA was specified, plot the latent space in PCA
    if "pca" in plot_types:
        # Plot latent space pca
        sc.pl.pca(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            show=show,
            return_fig=show,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca.png",
                )
            )
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca.svg",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

    # If UMAP was specified, plot the latent space in UMAP
    if "umap" in plot_types:
        # Plot latent space umap
        sc.pl.umap(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            show=show,
            return_fig=show,
            frameon=False,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap.png",
                )
            )
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap.svg",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

# Slightly adjusted version of the above function to be used during latent plotting of prediction / testing runs
def latent_space_plot_pred(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    # These variables should be present in the data.obs dataframe, and should be categorical or continuous.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The type of data being plotted. Used for the plot title and file name.
    data_type: str,
    # Name of the atlas / model whose latent space is being plotted.
    run_id: str,
    # What colors should be used for plotting the diagnoses? 
    diagnosis_colors: Dict,
    # The types of plots to generate. The function can generate PCA and UMAP plots. This functione expects the anndata to already
    plot_types: List[str] = ["umap", "pca"],
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
    size: int = 30,
):  

    # If PCA was specified, plot the latent space in PCA
    if "Diagnosis" in data.obs:
        data.uns["Diagnosis_colors"] = [diagnosis_colors[cat] for cat in data.obs["Diagnosis"].cat.categories if cat in diagnosis_colors]
    
    if "Diagnosis_pred" in data.obs:
        data.uns["Diagnosis_pred_colors"] = [diagnosis_colors[cat] for cat in data.obs["Diagnosis_pred"].cat.categories if cat in diagnosis_colors]

    # If PCA was specified, plot the latent space in PCA
    if "pca" in plot_types:
        # Plot latent space pca with all variables in one plot
        fig = sc.pl.pca(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=size,
            show=show,
            return_fig=True,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca_atlas-{run_id}.png",
                )
            )
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_pca_atlas-{run_id}.svg",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()

    # If UMAP was specified, plot the latent space in UMAP
    if "umap" in plot_types:
        # Plot latent space umap with all variables in one plot
        fig = sc.pl.umap(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=size,
            #marker=custom_markers_list,
            show=show,
            return_fig=True,
            frameon=False,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap_atlas-{run_id}.png",
                )
            )
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_umap_atlas-{run_id}.svg",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()
    
    if "tsne" in plot_types:
        # Plot latent space t-SNE with all variables in one plot
        fig = sc.pl.tsne(
            data,
            ncols=len(plot_by),
            color=plot_by,
            wspace=0.3,
            size=30,
            show=show,
            return_fig=True,
        )

        # Save the plot if specified
        if save:
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_tsne_atlas-{run_id}.png",
                )
            )
            plt.savefig(
                os.path.join(
                    save_path,
                    f"{timestamp}_e{epoch}_{data_type}_latent_tsne_atlas-{run_id}.svg",
                )
            )
        # Show the plot if specified
        plt.show() if show else plt.close()


# Outputs a violin plot showing the distribution of confidence scores for correct and incorrect predictions. 
def prediction_confidence_plot(
        true_labels: List,
        prediction_logits: List,
        prediction_probs: List,
        epoch: int,
        # The path to save the plots to.
        save_path: str,
        # The timestamp of the run, used for the file name when saving the plots.
        timestamp: str,
        atlas_name: str,
        save: bool = True,
        show: bool = False
    ):

    correct_confidences = []
    incorrect_confidences = []
    ctt_confidences = []
    ctt_mdd_confidences = []
    ctt_schz_confidences = []
    ctt_hc_confidences = []

    for i in range(len(true_labels)):
        if true_labels[i] == prediction_logits[i]:
            correct_confidences.append(prediction_probs[i])
        elif true_labels[i] != prediction_logits[i] and true_labels[i] != -1:
            incorrect_confidences.append(prediction_probs[i])
        elif true_labels[i] != prediction_logits[i] and true_labels[i] == -1: 
            ctt_confidences.append(prediction_probs[i])

            # Further check as what CTT was misdiagnosed
            if prediction_logits[i] == 0:
                ctt_hc_confidences.append(prediction_probs[i])
            if prediction_logits[i] == 1:
                ctt_mdd_confidences.append(prediction_probs[i])
            if prediction_logits[i] == 2:
                ctt_schz_confidences.append(prediction_probs[i])
    
    correct_confidences = np.array(correct_confidences)
    incorrect_confidences = np.array(incorrect_confidences)
    ctt_confidences = np.array(ctt_confidences)

    # Combine into a DataFrame for easy plotting with Seaborn
    df = pd.DataFrame({
        'Confidence Score': np.concatenate([correct_confidences, incorrect_confidences, ctt_confidences, ctt_hc_confidences, ctt_mdd_confidences, ctt_schz_confidences]),
        'Prediction Outcome': ['Correct'] * len(correct_confidences) + ['Incorrect']
    })
    
    # Create the violin plot
    plt.figure(figsize=(24, 8))
    sns.violinplot(x='Prediction Outcome', 
                   y='Confidence Score', 
                   data=df, 
                   inner='quartile', 
                   hue='Prediction Outcome', 
                   palette={'Correct': 'palegreen', 
                            'Incorrect': 'tomato', 
                            }, 
                   legend=False)
    plt.title('Distribution of Confidence Scores by Prediction Outcome')
    plt.ylim(0, 1) # Assuming confidence scores are between 0 and 1
    plt.grid(axis='y', linestyle='--')
    
    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_confidence_violin-{atlas_name}.png",
            )
        )
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_confidence_violin-{atlas_name}.svg",
            )
        )
    # Show the plot if specified
    plt.show() if show else plt.close()
    return correct_confidences, incorrect_confidences, ctt_confidences


# This function plots the latent space of the model, broken down by diagnosis. Each diagnosis will be plotted in a separate
# column, and each row will be one of the datasets in the data. The first row is all the datasets combined, as an overview.
# The function can save the plots to a specified path, and can also show the plots in the console.
def latent_space_batch_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The type of data being plotted, such as "training" or "validation". Used for the plot title and file name.
    data_type: str,
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
    # A descriptor to add to the end of file name of the plot.
    descriptor: str = "",
    # The size of the points in the plot.
    size: int = 30,
):
    # Get the unique diagnoses and datasets in the data
    diagnoses = data.obs["Diagnosis"].unique()
    datasets = data.obs["Dataset"].unique()

    # The number of rows and columns in the plot, 1 column per diagnosis, 1 row per dataset +1 for all datasets combined
    nrows = len(datasets) + 1
    ncols = len(diagnoses)

    # Initialize the plot
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    # Assign a color to each dataset so that the palette is consistent across all subplots
    palette = {dataset: sns.color_palette()[i] for i, dataset in enumerate(datasets)}

    # For each diagnosis; dia is used for the column index, and diagnosis is the diagnosis name
    for dia, diagnosis in enumerate(diagnoses):

        # Get the data for only this diagnosis
        diagnosis_data = data[data.obs["Diagnosis"] == diagnosis].copy()

        # Plot all the data in this diagnosis
        sc.pl.umap(
            diagnosis_data,
            ax=axs[0, dia],
            title=f"{diagnosis} - All",
            color="Dataset",
            palette=palette,
            size=size,
            show=False,
            return_fig=False,
            frameon=False,
        )

        # For each dataset; dat is used for the row index, and dataset is the dataset name
        for dat, dataset in enumerate(data.obs["Dataset"].unique(), start=1):

            # Get the data for this dataset and diagnosis
            subset_data = data[
                (data.obs["Diagnosis"] == diagnosis) & (data.obs["Dataset"] == dataset)
            ].copy()

            # If there is no data for this dataset and diagnosis, skip this subplot
            if len(subset_data) == 0:
                # remove distracting axis
                axs[dat, dia].axis("off")
                continue

            # plot grey background of all the data in this diagnosis
            sc.pl.umap(
                diagnosis_data,
                ax=axs[dat, dia],
                show=False,
                return_fig=False,
                size=size,
                frameon=False,
            )

            # plot the data for this dataset and diagnosis in color
            sc.pl.umap(
                subset_data,
                ax=axs[dat, dia],
                title=f"{diagnosis} - {dataset}",
                color="Dataset",
                palette=palette,
                size=size,
                show=False,
                return_fig=False,
                frameon=False,
            )

    # Set the plot title and layout
    plt.suptitle(
        f"Batch Effect Breakdown of {data_type.capitalize()} Data (UMAP)",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.90)

    # Save if specified, show if specified
    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_batch_{data_type}{descriptor}.png",
            )
        )
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_batch_{data_type}{descriptor}.svg",
            )
        )

    plt.show() if show else plt.close()


# This plot breaks down a set of variables by dataset in the latent space. Each variable passed through plot_by will be
# plotted in the latent space as a column. And then each row will be one of the datasets in the data. The first row is
# all the datasets combined, as an overview. The function can save the plots to a specified path, and can also show the
# plots in the console.
def latent_space_details_plot(
    # The AnnData object containing the latent representation of data. Usually the output for model.extract_latent_space().
    data: ad.AnnData,
    # The type of data being plotted, such as "training" or "validation". Used for the plot title and file name.
    data_type: str,
    # The variables to plot the latent space by. The function will plot the latent space for each variable in this list.
    plot_by: List[str],
    # The path to save the plots to.
    save_path: str,
    # The timestamp of the run, used for the file name when saving the plots.
    timestamp: str,
    # The current epoch of the model, used for the plot title and file name.
    epoch: int,
    # What colors should be used for plotting the diagnoses? 
    diagnosis_colors: Dict,
    # Should the plots be shown in the console?
    show: bool = True,
    # Should the plots be saved? If so, you must set the save_path.
    save: bool = False,
    # A descriptor to add to the end of file name of the plot.
    descriptor: str = "",
    # The size of the points in the plot.
    size: int = 30,
):

    # Function to set colors for diagnosis-related columns
    def set_diagnosis_colors(column_name):
        if column_name in data.obs:
            # Convert to categorical if not already
            if not pd.api.types.is_categorical_dtype(data.obs[column_name]):
                data.obs[column_name] = pd.Categorical(data.obs[column_name])
            
            # Set colors in uns
            colors = []
            for cat in data.obs[column_name].cat.categories:
                if cat in diagnosis_colors:
                    colors.append(diagnosis_colors[cat])
                else:
                    colors.append('#808080')  # Default gray for unknown categories
            
            data.uns[f"{column_name}_colors"] = colors

    # Set colors for diagnosis columns
    set_diagnosis_colors("Diagnosis")
    set_diagnosis_colors("Diagnosis_pred")

    # One row for each dataset, +1 row for all the datasets combined
    nrows = len(data.obs["Dataset"].unique()) + 1
    # One column for each variable to plot by
    ncols = len(plot_by)
    # Initalize the plot
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))

    # Make sure every subplot has the same color palette
    palettes = {}
    # For each variable to plot by
    for variable in plot_by:
        # If the variable is categorical, assign a color palette to it (non-categorical variables will use the default palette)
        if data.obs[variable].dtype.name == "category":
            # Special handling for diagnosis-related variables
            if variable in ["Diagnosis", "Diagnosis_pred"]:
                # Use the custom diagnosis colors
                palettes[variable] = {
                    subgroup: diagnosis_colors.get(subgroup, '#808080')  # Use custom colors or default gray
                    for subgroup in data.obs[variable].unique()
                }
            else:
                # Assign a color to each subgroup of the variable, and then by passing this palette later, each subgroup will have
                # the same color in every subplot
                palettes[variable] = {
                    subgroup: sns.color_palette()[i]
                    for i, subgroup in enumerate(data.obs[variable].unique())
                }

    # For each variable to plot by: c is used for the column index, and category is the variable name
    for c, category in enumerate(plot_by):
        # Is it a categorical variable? We want to know if we need to pass the palette from above.
        is_category = data.obs[category].dtype.name == "category"
        # Is the variable "Diagnosis"?
        is_diagnosis = category == "Diagnosis"
        # If it is DIagnosis, the legend should be on the data, otherwise on the right margin
        legend_loc = "right_margin" #on data" if is_diagnosis else "right margin"

        # Plot a grey background of the data
        sc.pl.umap(
            data,
            ax=axs[0, c],
            show=False,
            return_fig=False,
            size=size,
            frameon=False,
        )
        # Plot all the data, colored by this variable
        sc.pl.umap(
            data[data.obs["Data_Type"] == data_type].copy(),
            ax=axs[0, c],
            title=f"{category} - All",
            color=[category],
            show=False,
            return_fig=False,
            # pass the palette if this is one of the categorical variables
            palette=(palettes[category] if is_category else None),
            size=size*3,
            #legend_loc=legend_loc,
            # For the "Age" variable we set a colorbar manually
            vmax=80 if category == "Age" else None,
            vmin=20 if category == "Age" else None,
            frameon=False,
        )

        # Then for each dataset, repeat the process. d is used for the row index, and dataset is the dataset name
        for d, dataset in enumerate(data.obs["Dataset"].unique(), start=1):
            # Grey background of all the data
            sc.pl.umap(
                data,
                ax=axs[d, c],
                show=False,
                return_fig=False,
                size=size,
                frameon=False,
            )
            # Foreground of this datatset's data, colored by this variable
            sc.pl.umap(
                data[
                    (data.obs["Data_Type"] == data_type)
                    & (data.obs["Dataset"] == dataset)
                ].copy(),
                ax=axs[d, c],
                title=f"{category} - {dataset}",
                color=[category],
                show=False,
                return_fig=False,
                # pass the palette if this is one of the categorical variables
                palette=palettes[category] if is_category else None,
                size=size*3,
                #legend_loc=legend_loc,
                # For the "Age" variable we set a colorbar manually
                vmax=(80 if category == "Age" else None),
                vmin=(20 if category == "Age" else None),
                frameon=False,
            )

    # Set the plot title and layout
    plt.suptitle(
        f"Latent Space of {data_type.capitalize()} Data (UMAP)",
        fontsize=30,
    )
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.90)

    # Save if specified, show if specified
    if save:
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_details_{data_type}{descriptor}.png",
            )
        )
        plt.savefig(
            os.path.join(
                save_path,
                f"{timestamp}_e{epoch}_latent_details_{data_type}{descriptor}.svg",
            )
        )

    plt.show() if show else plt.close()
