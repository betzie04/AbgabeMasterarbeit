import os
from scipy.stats import spearmanr
import torch
import itertools
from collections import defaultdict
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib as mpl

import seaborn as sns
import os
import sys
import argparse
import random
import pandas as pd
import torch.nn as nn
import geoopt
import re
import pickle

from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def make_path(root, models_type, task, split, seed_suffix):
    return os.path.join(f"{root}/{models_type}/task-{task}-{split}{seed_suffix}")

def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes Spearman rank correlation between two tensors on GPU.
    Input: x, y of shape (n_samples, d)
    Output: average Spearman correlation across d dimensions
    
    Correctly handles tied values by assigning average ranks.
    """
    # Ensure tensors are float type
    x = x.float()
    y = y.float()
    
    # Flatten the first dimension if necessary (e.g., batch size of 1)
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))

    # If only one sample, correlation is undefined
    if x.size(0) <= 1:
        return float('nan')

    def compute_ranks_with_ties(tensor):
        """Compute ranks handling tied values correctly (average rank method)"""
        ranks = torch.zeros_like(tensor, dtype=torch.float32)
        for col in range(tensor.size(1)):
            col_data = tensor[:, col]
            # Get sorting indices
            sorted_indices = col_data.argsort()
            # Create ranks array
            col_ranks = torch.zeros_like(col_data, dtype=torch.float32)
            
            # Handle ties by assigning average ranks
            i = 0
            while i < len(sorted_indices):
                current_value = col_data[sorted_indices[i]]
                # Find all indices with the same value
                j = i
                while j < len(sorted_indices) and torch.abs(col_data[sorted_indices[j]] - current_value) < 1e-10:
                    j += 1
                
                # Assign average rank to all tied values
                avg_rank = (i + j - 1) / 2.0  # Average of ranks i to j-1
                for k in range(i, j):
                    col_ranks[sorted_indices[k]] = avg_rank
                
                i = j
            
            ranks[:, col] = col_ranks
        
        return ranks

    x_rank = compute_ranks_with_ties(x)
    y_rank = compute_ranks_with_ties(y)

    x_mean = x_rank.mean(dim=0, keepdim=True)
    y_mean = y_rank.mean(dim=0, keepdim=True)

    cov = ((x_rank - x_mean) * (y_rank - y_mean)).mean(dim=0)
    std_x = x_rank.std(dim=0, unbiased=False)
    std_y = y_rank.std(dim=0, unbiased=False)

    # Initialize correlation tensor
    corr = torch.zeros_like(cov)
    
    # Handle non-constant columns (standard case)
    valid = (std_x > 1e-8) & (std_y > 1e-8)
    if valid.sum() > 0:
        corr[valid] = cov[valid] / (std_x[valid] * std_y[valid])
    
    # Handle constant columns: if both columns are constant (std ≈ 0)
    # For identical tensors, this will be 1.0; for different constant values, it's undefined but we use 0
    constant_cols = (std_x <= 1e-8) & (std_y <= 1e-8)
    if constant_cols.sum() > 0:
        # For constant columns, check if the constant values are the same
        for col in range(x.size(1)):
            if constant_cols[col]:
                # All values in this column are the same, so check the first value
                if torch.abs(x[0, col] - y[0, col]) < 1e-8:
                    corr[col] = 1.0  # Same constant value
                else:
                    corr[col] = 0.0  # Different constant values (undefined, but we use 0)
    
    # Handle mixed case: one constant, one variable - correlation is 0
    mixed_cols = ((std_x <= 1e-8) & (std_y > 1e-8)) | ((std_x > 1e-8) & (std_y <= 1e-8))
    corr[mixed_cols] = 0.0
    
    # Return mean correlation
    return corr.mean().item()

def get_performance(path):
    predictions = torch.load(path)
    
    performance = []
    
    for (lr1, pred_dict1) in predictions.items():
        preds1 = torch.tensor(pred_dict1['test_pred'], dtype=torch.float32)
        probs1 = torch.tensor(pred_dict1['test_probs'], dtype=torch.float32)
        labels1 = torch.tensor(pred_dict1['test_labels'])

        if labels1[0][0] != -1:
            # Convert to appropriate formats
            labels1_np = labels1.cpu().numpy()
            preds1_np = preds1.cpu().numpy()
            probs1_np = probs1.cpu().numpy()

            # Handle multi-label/multi-output format
            if probs1_np.ndim == 3 and probs1_np.shape[-1] == 2:
                # Multi-label case: extract positive class probabilities and convert to binary predictions
                probs1_positive = probs1_np[:, :, 1]  # Shape: (N, 8) - probabilities for positive class
                
                
                # Flatten for metrics calculation
                labels1_flat = labels1_np.flatten()
                preds1_flat = preds1.flatten()
                probs1_flat = probs1_positive.flatten()
                
                # Calculate metrics on flattened data
                accuracy_score1 = accuracy_score(labels1_flat, preds1_flat)
                f1_1 = f1_score(labels1_flat, preds1_flat, average='weighted')
                
                # For Brier score, use positive class probabilities
                try:
                    brier1 = torch.tensor(log_loss(labels1_flat, probs1_flat))
                except ValueError:
                    # Fallback if log_loss fails
                    brier1 = torch.tensor(np.mean((labels1_flat - probs1_flat) ** 2))
                    
            else:
                # Single-label case (original logic)
                accuracy_score1 = accuracy_score(labels1_np, preds1_np.argmax(axis=1))
                f1_1 = f1_score(labels1_np, preds1_np.argmax(axis=1), average='weighted')
                
                # Brier score for single-label
                try:
                    brier1 = log_loss(labels1_np, probs1_np, labels=np.unique(labels1_np))
                except ValueError:
                    brier1 = 0.0
                    brier2 = 0.0
            
            has_label_metrics = True
          
        # Confidence und Korrelation
        confidence1 = probs1.mean()


        entry = {
            "lr": lr1,
            "total": len(preds1),
            "confidence": confidence1.item(),
            "brier1": brier1.item() if hasattr(brier1, 'item') else float(brier1),
            "acc_model1": accuracy_score1,
            "f1_model1": f1_1
        }
        
        performance.append(entry)
    
    return performance

def get_predictions(first_path, second_path, device, batch_size=32):
    first_predictions = torch.load(first_path)
    second_predictions = torch.load(second_path)
    agreements = []
    has_label_metrics = False
    
    for (lr1, pred_dict1), (lr2, pred_dict2) in zip(first_predictions.items(), second_predictions.items()):
        preds1 = torch.tensor(pred_dict1['test_pred'], dtype=torch.float32)
        preds2 = torch.tensor(pred_dict2['test_pred'], dtype=torch.float32)
        probs1 = torch.tensor(pred_dict1['test_probs'], dtype=torch.float32)
        probs2 = torch.tensor(pred_dict2['test_probs'], dtype=torch.float32)
        labels1 = torch.tensor(pred_dict1['test_labels'])
        labels2 = torch.tensor(pred_dict2['test_labels'])

        # Prüfe Dimensionen und NaN
        assert preds1.shape == preds2.shape, f"Shape mismatch: {preds1.shape} vs {preds2.shape}"
        assert not torch.isnan(preds1).any(), "NaN in preds1"
        assert not torch.isnan(preds2).any(), "NaN in preds2"


        # Metriken
        same_class = (preds1.argmax(dim=1) == preds2.argmax(dim=1)).sum().item()
        same_class_percentage = (same_class / len(preds1)) * 100

        similar_probs = torch.isclose(probs1, probs2, rtol=0.05, atol=0.01).sum().item()            
        
        brier1 = torch.tensor(0.0)
        brier2 = torch.tensor(0.0)
        if labels1[0][0] != -1:
            # Convert to appropriate formats
            labels1_np = labels1.cpu().numpy()
            labels2_np = labels2.cpu().numpy()
            preds1_np = preds1.cpu().numpy()
            preds2_np = preds2.cpu().numpy()
            probs1_np = probs1.cpu().numpy()
            probs2_np = probs2.cpu().numpy()

            # Handle multi-label/multi-output format
            if probs1_np.ndim == 3 and probs1_np.shape[-1] == 2:
                # Multi-label case: extract positive class probabilities and convert to binary predictions
                probs1_positive = probs1_np[:, :, 1]  # Shape: (N, 8) - probabilities for positive class
                probs2_positive = probs2_np[:, :, 1]  # Shape: (N, 8)
                
                
                # Flatten for metrics calculation
                labels1_flat = labels1_np.flatten()
                labels2_flat = labels2_np.flatten()
                preds1_flat = preds1.flatten()
                preds2_flat = preds2.flatten()
                probs1_flat = probs1_positive.flatten()
                probs2_flat = probs2_positive.flatten()
                
                # Calculate metrics on flattened data
                accuracy_score1 = accuracy_score(labels1_flat, preds1_flat)
                accuracy_score2 = accuracy_score(labels2_flat, preds2_flat)
                f1_1 = f1_score(labels1_flat, preds1_flat, average='weighted')
                f1_2 = f1_score(labels2_flat, preds2_flat, average='weighted')
                
                # For Brier score, use positive class probabilities
                try:
                    brier1 = torch.tensor(log_loss(labels1_flat, probs1_flat))
                    brier2 = torch.tensor(log_loss(labels2_flat, probs2_flat))
                except ValueError:
                    # Fallback if log_loss fails
                    brier1 = torch.tensor(np.mean((labels1_flat - probs1_flat) ** 2))
                    brier2 = torch.tensor(np.mean((labels2_flat - probs2_flat) ** 2))
                    
            else:
                # Single-label case (original logic)
                accuracy_score1 = accuracy_score(labels1_np, preds1_np.argmax(axis=1))
                accuracy_score2 = accuracy_score(labels2_np, preds2_np.argmax(axis=1))
                f1_1 = f1_score(labels1_np, preds1_np.argmax(axis=1), average='weighted')
                f1_2 = f1_score(labels2_np, preds2_np.argmax(axis=1), average='weighted')
                
                # Brier score for single-label
                try:
                    brier1 = log_loss(labels1_np, probs1_np, labels=np.unique(labels1_np))
                    brier2 = log_loss(labels2_np, probs2_np, labels=np.unique(labels2_np))
                except ValueError:
                    brier1 = 0.0
                    brier2 = 0.0
            
            has_label_metrics = True
          
        # Confidence und Korrelation
        confidence1 = probs1.mean()
        confidence2 = probs2.mean()
        
        # Handle Spearman correlation for different tensor shapes with robust NaN checking
        spearman_corr = spearman_corr_torch(preds1, preds2)
        spearmanr = spearman_corr_torch(probs1, probs2)
        print(spearmanr)

        entry = {
            "lr": lr1,
            "same_class": same_class,
            "same_class_percentage": same_class_percentage,
            "similar_probs": similar_probs,
            "total": len(preds1),
            "confidence1": confidence1.item(),
            "confidence2": confidence2.item(),
            "spearman_corr": spearmanr
        }
        
        # Add multi-label specific metrics if available
        if has_label_metrics:
            # Handle brier scores - they can be float or tensor
            brier1_val = brier1.item() if hasattr(brier1, 'item') else float(brier1)
            brier2_val = brier2.item() if hasattr(brier2, 'item') else float(brier2)
            
            entry.update({            
            "brier1": brier1_val,
            "brier2": brier2_val,
            "brier_diff": brier1_val - brier2_val,
            "acc_model1": accuracy_score1,
            "acc_model2": accuracy_score2,
            "f1_model1": f1_1,
            "f1_model2": f1_2,})
        
        agreements.append(entry)
    
    return agreements



def extract_seed(model_str):
    # Extrahiert die Zahl in Klammern, z.B. multiberts(10) -> 10
    match = re.search(r"\((\d+)\)", model_str)
    return int(match.group(1)) if match else -1

def sort_key(pair):
    # Zerlegt "multiberts(1) vs multiberts(2)" in zwei Teile und sortiert numerisch
    left, right = pair.split(" vs ")
    left_name = re.sub(r"\(\d+\)", "", left)
    right_name = re.sub(r"\(\d+\)", "", right)
    left_seed = extract_seed(left)
    right_seed = extract_seed(right)
    return (left_name, left_seed, right_name, right_seed)
def model_sort_key(name):
    # Sortiert multiberts(X) numerisch, andere Modelle alphabetisch davor/danach
    import re
    if name.startswith("multiberts"):
        match = re.search(r"\((\d+)\)", name)
        return (1, int(match.group(1))) if match else (1, float('inf'))
    elif name == "roberta":
        return (0, 0)
    elif name == "electra":
        return (0, 1)
    else:
        return (2, name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["mrpc"], nargs="+", help="Name(s) of the task(s) (e.g., sst2, mnli).")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--layer", type=int, default=12, help="Layer of the model to extract embeddings from.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the model.")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1], help="List of learning rates to use for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, nargs="+", default=["roberta", "electra", "multiberts"],  help="Type of model to use (bert, roberta, electra).")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Directory to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--activation", type=str, default="_3_ReLu_save_logits_andProb", help="Name of the activation function used in the model (e.g., _3_ReLu, _4_Sigmoid).")

    args = parser.parse_args()

    ROOT_PATH = args.root_path
    DEVICE = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_size = args.hidden_size
    layer = args.layer
    learning_rates = args.learning_rates
    activation = args.activation

    all_model_names = []
    all_distances = []
    all_performance = []
    logger.info(f"running on {DEVICE}")
    all_comparisons = []
    for task_name in args.task_name:
        if task_name == "mnli":
            output_size = 3
        else:
            output_size = 2

        
        for first_model, second_model in list(itertools.product(args.model_type, repeat=2)):
            # set seed range if first_model is multiberts, otherwise use None
            first_seed_range = range(args.num_seeds) if first_model == "multiberts" else [None]
            #for second_model in args.model_type:
                #if first_model != second_model:          
            second_seed_range = range(args.num_seeds) if second_model == "multiberts" else [None]     
            # iterate over all multibert models and compare them with electra and roberta
            for first_seed in first_seed_range:
                
                for second_seed in second_seed_range:
                    print(first_model, first_seed, second_model, second_seed)
                    #if first_model == second_model and first_seed is not None and second_seed is not None and first_seed > second_seed:
                    #    continue  # Nur eine Richtung bei gleichen Modellen/Seeds
                    # Define paths dynamically based on the model type
                    first_seed_suffix = f"-seed-{first_seed}" if first_model == "multiberts" else ""
                    second_seed_suffix = f"-seed-{second_seed}" if second_model == "multiberts" else ""
                    # Load the prediction dict
                    first_pred_labels = make_path(f"{ROOT_PATH}predictions{activation}", first_model, task_name, "predictions", first_seed_suffix)+".pt"
                    second_pred_labels = make_path(f"{ROOT_PATH}predictions{activation}", second_model, task_name, "predictions", second_seed_suffix)+".pt"
                    comparison_metrics = get_predictions(first_pred_labels, second_pred_labels, DEVICE, batch_size)
                    performance = get_performance(first_pred_labels)

                    for metrics in comparison_metrics:
                        entry = {
                            "task": task_name,
                            "models": f"{first_model}({first_seed}) vs {second_model}({second_seed})",
                            "lr": metrics.get("lr", None),  # Falls Lernrate in den Metriken ist
                            **metrics  # Alle Metriken aus der Prediction
                        }
                        all_comparisons.append(entry)
                    for performances in performance:
                        entry = {
                            "task": task_name,
                            "models": f"{first_model}({first_seed})",
                            "lr": metrics.get("lr", None),  # Falls Lernrate in den Metriken ist
                            **performances  # Alle Metriken aus der Prediction
                        }
                        all_performance.append(entry)
                
                        plot_title = f"{first_model} vs {second_model} (Seed: {first_seed} vs {second_seed})"
                    logger.info(f"Comparing {first_model} (Seed: {first_seed}) with {second_model} (Seed: {second_seed})")

    #all_comparisons = []
    #for task_name in args.task_name:
    #    with open("data/all_comparisons.pkl", "rb") as f:
    #        all_comparisons = pickle.load(f)

        task_comparisons = [c for c in all_comparisons if c['task'] == task_name]
        if not task_comparisons:
            continue
        df = pd.DataFrame(task_comparisons)
        # 1. Prepare data
        df['same_class_percentage'] = df['same_class_percentage'].astype(str).str.split('.').str[0].astype(float)
        df['models'] = df['models'].str.replace('(None)', '')

        # 2. Extract model names and create comparison labels
        df[['first_model', 'second_model']] = df['models'].str.split(' vs ', expand=True)
        df['model_pair'] = df['first_model'] + " vs " + df['second_model']  # New column for paired names
        df = df.sort_values(by=["models"]).reset_index(drop=True)

       # df = df.sort_values(by="model_pair").reset_index(drop=True)
        # 3. Create plots
        #plt.figure(figsize=(12, 12))
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # 2 rows, 2 columns

        # 📊 1. Heatmap (sortiert nach Modellnummer)
        heatmap_data = df.pivot_table(
            index='first_model',
            columns='second_model',
            values='same_class_percentage',
            aggfunc='mean'
        )

        # Symmetrisieren der Matrix: NaNs durch Werte der transponierten Matrix ersetzen
        heatmap_data_sym = heatmap_data.copy()

        # Modelle nach model_sort_key sortieren
        all_models = sorted(list(set(heatmap_data_sym.index) | set(heatmap_data_sym.columns)), key=model_sort_key)
        if "roberta" in all_models:
            all_models.remove("roberta")
            all_models = ["roberta"] + all_models 
        heatmap_data_sym = heatmap_data_sym.reindex(index=all_models, columns=all_models)

        # Nun heatmap_data_sym in der Heatmap verwenden
        sns.heatmap(
            heatmap_data_sym,
            ax=axs[0],
            cmap="YlGnBu",
            annot=False,
            vmin=0,
            vmax=100,
            cbar_kws={'label': '% Same Class Predictions'}, 
            square=True
        )
        axs[0].set_title("Heatmap: Prediction Agreement")
        axs[0].set_xlabel("Second Model")
        axs[0].set_ylabel("First Model")
        
        
        ## 📊 2. Heatmap of Spearman correlation(sortiert nach Modellnummer)
        spearman_heatmap_data = df.pivot_table(
            index='first_model',
            columns='second_model',
            values='spearman_corr',
            aggfunc='mean'
        )

        # Modelle nach model_sort_key sortieren
        #all_models = sorted(list(set(heatmap_data_sym.index) | set(heatmap_data_sym.columns)))
        all_models = sorted(list(set(heatmap_data_sym.index) | set(heatmap_data_sym.columns)), key=model_sort_key)
        if "roberta" in all_models:
            all_models.remove("roberta")
            all_models = ["roberta"] + all_models 
        spearman_heatmap_data = spearman_heatmap_data.reindex(index=all_models, columns=all_models)

        # Nun heatmap_data_sym in der Heatmap verwenden
        sns.heatmap(
            spearman_heatmap_data,
            ax=axs[1],
            cmap="YlGnBu",
            annot=False,
            vmin=0,
            vmax=1,
            cbar_kws={'label': '% Spearman Correlation'},
            square=True
        )
        axs[1].set_title("Heatmap: Spearman Correlation")
        axs[1].set_xlabel("Second Model")
        axs[1].set_ylabel("First Model")
        plt.suptitle(f"Task: {task_name}", fontsize=16, y=1.02)

        plt.tight_layout()
        plt.savefig(f"data/plots/{activation}_HeatMap_Spearman_Accuracy_{task_name}_subplots.png", dpi=200, bbox_inches='tight')
        plt.show()
       # Accuracy-Daten nach Modellen gruppieren (Durchschnitt über alle Learning Rates)
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # 2 rows, 2 columns

        if task_name == "mnli":
        # DataFrame erstellen
            all_performance_df = pd.DataFrame(all_performance)
            all_performance_df['models'] = all_performance_df['models'].str.replace('(None)', '')
            accuracy_by_model = all_performance_df.groupby('models')['acc_model1'].mean().sort_values(ascending=False)
            f1_by_model = all_performance_df.groupby('models')['f1_model1'].mean().sort_values(ascending=False)

            # Subplots erstellen
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            # X-Tick-Labels ggf. kürzen für bessere Lesbarkeit
            #def shorten(label):
            #    return label.replace("multiberts", "mb").replace("electra", "el").replace("roberta", "ro")

            acc_labels = [label for label in accuracy_by_model.index]
            f1_labels = [label for label in f1_by_model.index]

            # Accuracy-Plot
            axes[0].bar(acc_labels, accuracy_by_model.values, alpha=0.7, color='lightblue', edgecolor='black', linewidth=1)
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('Score')
            axes[0].set_xticks(range(len(acc_labels)))
            axes[0].set_xticklabels(acc_labels, rotation=45, ha='right', fontsize=9)
            axes[0].grid(True, alpha=0.3)

            # F1-Score-Plot
            axes[1].bar(f1_labels, f1_by_model.values, alpha=0.7, color='lightgreen', edgecolor='black', linewidth=1)
            axes[1].set_title('Model F1 Score')
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score')  # <== Hinzugefügt!
            axes[1].set_xticks(range(len(f1_labels)))
            axes[1].set_xticklabels(f1_labels, rotation=45, ha='right', fontsize=9)
            axes[1].grid(True, alpha=0.3)

            # Layout justieren
            plt.subplots_adjust(bottom=0.3, wspace=0.2)
            plt.savefig(f"data/plots/{activation}_Accuracy_F1_Bar_Plots_clean.png", dpi=200, bbox_inches='tight')
            plt.show()