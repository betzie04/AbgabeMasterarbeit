import json
import os
import logging
from torch.nn.utils import spectral_norm
import geoopt.optim
import multiprocessing as mp
import argparse
from scipy.linalg import svdvals

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from transformers import set_seed
import sys


# Configure logger
logger = logging.getLogger(__name__)# Setup logging

log_level = logging.INFO
logger.setLevel(log_level)

class LipschitzControlledNonlinear(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, output_dim))
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_lipschitz_constant(self):
        lipschitz = 1.0
        for layer in self.net:
            if hasattr(layer, 'weight'):
                singular_values = torch.linalg.svdvals(layer.weight)
                lipschitz *= singular_values.max().item()
        return lipschitz

        
def worker(args_tuple):
    args, task_name = args_tuple

    # Neues Logging-Setup im Subprozess
    logger = logging.getLogger(f"{__name__}")
    logger.setLevel(logging.INFO)

    # Alte Handler löschen (wichtig!)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Eigener Logfile pro Prozess
    logfile = f"Logfiles/reproducable_logfile_extrinsic_{task_name}.log"
    file_handler = logging.FileHandler(logfile, mode='w')
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized in subprocess for {task_name}")

    # WICHTIGE KORREKTUR: args ist bereits das argparse.Namespace Objekt
    # Nicht args[0] oder andere Indexierung!
    main(args, task_name)

    
def estimate_lipschitz_constant(
    model, batch_size, g_test, DEVICE, n=20
):
    """
    Estimate the Lipschitz constant of function g over [a, b].
    
    Parameters:
    - g: a callable (e.g., neural network) mapping torch tensors to tensors
    - a, b: bounds of the input interval
    - delta: proximity threshold for pair sampling
    - n: number of slope samples per round
    - m: number of repetitions to get max slopes
    - seed: random seed for reproducibility
    
    Returns:
    - M_hat: estimated Lipschitz constant
    - l_values: list of m max slope values
    - params: fitted Reverse Weibull distribution parameters
    """
  
    grad = []
    for _ in range(n):
        model.eval()  # Set the model to evaluation mode
        random_tensor = torch.randn(batch_size, g_test.shape[-1], device=DEVICE)
        jacobian = torch.autograd.functional.jacobian(model, random_tensor)
        sing_val = torch.linalg.svdvals(jacobian)
        lipschitz = sing_val.max().item()

        grad.append(lipschitz)

    L_lower = max(grad)



    return L_lower   

def get_model_paths(root_path, model_name, task_name, seed):
    suffix = f"-seed-{seed}" if seed is not None else ""
    return {
        "train": os.path.join(root_path, f"models/{model_name}/task-{task_name}-train{suffix}"),
        "eval":  os.path.join(root_path, f"models/{model_name}/task-{task_name}-eval{suffix}"),
        "test":  os.path.join(root_path, f"models/{model_name}/task-{task_name}-test{suffix}"),
        "suffix": suffix,
        "display": seed if seed is not None else ""
    }

#random inputs in netz werfen 
def compute_lipschitz_constant(model):
    """
    Computes the Lipschitz constant of a neural network by considering the spectral norms
    (largest singular values) of the weight matrices.

    Args:
        model (nn.Module): The neural network model (e.g., NonLinearNetwork).

    Returns:
        float: The Lipschitz constant of the model.
    """
    lipschitz_constant = 1.0
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            # Compute the largest singular value (spectral norm) of the weight matrix
            weight = layer.weight
            singular_values = torch.linalg.svdvals(weight)  # Compute singular values
            largest_singular_value = singular_values.max()  # Get the largest singular value
            lipschitz_constant *= largest_singular_value.item()  # Multiply to get the overall Lipschitz constant
        elif isinstance(layer, nn.ReLU):
            # ReLU has a Lipschitz constant of 1
            lipschitz_constant *= 1
        elif isinstance(layer, nn.ELU):
            # ELU has a Lipschitz constant of 1 for positive inputs and exp(alpha) for negative inputs
            lipschitz_constant *= 1  # Assuming alpha=1 (default)
        elif isinstance(layer, nn.SiLU):
            # SiLU (Swish) has a Lipschitz constant <= 1
            lipschitz_constant *= 1
        elif isinstance(layer, nn.Tanh):
            # Tanh has a Lipschitz constant of 1
            lipschitz_constant *= 1
        elif isinstance(layer, nn.Sigmoid):
            # Sigmoid has a Lipschitz constant of 0.25 (maximal Steigung)
            lipschitz_constant *= 0.25
        elif isinstance(layer, nn.Dropout):
            # Dropout is ignored in Lipschitz constant calculation
            continue
  # GELU is approximately Lipschitz continuous with constant <= 1
        else:
            logger.warning(f"Layer {layer} is not a recognized activation function. Assuming Lipschitz constant of 1.")
            lipschitz_constant *= 1
    return lipschitz_constant



   


def load_model(model_path, seed, layer, device):
    if seed is not None:
        model_path = model_path.format(seed=seed)
    model = torch.load(model_path, map_location=device).select(0, layer)
    return model, model.shape[-1]


def train_transformation_model(g_train, g_eval, h_train, h_eval, g_train_size, h_train_size, device, num_epochs, lr, batch_size, hidden_size, patience, output_size):
    """
    Trains the transformation model and applies early stopping based on validation loss.
    Returns the model and loss statistics.
    """ 
    dataset_g = TensorDataset(torch.Tensor(g_train))
    dataloader_g = DataLoader(dataset_g, batch_size=batch_size, shuffle=True)
    dataset_h = TensorDataset(torch.Tensor(h_train))
    dataloader_h = DataLoader(dataset_h, batch_size=batch_size, shuffle=True)
    assert len(dataloader_g) == len(dataloader_h), "Dataloader lengths do not match!"
    
    g = LipschitzControlledNonlinear(g_train_size, hidden_size, output_size)
    h = LipschitzControlledNonlinear(h_train_size, hidden_size, output_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    g.apply(init_weights)
    h.apply(init_weights)
    g.to(device)
    h.to(device)
    
    criterion = nn.MSELoss(reduction='none')
    opt_g = torch.optim.Adam(g.parameters(), lr=lr)
    opt_h = torch.optim.Adam(h.parameters(), lr=lr)
                    
    best_max_dist = -float("inf")
    best_min_dist = float("inf")
    epochs_no_improve = 0
    patience = 10
    convergence_threshold = 1e-6
    
    # KORREKTUR: Korrekte Initialisierung der Metriken
    max_train_loss_list = []
    val_loss_list = []  # Diese Liste war nicht korrekt initialisiert
    
    train_metrics = {
        'batch_losses_g': [],
        'batch_losses_h': [],
        'batch_distances': []
    }
    
    # WICHTIG: eval_metrics nur EINMAL außerhalb der Schleife initialisieren
    eval_metrics = {
        'epoch_max_loss_g': [],
        'epoch_min_loss_g': [],
        'epoch_avg_loss_g': [],
        'epoch_avg_loss_h': [],
        'epoch_nash_gaps': []
    }
    
    for epoch in range(num_epochs):
        g.train()
        h.train()
        epoch_loss = 0.0
        epoch_max_loss = 0.0
        epoch_min_loss = 0.0
        # Alternating training
        for train_index, (batch_g, batch_h) in enumerate(zip(dataloader_g, dataloader_h)):
            batch_g_input = batch_g[0].to(device)
            batch_h_input = batch_h[0].to(device)
        
            if train_index % 2 == 0:  # train g
                opt_g.zero_grad()
                
                output = g(batch_g_input)
                target = h(batch_h_input).detach()
                loss_g = criterion(output, target).mean()
                
                (-loss_g).backward()
                torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=1.0)
                opt_g.step()
                epoch_loss += loss_g.item()

                train_metrics['batch_losses_g'].append(loss_g.item())
                batch_dist = torch.max(torch.abs(output - target)).item()
                train_metrics['batch_distances'].append(batch_dist)
                epoch_min_loss = max(-loss_g, epoch_min_loss)

            else:  # train h
                opt_h.zero_grad()
                
                output = h(batch_h_input)
                target = g(batch_g_input).detach()
                loss_h = criterion(output, target).mean()
                
                loss_h.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)
                opt_h.step()
                epoch_loss += loss_h.item()
                
                train_metrics['batch_losses_h'].append(loss_h.item())
                batch_dist = torch.max(torch.abs(output - target)).item()
                train_metrics['batch_distances'].append(batch_dist)
                epoch_max_loss = max(loss_h, epoch_max_loss)

        max_train_loss_list.append(epoch_loss)
        avg_loss = epoch_loss / len(dataloader_g)
        

        # Evaluation
        eval_dataset_g = TensorDataset(torch.Tensor(g_eval))
        eval_dataloader_g = DataLoader(eval_dataset_g, batch_size=batch_size, shuffle=False)
        eval_dataset_h = TensorDataset(torch.Tensor(h_eval))
        eval_dataloader_h = DataLoader(eval_dataset_h, batch_size=batch_size, shuffle=False)
        
        g.eval()
        h.eval()
        eval_losses_g = []
        eval_losses_h = []
        # KORREKTUR: Entferne diese Zeile die eval_metrics überschreibt!
        # eval_metrics = {}  <- DIESE ZEILE LÖSCHEN!

        with torch.no_grad():
            for eval_batch_g, eval_batch_h in zip(eval_dataloader_g, eval_dataloader_h):
                eval_g_input = eval_batch_g[0].to(device)
                eval_h_input = eval_batch_h[0].to(device)
                
                psi_g = g(eval_g_input)
                phi_h = h(eval_h_input)
                loss_g_batch = criterion(psi_g, phi_h).mean().item()
                eval_losses_g.append(loss_g_batch)
                
                loss_h_batch = criterion(phi_h, psi_g).mean().item()
                eval_losses_h.append(loss_h_batch)
                
        current_max_loss_g = max(eval_losses_g)
        current_min_loss_g = min(eval_losses_h)
        current_avg_loss_g = sum(eval_losses_g) / len(eval_losses_g)
        current_avg_loss_h = sum(eval_losses_h) / len(eval_losses_h)

        logger.info(f"Eval - Max loss_g: {current_max_loss_g:.6f}")
        logger.info(f"Eval - Avg loss_g: {current_avg_loss_g:.6f}, Avg loss_h: {current_avg_loss_h:.6f}")
      
        # KORREKTUR: Jetzt werden die Werte korrekt zu eval_metrics hinzugefügt
        eval_metrics['epoch_max_loss_g'].append(current_max_loss_g)
        eval_metrics['epoch_min_loss_g'].append(current_min_loss_g)
        eval_metrics['epoch_avg_loss_g'].append(current_avg_loss_g)
        eval_metrics['epoch_avg_loss_h'].append(current_avg_loss_h)
        eval_metrics['epoch_nash_gaps'].append(current_max_loss_g - current_min_loss_g)

        # KORREKTUR: val_loss_list korrekt befüllen
        val_loss_list.append(current_avg_loss_g)

        eval_distances = []
        with torch.no_grad():
            for eval_g_batch, eval_h_batch in zip(eval_dataloader_g, eval_dataloader_h):
                eval_g_input = eval_g_batch[0].to(device)
                eval_h_input = eval_h_batch[0].to(device)
                psi_g_eval = g(eval_g_input)
                phi_h_eval = h(eval_h_input)
                eval_dist = torch.max(torch.abs(psi_g_eval - phi_h_eval)).item()
                eval_distances.append(eval_dist)
        
        avg_eval_distance = sum(eval_distances) / len(eval_distances)
        logger.info(f"Eval - Avg L-inf distance ||ψ(g) - φ(h)||_∞: {avg_eval_distance:.6f}")
        
        # Early stopping check
        improved = False
        
        if current_max_loss_g > best_max_dist + convergence_threshold:
            best_max_dist = current_max_loss_g
            improved = True

        if current_min_loss_g < best_min_dist - convergence_threshold:
            best_min_dist = current_min_loss_g
            improved = True
        
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            nash_gap = abs(best_max_dist - best_min_dist)
            logger.info(f"Early stopping at epoch {epoch+1}, Nash gap: {nash_gap:.6f}")
            break

    # KORREKTUR: Sicherstellen, dass val_loss_list nicht leer ist
    val_loss_max = max(val_loss_list) if val_loss_list else 0.0
    epoch_loss_tensor = torch.tensor(epoch_loss, dtype=torch.float32)
    epoch_max_loss_tensor = torch.tensor(epoch_max_loss, dtype=torch.float32)
    epoch_min_loss_tensor = torch.tensor(epoch_min_loss, dtype=torch.float32)

    return g, h, epoch_loss_tensor.cpu().detach().numpy(), epoch_max_loss_tensor.cpu().detach().numpy(), epoch_min_loss_tensor.cpu().detach().numpy(), {
        #"train_loss": max_train_loss_list,
        #"val_loss": eval_metrics,
        "train_loss_max": max(max_train_loss_list) if max_train_loss_list else 0.0,
        "val_loss_max": val_loss_max,
        "final_epoch": epoch + 1,
        "train_metrics": train_metrics
    }




# LIPSCHITZ ANALYSIS FUNCTIONS
def compute_empirical_lipschitz_bounds(model, test_data, device, n_samples=20, batch_size=8):
    """
    Berechnet empirische obere und untere Lipschitz-Schranken sowie theoretische obere Schranke
    """
    model.eval()
    grad = []
    
    input_dim = test_data.shape[-1] if hasattr(test_data, 'shape') else None
    if input_dim is None:
        for batch in test_data:
            input_dim = batch[0].shape[-1]
            break
    
    logger.info(f"Computing Lipschitz bounds (n_samples={n_samples})...")
    
    # 1. UNTERE SCHRANKE: Maximale Spektralnorm der Jacobian-Matrix
    for i in range(n_samples):
        try:
            random_tensor = torch.randn(batch_size, input_dim, device=device, requires_grad=True)
            
            def model_func(x):
                return model(x)
            
            jacobian = torch.autograd.functional.jacobian(model_func, random_tensor)
            
            for batch_idx in range(batch_size):
                jacobian_sample = jacobian[batch_idx, :, batch_idx, :]
                sing_val = torch.linalg.svdvals(jacobian_sample)
                lipschitz = sing_val.max().item()
                grad.append(lipschitz)
                
        except Exception as e:
            if i < 5:
                logger.warning(f"Jacobian error: {e}")
            continue
    
    L_lower = max(grad) if grad else 0.0
    
    # 2. THEORETISCHE OBERE SCHRANKE: Produkt der Layer-Lipschitz-Konstanten
    theoretical_upper_bound = 1.0
    layer_lipschitz_constants = []
    
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            # Für spektral normalisierte Layer
            if hasattr(layer, 'weight_u'):
                # Spektral-Norm garantiert σ_max ≤ 1
                layer_lipschitz = 1.0
            else:
                # Für normale Layer: größter Singulärwert
                singular_values = torch.linalg.svdvals(layer.weight)
                layer_lipschitz = singular_values.max().item()
            
            layer_lipschitz_constants.append(layer_lipschitz)
            theoretical_upper_bound *= layer_lipschitz
        elif isinstance(layer, nn.ReLU):
            # ReLU hat Lipschitz-Konstante 1
            layer_lipschitz_constants.append(1.0)
            # theoretical_upper_bound *= 1.0  # Bleibt unverändert
    
    # 3. EMPIRISCHE OBERE SCHRANKE: Finite Differences Methode
    max_lipschitz_ratio = 0.0
    with torch.no_grad():
        sample_count = 0
        for batch in test_data:
            if sample_count >= n_samples:
                break
                
            x = batch[0].to(device)
            current_batch_size = x.shape[0]
            
            for i in range(min(current_batch_size, n_samples - sample_count)):
                x_i = x[i:i+1]
                epsilon = 1e-4
                noise = epsilon * torch.randn_like(x_i)
                x_perturbed = x_i + noise
                
                y = model(x_i)
                y_perturbed = model(x_perturbed)
                
                output_diff = torch.norm(y - y_perturbed, dim=1)
                input_diff = torch.norm(noise, dim=1)
                
                lipschitz_ratio = (output_diff / (input_diff + 1e-10)).max().item()
                max_lipschitz_ratio = max(max_lipschitz_ratio, lipschitz_ratio)
                
                sample_count += 1
                if sample_count >= n_samples:
                    break
    
    L_upper_empirical = max_lipschitz_ratio
    L_upper_theoretical = theoretical_upper_bound
    
    return {
        'empirical_lower_bound': L_lower,
        'empirical_upper_bound': L_upper_empirical,
        'theoretical_upper_bound': L_upper_theoretical,  # NEU!
        'layer_lipschitz_constants': layer_lipschitz_constants,  # NEU!
        'jacobian_samples': grad,
        'bound_gap_empirical': L_upper_empirical - L_lower,
        'bound_gap_theoretical': L_upper_theoretical - L_lower,  # NEU!
        'n_samples_used': len(grad)
    }
def enhanced_log_lipschitz_bounds(model_g, model_h, test_dataloader_g, test_dataloader_h, lr, device, prefix=""):
    theoretical_lipschitz_g = model_g.get_lipschitz_constant()
    theoretical_lipschitz_h = model_h.get_lipschitz_constant()
    
    logger.info(f"Theoretical: G={theoretical_lipschitz_g:.4f}, H={theoretical_lipschitz_h:.4f}")
    
    empirical_bounds_g = compute_empirical_lipschitz_bounds(model_g, test_dataloader_g, device, n_samples=10, batch_size=4)
    empirical_bounds_h = compute_empirical_lipschitz_bounds(model_h, test_dataloader_h, device, n_samples=10, batch_size=4)
    
    # ERWEITERTE SUMMARY LOGGING
    logger.info(f"Lipschitz Bounds Summary:")
    logger.info(f"  Model G:")
    logger.info(f"    Lower Bound (Jacobian):     {empirical_bounds_g['empirical_lower_bound']:.4f}")
    logger.info(f"    Upper Bound (Empirical):    {empirical_bounds_g['empirical_upper_bound']:.4f}")
    logger.info(f"    Upper Bound (Theoretical):  {empirical_bounds_g['theoretical_upper_bound']:.4f}")
    logger.info(f"    Layer Products:             {empirical_bounds_g['layer_lipschitz_constants']}")
    
    logger.info(f"  Model H:")
    logger.info(f"    Lower Bound (Jacobian):     {empirical_bounds_h['empirical_lower_bound']:.4f}")
    logger.info(f"    Upper Bound (Empirical):    {empirical_bounds_h['empirical_upper_bound']:.4f}")
    logger.info(f"    Upper Bound (Theoretical):  {empirical_bounds_h['theoretical_upper_bound']:.4f}")
    logger.info(f"    Layer Products:             {empirical_bounds_h['layer_lipschitz_constants']}")
    
    # Vergleich der oberen Schranken
    logger.info(f"  Upper Bound Comparison:")
    logger.info(f"    G: Theoretical vs Empirical = {empirical_bounds_g['theoretical_upper_bound']:.4f} vs {empirical_bounds_g['empirical_upper_bound']:.4f}")
    logger.info(f"    H: Theoretical vs Empirical = {empirical_bounds_h['theoretical_upper_bound']:.4f} vs {empirical_bounds_h['empirical_upper_bound']:.4f}")
    
    # Welche obere Schranke ist tighter?
    tighter_bound_g = min(empirical_bounds_g['theoretical_upper_bound'], empirical_bounds_g['empirical_upper_bound'])
    tighter_bound_h = min(empirical_bounds_h['theoretical_upper_bound'], empirical_bounds_h['empirical_upper_bound'])
    
    logger.info(f"  Tightest Upper Bounds: G={tighter_bound_g:.4f}, H={tighter_bound_h:.4f}")
    
    return {
        'theoretical_lipschitz_g': theoretical_lipschitz_g,
        'theoretical_lipschitz_h': theoretical_lipschitz_h,
        'empirical_bounds_g': empirical_bounds_g,
        'empirical_bounds_h': empirical_bounds_h,
        'layer_info_g': [],
        'layer_info_h': [],
        'bounds_summary': {
            'g_lower': empirical_bounds_g['empirical_lower_bound'],
            'g_upper_empirical': empirical_bounds_g['empirical_upper_bound'],
            'g_upper_theoretical': empirical_bounds_g['theoretical_upper_bound'],  # NEU!
            'g_upper_tightest': tighter_bound_g,  # NEU!
            'g_theoretical': theoretical_lipschitz_g,
            'h_lower': empirical_bounds_h['empirical_lower_bound'],
            'h_upper_empirical': empirical_bounds_h['empirical_upper_bound'],
            'h_upper_theoretical': empirical_bounds_h['theoretical_upper_bound'],  # NEU!
            'h_upper_tightest': tighter_bound_h,  # NEU!
            'h_theoretical': theoretical_lipschitz_h
        }
    }

def extract_weights_from_model_dict(model_dict):
    # Finde alle Gewichtsmatrizen der Linear-Layer (z.B. 'network.0.weight', 'network.2.weight', ...)
    linear_weight_keys = [k for k in model_dict.keys() if "weight" in k and model_dict[k].ndim == 2]
    # Sortiere nach Layer-Index (z.B. 'network.0.weight' < 'network.2.weight')
    linear_weight_keys = sorted(linear_weight_keys, key=lambda x: int([s for s in x.split('.') if s.isdigit()][0]))
    # Extrahiere und konvertiere zu numpy arrays
    weights = [model_dict[k].cpu().numpy() for k in linear_weight_keys]
    return weights


def pearson_corr(all_g: torch.Tensor, all_h: torch.Tensor) -> float:
    """
    Berechnet den Pearson-Korrelationskoeffizienten zwischen zwei 2D-Tensoren x und y.
    Erwartet: [n_samples, n_features]
    """
    # Deine Inputs: (8, 756) und (8, 256)
    all_g = torch.cat(all_g, dim=0)  # shape (8, 756)
    all_h = torch.cat(all_h, dim=0)  # shape (8, 256)

    # Jetzt: Korrelation zwischen allen Spalten von all_g mit allen Spalten von all_h
    # Ergebnis: Matrix der Größe (756, 256)

    # Mittelwerte und Standardabweichungen
    g_mean = all_g.mean(dim=0, keepdim=True)  # shape (1, 756)
    h_mean = all_h.mean(dim=0, keepdim=True)  # shape (1, 256)
    g_std = all_g.std(dim=0, unbiased=False, keepdim=True)  # shape (1, 756)
    h_std = all_h.std(dim=0, unbiased=False, keepdim=True)  # shape (1, 256)

    # Standardisieren
    g_z = (all_g - g_mean) / g_std  # (8, 756)
    h_z = (all_h - h_mean) / h_std  # (8, 256)

    # Pearson correlation: (g_z^T @ h_z) / n
    n = all_g.shape[0]  # = 8
    corr = g_z.T @ h_z / n  # shape (756, 256)

    return corr.mean().item()  # Durchschnittliche Korrelation über alle Features

def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Computes Spearman rank correlation between two tensors on GPU.
    Input: x, y of shape (n_samples, d)
    Output: average Spearman correlation across d dimensions
    """
    # Flatten the first dimension if necessary (e.g., batch size of 1)
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))

    x_rank = x.argsort(dim=0).argsort(dim=0).float()
    y_rank = y.argsort(dim=0).argsort(dim=0).float()

    x_mean = x_rank.mean(dim=0, keepdim=True)
    y_mean = y_rank.mean(dim=0, keepdim=True)

    cov = ((x_rank - x_mean) * (y_rank - y_mean)).mean(dim=0)
    std_x = x_rank.std(dim=0)
    std_y = y_rank.std(dim=0)

    # Handle cases where std_x or std_y is zero to avoid NaN
    valid = (std_x > 1e-8) & (std_y > 1e-8)
    corr = torch.zeros_like(cov)
    corr[valid] = cov[valid] / (std_x[valid] * std_y[valid])

    # Return the mean correlation across all valid dimensions
    return corr.mean().item()


def main(args, task_name):

    ROOT_PATH = args.root_path
    DEVICE = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    layer_num = args.layer
    learning_rates = args.learning_rates
    patience = args.patience

    logger.info(f"running on {DEVICE}")
    logger.info(f"layer: {layer_num}")
    logger.info(f"batch size: {args.batch_size}")
    logger.info(f"num epochs: {num_epochs}")
    logger.info(f"learning rates: {learning_rates}")
    output_size = 3 if task_name == "mnli" else 2

    for model_g in args.model_type:
        os.makedirs(f"{ROOT_PATH}intrinsic_models/", exist_ok=True)
        if model_g == "multiberts":
            seed_range = range(args.num_seeds)
            train_on_models = args.model_type
        elif model_g == "roberta":
            seed_range = [None]
            train_on_models = [model for model in args.model_type if model != model_g]
        elif model_g == "electra":
            seed_range = [None]
            train_on_models = [model for model in args.model_type if model != model_g]
        else:
            raise ValueError(f"Unsupported model type: {model_g}")

        seed_range = range(args.num_seeds) if model_g == "multiberts" else [None]
        losses_seed = []
        for seed_g in seed_range:
            g_paths = get_model_paths(ROOT_PATH, model_g, task_name, seed_g)
            logger.info(f"Starting processing for model: {model_g}, seed: {g_paths['display']}, task: {task_name}")
            g_train, g_train_size = load_model(g_paths["train"], seed_g, layer_num, DEVICE)
            g_eval, g_eval_size = load_model(g_paths["eval"], seed_g, layer_num, DEVICE)
            g_test, g_test_size = load_model(g_paths["test"], seed_g, layer_num, DEVICE)

            seed_display = g_paths["display"]
            logger.info(f"Starting processing for model: {model_g}, seed: {seed_display}, task: {task_name}")


            #||h-\psi(g)||
            for model_h in [model for model in args.model_type]:
                logger.info(f"Using  Network for {model_g} on labels of model {model_h}")                        
                logger.info(f"Training transformation for model g: {model_g} on labels of model h: {model_h} on task {task_name}")
                if model_h == "multiberts" and model_g != "multiberts":
                    seed_range_h = [i for i in range(args.num_seeds) if i != seed_g]
                elif model_h == "multiberts":
                    seed_range_h = range(args.num_seeds)
                else:
                        seed_range_h = [None]

                
                for seed_h in seed_range_h:
                    h_paths = get_model_paths(ROOT_PATH, model_h, task_name, seed_h)
                    logger.info(f"Training  transformation: {model_g}({g_paths['display']}) → {model_h}({h_paths['display']})")
  
                    h_train, h_train_size = load_model(h_paths["train"], seed_h, layer_num, DEVICE)
                    h_eval, h_eval_size = load_model(h_paths["eval"], seed_h, layer_num, DEVICE)

                    h_test, h_test_size = load_model(h_paths["test"], seed_h, layer_num, DEVICE)

                    save_dir = os.path.join(
                            ROOT_PATH,
                            f"models_{args.activation}/{model_g}{g_paths['suffix']}_vs_{model_h}{h_paths['suffix']}"
                        )
                    os.makedirs(save_dir, exist_ok=True)
                    criterion = nn.MSELoss(reduction='none')
                    accuracy = []
                    for lr in learning_rates:

                        g, h, epoch_loss_tensor, epoch_max_loss_tensor, epoch_min_loss_tensor, loss_dict = train_transformation_model(g_train, g_eval, h_train, h_eval, g_train_size, h_train_size, DEVICE, num_epochs, lr, batch_size, args.hidden_size, patience, output_size)
                # Test evaluation
                        test_dataset_g = TensorDataset(torch.Tensor(g_test))
                        test_dataloader_g = DataLoader(test_dataset_g, batch_size=batch_size, shuffle=True)
                        test_dataset_h = TensorDataset(torch.Tensor(h_test))
                        test_dataloader_h = DataLoader(test_dataset_h, batch_size=batch_size, shuffle=True)
                        g.eval()
                        h.eval()
                        all_dist = []
                        all_preds_g = []
                        all_preds_h = []
                        all_g = []
                        all_h = []
                            
                        test_losses_g = []
                        test_losses_h = []
                        test_batch_metrics = {
                            'batch_losses_g': [],
                            'batch_losses_h': [],
                            'batch_linf_norms': [],
                            'batch_indices': []
                        }
                        all_accuracy = []
                        with torch.no_grad():
                            for train_index, (batch_g, batch_h) in enumerate(zip(test_dataloader_g, test_dataloader_h)):
                                test_g_input = torch.Tensor(batch_g[0]).to(DEVICE)
                                test_h_input = torch.Tensor(batch_h[0]).to(DEVICE)
                                psi_g = g(test_g_input)
                                phi_h = h(test_h_input)
                                
                                test_loss_g_batch = criterion(psi_g, phi_h).mean().item()
                                test_loss_h_batch = criterion(phi_h, psi_g).mean().item()
                                test_losses_g.append(test_loss_g_batch)
                                test_losses_h.append(test_loss_h_batch)
                                
                                linf_norm = torch.max(torch.abs(psi_g - phi_h)).item()
                                logger.info(f"L-infinity-Norm (test): {linf_norm}")
                                
                                test_batch_metrics['batch_losses_g'].append(test_loss_g_batch)
                                test_batch_metrics['batch_losses_h'].append(test_loss_h_batch)
                                test_batch_metrics['batch_linf_norms'].append(linf_norm)
                                test_batch_metrics['batch_indices'].append(train_index)
                                
                                all_g.append(test_g_input)
                                all_h.append(test_h_input)
                                all_preds_g.append(psi_g)
                                all_preds_h.append(phi_h)
                                all_dist.append(linf_norm)
                                test_prediction_accuracy = (
                                    (psi_g.argmax(dim=1) == phi_h.argmax(dim=1))
                                    .float()
                                    .mean()
                                    .cpu()
                                    .numpy()
                                )

                                all_accuracy.append(test_prediction_accuracy)
                                        
                            overall_max = max(all_dist)
                            test_max_loss_g = max(test_losses_g)
                            test_min_loss_h = min(test_losses_h)
                            test_avg_loss_g = sum(test_losses_g) / len(test_losses_g)
                            test_avg_loss_h = sum(test_losses_h) / len(test_losses_h)

                        logger.info(f"Overall  Distance on Test Data ||h-\psi(g)||_inf for lr={lr}: {overall_max}")
                        accuracy.append((epoch_loss_tensor, epoch_max_loss_tensor, epoch_min_loss_tensor, all_accuracy))
                        preds_g_combined = torch.cat(all_preds_g, dim=0) 
                        preds_h_combined = torch.cat(all_preds_h, dim=0)
                        spearman_val = spearman_corr_torch(preds_g_combined, preds_h_combined)
                        logger.info(f"Spearman correlation on Test Data  : {spearman_val:.4f}")
                        
                        rho = pearson_corr(all_g, all_h)
                        logger.info(f"Pearson correlation on Test Data  : {rho:4f}")

                        enhanced_lipschitz_results = enhanced_log_lipschitz_bounds(
                            g, h, test_dataloader_g, test_dataloader_h, lr, DEVICE, prefix="FINAL "
                        )
                        metrics_entry = {
                        "task": task_name,
                        "model_g": model_g,
                        "model_h": model_h,
                        "seed_g": seed_g,
                        "seed_h": seed_h,
                        "lr": lr,
                        "all_distances": all_dist,
                        "distance_inf_overall": overall_max,
                        "distance_mean": float(np.mean(all_dist)),
                        "distance_std": float(np.std(all_dist)),
                        "spearman": spearman_val,
                        "pearson": rho,
                        **loss_dict,
                        **enhanced_lipschitz_results
                        }


                        # Create directory if it doesn't exist
                        path_to_save_h = os.path.join(save_dir, f"{task_name}_{model_h}{h_paths['display']}_lr{lr}.pt")
                        path_to_save_g = os.path.join(save_dir, f"{task_name}_{model_g}{g_paths['display']}_lr{lr}.pt")

                        #os.makedirs(os.path.dirname(path_to_save_h), exist_ok=True)
                        torch.save(g, path_to_save_g)
                        logger.info(f"Model {model_g}saved to {path_to_save_g}")
                        torch.save(h, path_to_save_h)
                        logger.info(f"Model {model_h}saved to {path_to_save_h}")
                        # JSON-Datei pro Task speichern 
                        json_filename = f"data/results_max_loss/extrinsic/reproducable_extrinsic_{task_name}_final_training.json"
                        with open(json_filename, "a", encoding="utf-8") as f:
                            f.write(json.dumps(metrics_entry) + "\n")
                    losses_seed.append(accuracy)
            with open(
        os.path.join(
            ROOT_PATH,
            f"data/results_max_loss/extrinsic/reproducable_v2-extrinsic-maxloss-accuracies-{task_name}-small-{seed_g}.npy",
                ),
                "wb",
            ) as f:
                np.save(f, np.array(losses_seed, dtype=object))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["cola", "mrpc", "qnli", "qqp", "sst2","rte"], nargs="+", help="Name(s) of the task(s) (e.g., sst2, mnli).")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs for training.")
    parser.add_argument("--layer", type=int, default=12, help="Layer of the model to extract embeddings from.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the model.")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.00001, 0.0001, 0.001, 0.01, 0.1], help="List of learning rates to use for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default=["electra", "roberta", "multiberts"],  help="Type of model to use (bert, roberta, electra).")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Directory to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--activation", type=str, default="extrinsic_3Relu_reproducable", help="Name of the activation function used in the model (e.g., _3_ReLu, _4_Sigmoid).")
    parser.add_argument("--patience", type=int, default=20, help="Number of epochs with no improvement after which training will be stopped.")

    args = parser.parse_args()
    tasks = args.task_name
    mp.set_start_method("spawn", force=True)  # <- wichtig für CUDA-Kompatibilität

    # Erstelle Liste aller Kombinationen
    job_args = [(args, task_name) for task_name in tasks]
    # Nur EINMAL initialisieren
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("Logfiles/log_main_extrinsic_reproducable.log", mode='w')
        ],
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logger initialized in main process")

    logger.info(f"Running with multiprocessing using {mp.cpu_count()} CPUs")

    # Optional: Begrenze Anzahl der Prozesse mit processes=n
    with mp.Pool(processes=min(len(job_args), mp.cpu_count())) as pool:
        pool.map(worker, job_args)

    logger.info("All tasks completed successfully!")