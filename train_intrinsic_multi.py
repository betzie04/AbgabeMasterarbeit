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

# Define the model
class LinearMapOld(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearMap, self).__init__()
        self.network = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.network(x)

    def __iter__(self):
        """
        Allows the model to be iterable by yielding its layers.
        """
        yield self.network

# Define the model
class LinearMap(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearMap, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.linear(x)
        


class NonLinearNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=256):
        """
        Initializes the NonLinearNetwork class.

        Args:
            input_size (int): The size of the input layer.
            output_size (int): The size of the output layer.
        """
        super(NonLinearNetwork, self).__init__()
        # Sequential model with a linear layer followed by a ReLU activation. Kombination allows a nonlinear transfomation of the input data.
        # Can be modified to include more layers or different activation functions.
        # example: nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(
            spectral_norm(nn.Linear(input_size, hidden_size), n_power_iterations=20),
            nn.ReLU(),  # Erste Schicht mit ReLU
            spectral_norm(nn.Linear(hidden_size, hidden_size), n_power_iterations=20),
            nn.ELU(),   # Zweite Schicht mit ELU
            spectral_norm(nn.Linear(hidden_size, hidden_size), n_power_iterations=20),
            nn.SiLU(),  # Dritte Schicht mit Swish (SiLU)
            spectral_norm(nn.Linear(hidden_size, hidden_size), n_power_iterations=20),
            nn.Sigmoid(),  # Dritte Schicht mit Swish (SiLU)
            spectral_norm(nn.Linear(hidden_size, output_size), n_power_iterations=20)
        )
        #self.network = nn.Sequential(
        #    nn.Linear(input_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(hidden_size, output_size)
        #)

    def forward(self, x):
        return self.network(x)


def worker(args_tuple):
    args, task_name, lin_type = args_tuple

    # Neues Logging-Setup im Subprozess
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Alte Handler löschen (wichtig!)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Eigener Logfile pro Prozess
    logfile = f"Logfiles/reproducable_logfile_{task_name}_{lin_type}_3ReLU_final_training.log"
    file_handler = logging.FileHandler(logfile, mode='w')
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized in subprocess for {task_name} ({lin_type})")

    main(args, task_name, lin_type)
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



def kernel_cka_dist(A, B, sigma=None):
    """
    Computes Kernel CKA distance between representations A and B using an RBF kernel.
    :param A: torch.Tensor of size n x d (n samples, d features)
    :param B: torch.Tensor of size n x d (n samples, d features)
    :param sigma: bandwidth parameter for the RBF kernel. If None, it is set to the median distance heuristic.
    :return: Kernel CKA distance
    """
    device = A.device  # Ensure computations are done on the same device as the input tensors
    print(A.shape, B.shape)
    pairwise_sq_dists_A = torch.sum(A**2, dim=1).view(-1, 1) + torch.sum(A**2, dim=1) - 2 * torch.mm(A, A.T)
    pairwise_sq_dists_B = torch.sum(B**2, dim=1).view(-1, 1) + torch.sum(B**2, dim=1) - 2 * torch.mm(B, B.T)

    if sigma is None:
        sigma_A = torch.sqrt(torch.median(pairwise_sq_dists_A[pairwise_sq_dists_A > 0]) / 2)
        sigma_B = torch.sqrt(torch.median(pairwise_sq_dists_B[pairwise_sq_dists_B > 0]) / 2)
    else:
        sigma_A = sigma_B = sigma

    K_A = torch.exp(-pairwise_sq_dists_A / (2 * sigma_A**2))
    K_B = torch.exp(-pairwise_sq_dists_B / (2 * sigma_B**2))

    # Center the kernel matrices
    n = K_A.size(0)
    H = torch.eye(n, device=device) - (1 / n) * torch.ones((n, n), device=device)
    K_A_centered = H @ K_A @ H
    K_B_centered = H @ K_B @ H

    # Compute the similarity and normalization terms
    similarity = torch.trace(K_A_centered @ K_B_centered)
    normalization = torch.sqrt(torch.trace(K_A_centered @ K_A_centered) * torch.trace(K_B_centered @ K_B_centered))

    return 1 - similarity / normalization
    


def load_model(model_path, seed, layer_num, device):
    """
    Load a model from the specified path, seed, and layer.

    Args:
        model_path (str): Path to the model.
        seed (int or None): Seed value for the model.
        layer (int): Layer to extract embeddings from.
        device (str): Device to load the model on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Loaded model embeddings for the specified layer.
    """
    if seed is not None:
        model_path = model_path.format(seed=seed)
    model = torch.load(model_path, map_location=device).select(0,layer_num)
    layers = model[:layer_num+1]
    return model

def train_transformation_model(g_train, g_eval, h_train, h_eval, DEVICE, num_epochs, lr, batch_size, patience, lin_type):
    """
    Trains the transformation model and applies early stopping based on validation loss.
    Returns the model and loss statistics.
    """ 
    if lin_type == "linear":
        model = LinearMap(g_train.shape[-1], h_train.shape[-1])
    elif lin_type == "nonlinear":
        model = NonLinearNetwork(g_train.shape[-1], h_train.shape[-1])

    model.to(DEVICE)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    optimizer = geoopt.optim.RiemannianSGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    #nn.HuberLoss(delta=1.0, reduction='none')

    dataset = TensorDataset(torch.Tensor(g_train), torch.Tensor(h_train))
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    eval_dataset = TensorDataset(torch.Tensor(g_eval), torch.Tensor(h_eval))
    eval_dataloader = DataLoader(eval_dataset, batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0  # Initialize epoch loss
        epoch_max_loss = 0.0
        for batch_g, batch_h in dataloader:
            optimizer.zero_grad()
            out_x = model(batch_g)
            loss_vec = criterion(out_x, batch_h)
            loss = loss_vec.mean()
            max_loss = loss_vec.mean(axis=1).max()
            max_loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_g)  # Aggregate loss over the epoch
            epoch_max_loss = max(max_loss, epoch_max_loss)

        epoch_loss /= len(dataset)
        train_loss_list.append(epoch_max_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_losses = []
        with torch.no_grad():
            for batch_g, batch_h in eval_dataloader:
                batch_g, batch_h = batch_g.to(DEVICE), batch_h.to(DEVICE)
                val_pred = model(batch_g)
                loss = nn.MSELoss()(val_pred, batch_h)
                val_losses.append(loss.item())
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(eval_dataloader)
        val_loss_list.append(epoch_val_loss)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_max_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss - 1e-4:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    epoch_loss_tensor = torch.tensor(epoch_loss, dtype=torch.float32)
    epoch_max_loss_tensor = torch.tensor(epoch_max_loss, dtype=torch.float32)

    return model, epoch_loss_tensor.cpu().detach().numpy(), epoch_max_loss_tensor.cpu().detach().numpy(), {
        "train_loss": [float(x) for x in train_loss_list],
        "val_loss": [float(x) for x in val_loss_list],
        "train_loss_max": float(max(train_loss_list)),
        "val_loss_max": float(max(val_loss_list)),
    }



def test_lipschitz(model, hidden_dim, batch_size=32, num_tests=100, input_scale=1.0, DEVICE="cuda"):
    """
    Test if the model is Lipschitz continuous with constant < 1
    
    Args:
        model: Your PyTorch model
        hidden_dim: Dimension of hidden states (last dimension from hidden_states_input_eval)
        batch_size: Batch size for testing
        num_tests: Number of random tests to perform
        input_scale: Scale factor for random inputs
    """
    model.to(DEVICE)
    model.eval()
    
    lipschitz_constants = []
    
    with torch.no_grad():  # Disable gradient calculation for testing
        for _ in range(num_tests):
            # Create two random input tensors exactly as you specified
            x1 = input_scale * torch.randn(batch_size, hidden_dim, device=DEVICE)
            x2 = input_scale * torch.randn(batch_size, hidden_dim, device=DEVICE)
            
            # Get model outputs
            y1 = model(x1)
            y2 = model(x2)
            
            # Calculate distances (using L2 norm)
            input_dist = torch.norm(x1 - x2, p=2)
            output_dist = torch.norm(y1 - y2, p=2)
            
            # Avoid division by zero
            if input_dist > 1e-6:
                lipschitz_ratio = output_dist / input_dist
                lipschitz_constants.append(lipschitz_ratio.item())


    return lipschitz_constants

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


def main(args, task_name, lin_type):

    ROOT_PATH = args.root_path
    DEVICE = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    layer_num = args.layer
    learning_rates = args.learning_rates

    logger.info(f"running on {DEVICE}")
    logger.info(f"layer: {layer_num}")
    logger.info(f"batch size: {args.batch_size}")
    logger.info(f"num epochs: {num_epochs}")
    logger.info(f"learning rates: {learning_rates}")

    if lin_type == "nonlinear":
        ROOT_PATH = "data/nonlin/"
               
    elif lin_type == "linear":
        ROOT_PATH = "data/lin/"
    for model_g in args.model_type:
        os.makedirs(f"{ROOT_PATH}intrinsic_models/", exist_ok=True)

        seed_range = range(args.num_seeds) if model_g == "multiberts" else [None]
        for seed_g in seed_range:
            losses_seed = []
            g_paths = get_model_paths(ROOT_PATH, model_g, task_name, seed_g)
            logger.info(f"Starting processing for model: {model_g}, seed: {g_paths['display']}, task: {task_name}")
            g_train = load_model(g_paths["train"], seed_g, layer_num, DEVICE)
            g_eval = load_model(g_paths["eval"], seed_g, layer_num, DEVICE)

            seed_display = g_paths["display"]
            logger.info(f"Starting processing for model: {model_g}, seed: {seed_display}, task: {task_name}")

            
            #||h-\psi(g)||
            for model_h in args.model_type: #[model for model in args.model_type if model != model_g]:
                logger.info(f"Using {lin_type} Network for {model_g} on labels of model {model_h}")                        
                logger.info(f"Training transformation for model g: {model_g} on labels of model h: {model_h} on task {task_name}")
                if model_h == "multiberts":
                    seed_range_h = range(args.num_seeds)
                else:
                    seed_range_h = [None]
                for seed_h in seed_range_h:
                    h_paths = get_model_paths(ROOT_PATH, model_h, task_name, seed_h)
                    logger.info(f"Training {lin_type} transformation: {model_g}({g_paths['display']}) → {model_h}({h_paths['display']})")
  
                    h_train = load_model(h_paths["train"], seed_h, layer_num, DEVICE)
                    h_eval = load_model(h_paths["eval"], seed_h, layer_num, DEVICE)

                    save_dir = os.path.join(
                            ROOT_PATH,
                            f"intrinsic_models{args.activation}/{model_g}{g_paths['suffix']}_vs_{model_h}{h_paths['suffix']}"
                        )
                    os.makedirs(save_dir, exist_ok=True)
                    
                    losses = []
                    for lr in learning_rates:

                        model, epoch_loss, epoch_max_loss, loss_dict = train_transformation_model(g_train, g_eval, h_train, h_eval, DEVICE, num_epochs, lr, batch_size, patience=20, lin_type=lin_type)
                        model.eval()
                        losses.append([epoch_loss, epoch_max_loss])

                        g_test = load_model(g_paths["test"], seed_g, layer_num, DEVICE)
                        h_test = load_model(h_paths["test"], seed_h, layer_num, DEVICE)

                        
                        test_dataset = TensorDataset(torch.Tensor(g_test), torch.Tensor(h_test))
                        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
                        all_dist = []
                        all_test_preds = []
                        all_test_labels = []
                        all_g = []
                        all_h = []
                        # start testing and calculate distances
                        with torch.no_grad():    
                            for g_batch, h_batch in test_dataloader:
                                output = model(g_batch)
                                dist = torch.max(torch.abs(h_batch - output))  # Compute infinity norm distance
                                logger.info(f"Distance {lin_type}  on Test Data ||h-\psi(g)||_inf for lr={lr}: {dist.item()}")
                                all_dist.append(dist.item())
                                #dist_last_layer = torch.max(torch.abs(h_batch[-1] - output[-1])).item()
                                #logger.info(f"Distance last layer ||h-\psi(g)||_\infty: {dist_last_layer}")
                                all_test_labels.append(h_batch)
                                all_test_preds.append(output)
                                all_g.append(g_batch)
                                all_h.append(h_batch)
                        overall_max = max(all_dist)
                        logger.info(f"Overall {lin_type} Distance on Test Data ||h-\psi(g)||_inf for lr={lr}: {overall_max}")
                        preds_combined = torch.cat(all_test_preds, dim=0) 
                        labels_combined = torch.cat(all_test_labels, dim=0)

                        spearman_val = spearman_corr_torch(preds_combined, labels_combined)
                        logger.info(f"Spearman correlation on Test Data {lin_type} : {spearman_val:.4f}")
                        
                        rho = pearson_corr(all_g, all_h)
                        logger.info(f"Pearson correlation on Test Data {lin_type} : {rho:4f}")
                        #if lin_type == "nonlinear":
                        #    lipschitz_upper_bound = compute_lipschitz_constant(model)
                        #    logger.info(f"Lipschitz upper bound of the model: {lipschitz_upper_bound}")
                        #else:
                        #    lipschitz_constant = 1.0
                        #    for layer in model:
                        #        if isinstance(layer, nn.Linear):
                        #            # Compute the largest singular value (spectral norm) of the weight matrix
                        #            weight = layer.weight
                        #            singular_values = torch.linalg.svdvals(weight)  # Compute singular values
                        #            largest_singular_value = singular_values.max()  # Get the largest singular value
                        #            lipschitz_constant *= largest_singular_value.item() 
                        #        logger.info(f"Lipschitz upper bound of the model: {lipschitz_constant}")

                        #lipschitz_lower_bound = estimate_lipschitz_constant(model, batch_size, g_test, DEVICE)
                        lipschitz_lower_bound = None
                        lipschitz_upper_bound = None
                        logger.info(f"Lipschitz lower bound for lr={lr}: {lipschitz_lower_bound}")
                        metrics_entry = {
                        "task": task_name,
                        "model_g": model_g,
                        "model_h": model_h,
                        "seed_g": seed_g,
                        "seed_h": seed_h,
                        "lr": lr,
                        "type": lin_type,
                        "distance_inf_overall": overall_max,
                        "distance_mean": float(np.mean(all_dist)),
                        "distance_std": float(np.std(all_dist)),
                        "spearman": spearman_val,
                        "pearson": rho,
                        "lipschitz_upper": lipschitz_upper_bound if lin_type == "nonlinear" else lipschitz_constant,
                        "lipschitz_lower": lipschitz_lower_bound,
                        
                        }
                        metrics_entry.update(loss_dict)
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                        path_to_save = os.path.join(save_dir, f"{task_name}_{model_g}{g_paths['display']}_lr{lr}.pt")

                        torch.save(model, path_to_save)
                        logger.info(f"Model saved to {path_to_save}")
                        # JSON-Datei pro Task speichern
                        json_filename = f"data/results_max_loss/intrinsic/reproducable_3ReLU_final_training{task_name}_{lin_type}.json"
                        with open(json_filename, "a", encoding="utf-8") as f:
                            f.write(json.dumps(metrics_entry) + "\n")
                    losses_seed.append(losses)
            with open(os.path.join(
                            f"data/results_max_loss/intrinsic/reproducable_v2-intrinsic-maxloss-accuracies-{task_name}_{lin_type}-small-{model_g}-{seed_g}.npy",
                        ),
                        "wb",
                    ) as f:
                        np.save(f, np.array(losses_seed))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["cola", "mrpc", "qnli", "qqp", "sst2","rte"], nargs="+", help="Name(s) of the task(s) (e.g., sst2, mnli).")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the dataloaders.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--layer", type=int, default=12, help="Layer of the model to extract embeddings from.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the model.")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1], help="List of learning rates to use for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default=["multiberts", "roberta", "electra"],  help="Type of model to use (bert, roberta, electra).")
    parser.add_argument("--cache_dir", type=str, default="data/cache", help="Directory to store the pretrained models downloaded from huggingface.co.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
    parser.add_argument("--activation", type=str, default="3ReLU_final_training_reproducable", help="Name of the activation function used in the model (e.g., _3_ReLu, _4_Sigmoid).")

    args = parser.parse_args()
    tasks = args.task_name
    lin_types = ["nonlinear", "linear"]
    mp.set_start_method("spawn", force=True)  # <- wichtig für CUDA-Kompatibilität

    # Erstelle Liste aller Kombinationen
    job_args = [(args, task_name, lin_type) for task_name in tasks for lin_type in lin_types]
    # Nur EINMAL initialisieren
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("Logfiles/3ReLU_final_training_reproducable.log", mode='w')
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