from collections import defaultdict
import logging
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import geoopt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


logger = logging.getLogger(__name__)# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("train_\psi(g)saveLogitsandProb.log")],
)
log_level = logging.INFO
logger.setLevel(log_level)


# Define a nonlinear classifier
class Nonlinear(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)



def convert_class_indicies_to_one_hot(
    class_indices: torch.tensor, n_classes: int
) -> torch.tensor:
    one_hot_labels = torch.zeros((class_indices.shape[0], n_classes))
    one_hot_labels[range(class_indices.shape[0]), class_indices.to(int)] = 1
    return one_hot_labels



def get_all_models(data_folder):
    """
    Select models for comparison. Multiberts models with different seeds
    will only be compared to roberta and electra, not to each other.

    Args:
        data_folder (str): Path to the folder containing model subfolders.

    Returns:
        dict: Dictionary of models grouped by type (multiberts, roberta, electra).
    """
    model_folders = [os.path.join(data_folder, folder) for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    multiberts_models = [folder for folder in model_folders if "multiberts" in folder]
    roberta_models = [folder for folder in model_folders if "roberta" in folder]
    electra_models = [folder for folder in model_folders if "electra" in folder]

    return {
        "multiberts": multiberts_models,
        "roberta": roberta_models,
        "electra": electra_models,
    }


def load_labels(label_path, seed, device):
    """
    Load labels from the specified path and seed.

    Args:
        label_path (str): Path to the labels.
        seed (int or None): Seed value for the labels.
        device (str): Device to load the labels on (e.g., 'cuda' or 'cpu').

    Returns:
        torch.Tensor: Loaded labels.
    """
    if seed is not None:
        label_path = label_path.format(seed=seed)
    labels = torch.load(label_path, map_location=device)
    return labels

def load_model(model_path, seed, layer, device):
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
    model = torch.load(model_path, map_location=device).select(0, layer)
    return model

def make_path(root, models_type, task, split, seed_suffix):
    return os.path.join(root, f"models/{models_type}/task-{task}-{split}{seed_suffix}")



def main():
    parser = argparse.ArgumentParser(description="Extract embeddings and labels from a model.")
    parser.add_argument("--root_path", type=str, default="data/nonlin/", help="Path to the directory where the extracted embeddings are stored.")
    parser.add_argument("--task_name", type=str, default=["cola", "mrpc", "qnli", "qqp", "sst2","rte"], nargs="+", help="Name(s) of the task(s) (e.g., sst2, mnli).")
    parser.add_argument("--num_seeds", type=int, default=25, help="Number of seeds to use for generating embeddings.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for the classification task.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached preprocessed datasets or not.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory where the embeddings and labels will be saved.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the dataloaders.")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of epochs for training.")
    parser.add_argument("--layer", type=int, default=12, help="Layer of the model to extract embeddings from.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of the hidden layer in the model.")
    parser.add_argument("--learning_rates", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1], help="List of learning rates to use for training.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation (e.g., cpu or cuda).")
    parser.add_argument("--model_name_path", default="roberta-base", type=str, help="Path to the pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--model_type", type=str, default=["roberta", "electra", "multiberts"],  help="Type of model to use (bert, roberta, electra).")
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
    patience = 20

    logger.info(f"running on {DEVICE}")

    logger.info(f"running on {DEVICE}")
    logger.info(f"layer: {layer}")
    logger.info(f"batch size: {args.batch_size}")
    logger.info(f"num epochs: {num_epochs}")
    logger.info(f"learning rates: {learning_rates}")

    models = get_all_models(args.root_path)

   


    for task_name in args.task_name:
        if task_name == "mnli":
            output_size = 3
        else:
            output_size = 2
        
        
        for models_type in models:
            os.makedirs(f"{ROOT_PATH}predictions{args.activation}/{models_type}/", exist_ok=True)
            os.makedirs(f"{ROOT_PATH}classifier{args.activation}/{models_type}/", exist_ok=True)

            # set seed range if models_type is multiberts, otherwise use None
            seed_range = range(args.num_seeds) if models_type == "multiberts" else [None]
           
       
            # iterate over all multibert models and compare them with electra and roberta
            for SEED in seed_range:
                # Define paths dynamically based on the model type
                seed_suffix = f"-seed-{SEED}" if models_type == "multiberts" else ""
                # Usage:
                hidden_states_train_path = make_path(ROOT_PATH, models_type, task_name, "train", seed_suffix)
                hidden_states_eval_path = make_path(ROOT_PATH, models_type, task_name, "eval", seed_suffix)
                hidden_states_test_path = make_path(ROOT_PATH, models_type, task_name, "test", seed_suffix)
                # Define paths for labels
                train_label_path = make_path(ROOT_PATH, models_type, task_name, "labels_train", seed_suffix)
                eval_label_path = make_path(ROOT_PATH, models_type, task_name, "labels_eval", seed_suffix)
                test_label_path = make_path(ROOT_PATH, models_type, task_name, "labels_test", seed_suffix)
               
                # Load hidden states (embeddings) for train and eval splits for the current model/seed/layer
                hidden_states_train = load_model(hidden_states_train_path, SEED, layer, DEVICE)
                hidden_states_eval = load_model(hidden_states_eval_path, SEED, layer, DEVICE)
                hidden_states_test = load_model(hidden_states_test_path, SEED, layer, DEVICE)
                # Load labels   
                hidden_states_train_labels = load_labels(train_label_path, SEED, DEVICE)
                hidden_states_eval_labels = load_labels(eval_label_path, SEED, DEVICE)      
                hidden_states_test_labels = load_labels(test_label_path, SEED, DEVICE)      
                if hidden_states_test_labels[0].item() != -1:
                    print(task_name)


                logger.info(f"task name: {task_name}")
                logger.info(f"seed: {SEED}")

                # for layer in range(13):
                logger.info(f"layer: {layer}")


                # Convert class indices to one-hot encoded labels for loss calculation
                hidden_states_train_oh = convert_class_indicies_to_one_hot(hidden_states_train_labels, output_size)
                hidden_states_eval_oh = convert_class_indicies_to_one_hot(hidden_states_eval_labels, output_size)
                hidden_states_test_oh = convert_class_indicies_to_one_hot(hidden_states_test_labels, output_size)

                hidden_states_train_oh = hidden_states_train_oh.to(DEVICE)
                hidden_states_eval_oh =hidden_states_eval_oh.to(DEVICE)
                hidden_states_test_oh = hidden_states_test_oh.to(DEVICE)


                # train the transformation \psi(g) for all models, then they will be compared later
                # transfomration is trained on the hidden_states of g and its labels that are extracted in get-embeddings from hugginface
                transform_input_size = hidden_states_train.shape[-1]   # d_g

                dataset = TensorDataset(torch.Tensor(hidden_states_train), torch.Tensor(hidden_states_train_labels))  # g → h
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                test_dataset = TensorDataset(torch.Tensor(hidden_states_test), torch.Tensor(hidden_states_test_labels))  # g → h
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                eval_dataset = TensorDataset(torch.Tensor(hidden_states_eval), torch.Tensor(hidden_states_eval_labels))  # g → h
                eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

                # Define loss function
                criterion = nn.CrossEntropyLoss()
                test_accuracies = {}
                accuracies = {}
                test_f1s = {}
                eval_f1s = {}
                eval_pred_labels = {}
                test_predicted_labels = {}

                for lr in args.learning_rates:
                    # Initialize a fresh model for each learning rate
                    model = Nonlinear(transform_input_size, hidden_size, output_size)
                    model.to(DEVICE)
                    
                    best_linf = float("inf")
                    best_model_state = None
                    epochs_no_improve = 0
                    logger.info(f"lr: {lr}")
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
                    
                    # Training loop
                    for epoch in range(num_epochs):
                        model.train()
                        epoch_loss = 0.0
                        batch_count = 0

                        # Forward pass and loss computation for each batch
                        for batch_hs_g, batch_labels in dataloader: 
                            outputs = model(batch_hs_g) 
                            loss = criterion(outputs, batch_labels)      
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                            batch_count += 1
                            
                        epoch_loss /= batch_count  # Average loss per batch

                        logger.info(
                            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}"
                        )

                        # Evaluate model on validation set after each epoch
                        
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for batch in eval_dataloader:  # Iterate over validation DataLoader
                                g_eval, label_eval = batch  # Unpack the batch
                                g_eval, label_eval = g_eval.to(DEVICE), label_eval.to(DEVICE)  # Move data to DEVICE
                                val_pred = model(g_eval)  # Forward pass
                                val_loss += criterion(val_pred, label_eval).item()  # Accumulate batch loss

                            val_loss /= len(eval_dataloader)  # Average validation loss over all batches

                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.6f}")

                        # Early stopping
                        if val_loss < best_linf - 1e-4:
                            best_linf = val_loss
                            best_model_state = model.state_dict()
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                logger.info(f"Early stopping at epoch {epoch+1}")
                                break


                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        # Calculate training and evaluation accuracy
                        # Listen zum Speichern der Labels und Vorhersagen
                        all_test_preds = []
                        all_test_labels = []
                        all_test_probs = []
                        # Testphase: Iteriere über alle Batches. Speichere die Vorhersagen da Labels alle -1 im test split
                        model.eval()
                        with torch.no_grad():
                            for batch in test_dataloader:
                                batch_inputs, batch_labels_test = batch
                                batch_inputs = batch_inputs.to(DEVICE)

                                # Modellvorhersagen (Logits)
                                test_logits = model(batch_inputs)
                                
                                # Konvertiere Logits zu Wahrscheinlichkeiten
                                test_probs = torch.softmax(test_logits, dim=-1)
                                
                                # Vorhersagen (Klassen)
                                test_preds = torch.argmax(test_logits, dim=1)

                                # Speichere die Labels und Vorhersagen
                                all_test_probs.append(test_probs.cpu().numpy())  # Normalized probabilities
                                all_test_preds.append(test_preds.cpu().numpy())
                                all_test_labels.append(batch_labels_test.cpu().numpy())
                                
                                # Debug: Prüfe auf konstante Wahrscheinlichkeiten
                                if len(all_test_probs) == 1:  # Nur beim ersten Batch
                                    prob_var = torch.var(test_probs).item()
                                    logger.info(f"LR {lr}: Test probs variance = {prob_var:.8f}, range = [{test_probs.min():.6f}, {test_probs.max():.6f}]")

                        # Berechne Accuracy
                        # Concatenate all_preds_np and all_labels into single arrays
                        all_test_preds_np = np.concatenate(all_test_preds)
                        all_test_labels_np = np.concatenate(all_test_labels)

                        if all_test_labels_np[0] != -1:
                            # Calculate accuracy
                            test_acc = (all_test_preds_np == all_test_labels_np).sum() / len(all_test_labels_np)

                            average_method = "weighted" if output_size > 2 else "binary"
                            # Calculate F1-Score
                            test_f1 = f1_score(
                                all_test_labels_np,
                                all_test_preds_np,
                                average=average_method
                            )
                        else:
                            test_acc = 0.0
                            test_f1 = 0.0


                        all_eval_preds = []
                        all_eval_labels = []
                        all_eval_probs= []


                        with torch.no_grad():
                            for batch in eval_dataloader:  # Iteriere über den Validation DataLoader
                                g_eval, label_eval = batch
                                g_eval = g_eval.to(DEVICE)
                                label_eval = label_eval.to(DEVICE)

                                # Modellvorhersagen (Logits)
                                val_logits = model(g_eval)
                                
                                # Konvertiere Logits zu Wahrscheinlichkeiten
                                val_probs = torch.softmax(val_logits, dim=-1)
                                
                                # Vorhersagen (Klassen)
                                val_preds = torch.argmax(val_logits, dim=1)

                                # Speichere die Labels und Vorhersagen
                                all_eval_probs.append(val_probs.cpu().numpy())  # Normalized probabilities
                                all_eval_labels.append(label_eval.cpu().numpy())
                                all_eval_preds.append(val_preds.cpu().numpy())
                        # Konvertiere die gesammelten Daten in Arrays
                        #all_labels = np.concatenate(all_labels)
                        #all_preds = np.concatenate(all_preds)


                        # Berechne Accuracy
                        # Concatenate all_preds_np and all_labels into single arrays
                        all_preds_np = np.concatenate(all_eval_preds)
                        all_labels_np = np.concatenate(all_eval_labels)

                        # Calculate accuracy
                        eval_acc = (all_preds_np == all_labels_np).sum() / len(all_labels_np)

                        # Berechne F1-Score
                        eval_f1 = f1_score(
                            all_labels_np,
                            all_preds_np,
                            average="weighted" if output_size > 2 else "binary"
                        )
                        test_predicted_labels[lr] = {'test_acc':test_acc, 'test_f1':test_f1, 'eval_acc':eval_acc, 'f1_score':eval_f1, 'test_labels':all_test_labels, "test_probs":all_test_probs, "test_pred":all_test_preds,'eval_label':all_eval_labels, "eval_pred":all_eval_preds, 'eval_probs':all_eval_probs }
                        logger.info(f"Validation Accuracy: {eval_acc:.4f}, Validation F1-Score: {eval_f1:.4f}")

                    # Save predictions and model checkpoints for later analysis
                    for name, tensor in [("predictions", all_preds_np)]:
                        # make_path gibt dir den Ordnerpfad, z.B. .../models/{models_type}/task-{task_name}-eval-seed-{SEED}
                        save_dir = make_path(ROOT_PATH, models_type, task_name, name, seed_suffix)
                        # Ersetze ggf. "models" durch "predictions" im Pfad
                        save_dir = save_dir.replace("models", f"predictions{args.activation}")
                        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                        # Dateinamen anhängen
                        save_path = f"{save_dir}-lr{lr}"
                        torch.save(tensor, save_path)

                #save_path_eval_f1 = make_path(ROOT_PATH, models_type, task_name, "f1-scores", seed_suffix).replace("models", f"predictions{args.activation}") + ".pt"
                #save_path_eval_acc = make_path(ROOT_PATH, models_type, task_name, "prediction-accuracies", seed_suffix).replace("models", f"predictions{args.activation}") + ".pt"
                save_path_test_pred = make_path(ROOT_PATH, models_type, task_name, "predictions", seed_suffix).replace("models", f"predictions{args.activation}") + ".pt"
                #save_path_eval_pred = make_path(ROOT_PATH, models_type, task_name, "eval_predictions_labels", seed_suffix).replace("models", f"predictions{args.activation}") + ".pt"


                #torch.save(eval_f1s, save_path_eval_f1)
                #torch.save(accuracies, save_path_eval_acc)
                torch.save(test_predicted_labels, save_path_test_pred)
                #torch.save(eval_pred_labels, save_path_eval_pred)
                # save model checkpoint
                save_path = make_path(ROOT_PATH, models_type, task_name, "classifier_full", seed_suffix).replace("models", f"classifier{args.activation}") + ".pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_dim': transform_input_size,
                    'hidden_dim': hidden_size,
                    'output_dim': output_size,
                }, save_path)
                logger.info(f"Model {models_type} saved to {save_path}")

if __name__ == "__main__":
    main()
