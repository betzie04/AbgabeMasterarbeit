import torch, torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

class NonLinearNetwork(nn.Module):
    """A simple non-linear neural network with:
    - one linear layer (spectrally normalized for training stability)
    - a ReLU activation function to introduce non-linearity"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=20),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

def train_neural_network(X_train, Y_train, device, epochs, lr):
    """This function trains the network to learn a transformation from input to target data
    It adjusts the network's parameters so that its output matches the target as closely as possible"""
    
    model = NonLinearNetwork(X_train.shape[-1], Y_train.shape[-1]).to(device) # Create the network and move it to the CPU or GPU
    opt = torch.optim.Adam(model.parameters(), lr=lr) # Define the optimizer (Adam adjusts the model's weights during training)

    # Mean Squared Error (MSE) measures how close the network's output is to the target
    loss_fn = nn.MSELoss(reduction='none')  # "none" keeps loss values for each sample

    # Create batches of data for efficient training
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)), batch_size=128, shuffle=True)

    # Loop over the training process for the given number of epochs
    for ep in range(epochs):
        model.train()
        total_loss, max_loss = 0, 0  # Track average and worst-case error

        # Go through all batches
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()         # Reset gradients from previous step
            out = model(x)          # Run input through the network
            loss = loss_fn(out, y)  # Compute squared error
            max_l = loss.mean(dim=1).max()  # Get the worst error in the batch
            max_l.backward()        # Backpropagation: compute gradients
            opt.step()              # Update model weights

            # Accumulate loss statistics
            total_loss += loss.mean().item() * len(x)
            max_loss = max(max_loss, max_l.item())

        # Print training progress for this epoch
        print(f"Epoch {ep+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}, Max-Loss: {max_loss:.4f}")
    
    return model


