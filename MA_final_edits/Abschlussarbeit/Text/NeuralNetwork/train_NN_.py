import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

class NonLinearNetwork(nn.Module):
	"""A simple non-linear neural network with:
	- input layer, one hidden layer (with ReLU activation), output layer, spectral normalization for training stability"""
	def __init__(self, in_dim, out_dim, hidden_dim=128):
		super().__init__()
		self.net = nn.Sequential(
		spectral_norm(nn.Linear(in_dim, hidden_dim), n_power_iterations=20),
		nn.ReLU(),
		nn.Linear(hidden_dim, out_dim)
		)
		
	def forward(self, x):
		return self.net(x)

def train_neural_network(X_train, Y_train, device, epochs=100, lr=1e-3):
	"""Trains simple neural network to learn mapping X to Y."""

	model = NonLinearNetwork(X_train.shape[-1], Y_train.shape[-1]).to(device) # Create the network and move it to the CPU or GPU
	opt = torch.optim.Adam(model.parameters(), lr=lr) # Define the optimizer (Adam adjusts the weights during training)
	# Mean Squared Error (MSE) measures how close the network's output is to the target
	loss_fn = nn.MSELoss(reduction='none')  # Sample-wise loss
	
	loader = DataLoader(
			TensorDataset(torch.tensor(X_train, dtype=torch.float32),
				torch.tensor(Y_train, dtype=torch.float32)),
				batch_size=128, shuffle=True
				)
	
	# Loop over the training process for the number of epochs
	for epoch in range(epochs):
		model.train()
		total_loss, max_loss = 0.0, 0.0
		
		for x_batch, y_batch in loader:
			x_batch, y_batch = x_batch.to(device), y_batch.to(device)
				opt.zero_grad()	# Reset gradients from previous step
			output = model(x_batch)	# Run input through the network
			loss = loss_fn(output, y_batch)	# Compute squared error
			sample_loss = loss.mean(dim=1)           
			batch_loss = sample_loss.mean()
			batch_loss.backward()	# Backpropagation: compute gradients
			opt.step()	# Update model weights
			
			total_loss += batch_loss.item()
			max_loss = max(max_loss, sample_loss.max().item())
			
		print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {total_loss:.4f} | Max Sample Loss: {max_loss:.4f}")
	
	return model
