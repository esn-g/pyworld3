import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


import torch
import torch.nn as nn

class Neural_Network(nn.Module):
    def __init__(self, input_size=12, hidden_sizes=[20,20,20], output_size=12, activation=nn.ReLU()):
        """
        Constructor for the Neural_Network class.

        Parameters:
            input_size (int): Dimensionality of the input data.
            hidden_sizes (list): List of integers specifying the number of units in each hidden layer.
            output_size (int): Dimensionality of the output data.
            activation (torch.nn.Module): Activation function to be used between hidden layers.
        """
        super(Neural_Network, self).__init__()     #Init motherclass functionality
        
        # Define input and output layers
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        # Define hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        # Add the last hidden layer
        self.hidden_layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Activation function
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the neural network.
        Parameters:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Input layer
        x = self.input_layer(x)
        x = self.activation(x)
        
        # Hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Train the neural network model.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
        criterion: The loss function used for training.
        optimizer: The optimizer used for updating model parameters.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.float()  # Convert inputs to float tensor
            labels = labels.long()    # Convert labels to long tensor
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
