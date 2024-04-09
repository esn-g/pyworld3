import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

activation = nn.LeakyReLU()

class Neural_Network(nn.Module):
    def __init__(self, input_size=12, hidden_sizes=[20,20,20,20,20], output_size=12, activation=activation):
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

        # Add the first hidden layer
        #self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        # Add the last hidden layer
        #self.hidden_layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Activation function
        self.activation = activation
    
    def __str__(self) -> str:
        return super().__str__()

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


def validate_model(model, val_loader, criterion, device):
    """
    Validate the neural network model.

    Parameters:
        model (torch.nn.Module): The neural network model to be validated.
        val_loader (torch.utils.data.DataLoader): DataLoader containing the validation dataset.
        criterion: The loss function used for validation.
        device: The device used.

    Returns:
        float: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)  # Convert inputs to float tensor
            labels = labels.long().to(device)    # Convert labels to long tensor

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute validation loss

            val_running_loss += loss.item() * inputs.size(0)

    val_loss = val_running_loss / len(val_loader.dataset)
    return val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train the neural network model.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader containing the validation dataset.
        criterion: The loss function used for training.
        optimizer: The optimizer used for updating model parameters.
        num_epochs (int): The number of epochs to train the model.
        device: The device used.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)  # Convert inputs to float tensor
            labels = labels.long().to(device)    # Convert labels to long tensor
            
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute training loss

            loss.backward()  # Backward pass
            optimizer.step()  # Optimize

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Validate the model after each epoch
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

# Example usage:
# train_loader and val_loader should be DataLoader instances containing your training and validation datasets
# model, criterion, optimizer, num_epochs, and device should be defined as in your existing code

# train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

