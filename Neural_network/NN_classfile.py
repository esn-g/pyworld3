
#Import statements for basic PINN from video https://www.youtube.com/watch?v=MtHvs5FvtME
import math
import numpy as np
#import seaborn     #For visualization
import matplotlib.pyplot as plt
from collections import OrderedDict
#from tqdm import tqdm  #Progress bar

import torch
import torch.nn as nn

#Import statements specific for our project
#Import pyworld3



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, act):
        """
        Initialize the neural network module.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layers.
            output_size (int): Number of output features.
            depth (int): Number of hidden layers.
            act (str): Activation function for the hidden layers ('relu', 'sigmoid', or 'tanh').
        """
        super(NeuralNetwork, self).__init__()

        # Define layers using OrderedDict to maintain order
        layers = OrderedDict()
        layers['input_layer'] = nn.Linear(input_size, hidden_size)  # Input layer
        layers['activation'] = self.get_activation(act)  # Activation function
        for i in range(depth - 2):
            layers[f'hidden_layer_{i+1}'] = nn.Linear(hidden_size, hidden_size)  # Hidden layers
            layers[f'activation_{i+1}'] = self.get_activation(act)  # Activation function
        layers['output_layer'] = nn.Linear(hidden_size, output_size)  # Output layer
        
        # Create a sequential model from the layers
        self.model = nn.Sequential(layers)

    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def get_activation(self, act):
        """
        Return the specified activation function.

        Args:
            act (str): Activation function name ('relu', 'sigmoid', or 'tanh').

        Returns:
            torch.nn.Module: Activation function module.
        """
        if act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f'Unknown activation function: {act}')

# Example usage:
input_size = 10
hidden_size = 20
output_size = 1
depth = 3
act = 'relu'

# Create an instance of the NeuralNetwork class
model = NeuralNetwork(input_size, hidden_size, output_size, depth, act)
print(model)
