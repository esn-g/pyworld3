import os
import torch
from torch import nn

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
