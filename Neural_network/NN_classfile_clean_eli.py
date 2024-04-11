import os
import torch
from torch import nn
import matplotlib.pyplot as plt

class Neural_Network(nn.Module):
    def __init__(self, input_size=12, hidden_sizes=[20,20,20,20,20], output_size=12, activation=nn.ReLU()):
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
    
    def plot_weights_heatmap(self):
        """
        Plot heatmaps of the weights in each layer of the neural network.
        """
        plt.figure(figsize=(15, 5))

        # Plot input layer weights
        plt.subplot(1, len(self.hidden_layers) + 1, 1)
        plt.imshow(self.input_layer.weight.detach().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Input Layer Weights')

        # Plot hidden layers' weights
        for i, hidden_layer in enumerate(self.hidden_layers):
            plt.subplot(1, len(self.hidden_layers) + 1, i + 2)
            plt.imshow(hidden_layer.weight.detach().numpy(), cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title(f'Hidden Layer {i+1} Weights')

        # Plot output layer weights
        plt.subplot(1, len(self.hidden_layers) + 1, len(self.hidden_layers) + 2)
        plt.imshow(self.output_layer.weight.detach().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Output Layer Weights')

        plt.tight_layout()
        plt.show()


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
