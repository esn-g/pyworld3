import os
from sympy import Min
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from NN_classfile_clean_eli import Neural_Network
from Dataset_classfile import CustomDataset

########################### --- Model --- ##########################

hidden_sizes = [20,20,20,20,20]
model=Neural_Network(hidden_sizes=hidden_sizes, activation=nn.PReLU())

########################### --- Hyperparameters --- #################
# Larger batch sizes usually require larger learning rates, and smaller batch sizes usually require smaller learning rates to achieve convergence.
# When using larger batch sizes, you might need to increase the learning rate to compensate for the reduced noise in parameter updates.
# Conversely, when using smaller batch sizes, you might need to decrease the learning rate to prevent overshooting the minimum.

learning_rate = 1e-3 # 1e-6 bra början utan scheduler, 1e-3?
batch_size = 20
epochs = 2000
criterion=nn.MSELoss()    #Saves lossfunc
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# Adaptive LR
gamma = 0.9 # stock value 0.9? exponential 

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8, last_epoch=(-1), verbose='deprecated')

########## --- dataset --- #######################
dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json") # Create an instance of your map-style dataset
train_loader = DataLoader(dataset, batch_size=batch_size)

########## --- Device --- ########################
# small batch size ->> cpu
# small < 1000
# large batch size ->> mps/gpu
# large > 1000 roughly
# https://github.com/pytorch/pytorch/issues/77799

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and batch_size > 1000
    else "cpu"
)
print(f"Device: {device}")
model.to(device)

########################## --- GPT model trainer --- ######################
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
    # For plotting
    epoch_losses = []  # To store epoch losses
    epoch_learning_rates = []  # To store epoch learning rates

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:

            inputs = inputs.float().to(device)  # Convert inputs to float tensor
            labels = labels.float().to(device)
            # print("Input: ",inputs, "\nShape: ", inputs.shape)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            delta_outputs = model(inputs)
            # residual 
            outputs = inputs + delta_outputs
            
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)

        # scheduler update per epoch.
        scheduler.step()
        learning_rate_print = scheduler.get_lr()
        epoch_loss = running_loss / len(train_loader.dataset)

        # Append epoch loss and learning rate for plotting
        epoch_losses.append(epoch_loss)
        epoch_learning_rates.append(learning_rate_print[0])

        # prints to monitor
        print('[Epoch Learning rate]: ')
        print(learning_rate_print)
        print("Loss.item()=",loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.8f}")
    
    # after run, save the model
    torch.save(model, "Neural_network/model/model_goldGen_bsize_" + str(batch_size) + "_lr_"+ str(learning_rate) + "_epochs_" + str(num_epochs) + 'x' +".pt")

    # Plotting epoch loss
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss')
    plt.show()
    
    # Plotting epoch learning rate
    plt.plot(range(1, num_epochs + 1), epoch_learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Epoch Learning Rate')
    plt.show()

train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=epochs)


