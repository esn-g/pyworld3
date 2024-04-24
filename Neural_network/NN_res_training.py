import os
from turtle import color
from sympy import N, Min
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from NN_classfile_clean_eli import Neural_Network
from Dataset_classfile import CustomDataset

################### --- PLOTTING SETTINGS --- ###################################
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'  # Use the Computer Modern font for math text (LaTeX style)

# Set the font size for different plot elements
plt.rcParams['axes.labelsize'] = 12  # Font size for labels (x and y)
plt.rcParams['axes.titlesize'] = 14  # Font size for the title
plt.rcParams['xtick.labelsize'] = 10  # Font size for the x ticks
plt.rcParams['ytick.labelsize'] = 10  # Font size for the y ticks
plt.rcParams['legend.fontsize'] = 10  # Font size for legend

# Adjust line width and dot size for better visibility in print
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 6

########################### --- Model --- ##########################

hidden_sizes = [20,20,20,20,20,20,20,20,20,20]
activation=nn.Tanh()
model=Neural_Network(hidden_sizes=hidden_sizes, activation=activation)

########################### --- Hyperparameters --- #################
# Larger batch sizes usually require larger learning rates, and smaller batch sizes usually require smaller learning rates to achieve convergence.
# When using larger batch sizes, you might need to increase the learning rate to compensate for the reduced noise in parameter updates.
# Conversely, when using smaller batch sizes, you might need to decrease the learning rate to prevent overshooting the minimum.

learning_rate = 1e-3 # 1e-6 bra början utan scheduler, 1e-3?
batch_size = 100
epochs = 600
criterion=nn.MSELoss()    #Saves lossfunc
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)


# Adaptive LR 
gamma = 0.9 # stock value 0.9? exponential 

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6, last_epoch=(-1), verbose='deprecated')
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, verbose=False)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular2', )
# scheduler = torch.optim.lr_scheduler.OneCycleLR()

#L1 regularization parameter
# for gold2000 parameters, lamda 0.00001 is too bad? lambda 0.0000001 Standard result
l1_lambda = 0.00000001

########## --- dataset --- #######################
# updated with validation set 19/4 2024

dataset = CustomDataset("create_dataset/dataset_storage/W3data_len100_ppmvar5000000_norm.json") # Create an instance of your map-style dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # shuffle True?
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#   Gather the variance:            ADDED BY ELI 16/APR // Thanks Dawg 
ppm_variance=dataset.ppmvar

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
    if torch.backends.mps.is_available() and batch_size > 2000
    else "cpu"
)
print(f"Device: {device}")
model.to(device)

########################## --- GPT model trainer --- ######################
def train_model(model, train_loader, criterion, optimizer, num_epochs, ppmvar="-", L1regBool = False):
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
    training_losses = []  # To store epoch losses
    learning_rates = []  # To store epoch learning rates
    validation_losses =[]

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
            # L1 Regularization - after loss, before lossBACKWARD
            if L1regBool == True:
                l1_reg = 0
                for param in model.parameters():
                    l1_reg += torch.sum(abs(param))
                l1_reg /= len(list(model.parameters()))
            
                # Append l1 loss to total loss
                loss = loss + (l1_lambda*l1_reg)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)
        
        # calculate losses
        epoch_training_loss = running_loss / len(train_loader.dataset)
        epoch_validation_loss = validate_model(model, val_loader, criterion)
        
        # scheduler step, needs validation loss for RONL plateu.... epoch_validation_loss
        scheduler.step()

        # get LR for plotting
        learning_rate_print = scheduler.get_last_lr()

        # Append epoch loss and learning rate for plotting
        training_losses.append(epoch_training_loss)
        validation_losses.append(epoch_validation_loss)
        learning_rates.append(learning_rate_print)
       
        # prints to monitor
        # print("Loss.item()=",loss.item())
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_training_loss:.10f}")
        print(f"Validation Loss: {epoch_validation_loss:.10f}")
    
    # after run, save the model
    torch.save(model, "Neural_network/model/ppmvar_" +input_tag +str(ppmvar) +"_L1"+str(L1regBool)+"_lambda_" + str(l1_lambda) + "_PReLU_hiddenSz_"+ str(len(hidden_sizes)) + '_BSz_'+ str(batch_size) + "_COSAnn_Start_" + str(learning_rate) + "_epochs_" + str(num_epochs) + 'Last_TrainingLoss_' + str(epoch_training_loss) +  'Last_ValidationLoss_' + str(epoch_validation_loss) +".pt")

    fig, axs = plt.subplots(1, 2, figsize=(8, 10))

    # Plotting epoch loss
    axs[0].plot(range(1, num_epochs + 1), training_losses, label= 'Training Loss', color ='b')
    axs[0].plot(range(1, num_epochs + 1), validation_losses, label='Validation Loss', color ='r')
    axs[0].set_yscale('log')  # Set y-axis to logarithmic scale
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Log Loss')
    axs[0].set_title('Epoch Log Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting epoch learning rate
    axs[1].plot(range(1, num_epochs + 1), learning_rates)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Learning Rate')
    axs[1].set_title('Epoch Learning Rate')
    axs[1].grid(True)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

def validate_model(model, val_loader, criterion):
    model.eval().to(device)  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for inputs, labels in val_loader:
            inputs = inputs.float().to(device)  # Convert inputs to float tensor
            labels = labels.float().to(device)
            delta_outputs = model(inputs)
            outputs = inputs + delta_outputs
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    return val_loss / len(val_loader.dataset)

ans = input('L1reg? y/n \n')
input_tag = input('custom name,specify string: ')
if ans == 'y':
    print('L1 regularization used')
    userL1 = True
else:
    print('No Regularization used.')
    userL1 = False
train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=epochs, ppmvar=ppm_variance, L1regBool=userL1)





