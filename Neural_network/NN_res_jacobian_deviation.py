import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

sys.path.append("create_dataset")
from generate_dataset_classfile import Generate_dataset
standard_state_matrix = np.array(Generate_dataset.fetch_dataset("create_dataset/dataset_storage/dataset_runs_1_variance_0_normalized_.json")["Model_runs"]["Run_0_State_matrix"])

modelstrings = [
                'Neural_network/model/L1X_lambda:1e-07_PReLU_hiddenSz:10_BSz:20_COSAnn_Start:0.001_epochs_2000Last_Loss:5.294184613073109e-07.pt',
                'Neural_network/model/gold2000.pt',
                'Neural_network/model/Plen_100LASSO_OKppmvar_500000_L1YES_lambda_1e-07_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_5.501026285514854e-07.pt', #regularized
                'Neural_network/model/Plen_100ppmvar_500000_L1X_lambda_1e-06_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_9.909764973059509e-08.pt', # not regularized
                # add paths
                 ]
model_val_span = [ 0,
                   0,
                   0.05,
                   0.05,
                   ]

std_run_initials = standard_state_matrix[0,:]
data_points = input('Antal datapunkter: [DEF=100]')

def compute_sweep_vec(var, initials, data_points):
    # om variansen är för liten, använd 0.1
    if var < 0.1:
        span = 0.1
    else:
        span = i
    deviation = np.zeros(len(initials))
    deviation[:] = span
    lower = initials - deviation
    upper =  + initials + deviation
    sweep_arrays = []
 
    # Create a linspace for each element in the vector
    for i in range(len(initials)):
        linspace_array = np.linspace(lower[i], upper[i], int(data_points))
        sweep_arrays.append(linspace_array)

    # Convert the list of arrays to a NumPy array
    sweep_arrays= np.array(sweep_arrays)
    # sweep arrays is a list of np arrays 
    return sweep_arrays
def load_models(modelstrings):
    
    models = [torch.load(modelstring) for modelstring in modelstrings]
    for model in models:
        model.eval()
        model.to('cpu')
    return models
def calculate_jacobians(state_matrix, model):
    jacobians = []
    for state in state_matrix.T:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Adding batch dimension
        jac = torch.autograd.functional.jacobian(model, state_tensor).squeeze(0)  # Reducing batch dimension
        jac = jac.squeeze(1)  # Attempt to remove the unexpected singleton dimension
        # jac_status = isHurwitz(jac).item()
        # print("Is Hurwitz:", jac_status) # toggle hurwitz
        jac = torch.abs(jac)
        jacobians.append(jac)
    return jacobians
def isHurwitz(jacobian):
    eigenvalues = torch.linalg.eigvals(jacobian)
    return torch.all(torch.real(eigenvalues) < 0)
def compute_mean_variance(jacobians):
    stacked_jacobians = torch.stack(jacobians)
    mean_jacobian = torch.mean(stacked_jacobians, dim=0)
    var_jacobian = torch.var(stacked_jacobians, dim=0)
    # mean_norm = (mean_jacobian - torch.mean(mean_jacobian)) / torch.std(mean_jacobian)
    # var_norm = (var_jacobian - torch.mean(var_jacobian)) / torch.std(var_jacobian)
    return mean_jacobian, var_jacobian
def plot_heatmaps(mean_jacobian, var_jacobian, i):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(mean_jacobian.numpy(), cmap='jet')
    ax1.set_title('Abolute Mean of Jacobian' + str(i))
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(var_jacobian.numpy(), cmap='jet')
    ax2.set_title('Variance of Jacobian' + str(i))
    fig.colorbar(im2, ax=ax2)
    plt.show()
def plot_heatmaps_log(mean_jacobian, var_jacobian, i):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot absolute mean of Jacobian
    im1 = ax1.imshow(mean_jacobian.numpy(), cmap='jet', norm=colors.LogNorm())
    ax1.set_title('Logarithmic Scale: Absolute Mean of Jacobian ' + str(i))
    fig.colorbar(im1, ax=ax1)
    
    # Plot variance of Jacobian
    im2 = ax2.imshow(var_jacobian.numpy(), cmap='jet', norm=colors.LogNorm())
    ax2.set_title('Logarithmic Scale: Variance of Jacobian ' + str(i))
    fig.colorbar(im2, ax=ax2)
    
    plt.show()

model_sweep_list = [] # index gives sweep range 
for i in model_val_span:
    x = compute_sweep_vec(i, std_run_initials, data_points)
    model_sweep_list.append(x)

models = load_models(modelstrings)

# 'main'
i=0
for model in models:
    # väljer en del av namnet
    modelstring = modelstrings[i]
    modelstring_select = modelstring[21:34]
    # sweep
    model_sweep = model_sweep_list[i] 
    i += 1
    jacobians = calculate_jacobians(model_sweep, model)
    mean_jacobian, var_jacobian = compute_mean_variance(jacobians)
    plot_heatmaps(mean_jacobian, var_jacobian, modelstring_select)