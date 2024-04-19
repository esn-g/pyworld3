
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
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

models = [torch.load(modelstring) for modelstring in modelstrings]
for model in models:
    model.eval()
    model.to('cpu')

def isHurwitz(jacobian):
    eigenvalues = torch.linalg.eigvals(jacobian)
    return torch.all(torch.real(eigenvalues) < 0)

def calculate_jacobians(state_matrix, model):
    jacobians = []
    for state in state_matrix:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # Adding batch dimension
        jac = torch.autograd.functional.jacobian(model, state_tensor).squeeze(0)  # Reducing batch dimension
        jac = jac.squeeze(1)  # Attempt to remove the unexpected singleton dimension
        #if jac.shape[0] == 12 and jac.shape[1] == 12:  # Confirm square matrix of correct size
        #print("Jacobian shape:", jac.shape)
        jac_status = isHurwitz(jac)
        # print("Is Hurwitz:", jac_status) # toggle hurwitz
        jac = torch.abs(jac)
        jacobians.append(jac)
    return jacobians

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

# Execution part
i=0
for model in models:
    i += 1
    jacobians = calculate_jacobians(standard_state_matrix, model)
    mean_jacobian, var_jacobian = compute_mean_variance(jacobians)
    plot_heatmaps(mean_jacobian, var_jacobian, i)
