
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("create_dataset")
from generate_dataset_classfile import Generate_dataset
standard_state_matrix = np.array(Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len1_state_ppmvar0_norm.json")['Model_runs']["Run_0_State_matrix"])

modelstrings = ['Neural_network/model/XnewGenthinPreluppmvar_400000.0_L1True_lambda_1e-08_PReLU_hiddenSz_10_BSz_100_COSAnn_Start_0.001_epochs_400Last_TrainingLoss_3.088466915842266e-07Last_ValidationLoss_1.99721821140623e-07.pt'
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
        #jac_status = isHurwitz(jac)
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
    # print(str((mean_jacobian)))
    plot_heatmaps(mean_jacobian, var_jacobian, i)
