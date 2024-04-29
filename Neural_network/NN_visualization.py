import os
import sys

import torch
from torch import nn

from NN_eli import Neural_Network
from Dataset_classfile import CustomDataset

import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3 
from pyworld3.utils import plot_world_variables

################# ----- LOAD MODEL ------ ##############
modelstring = "Neural_network/model/XnewGenthinPreluppmvar_400000.0_L1True_lambda_1e-08_PReLU_hiddenSz_10_BSz_100_COSAnn_Start_0.001_epochs_400Last_TrainingLoss_3.088466915842266e-07Last_ValidationLoss_1.99721821140623e-07.pt"
model=torch.load(modelstring)
def weightHeatmap(model):
    num_valid_params = sum(1 for param in model.parameters() if param.dim() == 2)
    fig, axes = plt.subplots(nrows=1, ncols=num_valid_params, figsize=(20, 5))

    plot_index = 0  # Counter for valid plots

    all_params = [param for param in model.parameters() if param.dim() == 2]
    for param in all_params:
        param.requires_grad = False

    all_param_values = np.concatenate([param.numpy().flatten() for param in all_params])

    #vmin = np.min(all_param_values)
    #vmax = np.max(all_param_values)

    #centered around 0
    vmin = 0
    vmax = np.max(abs(all_param_values))

    for name, param in model.named_parameters():
        print('Name: '+ name + '\nParameters: '+ str(param)+ '\nSdim: ' + str(param.dim()) + '\n'+str(param.size(dim=-1)) )
        if param.dim() == 2:
            im = axes[plot_index].imshow(abs(param), cmap='viridis',vmin=vmin, vmax=vmax)
            axes[plot_index].set_title(name)
            cbar = fig.colorbar(im, ax=axes[plot_index])
            plot_index += 1
        
        #for i in param:
            #print(np.mean(np.abs(i)))
            #pass
            # np.mean(abs(i))
            #plot_index += 1


plt.show()

# varje rad är en nod : källa elliot
# rader stegas igenom först : källa elliot
        