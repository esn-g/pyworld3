import os
import sys

import torch
import matplotlib.pyplot as plt
import numpy as np

def weightHeatmap(models):
    num_models = len(models)
    max_layers = max([len([param for param in model.parameters() if param.dim() == 2]) for model in models])
    fig, axes = plt.subplots(nrows=num_models, ncols=max_layers, figsize=(5*max_layers, 5*num_models))

    for model_index, model in enumerate(models):
        all_params = [param for param in model.parameters() if param.dim() == 2]
        all_param_values = np.concatenate([param.detach().numpy().flatten() for param in all_params])
        vmax = np.max(abs(all_param_values))
        vmin=0
        plot_index = 0  # Counter for valid plots

        for name, param in model.named_parameters():
            if param.dim() == 2:
                ax = axes[model_index][plot_index]
                param.requires_grad = False
                param_values = abs(param.detach().numpy())
                #vmax = np.max(param_values)
                im = ax.imshow(param_values, cmap='viridis', vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax)
                ax.set_title(f"Model {model_index+1} - {name}")
                
                plot_index += 1
    plt.tight_layout()
    plt.show()


# Example usage:
models = [torch.load("Neural_network/model/Plen_100ppmvar_500000_L1X_lambda_1e-06_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_9.909764973059509e-08.pt"), torch.load("Neural_network/model/Plen_100LASSO_OKppmvar_500000_L1YES_lambda_1e-07_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_5.501026285514854e-07.pt")]
axes = None  # You can also provide your own list of axes
weightHeatmap(models)
