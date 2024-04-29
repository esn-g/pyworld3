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
                ax = axes[plot_index]
                param.requires_grad = False
                param_values = abs(param.detach().numpy())
                vmax = np.max(param_values)
                im = ax.imshow(param_values, cmap='viridis', vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax)
                ax.set_title(f"Model {model_index+1} - {name}")
                
                plot_index += 1
    plt.tight_layout()
    plt.show()


# Example usage:
models = [torch.load("Neural_network/model/XnewGenthinPreluppmvar_400000.0_L1True_lambda_1e-08_PReLU_hiddenSz_10_BSz_100_COSAnn_Start_0.001_epochs_400Last_TrainingLoss_3.088466915842266e-07Last_ValidationLoss_1.99721821140623e-07.pt")]
axes = None  # You can also provide your own list of axes
weightHeatmap(models)
