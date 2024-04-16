import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from NN_eli import Neural_Network
from Dataset_classfile import CustomDataset

from re import A
import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3 
from pyworld3.utils import plot_world_variables

import sys

# required to reach create_dataset folder
sys.path.append("create_dataset")

from generate_dataset_classfile import Generate_dataset
modelstring =input('Copy relative path to model:')
#modelstring = "Neural_network/model/L1_lambda:1_PReLU_hiddenSz:10_BSz:20_COSAnn_Start:0.001_epochs_2000Last_Loss:819.3883056640625.pt"

#dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json") # Create an instance of your map-style dataset
model=torch.load(modelstring)  

# FUNKAR EJ model.plot_weights_heatmap()
# Neural_network/model/model_res_gen4_bsize_20_lr_0.001_epochs_600x.pt

# set model to evaluation mode, required for testing
model.eval()

# get standard run
standard_state_matrix=np.array(     
    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
    )

normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())

def next_state_estimate(model,current_state=torch.empty([]) ):
    # principally 'forward'
    next_state=model(current_state)
    return next_state

def estimated_model(model, state_matrix=np.empty([]),  number_of_states=601, start_state_index=0):
    
    current_state=state_matrix[start_state_index,:]

    estimated_state_matrix=np.empty( (number_of_states, 12))
    
    for k, state_vector in enumerate(state_matrix):
        # # # # # # # #  
        sum=0
        for i in state_vector:
            sum+=state_vector-current_state

        # print("Sum of error at run k=",k," SUM=",sum)
        estimated_state_matrix[k,:]= current_state
        current_state=torch.tensor(current_state).float()   #Format for NN forward
        # residualnätverket tar fram differnensen
        # dx = network(x_est(current_state)) 
        next_state_diff=next_state_estimate(model=model, current_state=current_state)
        # next_state är gamla + diffen
        next_state = current_state + next_state_diff

        next_state = next_state.detach().numpy()

        current_state=next_state

    estimated_state_matrix[number_of_states-1,:]=current_state

    return estimated_state_matrix
    


normalized_states_estimated=estimated_model(model=model, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)

#
states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())
error_matrix=standard_state_matrix-states_estimated
print("Error matrix: \n", error_matrix)

al_est_full=states_estimated[:,0]
pal_est_full=states_estimated[:,1]
uil_est_full=states_estimated[:,2]
lfert_est_full=states_estimated[:,3]
print("est shape full: ",lfert_est_full.shape)

al=standard_state_matrix[:,0]
pal=standard_state_matrix[:,1]
uil=standard_state_matrix[:,2]
lfert=standard_state_matrix[:,3]

number_of_states=601
start_state_index=0
test_time=np.arange(0,300+.5,0.5)

plot_world_variables(
    test_time,
    [al, pal, uil, lfert, al_est_full, pal_est_full, uil_est_full, lfert_est_full ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, 2*max(al)], [0, 2*max(pal)], [0, 2*max(uil)], [0, 2*max(lfert)], [0, 2*max(al)], [0, 2*max(pal)], [0, 2*max(uil)], [0, 2*max(lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title='Model: ' + modelstring,
)
plt.show()
# plt.savefig(modelstring +'TEST_AGRIC_SEC_Lr_SCHED.pdf')

# funkar ej
# x =input('Save? <any>=save, <n> =no)')
# if x == 'n':
#    pass
# else: 
#    plt.savefig('TEST_AGRIC_SEC_' + modelstring)


