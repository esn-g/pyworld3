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
sys.path.append("create_dataset")

from generate_dataset_classfile import Generate_dataset



#dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json") # Create an instance of your map-style dataset
model=torch.load("Neural_network/model/model_gen1_bsize_100_lr_0.0001_epochs_1000.pt")  #Funkar inte helt kefft

#model=torch.load("Neural_network/model/model_gen1_bsize_50_lr_0.0001_epochs_1500.pt")  #Funkar kefft
model=torch.load("Neural_network/model/model_gen1_bsize_50_lr_0.0001_epochs_1000.pt")   #Funkar asbra -tur med init kanske
model.eval()




#standard_state_matrix=torch.tensor(     
#    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
#    )
standard_state_matrix=np.array(     
    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
    )

##print(standard_state_matrix)
normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())

#print(standard_state_matrix)
#print("normstandardrun:\n",normalized_state_matrix)

def next_state_estimate(model,current_state=torch.empty([]) ):
    next_state=model(current_state)
    return next_state

#state_1=next_state_estimate(X_state_matrix[0,:], A_state_transition_matrix )
#print("\nreal state 1: \n", X_state_matrix[1,:] ,"\nestimated state_1: \n", state_1[:])


def estimated_model(model, state_matrix=np.empty([]),  number_of_states=601, start_state_index=0):
    
    current_state=state_matrix[start_state_index,:]

    estimated_state_matrix=np.empty( (number_of_states, 12))
    
    for k, state_vector in enumerate(state_matrix):
        sum=0
        for i in state_vector:
            sum+=state_vector-current_state

        print("Sum of error at run k=",k," SUM=",sum)
        estimated_state_matrix[k,:]=current_state
        current_state=torch.tensor(current_state).float()   #Format for NN forward
        next_state=next_state_estimate(model=model, current_state=current_state)

        next_state = next_state.detach().numpy()

        current_state=next_state
        #print("\ncurr state\n",current_state)

    estimated_state_matrix[number_of_states-1,:]=current_state

    return estimated_state_matrix
    


normalized_states_estimated=estimated_model(model=model, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)
#print("normestimaterun:\n",normalized_states_estimated)


states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())
#states_estimated=Generate_dataset.min_max_DEnormalization(normalized_state_matrix.copy())
#standard_state_matrix=normalized_state_matrix

#print("normestimaterun:\n",normalized_states_estimated)
#print("normrun:\n",normalized_state_matrix)
#states_estimated=normalized_states_estimated
#standard_state_matrix=normalized_state_matrix

error_matrix=standard_state_matrix-states_estimated

#print("estimaterun:\n",states_estimated)

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
    [[0, 1.1*max(al)], [0, 1.1*max(pal)], [0, 1.1*max(uil)], [0, 1.1*max(lfert)], [0, 1.1*max(al)], [0, 1.1*max(pal)], [0, 1.3*max(uil)], [0, 1.3*max(lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test NN model 1ppm 2 runs",
)
plt.show()
#plt.savefig("fig_world3_AR_test_poli1_diff_big.png")


