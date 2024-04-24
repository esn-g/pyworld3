
import torch
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
#
#from NN_eli import Neural_Network
#from Dataset_classfile import CustomDataset

from re import A
import matplotlib.pyplot as plt
import numpy as np
import sys
#print("\nsyspath\n: ",sys.path,"\n")
#from pyworld3 import World3, world3

#import pyworld3
from pyworld3.utils import plot_world_variables

import sys
sys.path.append("create_dataset")

#pyworld3.world3.



from generate_dataset_classfile import Generate_dataset

from NN_utils import plot_state_vars

# Attempted to make own branch of pyworld and make it editable, passed on this for now because it raised to much issues
# Instead we now create an extra utils-file to add improved functions
#pip install git+file:///Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction
#/Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction
#pip install -e git+file:///Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction#egg=pyworld3


''' Before main()
#dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json") # Create an instance of your map-style dataset
#model=torch.load("Neural_network/model/model_gen1_bsize_100_lr_0.0001_epochs_1000.pt")  #Funkar inte helt kefft

#model=torch.load("Neural_network/model/model_gen1_bsize_50_lr_0.0001_epochs_1500.pt")  #Funkar kefft
model=torch.load("Neural_network/model/model_gen1_bsize_50_lr_0.0001_epochs_1000.pt")   #Funkar asbra -tur med init kanske
model.eval()    #eval mode




#standard_state_matrix=torch.tensor(     
#    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
#    )

#Fetches the standard run
standard_state_matrix=np.array(     
    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
    )


# Normalizes the standard run and saves without altering original matrix
normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())

'''


def next_state_estimate(model,current_state=torch.empty([]) ):
    # principally 'forward'
    next_state=model(current_state)
    return next_state


def estimated_model(model, residual=True, state_matrix=np.empty([]),  number_of_states=601, start_state_index=0):
    ''' estimated_model function to generate the estimated state-matrix based on the NN-model
    Takes in arguments: 
    model - the NN model to estimate the states
    state_matrix - the original normalized matrix to compare to
    number_of_states - how many states should be estimated
    start_state_index - from which k the estimation starts

    The state_matrix of k=[starts_state_index] gives the initial value to the model which is passed forward to generate the next
    this is then repeated for the entire state-matrix
    '''
    
    current_state=state_matrix[start_state_index,:]

    estimated_state_matrix=np.empty( (number_of_states, 12) )
    
    for k, state_vector in enumerate(state_matrix):
        '''sum=0
        
        for i in state_vector:
            sum+=state_vector-current_state
        print("Sum of error at run k=",k," SUM=",sum)
        '''
        estimated_state_matrix[k,:]=current_state

        #Format into tensor for NN forward
        current_state=torch.tensor(current_state).float()   

        #Estimate the next state via helper function, depends on residual bool
        if residual:
            next_state_diff=next_state_estimate(model=model, current_state=current_state)
            # next_state är gamla + diffen
            next_state = current_state + next_state_diff
        else:
            next_state=next_state_estimate(model=model, current_state=current_state)

        #Reformat as numpy array
        next_state = next_state.detach().numpy()

        # Step forward
        current_state=next_state

        #print("\ncurr state\n",current_state)

    estimated_state_matrix[number_of_states-1,:]=current_state

    return estimated_state_matrix
    

''' Before main()
# Estimates the normalized state-matrix
normalized_states_estimated=estimated_model(model=model, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)
#print("normestimaterun:\n",normalized_states_estimated)


# Denormalizes the estimated state matrix
states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())

'''

#print("normestimaterun:\n",normalized_states_estimated)
#print("normrun:\n",normalized_state_matrix)
#states_estimated=normalized_states_estimated
#standard_state_matrix=normalized_state_matrix



def generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated):
    ''' Generate error matrix
    Takes in arguments - real and estimated matrix, original and normalized
    All arrays are on the form: each row corresponds to a state at value k
    the columns are the different variables
    '''
        
    # Generate the total error matrix
    #print("\nsum of standard matrix: ",np.sum(standard_state_matrix),"\n\n")
    #print("\nsum of estimated matrix: ",np.sum(states_estimated),"\n\n")
    #print("estimate run:\n",states_estimated)
    #print("standard run:\n",standard_state_matrix)
    error_matrix=standard_state_matrix-states_estimated

    #print("\nsum of standard matrix: ",np.sum(standard_state_matrix),"\n\n")
    #print("\nsum of estimated matrix: ",np.sum(states_estimated),"\n\n")
    #
    #print("Error matrix: \n", error_matrix)

    ### Generate the NORMALIZED VERSION error matrix 
    #print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    #print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")
    #print("normalized estimate run:\n",normalized_states_estimated)
    #print("normalized standard run:\n",normalized_state_matrix)
    normalized_error_matrix=normalized_state_matrix-normalized_states_estimated

    print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")

    #print("Normalized error matrix: \n", normalized_error_matrix)
    #print("estimaterun:\n",states_estimated)

    #mean_of_variable_errors=np.mean(normalized_error_matrix, axis=0)
    #print("mean vector: \n",mean_of_variable_errors)
    norm_mean_of_variable_errors=np.mean(abs(normalized_error_matrix), axis=0)
    print("Absolute mean of normalized errormatrix, vector: \n",norm_mean_of_variable_errors)

    mean_of_variable_errors=np.mean( np.square(error_matrix) , axis=0)
    print("mean square error of errormatrix, vector of vars: \n",mean_of_variable_errors)

    return error_matrix , normalized_error_matrix

#generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)
''' Outdated stuff:
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
'''


#Best 'Neural_network/model/Plen_100LASSO_OKppmvar_500000_L1YES_lambda_1e-07_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_5.501026285514854e-07.pt' #regularized


def main():
    modelstring =input('Copy relative path to model: ')
    if modelstring=="":
        modelstring="Neural_network/model/gold2000.pt"
    model=torch.load(modelstring)  
    residual= input("Residual? (T/F): ")
    if residual=="F":
        residual=False
    else:
        residual=True
    sectors={"agr" : ["al", "pal", "uil", "lfert"],
            "cap" : ["ic", "sc"],
            "pol" : ["ppol"],
            "pop" : ["p1", "p2", "p3", "p4"],
            "res" : ["nr"]
            }
    #sectors=["agr","cap", "pol", "pop", "res"]
    print("Input specified variables or a sector to plot. \nChoose from ", list(sectors.items()))
    spec_vars=input("SKIP for 'all': ")

    #spec_vars=str(spec_vars)
    print(spec_vars)
    if spec_vars =="":
        print("all")
        spec_vars="all"
    elif spec_vars in sectors:
        print("sector")
        spec_vars=sectors[spec_vars]
        print(spec_vars)
    elif any(spec_vars in var_list for var_list in sectors.values()):
        print("var")
        pass
    
    else:
        print("incorrect sector/variable name")
        spec_vars="all"
    # set model to evaluation mode, required for testing
    model.to('cpu') #Incased trained on gpu, transfer back
    model.eval()

    ##Fetches the standard run
    #standard_state_matrix=np.array(     
    #    Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
    #    )

    #Fetches the standard run
    standard_state_matrix=np.array(     
        Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len100_ppmvar500000.json")["Model_runs"]["Run_4_State_matrix"]  
        )


    # Normalizes the standard run and saves without altering original matrix
    normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())
    # Estimates the normalized state-matrix
    normalized_states_estimated=estimated_model(model=model, 
    residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)

    # Denormalizes the estimated state matrix
    states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())

    #Gather error matrices and print some error values
    generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)

    #Plot the state variables chosen standard is "all"
    plot_state_vars(state_matrix=standard_state_matrix, est_matrix=states_estimated, variables_included=[spec_vars]) #, variables_included= ["nr", "ppol","sc"] )

main()

'''

plot_world_variables(
    test_time,
    [al, pal, uil, lfert, al_est_full, pal_est_full, uil_est_full, lfert_est_full ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, 1.1*max(al)], [0, 1.1*max(pal)], [0, 1.1*max(uil)], [0, 1.1*max(lfert)], [0, 1.1*max(al)], [0, 1.1*max(pal)], [0, 1.3*max(uil)], [0, 1.3*max(lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test NN model 1ppm 2 runs",
    line_styles=["-", "-", "-", "-", "--", "--", "--", "--"],
)
plt.show()
'''
#plt.savefig("fig_world3_AR_test_poli1_diff_big.png")

#Fixa error för hela variablerna, över hela, typ sum eller mean

'''
State variables:
(associated colours)

Agriculture: - (green)
al
pal
uil
lfert

["al",
"pal",
"uil",
"lfert"]

Capital: - (blue)
ic 
sc

["ic", 
"sc"]

Pollution:   - (gray/brown)
ppol

["ppol"]

Population: -   (red)
p1
p2
p3
p4

["p1",
"p2",
"p3",
"p4"]

Resource:   -   (pink/purple)
nr

["nr"]
'''

'''

Jacobian  - (TORCH.AUTOGRAD.FUNCTIONAL.JACOBIAN ?)
using only how inputs influence the outputs, gradient on xi

Then heatmap on jacobian

x=f(x0+delta_xi)
grad( f(x) ) med avseende på xi

Jk
mean of J=1/n * [J1+J2]
variance of J=1/n * (J1-mean of J)^2



Hurwitz- stability


Loss function using L1 regulariziation
Loss= Sum(y-f(x))^2 + |theta|


Do the learning convergence plot using logarithmic scale - how it is generally done

Log(loss)
^
 |
 |
 |
_|____________> k steps
 |

or 

|theta_k - theta_k-1|^2
^
 |
 |
 |
_|_________________> k_steps
 |


 Do everything for standard run
 - hypothesis for the correlatins

 Then create varied dataset
 use it to verify

 



To do list :

    - Spruta ut varied dataset  -ELLIO - CHECK/ELL (10 runs for 1k var resp 10k , 100k - [ppm])
    - Plot alla vars - elx

    - heatmap - 0=black 

    - Log för convergence   CHECK/ESN

    - Add get ppmvar in dataset classfile - use to name our trained models  -ELL

    - Kolla upp Jacobian - hur gör man på bästa sätt    -TEaMWORK
        - extrahera J och gör heatmap för ett visst k - k=0 eller 1 
        -HEATMAPS
        - gör för flera k , kanske stega igenom standard run, J för alla k%50=0
        - ta mean och var 

    - L1 regularization - CHECK/ESN
        -Gör Jacobian igen

    - L1 och varied dataset     -POST JAC
        -Gör J igen


    -optional task - add evaluation loss

    
    
    '''