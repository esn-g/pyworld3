from email import utils
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
import sys
print("\nsyspath\n: ",sys.path,"\n")
from pyworld3 import World3, world3

from pyworld3.utils import plot_world_variables

import sys
sys.path.append("create_dataset")



from generate_dataset_classfile import Generate_dataset

#pip install git+file:///Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction
#/Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction


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


def next_state_estimate(model,current_state=torch.empty([]) ):
    next_state=model(current_state)
    return next_state


def estimated_model(model, state_matrix=np.empty([]),  number_of_states=601, start_state_index=0):
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

    estimated_state_matrix=np.empty( (number_of_states, 12))
    
    for k, state_vector in enumerate(state_matrix):
        '''sum=0
        
        for i in state_vector:
            sum+=state_vector-current_state
        print("Sum of error at run k=",k," SUM=",sum)
        '''
        estimated_state_matrix[k,:]=current_state

        #Format into tensor for NN forward
        current_state=torch.tensor(current_state).float()   

        #Estimate the next state via helper function
        next_state=next_state_estimate(model=model, current_state=current_state)

        #Reformat as numpy array
        next_state = next_state.detach().numpy()

        # Step forward
        current_state=next_state

        #print("\ncurr state\n",current_state)

    estimated_state_matrix[number_of_states-1,:]=current_state

    return estimated_state_matrix
    

# Estimates the normalized state-matrix
normalized_states_estimated=estimated_model(model=model, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)
#print("normestimaterun:\n",normalized_states_estimated)


# Denormalizes the estimated state matrix
states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())



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
    print("\nsum of standard matrix: ",np.sum(standard_state_matrix),"\n\n")
    print("\nsum of estimated matrix: ",np.sum(states_estimated),"\n\n")
    print("estimate run:\n",states_estimated)
    print("standard run:\n",standard_state_matrix)
    error_matrix=standard_state_matrix-states_estimated

    print("\nsum of standard matrix: ",np.sum(standard_state_matrix),"\n\n")
    print("\nsum of estimated matrix: ",np.sum(states_estimated),"\n\n")

    print("Error matrix: \n", error_matrix)

    ### Generate the NORMALIZED VERSION error matrix 
    print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")
    print("normalized estimate run:\n",normalized_states_estimated)
    print("normalized standard run:\n",normalized_state_matrix)
    normalized_error_matrix=normalized_state_matrix-normalized_states_estimated

    print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")

    print("Normalized error matrix: \n", normalized_error_matrix)
    #print("estimaterun:\n",states_estimated)

    mean_of_variable_errors=np.mean(normalized_error_matrix, axis=0)
    print("mean vector: \n",mean_of_variable_errors)
    mean_of_variable_errors=np.mean(abs(normalized_error_matrix), axis=0)
    print("abs mean vector: \n",mean_of_variable_errors)

    return error_matrix , normalized_error_matrix

generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)

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



def generate_statevars_dict(state_matrix=np.empty([601, 12]), est_matrix=np.empty([601, 12]), ):
    #List of variable names for all variables examined
        var_str_list=np.array(["al",
                            "pal",
                            "uil",
                            "lfert",
                            "ic",
                            "sc",
                            "ppol",
                            "p1",
                            "p2",
                            "p3",
                            "p4",
                            "nr"]).T       #Transposes it to get each variable as its own column - X matrix but names
        # Zip variable names to variable vectors in the state matrix
        state_vars_dict=dict( zip( var_str_list, state_matrix.T  ) )     # Transpose state matrix for correct alignment

        var_str_list_est=np.char.add(var_str_list,"_est")
        est_state_vars_dict=dict( zip( var_str_list_est, est_matrix.T  ) )     # Transpose state matrix for correct alignment
        print(state_vars_dict)
        print(est_state_vars_dict)
        return state_vars_dict, est_state_vars_dict
        

state_vars_dict, est_state_vars_dict=generate_statevars_dict(standard_state_matrix, states_estimated)
print(state_vars_dict)
for i in range(len(state_vars_dict["al"][0])):
    #print(state_vars_dict["al"][0][i])
     
    print("i: ",i,", real: ",state_vars_dict["al"][0][i],", est: ", state_vars_dict["al"][1][i])


def plot_auto(est_matrix,state_matrix=None,name=None, variables_included="all"):

    pass



     
#states_estimated    #on form 601*12
#states_estimated.T on form 12*601

for var in enumerate(states_estimated.T):
     pass
     

#    Make the rows into a list of np arrays instead

#   Take in a list of inddexes for which variables to be included

orig_est_data=list( state_vars_dict.values() )+list( est_state_vars_dict.values() )
#state_vars_dict.values().extend( est_state_vars_dict.values() )

var_names= list( state_vars_dict.keys() ) + list( est_state_vars_dict.keys() )
#state_vars_dict.keys().extend( est_state_vars_dict.keys() )

var_limits= np.amax( orig_est_data, axis=1 )

#np.amax( state_vars_dict.values().extend( state_vars_dict.values() )

len(state_vars_dict.values())


dict_of_plotvars= {
        "var_data" : orig_est_data ,
        "var_names" : var_names ,
        "var_lims" : var_limits ,    #ADd axis=1
        "img_background" : None,
        "title" :  None,    #Add
        "figsize" : (7, 5),                                   
        "grid" : True,
        "line_styles" : ["-"] }    #"dist_spines" : float = 0.09,




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
#plt.savefig("fig_world3_AR_test_poli1_diff_big.png")

#Fixa error för hela variablerna, över hela, typ sum eller mean

'''
State variables:

Agriculture:
al
pal
uil
lfert


Capital:
ic 
sc

Pollution:
ppol

Population:
p1
p2
p3
p4

Resource:
nr'''

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

    - L1 regularization - ESN
        -Gör Jacobian igen

    - L1 och varied dataset     -POST JAC
        -Gör J igen


    -optional task - add evaluation loss

    
    
    '''