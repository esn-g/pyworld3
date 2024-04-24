
import torch

from re import A
import matplotlib.pyplot as plt
import numpy as np
import sys

from pyworld3.utils import plot_world_variables

import sys
sys.path.append("create_dataset")

from generate_dataset_classfile import Generate_dataset

from NN_utils import plot_state_vars



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
    


def generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated):
    ''' Generate error matrix
    Takes in arguments - real and estimated matrix, original and normalized
    All arrays are on the form: each row corresponds to a state at value k
    the columns are the different variables
    '''
        

    error_matrix=standard_state_matrix-states_estimated

    #print("\nsum of standard matrix: ",np.sum(standard_state_matrix),"\n\n")
    #print("\nsum of estimated matrix: ",np.sum(states_estimated),"\n\n")


    ### Generate the NORMALIZED VERSION error matrix 

    normalized_error_matrix=normalized_state_matrix-normalized_states_estimated

    print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")


    #norm_mean_of_variable_errors=np.mean(abs(normalized_error_matrix), axis=0)
    #print("Absolute mean of normalized errormatrix, vector: \n",norm_mean_of_variable_errors)

    mean_of_variable_errors=np.mean( np.abs(error_matrix) , axis=0)
    print("mean square error of errormatrix, vector of vars: \n",mean_of_variable_errors)

    mean_per_var_world3run=np.mean( standard_state_matrix, axis=0 )

    print("mean of world3 run per var: \n", mean_per_var_world3run)

    MSE_normalized=np.divide( mean_of_variable_errors, mean_per_var_world3run )

    print("Mse normalized vector: \n", MSE_normalized)

    return error_matrix , normalized_error_matrix

#generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)



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
        spec_vars=["all"]
    elif spec_vars in sectors:
        print("sector")
        spec_vars=sectors[spec_vars]
        print(spec_vars)
    elif any(spec_vars in var_list for var_list in sectors.values()):
        spec_vars=[spec_vars]
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

    #Fetches other run
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
    plot_state_vars(state_matrix=standard_state_matrix, est_matrix=states_estimated, variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )

main()

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
