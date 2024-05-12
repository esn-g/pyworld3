
from sympy import plot
import torch
#from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
#
#from NN_eli import Neural_Network
#from Dataset_classfile import CustomDataset

#from re import A
import matplotlib.pyplot as plt
import numpy as np
import sys
#print("\nsyspath\n: ",sys.path,"\n")
#from pyworld3 import World3, world3

import sys
sys.path.append("create_dataset")





from generate_dataset_classfile import Generate_dataset

from NN_utils import plot_state_vars

# Attempted to make own branch of pyworld and make it editable, passed on this for now because it raised to much issues
# Instead we now create an extra utils-file to add improved functions
#pip install git+file:///Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction
#/Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction
#pip install -e git+file:///Users/elliottapperkarlsson/Github/pyworld3@AR_data_extraction#egg=pyworld3



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
    
    #estimated_state_matrix[number_of_states-1,:]=current_state     #To get last step , but not needed


    return estimated_state_matrix

def generate_single_step_matrix(model, residual=True, state_matrix=np.empty([]),  number_of_states=601, start_state_index=0):
    ''' function to generate estimated steps based on the NN-model from each k of the original state-matrix 
    Takes in arguments: 
    model - the NN model to estimate the states
    state_matrix - the original normalized matrix to compare to
    number_of_states - how many states should be estimated
    start_state_index - from which k the estimation starts

    The state_matrix of k=[starts_state_index] gives the initial value to the model which is passed forward to generate estimation
    this is done for each k and saved in a matrix
    '''
    
    #   Gets first state vector
    est_next_state=state_matrix[start_state_index,:]

    #   Initialize matrices
    singlestep_state_matrix=np.empty( (number_of_states, 12) )

    singlestep_delta_x_matrix=np.empty( (number_of_states, 12) )

    #   First value is the same because we cant estimate it
    #singlestep_state_matrix[0,:]=current_state     #Is done anyway

    #   First delta is zero , previous state plus delta is current state - begins att k=0 deltax=0
    #singlestep_delta_x_matrix[0,:]=0
    est_next_state_diff=0
    
    for k, state_vector in enumerate(state_matrix):

        #   Save into delta matrix
        singlestep_delta_x_matrix[k,:]=est_next_state_diff

        #   Save into singlestep state matrix , takes the previous state(k-1) state plus the delta x to get state at k 
        singlestep_state_matrix[k,:]=est_next_state
        #   state at k
        current_state=state_matrix[k,:]



       

        #Format into tensor for NN forward
        current_state=torch.tensor(current_state).float()   

        #Estimate the next state via helper function, depends on residual bool
        if residual:
            est_next_state_diff=next_state_estimate(model=model, current_state=current_state)

            # est_next_state is current state(standard matrix) + estimated diff
            est_next_state = current_state + est_next_state_diff     
        else:
            print("\nNOT FIXED THIS FUNCTIONALITY\n")
            #next_state=next_state_estimate(model=model, current_state=current_state)      #   NOT USED NOW

        #Reformat as numpy array
        est_next_state_diff=est_next_state_diff.detach().numpy() 
        est_next_state=est_next_state.detach().numpy() 
        

    return singlestep_state_matrix
    



def generate_single_step_error_matrix(standard_state_matrix, normalized_state_matrix, singlestep_states_estimated, normalized_singlestep_states_estimated):
    ''' Generate error matrix
    Takes in arguments - real and estimated matrix, original and normalized
    All arrays are on the form: each row corresponds to a state at value k
    the columns are the different variables
    '''

            
    #       Get matrix of all delta x values where X[k]=X[k-1] + delta_x[k]
    delta_x_matrix=np.empty( singlestep_states_estimated.shape) 
    delta_x_matrix[0,:]=0
    delta_x_matrix[1:,:]=standard_state_matrix[1:]-standard_state_matrix[:-1]

    est_delta_x_matrix=np.empty( singlestep_states_estimated.shape) 
    est_delta_x_matrix[0,:]=0
    est_delta_x_matrix[1:,:]=singlestep_states_estimated[1:]-standard_state_matrix[:-1]


     #################################### GET ERROR MATRICES ############################################
    #   Given standard run and single step est - > error matrix per element = x[k-1]+est_delta_x[k] - x[k-1]+delta_x[k]
    step_error_matrix=singlestep_states_estimated-standard_state_matrix      # Generate total singlestep error matrix

    abs_step_error_matrix=abs(step_error_matrix)

   

    '''
    # sets zero values to small value
    supersmall_delta=1e-10
    delta_x_matrix[delta_x_matrix==0]=supersmall_delta
    for k, delta_x in enumerate(delta_x_matrix):
        if np.any(delta_x==0):

            print("0 at k=",k)
            print(delta_x)
    '''
    
                        ###################### DIVIDE BY DELTA X SQUARED MEAN, or MINIMUM VALUE MAX VAL OF DELTA X - GIVES SCALE



    #################################### GET NORMALIZER MEANS ############################################
    #       Fetch the mean of standard matrices per variable for normalizing
    #mean_of_standard_run=np.mean(standard_state_matrix, axis=0)
    #print("mean of standard run: \n", mean_of_standard_run)

    #       Fetch the mean of standard step/delta matrices per variable for normalizing

    abs_mean_of_standard_run_STEPS=np.mean(abs(delta_x_matrix), axis=0) #Added absolute value of the steps to get just stepsize mean
    #print("mean of standard run abs_delta_x step: \n", abs_mean_of_standard_run_STEPS)

   

    #################################### MAKE RELATIVE ############################################
     #relative_step_error_matrix=np.divide(step_error_matrix, delta_x_matrix)    # For normalizing over step
    
    #mean_of_variable_step_errors=np.mean(abs(relative_step_error_matrix), axis=0)  #For abs mean of normalized step error
    #print("Absolute mean of relative step-error-matrix, vector: \n",mean_of_variable_step_errors)

    #   Divides the diff, error  by mean of the real steps - makes it somewhat relative error
    relative_to_mean_step_error_matrix=np.divide(abs_step_error_matrix, abs_mean_of_standard_run_STEPS)
    #print("REL shape:", relative_to_mean_step_error_matrix.shape)

    mean_of_variable_step_errors=np.mean(abs(step_error_matrix), axis=0)
    print("Absolute mean of step-error-matrix, vector: \n",mean_of_variable_step_errors)

    tot_rel_to_mean_step_error_matrix=np.mean(abs(relative_to_mean_step_error_matrix), axis=1)

    #reshape
    tot_rel_to_mean_step_error_matrix=tot_rel_to_mean_step_error_matrix.reshape(-1,1)
    print("totvar REL shape:", tot_rel_to_mean_step_error_matrix.shape)


    mean_of_step_error=np.mean(tot_rel_to_mean_step_error_matrix,axis=0)
    print("\n\n\nMEAN VALUE OF ERROR ACROSS ALL X[K] AND ACROSS VARS    STEPPPPPP:",mean_of_step_error,"\n\n\n")
    #return tot_rel_to_mean_step_error_matrix#relative_to_mean_step_error_matrix#abs_step_error_matrix
    return tot_rel_to_mean_step_error_matrix, relative_to_mean_step_error_matrix#step_error_matrix , relative_step_error_matrix


def generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated):
    ''' Generate error matrix
    Takes in arguments - real and estimated matrix, original and normalized
    All arrays are on the form: each row corresponds to a state at value k
    the columns are the different variables
    '''

    #################################### GET NORMALIZER MEANS ############################################
    #       Fetch the mean of standard matrices per variable for normalizing
    mean_of_standard_run=np.mean(standard_state_matrix, axis=0)
    #print("mean of standard run: \n", mean_of_standard_run)

    #################################### GET ERROR MATRICES ############################################
    # Generate the total error matrix
    error_matrix=standard_state_matrix-states_estimated

    abs_error_matrix=abs(error_matrix)

    #################################### MAKE RELATIVE ############################################
    ### Generate the NORMALIZED VERSION error matrix 
    #normalized_error_matrix=normalized_state_matrix-normalized_states_estimated

    #print("\nsum of normalized standard matrix: ",np.sum(normalized_state_matrix),"\n\n")
    #print("\nsum of normalized estimated matrix: ",np.sum(normalized_states_estimated),"\n\n")

    #norm_mean_of_variable_errors=np.mean(abs(normalized_error_matrix), axis=0)
    #print("Absolute mean of normalized errormatrix, vector: \n",norm_mean_of_variable_errors)

    #mean_of_variable_errors=np.mean( np.square(error_matrix) , axis=0)
    #print("mean square error of errormatrix, vector of vars: \n",mean_of_variable_errors)


     #   Divides the diff, error  by mean of the real x values - makes it somewhat relative error
    relative_to_mean_error_matrix=np.divide(abs_error_matrix, abs(mean_of_standard_run))
    print("RUN REL shape:", relative_to_mean_error_matrix.shape)

    mean_of_var_errors=np.mean(abs(error_matrix), axis=0) #For seeing mean error per var
    print("Absolute mean of error-matrix per var, vector: \n",mean_of_var_errors)

    tot_rel_to_mean_error_matrix=np.mean(abs(relative_to_mean_error_matrix), axis=1)

    #reshape
    tot_rel_to_mean_error_matrix=tot_rel_to_mean_error_matrix.reshape(-1,1)
    print("totvar RUN REL shape:", tot_rel_to_mean_error_matrix.shape)

    mean_of_error=np.mean(tot_rel_to_mean_error_matrix,axis=0)

    print("\n\n\nMEAN VALUE OF ERROR ACROSS ALL X[K] AND ACROSS VARS:",mean_of_error,"\n\n\n")

    return  tot_rel_to_mean_error_matrix, relative_to_mean_error_matrix






    return error_matrix #, normalized_error_matrix



def report_models():    #For getting the models in the report
    NN_0= "Neural_network/model/XY_RESULTS0000000L1XXXXXXXXLowerL1ppmvar_0_L1True_lambda_1e-09_PReLU_hiddenSz_10_BSz_16_COSAnn_Start_0.001_epochs_12000Last_TrainingLoss_1.2536813986940842e-08Last_ValidationLoss_1.8733208650978384e-09.pt"
    NN_20= "Neural_network/model/XY_RESULTS202020L1YESXXXXXX_20percentAttemptppmvar_200000.0_L1True_lambda_1e-09_PReLU_hiddenSz_10_BSz_32_COSAnn_Start_0.001_epochs_600Last_TrainingLoss_3.742628736347342e-08Last_ValidationLoss_1.609640550093161e-08.pt"
    NN_40="Neural_network/model/XY_RESULT40000L1TRUEXXXXXXyppmvar_400000.0_L1True_lambda_1e-09_PReLU_hiddenSz_10_BSz_100_COSAnn_Start_0.001_epochs_600Last_TrainingLoss_1.0176896774183319e-07Last_ValidationLoss_1.6479816197604673e-07.pt"
    report_NN_dict={ "NN_0" : NN_0 , "NN_20" : NN_20 , "NN_40" : NN_40 }
    return report_NN_dict

#Best 'Neural_network/model/Plen_100LASSO_OKppmvar_500000_L1YES_lambda_1e-07_PReLU_hiddenSz_10_BSz_600_COSAnn_Start_0.001_epochs_2000Last_Loss_5.501026285514854e-07.pt' #regularized

def specify_plotting():
    report_NN_dict=report_models()
    print("\n",'_' * 75)  # Line separator
    modelstring =input('\n\nCopy relative path to model: ')
    multi_NN=False
    if modelstring=="":
        modelstring="Neural_network/model/XnewGen1%TANHMOOOOREppmvar_10000.0_L1False_lambda_1e-08_PReLU_hiddenSz_10_BSz_50_COSAnn_Start_0.001_epochs_800Last_TrainingLoss_6.735498562365772e-09Last_ValidationLoss_6.849705511124959e-09.pt"
        #"Neural_network/model/gold2000.pt"
    elif modelstring=="NN":
        print("\nAll report models\n")
        print(modelstring)
        print(report_NN_dict.values())
        
        model_list = list( report_NN_dict.values() )
        
        multi_NN=True


    elif modelstring in report_NN_dict.keys():
        print("\nModel ",modelstring, " from report models\n")
        modelstring=report_NN_dict[modelstring] #Load actual string from dict

    
    if multi_NN!=True:
        model=torch.load(modelstring)  
    else:
        for nr, modelstring in enumerate(model_list):
            model=torch.load(modelstring)  
            model_list[nr]=model
        model=model_list



    #   residual= input("Residual? (T/F): ")    # REMOVED FOR NOW
    residual=True           # ADDED FOR NOW
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

    
    print('_' * 75)  # Line separator
    print("\nInput specified variables or a sector to plot. \n\nChoose from:\n" )#, list(sectors.items()))
    for key, value in sectors.items():
        print(f'{key}: {value}')
    print("or step, acc for step error or estimation error plots\n")
    spec_vars=input("SKIP for 'all': ")
    

    #spec_vars=str(spec_vars)
    #print(spec_vars)
    if spec_vars =="":
        print("Plotting all state-variables")
        spec_vars=["all"]
    elif spec_vars in sectors:
        print("Plotting sector ", spec_vars)
        spec_vars=sectors[spec_vars]
        print(spec_vars)
    elif any(spec_vars in var_list for var_list in sectors.values()):
        print("Plotting variable ", spec_vars)
        spec_vars=[spec_vars]
        
    elif spec_vars=="std":
        spec_vars=["std"]
        print("Plotting std variables")

        #______________________Errors versions for report
    elif spec_vars=="step":
        print("Plotting step-errors")
        spec_vars=[spec_vars]
        if isinstance(model, list): #If report models
            NN_names=[model for model in list( report_NN_dict.keys() ) ]
            print(NN_names)
            spec_vars.extend( NN_names )
        
        print(spec_vars)
        
    elif spec_vars=="acc":
        print("Plotting estimation-errors")
        spec_vars=[spec_vars]
        if isinstance(model, list): #If report models
            NN_names=[model for model in list( report_NN_dict.keys() ) ]
            spec_vars.extend( NN_names )
        print(spec_vars)
    
    else:
        print("NOT sector/variable name, alternative testing")
        spec_vars=[spec_vars]
    print('_' * 75, "\n")  # Line separator

    return model, residual, spec_vars

def report_error_plotting(model_list, normalized_state_matrix, standard_state_matrix, spec_vars=["step"], residual=True, number_of_states=601, start_state_index=0):        #iterate for the three models
    '''Generates error plots for three neural networks as used inm the report results'''
    array_of_errors=np.zeros((number_of_states, len(model_list)))
    

    for col, model in enumerate(model_list):
        print("Getting errors of model nr ", col)
            

        ######################################### Estimate using the model #########################################################

        # Estimates the normalized state-matrix
        normalized_states_estimated=estimated_model(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)


        #   Generate the single step model matrix
        norm_singlestep_estimations=generate_single_step_matrix(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)

        ########################################## DE-Normalizing ########################################################
        
        # Denormalizes the estimated state matrix
        states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())


        # Denormalizes the estimated state matrix
        singlestep_estimations=Generate_dataset.min_max_DEnormalization(norm_singlestep_estimations.copy())
        #print(singlestep_estimations,"\n\n")

        print("\n\n",spec_vars[col+1],"\n\n")
        ######################################### Calculate errors ####### ##################################################
        if spec_vars[0]=="acc":
            print(" if spec_vars[0]==acc")
             #   Gather error matrices and print some error values
            est_error, error_matrix=generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)
            #est_error=est_error.reshape(-1,1)
            est_error=est_error.reshape(-1,)
            array_of_errors[:, col]=est_error

        if spec_vars[0]=="step":
            print(" if spec_vars[0]==step")
            #   Gather the single step error matrix
            step_error_vect, step_error_matrix= generate_single_step_error_matrix(standard_state_matrix, normalized_state_matrix, singlestep_estimations, norm_singlestep_estimations)
            #step_error_vect=step_error_vect.reshape(-1,1)
            #print("col=",col,"\narrayoferrors shape=", array_of_errors.shape,"\nsteperror shape=", step_error_vect.shape, "\none column of array shape=", array_of_errors[:,col].shape)
            step_error_vect=step_error_vect.reshape(-1,)
            array_of_errors[ : , col]=step_error_vect

            print("SHAPE", array_of_errors.shape)
            

    return array_of_errors

def main():

    #   Fetch parameters for plotting
    model, residual, spec_vars = specify_plotting()
    
    # set model to evaluation mode, required for testing
    model.to('cpu') #Incased trained on gpu, transfer back
    model.eval()

    ################################## Fetching runs ################################################################
    ##Fetches the standard run
    standard_state_matrix=np.array(     
        Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
        )

    ##Fetches the standard run
    #standard_state_matrix=np.array(     
    #    Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len2_state_ppmvar2e+05.json")["Model_runs"]["Run_1_State_matrix"]  
    #    )

    alt_state_matrix=np.array(     
    Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len100_ppmvar100000.0.json")["Model_runs"]["Run_4_State_matrix"]  
    )
    ########################################## Normalizing ########################################################

    # Normalizes the standard run and saves without altering original matrix
    normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())
    

    ######################################### Estimate using the model #########################################################


    # Estimates the normalized state-matrix
    normalized_states_estimated=estimated_model(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)


     #   Generate the single step model matrix
    norm_singlestep_estimations=generate_single_step_matrix(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)

    ########################################## DE-Normalizing ########################################################
    
    # Denormalizes the estimated state matrix
    states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())


    # Denormalizes the estimated state matrix
    singlestep_estimations = Generate_dataset.min_max_DEnormalization(norm_singlestep_estimations.copy())

    

    ######################################### Calculate errors #########################################################

        #   Gather error matrices and print some error values
    est_error, error_matrix = generate_error_matrix( standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)

       
        #   Gather the single step error matrix

    step_error_vect, step_error_matrix = generate_single_step_error_matrix( standard_state_matrix, normalized_state_matrix, singlestep_estimations, norm_singlestep_estimations )
    print( step_error_vect.shape )       
    print( step_error_vect[:,0] )
    #plot_state_vars(state_matrix=error_matrix,  variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )
    

    ######################################### Plot estimation or varied run #########################################################
    #Plot the state variables chosen, standard is "all"
    #plot_state_vars(state_matrix=standard_state_matrix, est_matrix=states_estimated, variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )

    ##Plot the state variables chosen, standard is "all"
    #plot_state_vars(state_matrix=standard_state_matrix, est_matrix=alt_state_matrix, variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )

#main()



def report_main():

    #   Fetch parameters for plotting
    model, residual, spec_vars = specify_plotting()
    
    if isinstance(model, list):
        for NN in model:
            # set model to evaluation mode, required for testing
            NN.to('cpu') #Incased trained on gpu, transfer back
            NN.eval()
    else:
        # set model to evaluation mode, required for testing
        model.to('cpu') #Incased trained on gpu, transfer back
        model.eval()

    ################################## Fetching runs ################################################################
    ##Fetches the standard run
    standard_state_matrix=np.array(     
        Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run"]  
        )

    ##Fetches the standard run
    #standard_state_matrix=np.array(     
    #    Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len2_state_ppmvar4e+05.json")["Model_runs"]["Run_1_State_matrix"]  
    #    )

    alt_state_matrix=np.array(     
    Generate_dataset.fetch_dataset("create_dataset/dataset_storage/W3data_len100_ppmvar100000.0.json")["Model_runs"]["Run_4_State_matrix"]  
    )
    ########################################## Normalizing ########################################################

    # Normalizes the standard run and saves without altering original matrix
    normalized_state_matrix=Generate_dataset.min_max_normalization(standard_state_matrix.copy())

    if isinstance(model, list):
        

        #   Gets the errormatrices for the three models
        multi_model_e_array=report_error_plotting(model_list=model, normalized_state_matrix=normalized_state_matrix, standard_state_matrix=standard_state_matrix, spec_vars=spec_vars, residual=True, number_of_states=601, start_state_index=0)        #iterate for the three models
        print(multi_model_e_array.shape)
        #print(multi_model_e_array[:,0])
        plot_state_vars(state_matrix=multi_model_e_array,  variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )
        return


    ######################################### Estimate using the model #########################################################
    
    # Estimates the normalized state-matrix
    normalized_states_estimated=estimated_model(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)


     #   Generate the single step model matrix
    norm_singlestep_estimations=generate_single_step_matrix(model=model, residual=residual, state_matrix=normalized_state_matrix,  number_of_states=601, start_state_index=0)

    ########################################## DE-Normalizing ########################################################
    
    # Denormalizes the estimated state matrix
    states_estimated=Generate_dataset.min_max_DEnormalization(normalized_states_estimated.copy())


    # Denormalizes the estimated state matrix
    singlestep_estimations=Generate_dataset.min_max_DEnormalization(norm_singlestep_estimations.copy())

    
    if spec_vars[0] in ["step", "acc"]:
        ######################################### Calculate errors #########################################################

            #   Gather error matrices and print some error values
        rel_est_error, error_matrix=generate_error_matrix(standard_state_matrix, normalized_state_matrix, states_estimated, normalized_states_estimated)

        
            #   Gather the single step error matrix

        step_error_vect, step_error_matrix= generate_single_step_error_matrix(standard_state_matrix, normalized_state_matrix, singlestep_estimations, norm_singlestep_estimations)
        
        '''# If wanting to show average error value for every 10(step_len) years
        er_mat=rel_error_matrix.copy()
        step_len=7
        portion=np.zeros((step_len,12))
        for k, state in enumerate(er_mat):
            idx=k%step_len
            portion[idx]=state
            
            if idx==(step_len-1):
                print("k=",k)
                portion=np.mean(portion, axis=0)
                print(portion)

                print(er_mat[ k-(step_len-1) :k, :])
                er_mat[ k-(step_len-1) :k, :] = portion
                portion=np.zeros((step_len,12))
        rel_error_matrix=er_mat '''
        

        #plot_state_vars(state_matrix=error_matrix,  variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )
        if spec_vars[0] =="step":
            plot_state_vars(est_matrix=step_error_matrix,)#  variables_included=["step_error"]) #, variables_included= ["nr", "ppol","sc"] )
            return
        if spec_vars[0] =="acc":
            plot_state_vars(est_matrix=error_matrix)#, variables_included=["std"])# variables_included=["step_error"]) #, variables_included= ["nr", "ppol","sc"] )
            return

    ######################################### Plot estimation or varied run #########################################################
    #Plot the state variables chosen, standard is "all"
    plot_state_vars(state_matrix=standard_state_matrix, est_matrix=states_estimated, variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )

    ##Plot the state variables chosen, standard is "all"
    #plot_state_vars(state_matrix=standard_state_matrix, est_matrix=alt_state_matrix, variables_included=spec_vars) #, variables_included= ["nr", "ppol","sc"] )
    
report_main()



#AlTERED: ax_.yaxis.set_major_locator(plt.MaxNLocator(3))#5 , "figsize" : (3.5, 2.5), 


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