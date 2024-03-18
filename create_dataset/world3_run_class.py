from re import A
import matplotlib.pyplot as plt
import numpy as np
from pyworld3 import World3
from pyworld3.utils import plot_world_variables


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


class World3_run:
    #Methods and class functions regarding running the model

    def __init__(self ):
        pass

    def run_model(init_values=False, controll=False): #Run the world3 model based on the given parameters

        if init_values==False and controll==False: #Basic run
            world3 = World3()
            world3.set_world3_control()
            world3.init_world3_constants()
            world3.init_world3_variables()
            world3.set_world3_table_functions()
            world3.set_world3_delay_functions()
            world3.run_world3(fast=False)
        elif init_values!=False:
            world3 = World3()
            world3.set_world3_control()
            world3.init_world3_constants(**init_values) #Pass the dict of all init values passed as keyword arguments
            world3.init_world3_variables()
            world3.set_world3_table_functions()
            world3.set_world3_delay_functions()
            world3.run_world3(fast=False)
        else:
            print("Within World3_run function run_model(): This functionality is not yet here")
        
        return world3
        

######################################### Create the State Matrix X  ########################################
    def generate_state_matrix(world3_obj): #Takes in a world3 object and returns its state-matrix
        X_state_matrix=np.array([world3_obj.al,
            world3_obj.pal,
            world3_obj.uil,
            world3_obj.lfert,
            world3_obj.ic,
            world3_obj.sc,
            world3_obj.ppol,
            world3_obj.p1,
            world3_obj.p2,
            world3_obj.p3,
            world3_obj.p4,
            world3_obj.nr]).T
        return X_state_matrix


    def generate_full_state_matrix():
        pass        #Matrix of all variables in world3 model - will be used to train NN later
        


            

    def fit_varnames( k_max):
        var_name_matrix=np.empty((k_max,12), dtype=object) #Init matrix
    
        for k in np.arange(k_max):   #Go through rows and assign k value and variable list
            var_name_matrix[k,:]=var_str_list
            var_name_matrix[k,:]+="["+str(k)+"]:"





    def __str__(self):
        return "World3 run"




######################################### Create the State Transition Matrix A  ########################################
    #Calculate a row of the theta matrix via linalg.lstsq, least dquare method
    def calculate_theta_row(var_index=0, row_of_state_var=np.array([]) , state_array= np.array([])):
        
        theta, residuals, rank, s = np.linalg.lstsq(state_array, row_of_state_var, rcond=None)
    
        return theta

    #Construct the State transitions matrix A by calling calculate_theta_row() for each state variable
    def construct_A_matrix(world3_obj, state_array=np.array([]) ):

        A_matrix=np.empty((12,12), dtype=float)    #Initialize the A-matrix
        #Goes through row by row of the Transposed state_array (Col by Col), means going through each variable one at a time
        for var_index, var_row in enumerate(state_array.T): 

            #Truncate the X matrix and state variable vector according to x1{1-kmax}=X{0-(kmax-1)}*theta_1
            var_row_truncated=var_row[ 1: ,np.newaxis]
            #print("var_row_truncated shape: ", var_row_truncated.shape)
            
            state_array_truncated=state_array[:(world3_obj.n-1), : ]
            #print("state_array_truncated shape: ", state_array_truncated.shape)

            theta_row=World3_run.calculate_theta_row(var_index, var_row_truncated, state_array_truncated)

            A_matrix[var_index,:]=theta_row.reshape(-1)
        return A_matrix




