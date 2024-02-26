from re import A
import matplotlib.pyplot as plt
import numpy as np
import sys

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

np.set_printoptions(precision=3, suppress=False, linewidth=150, threshold=sys.maxsize)
#unicode_theta="\u03B8"

########################################### Create a world3-run ########################################

world3 = World3()
world3.set_world3_control()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

#np.shape(world3.nrfr)


########################################### Create a matrix of variable names ########################################

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



var_name_matrix=np.empty((world3.n,12), dtype=object) #Init matrix

for var_index, variable_name in enumerate(var_str_list):  #Go through variable names and add to matrix columns
    #print(var_index, variable_name)
    var_name_matrix[:,var_index]=variable_name+"["
    
for k in np.arange(world3.n):   #Go through rows and assign k value
    var_name_matrix[k,:]+=str(k)+"]:"

#print(var_name_matrix)    
######################################### Create the State Matrix X  ########################################

X_state_matrix=np.array([world3.al,
    world3.pal,
    world3.uil,
    world3.lfert,
    world3.ic,
    world3.sc,
    world3.ppol,
    world3.p1,
    world3.p2,
    world3.p3,
    world3.p4,
    world3.nr]).T




############################# Create the theta-matrix (A) ########################################


#Calculate a row of the theta matrix via linalg.lstsq, least dquare method
def calculate_theta_row(var_index=0, row_of_state_var=np.array([]) , state_array= np.array([])):
    
    theta, residuals, rank, s = np.linalg.lstsq(state_array, row_of_state_var, rcond=None)
    
    
    return theta

#Construct the State transitions matrix A by calling calculate_theta_row() for each state variable
def construct_A_matrix( state_array=np.array([]) ):

    A_matrix=np.empty((12,12), dtype=float)    #Initialize the A-matrix
    #Goes through row by row of the Transposed state_array (Col by Col), means going through each variable one at a time
    for var_index, var_row in enumerate(state_array.T): 

        #Truncate the X matrix and state variable vector according to x1{1-kmax}=X{0-(kmax-1)}*theta_1
        var_row_truncated=var_row[ 1: ,np.newaxis]
        #print("var_row_truncated shape: ", var_row_truncated.shape)
        
        state_array_truncated=state_array[:(world3.n-1), : ]
        #print("state_array_truncated shape: ", state_array_truncated.shape)

        theta_row=calculate_theta_row(var_index, var_row_truncated, state_array_truncated)

        A_matrix[var_index,:]=theta_row.reshape(-1)
    return A_matrix



A_state_transition_matrix=construct_A_matrix(X_state_matrix)

#print(var_name_matrix[205,1],X_state_matrix[205,1])
#print(var_name_matrix)
#print("\n\n\n\n\n",var_name_matrix[0:600:100,:], "\n\n\n\n\n")
#print("\n\n\n\n\n",A_state_transition_matrix[:,:], "\n\n\n\n\n")
print("\n\n\n\n\n",A_state_transition_matrix, "\n\n\n\n\n")

