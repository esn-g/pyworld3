from re import A
import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables


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
    var_name_matrix[:,var_index]=variable_name+"_k="
    
for k in np.arange(world3.n):   #Go through rows and assign k value
    var_name_matrix[k,:]+=str(k)

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




#print("state array shape: ", X_state_matrix.shape,"\n state_array:\n", X_state_matrix)


#X_state_matrix_0=X_state_matrix[:(world3.n-1),:]  #Removes last row corresponding to kmax

############################# Create the theta-matrix (A) ########################################


#Calculate a row of the theta matrix via linalg.lstsq, least dquare method
def calculate_theta_row(var_index=0, row_of_state_var=np.array([]) , state_array= np.array([])):
    
    print(f"\n_____{var_str_list[var_index]}:_____\n") 
    print("var_row shape: ", row_of_state_var.shape)
    print("state_array_truncated shape: ", state_array.shape)
    
    theta, residuals, rank, s = np.linalg.lstsq(state_array, row_of_state_var, rcond=None)
    
    print("shape of theta: ", theta.shape, "\ntheta: ",theta)
    return theta

#Construct the State transitions matrix A by calling calculate_theta_row() for each state variable
def construct_A_matrix( state_array=np.array([]) ):

    A_matrix=np.empty((12,12), dtype=object)    #Initialize the A-matrix
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

print("A_state_transition_matrix.shape: ", A_state_transition_matrix.shape)
np.set_printoptions(precision=3, suppress=True)

print(np.get_printoptions())


print("A_matrix: \n\n",A_state_transition_matrix)
print(A_state_transition_matrix)


def next_state_estimate(current_state=np.array([]), transition_matrix=np.array([]) ):
    next_state=transition_matrix@current_state
    return next_state

#state_1=next_state_estimate(X_state_matrix[0,:], A_state_transition_matrix )
#print("\nreal state 1: \n", X_state_matrix[1,:] ,"\nestimated state_1: \n", state_1[:])


def estimated_model(state_matrix=np.array([]), transition_matrix=np.array([]), number_of_states=601, start_state_index=0):
    
    current_state=state_matrix[start_state_index,:]
    estimated_state_matrix=np.empty( (number_of_states, 12), dtype=object )
    
    for k in range(start_state_index, start_state_index+number_of_states):
        estimated_state_matrix[k,:]=current_state
    
        next_state=next_state_estimate(current_state, transition_matrix)

        current_state=next_state
        print("\ncurr state\n",current_state)

    estimated_state_matrix[number_of_states-1,:]=current_state

    return estimated_state_matrix
    

states_estimated=estimated_model(X_state_matrix, A_state_transition_matrix, 10, 0)

print("\nreal state 1-10: \n", X_state_matrix[0:10,:] ,"\nestimated state 1-10: \n", states_estimated)

error_matrix=X_state_matrix[0:10,:]-states_estimated

print("Error matrix: \n", error_matrix)




states_estimated=estimated_model(X_state_matrix, A_state_transition_matrix)

al_est_full=states_estimated[:,0]
pal_est_full=states_estimated[:,1]
uil_est_full=states_estimated[:,2]
lfert_est_full=states_estimated[:,3]
print("est shape full: ",lfert_est_full.shape)

plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert, al_est_full, pal_est_full, uil_est_full, lfert_est_full ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)], [0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test AR",
)
plt.savefig("fig_world3_AR_test_full.pdf")


'''
plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert, al_est, pal_est, uil_est, lfert_est ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)], [0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test AR",
)
plt.savefig("fig_world3_AR_test.pdf")

'''
    

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
nr


####################### plot for a set length of k
al_real=X_state_matrix[:10,0]
pal_real=X_state_matrix[:10,1]
uil_real=X_state_matrix[:10,2]
lfert_real=X_state_matrix[:10,3]

print("real shape: ",lfert_real.shape)


al_est=states_estimated[:,0]
pal_est=states_estimated[:,1]
uil_est=states_estimated[:,2]
lfert_est=states_estimated[:,3]
print("est shape: ",lfert_est.shape)

test_time=np.arange(0,5,0.5)

plot_world_variables(
    test_time,
    [ al_real, pal_real, uil_real, lfert_real, al_est, pal_est, uil_est, lfert_est ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)], [0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test AR",
)
plt.savefig("fig_world3_AR_test.png")


'''
