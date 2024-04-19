from re import A
import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables

from init_transition_matrix_old import A_state_transition_matrix #Brings transition matrix from default run


########################################### Create a world3-run ########################################

world3 = World3()
world3.set_world3_control()
world3.init_world3_constants(p1i=77e7) # p1i=67e7
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)


#np.shape(world3.nrfr)

######################################### Create the State Matrix X for this run  ########################################

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
    







states_estimated=estimated_model(X_state_matrix, A_state_transition_matrix)


error_matrix=X_state_matrix[:,:]-states_estimated

print("Error matrix: \n", error_matrix)

al_est_full=states_estimated[:,0]
pal_est_full=states_estimated[:,1]
uil_est_full=states_estimated[:,2]
lfert_est_full=states_estimated[:,3]
print("est shape full: ",lfert_est_full.shape)

plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert, al_est_full, pal_est_full, uil_est_full, lfert_est_full ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, 1.1*max(world3.al)], [0, 1.1*max(world3.pal)], [0, 1.1*max(world3.uil)], [0, 1.1*max(world3.lfert)], [0, 1.1*max(world3.al)], [0, 1.1*max(world3.pal)], [0, 1.3*max(world3.uil)], [0, 1.3*max(world3.lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test AR poli1_diff_15%",
)
plt.savefig("fig_world3_AR_test_poli1_diff_big.png")


