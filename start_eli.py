from re import A
import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables


world3 = World3()
world3.set_world3_control()
world3.init_world3_constants()
world3.init_world3_variables()
world3.set_world3_table_functions()
world3.set_world3_delay_functions()
world3.run_world3(fast=False)

#np.shape(world3.nrfr)

state_variables=np.array([world3.al,
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
    world3.nr])

var_str_list=["al",
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
    "nr"]


state_array=state_variables

state_array_k=state_variables[:, 1: ]

state_array_prev=state_variables[:, : (world3.n-1)]


def calculate_theta_row(var_index=0, var_array=np.array([]) ):
    var_array_k=var_array[ 1: ,np.newaxis]
    var_array_prev=var_array[ : (world3.n-1) ]
    print(f"\n_____{var_str_list[var_index]}:_____\n")
    print(state_array_prev.shape)
    print(var_array_k.shape)
    # print(var_array_k)
    theta = np.linalg.lstsq(state_array_prev.T, var_array_k, rcond=None)
    print(theta)
    # theta=theta.reshape(-1)
    # print(theta.shape)

def construct_A_matrix( array_of_states=np.array([]) ):
    
    for var_index, var_array in enumerate(array_of_states):
        calculate_theta_row(var_index, var_array )


construct_A_matrix(state_array)

    


#for i in state_variables:
    

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






'''
