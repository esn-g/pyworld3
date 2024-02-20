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

# order alphabetical, codeline


# agr states
world3.al 
world3.pal
world3.uil
world3.lfert
# cap stats
world3.ic
world3.sc
#pol states
world3.ppol
# pop states
world3.p1
world3.p2
world3.p3
world3.p4
# resource states
#print(world3.nr)


dim2=world3.pal.shape
state_array = np.array([world3.al,
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
dim=state_array.shape
# print(dim)
# print(dim2)


theta_list = []
residuals = []
for i in state_array:
    
    # print(i)
    # print(i.shape)
    i = i[:,np.newaxis] # från alejandro
    # print(i)
    # print(i.shape)
    x, residual, _ , _ = np.linalg.lstsq(state_array.T , i , rcond=None)
    # print(x.shape)
    x = x.reshape(-1)
    # print(x.shape)

    theta_list.append(x)
    residuals.append(residual)
    # residuals = np.concatenate(residuals, residual, axis=0 )

theta_array = np.array(theta_list)
# theta_array = theta_array.reshape(-2)
residuals = np.array(residuals)
print(theta_array.shape)
print(theta_array)
# print(residuals)
# print(residuals)

