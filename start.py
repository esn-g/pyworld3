import platform
import matplotlib.pyplot as plt
import numpy as np

from pyworld3 import World3
from pyworld3.utils import plot_world_variables


world3 = World3()
world3.set_world3_control()
world3.init_world3_constants(nri=-1)
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
world3.nr

state_array_flat = np.array([
    world3.al,
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
    world3.nr
    ])
state_array = state_array_flat.T # state_array_tall, som det ska vara  
# print(state_array_flat.shape)         
# print(state_array.shape) # shape(rows, columns), #601 rader 12 kolumner. Korrekt.
theta_list = []
residuals = []
for i in state_array_flat: # tar en rad från den långa, som är en kolumn i dne vanliga
    i = i[1: ,np.newaxis]
    x, residual, _ , _ = np.linalg.lstsq(state_array[ : (world3.n-1)] , i , rcond=None)
    x = x.T
    print(x.shape)
    x = x.reshape(-1)
    # print(x.shape)
    
    theta_list.append(x)
    residuals.append(residual)

theta_array = np.array(theta_list)
# theta_array = theta_array.reshape(-2)
# residuals = np.array(residuals)
print(theta_array.shape)
print(theta_array)
# theta_
# print(residuals)
# print(residuals)


plot_world_variables(
    world3.time,
    [world3.nr, world3.iopc, world3.fpc, world3.pop, world3.ppolx],
    ["NRFR", "IOPC", "FPC", "POP", "PPOLX"],
    [[0,1.2*max(world3.nr)], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    figsize=(7, 5),
    grid=1,
    title="World3 standard run",
)
plt.show()