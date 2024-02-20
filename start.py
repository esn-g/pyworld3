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

statelist = [world3.al,
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
    world3.nr]
dim2=world3.pal.shape
state_array = np.array(statelist)
dim=state_array.shape
print(dim)
print(dim2)


for i in statelist:
    
    print(i)
    print(i.shape)
    i = i[:,np.newaxis]
    print(i)
    print(i.shape)
    x = np.linalg.lstsq(state_array,i)
    print(x)