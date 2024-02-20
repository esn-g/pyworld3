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

state_array_k=state_variables[:, : (world3.n-1)]

state_array_prev=state_variables[:, 1: ]



print(state_array_prev[0,0:2])

print(state_array_k[0,0:2])


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
