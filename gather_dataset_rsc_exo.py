from pyworld3 import Resource
from pyworld3 import World3


from pyworld3.utils import plot_world_variables

import matplotlib.pyplot as plt


############################################# Perform rsc and world3 runs ####################################################### 

rsc = Resource(2000,2010)
rsc.set_resource_control()
rsc.set_resource_table_functions()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()

state_vars=[rsc.nr , rsc.nrfr , rsc.nruf , rsc.nrur , rsc.pcrum, rsc.fcaor]
#print(rsc.time)
#print(length(state_vars))

for k, year in enumerate(rsc.time):
    current_state=[]
    for i in state_vars:
        current_state.append(i[k])
        #print[i[k]]
    
    print(f"k={k},year={year},current_state:nr{current_state[0]},nrfr{current_state[1]},nruf{current_state[2]},nrur{current_state[3]},pcrum{current_state[4]},fcaor{current_state[5]}")
    print(rsc.nr[k])

#for i in current_state_vars_list:
#    print()

'''
#plt.figure()
#Plots resource variables from the exogenous resource-run
plot_world_variables(
    rsc.time,
    [ rsc.nr , rsc.nrfr , rsc.nruf , rsc.nrur , rsc.pcrum, rsc.fcaor],
    [ "rsc_NR" , "rsc_NRFR" , "rsc_NRUF" , "rsc_NRUR" , "rsc_PCRUM", "rsc_FCAOR"],
    [ [0, 1.2*max(rsc.nr)], [0, 1.2*max(rsc.nrfr)], [0, 1.2*max(rsc.nruf)], [0, 1.2*max(rsc.nrur)], [0, 1.2*max(rsc.pcrum)], [0, 1.2*max(rsc.fcaor)]],
    img_background=None,
    figsize=(14, 8),
    title="Resource Test",
    grid=True
)
plt.savefig("test_figures/rsc_exo_run_test")

'''


''' 
with open("dataset_rsc_exo.txt","w") as file:
    current_state_vars_list=[rsc.nr , rsc.nrfr , rsc.nruf , rsc.nrur , rsc.pcrum, rsc.fcaor]



'''