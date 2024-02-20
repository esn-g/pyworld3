'''
    ####
    from resource class
    ####

    Nonrenewable Resource sector. Can be run independantly from other sectors
    with exogenous inputs. The initial code is defined p.405.

    Examples
    --------
    Running the nonerenewable resource sector alone requires artificial
    (exogenous) inputs which should be provided by the other sectors. Start
    from the following example:

    >>> rsc = Resource()
    >>> rsc.set_resource_table_functions()
    >>> rsc.init_resource_variables()
    >>> rsc.init_resource_constants()
    >>> rsc.set_resource_delay_functions()
    >>> rsc.init_exogenous_inputs()
    >>> rsc.run_resource()
'''

from pyworld3 import Resource
from pyworld3 import World3


from pyworld3.utils import plot_world_variables

import matplotlib.pyplot as plt




############################################# Perform rsc and world3 runs ####################################################### 

rsc = Resource()
rsc.set_resource_control()
rsc.set_resource_table_functions()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()


world3 = World3()                    # choose the time limits and step.
world3.set_world3_control()          # choose your controls
world3.init_world3_constants()       # choose the model constants.
world3.init_world3_variables()       # initialize all variables.
world3.set_world3_table_functions()  # get tables from a json file.
world3.set_world3_delay_functions()  # initialize delay functions.
world3.run_world3()

############################################ Create plots for all resource-parameters ######################################################## 

#Plot resource-variables from the world3-run
plot_world_variables(
    world3.time,
    [world3.nr , world3.nrfr , world3.nruf , world3.nrur , world3.pcrum, world3.fcaor],
    ["w3_NR" , "w3_NRFR" , "w3_NRUF" , "w3_NRUR" , "w3_PCRUM", "w3_FCAOR"],
    [ [0, 1.2*max(rsc.nr)], [0, 1.2*max(rsc.nrfr)], [0, 1.2*max(rsc.nruf)], [0, 1.2*max(rsc.nrur)], [0, 1.2*max(rsc.pcrum)], [0, 1.2*max(rsc.fcaor)]],
    img_background=None,
    figsize=(14, 8),
    title="Resource Test",
    grid=True
)

plt.savefig("test_figures/rsc_w3_run_test")


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
#plt.show(block=True)

############################################# Plots fcaor and nrfr ####################################################### 
#fcaor: fraction of capital allocated to obtaining resources
#nrfr: nonrenewable resource fraction remaining

#Examine relationship between fcor and nrfr

plot_world_variables(
    rsc.time,
    [rsc.fcaor, rsc.nrfr],
    ["FCAOR", "NRFR"],
    [[0, max(rsc.fcaor)], [0, max(rsc.nrfr)]],
    img_background=None,
    figsize=(7, 5),
    title="FCAOR and NRFR rsc exo run",
    grid=True
)
plt.savefig("test_figures/fcaor_nrfr_rsc_exo")

#Examine relationship between fcor and nrfr
plot_world_variables(
    world3.time,
    [world3.fcaor, world3.nrfr],
    ["FCAOR", "NRFR"],
    [[0, max(world3.fcaor)], [0, max(world3.nrfr)]],
    img_background=None,
    figsize=(7, 5),
    title="FCAOR and NRFR world3 run",
    grid=True
)
plt.savefig("test_figures/fcaor_nrfr_w3_run")

####################################### Plot NRFR/FCOR ############################################################# 

#rsc run
plot_world_variables(
    rsc.time,
    [rsc.fcaor/rsc.nrfr],
    ["FCAOR/NRFR"],
    [[0, max(rsc.fcaor/rsc.nrfr)]],
    img_background=None,
    figsize=(7, 5),
    title="FCAOR/NRFR rsc exo run",
    grid=True
)

plt.savefig("test_figures/fcaor_over_nrfr_rsc_exo")

'''
Try to run in a different way
plot_world_variables(
    rsc.nrfr,
    [rsc.fcaor],
    ["FCAOR/NRFR"],
    [[0, max(rsc.fcaor/rsc.nrfr)]],
    img_background=None,
    figsize=(7, 5),
    title="FCAOR/NRFR rsc exo run",
    grid=True
)
'''


#Examine relationship between fcor and nrfr
#world3-run
plot_world_variables(
    world3.time,
    [world3.fcaor/world3.nrfr],
    ["FCAOR/NRFR"],
    [[0, max(world3.fcaor/world3.nrfr)]],
    img_background=None,
    figsize=(7, 5),
    title="FCAOR/NRFR world3 run",
)
plt.savefig("test_figures/fcaor_over_nrfr_w3_run")

############################################################################################################ 


'''
#Try to plot all variables from both runs simultaneously, Gives error, likely due to the rsc.time=!world3.time
plot_world_variables(
    world3.time,
    [world3.nr , rsc.nr , world3.nrfr  , rsc.nrfr , world3.nruf  , rsc.nruf , world3.nrur  , rsc.nrur , world3.pcrum , rsc.pcrum, world3.fcaor , rsc.fcaor],
    ["w3_NR" ,  "rsc_NR" , "w3_NRFR"  , "rsc_NRFR" , "w3_NRUF"  , "rsc_NRUF" , "w3_NRUR"  , "rsc_NRUR" , "w3_PCRUM" , "rsc_PCRUM", "w3_FCAOR" , "rsc_FCAOR"],
    [[0, 1.2*max(rsc.nr)], [0, 1.2*max(rsc.nr)], [0, 1.2*max(rsc.nrfr)] , [0, 1.2*max(rsc.nrfr)], [0, 1.2*max(rsc.nruf)] , [0, 1.2*max(rsc.nruf)], [0, 1.2*max(rsc.nrur)] , [0, 1.2*max(rsc.nrur)], [0, 1.2*max(rsc.pcrum)] , [0, 1.2*max(rsc.pcrum)], [0, 1.2*max(rsc.fcaor)] , [0, 1.2*max(rsc.fcaor)]],
    img_background=None,
    figsize=(12, 8),
    title="Resource Test: full World3- and exogenous- run comparison",
    grid=True
)
'''

#plt.show()

'''
plot_world_variables(
    rsc.time,
    [rsc.nrfr, rsc.iopc, rsc.pop],
    ["NRFR", "IOPC", "POP"],
    [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    img_background="./img/fig7-7.png",
    figsize=(7, 5),
    title="Resource Test",
)
plt.show(block=True)
'''

'''
plot_world_variables(
    rsc.time,
    [rsc.nri, rsc.nr, rsc.nrfr, rsc.nruf, rsc.nrur],
    ["NRI", "NR", "NRFR", "NRUF", "NRUR"],
    [[0, 1], [0, 1e3], [0, 1e3], [0, 16e9], [0, 32]],
    img_background="./img/fig7-7.png",
    figsize=(7, 5),
    title="World3 control run - General",
)
'''


'''
Attributes:
nri
nr 
nrfr 
nruf 
nrur 
pcrum
fcaor

Control signals:
nruf_control 
fcaor_control


Plotting json fraction relations example pcrum of iopc:
iopc= linspace(0,1600)
plt.plot(iopc, rsc.pcrum_f(iopc))
pl.show()


'''

'''Fcaor nrfr find relation and why'''