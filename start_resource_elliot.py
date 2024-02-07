

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


from pyworld3.utils import plot_world_variables

import matplotlib.pyplot as plt


rsc = Resource()
rsc.set_resource_control()
rsc.set_resource_table_functions()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()


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
nri

nr 

nrfr 

nruf 

nrur 

pcrum

fcaor
'''

'''Fcaour nrfr find relation and why'''