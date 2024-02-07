
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
import matplotlib.pyplot as plt
from pyworld3 import Resource

from pyworld3.utils import plot_world_variables

rsc = Resource()
rsc.set_resource_table_functions('./my_modified_functions_table_world3.json')
rsc.set_resource_control()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()

plot_world_variables(
    rsc.time,
    [rsc.nrfr, rsc.iopc,  rsc.pop, rsc.nrur ],
    ["NRFR", "IOPC",  "POP", 'NRUR'],
    [[0, 1], [0, 1e3], [0, 16e9], [0, 1.5*max(rsc.nrur)], [0, 32]],
    img_background="./img/fig7-7.png",
    figsize=(7, 5),
    title="rsc run",
)

plt.show()

#print(rsc.nrfr)