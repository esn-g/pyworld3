
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

     ## från resourceklass
        Attributes
    ----------
    nri : float, optional
        nonrenewable resources initial [resource units]. The default is 1e12.
    nr : numpy.ndarray
        nonrenewable resources [resource units]. It is a state variable.
    nrfr : numpy.ndarray                                                        # nrfr output
        nonrenewable resource fraction remaining [].
    nruf : numpy.ndarray
        nonrenewable resource usage factor [].
    nrur : numpy.ndarray
        nonrenewable resource usage rate [resource units/year].                 
    pcrum : numpy.ndarray                                                       # pcrum  y values from JSON-file
        per capita resource usage multiplier [resource units/person-year].
    fcaor : numpy.ndarray                                                       # fcaor, y values from JSON-filen      
        fraction of capital allocated to obtaining resources [].

    **Control signals**
    nruf_control : function, optional
        nruf, control function with argument time [years]. The default is 1.
    fcaor_control : function, optional
        fraction of normal fcaor used, control function with argument time [years]. The default is 1.0
'''
import matplotlib.pyplot as plt
from pyworld3 import Resource

from pyworld3.utils import plot_world_variables


rsc = Resource()

# tre argument - feedbackkontroll, ett argument, open loop control
def fcaor_control(t, rsc, k):
    return rsc.pcrum[k]

rsc.set_resource_table_functions()                # './my_modified_functions_table_world3.json'
rsc.set_resource_control()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.set_resource_control(fcaor_control = fcaor_control)
rsc.run_resource()

plot_world_variables(
    rsc.time,
    [rsc.nrfr, rsc.iopc,  rsc.pop, rsc.nrur, rsc.fcaor],
    ["NRFR", "IOPC",  "POP", 'NRUR','FCAOR', 'NRI'],
    [[0, 1], [0, 1e3], [0, 16e9], [0, 1.5*max(rsc.nrur)], [0, 1.2*max(rsc.fcaor)]],  ##MAX  
    img_background="./img/fig7-7.png",
    figsize=(7, 5),
    title="rsc run, pcrum feedback control",
)

plt.show()

# print(rsc.nrfr)