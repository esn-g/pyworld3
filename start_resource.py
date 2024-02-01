
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

rsc = Resource()
rsc.set_resource_table_functions()
rsc.init_resource_variables()
rsc.init_resource_constants()
rsc.set_resource_delay_functions()
rsc.init_exogenous_inputs()
rsc.run_resource()