from pyworld3 import World3

world3 = World3()                    # choose the time limits and step.
world3.set_world3_control()          # choose your controls
world3.init_world3_constants()       # choose the model constants.
world3.init_world3_variables()       # initialize all variables.
world3.set_world3_table_functions()  # get tables from a json file.
world3.set_world3_delay_functions()  # initialize delay functions.
world3.run_world3()

# plottar ej
