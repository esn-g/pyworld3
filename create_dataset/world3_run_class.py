from re import A
import matplotlib.pyplot as plt
import numpy as np
from pyworld3 import World3
from pyworld3.utils import plot_world_variables

var_str_list=np.array(["al",
    "pal",
    "uil",
    "lfert",
    "ic",
    "sc",
    "ppol",
    "p1",
    "p2",
    "p3",
    "p4",
    "nr"]).T       #Transposes it to get each variable as its own column - X matrix but names

class World3_run:
    def __init__(self, init_values_mod=False, controllable=False, timespan=range(1900,2100), number_of_runs=1 ):
        self.init_values_mod=init_values_mod
        self.controllable=controllable
        self.timespan=timespan
        self.var_name_matrix=var_str_list
        self.k_max=1
        self.number_of_runs=number_of_runs

    def run_model(self): #Run the world3 model based on the given parameters
        if self.init_values_mod==False and self.controllable==False: #Basic run
            world3 = World3()
            world3.set_world3_control()
            world3.init_world3_constants()
            world3.init_world3_variables()
            world3.set_world3_table_functions()
            world3.set_world3_delay_functions()
            world3.run_world3(fast=False)
        

        self.k_max=world3.n


    def generate_varname_array(self, k_max):
        #Transposes it to get each variable as its own column - X matrix but names
        var_str_list=np.array(["al","pal","uil","lfert","ic","sc","ppol","p1","p2","p3","p4","nr"]).T   
        var_name_matrix=np.empty((k_max,12), dtype=object) #Init matrix
        for var_index, variable_name in enumerate(var_str_list):  #Go through variable names and add to matrix columns
            #print(var_index, variable_name)
            var_name_matrix[:,var_index]=variable_name+"_k="
    
        for k in np.arange(k_max):   #Go through rows and assign k value
            var_name_matrix[k,:]+=str(k)
        self.var_name_matrix=var_name_matrix

    





