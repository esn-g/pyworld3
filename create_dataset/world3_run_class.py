from re import A
import matplotlib.pyplot as plt
import numpy as np
from pyworld3 import World3
from pyworld3.utils import plot_world_variables
import json

#List of variable names for all variables examined
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
    #Environment for generating world3-model-values - i.e creating our dataset

    def __init__(self, init_values_mod=False, controllable=False, timespan=range(1900,2100), number_of_runs=1 ):
        self.init_values_mod=init_values_mod
        self.controllable=controllable
        self.timespan=timespan
        #self.var_name_matrix=var_str_list
        self.number_of_runs=number_of_runs
        self.world3_objects_array=np.empty(number_of_runs, dtype=object)

    def run_model(self): #Run the world3 model based on the given parameters
        if self.init_values_mod==False and self.controllable==False: #Basic run
            world3 = World3()
            world3.set_world3_control()
            world3.init_world3_constants()
            world3.init_world3_variables()
            world3.set_world3_table_functions()
            world3.set_world3_delay_functions()
            world3.run_world3(fast=False)
        
        return world3
        
    def generate_models(self):
        for run in range(self.number_of_runs):
            self.world3_objects_array[run]=self.run_model()

######################################### Create the State Matrix X  ########################################
    def generate_state_matrix(world3_obj):
        X_state_matrix=np.array([world3_obj.al,
            world3_obj.pal,
            world3_obj.uil,
            world3_obj.lfert,
            world3_obj.ic,
            world3_obj.sc,
            world3_obj.ppol,
            world3_obj.p1,
            world3_obj.p2,
            world3_obj.p3,
            world3_obj.p4,
            world3_obj.nr]).T
        return X_state_matrix

            

    def fit_varnames( k_max):
        var_name_matrix=np.empty((k_max,12), dtype=object) #Init matrix
    
        for k in np.arange(k_max):   #Go through rows and assign k value and variable list
            var_name_matrix[k,:]=var_str_list
            var_name_matrix[k,:]+="["+str(k)+"]:"



#############################################Currently working with JSON#################################################
#Will likely switch this later for better efficiency - binary formats like NumPy's .npy or .npz formats, or HDF5, designed for efficient storage and retrieval of numerical data.
    def save_run(self, current_run, file_path):
        
        data_runs=[World3_run.format_data(run_nr, w3_object) for run_nr, w3_object in enumerate(self.world3_objects_array)]

        with open(file_path, "w") as json_file:
            json.dump(current_run, json_file, indent=4)  # indent parameter for pretty formatting


    def format_data(run, object):
        World3_run.fit_varnames(object.n)

        formatted_data={
            "Run_index":run,
            "Time_span":[object.year_min ,object.year_max],
            "K_max": object.n,
            "State_matrix": World3_run.generate_state_matrix(object).tolist()
            }
        return formatted_data







