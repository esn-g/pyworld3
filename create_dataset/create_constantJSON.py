from re import A
import matplotlib.pyplot as plt
import numpy as np
from world3_run_class import World3_run
from pyworld3 import World3
from pyworld3.utils import plot_world_variables
import json

initial_values_dict={

    "p1i" : 65e7 ,
    "p2i" : 70e7 ,
    "p3i" : 19e7 ,
    "p4i" : 6e7 ,
    "dcfsn" : 4 ,
    "fcest" : 4000 ,
    "hsid" : 20 ,
    "ieat" : 3 ,
    "len" : 28 ,
    "lpd" : 20 ,
    "mtfn" : 12 ,
    "pet" : 4000 ,
    "rlt" : 30 ,
    "sad" : 20 ,
    "zpgt" : 4000 ,
    "ici" : 2.1e11 ,
    "sci" : 1.44e11 ,
    "iet" : 4000 ,
    "iopcd" : 400 ,
    "lfpf" : 0.75 ,
    "lufdt" : 2 ,
    "ali" : 0.9e9 ,
    "pali" : 2.3e9 ,
    "lfh" : 0.7 ,
    "palt" : 3.2e9 ,
    "pl" : 0.1 ,
    "io70" : 7.9e11 ,
    "sd" : 0.07 ,
    "uili" : 8.2e6 ,
    "alln" : 6000 ,
    "uildt" : 10 ,
    "lferti" : 600 ,
    "ilf" : 600 ,
    "fspd" : 2 ,
    "sfpc" : 230 ,
    "ppoli" : 2.5e7 ,
    "ppol70" : 1.36e8 ,
    "ahl70" : 1.5 ,
    "amti" : 1 ,
    "imti" : 10 ,
    "imef" : 0.1 ,
    "fipm" : 0.001 ,
    "frpm" : 0.02 ,
    "nri" : 1e12 }

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


def save_data(dict, file_path=None ):
    if file_path==None:
        file_path=f"create_dataset/constants_standards.json"



        with open(file_path, "w") as json_file:
            json.dump(dict, json_file, indent=4)  # indent parameter for pretty formatting

def format_state_matrix(object):
    formatted_data=World3_run.generate_state_matrix(object).tolist()  
    return formatted_data



def create_extremes_dict(matrix):
    extremes_dict = { }
    for i, var_col in enumerate(matrix.T):
        label = var_str_list[i]
        min = np.min(var_col)
        max = np.max(var_col)

        extremes_dict[f"{label}_extremes"] = [min, max]
    return extremes_dict

def main():
    standard_run = World3_run.run_model()
    formatted_standard_run = format_state_matrix(standard_run)
    # print(formatted_standard_run)
    state_matrix = World3_run.generate_state_matrix(standard_run)
    extremes_dict = create_extremes_dict(state_matrix)
    
    standard_run_dict = {
        'Standard Run MinMax' : extremes_dict,
        'Standard Run Initial Values' : initial_values_dict,
        'Standard Run' : formatted_standard_run,
    }
    save_data(standard_run_dict)

    

main()
