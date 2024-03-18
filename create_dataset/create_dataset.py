from re import A
import matplotlib.pyplot as plt
import numpy as np
from pyworld3 import World3
from pyworld3.utils import plot_world_variables
from world3_run_class import World3_run
from dataset_class import Dataset_class

datasetting=Dataset_class(max_initval_variance_percent=1, number_of_runs=2 )
datasetting.generate_models()
modded_model=datasetting.world3_objects_array[0]

model=World3_run.run_model()
#print(World3_run.generate_state_matrix(model))
#print(World3_run.generate_state_matrix(modded_model))
#print(World3_run.generate_state_matrix(modded_model).tolist())

datasetting.save_runs()

print(datasetting)


#create_dataset/dataset_storage/dataset_runs_1_variance_1.json