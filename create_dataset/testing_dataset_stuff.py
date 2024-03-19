from re import A
import matplotlib.pyplot as plt
import numpy as np
from pyworld3 import World3
from pyworld3.utils import plot_world_variables
from world3_run_class import World3_run
from generate_dataset_classfile import Generate_dataset

#al=Generate_dataset.fetch_dataset("create_dataset/dataset_storage/dataset_runs_2_variance_1.json")["Model_runs"]["Run_0_State_matrix"][0][:]
#al_norm=Generate_dataset.fetch_dataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json")["Model_runs"]["Run_0_State_matrix"][0][:]

dict=Generate_dataset.fetch_dataset("create_dataset/dataset_storage/dataset_runs_2_variance_1.json") #fetches a list of all state_arrays (nested lists)
dict_norm=Generate_dataset.fetch_dataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json")

dict_runs=dict["Model_runs"]
dict_runs_norm=dict_norm["Model_runs"]

run=dict_runs["Run_0_State_matrix"]
run_norm=dict_runs_norm["Run_0_State_matrix"]
#al=np.array(al)
#al_norm=np.array(al_norm)

run=np.array(run)
run_norm=np.array(run_norm)

print(run.shape,run_norm.shape)

al=run[:,0]
al_norm=run_norm[:,0]

#al=run[0][:]
#al_norm=run_norm[0][:]

#al=np.array(al)
#al_norm=np.array(al_norm)

print(al.shape,al_norm.shape)

test_time=np.arange(0,300+.5,0.5)
plot_world_variables(
    test_time,
    [ al, al_norm],
    [ "AL", "AL_NORM"],
    [[0, max(al)], [0, max(al_norm)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test norm",
)
plt.show()

#model=World3_run.run_model()
#orignal_model=World3_run.generate_state_matrix(model)
#print(World3_run.generate_state_matrix(model))
#print(World3_run.generate_state_matrix(modded_model))
#print(World3_run.generate_state_matrix(modded_model).tolist())



#create_dataset/dataset_storage/dataset_runs_1_variance_1.json




'''
plot_world_variables(
    world3.time,
    [world3.al, world3.pal, world3.uil, world3.lfert, al_est, pal_est, uil_est, lfert_est ],
    [ "AL", "PAL", "UIL","LFERT", "al_est", "pal_est", "uil_est", "lfert_est"],
    [[0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)], [0, max(world3.al)], [0, max(world3.pal)], [0, max(world3.uil)], [0, max(world3.lfert)] ],
    img_background=None,
    figsize=(7, 5),
    title="Test AR",
)
plt.savefig("fig_world3_AR_test.pdf")

'''