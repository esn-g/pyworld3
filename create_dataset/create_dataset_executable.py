from re import A

from generate_dataset_classfile import Generate_dataset

datasetting=Generate_dataset(max_initval_variance_ppm=0, number_of_runs=10)
datasetting.generate_models()


#modded_model=datasetting.world3_objects_array[0]



#model=World3_run.run_model()
#orignal_model=World3_run.generate_state_matrix(model)
#print(World3_run.generate_state_matrix(model))
#print(World3_run.generate_state_matrix(modded_model))
#print(World3_run.generate_state_matrix(modded_model).tolist())

datasetting.save_runs(norm=True)

print(datasetting)

#print(modded_model.hsid)
#create_dataset/dataset_storage/dataset_runs_1_variance_1.json