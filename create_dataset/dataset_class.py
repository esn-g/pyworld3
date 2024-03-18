from re import A
import matplotlib.pyplot as plt
import numpy as np
from world3_run_class import World3_run
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

class Dataset_class(): 

    #Environment for generating world3-model-values - i.e creating our dataset

    def __init__(self, controllable=False, max_initval_variance_percent=0, timespan=[1900,2100, 0.5], number_of_runs=1 ):
        self.controllable=controllable  #Might take in bool false or control function
        self.max_initval_variance_percent=max_initval_variance_percent  #takes in percentage nr of max variance of initvals
        self.timespan=timespan
        self.number_of_runs=number_of_runs
        self.world3_objects_array=np.empty(number_of_runs, dtype=object)
        self.initial_values_dict=initial_values_dict


        
    def generate_models(self):
        #For now this is stupid in case youre doing base run, because then you dont need to pass anything to run_model()
        for run in range(self.number_of_runs):
            initial_vals=self.randomize_init_state()
            self.world3_objects_array[run]=World3_run.run_model(init_values=initial_vals)
        



    def randomize_init_state(self): #Generate initial values based on initiation of class instance

        augmented_init_state_array=np.array(list(self.initial_values_dict.values())) #Takes original init values into array

        #Creates a random variance for each respective variable within the range set in the class initiation 
        percentages = np.random.random_integers(100-self.max_initval_variance_percent, 100+self.max_initval_variance_percent, augmented_init_state_array.shape)
        
        fractions=percentages/100 #Convert percentage to fractions
        
        augmented_init_state_array=augmented_init_state_array*fractions #Augment the init values 

        augmented_init_state_dict=dict(zip(initial_values_dict.keys(), augmented_init_state_array)) #Adds the new values to similar dict

        return augmented_init_state_dict
        



############################################# Currently working with JSON #################################################
#Will likely switch this later for better efficiency - binary formats like NumPy's .npy or .npz formats, or HDF5, designed for efficient storage and retrieval of numerical data.
    def save_runs(self, file_path=None):
        if file_path==None:
            file_path=f"create_dataset/dataset_storage/dataset_runs_{self.number_of_runs}_variance_{self.max_initval_variance_percent}.json"

        #dataset_params=self.__str__()    #Add for title, currently causing issues because str doesnt return dict, create a seperate method for this
        
        data_runs=[self.format_data(run_nr, w3_object) for run_nr, w3_object in enumerate(self.world3_objects_array)]
        
        with open(file_path, "w") as json_file:
            #json.dump(dataset_params, json_file, indent=4)  # indent parameter for pretty formatting   #TITLE
            json.dump(data_runs, json_file, indent=4)  # indent parameter for pretty formatting



    def format_data(self,run, object):
        #Dataset_class.fit_varnames(object.n)   #in case one wants to label the matrix elements

        formatted_data={
            "Run_index":run,
            #"Time_span":[object.year_min ,object.year_max],
            #"K_max": object.n,
            #"Max_init_variance": self.max_initval_variance_percent,
            "State_matrix": World3_run.generate_state_matrix(object).tolist()
            }
        return formatted_data


    def __str__(self):
        '''
        arguments=( "controllable:" + str(self.controllable) + "\nmax_initval_variance_percent:" 
                + str(self.max_initval_variance_percent) + "\ntimespan:" + str(self.timespan) 
                + "\nnumber_of_runs:" + str(self.number_of_runs)     )    #+ "\nworld3_objects_array:" )
                #+ str(self.world3_objects_array) + "\ninitial_values_dict:" + str(json.dumps(self.initial_values_dict,indent=2))  )
        '''
        arguments_dict={ 
                    "Dataset_parameters": {

                    "controllable:" : str(self.controllable) ,
                    "max_initval_variance_percent" :  str(self.max_initval_variance_percent) ,
                    "timespan," : str(self.timespan) ,
                    "number_of_runs:" : str(self.number_of_runs)   }    }
        
        #return "Dataset parameters:\n"+arguments

        return json.dumps(arguments_dict,indent=4)
        


############################################### Experimental normalization and heat maps ####################################
    def test_normalization(matrix, orig_matrix):  #https://en.wikipedia.org/wiki/Normalization_(statistics)
        for i,var in enumerate(matrix.T):
            scope=max(orig_matrix[:,i])-min(orig_matrix[:,i])
            diff=var-min(orig_matrix[:,i])
            matrix[:,i]=diff/scope
        return matrix
    
    '''
    X=World3_run.generate_state_matrix(runs[0])
normalized_x=test_normalization(X,X)
#normalized_A=test_normalization(A_state_transition_matrix,A_state_transition_matrix)

# Create the heat map using Matplotlib
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.imshow(normalized_x, cmap='viridis', interpolation='nearest', aspect="auto")

# Add a color bar for reference
plt.colorbar()

# Customize axis labels and title
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('12x12 Heat Map')

# Show the plot
plt.show()
    '''