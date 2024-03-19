from re import A
import matplotlib.pyplot as plt
import numpy as np
from world3_run_class import World3_run
from pyworld3 import World3, world3
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

class Generate_dataset(): 

    #Environment for generating world3-model-values - i.e creating our dataset

    def __init__(self, controllable=False, max_initval_variance_ppm=0, timespan=[1900,2200, 0.5], number_of_runs=1 ):
        self.controllable=controllable  #Might take in bool false or control function
        self.max_initval_variance_ppm=max_initval_variance_ppm  #takes in ppm(parts per million - 1/10^-6) nr of max variance of initvals
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
        ppms = np.random.random_integers(10e6-self.max_initval_variance_ppm, 10e6+self.max_initval_variance_ppm, augmented_init_state_array.shape)
        
        fractions=ppms/(10e6) #Convert ppm to fractions
        
        augmented_init_state_array=augmented_init_state_array*fractions #Augment the init values 

        augmented_init_state_dict=dict(zip(initial_values_dict.keys(), augmented_init_state_array)) #Adds the new values to similar dict

        return augmented_init_state_dict
        



############################################# Currently working with JSON #################################################
#Will likely switch this later for better efficiency - binary formats like NumPy's .npy or .npz formats, or HDF5, designed for efficient storage and retrieval of numerical data.
    def save_runs(self, file_name=None, norm=False):
        

        if norm==2:  #When called recorsively for normalizing
            add_norm_str="_normalized_"
        else:
            add_norm_str=""     #Added if not normalized

        directory="create_dataset/dataset_storage/"

        if file_name==None:
            file_name=f"dataset_runs_{self.number_of_runs}_variance_{self.max_initval_variance_ppm}"
        file_name=f"{file_name}{add_norm_str}"  #Append normalized or ""
        
        file_path_full=f"{directory}{file_name}.json"
        
        title=f"{add_norm_str} World3 runs from file {file_path_full}"   #Add for title

    
        dataset_params=self.parameters_dict()      #Dataset parameters

        data_runs=self.format_data()    #Fetch dict of state matrices

        if norm==2:
            for run_key, run_matrix in data_runs.items():   #each state_matrix is normalized 
                data_runs[run_key]=Generate_dataset.min_max_normalization(np.array(run_matrix)).tolist()   #sends matrix for normalizing
        
        dataset_dict={
            "Title": title ,
            "Parameters": dataset_params ,
            "Model_runs": data_runs
        }

        with open(file_path_full, "w") as json_file:
            json.dump(dataset_dict, json_file, indent=8)  # indent parameter for pretty formatting

        if norm==True:      ######### If norm true, it creates another save_run 
            self.save_runs(file_name=file_name, norm=2) #Runs again with new path and norm=0 for normalizing input

    def format_data(self):
        #Generate_dataset.fit_varnames(object.n)   #in case one wants to label the matrix elements

        #Create a dict for all runs
        formatted_data_of_runs={}
        
        for run_index, world3_object in enumerate(self.world3_objects_array):
            formatted_data_of_runs[f"Run_{run_index}_State_matrix"]=World3_run.generate_state_matrix(world3_object).tolist()
        return formatted_data_of_runs
 
        
    def parameters_dict(self):
        Dataset_parameters= {
            "timespan" : self.timespan ,
            "number_of_runs" : self.number_of_runs ,
            "max_initval_variance_ppm" :  self.max_initval_variance_ppm ,
            "controllable" : self.controllable }                    
              
        return Dataset_parameters



    def __str__(self):
        '''
        arguments=( "controllable:" + str(self.controllable) + "\nmax_initval_variance_ppm:" 
                + str(self.max_initval_variance_ppm) + "\ntimespan:" + str(self.timespan) 
                + "\nnumber_of_runs:" + str(self.number_of_runs)     )    #+ "\nworld3_objects_array:" )
                #+ str(self.world3_objects_array) + "\ninitial_values_dict:" + str(json.dumps(self.initial_values_dict,indent=2))  )
        '''
        arguments_dict={ 
                    "Dataset_parameters": {

                    "controllable:" : str(self.controllable) ,
                    "max_initval_variance_ppm" :  str(self.max_initval_variance_ppm) ,
                    "timespan," : str(self.timespan) ,
                    "number_of_runs:" : str(self.number_of_runs)   }    }
        
        #return "Dataset parameters:\n"+arguments

        return json.dumps(arguments_dict,indent=4)
        
    def fetch_dataset(filepath):    #General fetching of dataset from jsonfile
        
        #Fetch nr of runs and k_max from title in json file
        #Then fetch all matrices and append to a tensor
        with open(filepath, "r") as json_file:
            # returns JSON object as a dictionary
            dataset_dict = json.loads(json_file.read())
        '''
        title=dataset_dict["Title"]

        parameters_dict=dataset_dict["Parameters"]   #Dict of datasetparameters for saving nr of runs and timespan
        nr_of_runs=parameters_dict["number_of_runs"]   
        timespan=parameters_dict["timespan"] #List on form [start year, end year, step size]

        state_matrices_arraylist=dataset_dict["Model_runs"].values() #fetches a list of all state_arrays (nested lists)
        '''
        return dataset_dict 

############################################### Experimental normalization and heat maps ####################################
    def min_max_normalization(matrix):  #https://en.wikipedia.org/wiki/Normalization_(statistics)
        #Matrix must be np array

        #matrix - state matrix to be normalized based on min and max from basic run matrix
        #Fetch dict from constants file, take out "Standard Run MinMax" dict
        min_max_dict=Generate_dataset.fetch_dataset("create_dataset/constants_standards.json")["Standard Run MinMax"]
        min_max_list=list(min_max_dict.values())

        for i,var in enumerate(matrix.T):       
            min,max=min_max_list[i] #gets min and max for current variable in basic run
            scope=max-min
            diff=var-min
            #diff=matrix[:,i]-min
            #diff=matrix[:][i]-min  #IF Matrix is not array - doesnt work
            #matrix[:][i]=diff/scope
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