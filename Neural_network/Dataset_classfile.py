


import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import json

#from symbol import parameters

class CustomDataset(Dataset):
    #init might be made into a fetch json function
    def __init__(self, fetch_data_path="create_dataset/dataset_storage/dataset_runs_1_variance_1.json"):
        
        #Fetch nr of runs and k_max from title in json file
        #Then fetch all matrices and append to a tensor
        with open(fetch_data_path, "r") as json_file:
            # returns JSON object as a dictionary
            dataset_dict = json.loads(json_file.read())

        self.title=dataset_dict["Title"]

        parameters_dict=dataset_dict["Parameters"]   #Dict of datasetparameters for saving nr of runs and timespan
        self.nr_of_runs=parameters_dict["number_of_runs"]   
        self.timespan=parameters_dict["timespan"] #List on form [start year, end year, step size]
        self.k_max=((self.timespan[1]-self.timespan[0])/self.timespan[2])+1 #List on form [start year, end year, step size]

        state_matrices_arraylist=list(dataset_dict["Model_runs"].values()) #fetches a list of all state_arrays (nested lists)

        self.state_matrices_dataset=torch.tensor(state_matrices_arraylist) #Makes each statematrix (list) into tensors and saves in a tensor 
        print("data shape tensoer",self.state_matrices_dataset.shape)

    def __len__(self):

        nr_of_state_vectors=torch.numel(self.state_matrices_dataset[:,:,0]) #numel returns amount of objects
        nr_of_state_vectors=nr_of_state_vectors-self.nr_of_runs
        return nr_of_state_vectors

    def __getitem__(self, index):
        #want to fetch the state at a given k and k+1 at a given run - index refers to the n:th state-vector
        #Counting from 0 up to (k-max * nr_of_state_matrices) 
        #The state matrices are already made into tensors in init therefore dont need to convert, simpy fetch k and k+1

        # Convert state matrix and label to tensors if necessary
        #state_matrix_tensor = torch.tensor(state_matrix, dtype=torch.float32)
        #print("Sample_index: ", index) if index % 200 == 0 else None


        
        quotient, remainder = divmod(index, self.k_max)
        quotient=int(quotient)
        remainder=int(remainder)
        #print(quotient, remainder )

        if remainder==600:
            quotient+=1
            remainder=0

        input_state=self.state_matrices_dataset[quotient,remainder]
        #print(input_state)

        # label_tensor = torch.tensor(label, dtype=torch.long)
        output_state_target=self.state_matrices_dataset[quotient,remainder+1]


        return input_state, output_state_target
'''
# Define a custom sampler if needed (e.g., RandomSampler, SequentialSampler)
class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).tolist())  # Example: RandomSampler
'''


'''

# Create an instance of your map-style dataset
dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_2_variance_1_normalized_.json")

# Create a sampler instance
#sampler = CustomSampler(dataset)

# Instantiate DataLoader with the dataset and sampler
batch_size = 32  # Define your desired batch size
data_loader = DataLoader(dataset, batch_size=batch_size)#, sampler="RandomSampler")

# Iterate over batches of data
for batch in data_loader:
    # Process each batch as needed
    print("\n\nBatch:\n",batch,"\n\n\n")  # Example: Printing each batch of data
'''