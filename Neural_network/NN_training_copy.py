import os
from re import L
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from NN_eli import Neural_Network
from Dataset_classfile import CustomDataset

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("create_dataset")
from generate_dataset_classfile import Generate_dataset
# Or:
import json

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")





'''
# Iterate over batches of data
for batch in train_loader:
    # Process each batch as needed
    print("\n\nBatch:\n",batch,"\n\n\n")  # Example: Printing each batch of data
'''


#MAybe need more layers, maybe do residual learning instead, 
#Maybe use AR and improve with NN







####################################################################################################
    
class NN_training_run():
    #Class for running a training loop on a certain NN
        
    #Initialize a training run, given hyperparameters
    
        #model,   # MAYBE DO THIS WITHIN THE INIT AND SIMPLY PASS THE SHAPE    
    def __init__(self, 
                dataset, 
                num_epochs=1000, 
                criterion='sum', 
                learning_rate=1e-4,
                batch_size=100,                 
                NN_depth=3,
                NN_width=20,
                activation_function=nn.ReLU()): #dataset_path=None):     #Hyperparameters
                
    
        '''    THIS IS PREFFERABLY DONE OUTSIDE OF THIS CLASS - PASSING THE FINISHED DATASET MIGHT MAKE MORE SENSE
        if dataset_path==None: 
            dataset_path="create_dataset/dataset_storage/dataset_runs_1_variance_0_normalized_.json"
            
        self.dataset=CustomDataset(dataset_path) # Create an instance of your map-style dataset
        '''

        

        # Instantiate DataLoader with the dataset and sampler
        train_loader = DataLoader(dataset, batch_size=batch_size)#, sampler="RandomSampler")
        self.learning_rate=learning_rate
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.train_loader=train_loader
        

        self.hyperparameters_dict={
                "criterion": criterion ,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "NN_depth": NN_depth,
                "NN_width": NN_width,
                }

        #Values for the NN model
        self.NN_depth=NN_depth
        self.NN_width=NN_width
        NN_shape=[NN_width]*NN_depth

        self.NN_shape=NN_shape

        #Initialize our NN model
        self.model=Neural_Network(input_size=12, hidden_sizes=NN_shape, output_size=12, activation=activation_function)

        self.criterion=nn.MSELoss(reduction=criterion)    #Saves lossfunc
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.epoch_loss_evolution=torch.empty((self.num_epochs))


        


        #From CHATGPT:
        #loss_fn = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    def single_epoch_training_loop(self): #, model, train_loader, criterion, optimizer): #Not needed in class
        ''' Goes through one epoch of training
        Runs through the dataloader batch per batch, calculating loss to be returned
        '''

        # Set the model to training mode - important for batch normalization and dropout layers
        # Possibly Unnecessary in this situation but added for best practices
        self.model.train()  # Set model to training mode
        running_loss = 0.0
        for batch_nr, (inputs, labels) in enumerate(self.train_loader):

            inputs = inputs.float()  # Convert inputs to float tensor
            labels = labels.float()#.long()    # Convert labels to long tensor
            #print("Input: ",inputs, "\nShape: ", inputs.shape)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass  # Compute prediction and loss
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization    # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)

            #batch_loss_evolution[batch_nr]=running_loss  #add batchloss

        return running_loss #Return the current loss to be saved for each epoch


    def train_model(self, min_delta_error=1e-4): #, model, train_loader, criterion, optimizer, num_epochs=100, save=True, savename=None): #Not needed in class

        """
        Train the neural network model epoch for epoch

        Parameters:
            model (torch.nn.Module): The neural network model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
            criterion: The loss function used for training.
            optimizer: The optimizer used for updating model parameters.
            num_epochs (int): The number of epochs to train the model.

        Returns:
            None
        """

        #Create a tensor for storing the evolution of the loss for each batch

        #Below is for checking batchloss - might be totally unneccessary
        #batches_per_epoch=len(train_loader)
        #nr_of_batches=batches_per_epoch*num_epochs
        #print(nr_of_batches)
        #batch_loss_evolution=torch.empty((nr_of_batches))
        
        #epoch_loss_evolution=torch.empty((self.num_epochs))    #NOW DOING THIS AS A CLASS-ATTRIBUTE




        for epoch in range(self.num_epochs): #Run through epochs, go through the data and train, then check loss, run again
            #print(f"Epoch {epoch+1}\n-------------------------------")
            running_loss=self.single_epoch_training_loop()  #model, train_loader, criterion, optimizer) #Not needed for class
            #test_loop(test_dataloader, model, loss_fn)     # TO BE ADDED LATER
        
            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.epoch_loss_evolution[epoch]=epoch_loss  #Add the loss of the current epoch to the tensor of losses
            #print("len train_loader.dataset : ", len(train_loader.dataset))
            #print("Loss.item()=",loss.item())
            #print("size: ",inputs.size(0))

            if abs(self.epoch_loss_evolution[epoch-1]-epoch_loss)<min_delta_error:     #Check if we are still improving
                print("Ended at epoch ", epoch, " because learning stagnated" )
                return  #Exiting the epoch loop would be better, dont have internet right now to check the syntax

            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}\n-------------------------------")

        '''
        if save==True:
            if savename==None:
                savename="model_b["+ str(self.batch_size) + "]_lr["+ str(self.learning_rate) + "]_ep[" + str(self.num_epochs) + "].pt"
            torch.save(self.model, "Neural_network/model/"+savename)
        print("\nDone!\n")
        '''
        print("\nDone!\n")
        return self.epoch_loss_evolution #, batch_loss_evolution
        
    #def return_hyperparams(self):

    def save_model(self, savename=None):
        if savename==None:
            savename="model_b["+ str(self.batch_size) + "]_lr["+ str(self.learning_rate) + "]_ep[" + str(self.num_epochs) + "].pt"
        torch.save(self.model, "Neural_network/model/"+savename)
    
    def __str__(self):
        return "Model with hyperparams: " +  str(self.hyperparameters_dict) + "\n\nGives loss: " +  str(self.epoch_loss_evolution[-1]) 
        



####################################################################################################
    
class NN_hyperparameters_optimization():
    
    #Class for training the NN network, 
    # either for to train a model to be used 
    # or to test which model is most effective
    def __init__(self, dataset_path="create_dataset/dataset_storage/dataset_runs_1_variance_0_normalized_.json", fetch_hyperparams_path="hyperparams.json"):  #train=False, test=False, save=False,
        #self.train=train
        #self.test=test
        #self.save=save


        '''  This is done in the other class
        #Start with this as basic, maybe test changing later
        self.criterion=nn.MSELoss(reduction='sum')    #Saves lossfunc
        self.optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

        '''

        self.dataset=CustomDataset(dataset_path) # Create an instance of your map-style dataset
        self.hyperparams_path=fetch_hyperparams_path

        # For passing as #Pass the dict of all init values passed as keyword arguments
        if fetch_hyperparams_path==None:
            
            hyperparams_dict={
                "criterion":"sum" ,
                "learning_rate": 1e-4,
                "batch_size":100,
                "NN_depth":3,
                "NN_width":20,
                "activation_function" : nn.ReLU()
                }
        else:
            #Generate_dataset.fetch_dataset(fetch_hyperparams)  #does the same as below

            #Fetch hyperparamters stored in json file fetch_hyperparams
            with open(fetch_hyperparams_path, "r") as json_file:
                # returns JSON object as a dictionary
                hyperparams_dict = json.loads(json_file.read())
            
        self.hyperparams_dict=hyperparams_dict
        self.current_training=NN_training_run()
        self.hyperparams_stepsize={
            #"criterion":"sum",
            "learning_rate": 5e-6, # [1,1e-6, 5e-6],
            "batch_size":  10, #[1,600, 10],
            "NN_depth":  1, #[2,10, 1],
            "NN_width":  2, #[10,30, 2],
            #"activation_function" : nn.ReLU()
            }

    
        
        
    def set_hyperparams(self, dict_of_parameters):
        self.hyperparams_dict=dict_of_parameters
        print("Hyperparameters updated")
    
    def save_hyperparams(self):
        #Fetch hyperparamters from self.hyperparams_dict and save in hyperparams path
        with open(self.hyperparams_path, "w") as json_file:
            # Dumps hyperparameters in jsonfile
            json.dump(self.hyperparams_dict, json_file, indent=8)  # indent parameter for pretty formatting

    def run_training(self, dict_of_parameters=None):    #Creates a training_run_object based on hyperparams
        #Create a training run using decided hyperparameters
        if dict_of_parameters==None:
            current_training_run=NN_training_run(**self.hyperparams_dict)
        else:
            current_training_run=NN_training_run(**dict_of_parameters)
        self.current_training=current_training_run            #return current_training_run



    def evaluate_hyperparameters(self, dict_of_parameters=None):

        if dict_of_parameters==None:
            dict_of_parameters=self.hyperparams_dict


    def hyperparameter_space(self):

        Learning_rate_stepsize=5e-6
        batch_size_stepsize=5
        NN_depth_stepsize=1
        NN_width_stepsize=1

        Learning_rate_num_steps=2
        batch_size_num_steps=2
        NN_depth_num_steps=1
        NN_width_num_steps=1
        

        Learning_rate_steps=np.linspace(self.hyperparams_dict["learning_rate"])
        batch_size_steps=np.linspace(self.hyperparams_dict["batch_size"])
        NN_depth_steps=np.linspace(self.hyperparams_dict["NN_depth"])
        NN_width_steps=np.linspace(self.hyperparams_dict["NN_width"])

        for lr_val in np.arange(Learning_rate_steps):
            pass

        for i in self.hyperparams_stepsize.keys():
            #self.hyperparams_stepsize(i)[2]
            pass

        
        
        

    def sweep_training(self, iters=10, parameter="Epochs"):

        current_hyperparams=self.hyperparams_dict

        #Create a room of variance of hyperparams
        #Call the sweeping for all these
        #Find best one, 
        # Create room again, a lot smaller variance
        # Find best one
        #Until certain smallest error diff is reached

        self.run_training()    #Creates a NN_training_run object and saves in self.current_training_run



        #Create a tensor a room of possible hyperparams and their respective loss
        #The best one is saved


        
        #current_model=Neural_Network(hidden_sizes=model_hidden_shape)
        for i in np.arange(iters):
            self.run_training()



            #train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=epochs)




#model=Neural_Network()


dataset = CustomDataset("create_dataset/dataset_storage/dataset_runs_1_variance_0_normalized_.json") # Create an instance of your map-style dataset


training_obj=NN_training_run(dataset)
training_obj.train_model()
print(training_obj)








####################################################################################################

#batch_loss_evolution, epoch_loss_evolution= train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, num_epochs=epochs, save=False)


'''
plt.plot(epoch_loss_evolution,
        label="epoch_loss_evolution")
plt.xlabel="Epoch"
plt.ylabel("loss")
plt.title("training loss over epochs")
                          
plt.legend()
plt.show()

'''


'''


learn_training NN

    startparams=---

    for test in testing_parameters:
    
        NN_model=new_model(params)
        parameters=newparamaters

        current_loss = Model_training( NN_model    ,   parameters)
                            return loss
        

    save best params
'''



'''

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


'''



'''


    def train_model(self, model, train_loader, criterion, optimizer, num_epochs, save=True, savename=None):
        """
        Train the neural network model.

        Parameters:
            model (torch.nn.Module): The neural network model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader containing the training dataset.
            criterion: The loss function used for training.
            optimizer: The optimizer used for updating model parameters.
            num_epochs (int): The number of epochs to train the model.

        Returns:
            None
        """

        #Create a tensor for storing the evolution of the loss for each batch
        
        batches_per_epoch=len(train_loader)
        nr_of_batches=batches_per_epoch*num_epochs
        print(nr_of_batches)
        #batch_loss_evolution=torch.tensor((nr_of_batches))
        batch_loss_evolution=torch.empty((nr_of_batches))
        
        epoch_loss_evolution=torch.empty((num_epochs))


        for epoch in range(num_epochs):

            # Set the model to training mode - important for batch normalization and dropout layers
            # Possibly Unnecessary in this situation but added for best practices
            model.train()  # Set model to training mode
            running_loss = 0.0
            for batch_nr, (inputs, labels) in enumerate(train_loader):

                inputs = inputs.float()  # Convert inputs to float tensor
                labels = labels.float()#.long()    # Convert labels to long tensor
                #print("Input: ",inputs, "\nShape: ", inputs.shape)
                
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass  # Compute prediction and loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization    # Backpropagation
                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.item() * inputs.size(0)

                batch_loss_evolution[batch_nr]=running_loss  #add batchloss
                

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_loss_evolution[epoch]=epoch_loss  #Add the loss of the current epoch to the tensor of losses
            #print("len train_loader.dataset : ", len(train_loader.dataset))
            #print("Loss.item()=",loss.item())
            #print("size: ",inputs.size(0))
            if (epoch+1) % 25 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}\n-------------------------------")
            
        if save==True:
            if savename==None:
                savename="model_b["+ str(self.batch_size) + "]_lr["+ str(self.learning_rate) + "]_ep[" + str(num_epochs) + "].pt"
            torch.save(model, "Neural_network/model/"+savename)
        return batch_loss_evolution, epoch_loss_evolution


'''