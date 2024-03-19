import torch
import torchvision.models as models

print(torch.cuda.is_available())
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Step 1: Define your neural network architecture
class StatePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StatePredictor, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Step 2: Prepare your dataset
# Assume you have a list of states where each state is a 12-valued vector
# Construct your dataset as pairs of (current_state, next_state)
# For simplicity, let's create a synthetic dataset
# Replace this with your actual data loading code
def generate_synthetic_data(num_samples, input_size):
    return torch.randn(num_samples, input_size)

# Hyperparameters
input_size = 12
hidden_size = 64
output_size = 12
batch_size = 32
num_epochs = 1000
learning_rate = 0.001

# Generate synthetic dataset
data = generate_synthetic_data(1000, input_size)
# Define dataset and dataloader
dataset = TensorDataset(data[:-1], data[1:])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 3: Define your loss function
loss_function = nn.MSELoss()

# Step 4: Set up your optimizer
model = StatePredictor(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 5: Write your training loop
losses = []  # To store the loss values for plotting

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

# Plot the training loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()