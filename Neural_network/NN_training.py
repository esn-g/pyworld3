
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
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
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)  # Convert inputs to float tensor
            labels = labels.long().to(device)    # Convert labels to long tensor
            

            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")