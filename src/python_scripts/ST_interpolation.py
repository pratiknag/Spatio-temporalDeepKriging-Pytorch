#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd


# Define tilted loss function
def tilted_loss1(y, f):
    q=0.5
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum
    
def tilted_loss2(y, f):
    q=0.975
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum
    
def tilted_loss3(y, f):
    q=0.025
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum

# Define the model architecture (using nn.Module in PyTorch)
class DeepKrigingModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepKrigingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 1)  # Output layer
        
         # Trainable parameter for x_
        self.x_ = nn.Parameter(torch.tensor(0.5))  # initialize at some value (e.g., 0.5)
        
        # Final output layer for x_
        self.fc_x_ = nn.Linear(50, 1)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x1 = self.fc9(x)
        
        # x_ output (with trainable parameter, transformed via softplus)
        x_ = torch.nn.functional.softplus(self.fc_x_(x)) * self.x_
        
        # Multiply x by 10
        #x = x * 10
        
        # x2 = x1 + x_
        x2 = x1 + x_
        
        # x3 = x1 - x_
        x3 = x1 - x_
        
        return x1, x2, x3

def main():

    # Load your data (example placeholders for phi_reduce_train and y)
    # phi_reduce_train and y should be converted to torch tensors
    df = pd.read_csv("datasets/dataset-10DAvg-sample.csv")
    y = np.array(df["Station_Value"]).reshape(len(df),1)
    del(df)
    phi_reduce = np.load("datasets/phi_float16.npy")
    x_train, x_test, y_train, y_test = train_test_split(phi_reduce, y, test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    # Convert to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Initialize model
    model = DeepKrigingModel(input_dim=x_train.shape[1])
    step_size = 20
    gamma = 0.1
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Create data loaders for training and testing (batching)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Training loop
    start_time = time.time()
    num_epochs = 350
    patience = 30  # Early stopping patience
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    

    directory = 'models/'

    # Check if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            x1, x2, x3 = model(inputs)
            
            # Compute loss
            loss = tilted_loss1(targets, x1) + tilted_loss2(targets,x2) + tilted_loss3(targets,x3)
            #print(loss.shape)
            # Backward pass and optimize
            loss.sum().backward()
            optimizer.step()
            
            running_loss += loss.sum().item()
        scheduler.step()
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                x1, x2, x3 = model(inputs)
                loss = tilted_loss1(targets, x1) + tilted_loss2(targets,x2) + tilted_loss3(targets,x3)
                val_loss += loss.sum().item()

        # Print training and validation loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(test_loader)}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model
            torch.save(model.state_dict(), 'models/model_interpolation.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        

    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")

    # Loading the best model
    model.load_state_dict(torch.load('models/model_interpolation.pth'))

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor)

    # Evaluate model performance (e.g., Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred.numpy())
    print(f"Mean Squared Error: {mse}")

#if __name__ == '__main__':
 #   main()
