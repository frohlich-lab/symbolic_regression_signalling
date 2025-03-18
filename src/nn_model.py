import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os

# Hyperparameters for the neural network
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 25
HIDDEN_LAYERS = [128, 64, 32]
ACTIVATION = nn.ReLU

class NeuralNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNet, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(ACTIVATION())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if dataset size is specified, 
    and select specific features if provided.
    """
    data = pd.read_csv(file_path)
    
    # Select specific columns if features are provided
    if features and features != "all":
        data = data[features.split(',')]
    
    # Normalize input features
    scaler_X = StandardScaler()
    data.iloc[:, :-1] = scaler_X.fit_transform(data.iloc[:, :-1])
    
    # Sample the dataset if dataset size is specified
    sampled_data = data.sample(n=min(dataset_size, len(data))) if dataset_size else data

    return sampled_data, data

def train_and_evaluate_model(sampled_data, full_data):
    """
    Train a neural network on the sampled dataset and calculate MAE on the full dataset.
    """
    # Prepare sampled data for training
    X_sampled, y_sampled = sampled_data.iloc[:, :-1].values, sampled_data.iloc[:, -1].values
    X_sampled = torch.tensor(X_sampled, dtype=torch.float32)
    y_sampled = torch.tensor(y_sampled, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X_sampled, y_sampled)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = NeuralNet(input_dim=X_sampled.shape[1], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    # Evaluate on the full dataset
    X_full, y_full = full_data.iloc[:, :-1].values, full_data.iloc[:, -1].values
    X_full = torch.tensor(X_full, dtype=torch.float32)
    y_full = torch.tensor(y_full, dtype=torch.float32).view(-1, 1)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_full)
        mae = torch.mean(torch.abs(predictions - y_full)).item()
    
    print(f"Mean Absolute Error (MAE) on the full dataset- Neural Network: {mae}")

def main():
    """
    Main function to parse arguments and train the neural network.
    """
    parser = argparse.ArgumentParser(description='Train a neural network and evaluate MAE')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Maximum number of samples to load from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--output', required=True, help='File path to save the model') 
    
    args = parser.parse_args()
    sampled_data, full_data = load_dataset(args.dataset, args.dataset_size, args.features)
    train_and_evaluate_model(sampled_data, full_data)

if __name__ == '__main__':
    main()
