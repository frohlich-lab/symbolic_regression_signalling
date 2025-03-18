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
EPOCHS = 300
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
    
    # Sample the dataset if dataset size is specified
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)))
    
    scaler_X = StandardScaler()
    data.iloc[:, :-1] = scaler_X.fit_transform(data.iloc[:, :-1])  # Normalize input features

    return data

def train_model(data, temp_file):
    """
    Train a neural network on the dataset and save predictions to a file.
    """
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = NeuralNet(input_dim=X.shape[1], output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    
    torch.save(model.state_dict(), temp_file)
    print(f"Model trained and saved to {temp_file}")

def main():
    """
    Main function to parse arguments and train the neural network.
    """
    parser = argparse.ArgumentParser(description='Train a neural network on the dataset')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Maximum number of samples to load from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to save the trained model')
    
    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    train_model(data, args.temp_file)

if __name__ == '__main__':
    main()

