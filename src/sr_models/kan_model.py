import numpy as np
import pandas as pd
import sympy as sp
import argparse
import torch
from kan import KAN
from tqdm import tqdm
import os

# Hyperparameters
# Data Configuration
TRAIN_SIZE = 0.8  # Percentage of the dataset to use for training
SEED = 1  # Random seed for reproducibility

# Model Settings
N_ITERATIONS = 3  # Number of iterations to run the KAN model
WIDTH = [5, 5, 5, 1]  # Width of the neural network layers
GRID = 35  # Grid size for the KAN model
K = 3  # Degree of the polynomial features
THRESHOLD = 0.01  # Threshold for pruning the KAN model

# Training Configuration
OPTIMIZER = "BFGS"  # Optimizer used for training the KAN model
STEPS = 20  # Number of optimization steps to take during training
LAMB = 0.1  # Regularization parameter for the KAN model
LAMB_ENTROPY = 10  # Entropy regularization parameter for the KAN model

# Function Library
LIBRARY = ['x', 'x^2', 'x^3', 'exp', 'log', 'abs']  # Library of functions to use in the KAN model

def load_dataset(file_path, dataset_size=None, features=None):
    """Load dataset from a CSV file, sample it if a dataset size is specified, and select specific features if provided."""
    data = pd.read_csv(file_path)

    # Filter dataset columns based on features
    if features and features != "all":
        feature_list = features.split(',')
        data = data[feature_list]

    # Sample the dataset if dataset size is specified
    if dataset_size:
        dataset_size = min(dataset_size, len(data))  # Ensure we don't exceed dataset size
        data = data.sample(n=dataset_size)

    return data

def convert_to_symbolic(expression):
    """Convert a KAN model expression to symbolic form."""
    # Assuming the expression is provided as a KAN model's internal representation, convert it to a SymPy expression.
    # You would replace this placeholder logic with the actual conversion logic provided by KAN.
    symbolic_expression = sp.sympify(expression)
    return symbolic_expression

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """Run KAN to find the best formula, saving progress periodically and converting the output to symbolic form."""
    # Convert data to tensors suitable for KAN
    dataset_size = len(data)
    test_size = int((1 - TRAIN_SIZE) * dataset_size)
    train_size = dataset_size - test_size

    # Convert data to torch tensors
    dataset = torch.tensor(data.values, dtype=torch.float32)
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Extracting data from these subsets
    train_tensor = dataset[train_dataset.indices]
    test_tensor = dataset[test_dataset.indices]

    # Now split these tensors into inputs and labels
    train_input = train_tensor[:, :-1]
    train_label = train_tensor[:, -1].unsqueeze(1)  # Reshape for consistency as 2D

    test_input = test_tensor[:, :-1]
    test_label = test_tensor[:, -1].unsqueeze(1)  # Reshape for consistency as 2D

    # Now you can create the dataset dictionary
    dataset_dict = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    # Initialize the KAN model
    model = KAN(width=WIDTH, 
                grid=GRID, 
                k=K, 
                seed=SEED)

    with tqdm(total=n_iterations, desc="KAN Model Training Progress") as pbar:
        for iterations in range(N_ITERATIONS):
            model.fit(dataset_dict, 
                      opt=OPTIMIZER, 
                      steps=STEPS, 
                      lamb=LAMB,
                      lamb_entropy=LAMB_ENTROPY)  # Perform a training step
            
            model = model.prune(threshold=THRESHOLD)  # Prune the model to remove unnecessary terms
            
            pbar.update(1)

    model.auto_symbolic(lib=LIBRARY)
    best_formula = model.symbolic_formula(3)[0][0]
    # Final save of the best formula after all epochs
    with open(temp_file, 'w') as f:
        f.write(str(best_formula))

def main():
    parser = argparse.ArgumentParser(description='Find the best formula using KAN')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to the temporary file to save intermediate results')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file)

if __name__ == '__main__':
    main()