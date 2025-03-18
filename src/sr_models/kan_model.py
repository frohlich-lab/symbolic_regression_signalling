"""
This module loads a dataset, trains a KAN model to find the best symbolic formula,
and saves the intermediate results to a temporary file.

Usage:
    python kan_model.py --dataset <path_to_dataset> --temp_file <path_to_temp_file> [--dataset_size <size>] [--features <feature_list>]
"""

import numpy as np
import pandas as pd
import sympy as sp
import argparse
import torch
from kan import KAN
from tqdm import tqdm
import os

# Hyperparameters
TRAIN_SIZE = 0.8  # Training set size as a percentage of the dataset
SEED = 1  # Random seed for reproducibility

# Model Settings
N_ITERATIONS = 3  # Number of KAN model iterations
WIDTH = [5, 7, 5, 1]  # Width of neural network layers
GRID = 40  # Grid size for KAN
K = 3  # Degree of polynomial features
THRESHOLD = 0.01  # Pruning threshold

# Training Configuration
OPTIMIZER = "Adam"  # Optimizer for KAN training
STEPS = 150  # Optimization steps per iteration
LAMB = 0.001  # Regularization parameter
LAMB_ENTROPY = 1.0  # Entropy regularization

# Function Library for Symbolic Representation
LIBRARY = ['x', 'x^2', 'x^3', 'exp', 'log', 'abs']  # KAN function library

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if specified, 
    and select specific columns if features are provided.
    """
    data = pd.read_csv(file_path)
    if features and features != "all":
        data = data[features.split(',')]
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)))
    return data

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """
    Run KAN model to find the best formula, saving results 
    periodically and converting output to symbolic form.
    """
    # Convert data to torch tensors
    dataset = torch.tensor(data.values, dtype=torch.float32)
    dataset_size = len(dataset)
    test_size = int((1 - TRAIN_SIZE) * dataset_size)
    train_size = dataset_size - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Verify feature and label dimensions match the dataset length
    train_input = torch.stack([train_dataset[i][:-1] for i in range(len(train_dataset))])
    train_label = torch.stack([train_dataset[i][-1] for i in range(len(train_dataset))]).unsqueeze(1)
    test_input = torch.stack([test_dataset[i][:-1] for i in range(len(test_dataset))])
    test_label = torch.stack([test_dataset[i][-1] for i in range(len(test_dataset))]).unsqueeze(1)
    
    # Prepare dataset dictionary
    dataset_dict = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    # Initialize KAN model
    model = KAN(width=WIDTH, grid=GRID, k=K, seed=SEED)

    with tqdm(total=n_iterations, desc="KAN Model Training") as pbar:
        for iteration in range(n_iterations):
            try:
                # Train model
                results = model.fit(
                    dataset_dict,
                    opt=OPTIMIZER,
                    steps=STEPS,
                    lamb=LAMB,
                    lamb_entropy=LAMB_ENTROPY
                )
            
                # Prune the model and extract the formula
                model.prune(edge_th=THRESHOLD)
                model.auto_symbolic(lib=LIBRARY)
                current_formula = model.symbolic_formula()[0][0]
                loss_value = results['test_loss'][-1]
                
                # Save progress
                with open(temp_file, 'a') as f:
                    f.write(f"Iteration {iteration + 1}:\n")
                    f.write(f"Formula: {current_formula}\n")
                    f.write(f"Loss: {loss_value}\n\n")
                
                pbar.update(1)

            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Run KAN for Symbolic Regression')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Max number of samples from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save intermediate results')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file)

if __name__ == '__main__':
    main()