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
TRAIN_SIZE = 0.8  # Percentage of the dataset to use for training
SEED = 1  # Random seed for reproducibility
N_ITERATIONS = 3  # Number of iterations to run the KAN model
WIDTH = [5, 5, 5, 1]  # Width of the neural network layers
GRID = 35  # Grid size for the KAN model
K = 3  # Degree of the polynomial features
THRESHOLD = 0.01  # Threshold for pruning the KAN model
OPTIMIZER = "LBFGS"  # Optimizer used for training the KAN model
STEPS = 20  # Number of optimization steps to take during training
LAMB = 0.01  # Regularization parameter for the KAN model
LAMB_ENTROPY = 10.0  # Entropy regularization parameter for the KAN model
LIBRARY = ['x', 'x^2', 'x^3', 'exp', 'log', 'abs']  # Library of functions to use in the KAN model

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if a dataset size is specified, and select specific features if provided.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        dataset_size (int, optional): Number of samples to use from the dataset. Defaults to None.
        features (str, optional): Comma-separated list of features to use from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: Loaded and optionally filtered dataset.
    """
    data = pd.read_csv(file_path)

    if features and features != "all":
        feature_list = features.split(',')
        data = data[feature_list]

    if dataset_size:
        dataset_size = min(dataset_size, len(data))
        data = data.sample(n=dataset_size)

    return data

def convert_to_symbolic(expression):
    """
    Convert a KAN model expression to symbolic form.

    Args:
        expression (str): Expression in KAN model's internal representation.

    Returns:
        sp.Expr: SymPy expression.
    """
    symbolic_expression = sp.sympify(expression)
    return symbolic_expression

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """
    Run KAN to find the best formula, saving progress periodically and converting the output to symbolic form.

    Args:
        data (pd.DataFrame): Dataset to use for training and testing.
        temp_file (str): Path to the temporary file to save intermediate results.
        n_iterations (int, optional): Number of iterations to run the KAN model. Defaults to N_ITERATIONS.
    """
    dataset_size = len(data)
    test_size = int((1 - TRAIN_SIZE) * dataset_size)
    train_size = dataset_size - test_size

    dataset = torch.tensor(data.values, dtype=torch.float32)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_tensor = dataset[train_dataset.indices]
    test_tensor = dataset[test_dataset.indices]

    train_input = train_tensor[:, :-1]
    train_label = train_tensor[:, -1].unsqueeze(1)

    test_input = test_tensor[:, :-1]
    test_label = test_tensor[:, -1].unsqueeze(1)

    dataset_dict = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    model = KAN(width=WIDTH, grid=GRID, k=K, seed=SEED)

    with tqdm(total=n_iterations, desc="KAN Model Training Progress") as pbar:
        for iteration in range(n_iterations):
            results = model.fit(
                dataset_dict,
                opt=OPTIMIZER,
                steps=STEPS,
                lamb=LAMB,
                lamb_entropy=LAMB_ENTROPY
            )
            
            model = model.prune(node_th=THRESHOLD, edge_th=THRESHOLD)
            model.auto_symbolic(lib=LIBRARY)
            current_formula = model.symbolic_formula()[0][0]
            loss_value = results['test_loss'][-1]
            
            with open(temp_file, 'a') as f:
                f.write(f"Iteration {iteration + 1}:\n")
                f.write(f"Formula: {current_formula}\n")
                f.write(f"Loss: {loss_value}\n\n")
            
            pbar.update(1)

def main():
    """
    Main function to parse arguments and run the KAN model training.
    """
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