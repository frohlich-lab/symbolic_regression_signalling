"""
This module provides functionality for symbolic regression using the PySR algorithm.
It includes functions to load datasets, find the best formula, and run the main function for command-line execution.

Usage:
    python pysr_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import numpy as np
import pandas as pd
import sympy as sp
import argparse
from pysr import PySRRegressor
from tqdm import tqdm
import os

# Hyperparameters for PySR (Python Symbolic Regression)
N_ITERATIONS = 1000  # Total number of iterations for training the model
POPULATION_SIZE = 30  # Number of symbolic expressions in the population
POPULATIONS = 15  # Number of populations to evolve
MUTATION_RATE = 0.1  # Probability of mutating an individual in the population
MAX_SIZE = 20  # Maximum size of the symbolic expressions
LOG_SPACE_LOSS = "my_loss(x,y)=(log(max(x,0)+1e-25)-log(max(y,0)+1e-25))^2"  # Custom log-space loss function
PARSIMONY = 1  # Regularization strength to reduce the complexity of expressions
VERBOSITY = 0  # Level of detail in the output during training
BATCHING = True  # Use mini-batch training
ANNEALING = True  # Use annealing during training to escape local minima

# Function Library for Symbolic Regression
BINARY_OPERATORS = ["+", "*", "/", "-"]  # Binary operators used in the symbolic expressions
UNARY_OPERATORS = ["exp", "log", "abs", "neg"]  # Unary operators used in the symbolic expressions

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if a dataset size is specified, and select specific features if provided.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        dataset_size (int, optional): Number of samples to use from the dataset. Defaults to None.
        features (str, optional): Comma-separated list of features to use from the dataset. Defaults to None.

    Returns:
        pd.DataFrame: Loaded and optionally filtered/sampled dataset.
    """
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

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """
    Find the best symbolic regression formula using PySR.

    Args:
        data (pd.DataFrame): The dataset containing input features and target labels.
        temp_file (str): Path to the temporary file to save intermediate results.
        n_iterations (int, optional): Number of iterations for training the model. Defaults to N_ITERATIONS.
    """
    # Split data into inputs (X) and labels (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    try:
        model = PySRRegressor(
            niterations=n_iterations,
            population_size=POPULATION_SIZE,
            populations=POPULATIONS,
            binary_operators=BINARY_OPERATORS,
            unary_operators=UNARY_OPERATORS,
            maxsize=MAX_SIZE,
            parsimony=PARSIMONY,
            verbosity=VERBOSITY,
            batching=BATCHING,
            annealing=ANNEALING,
            elementwise_loss=LOG_SPACE_LOSS,
            equation_file=temp_file  # PySR saves its best formulas in this file
        )
        model.fit(X, y)

    except TimeoutError:
        print("PySR process timed out.")

def main():
    """
    Main function to parse arguments and find the best formula using PySR.
    """
    parser = argparse.ArgumentParser(description='Find the best formula using PySR')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to the temporary file to save intermediate results')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)

    find_best_formula(data, args.temp_file)

if __name__ == '__main__':
    main()