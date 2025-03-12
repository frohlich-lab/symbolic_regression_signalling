"""
This module provides functionality for symbolic regression using the PySR algorithm.
It includes functions to load datasets, find the best formula, and run the main function for command-line execution.

Usage:
    python pysr_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import numpy as np
import pandas as pd
import argparse
from pysr import PySRRegressor
import os

# Hyperparameters for PySR (Python Symbolic Regression)
N_ITERATIONS = 100  # Total iterations for model training
POPULATION_SIZE = 30  # Symbolic expressions in the population
POPULATIONS = 15  # Number of populations to evolve
MUTATION_RATE = 0.1  # Mutation probability per individual
MAX_SIZE = 20  # Maximum size of symbolic expressions
PARSIMONY = 1  # Regularization to reduce expression complexity
VERBOSITY = 0  # Verbosity level during training
BATCHING = True  # Enable mini-batch training
ANNEALING = True  # Enable annealing to escape local minima

# Operators for Symbolic Regression
BINARY_OPERATORS = ["+", "*", "/", "-"]  # Binary operators for expressions
UNARY_OPERATORS = ["exp", "log"]  # Unary operators for expressions

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

    return data

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """
    Run PySR to find the best formula, saving results to a specified file.
    """
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values  # Split data into inputs (X) and output (y)

    try:
        # Set up and train the PySR model
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
            equation_file=temp_file  # Save best formulas to this file
        )
        model.fit(X, y)

    except TimeoutError:
        print("PySR process timed out. Try reducing the number of iterations or adjusting other hyperparameters.")
    except Exception as e:
        print(f"An error occurred during PySR training: {e}")

def main():
    """
    Main function to parse arguments and find the best formula using PySR.
    """
    parser = argparse.ArgumentParser(description='Find the best formula using PySR')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Maximum number of samples to load from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to save the intermediate results from PySR')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file)

if __name__ == '__main__':
    main()