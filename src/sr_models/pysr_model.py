"""
This module provides functionality for symbolic regression using the PySR algorithm.
It includes functions to load datasets, find the best formula, and run the main function for command-line execution.

Usage:
    python pysr_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import argparse
import random
from typing import Optional

import numpy as np
import pandas as pd
from pysr import PySRRegressor
import os

TARGET_COLUMN = 'kcat_cg'

# Hyperparameters for PySR (Python Symbolic Regression)
N_ITERATIONS = 200  # Total iterations for model training
POPULATION_SIZE = 30  # Symbolic expressions in the population
POPULATIONS = 15  # Number of populations to evolve
MUTATION_RATE = 0.1  # Mutation probability per individual
MAX_SIZE = 20  # Maximum size of symbolic expressions
PARSIMONY = 1  # Regularization to reduce expression complexity
VERBOSITY = 0  # Verbosity level during training
BATCHING = True  # Enable mini-batch training
ANNEALING = True  # Enable annealing to escape local minima
LOG_SPACE_LOSS = "my_loss(x,y)=(log(max(x,0)+1e-25)-log(max(y,0)+1e-25))^2"  # Custom log-space loss function

# Operators for Symbolic Regression
BINARY_OPERATORS = ["+", "*", "/", "-"]  # Binary operators for expressions
UNARY_OPERATORS = ["exp", "log"]  # Unary operators for expressions

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if dataset size is specified, 
    and select specific features if provided.
    """
    data = pd.read_csv(file_path)

    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]

    # Select specific columns if features are provided
    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for PySR: {missing}. Using available columns {available}.")
        selected = available + ([target] if target not in available else [])
    else:
        selected = list(data.columns)
    data = data[selected]

    # Sample the dataset if dataset size is specified
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)))

    data = data.map(np.exp)

    return data

def find_best_formula(
    data,
    tempdir,
    n_iterations: int = N_ITERATIONS,
    population_size: int = POPULATION_SIZE,
    max_size: int = MAX_SIZE,
    parsimony: float = PARSIMONY,
    seed: Optional[int] = None,
):
    """
    Run PySR to find the best formula, saving results to a specified file.
    """
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values  # Split data into inputs (X) and output (y)
    
    try:
        if seed is not None:
            set_seed(seed)
        # Set up and train the PySR model
        model = PySRRegressor(
            niterations=n_iterations,
            population_size=population_size,
            populations=POPULATIONS,
            binary_operators=BINARY_OPERATORS,
            unary_operators=UNARY_OPERATORS,
            maxsize=max_size,
            parsimony=parsimony,
            verbosity=VERBOSITY,
            batching=BATCHING,
            elementwise_loss=LOG_SPACE_LOSS,
            annealing=ANNEALING,
            output_directory=os.path.dirname(tempdir),  # Save best formulas to this file
            run_id="temp",
            random_state=seed,
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
    parser.add_argument('--n_iterations', type=int, default=N_ITERATIONS, help='Number of PySR iterations')
    parser.add_argument('--population_size', type=int, default=POPULATION_SIZE, help='Population size for PySR')
    parser.add_argument('--max_size', type=int, default=MAX_SIZE, help='Maximum symbolic expression size')
    parser.add_argument('--parsimony', type=float, default=PARSIMONY, help='Parsimony coefficient')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    tempdir = os.path.dirname(args.temp_file)
    find_best_formula(
        data,
        tempdir,
        n_iterations=args.n_iterations,
        population_size=args.population_size,
        max_size=args.max_size,
        parsimony=args.parsimony,
        seed=args.seed,
    )

if __name__ == '__main__':
    main()
