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

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
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
        model.fit(X, y, progress=True)

    except TimeoutError:
        print("PySR process timed out.")

    


def main():
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