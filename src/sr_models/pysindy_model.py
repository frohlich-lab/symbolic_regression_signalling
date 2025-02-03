"""
This module provides functionality for symbolic regression using the SINDy-PI algorithm.
It includes functions to load datasets, convert models to symbolic form, and find the best formula
through iterative training. The main function allows for command-line execution.

Usage:
    python pysindy_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import numpy as np
import pandas as pd
import sympy as sp
import argparse
import pysindy as ps
from tqdm import tqdm

# Parameters for trajectory selected from data
N_SAMPLE = 0
N_TIME_STEPS = 22
# Hyperparameters for SINDy-PI
N_ITERATIONS = 100
ALPHA = 0.1

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if a dataset size is specified, and select specific features if provided.

    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    dataset_size (int, optional): Number of samples to use from the dataset. Defaults to None.
    features (str, optional): Comma-separated list of features to use from the dataset. Defaults to None.

    Returns:
    pd.DataFrame: The loaded and possibly filtered dataset.
    """
    data = pd.read_csv(file_path)

    if features and features != "all":
        feature_list = features.split(',')
        data = data[feature_list]

    if dataset_size:
        dataset_size = min(dataset_size, len(data))
        data = data.sample(n=dataset_size)

    return data

def convert_to_symbolic(model, input_features):
    """
    Convert the learned SINDy-PI model to symbolic form.

    Parameters:
    model (ps.SINDy): The trained SINDy-PI model.
    input_features (list): List of input feature names.

    Returns:
    list: List of symbolic equations.
    """
    equations = model.print(precision=3)
    symbolic_equations = [sp.sympify(equation) for equation in equations]
    return symbolic_equations

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """
    Run SINDy-PI to find the best formula, saving progress periodically and converting the output to symbolic form.

    Parameters:
    data (pd.DataFrame): The dataset to use for training.
    temp_file (str): Path to the temporary file to save intermediate results.
    n_iterations (int, optional): Number of iterations for training. Defaults to N_ITERATIONS.
    """
    sample = data.iloc[N_SAMPLE*N_TIME_STEPS:(N_SAMPLE+1)*N_TIME_STEPS, :].reset_index()
    X = sample.iloc[:, :-1].values
    y = sample.iloc[:, -1].values

    t = np.log(np.array(data.index[:-1]) + 1)

    pde_lib = ps.PDELibrary(
        library_functions=ps.PolynomialLibrary(),
        temporal_grid=t,
        derivative_order=1,
        implicit_terms=True,
    )
    optimizer = ps.STLSQ(threshold=ALPHA)
    model = ps.SINDy(
        feature_library=pde_lib, 
        optimizer=optimizer, 
        feature_names=['P_p', 'tK', 'P_u'],
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )

    best_formula = None

    with tqdm(total=n_iterations, desc="SINDy-PI Training Progress") as pbar:
        for iteration in range(n_iterations):
            model.fit(X, y)
            pbar.update(1)

            symbolic_formulas = convert_to_symbolic(model, input_features=data.columns[:-1])

            if iteration % 10 == 0:
                best_formula = symbolic_formulas
                with open(temp_file, 'w') as f:
                    f.write("\n".join(str(formula) for formula in best_formula))

    with open(temp_file, 'w') as f:
        f.write("\n".join(str(formula) for formula in best_formula))

def main():
    """
    Main function to execute the script from the command line.
    Parses command-line arguments and initiates the process to find the best formula using SINDy-PI.
    """
    parser = argparse.ArgumentParser(description='Find the best formula using SINDy-PI')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to the temporary file to save intermediate results')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file)

if __name__ == '__main__':
    main()