import numpy as np
import pandas as pd
import sympy as sp
import argparse
import pysindy as ps
from tqdm import tqdm

# Hyperparameters for SINDy-PI
N_ITERATIONS = 100

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

def convert_to_symbolic(model, input_features):
    """Convert the learned SINDy-PI model to symbolic form."""
    equations = model.print(precision=3)
    symbolic_equations = []

    for equation in equations:
        # Convert the string representation of each equation into a symbolic expression
        symbolic_equations.append(sp.sympify(equation))

    return symbolic_equations

def find_best_formula(data, temp_file, n_iterations=N_ITERATIONS):
    """Run SINDy-PI to find the best formula, saving progress periodically and converting the output to symbolic form."""
    # Split data into inputs (X) and labels (y)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Initialize the SINDy-PI model
    pde_lib = ps.PDELibrary(
        function_library=ps.PolynomialLibrary(),
        temporal_grid=t,
        derivative_order=1,
        implicit_terms=True,
    )
    optimizer = ps.STLSQ(threshold=alpha)
    model = ps.SINDy(feature_library=pde_lib, 
                    optimizer=optimizer, 
                    feature_names=['P_p', 'tK', 'P_u'],
                    differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )

    best_formula = None

    with tqdm(total=n_iterations, desc="SINDy-PI Training Progress") as pbar:
        for iteration in range(n_iterations):
            model.fit(X, y)
            pbar.update(1)

            # Retrieve the best formula in symbolic form
            symbolic_formulas = convert_to_symbolic(model, input_features=data.columns[:-1])

            # Save the best formula periodically (every 10 iterations)
            if iteration % 10 == 0:
                best_formula = symbolic_formulas
                with open(temp_file, 'w') as f:
                    f.write("\n".join(str(formula) for formula in best_formula))

    # Final save of the best formula after all iterations
    with open(temp_file, 'w') as f:
        f.write("\n".join(str(formula) for formula in best_formula))

def main():
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