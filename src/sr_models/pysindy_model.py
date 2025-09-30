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

TARGET_COLUMN = 'kcat_cg'

from sympy import symbols, diff, symbols, Derivative, Eq, simplify, solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

# Parameters
N_SAMPLES = 300
N_TIME_STEPS = 22
N_ITERATIONS = 100
ALPHA = 1e4
THRESHOLD = 1e-4
# Define custom library functions for SINDy-PI
LIBRARY_FUNCTIONS = [
    lambda x: 1,
    lambda x: x,
    lambda x, y: x * y,
    lambda x, y, z: x * y * z,
    lambda x, y, z, t: x * y * z * t,
]

FUNCTION_NAMES = [
    lambda x: "1",
    lambda x: x,
    lambda x, y: f"{x}*{y}",
    lambda x, y, z: f"{x}*{y}*{z}",
    lambda x, y, z, t: f"{x}*{y}*{z}*{t}",
]

def load_dataset(
    file_path: str, dataset_size: int = None, features: str = None
) -> pd.DataFrame:
    """Load a dataset, sample it, and select specified features."""
    data = pd.read_csv(file_path)

    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]

    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        feature_list = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for PySINDy: {missing}. Using available columns {feature_list}.")
        cols_to_use = ['time'] if 'time' in data.columns else []
        cols_to_use += feature_list
        if target not in cols_to_use:
            cols_to_use.append(target)
        data = data[cols_to_use].dropna()

    if dataset_size:
        dataset_size = min(dataset_size, len(data))
        data = data.iloc[:dataset_size]

    # Apply exponential transformation
    if 'feature_list' in locals():
        data[feature_list] = np.exp(data[feature_list])
    
    return data



def convert_to_symbolic(equations_list, input_features):
    """
    Transforms a list of equation strings into SymPy equations,
    handling glued variables and automatically interpreting '_t' variables
    as time derivatives of given base variables.

    Args:
    - equations_list (list of str): List of equation strings.
    - variables (list of str): List of known base variable names.

    Returns:
    - list of sympy expressions.
    """
    t = sp.symbols('t')  # time symbol for derivatives

    # Create sympy symbols for base variables
    sympy_vars = {var: sp.symbols(var) for var in input_features}
    # sympy_derivatives = {var:sp.symbols(var+'_t') for var in input_features}
    # sympy_vars = {**sympy_vars, **sympy_derivatives}

    # Sort variables by length for proper replacement
    sorted_vars = sorted(input_features, key=len, reverse=True)
    sympy_equations = []

    for eq in equations_list:
        if eq.strip() == '0.000':
            sympy_equations.append(sp.sympify(0))
            continue

        # Step 1: Insert '*' between adjacent numbers and variables
        eq = re.sub(r'(\d*\.?\d+)\s+(\d*\.?\d+)', r'\1*\2', eq)
        eq = re.sub(r'(\d*\.?\d+)\s*([A-Za-z_])', r'\1*\2', eq)

        # Step 2: Separate all known variables even if glued together
        for var1 in sorted_vars:
            for var2 in sorted_vars:
                eq = re.sub(rf'({var1})(?={var2})', r'\1*', eq)

        # Step 3: Handle '_t' derivatives AFTER splitting
        def derivative_replacer(match):
            var_name = match.group(1)
            return f"Derivative({var_name}, t)"

        eq = re.sub(r'([A-Za-z_][A-Za-z0-9_]*)_t', derivative_replacer, eq)

        # Step 4: Final cleanup (remove duplicate '*')
        eq = re.sub(r'\*{2,}', '*', eq)

        # Step 5: Parse with sympy
        try:
            sympy_expr = sp.sympify(eq, locals={**sympy_vars, 'Derivative': sp.Derivative, 't': t})
            sympy_equations.append(sympy_expr)
        except Exception as e:
            print(f"Error parsing equation: {eq}\n{e}")
            sympy_equations.append(None)

    print(sympy_equations)
    return sympy_equations

def convert_to_explicit_ode_system(rhs_expressions, input_features, t=symbols('t')):
    """
    Converts a system of ODEs given as SymPy RHS expressions into an explicit ODE system,
    handling cross-derivatives properly.

    Parameters:
    ----------
    rhs_expressions : list of sympy.Expr
        The RHS expressions of the ODEs corresponding to each input feature.
    input_features : list of str
        The list of input feature names corresponding to the ODEs.
    t : sympy.Symbol
        The time symbol used in the derivatives (default is 't').

    Returns:
    -------
    list of sympy.Eq
        Explicit ODE system with LHS as time derivatives and RHS as simplified expressions.
    """
    # Create derivative symbols: Derivative(x, t), Derivative(y, t), etc.
    derivative_symbols = {var: Derivative(symbols(var), t) for var in input_features}

    # Step 1: Form a list of equations in SymPy Eq format
    equations = []
    for var, rhs_expr in zip(input_features, rhs_expressions):
        lhs = derivative_symbols[var]
        equations.append(Eq(lhs, rhs_expr))

    # Step 2: Flatten all equations into a system and solve for each derivative explicitly
    # We solve for all time derivatives simultaneously
    solution = solve(equations, list(derivative_symbols.values()), dict=True)

    if not solution:
        raise ValueError("No explicit solution found for the given ODE system.")

    solution = solution[0]  # Extract the first solution dictionary

    # Step 3: Construct explicit ODE equations: d(var)/dt = RHS
    explicit_odes = []
    for var in input_features:
        lhs = derivative_symbols[var]
        rhs = simplify(solution.get(lhs, 0))
        explicit_odes.append(Eq(lhs, rhs))

    return explicit_odes

def sample_splitter(data: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Split data into time, features (X), and targets (y)."""
    samples = [
        data.iloc[i * N_TIME_STEPS:(i + 1) * N_TIME_STEPS - 1].reset_index(drop=True)
        for i in range(N_SAMPLES)
    ]
    X = [sample.iloc[:, 1:-1].values for sample in samples]
    y = [sample.iloc[:, -1].values for sample in samples]
    t = samples[0]['time'].values
    return t, X, y

def get_x_dot(data: pd.DataFrame) -> list[np.ndarray]:
    """Compute derivatives of features with respect to time."""
    t, _, y = sample_splitter(data)
    samples = [
        data.iloc[i * N_TIME_STEPS:(i + 1) * N_TIME_STEPS - 1].reset_index(drop=True)
        for i in range(N_SAMPLES)
    ]
    x_dots = []
    for i, sample in enumerate(samples):
        P_u_values = sample['P_u'].values  # Replace with actual column name
        P_u_dot = np.gradient(P_u_values, t)
        P_p_dot = y[i]
        cst_dot = np.zeros(N_TIME_STEPS - 1)
        x_dot_sample = np.vstack([P_p_dot, cst_dot, P_u_dot, cst_dot, cst_dot, cst_dot, cst_dot]).T
        x_dots.append(x_dot_sample)
    return x_dots

def custom_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate custom log loss."""
    epsilon = 1e-50
    y_pred = np.clip(y_pred, epsilon, np.inf)
    y_true = np.clip(y_true, epsilon, np.inf)
    loss = (np.log10(y_pred) - np.log10(y_true))
    return np.mean(loss)

def find_best_formula(
    data: pd.DataFrame, temp_file: str, n_iterations: int = N_ITERATIONS
) -> None:
    """Run SINDy-PI to find the best formula."""
    t, X, _ = sample_splitter(data)
    x_dots = get_x_dot(data)
    
    feature_names = ['P_p', 'tK', 'P_u', 'k_cat', 'k_on', 'k_off', 'k_inact']

    # Initialize SINDy-PI model
    pde_lib = ps.PDELibrary(
        library_functions=LIBRARY_FUNCTIONS,
        function_names=FUNCTION_NAMES,
        temporal_grid=t,
        derivative_order=1,
        implicit_terms=True,
    )
    optimizer = ps.STLSQ(alpha=ALPHA, threshold=THRESHOLD)
    model = ps.SINDy(
        feature_library=pde_lib,
        optimizer=optimizer,
        feature_names=feature_names,
    )

    best_formula = None
    model.fit(X, t=t, x_dot=x_dots, multiple_trajectories=True)
    print("Calculating loss...")
    loss = model.score(X, t=t, x_dot=x_dots, multiple_trajectories=True, metric=custom_log_loss)
    print(f"Loss = {loss}")
    expressions = model.equations(precision=3)
    print(expressions)
    symbolic_formulas = convert_to_symbolic(expressions, input_features=feature_names)
    symbolic_formulas = convert_to_explicit_ode_system(symbolic_formulas, input_features=feature_names)
    
    best_formula = symbolic_formulas[0].rhs
    with open(temp_file, 'w') as f:
        f.write('Equation ' + str(best_formula))
        f.write('\nLoss ' + str(loss))

def grid_search(data: pd.DataFrame, temp_file: str) -> None:
    """Perform grid search over alpha and threshold to find the best parameters."""
    t, X, _ = sample_splitter(data)
    x_dots = get_x_dot(data)
    
    feature_names = ['P_p', 'tK', 'P_u', 'k_cat', 'k_on', 'k_off', 'k_inact']
    best_loss = float('inf')
    best_params = None
    best_formula = None
    results = []

    alphas = [1, 1e2, 1e3, 1e4, 1e5]
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    for alpha in alphas:
        for threshold in thresholds:
            print(f"Testing alpha={alpha}, threshold={threshold}")
            pde_lib = ps.PDELibrary(
                library_functions=LIBRARY_FUNCTIONS,
                function_names=FUNCTION_NAMES,
                temporal_grid=t,
                derivative_order=1,
                implicit_terms=True,
            )
            optimizer = ps.STLSQ(alpha=alpha, threshold=threshold)
            model = ps.SINDy(
                feature_library=pde_lib,
                optimizer=optimizer,
                feature_names=feature_names,
            )
            model.fit(X, t=t, x_dot=x_dots, multiple_trajectories=True)
            loss = model.score(X, t=t, x_dot=x_dots, multiple_trajectories=True, metric=custom_log_loss)
            print(f"Loss for alpha={alpha}, threshold={threshold}: {loss}")

            results.append((alpha, threshold, loss))

            if loss < best_loss:
                best_loss = loss
                best_params = (alpha, threshold)

    results.sort(key=lambda x: x[2])
    print("Ranking of parameters based on loss:")
    for rank, (alpha, threshold, loss) in enumerate(results, start=1):
        print(f"Rank {rank}: alpha={alpha}, threshold={threshold}, loss={loss}")

    print(f"Best parameters: alpha={best_params[0]}, threshold={best_params[1]}")
    print(f"Best loss: {best_loss}")

                           
def main() -> None:
    parser = argparse.ArgumentParser(description="Find the best formula using SINDy-PI")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file")
    parser.add_argument("--dataset_size", type=int, help="Number of samples to use from the dataset")
    parser.add_argument("--features", type=str, help="Comma-separated list of features to use")
    parser.add_argument("--temp_file", required=True, help="Path to save intermediate results")
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file)

if __name__ == "__main__":
    main()
