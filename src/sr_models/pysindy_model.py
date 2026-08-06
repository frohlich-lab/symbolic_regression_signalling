"""
This module provides functionality for symbolic regression using the SINDy-PI algorithm.
It includes functions to load datasets, convert models to symbolic form, and find the best formula
through iterative training. The main function allows for command-line execution.

Usage:
    python pysindy_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pysindy as ps
import sympy as sp
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

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
        # Keep condition_id (trajectory grouping) and time (integration axis) so
        # SINDy operates on genuine per-condition time series rather than on
        # arbitrary fixed-length row chunks. Mixing conditions produced
        # non-monotonic time and NaN finite-difference derivatives.
        cols_to_use = [c for c in ('condition_id', 'time') if c in data.columns]
        cols_to_use += feature_list
        if target not in cols_to_use:
            cols_to_use.append(target)
        data = data[cols_to_use].dropna()
        # The datasets encode a steady-state sample as time=inf (t -> infinity).
        # An infinite time gap makes finite-difference derivatives NaN, so drop
        # any non-finite time rows before building trajectories.
        if 'time' in data.columns:
            data = data[np.isfinite(data['time'])]

    # Sample whole trajectories (by condition), never mid-trajectory rows.
    if dataset_size and 'condition_id' in data.columns:
        approx_traj_len = max(1, int(data.groupby('condition_id').size().median()))
        n_conditions = max(1, dataset_size // approx_traj_len)
        keep_ids = data['condition_id'].drop_duplicates().iloc[:n_conditions]
        data = data[data['condition_id'].isin(keep_ids)]
    elif dataset_size:
        dataset_size = min(dataset_size, len(data))
        data = data.iloc[:dataset_size]

    # Apply exponential transformation to the feature columns only.
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

CHUNK_LENGTH = N_TIME_STEPS - 1


def _enforce_increasing_time(chunk: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a strictly increasing 'time' column (or a synthetic one)."""
    chunk = chunk.reset_index(drop=True)
    if "time" in chunk.columns:
        t_values = chunk["time"].to_numpy(dtype=float, copy=True)
        for j in range(1, len(t_values)):
            prev = t_values[j - 1]
            if t_values[j] <= prev:
                step = max(1e-9, abs(prev) * 1e-9)
                t_values[j] = prev + step
        chunk["time"] = t_values
    else:
        chunk["time"] = np.linspace(0.0, float(len(chunk) - 1), num=len(chunk))
    return chunk


def _build_samples(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Create per-trajectory data slices with a strictly increasing time grid.

    When ``condition_id`` is present each condition is one trajectory (sorted by
    time); the id column is dropped so the layout is [time, features..., target].
    Otherwise we fall back to fixed-length row chunks.
    """
    samples: list[pd.DataFrame] = []
    if data.empty:
        return samples

    if "condition_id" in data.columns:
        for _, group in data.groupby("condition_id", sort=False):
            if len(group) < 2:
                continue
            group = group.drop(columns=["condition_id"])
            if "time" in group.columns:
                group = group.sort_values("time")
            samples.append(_enforce_increasing_time(group))
        return samples

    if "time" in data.columns:
        data = data.sort_values("time").reset_index(drop=True)

    total_rows = len(data)
    n_chunks = total_rows // CHUNK_LENGTH

    for idx in range(n_chunks):
        start = idx * CHUNK_LENGTH
        end = start + CHUNK_LENGTH
        chunk = data.iloc[start:end].copy()
        if len(chunk) < 2:
            continue
        samples.append(_enforce_increasing_time(chunk))

    return samples


def sample_splitter(data: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Split data into time, features (X), and targets (y)."""
    samples = _build_samples(data)
    if not samples:
        raise ValueError("PySINDy received no valid trajectory segments.")
    X = [sample.iloc[:, 1:-1].values for sample in samples]
    y = [sample.iloc[:, -1].values for sample in samples]
    t = samples[0]['time'].values
    return t, X, y


def _sanitize_edge_order(order: Optional[int]) -> int:
    """
    Translate the requested finite difference order into a valid numpy.gradient edge_order.

    numpy only supports edge orders 1 and 2; fall back gracefully if the sweep requests a
    different value.
    """
    if order is None or order <= 1:
        return 1
    return 2


def get_x_dot(data: pd.DataFrame, finite_difference_order: Optional[int] = None) -> list[np.ndarray]:
    """Compute derivatives of features with respect to time."""
    samples = _build_samples(data)
    if not samples:
        raise ValueError("PySINDy received no valid trajectory segments for derivative estimation.")
    edge_order = _sanitize_edge_order(finite_difference_order)
    x_dots = []
    for sample in samples:
        time = sample["time"].to_numpy(dtype=float, copy=False)
        feature_matrix = sample.iloc[:, 1:-1].to_numpy(dtype=float, copy=True)
        derivatives = np.gradient(feature_matrix, time, axis=0, edge_order=edge_order)
        x_dots.append(derivatives)
    return x_dots

def custom_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate custom log loss."""
    epsilon = 1e-50
    y_pred = np.clip(y_pred, epsilon, np.inf)
    y_true = np.clip(y_true, epsilon, np.inf)
    loss = (np.log10(y_pred) - np.log10(y_true))
    return np.mean(loss)

def _feature_names_from_data(data: pd.DataFrame) -> list[str]:
    """Derive SINDy feature names from the dataframe columns.

    ``sample_splitter`` treats column 0 as the time axis and the last column as
    the target, using ``iloc[:, 1:-1]`` as the feature matrix. The feature names
    handed to PySINDy must line up with exactly those columns, otherwise PySINDy
    raises a shape mismatch. The previous hard-coded list (7 names) never matched
    the configured features and broke every run.
    """
    return list(data.columns[1:-1])


def find_best_formula(
    data: pd.DataFrame, temp_file: str, n_iterations: int = N_ITERATIONS
) -> None:
    """Run SINDy-PI to find the best formula."""
    t, X, _ = sample_splitter(data)
    x_dots = get_x_dot(data)

    feature_names = _feature_names_from_data(data)

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

def grid_search(
    data: pd.DataFrame,
    temp_file: str,
    alpha_override: Optional[float] = None,
    threshold_override: Optional[float] = None,
    finite_difference_order_override: Optional[int] = None,
) -> None:
    """Perform grid search over alpha and threshold to find the best parameters."""
    t, X, _ = sample_splitter(data)
    x_dots = get_x_dot(data, finite_difference_order_override)

    feature_names = _feature_names_from_data(data)
    best_loss = float('inf')
    best_params = None
    best_formula = None
    results = []

    if alpha_override is not None:
        alphas = [alpha_override]
    else:
        alphas = [1, 1e2, 1e3, 1e4, 1e5]

    if threshold_override is not None:
        thresholds = [threshold_override]
    else:
        thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    finite_order = _sanitize_edge_order(finite_difference_order_override)

    for alpha in alphas:
        for threshold in thresholds:
            print(f"Testing alpha={alpha}, threshold={threshold}")
            pde_lib = ps.PDELibrary(
                library_functions=LIBRARY_FUNCTIONS,
                function_names=FUNCTION_NAMES,
                temporal_grid=t,
                derivative_order=finite_order,
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

    if best_params is None:
        Path(temp_file).write_text("")
        print("No valid parameters produced a result.")
        return

    print(f"Best parameters: alpha={best_params[0]}, threshold={best_params[1]}")
    print(f"Best loss: {best_loss}")

    # Refit with best parameters and write out formula/score
    best_alpha, best_threshold = best_params
    pde_lib = ps.PDELibrary(
        library_functions=LIBRARY_FUNCTIONS,
        function_names=FUNCTION_NAMES,
        temporal_grid=t,
        derivative_order=finite_order,
        implicit_terms=True,
    )
    optimizer = ps.STLSQ(alpha=best_alpha, threshold=best_threshold)
    model = ps.SINDy(
        feature_library=pde_lib,
        optimizer=optimizer,
        feature_names=feature_names,
    )
    model.fit(X, t=t, x_dot=x_dots, multiple_trajectories=True)
    final_loss = model.score(X, t=t, x_dot=x_dots, multiple_trajectories=True, metric=custom_log_loss)
    expressions = model.equations(precision=3)
    symbolic_formulas = convert_to_symbolic(expressions, input_features=feature_names)
    symbolic_formulas = convert_to_explicit_ode_system(symbolic_formulas, input_features=feature_names)

    best_formula = symbolic_formulas[0].rhs
    with open(temp_file, 'w') as handle:
        handle.write(f"Equation {best_formula}\n")
        handle.write(f"Loss {final_loss}\n")

                           
def main() -> None:
    parser = argparse.ArgumentParser(description="Find the best formula using SINDy-PI")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file")
    parser.add_argument("--dataset_size", type=int, help="Number of samples to use from the dataset")
    parser.add_argument("--features", type=str, help="Comma-separated list of features to use")
    parser.add_argument("--temp_file", required=True, help="Path to save intermediate results")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--alpha", type=float, help="STLSQ alpha value to evaluate")
    parser.add_argument("--threshold", type=float, help="STLSQ threshold value to evaluate")
    parser.add_argument(
        "--finite_difference_order",
        type=int,
        help="Edge order used while estimating derivatives (1 or 2).",
    )

    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    data = load_dataset(args.dataset, args.dataset_size, args.features)
    grid_search(
        data,
        args.temp_file,
        alpha_override=args.alpha,
        threshold_override=args.threshold,
        finite_difference_order_override=args.finite_difference_order,
    )


if __name__ == "__main__":
    main()
