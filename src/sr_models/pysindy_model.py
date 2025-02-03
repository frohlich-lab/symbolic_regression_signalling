import numpy as np
import pandas as pd
import sympy as sp
import argparse
import pysindy as ps
from tqdm import tqdm

# Parameters
N_SAMPLES = 30
N_TIME_STEPS = 22
N_ITERATIONS = 100
ALPHA = 1e-3
THRESHOLD = 1e-20

# Define custom library functions for SINDy-PI
LIBRARY_FUNCTIONS = [
    lambda x: 1,
    lambda x: x,
    lambda x: x**(-1),
    lambda x, y: x * y,
    lambda x, y: x**(-1) * y**(-1),
    lambda x, y: x * y**(-1),
    lambda x, y, z: x * y * z,
    lambda x, y, z: x * y * z**(-1),
    lambda x, y, z: x * y**(-1) * z**(-1),
    lambda x, y, z: x**(-1) * y**(-1) * z**(-1),
]

FUNCTION_NAMES = [
    lambda x: "1",
    lambda x: x,
    lambda x: f"{x}^-1",
    lambda x, y: f"{x}*{y}",
    lambda x, y: f"{x}^-1*{y}^-1",
    lambda x, y: f"{x}*{y}^-1",
    lambda x, y, z: f"{x}*{y}*{z}",
    lambda x, y, z: f"{x}*{y}*{z}^-1",
    lambda x, y, z: f"{x}*{y}^-1*{z}^-1",
    lambda x, y, z: f"{x}^-1*{y}^-1*{z}^-1",
]

def load_dataset(
    file_path: str, dataset_size: int = None, features: str = None
) -> pd.DataFrame:
    """Load a dataset, sample it, and select specified features."""
    data = pd.read_csv(file_path)
    
    if features and features != "all":
        feature_list = ['P_p'] + features.split(',')
        data = data[['time'] + feature_list].dropna()

    if dataset_size:
        dataset_size = min(dataset_size, len(data))
        data = data.iloc[:dataset_size]

    # Apply exponential transformation
    if 'feature_list' in locals():
        data[feature_list] = np.exp(data[feature_list])
    
    return data

def convert_to_symbolic(model: ps.SINDy, input_features: list[str]) -> list[sp.Expr]:
    """Convert the SINDy-PI model to symbolic form."""
    print(model.coefficients())
    equations = model.print(precision=6)
    return [sp.sympify(equation) for equation in equations]

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
    with tqdm(total=n_iterations, desc="SINDy-PI Training Progress") as pbar:
        for iteration in range(n_iterations):
            model.fit(X, t=t, x_dot=x_dots, multiple_trajectories=True)
            symbolic_formulas = convert_to_symbolic(model, input_features=data.columns[:-1])

            if iteration % 10 == 0:
                best_formula = symbolic_formulas
                with open(temp_file, 'w') as f:
                    f.write("\n".join(str(formula) for formula in best_formula))
            pbar.update(1)

    if best_formula:
        with open(temp_file, 'w') as f:
            f.write("\n".join(str(formula) for formula in best_formula))

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