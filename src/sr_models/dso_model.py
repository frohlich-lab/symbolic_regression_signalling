import argparse
import json
import os
import random
import signal
import sys
from pathlib import Path
import numpy as np
import pandas as pd


class _TimeoutReached(Exception):
    """Raised from the SIGTERM handler so training can save its best-so-far."""


def _install_sigterm_salvage():
    """Turn an external `timeout` SIGTERM into an exception so best-so-far is kept."""
    def _handler(signum, frame):
        raise _TimeoutReached(f"received signal {signum}")
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):  # pragma: no cover - not in main thread
        pass

DSO_IMPORT_ERROR = None
try:
    from dso import DeepSymbolicRegressor
except Exception as exc:  # pragma: no cover - defensive
    repo_root = Path(__file__).resolve().parents[2]
    dso_path = repo_root / "src" / "dso"
    sys.path.insert(0, str(dso_path))
    try:
        from dso import DeepSymbolicRegressor
    except Exception as inner_exc:  # pragma: no cover - defensive
        DSO_IMPORT_ERROR = inner_exc
        DeepSymbolicRegressor = None

import tensorflow as tf
tf.keras.backend.clear_session()  # Clears TensorFlow state

# Hyperparameters for DSO (Deep Symbolic Optimization)
# The RL policy needs thousands of iterations to converge; 100 barely leaves
# random initialisation. Combined with the SIGTERM salvage below (best-so-far is
# written if the wall-clock timeout fires), a large value is safe.
DEFAULT_N_ITERATIONS = 2000
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_ENTROPY_WEIGHT = 0.03
DEFAULT_ENTROPY_GAMMA = 0.7
SAVE_ALL_ITERATIONS = True
FUNCTION_SET = ["add", "sub", "mul", "div", "log"]  # rational + log only (drop sin/cos/exp/poly)
OPTIMIZER = "adam"
POLY_REGRESSOR = "dso_least_squares"
CONFIG_FILE_PATH = "./data/dso/dso_config.json"

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except AttributeError:
        pass

def create_dso_config(
    dataset_path: str,
    n_iterations: int,
    batch_size: int,
    learning_rate: float,
    entropy_weight: float,
    entropy_gamma: float,
    seed: int = 0,
) -> None:
    """Creates a JSON configuration file for DSO based on specified hyperparameters."""
    effective_batch_size = max(1, batch_size)
    total_samples = max(1, n_iterations) * effective_batch_size

    config = {
        "task": {
            "task_type": "regression",
            "dataset": dataset_path,
            "function_set": FUNCTION_SET,
            "poly_optimizer_params": {"regressor": POLY_REGRESSOR}
        },
        "policy_optimizer": {
            "learning_rate": learning_rate,
            "optimizer": OPTIMIZER,
            "entropy_weight": entropy_weight,
            "entropy_gamma": entropy_gamma,
        },
        "training": {
            "n_samples": total_samples,
            "batch_size": effective_batch_size,
            "early_stopping" : False
        },
        "prior": {
            "const": {"on": True},
        },
        "experiment" : {
            "logdir" : "./data/dso/logs",
            "seed" : seed,
        }
    }
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    with open(CONFIG_FILE_PATH, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    print(f"Configuration file created at: {CONFIG_FILE_PATH}")

TARGET_COLUMN = 'kcat_cg'


def load_dataset(file_path, dataset_size=None, features=None, seed=None):
    """Loads dataset from a CSV file, samples if specified, and selects specified features."""
    data = pd.read_csv(file_path)
    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]
    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for DSO: {missing}. Using available columns {available}.")
        selected = available + ([target] if target not in available else [])
    else:
        selected = list(data.columns)
    data = data[selected]
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)), random_state=seed)

    if not data.empty:
        input_cols = data.columns[:-1]
        data.loc[:, input_cols] = np.exp(data.loc[:, input_cols])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("DSO received an empty dataset after cleaning.")
    return data

def _write_best_program(best_program, temp_file: str) -> None:
    if best_program is None:
        Path(temp_file).write_text("")
        print("DSO produced no valid expression.")
        return
    with open(temp_file, 'w') as f:
        f.write("Equation\tScore\n")
        best_equation = repr(best_program.sympy_expr)
        print(best_equation)
        score = best_program.r
        f.write(f"{best_equation}\t{score:.6f}\n")


def run_dso_training(temp_file: str) -> None:
    """Runs DSO model training and logs the best equations to a specified file."""
    model = DeepSymbolicRegressor(CONFIG_FILE_PATH)
    # DeepSymbolicOptimizer.train() calls setup() itself (tf.reset_default_graph
    # + new session); calling setup() here as well would build the graph twice.
    try:
        model.train()
    except _TimeoutReached as exc:
        # `timeout <N>` fired; keep the best expression discovered so far instead
        # of losing the whole (now much longer) run.
        print(f"DSO training interrupted ({exc}); saving best-so-far expression.")
    best_program = getattr(getattr(model, "trainer", None), "p_r_best", None)
    _write_best_program(best_program, temp_file)
    print("DSO model training completed. Results saved.")

def main():
    parser = argparse.ArgumentParser(description='Run DSO with a JSON configuration file for symbolic regression')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save best equations during training')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--n_iterations', type=int, help='Number of DSO training iterations')
    parser.add_argument('--n_samples', type=int, help='Batch size used per DSO iteration')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the policy optimizer')
    parser.add_argument('--entropy_weight', type=float, help='Entropy regularisation weight for policy optimizer')
    parser.add_argument('--entropy_gamma', type=float, help='Entropy decay factor for policy optimizer')

    args = parser.parse_args()
    if DSO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import the local DSO package. Please ensure the conda "
            "environment provides a compatible tensorflow/absl stack. "
            f"Original error: {DSO_IMPORT_ERROR}"
        )

    _install_sigterm_salvage()
    if args.seed is not None:
        seed_everything(args.seed)
    data = load_dataset(args.dataset, args.dataset_size, args.features, args.seed)
    dir_path = os.path.join(*args.dataset.split('/')[0:3])
    dataset_path = os.path.join("./", dir_path, "sr_comparison/dso/processed_dataset_dso.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    data.to_csv(dataset_path, index=False, header=False)

    create_dso_config(
        dataset_path=dataset_path,
        n_iterations=args.n_iterations or DEFAULT_N_ITERATIONS,
        batch_size=args.n_samples or DEFAULT_BATCH_SIZE,
        learning_rate=args.learning_rate or DEFAULT_LEARNING_RATE,
        entropy_weight=args.entropy_weight or DEFAULT_ENTROPY_WEIGHT,
        entropy_gamma=args.entropy_gamma or DEFAULT_ENTROPY_GAMMA,
        seed=args.seed if args.seed is not None else 0,
    )
    run_dso_training(args.temp_file)


if __name__ == '__main__':
    main()
