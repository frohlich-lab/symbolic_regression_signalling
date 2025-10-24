"""
This modules leverages the AI Feynman library to perform symbolic regression on a given dataset.
It allows for dataset sampling, feature selection, and saves intermediate results to a specified file.

Usage:
    python aifeynman_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import argparse
import random
import pandas as pd
import os
import shutil
import sys
from pathlib import Path
import tempfile
import numpy as np

AI_FEYNMAN_IMPORT_ERROR = None
try:
    from aifeynman import S_run_aifeynman
except ImportError as exc:  # pragma: no cover - optional dependency
    repo_root = Path(__file__).resolve().parents[2]
    local_pkg = repo_root / "src" / "aifeynman"
    if local_pkg.exists():
        sys.path.insert(0, str(local_pkg))
        try:
            from aifeynman import S_run_aifeynman  # type: ignore
        except ImportError as inner_exc:  # pragma: no cover - defensive
            AI_FEYNMAN_IMPORT_ERROR = inner_exc
    else:  # pragma: no cover - defensive
        AI_FEYNMAN_IMPORT_ERROR = exc

TARGET_COLUMN = 'kcat_cg'

# Hyperparameters for AI Feynman
OPERATORS = '+*-D~ILEA'  # Operators used in symbolic regression
BF_TRY_TIME = 30  # Max time (seconds) for brute-force search step
POLYFIT_DEGREE = 4  # Degree for polynomial fitting
NN_EPOCHS = 40  # Training epochs for neural network stage
DATA_PATHDIR = './data/aifeynman/'  # Directory for dataset
TEST_PERCENTAGE = 20  # Percentage of data for testing
FILENAME = 'mystery.txt'  # Dataset file name

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(file_path, dataset_size=None, features=None, seed=None):
    """Loads a dataset from CSV, samples it if specified, and selects columns if features are provided."""
    data = pd.read_csv(file_path)
    target = TARGET_COLUMN if TARGET_COLUMN in data.columns else data.columns[-1]
    if features and features != "all":
        requested = [col.strip() for col in features.split(',')]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for AI Feynman: {missing}. Using available columns {available}.")
        selected = available + ([target] if target not in available else [])
    else:
        selected = list(data.columns)
    data = data[selected]
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)), random_state=seed)

    # Convert inputs back to linear scale while leaving the target in log space
    if not data.empty:
        input_cols = data.columns[:-1]
        data.loc[:, input_cols] = np.exp(data.loc[:, input_cols])
    return data

def run_feynman(
    data,
    temp_file,
    features,
    bf_try_time: int,
    nn_epochs: int,
    polyfit_degree: int,
) -> None:
    """Runs AI Feynman to find the best formula, saving results to a specified file."""
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    data_array = np.concatenate((X, y[:, np.newaxis]), axis=1)
    os.makedirs(DATA_PATHDIR, exist_ok=True)
    np.savetxt(os.path.join(DATA_PATHDIR, FILENAME), data_array)
    np.savetxt(os.path.join(DATA_PATHDIR, '7ops.txt'), [OPERATORS], fmt='%s')

    try:
        # Run AI Feynman with specified parameters
        variable_names = (
            [name.strip() for name in features.split(',')]
            if features
            else [f"x_{idx}" for idx in range(X.shape[1])]
        )
        S_run_aifeynman.run_aifeynman(
            pathdir=DATA_PATHDIR,
            filename=FILENAME,
            BF_try_time=bf_try_time,
            BF_ops_file_type='7ops.txt',
            polyfit_deg=polyfit_degree,
            NN_epochs=nn_epochs,
            vars_name=variable_names,
            test_percentage=TEST_PERCENTAGE,
        )
        copy_solution_to_temp(os.path.join(DATA_PATHDIR, 'results', f'solution_{FILENAME}'), temp_file)

    except TimeoutError:
        print("AI Feynman process timed out.")
    except FileNotFoundError:
        print("Solution file not found. Check if AI Feynman completed successfully.")
    except Exception as e:
        print(f"Unexpected error during AI Feynman execution: {e}")

def copy_solution_to_temp(solution_file, temp_file):
    """Copies solution file to the specified temp file if it exists."""
    try:
        shutil.copy(solution_file, temp_file)
        print(f"Solution saved to {temp_file}")
    except FileNotFoundError:
        print(f"Solution file {solution_file} not found.")
    except Exception as e:
        print(f"Error copying solution file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run AI Feynman for symbolic regression')
    parser.add_argument('--dataset', required=True, help='Path to dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Sample size from dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save intermediate results')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--bf_try_time', type=int, help='Override brute-force search time (seconds)')
    parser.add_argument('--nn_epochs', type=int, help='Override neural network training epochs')
    parser.add_argument(
        '--polyfit_degree',
        type=int,
        help='Override polynomial fit degree used by AI Feynman',
    )

    args = parser.parse_args()
    if AI_FEYNMAN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import the AI Feynman package from either the environment or src/aifeynman.\n"
            f"Original error: {AI_FEYNMAN_IMPORT_ERROR}"
        )

    if args.seed is not None:
        seed_everything(args.seed)
    data = load_dataset(args.dataset, args.dataset_size, args.features, args.seed)
    run_feynman(
        data,
        args.temp_file,
        args.features,
        bf_try_time=args.bf_try_time or BF_TRY_TIME,
        nn_epochs=args.nn_epochs or NN_EPOCHS,
        polyfit_degree=args.polyfit_degree or POLYFIT_DEGREE,
    )

if __name__ == '__main__':
    main()
