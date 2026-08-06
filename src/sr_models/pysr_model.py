"""
This module provides functionality for symbolic regression using the PySR algorithm.
It includes functions to load datasets, find the best formula, and run the main function for command-line execution.

Usage:
    python pysr_model.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_file <path_to_temp_file>
"""

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pysr import PySRRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from shared.constants import PYSR_CONFIG, pysr_operator_config
from shared.regime_variants import VARIANTS, augment_for_variant

TARGET_COLUMN = 'kcat_cg'

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_dataset(file_path, dataset_size=None, features=None, variant: str = "sQSSA"):
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

    if variant and variant != "sQSSA":
        data = augment_for_variant(data, variant)

    return data

def find_best_formula(
    data,
    temp_file: str,
    override_config: Optional[Dict[str, object]] = None,
    seed: Optional[int] = None,
    variant: str = "sQSSA",
) -> None:
    """
    Run PySR to find the best formula, saving results to a specified file.
    """
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values  # Split data into inputs (X) and output (y)
    temp_path = Path(temp_file)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.unlink(missing_ok=True)
    run_id = f"trial_{variant}_{os.getpid()}_{int(time.time() * 1000)}"
    output_dir = temp_path.parent

    try:
        if seed is not None:
            set_seed(seed)
        # Set up and train the PySR model
        model_config = dict(PYSR_CONFIG)
        model_config.update(pysr_operator_config(variant))
        if override_config:
            model_config.update(override_config)
        # Enforce deterministic PySR runs (serial execution + fixed seed).
        model_config["deterministic"] = True
        model_config["parallelism"] = "serial"
        model_config["procs"] = 0
        for forbidden_key in ("output_directory", "run_id"):
            model_config.pop(forbidden_key, None)
        model = PySRRegressor(
            **model_config,
            output_directory=str(output_dir),
            run_id=run_id,
        )
        model.fit(X, y)
        result_csv = output_dir / run_id / "hall_of_fame.csv"
        if result_csv.exists():
            shutil.copyfile(result_csv, temp_path)
        else:
            backup = sorted(output_dir.glob(f"{run_id}*/hall_of_fame.csv"))
            if backup:
                shutil.copyfile(backup[0], temp_path)

    except TimeoutError:
        print("PySR process timed out. Try reducing the number of iterations or adjusting other hyperparameters.")
    except Exception as e:
        print(f"An error occurred during PySR training: {e}")
    finally:
        run_output = output_dir / run_id
        if run_output.exists():
            shutil.rmtree(run_output, ignore_errors=True)

def main():
    """
    Main function to parse arguments and find the best formula using PySR.
    """
    parser = argparse.ArgumentParser(description='Find the best formula using PySR')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Maximum number of samples to load from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to save the intermediate results from PySR')
    parser.add_argument('--n_iterations', type=int, help='Number of PySR iterations (overrides config)')
    parser.add_argument('--population_size', type=int, help='Population size for PySR (overrides config)')
    parser.add_argument('--max_size', type=int, help='Maximum symbolic expression size (overrides config)')
    parser.add_argument('--parsimony', type=float, help='Parsimony coefficient (overrides config)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--variant', type=str, choices=list(VARIANTS.keys()), default="sQSSA", help='Model variant determining feature preprocessing (default: sQSSA)')

    args = parser.parse_args()
    print(f"Running PySR for variant: {args.variant}")
    data = load_dataset(args.dataset, args.dataset_size, args.features, variant=args.variant)
    override_config: Dict[str, object] = {}
    if args.n_iterations is not None:
        override_config['niterations'] = args.n_iterations
    if args.population_size is not None:
        override_config['population_size'] = args.population_size
    if args.max_size is not None:
        override_config['maxsize'] = args.max_size
    if args.parsimony is not None:
        override_config['parsimony'] = args.parsimony
    if args.variant == "tQSSA" and "maxsize" not in override_config:
        override_config["maxsize"] = 45
    find_best_formula(
        data,
        args.temp_file,
        override_config=override_config or None,
        seed=args.seed,
        variant=args.variant,
    )

if __name__ == '__main__':
    main()
