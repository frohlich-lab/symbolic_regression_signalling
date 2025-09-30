import json
import argparse
import pandas as pd
import numpy as np
import os
from dso import DeepSymbolicRegressor

import tensorflow as tf
tf.keras.backend.clear_session()  # Clears TensorFlow state

# Hyperparameters for DSO (Deep Symbolic Optimization)
N_ITERATIONS = 100
N_SAMPLES = 200
SAVE_ALL_ITERATIONS = True
FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly"]
LEARNING_RATE = 0.0005
OPTIMIZER = "adam"
POLY_REGRESSOR = "dso_least_squares"
CONFIG_FILE_PATH = "./data/dso/dso_config.json"

def create_dso_config(dataset_path):
    """Creates a JSON configuration file for DSO based on specified hyperparameters."""
    config = {
        "task": {
            "task_type": "regression",
            "dataset": dataset_path,
            "function_set": FUNCTION_SET,
            "poly_optimizer_params": {"regressor": POLY_REGRESSOR}
        },
        "policy_optimizer": {
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
        },
        "training": {
            "n_samples": N_SAMPLES,
            "batch_size": 64,
            "early_stopping" : False
        },
        "prior": {
            "const": {"on": True},
        },
        "experiment" : {
            "logdir" : "./data/dso/logs"
        }
    }
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    with open(CONFIG_FILE_PATH, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    print(f"Configuration file created at: {CONFIG_FILE_PATH}")

TARGET_COLUMN = 'kcat_cg'


def load_dataset(file_path, dataset_size=None, features=None):
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
        data = data.sample(n=min(dataset_size, len(data)))

    if not data.empty:
        input_cols = data.columns[:-1]
        data.loc[:, input_cols] = np.exp(data.loc[:, input_cols])
    return data

def run_dso_training(temp_file):
    """Runs DSO model training and logs the best equations to a specified file."""
    model = DeepSymbolicRegressor(CONFIG_FILE_PATH)
    model.setup()

    with open(temp_file, 'w') as f:
        model.train()
        f.write("Equation\tScore\n")
        new_best_program = model.trainer.p_r_best
        new_best_equation = repr(new_best_program.sympy_expr)
        print(new_best_equation)
        score = new_best_program.r
        f.write(f"{new_best_equation}\t{score:.6f}\n")

    print("DSO model training completed. Results saved.")

def main():
    parser = argparse.ArgumentParser(description='Run DSO with a JSON configuration file for symbolic regression')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save best equations during training')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dir_path = os.path.join(*args.dataset.split('/')[0:3])
    dataset_path = os.path.join("./", dir_path, "sr_comparison/dso/processed_dataset_dso.csv")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    data.to_csv(dataset_path, index=False, header=False)

    create_dso_config(dataset_path)
    run_dso_training(args.temp_file)


if __name__ == '__main__':
    main()
