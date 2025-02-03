import json
import argparse
import pandas as pd
import os
from dso import DeepSymbolicRegressor

import tensorflow as tf
tf.keras.backend.clear_session()  # Clears TensorFlow state

# Hyperparameters for DSO (Deep Symbolic Optimization)
N_ITERATIONS = 50
N_SAMPLES = 40
SAVE_ALL_ITERATIONS = True
FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly"]
LEARNING_RATE = 0.001
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
            "batch_size": 500,
        },
        "prior": {
            "const": {"on": True},
        }
    }
    os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
    with open(CONFIG_FILE_PATH, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    print(f"Configuration file created at: {CONFIG_FILE_PATH}")

def load_dataset(file_path, dataset_size=None, features=None):
    """Loads dataset from a CSV file, samples if specified, and selects specified features."""
    data = pd.read_csv(file_path)
    if features and features != "all":
        data = data[features.split(',')]
    if dataset_size:
        data = data.sample(n=min(dataset_size, len(data)))
    return data

def run_dso_training(temp_file):
    """Runs DSO model training and logs the best equations to a specified file."""
    model = DeepSymbolicRegressor(CONFIG_FILE_PATH)
    model.setup()
    current_best = None

    with open(temp_file, 'w') as f:
        f.write("Iteration\tEquation\tScore\n")
        for iteration in range(N_ITERATIONS):
            try:
                model.train_one_step()
                new_best_program = model.trainer.p_r_best
                new_best_equation = repr(new_best_program.sympy_expr)
                score = new_best_program.r

                if new_best_equation != current_best:
                    current_best = new_best_equation
                    f.write(f"{iteration}\t{current_best}\t{score:.6f}\n")
                    print(f"New best equation found at iteration {iteration}: {current_best} with score: {score}")

            except Exception as e:
                print(f"Error at iteration {iteration}: {e}")
                continue

        f.write(f"\nFinal best equation after {N_ITERATIONS} iterations:\n")
        f.write(f"Iteration: {iteration}\tEquation: {current_best}\tScore: {score:.6f}\n")
    print("DSO model training completed. Results saved.")

def main():
    parser = argparse.ArgumentParser(description='Run DSO with a JSON configuration file for symbolic regression')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use')
    parser.add_argument('--temp_file', required=True, help='Path to save best equations during training')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dataset_path = "./data/dso/processed_dataset_dso.csv"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    data.to_csv(dataset_path, index=False, header=False)

    create_dso_config(dataset_path)
    run_dso_training(args.temp_file)

if __name__ == '__main__':
    main()