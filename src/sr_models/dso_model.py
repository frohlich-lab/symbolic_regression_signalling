import json
import argparse
import pandas as pd
from dso.dso import DeepSymbolicOptimizer
import os

# Hyperparameters for DSO (Deep Symbolic Optimization)
# General Settings
N_ITERATIONS = 100
N_SAMPLES = 100
LOGDIR = "../../data/temp_results/"
SAVE_ALL_ITERATIONS = True

# Evolutionary Algorithm Settings
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 5

# Model and Learning Settings
LEARNING_RATE = 0.001
EPSILON = 0.1
OPTIMIZER = "adam"
NUM_LAYERS = 2
MAX_LENGTH = 30

# Function Set
FUNCTION_SET = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "poly"]

# Polynomial Optimization Settings
GP_MELD = False
POLY_OPTIMIZER_DEGREE = 3
POLY_COEF_TOL = 1e-6
POLY_REGRESSOR = "dso_least_squares"

def create_dso_config(file_path, output_path, dataset_path, function_set=None):
    """Create a JSON configuration file for DSO."""
    config = {
        "task": {
            "task_type": "regression",
            "dataset": dataset_path,
            "function_set": function_set or FUNCTION_SET
        },
        "training": {
            "n_samples": N_SAMPLES,
            "epsilon": EPSILON,
            "logdir": LOGDIR,
            "save_all_iterations": SAVE_ALL_ITERATIONS
        },
        "policy": {
            "max_length": MAX_LENGTH,
            "num_layers": NUM_LAYERS
        },
        "policy_optimizer": {
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER
        },
        "gp_meld": GP_MELD,
        "poly_optimizer_params": {
            "degree": POLY_OPTIMIZER_DEGREE,
            "coef_tol": POLY_COEF_TOL,
            "regressor": POLY_REGRESSOR
        },
          "logging": {
            "save_all_iterations": True  # This ensures detailed statistics are saved for every iteration
        }
    }

    # Write the configuration to a JSON file
    with open(output_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)

    print(f"Configuration file created at: {output_path}")

def load_dataset(file_path, dataset_size=None, features=None):
    """Load dataset from a CSV file, sample it if a dataset size is specified, and select specific features if provided."""
    try:
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
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

def run_dso_training(config_file, temp_file):
    """Run the DSO model training using the given configuration file and save the best equation to a file."""
    try:
        # Initialize and train the DSO model using the configuration file
        model = DeepSymbolicOptimizer(config_file)
        current_best = None

        for iteration in range(N_ITERATIONS):
            model.train_step()  # Perform a single training step
            new_best = model.get_best_equation()  # Retrieve the best equation so far

            # Check if a new best equation has been found
            if new_best != current_best:
                current_best = new_best
                # Save the new best equation to the temporary file
                with open(temp_file, 'w') as f:
                    f.write(f"Iteration {iteration}: {current_best}\n")
                print(f"New best equation found and saved: {current_best}")

        # Final save of the best equation after all iterations (to ensure consistency)
        if current_best is not None:
            with open(temp_file, 'w') as f:
                f.write(f"Best equation after {N_ITERATIONS} iterations: {current_best}\n")

        print("DSO model training completed successfully. Check the log directory for results.")
    except Exception as e:
        raise ValueError(f"Error during DSO training: {e}")

def main():
    parser = argparse.ArgumentParser(description='Find the best formula using DSO with a JSON configuration file')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--config_file', required=True, help='Path to save the DSO configuration JSON file')
    parser.add_argument('--function_set', type=str, help='Comma-separated list of functions for symbolic regression', default="add,sub,mul,div,sin,cos,exp,log,poly")
    parser.add_argument('--temp_file', required=True, help='Path to save the temporary file with the best equation found during training')

    args = parser.parse_args()

    # Load and preprocess the dataset
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dataset_path = "./processed_dataset.csv"
    data.to_csv(dataset_path, index=False)

    # Create the configuration file for DSO
    function_set = args.function_set.split(',')
    create_dso_config(args.config_file, args.config_file, dataset_path, function_set)

    # Run DSO model training using the generated configuration file and save the best equation to the temporary file
    run_dso_training(args.config_file, args.temp_file)

if __name__ == '__main__':
    main()