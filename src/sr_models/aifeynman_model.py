import numpy as np
import pandas as pd
import argparse
from aifeynman import S_run_aifeynman
import os

# Hyperparameters for AI Feynman
BF_TRY_TIME = 30  # Maximum time (in seconds) allowed for the brute-force search step
POLYFIT_DEGREE = 4  # Degree of the polynomial used in the polynomial fitting stage
NN_EPOCHS = 40  # Number of training epochs for the neural network regression stage
DATA_PATHDIR = './'  # Path to the directory containing the text format dataset
TEST_PERCENTAGE = 20  # Percentage of data to use for testing

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

def find_best_formula(data, temp_file, features):
    """Run AI Feynman to find the best formula, saving progress to a specified temporary file."""
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    data_array = np.concatenate((X, y[:, np.newaxis]), axis=1)
    np.savetxt(DATA_PATHDIR + 'mystery_mm.txt', data_array)

    feature_list = features.split(',')

    try:
        # Run the AI Feynman process and save progress to the specified temporary file
        S_run_aifeynman.run_aifeynman(
            pathdir=DATA_PATHDIR,
            filename=temp_file,  # Use the temporary file specified by Snakemake
            BF_try_time=BF_TRY_TIME,
            BF_ops_file_type='7ops.txt',
            polyfit_deg=POLYFIT_DEGREE,
            NN_epochs=int(NN_EPOCHS),
            vars_name=feature_list,
            test_percentage=TEST_PERCENTAGE,
        )
        
    except TimeoutError:
        print("AI Feynman process timed out.")

def main():
    parser = argparse.ArgumentParser(description='Find the best formula using AI Feynman')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Number of samples to use from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    parser.add_argument('--temp_file', required=True, help='Path to the temporary file to save intermediate results')

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(data, args.temp_file, args.features)

if __name__ == '__main__':
    main()