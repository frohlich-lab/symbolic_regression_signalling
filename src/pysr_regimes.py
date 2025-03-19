"""
This module provides functionality for symbolic regression using the PySR algorithm.
It evaluates different **biochemical regimes** by filtering the dataset based on feature ratios,
running PySR separately for each regime, and comparing the results against a Michaelis-Menten fit.

Usage:
    python pysr_regimes.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_dir <path_to_temp_dir>
"""

import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from scipy.optimize import curve_fit

# Define different biochemical regimes
REGIMES = {
    "regime1": lambda df, ratio: df[(df["P_u"] / df["tK"]) >= ratio],
    "regime1_1": lambda df, ratio: df[((df["k_off"] + df["k_cat"] + df["k_inact"]) / (df["k_D"] * df["k_off"])) / df["P_u"] >= ratio],
    "regime1_2": lambda df, ratio: df[(df["P_u"] / ((df["k_off"] + df["k_cat"] + df["k_inact"]) / (df["k_D"] * df["k_off"]))) >= ratio],
    "regime2": lambda df, ratio: df[(df["tK"] / df["P_u"]) >= ratio],
    "regime2_1": lambda df, ratio: df[((df["k_off"] + df["k_cat"] + df["k_inact"]) / (df["k_D"] * df["k_off"])) / df["tK"] >= ratio],
    "regime2_2": lambda df, ratio: df[((df["k_off"] + df["k_cat"] + df["k_inact"]) / (df["k_D"] * df["k_off"])) / df["tK"] >= ratio],
}

# Operators for Symbolic Regression
BINARY_OPERATORS = ["+", "*", "-", "/"]
UNARY_OPERATORS = ["exp", "log"]

def michaelis_menten(P_u, tK, k_cat, k_off, k_inact, k_on, k_D):
    """
    Michaelis-Menten equation:
    k_cat * tK * P_u / (P_u + ((k_cat + k_off + k_inact) / (k_on * k_D)))
    """
    return (k_cat * tK * P_u) / (P_u + ((k_cat + k_off + k_inact) / (k_on * k_D)))

def load_dataset(file_path, dataset_size=None, features=None):
    """
    Load dataset from a CSV file, sample it if dataset size is specified, 
    and select specific features if provided.
    """
    data = pd.read_csv(file_path)
    
    # Select specific columns if features are provided
    if features and features != "all":
        data = data[features.split(',')]

    data = data.map(np.exp)
    
    return data

def find_best_formula(data, temp_file):
    """
    Run PySR to find the best formula, saving results to a specified file.
    """
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values  # Split data into inputs (X) and output (y)

    try:
        model = PySRRegressor(
            niterations=300,
            population_size=30,
            maxsize=20,
            parsimony=1,
            binary_operators=BINARY_OPERATORS,
            unary_operators=UNARY_OPERATORS,
            verbosity=0,
            equation_file=temp_file
        )
        model.fit(X, y)
        return model.loss()
    except Exception as e:
        print(f"An error occurred during PySR training: {e}")
        return None

def evaluate_regimes(data, temp_dir, dataset_size=None):
    """
    Evaluate all biochemical regimes, save formulas, and compare loss to Michaelis-Menten fit.
    """
    os.makedirs(temp_dir, exist_ok=True)
    losses = {}
    
    for regime, filter_func in REGIMES.items():
        filtered_data = filter_func(data, ratio=1.0)
        filtered_data = filtered_data.sample(n=min(dataset_size, len(filtered_data)))
        print(f"Evaluating regime: {regime} ({len(filtered_data)} samples)")
        if filtered_data.empty:
            continue
        temp_file = os.path.join(temp_dir, f"pysr_{regime}.csv")
        loss = find_best_formula(filtered_data, temp_file)
        losses[regime] = loss
        
        # Fit Michaelis-Menten model
        try:
            y = filtered_data["k_cat_cg"].values  # Assuming "k_cat_cg" is the target column
            P_u = filtered_data["P_u"].values
            tK = filtered_data["tK"].values
            k_cat = filtered_data["k_cat"].values
            k_off = filtered_data["k_off"].values
            k_inact = filtered_data["k_inact"].values
            k_on = filtered_data["k_on"].values
            k_D = filtered_data["k_D"].values
            mm_predictions = michaelis_menten(P_u, tK, k_cat, k_off, k_inact, k_on, k_D)
            mm_loss = np.mean((mm_predictions - y) ** 2)
            losses[f"{regime}_mm"] = mm_loss
        except:
            losses[f"{regime}_mm"] = None
    
    # Prepare data for grouped bar plot
    regimes = [r for r in losses.keys() if '_mm' not in r]
    pysr_losses = [losses[r] for r in regimes]
    mm_losses = [losses[f"{r}_mm"] for r in regimes]

    x = np.arange(len(regimes))  # the label locations
    width = 0.35  # the width of the bars

    # Plot grouped bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, pysr_losses, width, label='PySR Loss', color='blue')
    plt.bar(x + width / 2, mm_losses, width, label='MM Loss', color='red')

    # Add labels, title, and legend
    plt.xlabel("Regimes")
    plt.ylabel("Loss")
    plt.title("PySR vs. Michaelis-Menten Loss across Regimes")
    plt.xticks(x, regimes, rotation=45, ha='right')
    plt.legend()

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, "regime_loss_comparison.png"))
    plt.show()

def main():
    """
    Main function to evaluate all biochemical regimes and compare against Michaelis-Menten.
    """
    parser = argparse.ArgumentParser(description='Evaluate all biochemical regimes using PySR and compare with Michaelis-Menten.')
    parser.add_argument('--dataset', required=True, help='Path to the dataset CSV file')
    parser.add_argument('--dataset_size', type=int, help='Maximum number of samples to load from the dataset')
    parser.add_argument('--features', type=str, help='Comma-separated list of features to use from the dataset')
    
    temp_dir = 'data/pysr_regimes/'

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    evaluate_regimes(data, temp_dir, args.dataset_size)

if __name__ == '__main__':
    main()
