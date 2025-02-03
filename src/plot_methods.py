"""
This module provides methods for loading symbolic formulas, computing log-space MSE loss,
calculating formula complexity, and generating scatter plots of log-space MSE loss against
formula complexity.
"""

import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def load_formulas_from_file(file_path):
    """
    Load formulas from a specified file.

    Parameters:
    file_path (str): Path to the file containing formulas.

    Returns:
    list: A list of formulas as strings.
    """
    with open(file_path, 'r') as file:
        formulas = file.readlines()
    return [formula.strip() for formula in formulas]

def compute_log_mse_loss(formula, dataset):
    """
    Compute the log-space Mean Squared Error (MSE) loss for a given formula and dataset.

    Parameters:
    formula (sympy.Expr): The symbolic formula to evaluate.
    dataset (pandas.DataFrame): The dataset containing feature columns and target column.

    Returns:
    float: The computed log-space MSE loss.
    """
    feature_columns = dataset.columns[:-1]
    feature_symbols = sp.symbols(feature_columns.tolist())
    target_column = dataset.columns[-1]

    formula_func = sp.lambdify(feature_symbols, formula, 'numpy')
    predicted_values = formula_func(*[dataset[col] for col in feature_columns])

    log_predicted_values = np.log(predicted_values.clip(lower=1e-10))
    log_actual_values = np.log(dataset[target_column].clip(lower=1e-10))

    loss = np.mean((log_predicted_values - log_actual_values) ** 2)
    return loss

def calculate_complexity(formula):
    """
    Calculate the complexity of a given formula.

    Parameters:
    formula (sympy.Expr): The symbolic formula to evaluate.

    Returns:
    int: The complexity of the formula.
    """
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(formulas, dataset, output_file):
    """
    Generate a scatter plot of log-space MSE loss against formula complexity.

    Parameters:
    formulas (list): A list of symbolic formulas.
    dataset (pandas.DataFrame): The dataset containing feature columns and target column.
    output_file (str): Path to save the generated plot.
    """
    complexities = []
    losses = []

    for formula in formulas:
        complexity = calculate_complexity(formula)
        loss = compute_log_mse_loss(formula, dataset)
        complexities.append(complexity)
        losses.append(loss)

    plt.figure()
    plt.scatter(complexities, losses)
    plt.xlabel('Formula Complexity')
    plt.ylabel('Log-Space Loss (MSE)')
    plt.title('Scatter Plot of Log-Space MSE Loss vs. Formula Complexity')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    """
    Main function to parse arguments and generate the scatter plot.
    """
    parser = argparse.ArgumentParser(description='Plot Log-Space MSE Loss against Formula Complexity')
    parser.add_argument('--formulas', nargs='+', required=True, help='List of formulas or a file path containing formulas')
    parser.add_argument('--dataset', required=True, help='File path to a CSV file')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')

    args = parser.parse_args()

    if os.path.isfile(args.formulas[0]):
        formulas = load_formulas_from_file(args.formulas[0])
    else:
        formulas = args.formulas

    formulas = [sp.sympify(formula) for formula in formulas]
    dataset = pd.read_csv(args.dataset)

    scatter_plot_formulas(formulas, dataset, args.output)

if __name__ == '__main__':
    main()