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
    """Compute the mean squared error (MSE) loss between predictions of the formula and the target in log-space data."""
    formula_symbols = formula.free_symbols
    formula_variable_names = [str(symbol) for symbol in formula_symbols]  # Extract variable names in formula
    relevant_columns = [col for col in dataset.columns if col in formula_variable_names]

    feature_symbols = sp.symbols(relevant_columns)
    target_column = dataset.columns[-1]  # Assuming the last column is the target

    # Lambdify the formula function for relevant columns only
    formula_func = sp.lambdify(feature_symbols, formula, modules=['numpy', 'sympy'])

    # Calculate predictions
    predicted_values = formula_func(*[dataset[col].astype(float) for col in relevant_columns])

    # Compute MSE directly
    loss = np.mean((predicted_values - dataset[target_column]) ** 2)
    return loss

def calculate_complexity(formula):
    """Calculate the complexity of a formula based on the number of elements."""
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(formulas, dataset, output_file):
    """Generate a scatter plot of log-space MSE loss versus formula complexity and save it to a file."""
    complexities, losses = [], []

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
    parser.add_argument('--dataset', required=True, help='File path to a CSV dataset')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')

    args = parser.parse_args()

    # Determine if formulas are in a file or provided directly
    if os.path.isfile(args.formulas[0]):
        formulas = load_formulas_from_file(args.formulas[0])
    else:
        formulas = args.formulas

    # Convert formulas to sympy expressions
    formulas = [sp.sympify(formula) for formula in formulas]

    # Load dataset from CSV file
    dataset = pd.read_csv(args.dataset)

    # Generate and save the scatter plot
    scatter_plot_formulas(formulas, dataset, args.output)

if __name__ == '__main__':
    main()