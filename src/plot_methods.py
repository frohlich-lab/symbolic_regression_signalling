"""
This module provides methods for loading symbolic formulas, computing log-space MAE loss,
calculating formula complexity, and generating scatter plots of log-space MAE loss against
formula complexity.
"""

import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
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

def compute_log_MAE_loss(formula, dataset, discovery_scale):
    """Compute the mean squared error (MAE) loss between predictions of the formula and the target in log-space data."""
    formula_symbols = formula.free_symbols
    formula_variable_names = [str(symbol) for symbol in formula_symbols]  # Extract variable names in formula
    relevant_columns = [col for col in dataset.columns if col in formula_variable_names]

    feature_symbols = sp.symbols(relevant_columns)
    target_column = dataset.columns[-1]  # Assuming the last column is the target

    # Lambdify the formula function for relevant columns only
    formula_func = sp.lambdify(feature_symbols, formula, modules=['numpy', 'sympy'])

    # Calculate predictions
    predicted_values = formula_func(*[dataset[col].astype(float) for col in relevant_columns])

    # Compute MAE directly
    if discovery_scale == 'log':
        loss = np.mean(np.abs(predicted_values - dataset[target_column]))
    else:
        loss = np.mean(np.abs(np.log10(np.clip(predicted_values, a_min=1e-20, a_max=None) - np.log10(dataset[target_column]))))
    return loss

def calculate_complexity(formula):
    """Calculate the complexity of a formula based on the number of elements."""
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(methods, formulas, discovery_scales, dataset, output_file):
    """Generate a scatter plot of log-space MAE loss versus formula complexity and save it to a file."""
    complexities, losses = [], []

    for method, formula in zip(methods, formulas):
        complexity = calculate_complexity(formula)
        if discovery_scales[method] == 'log':
            loss = compute_log_MAE_loss(formula, dataset, discovery_scale='log')
        elif discovery_scales[method] == 'linear':
            dataset_linear = dataset.apply(lambda x: np.exp(x) if x.name in dataset.columns[2:] else x)
            loss = compute_log_MAE_loss(formula, dataset_linear, discovery_scale='linear')
        else:
            raise ValueError(f"Unsupported discovery scale: {discovery_scales[method]}")
        complexities.append(complexity)
        losses.append(loss)

    plt.figure()
    plt.scatter(complexities, losses)
    for i, method in enumerate(methods):
        plt.annotate(method, (complexities[i], losses[i]), fontsize=8)
    plt.xlabel('Formula Complexity')
    plt.ylabel('Log-Space Loss (MAE)')
    plt.title('Scatter Plot of Log-Space MAE Loss vs. Formula Complexity')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    """
    Main function to parse arguments and generate the scatter plot.
    """
    print('STARTED')
    parser = argparse.ArgumentParser(description='Plot Log-Space MAE Loss against Formula Complexity')
    parser.add_argument('--formulas', nargs='+', required=True, help='List of file paths containing formulas')
    parser.add_argument('--dataset', required=True, help='File path to a CSV dataset')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')
    parser.add_argument('--discovery-scales', required=True, type=str, help='JSON of method names and their corresponding discovery scales')
 
    args = parser.parse_args()
    methods = [formula.split('/')[-1].split('_')[1].split('.')[0] for formula in args.formulas]

    # Load all formulas from the provided file paths
    formulas = []
    for file_path in args.formulas:
        with open(file_path, 'r') as f:
            formula = f.readline().strip()  # Assuming each file contains only one formula
            formulas.append(sp.sympify(formula))  # Convert to sympy expression

    # Load dataset from CSV file
    dataset = pd.read_csv(args.dataset)
    discovery_scales = json.loads(args.discovery_scales)

    # Generate and save the scatter plot
    scatter_plot_formulas(methods, formulas, discovery_scales, dataset, args.output)

if __name__ == '__main__':
    main()
