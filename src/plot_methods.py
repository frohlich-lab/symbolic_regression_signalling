import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import csv

def load_formulas_from_file(file_path):
    with open(file_path, 'r') as file:
        formulas = file.readlines()
    return [formula.strip() for formula in formulas]

def compute_log_mse_loss(formula, dataset):
    # Generate symbols dynamically for each feature column (excluding the last column)
    feature_columns = dataset.columns[:-1]  # All except the last column
    feature_symbols = sp.symbols(feature_columns.tolist())  # Convert column names to sympy symbols
    target_column = dataset.columns[-1]  # The last column is the target

    # Create a lambda function that can operate on pandas DataFrame columns
    formula_func = sp.lambdify(feature_symbols, formula, 'numpy')

    # Apply the formula to the DataFrame
    predicted_values = formula_func(*[dataset[col] for col in feature_columns])

    # Compute log-space MSE
    log_predicted_values = np.log(predicted_values.clip(lower=1e-10))  # Avoid log(0) by clipping
    log_actual_values = np.log(dataset[target_column].clip(lower=1e-10))

    # Calculate MSE
    loss = np.mean((log_predicted_values - log_actual_values) ** 2)

    return loss

def calculate_complexity(formula):
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(formulas, dataset, output_file):
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
    parser = argparse.ArgumentParser(description='Plot Log-Space MSE Loss against Formula Complexity')
    parser.add_argument('--formulas', nargs='+', required=True, help='List of formulas or a file path containing formulas')
    parser.add_argument('--dataset', required=True, help='File path to a CSV file')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')

    args = parser.parse_args()

    # Load formulas from file if a file path is provided
    if os.path.isfile(args.formulas[0]):
        formulas = load_formulas_from_file(args.formulas[0])
    else:
        formulas = args.formulas

    # Convert formula strings to sympy expressions
    formulas = [sp.sympify(formula) for formula in formulas]

    # Load dataset from CSV file or provided points
    dataset = pd.read_csv(args.dataset)

    scatter_plot_formulas(formulas, dataset, args.output)

if __name__ == '__main__':
    main()