"""This module provides functions to extract the best formula from various symbolic regression methods' hall of fame files."""

import argparse
import os
import sympy as sp
import pandas as pd

def get_best_formula_from_pysindy(hall_of_fame_file):
    """
    Extract the best formula from PySINDy's output file.

    Parameters:
    hall_of_fame_file (str): Path to the PySINDy hall of fame file.

    Returns:
    str: The best formula as a string, or None if not found.
    """
    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            formula = f.readline().strip()
        return formula
    return None

def get_best_formula_from_aifeynman(hall_of_fame_file, feature_names):
    """
    Extract the best formula from AI Feynman's output file based on the lowest error.

    Parameters:
    hall_of_fame_file (str): Path to the AI Feynman hall of fame file.
    feature_names (str): Comma-separated feature names.

    Returns:
    sympy.Expr: The best formula as a SymPy expression, or None if not found.
    """
    best_formula = None
    best_error = float('inf')

    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    error = float(parts[3])
                    formula = " ".join(parts[4:])
                    if error < best_error:
                        best_error = error
                        best_formula = formula
                except ValueError:
                    continue

    if best_formula:
        try:
            best_formula = convert_x_i_variables(best_formula, feature_names, start_index=1)
            sympy_formula = sp.sympify(best_formula)
            return sympy_formula
        except (sp.SympifyError, ValueError):
            print("Failed to convert the formula to a SymPy expression.")
            return None
    return None

def get_best_formula_from_dso(hall_of_fame_file):
    """
    Extract the best formula from the DSO hall of fame CSV file based on the lowest score.

    Parameters:
    hall_of_fame_file (str): Path to the DSO hall of fame file.

    Returns:
    str: The best formula as a string, or None if not found.
    """
    if os.path.exists(hall_of_fame_file):
        hall_of_fame = pd.read_csv(hall_of_fame_file)
        if not hall_of_fame.empty and 'score' in hall_of_fame.columns and 'equation' in hall_of_fame.columns:
            best_formula_row = hall_of_fame.sort_values(by='score').iloc[0]
            best_formula = best_formula_row["equation"]
            return best_formula
    return None

def convert_x_i_variables(formula, feature_names, start_index=0, underscore=True):
    """
    Convert x_i variables in the formula to real feature names.

    Parameters:
    formula (str): The formula containing x_i variables.
    feature_names (str): Comma-separated feature names.
    start_index (int): Starting index for x_i variables.
    underscore (bool): Whether to use underscore in variable names.

    Returns:
    str: The formula with x_i variables replaced by feature names.
    """
    features_names_list = feature_names.split(',')
    for i, feature_name in enumerate(features_names_list):
        if underscore:
            formula = formula.replace(f"x_{i+start_index}", feature_name)
        else:
            formula = formula.replace(f"x{i+start_index}", feature_name)
    return formula

def get_best_formula_from_kan(hall_of_fame_file, feature_names):
    """
    Extract and convert the best formula from KAN's output file based on the lowest loss.

    Parameters:
    hall_of_fame_file (str): Path to the KAN hall of fame file.
    feature_names (str): Comma-separated feature names.

    Returns:
    str: The best formula as a string, or None if not found.
    """
    best_formula = None
    best_loss = float('inf')
    formula_found = False

    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Iteration"):
                    i += 1
                    continue
                try:
                    if "Formula:" in line:
                        formula_part = line.split("Formula: ")[1].strip()
                        i += 1
                        loss_line = lines[i].strip()
                        if "Loss:" in loss_line:
                            loss_value = float(loss_line.split("Loss: ")[1].strip())
                            formula_found = True
                            if loss_value < best_loss:
                                best_loss = loss_value
                                best_formula = formula_part
                except (IndexError, ValueError) as e:
                    print(f"Skipping KAN hall of fame line read due to parsing error: {e}")
                i += 1

    if not formula_found:
        raise ValueError("No valid formulas found in the hall of fame file.")
    
    return convert_x_i_variables(best_formula, feature_names)

def get_best_formula_from_pysr(hall_of_fame_file, feature_names):
    """
    Extract the best formula from PySR's hall of fame CSV file based on the lowest loss.

    Parameters:
    hall_of_fame_file (str): Path to the PySR hall of fame file.
    feature_names (str): Comma-separated feature names.

    Returns:
    str: The best formula as a string, or None if not found.
    """
    if os.path.exists(hall_of_fame_file):
        hall_of_fame = pd.read_csv(hall_of_fame_file)
        if not hall_of_fame.empty and 'Loss' in hall_of_fame.columns:
            best_formula = hall_of_fame.sort_values(by='Loss').iloc[0]["Equation"]
            return convert_x_i_variables(best_formula, feature_names, underscore=False)
    return None

def main():
    """
    Main function to extract the best formula from each hall of fame file based on the score.
    """
    parser = argparse.ArgumentParser(description='Extract the best formula from each hall of fame file based on the score')
    parser.add_argument('--hall_of_fame', nargs='+', required=True, help='Paths to the hall of fame files, one for each method')
    parser.add_argument('--save', nargs='+', required=True, help='Paths to save the best formulas, one for each method')
    parser.add_argument('--features', required=True, help='Feature names')
    parser.add_argument('--methods', nargs='+', required=True, help='List of symbolic regression methods to use (e.g., pysindy, aifeynman, dso, kan, pysr)')

    args = parser.parse_args()

    if not (len(args.methods) == len(args.hall_of_fame) == len(args.save)):
        raise ValueError("The number of methods, hall of fame files, and save paths must be the same.")

    for method, hall_of_fame_path, save_path in zip(args.methods, args.hall_of_fame, args.save):
        method = method.lower()
        if method == 'pysindy':
            best_formula = get_best_formula_from_pysindy(hall_of_fame_path)
        elif method == 'aifeynman':
            best_formula = get_best_formula_from_aifeynman(hall_of_fame_path, args.features)
        elif method == 'dso':
            best_formula = get_best_formula_from_dso(hall_of_fame_path)
        elif method == 'kan':
            best_formula = get_best_formula_from_kan(hall_of_fame_path, args.features)
        elif method == 'pysr':
            best_formula = get_best_formula_from_pysr(hall_of_fame_path, args.features)
        else:
            print(f"Unknown method: {method}")
            continue

        if best_formula:
            with open(save_path, 'w') as file:
                file.write(best_formula)
            print(f"Best formula for {method} saved to {save_path}: {best_formula}")
        else:
            print(f"No valid formula found in the hall of fame file for {method}.")

if __name__ == '__main__':
    main()