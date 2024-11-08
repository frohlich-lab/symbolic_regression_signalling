import argparse
import os
import sympy as sp
import pandas as pd

def get_best_formula_from_pysindy(hall_of_fame_file):
    """Extract the best formula from PySINDy's output file."""
    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            formula = f.readline().strip()  # PySINDy might output the formula directly as plain text
        return formula
    return None

def get_best_formula_from_aifeynman(hall_of_fame_file, feature_names):
    """Extract the best formula from AI Feynman's output file based on the lowest error and return as a SymPy expression."""
    best_formula = None
    best_error = float('inf')  # Initialize with a large value for error

    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts) < 5:
                    continue

                try:
                    error = float(parts[3])  # Extract the error from the 4th column
                    formula = " ".join(parts[4:])  # Extract the formula starting from the 5th column onwards

                    # Check if the current formula has a lower error
                    if error < best_error:
                        best_error = error
                        best_formula = formula
                        
                except ValueError:
                    continue  # Skip lines that can't be parsed

    if best_formula:
        try:
            # Convert the formula to a SymPy expression
            best_formula = convert_x_i_variables(best_formula, feature_names, start_index=1)
            sympy_formula = sp.sympify(best_formula)
            return sympy_formula
        except (sp.SympifyError, ValueError):
            print("Failed to convert the formula to a SymPy expression.")
            return None
    return None


def get_best_formula_from_dso(hall_of_fame_file):
    """Extract the best formula from the DSO hall of fame CSV file, based on the lowest score."""
    if os.path.exists(hall_of_fame_file):
        hall_of_fame = pd.read_csv(hall_of_fame_file)
        if not hall_of_fame.empty and 'score' in hall_of_fame.columns and 'equation' in hall_of_fame.columns:
            # Sort by score to find the best formula with the lowest score
            best_formula_row = hall_of_fame.sort_values(by='score').iloc[0]
            best_formula = best_formula_row["equation"]
            # Return the best formula
            return best_formula

    return None

def convert_x_i_variables(formula, feature_names, start_index=0, underscore=True):
    """Convert x_i variables in the KAN formula to real feature names."""
    features_names_list = feature_names.split(',')
    for i, feature_name in enumerate(features_names_list):
        if underscore:
            formula = formula.replace(f"x_{i+start_index}", feature_name)
        else:
            formula = formula.replace(f"x{i+start_index}", feature_name)
    return formula

def get_best_formula_from_kan(hall_of_fame_file, feature_names):
    """Extract and convert the best formula from KAN's output file based on the lowest loss across multiple iterations."""
    best_formula = None
    best_loss = float('inf')  # Initialize with a large loss value
    formula_found = False

    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            lines = f.readlines()  # Read all lines at once for easier sequential processing
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Iteration"):
                    i += 1
                    continue  # Skip iteration headers

                # Attempt to extract the formula and check the next line for the loss
                try:
                    if "Formula:" in line:
                        # Extract the formula part
                        formula_part = line.split("Formula: ")[1].strip()

                        # Move to the next line to find the loss
                        i += 1
                        loss_line = lines[i].strip()
                        if "Loss:" in loss_line:
                            # Extract the loss value
                            loss_value = float(loss_line.split("Loss: ")[1].strip())

                            # Check if this is the best formula (lowest loss) found so far
                            formula_found = True
                            if loss_value < best_loss:
                                best_loss = loss_value
                                best_formula = formula_part
                except (IndexError, ValueError) as e:
                    print(f"Skipping KAN hall of fame line read due to parsing error: {e}")  # Debug: show parsing error message

                i += 1  # Move to the next line

    if not formula_found:
        raise ValueError("No valid formulas found in the hall of fame file.")
    
    return convert_x_i_variables(best_formula, feature_names)

def get_best_formula_from_pysr(hall_of_fame_file, feature_names):
    """Extract the best formula from PySR's hall of fame CSV file, prioritizing the lowest loss."""
    if os.path.exists(hall_of_fame_file):
        hall_of_fame = pd.read_csv(hall_of_fame_file)
        if not hall_of_fame.empty and 'Loss' in hall_of_fame.columns:
            best_formula = hall_of_fame.sort_values(by='Loss').iloc[0]["Equation"]
            return convert_x_i_variables(best_formula, feature_names, underscore=False)
    return None

import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract the best formula from each hall of fame file based on the score')
    parser.add_argument('--hall_of_fame', nargs='+', required=True, help='Paths to the hall of fame files, one for each method')
    parser.add_argument('--save', nargs='+', required=True, help='Paths to save the best formulas, one for each method')
    parser.add_argument('--features', required=True, help='Feature names')
    parser.add_argument('--methods', nargs='+', required=True, help='List of symbolic regression methods to use (e.g., pysindy, aifeynman, dso, kan, pysr)')

    args = parser.parse_args()

    # Ensure that the number of methods, hall of fame files, and save paths are the same
    if not (len(args.methods) == len(args.hall_of_fame) == len(args.save)):
        raise ValueError("The number of methods, hall of fame files, and save paths must be the same.")

    # Loop through each method and its respective hall_of_fame and save paths
    for method, hall_of_fame_path, save_path in zip(args.methods, args.hall_of_fame, args.save):
        method = method.lower()
        
        # Choose the correct extraction function based on the method
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

        # Save the best formula for each method to its respective save path
        if best_formula:
            with open(save_path, 'w') as file:
                file.write(best_formula)
            print(f"Best formula for {method} saved to {save_path}: {best_formula}")
        else:
            print(f"No valid formula found in the hall of fame file for {method}.")

if __name__ == '__main__':
    main()