import pandas as pd
import argparse
import os
import sympy as sp

def get_best_formula_from_pysindy(hall_of_fame_file):
    """Extract the best formula from PySINDy's output file."""
    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            formula = f.readline().strip()  # PySINDy might output the formula directly as plain text
        return formula
    return None

def get_best_formula_from_aifeynman(hall_of_fame_file):
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

    return None, None, None  # Return None if the file doesn't exist or is invalid

def get_best_formula_from_kan(hall_of_fame_file):
    """Extract the best formula from KAN's output file."""
    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            formula = f.readline().strip()  # Assuming KAN stores the best formula directly as plain text
        return formula
    return None

def get_best_formula_from_pysr(hall_of_fame_file):
    """Extract the best formula from PySR's hall of fame CSV file, prioritizing the lowest loss."""
    if os.path.exists(hall_of_fame_file):
        hall_of_fame = pd.read_csv(hall_of_fame_file)
        if not hall_of_fame.empty and 'Loss' in hall_of_fame.columns:
            best_formula = hall_of_fame.sort_values(by='Loss').iloc[0]["Equation"]
            return best_formula
    return None

def main():
    parser = argparse.ArgumentParser(description='Extract the best formula from the hall of fame file based on the score')
    parser.add_argument('--hall_of_fame', required=True, help='Path to the hall of fame file')
    parser.add_argument('--save', required=True, help='Path to save the best formula')
    parser.add_argument('--method', required=True, help='Symbolic regression method used (e.g., pysindy, aifeynman, dso, kan, pysr)')

    args = parser.parse_args()

    # Choose the correct extraction function based on the method
    if args.method.lower() == 'pysindy':
        best_formula = get_best_formula_from_pysindy(args.hall_of_fame)
    elif args.method.lower() == 'aifeynman':
        best_formula = get_best_formula_from_aifeynman(args.hall_of_fame)
    elif args.method.lower() == 'dso':
        best_formula = get_best_formula_from_dso(args.hall_of_fame)
    elif args.method.lower() == 'kan':
        best_formula = get_best_formula_from_kan(args.hall_of_fame)
    elif args.method.lower() == 'pysr':
        best_formula = get_best_formula_from_pysr(args.hall_of_fame)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    if best_formula:
        with open(args.save, 'w') as file:
            file.write(best_formula)
        print(f"Best formula saved to {args.save}: {best_formula}")
    else:
        print("No valid formula found in the hall of fame file.")

if __name__ == '__main__':
    main()