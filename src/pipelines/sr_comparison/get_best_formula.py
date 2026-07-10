"""This module provides functions to extract the best formula from various symbolic regression methods' hall of fame files."""

import argparse
import os
import pandas as pd

from regime_variants import VARIANTS


def split_method_variant(method: str):
    for variant_key in VARIANTS.keys():
        suffix = f"_{variant_key}"
        if method.endswith(suffix):
            return method[: -len(suffix)], variant_key
    return method, None

def convert_x_i_variables(formula, feature_names, start_index=0, underscore=True):
    """Convert x_i variables in a formula to actual feature names."""
    feature_list = feature_names.split(',')
    for i, feature_name in enumerate(feature_list):
        var = f"x_{i + start_index}" if underscore else f"x{i + start_index}"
        formula = formula.replace(var, feature_name)
    return formula

def read_csv_hall_of_fame(hall_of_fame_file, delimiter='\t'):
    """Read a CSV-based hall of fame file if it exists and is not empty."""
    if os.path.exists(hall_of_fame_file) and os.path.getsize(hall_of_fame_file) > 0:
        return pd.read_csv(hall_of_fame_file, sep=delimiter)
    return None

def extract_best_formula_from_csv(
    hall_of_fame_file,
    feature_names,
    formula_column,
    score_column,
    delimiter='\t',
    start_index=0,
    underscore=True,
):
    """Extract the best formula from a CSV-based hall of fame file based on a score metric."""
    hall_of_fame = read_csv_hall_of_fame(hall_of_fame_file, delimiter=delimiter)
    if hall_of_fame is None or formula_column not in hall_of_fame.columns:
        return None

    candidate_columns = [
        score_column,
        "Score",
        "score",
        "Loss",
        "loss",
    ]
    score_col = next((col for col in candidate_columns if col and col in hall_of_fame.columns), None)
    if score_col is None:
        return None

    ascending = True
    if score_col.lower() == "score":
        ascending = False

    best_row = hall_of_fame.sort_values(by=score_col, ascending=ascending).iloc[0]
    best_formula = best_row[formula_column]
    return convert_x_i_variables(best_formula, feature_names, start_index=start_index, underscore=underscore)
    return None

def extract_best_formula_from_txt(hall_of_fame_file, feature_names, formula_indicator, score_indicator, start_index=0, underscore=True):
    """Extract the best formula from a TXT-based hall of fame file based on score indicator."""
    best_formula, best_score = None, float('inf')
    if os.path.exists(hall_of_fame_file):
        with open(hall_of_fame_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if formula_indicator in line:
                    formula = line.split(formula_indicator)[1].strip()
                    score_line = lines[i + 1] if i + 1 < len(lines) else ""
                    if score_indicator in score_line:
                        try:
                            score = float(score_line.split(score_indicator)[1].strip())
                            if score < best_score:
                                best_score, best_formula = score, formula
                        except ValueError:
                            continue
    if start_index == None:
        return best_formula
    else:
        return convert_x_i_variables(best_formula, feature_names, start_index=start_index, underscore=underscore) if best_formula else None

def main():
    parser = argparse.ArgumentParser(description='Extract best formulas from hall of fame files.')
    parser.add_argument('--hall_of_fame', nargs='+', required=True, help='Paths to the hall of fame files.')
    parser.add_argument('--save', nargs='+', required=True, help='Paths to save the best formulas.')
    parser.add_argument('--features', required=True, help='Comma-separated feature names.')
    parser.add_argument('--methods', nargs='+', required=True, help='List of symbolic regression methods (pysindy, aifeynman, dso, kan, pysr, odeformer).')

    args = parser.parse_args()

    if len(args.methods) != len(args.hall_of_fame) or len(args.methods) != len(args.save):
        raise ValueError("Mismatch between the number of methods, hall of fame files, and save paths.")

    method_to_extractor = {
        'pysindy': lambda file, features: extract_best_formula_from_txt(file, features, 'Equation', 'Loss', start_index=None),
        'aifeynman': lambda file, features: extract_best_formula_from_txt(file, features, "Formula:", "Error:", start_index=0, underscore=False),
        'dso': lambda file, features: extract_best_formula_from_csv(file, features, 'Equation', 'Score', start_index=1, underscore=False),
        'kan': lambda file, features: extract_best_formula_from_txt(file, features, "Formula:", "Loss:", start_index=1),
        'pysr': lambda file, features: extract_best_formula_from_csv(file, features, 'Equation', 'Score', delimiter=',', underscore=False),
        'odeformer': lambda file, features: extract_best_formula_from_csv(file, features, 'Equation', 'Score', start_index=0, underscore=False),
    }

    for method, hall_of_fame_path, save_path in zip(args.methods, args.hall_of_fame, args.save):
        base_method, variant_key = split_method_variant(method)
        method_key = base_method.lower()
        extractor_func = method_to_extractor.get(method_key)

        if extractor_func:
            best_formula = extractor_func(hall_of_fame_path, args.features)
            if best_formula:
                with open(save_path, 'w') as file:
                    file.write(best_formula)
                variant_msg = f" ({variant_key})" if variant_key else ""
                print(f"Best formula for {base_method}{variant_msg} saved to {save_path}: {best_formula}")
            else:
                print(f"No valid formula found for {base_method} in {hall_of_fame_path}.")
        else:
            print(f"Unknown method: {method}")

if __name__ == '__main__':
    main()
