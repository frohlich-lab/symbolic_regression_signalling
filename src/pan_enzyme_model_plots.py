import os
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sympy.utilities.lambdify import lambdify
from sklearn.metrics import r2_score
from matplotlib.cm import get_cmap

# Set Helvetica styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

def load_formula(formula_path):
    with open(formula_path, 'r') as f:
        formula_str = f.readline().strip()
    return sp.sympify(formula_str)

def compute_r2(formula, dataset, discovery_scale):
    target_column = dataset.columns[-1]
    formula_symbols = formula.free_symbols
    variable_names = [str(s) for s in formula_symbols]
    input_data = [dataset[var].astype(float) for var in variable_names]

    try:
        f = lambdify(formula_symbols, formula, modules='numpy')
        y_pred = np.array(f(*input_data), dtype=np.float64)
    except Exception as e:
        print(f"Failed to evaluate formula: {formula} with error {e}")
        return np.nan

    y_true = dataset[target_column].astype(float).values

    # Handle log/linear conversion
    if discovery_scale == "linear":
        # Model predicts in linear space → take exp(target) to match
        y_true = np.exp(y_true)
    elif discovery_scale == "log":
        # Model predicts in log space → take log(predicted) to match
        y_pred = np.log(np.clip(y_pred, a_min=1e-20, a_max=None))

    # Guard against invalid values
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        print("Invalid values in R² computation.")
        return np.nan

    return r2_score(y_true, y_pred)

def compute_complexity(formula):
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def generate_bar_plot(r2_scores, output_path_base):
    if not r2_scores:
        print("No R² scores to plot.")
        return

    sorted_items = sorted(r2_scores.items(), key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    model_names, scores = zip(*sorted_items)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, scores, color='skyblue')

    for i, score in enumerate(scores):
        if not np.isnan(score):
            plt.text(i, score + 0.01, f"{score:.2f}", ha='center', va='bottom', fontsize=10)

    plt.axhline(y=0.8, color='red', linestyle='--', label='Reference R² = 0.80')
    plt.legend()

    plt.ylabel('R² Score')
    plt.xlabel('Enzyme Model')
    plt.title('Symbolic Model R² Score per Enzyme System')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')

    plt.savefig(f"{output_path_base}_bar.png", dpi=300)
    plt.close()

def generate_scatter_plot(points, output_path_base):
    if not points:
        print("No data to plot in scatter.")
        return

    methods = list(set(p['method'] for p in points))
    colors = get_cmap("tab10")
    color_map = {m: colors(i % 10) for i, m in enumerate(methods)}

    plt.figure(figsize=(8, 6))
    for p in points:
        plt.scatter(p['complexity'], p['r2'], color=color_map[p['method']],
                    label=p['method'], alpha=0.7, edgecolor='k')
        plt.annotate(p['model'], (p['complexity'], p['r2']), fontsize=8)

    plt.xlabel("Formula Complexity")
    plt.ylabel("R² Score")
    plt.title("Complexity vs. Accuracy Across Enzyme and SR Models")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m],
                          markeredgecolor='k', label=m) for m in methods]
    plt.legend(handles=handles, title="Method")
    plt.grid(True)
    plt.tight_layout()
    print(f"Saving scatter plot to {output_path_base}_scatter.png")
    plt.savefig(f"{output_path_base}_scatter.png", dpi=300)
    plt.close()

def infer_method_from_name(folder_name):
    keywords = ['pysr', 'aifeynman', 'dso', 'sindy', 'kan']
    for kw in keywords:
        if kw.lower() in folder_name.lower():
            return kw
    return 'unknown'

def write_all_formulas(formula_registry, output_path):
    with open(output_path, 'w') as f:
        f.write("enzyme_model\tregime\tmethod\tformula\n")
        for entry in formula_registry:
            f.write(f"{entry['enzyme_model']}\t{entry['regime']}\t{entry['method']}\t{entry['formula']}\n")

def main(root_dir, dataset_path, discovery_scales):
    output_dir = os.path.join(root_dir, "panmodel_plots")
    os.makedirs(output_dir, exist_ok=True)

    r2_scores = {}
    scatter_data = []
    formula_registry = []

    for enzyme_model in os.listdir(root_dir):
        enzyme_model_path = os.path.join(root_dir, enzyme_model)
        if not os.path.isdir(enzyme_model_path):
            continue

        for regime in ['static', 'dynamic']:
            subdir_path = os.path.join(enzyme_model_path, regime)
            if not os.path.isdir(subdir_path):
                continue

            results_dir = os.path.join(subdir_path, "sr_comparison", "results")

            if not os.path.isdir(results_dir) or not os.path.exists(dataset_path):
                print(f"Skipping {enzyme_model}/{regime}: missing results or data")
                continue

            formula_files = [f for f in os.listdir(results_dir) if f.startswith("formula_") and f.endswith(".txt")]
            if not formula_files:
                print(f"Skipping {enzyme_model}/{regime}: no formula files")
                continue

            dataset = pd.read_csv(dataset_path)

            for fname in formula_files:
                method_name = fname.replace("formula_", "").replace(".txt", "")
                model_key = f"{enzyme_model}_{regime}_{method_name}"
                equation_path = os.path.join(results_dir, fname)

                try:
                    formula = load_formula(equation_path)

                    method = infer_method_from_name(fname)
                    discovery_scale = discovery_scales.get(method, "log")  # Default to log if unknown
                    r2 = compute_r2(formula, dataset, discovery_scale)
                    r2 = compute_r2(formula, dataset)
                    complexity = compute_complexity(formula)

                    method = infer_method_from_name(fname)
                    
                    formula_registry.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'formula': str(formula)
                    })

                    r2_scores[model_key] = r2
                    scatter_data.append({
                        'model': model_key,
                        'method': method,
                        'r2': r2,
                        'complexity': complexity,
                        'regime': regime
                    })
                except Exception as e:
                    print(f"Error processing {equation_path}: {e}")
                    r2_scores[model_key] = np.nan

    output_base = os.path.join(output_dir, "symbolic_model_r2_scores")
    generate_bar_plot(r2_scores, output_base)
    generate_scatter_plot(scatter_data, output_base)

    formula_output_path = os.path.join(output_dir, "all_symbolic_formulas.txt")
    write_all_formulas(formula_registry, formula_output_path)

    print(f"Saved R² bar and scatter plots to {output_base}_*.png")
    print(f"Saved all formulas to {formula_output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate symbolic regression summary plots across enzyme models.")
    parser.add_argument('--root-dir', required=True, help='Path to top-level directory containing enzyme model subfolders')
    parser.add_argument('--dataset', required=False, help='Path to a specific dataset file (overrides default dataset paths)')
    parser.add_argument('--discovery-scales', type=str, required=True, help='JSON string or path to JSON with discovery scales')
    args = parser.parse_args()
    
    if args.discovery_scales.endswith('.json'):
        with open(args.discovery_scales) as f:
            discovery_scales = json.load(f)
    else:
        discovery_scales = json.loads(args.discovery_scales)

    main(args.root_dir, dataset_path=args.dataset, discovery_scales=discovery_scales)