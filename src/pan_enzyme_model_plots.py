import os
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sympy.utilities.lambdify import lambdify
from sklearn.metrics import r2_score
from matplotlib.cm import get_cmap
from plot_style import apply_cell_systems_style

# Apply consistent publication styling
apply_cell_systems_style()

METHOD_LABELS = {
    'pysr': 'PySR',
    'aifeynman': 'AI Feynman',
    'dso': 'DSO',
    'kan': 'KAN',
    'pysindy': 'PySINDy',
    'unknown': 'Unknown'
}

METHOD_COLORS = {
    'pysr': '#1f77b4',
    'aifeynman': '#ff7f0e',
    'dso': '#2ca02c',
    'kan': '#d62728',
    'pysindy': '#9467bd',
    'unknown': '#7f7f7f'
}

def load_formula(formula_path):
    with open(formula_path, 'r') as f:
        formula_str = f.readline().strip()
    return sp.sympify(formula_str)

def compute_r2(formula, dataset, discovery_scale):
    target_column = dataset.columns[-1]
    # Ensure deterministic ordering of symbols for lambdify (avoid set ordering)
    formula_symbols_set = formula.free_symbols
    formula_symbols = sorted(list(formula_symbols_set), key=lambda s: s.name)
    variable_names = [str(s) for s in formula_symbols]
    # The processed dataset stores variables in log space; convert to linear for evaluation.
    input_data = [np.exp(dataset[var].astype(float)) for var in variable_names]

    try:
        f = lambdify(formula_symbols, formula, modules='numpy')
        y_pred = np.array(f(*input_data), dtype=np.float64)
    except Exception as e:
        print(f"Failed to evaluate formula: {formula} with error {e}")
        return np.nan

    # Targets are also stored in log space in the processed CSV; compare in linear space.
    y_true = np.exp(dataset[target_column].astype(float).values)

    # For pan plots: treat formulas as linear and compute R² on finite subset
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 2:
        print("Insufficient finite values for R² computation.")
        return np.nan
    score = r2_score(y_true[mask], y_pred[mask])
    # Negative R² indicates the model is worse than a mean baseline; cap at zero for visualization.
    return max(score, 0.0)

def compute_complexity(formula):
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def generate_bar_plot(entries, output_path_base):
    if not entries:
        print("No R² scores to plot.")
        return

    sorted_entries = sorted(
        entries,
        key=lambda e: (float('-inf') if e['score'] is None or np.isnan(e['score']) else e['score']),
        reverse=True
    )

    positions = np.arange(len(sorted_entries))
    display_scores = []
    annotations = []
    tick_labels = []
    bar_colors = []

    for entry in sorted_entries:
        raw_score = entry['score']
        score_for_plot = 0.0 if raw_score is None or np.isnan(raw_score) else raw_score
        score_for_plot = max(score_for_plot, 0.0)
        display_scores.append(score_for_plot)

        if raw_score is None or np.isnan(raw_score):
            annotations.append('N/A')
        else:
            annotations.append(f"{raw_score:.2f}")

        method = entry['method']
        method_label = _method_label(method)
        enzyme_label = _format_enzyme_name(entry['enzyme_model'])
        regime_label = entry['regime'].capitalize()
        tick_labels.append(f"{enzyme_label}\n{regime_label} · {method_label}")

        bar_colors.append(_method_color(method, '#1f77b4'))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(positions, display_scores, color=bar_colors, edgecolor='black', linewidth=0.4)

    for bar, annotation in zip(bars, annotations):
        height = bar.get_height()
        y_pos = max(height, 0.02)
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, annotation,
                 ha='center', va='bottom', fontsize=10)

    plt.ylabel('R² Score (capped at 0 for display)')
    plt.xlabel('Enzyme Model, Regime, and Method')
    plt.title('Symbolic Model Performance Across Enzyme Systems')
    plt.xticks(positions, tick_labels, rotation=30, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    legend_handles = []
    seen_methods = set()
    for entry, color in zip(sorted_entries, bar_colors):
        method = entry['method']
        if method in seen_methods:
            continue
        seen_methods.add(method)
        legend_handles.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                       markeredgecolor='black', label=_method_label(method))
        )
    if legend_handles:
        plt.legend(handles=legend_handles, title='Symbolic Regression Method', frameon=False, loc='upper right')

    plt.savefig(f"{output_path_base}_bar.png", dpi=300)
    plt.close()

def generate_scatter_plot(points, output_path_base):
    # Always produce a file to satisfy workflow expectations, even if empty
    if not points:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path_base}_scatter.png", dpi=300)
        plt.close()
        print("Saved empty scatter placeholder due to no data.")
        return

    valid_points = [p for p in points if p['r2'] is not None and not np.isnan(p['r2'])]
    if not valid_points:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'All R² scores are NaN', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_path_base}_scatter.png", dpi=300)
        plt.close()
        print("Saved empty scatter placeholder due to NaN scores.")
        return

    methods = sorted(set(p['method'] for p in valid_points))
    cmap = get_cmap("tab10")
    color_map = {m: _method_color(m, cmap(i % cmap.N)) for i, m in enumerate(methods)}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for method in methods:
        method_points = [p for p in valid_points if p['method'] == method]
        label = _method_label(method)
        color = color_map[method]

        complexities = [p['complexity'] for p in method_points]
        scores = [p['r2'] for p in method_points]

        ax.scatter(
            complexities,
            scores,
            color=color,
            label=label,
            alpha=0.85,
            edgecolor='k',
            linewidth=0.4,
            s=70
        )

        for x, y, point in zip(complexities, scores, method_points):
            annotation = label
            if point.get('regime'):
                annotation = f"{annotation} · {point['regime'].capitalize()}"
            if point.get('enzyme_model'):
                annotation = f"{annotation} · {_format_enzyme_name(point['enzyme_model'])}"
            ax.annotate(
                annotation,
                (x, y),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=9,
                color=color
            )

    ax.set_xlabel("Symbolic Formula Complexity")
    ax.set_ylabel("R² Score (capped at 0 for display)")

    regime_context = sorted({p['regime'].capitalize() for p in valid_points if p.get('regime')})
    enzyme_context = sorted({_format_enzyme_name(p['enzyme_model']) for p in valid_points if p.get('enzyme_model')})
    context_bits = []
    if enzyme_context:
        context_bits.append(f"{len(enzyme_context)} enzyme systems")
    if regime_context:
        context_bits.append("Regimes: " + ", ".join(regime_context))
    context_suffix = f" ({'; '.join(context_bits)})" if context_bits else ""

    ax.set_title(f"Symbolic Regression Accuracy vs Formula Complexity{context_suffix}")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title="Symbolic Regression Method", frameon=False, loc='best')
    plt.tight_layout()
    print(f"Saving scatter plot to {output_path_base}_scatter.png")
    plt.savefig(f"{output_path_base}_scatter.png", dpi=300)
    plt.close()

def infer_method_from_name(fname):
    name = fname.lower()
    if 'pysr' in name:
        return 'pysr'
    if 'aifeynman' in name:
        return 'aifeynman'
    if 'dso' in name:
        return 'dso'
    if 'kan' in name:
        return 'kan'
    # Normalize any sindy variants to pysindy key
    if 'pysindy' in name or 'sindy' in name:
        return 'pysindy'
    return 'unknown'


def _method_label(method):
    return METHOD_LABELS.get(method, method.replace('_', ' ').title())


def _method_color(method, fallback):
    return METHOD_COLORS.get(method, fallback)


def _format_enzyme_name(name):
    return name.replace('_', ' ').title()

def _map_xi_to_columns(formula_str, dataset_columns):
    # Replace x0, x1, ... (and x_0) with dataset column names in order (excluding target at the end)
    cols = list(dataset_columns[:-1])
    mapped = formula_str
    for i, col in enumerate(cols):
        mapped = mapped.replace(f"x_{i}", col)
        mapped = mapped.replace(f"x{i}", col)
    return mapped

def write_all_formulas(formula_registry, output_path):
    with open(output_path, 'w') as f:
        f.write("enzyme_model\tregime\tmethod\tformula\n")
        for entry in formula_registry:
            f.write(f"{entry['enzyme_model']}\t{entry['regime']}\t{entry['method']}\t{entry['formula']}\n")

def main(root_dir, dataset_path, discovery_scales):
    output_dir = os.path.join(root_dir, "panmodel_plots")
    os.makedirs(output_dir, exist_ok=True)

    scatter_data = []
    formula_registry = []
    bar_entries = []

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
                method = infer_method_from_name(fname)
                discovery_scale = discovery_scales.get(method, "log")  # Default to log if unknown

                try:
                    raw_formula = open(equation_path, 'r').readline().strip()
                    replaced = _map_xi_to_columns(raw_formula, dataset.columns)
                    formula = sp.sympify(replaced)

                    r2 = compute_r2(formula, dataset, discovery_scale)
                    complexity = compute_complexity(formula)

                    formula_registry.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'formula': str(formula)
                    })

                    bar_entries.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'score': r2
                    })
                    scatter_data.append({
                        'model': model_key,
                        'method': method,
                        'r2': r2,
                        'complexity': complexity,
                        'regime': regime,
                        'formula': str(formula),
                        'enzyme_model': enzyme_model
                    })
                except Exception as e:
                    print(f"Error processing {equation_path}: {e}")
                    bar_entries.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'score': np.nan
                    })

    output_base = os.path.join(output_dir, "symbolic_model_r2_scores")
    generate_bar_plot(bar_entries, output_base)
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
