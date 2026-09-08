"""
Comparison plot for the SR benchmark (the `results_plot` rule).

Loads each method's discovered formula, evaluates it on the test set respecting
per-method input/output scaling, and plots formula complexity (x) against the
**median relative absolute error (%)** (y, log scale) — one point per method,
styled to match the paper's Figure 1B.
"""

import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import json
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from shared.plot_style import apply_cell_systems_style
from shared.mm_models import TARGET_COLUMN as MM_TARGET_COLUMN
from shared.regime_variants import MODEL_COLORS, VARIANTS, augment_for_variant

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

EPS = 1e-20
apply_cell_systems_style()

BASE_METHOD_LABELS = {
    'pysr': 'PySR',
    'aifeynman': 'AI Feynman',
    'dso': 'DSO',
    'kan': 'KAN',
    'pysindy': 'PySINDy',
    'unknown': 'Unknown'
}

BASE_METHOD_COLORS = {
    'pysr': '#E69F00',       # gold
    'aifeynman': '#b22222',  # firebrick red
    'dso': '#2ca02c',        # green
    'kan': '#000000',        # black
    'pysindy': '#9467bd',
    'unknown': '#7f7f7f'
}

# Only this PySR variant is shown (Figure 1B has a single PySR point).
CANONICAL_PYSR_VARIANT = 'sQSSA'

# Per-method annotation placement: (dx_pts, dy_pts, ha, va). Hand-tuned so the
# four labels sit clear of each other and the markers, as in the paper figure.
LABEL_OFFSETS = {
    'aifeynman': (7, 9, 'left', 'bottom'),
    'dso': (7, -11, 'left', 'top'),
    'kan': (-9, 0, 'right', 'center'),
    'pysr': (8, 0, 'left', 'center'),
}
DEFAULT_LABEL_OFFSET = (8, 6, 'left', 'bottom')


def split_method_variant(method: str):
    for variant_key in VARIANTS.keys():
        suffix = f"_{variant_key}"
        if method.endswith(suffix):
            return method[: -len(suffix)], variant_key
    return method, None


def method_display_name(method: str) -> str:
    base_method, variant_key = split_method_variant(method)
    base_label = BASE_METHOD_LABELS.get(
        base_method,
        base_method.replace('_', ' ').title(),
    )
    if variant_key:
        variant_label = VARIANTS.get(variant_key, variant_key)
        return f"{base_label} ({variant_label})"
    return base_label


def method_color(base_method: str, variant_key: Optional[str]) -> str:
    if variant_key and (base_method, variant_key) in MODEL_COLORS:
        return MODEL_COLORS[(base_method, variant_key)]
    return BASE_METHOD_COLORS.get(base_method, '#7f7f7f')


def dataset_for_variant(dataset: pd.DataFrame, variant_key: Optional[str], cache: dict) -> pd.DataFrame:
    if variant_key and variant_key != "sQSSA":
        if variant_key not in cache:
            cache[variant_key] = augment_for_variant(dataset, variant_key)
        return cache[variant_key]
    return dataset

def _map_xi_to_columns(formula_str, dataset_columns):
    cols = list(dataset_columns[:-1])
    mapped = formula_str
    for i, col in enumerate(cols):
        mapped = mapped.replace(f"x_{i}", col)
        mapped = mapped.replace(f"x{i}", col)
    return mapped

def compute_median_rel_abs_error_pct(formula, dataset, discovery_scale="linear", use_log_input=False):
    """Median relative absolute error (%) of predicted vs true kcat_cg.

    Per-row RAE = |kcat_hat - kcat| / |kcat|; returns median(RAE) * 100.

    `dataset` columns are on the CSV's native LOG scale. KAN was trained on LOG
    inputs (`use_log_input=True`); every other method on LINEAR inputs, so those
    feature columns are exp()'d. 'log'-scale methods predict log(kcat_cg), so we
    exp() the formula output to recover kcat_cg. The target column is log(kcat_cg)
    and is exp()'d to the true linear kcat_cg for the relative error.
    """
    formula_variable_names = [str(symbol) for symbol in formula.free_symbols]
    relevant_columns = [col for col in dataset.columns if col in formula_variable_names]

    feature_symbols = sp.symbols(relevant_columns)
    target_column = dataset.columns[-1]  # Assuming the last column is the target
    formula_func = sp.lambdify(feature_symbols, formula, modules=['numpy'])

    # Inputs: log (as-is) for KAN, linear (exp) for every other method.
    inputs = []
    for col in relevant_columns:
        vals = dataset[col].astype(float).values
        inputs.append(vals if use_log_input else np.exp(vals))

    with np.errstate(all="ignore"):
        predicted_values = np.asarray(formula_func(*inputs), dtype=float)
        if predicted_values.ndim > 1:
            predicted_values = predicted_values.squeeze()
        if predicted_values.size != len(dataset):
            predicted_values = np.broadcast_to(predicted_values, len(dataset))
        # Recover kcat_cg: log-scale methods predict log(kcat_cg) -> exp() it.
        kfw = np.exp(predicted_values) if discovery_scale == "log" else predicted_values
        kcat_true = np.exp(dataset[target_column].astype(float).values)  # log(kcat_cg) -> kcat_cg
        rae = np.abs(kfw - kcat_true) / np.abs(kcat_true)

    mask = np.isfinite(rae)
    if not np.any(mask):
        return np.nan

    return float(np.median(rae[mask])) * 100.0

def calculate_complexity(formula):
    """Symbolic complexity = sympy operation count (matches the paper's x-axis)."""
    return int(sp.count_ops(formula))

def scatter_plot_formulas(methods, formulas, dataset, discovery_scales, output_file, plot_context=None, dataset_context=None):
    """Plot formula complexity vs median relative absolute error (%), Figure-1B style.

    One point per method (PySR collapsed to its canonical variant), log-scale y,
    method labels annotated directly on the points, no legend, padded axes.
    """
    points = []
    for method, formula in zip(methods, formulas):
        base_method, variant_key = split_method_variant(method)
        # Single PySR point: keep only the canonical variant, drop the rest (e.g. tQSSA).
        if base_method == 'pysr' and variant_key not in (None, CANONICAL_PYSR_VARIANT):
            continue

        use_log_input = (base_method == "kan")
        scale = discovery_scales.get(base_method, "linear") if discovery_scales else "linear"
        complexity = calculate_complexity(formula)
        err_pct = compute_median_rel_abs_error_pct(
            formula, dataset, discovery_scale=scale, use_log_input=use_log_input
        )
        if not np.isfinite(err_pct) or err_pct <= 0:
            continue

        points.append({
            'label': BASE_METHOD_LABELS.get(base_method, base_method.replace('_', ' ').title()),
            'base': base_method,
            'x': complexity,
            'y': err_pct,
            'color': BASE_METHOD_COLORS.get(base_method, '#7f7f7f'),
        })

    if not points:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        return

    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    for p in points:
        ax.scatter(p['x'], p['y'], s=75, color=p['color'], edgecolor='k',
                   linewidth=0.5, zorder=3)
        dx, dy, ha, va = LABEL_OFFSETS.get(p['base'], DEFAULT_LABEL_OFFSET)
        ax.annotate(p['label'], (p['x'], p['y']), textcoords='offset points',
                    xytext=(dx, dy), ha=ha, va=va, fontsize=10, color=p['color'])

    ax.set_yscale('log')

    # Pad the axes so the annotations fit and breathe.
    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]
    x_span = max(max(xs) - min(xs), 1)
    ax.set_xlim(min(xs) - 0.22 * x_span - 1.0, max(xs) + 0.45 * x_span + 1.5)
    ax.set_ylim(min(ys) * 0.05, max(ys) * 12)

    ax.grid(True, which='major', linestyle='--', alpha=0.3)
    ax.set_xlabel('Symbolic Formula Complexity', fontsize=11)
    ax.set_ylabel('Median Relative Absolute Error (%)', fontsize=11)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _infer_method_from_formula_path(path):
    fname = os.path.basename(path)
    fname_lower = fname.lower()
    variant_key = None
    for key in VARIANTS.keys():
        if f"_{key.lower()}" in fname_lower:
            variant_key = key
            break

    base_method = 'unknown'
    for key in BASE_METHOD_LABELS:
        if key in fname_lower:
            base_method = key
            break

    if variant_key:
        return f"{base_method}_{variant_key}"
    return base_method


def _build_dataset_context(dataset_path):
    path = Path(dataset_path)
    parts = [part.lower() for part in path.parts]
    enzyme = next((p for p in parts if p.endswith('enzyme')), None)
    regime = 'dynamic' if 'dynamic' in parts else 'static' if 'static' in parts else None

    labels = []
    if enzyme:
        labels.append(enzyme.replace('_', ' ').title())
    if regime:
        labels.append(regime.capitalize())

    return ' · '.join(labels) if labels else None

def main():
    """
    Main function to parse arguments and generate the scatter plot.
    """
    print('STARTED')
    parser = argparse.ArgumentParser(description='Plot Log-Space MAE Loss against Formula Complexity')
    parser.add_argument('--formulas', nargs='+', required=True, help='List of file paths containing formulas')
    parser.add_argument('--dataset', required=True, help='File path to a CSV dataset')
    parser.add_argument('--output', required=True, help='File path to save the generated plot')
    parser.add_argument('--discovery-scales', required=False, type=str,
                        help="JSON of base-method -> 'log'|'linear' (log methods predict log(kcat_cg)).")

    args = parser.parse_args()
    methods = [_infer_method_from_formula_path(formula) for formula in args.formulas]
    discovery_scales = json.loads(args.discovery_scales) if args.discovery_scales else {}

    # Load dataset on its native LOG scale; per-method input/output scaling is
    # applied inside compute_log_MAE_loss (features exp'd for non-KAN, formula
    # output exp'd for log-scale methods).
    dataset = pd.read_csv(args.dataset)

    # Load all formulas from the provided file paths
    formulas = []
    for file_path in args.formulas:
        with open(file_path, 'r') as f:
            raw = f.readline().strip()
            mapped = _map_xi_to_columns(raw, dataset.columns)
            formulas.append(sp.sympify(mapped))

    plot_context = 'Integrated Error' if 'integrated' in os.path.basename(args.output).lower() else 'Pointwise Error'
    dataset_context = _build_dataset_context(args.dataset)

    # Generate and save the scatter plot
    scatter_plot_formulas(
        methods,
        formulas,
        dataset,
        discovery_scales,
        args.output,
        plot_context=plot_context,
        dataset_context=dataset_context
    )

if __name__ == '__main__':
    main()
