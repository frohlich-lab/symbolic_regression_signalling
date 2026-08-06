"""
This module provides methods for loading symbolic formulas, computing log-space MAE loss,
calculating formula complexity, and generating scatter plots of log-space MAE loss against
formula complexity.
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
    'pysr': '#1f77b4',
    'aifeynman': '#ff7f0e',
    'dso': '#2ca02c',
    'kan': '#d62728',
    'pysindy': '#9467bd',
    'unknown': '#7f7f7f'
}


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

def compute_log_MAE_loss(formula, dataset, discovery_scale="linear", use_log_input=False):
    """log-MAE between predicted and actual log(kcat_cg), respecting per-method scaling.

    `dataset` columns are on the CSV's native LOG scale. Non-KAN formulas were
    trained on LINEAR inputs, so we exp() the feature columns; KAN was trained on
    LOG inputs, so its columns are used as-is (`use_log_input=True`). 'log'-scale
    methods predict log(kcat_cg), so we exp() the formula output to recover kcat_cg.
    The target column is already log(kcat_cg) and is used directly (not re-logged).
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
        pred_log = np.log(np.clip(kfw, a_min=EPS, a_max=None))

    y_log = dataset[target_column].astype(float).values  # already log(kcat_cg)
    mask = np.isfinite(pred_log) & np.isfinite(y_log)
    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs(pred_log[mask] - y_log[mask])))

def calculate_complexity(formula):
    """Calculate the complexity of a formula based on the number of elements."""
    return len(formula.atoms(sp.Symbol, sp.Number)) + len(formula.atoms(sp.Add, sp.Mul, sp.Pow, sp.Function))

def scatter_plot_formulas(methods, formulas, dataset, discovery_scales, output_file, plot_context=None, dataset_context=None):
    """Generate a scatter plot of log-space MAE loss versus formula complexity and save it to a file."""
    grouped_points = {}

    for method, formula in zip(methods, formulas):
        base_method, variant_key = split_method_variant(method)
        use_log_input = (base_method == "kan")
        scale = discovery_scales.get(base_method, "linear") if discovery_scales else "linear"
        complexity = calculate_complexity(formula)
        # Formulas reference base features; per-method input/output scaling handled inside.
        loss = compute_log_MAE_loss(formula, dataset, discovery_scale=scale, use_log_input=use_log_input)

        if not np.isfinite(loss):
            continue

        display_name = method_display_name(method)
        bucket = grouped_points.setdefault(
            display_name,
            {
                'complexity': [],
                'loss': [],
                'base': base_method,
                'variant': variant_key,
            },
        )
        bucket['complexity'].append(complexity)
        bucket['loss'].append(loss)

    if not grouped_points:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        return

    cmap = get_cmap('tab10')

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for idx, (label, payload) in enumerate(grouped_points.items()):
        color = method_color(payload['base'], payload['variant']) or cmap(idx % cmap.N)
        complexities = payload['complexity']
        losses = payload['loss']

        ax.scatter(
            complexities,
            losses,
            label=label,
            color=color,
            edgecolor='k',
            linewidth=0.4,
            s=70,
            alpha=0.85,
        )

        for x, y in zip(complexities, losses):
            ax.annotate(
                label,
                (x, y),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=9,
                color=color,
            )

    ax.set_xlabel('Symbolic Formula Complexity')
    ax.set_ylabel('Log-space MAE: Mean |ln(y_hat + ε) − ln(y + ε)| (ε = 1e−20)')

    context_bits = []
    if plot_context:
        context_bits.append(plot_context)
    if dataset_context:
        context_bits.append(dataset_context)
    context_str = f" ({'; '.join(context_bits)})" if context_bits else ''
    ax.set_title(f'Symbolic Regression Complexity vs Log-MAE{context_str}')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(title='Symbolic Regression Method', frameon=False, loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


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
