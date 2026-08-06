import os
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sympy.utilities.lambdify import lambdify
from sklearn.metrics import r2_score
from matplotlib.cm import get_cmap
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter
from shared.plot_style import apply_cell_systems_style

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

def compute_relative_mae(formula, dataset, discovery_scale):
    target_column = dataset.columns[-1]
    formula_symbols_set = formula.free_symbols
    formula_symbols = sorted(list(formula_symbols_set), key=lambda s: s.name)
    variable_names = [str(s) for s in formula_symbols]
    input_data = [np.exp(dataset[var].astype(float)) for var in variable_names]

    try:
        f = lambdify(formula_symbols, formula, modules='numpy')
        y_pred = np.array(f(*input_data), dtype=np.float64)
    except Exception as e:
        print(f"Failed to evaluate formula for relMAE: {formula} with error {e}")
        return np.nan

    y_true = np.exp(dataset[target_column].astype(float).values)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 2:
        print("Insufficient finite values for relMAE computation.")
        return np.nan

    eps = 1e-20
    denom = np.maximum(np.abs(y_true[mask]), eps)
    rel_mae = np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom) * 100.0
    return float(rel_mae)

def compute_relative_errors(formula, dataset, discovery_scale):
    if dataset is None:
        return np.array([])
    target_column = dataset.columns[-1]
    formula_symbols_set = formula.free_symbols
    formula_symbols = sorted(list(formula_symbols_set), key=lambda s: s.name)
    variable_names = [str(s) for s in formula_symbols]
    input_data = [np.exp(dataset[var].astype(float)) for var in variable_names]

    try:
        f = lambdify(formula_symbols, formula, modules='numpy')
        y_pred = np.array(f(*input_data), dtype=np.float64)
    except Exception as e:
        print(f"Failed to evaluate formula for rel errors: {formula} with error {e}")
        return np.array([])

    y_true = np.exp(dataset[target_column].astype(float).values)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 2:
        return np.array([])

    eps = 1e-20
    denom = np.maximum(np.abs(y_true[mask]), eps)
    rel_err = np.abs(y_pred[mask] - y_true[mask]) / denom * 100.0
    rel_err = rel_err[np.isfinite(rel_err)]
    return rel_err

def summarize_relative_errors(errors: np.ndarray):
    if errors is None or len(errors) == 0:
        return np.nan, np.nan
    return float(np.mean(errors)), float(np.median(errors))

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
        regime_label = _format_regime_label(entry['regime'])
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
    plt.grid(True, which="major", axis="y", linestyle='--', alpha=0.4)

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
                annotation = f"{annotation} · {_format_regime_label(point['regime'])}"
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

    regime_context = sorted({_format_regime_label(p['regime']) for p in valid_points if p.get('regime')})
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

def generate_regime_boxplot(rel_mae_by_enzyme_regime_split, output_path_base):
    if not rel_mae_by_enzyme_regime_split:
        print("No regime relMAE data available for boxplot.")
        return

    enzyme_models = sorted({key[0] for key in rel_mae_by_enzyme_regime_split.keys()})
    preferred_order = [
        "two_step_enzyme",
        "three_step_enzyme",
        "three_step_enzyme_dimer",
    ]
    ordered = [m for m in preferred_order if m in enzyme_models]
    ordered += [m for m in enzyme_models if m not in ordered]
    enzyme_models = ordered
    regime_order = ['static', 'dynamic']
    split_order = ['train', 'test']

    data = []
    labels = []
    positions = []
    colors = []
    base_positions = []

    train_color = '#bdbdbd'
    test_color = '#4c78a8'

    pos = 1.0
    gap = 1.0
    within_gap = 1.2
    split_offset = 0.28
    for enzyme in enzyme_models:
        for idx, regime in enumerate(regime_order):
            base_positions.append(pos)
            labels.append(f"{_format_enzyme_short_label(enzyme)}\n{_format_regime_label(regime)}")
            for split in split_order:
                values = rel_mae_by_enzyme_regime_split.get((enzyme, regime, split), [])
                values = [v for v in values if v is not None and np.isfinite(v) and v > 0]
                if not values:
                    values = [np.nan]
                data.append(values)
                offset = -split_offset if split == 'train' else split_offset
                positions.append(pos + offset)
                colors.append(train_color if split == 'train' else test_color)
            pos += within_gap
        pos += gap

    plt.figure(figsize=(5.5, 3.2))
    ax = plt.gca()
    box = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        whis=(5, 95),
    )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.85)

    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(1.2)

    # Annotate medians for train/test boxes
    median_labels = []
    for vals in data:
        clean = [v for v in vals if v is not None and np.isfinite(v) and v > 0]
        if not clean:
            median_labels.append(None)
            continue
        median_labels.append(float(np.median(clean)))

    plt.xticks(base_positions, labels, rotation=0, ha='center', fontsize=9)
    plt.yticks(fontsize=9)
    plt.yscale('log')
    # Major ticks at decades only
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))

    # No minor ticks
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Grid ONLY on major ticks (decades)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.4)
    ax.grid(False, which='minor', axis='y')
    plt.ylabel('Relative MAE (%)', fontsize=10)
    plt.xlabel('Enzyme Model and Regime', fontsize=10)
    plt.title('Symbolic Regression relMAE by Enzyme Model and Regime', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Add median value labels after scales/grid are set
    y_min, y_max = ax.get_ylim()
    for x_pos, med in zip(positions, median_labels):
        if med is None or not np.isfinite(med):
            continue
        y_text = med * 1.25
        if y_text > y_max:
            y_text = med * 0.9
        if y_text < y_min:
            y_text = med * 1.1
        ax.text(
            x_pos,
            y_text,
            f"{med:.2g}",
            ha='center',
            va='bottom',
            fontsize=7,
            color='black',
        )

    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=train_color,
                   markeredgecolor='black', label='Train'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=test_color,
                   markeredgecolor='black', label='Test')
    ]
    plt.legend(
        handles=legend_handles,
        title='Split',
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(-0.02, 1.0),
        fontsize=9,
        title_fontsize=9
    )

    plt.tight_layout(pad=0.2)
    plt.savefig(f"{output_path_base}_box.png", dpi=300, bbox_inches="tight")
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

def _format_enzyme_short_label(name):
    label_map = {
        "two_step_enzyme": "Two-step",
        "three_step_enzyme": "Three-step",
        "three_step_enzyme_dimer": "Four-step",
    }
    return label_map.get(name, _format_enzyme_name(name))

def _format_regime_label(regime):
    regime_map = {
        'static': 'Closed',
        'dynamic': 'Open'
    }
    return regime_map.get(regime, regime.replace('_', ' ').title())

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
        f.write("enzyme_model\tregime\tmethod\tformula\trelmae_train_mean\trelmae_train_median\trelmae_test_mean\trelmae_test_median\n")
        for entry in formula_registry:
            train_mean = entry.get("relmae_train_mean")
            train_median = entry.get("relmae_train_median")
            test_mean = entry.get("relmae_test_mean")
            test_median = entry.get("relmae_test_median")
            train_mean_str = f"{train_mean:.6g}" if isinstance(train_mean, (int, float)) and np.isfinite(train_mean) else "NA"
            train_median_str = f"{train_median:.6g}" if isinstance(train_median, (int, float)) and np.isfinite(train_median) else "NA"
            test_mean_str = f"{test_mean:.6g}" if isinstance(test_mean, (int, float)) and np.isfinite(test_mean) else "NA"
            test_median_str = f"{test_median:.6g}" if isinstance(test_median, (int, float)) and np.isfinite(test_median) else "NA"
            f.write(
                f"{entry['enzyme_model']}\t{entry['regime']}\t{entry['method']}\t{entry['formula']}"
                f"\t{train_mean_str}\t{train_median_str}\t{test_mean_str}\t{test_median_str}\n"
            )

def _normalize_variant(variant):
    if variant is None:
        return None
    variant_norm = variant.strip().lower()
    if variant_norm in {"sqssa", "s_qssa"}:
        return "sQSSA"
    if variant_norm in {"tqssa", "t_qssa"}:
        return "tQSSA"
    return None

def main(root_dir, dataset_path, discovery_scales, variant=None):
    variant_key = _normalize_variant(variant)
    output_dir = os.path.join(root_dir, "panmodel_plots")
    if variant_key:
        output_dir = os.path.join(output_dir, variant_key.lower())
    os.makedirs(output_dir, exist_ok=True)

    scatter_data = []
    formula_registry = []
    bar_entries = []
    rel_mae_by_enzyme_regime_split = {}

    for enzyme_model in os.listdir(root_dir):
        enzyme_model_path = os.path.join(root_dir, enzyme_model)
        if not os.path.isdir(enzyme_model_path):
            continue
        if enzyme_model == "experimental":
            print("Skipping experimental data in pan plots.")
            continue

        for regime in ['static', 'dynamic']:
            subdir_path = os.path.join(enzyme_model_path, regime)
            if not os.path.isdir(subdir_path):
                continue

            results_dir = os.path.join(subdir_path, "sr_comparison", "results")
            dataset_path_regime_test = dataset_path or os.path.join(subdir_path, "processed", "data_test.csv")
            if dataset_path and "data_test.csv" in dataset_path:
                dataset_path_regime_train = dataset_path.replace("data_test.csv", "data_train.csv")
            else:
                dataset_path_regime_train = os.path.join(subdir_path, "processed", "data_train.csv")

            if not os.path.isdir(results_dir) or not os.path.exists(dataset_path_regime_test):
                print(f"Skipping {enzyme_model}/{regime}: missing results or data")
                continue

            formula_files = [f for f in os.listdir(results_dir) if f.startswith("formula_") and f.endswith(".txt")]
            if variant_key:
                suffix = f"_{variant_key}.txt"
                formula_files = [f for f in formula_files if f.endswith(suffix)]
            if not formula_files:
                variant_msg = f" for {variant_key}" if variant_key else ""
                print(f"Skipping {enzyme_model}/{regime}: no formula files{variant_msg}")
                continue

            dataset_test = pd.read_csv(dataset_path_regime_test)
            dataset_train = None
            if os.path.exists(dataset_path_regime_train):
                dataset_train = pd.read_csv(dataset_path_regime_train)
            else:
                print(f"Warning: missing training dataset for {enzyme_model}/{regime}; train relMAE will be skipped.")

            for fname in formula_files:
                method_name = fname.replace("formula_", "").replace(".txt", "")
                model_key = f"{enzyme_model}_{regime}_{method_name}"
                equation_path = os.path.join(results_dir, fname)
                method = infer_method_from_name(fname)
                discovery_scale = discovery_scales.get(method, "log")  # Default to log if unknown

                try:
                    raw_formula = open(equation_path, 'r').readline().strip()
                    replaced = _map_xi_to_columns(raw_formula, dataset_test.columns)
                    formula = sp.sympify(replaced)

                    r2 = compute_r2(formula, dataset_test, discovery_scale)
                    rel_errors_test = compute_relative_errors(formula, dataset_test, discovery_scale)
                    rel_errors_train = compute_relative_errors(formula, dataset_train, discovery_scale) if dataset_train is not None else np.array([])
                    rel_mae_train_mean, rel_mae_train_median = summarize_relative_errors(rel_errors_train)
                    rel_mae_test_mean, rel_mae_test_median = summarize_relative_errors(rel_errors_test)
                    complexity = compute_complexity(formula)

                    formula_registry.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'formula': str(formula),
                        'relmae_train_mean': rel_mae_train_mean,
                        'relmae_train_median': rel_mae_train_median,
                        'relmae_test_mean': rel_mae_test_mean,
                        'relmae_test_median': rel_mae_test_median,
                    })

                    bar_entries.append({
                        'enzyme_model': enzyme_model,
                        'regime': regime,
                        'method': method,
                        'score': r2
                    })
                    key = (enzyme_model, regime)
                    for split_name, values in (("train", rel_errors_train), ("test", rel_errors_test)):
                        split_key = (enzyme_model, regime, split_name)
                        rel_mae_by_enzyme_regime_split.setdefault(split_key, [])
                        if values is None:
                            continue
                        for value in values:
                            if value is not None and np.isfinite(value) and value > 0:
                                rel_mae_by_enzyme_regime_split[split_key].append(float(value))
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
    relmae_output_base = os.path.join(output_dir, "symbolic_model_relmae_scores")
    generate_bar_plot(bar_entries, output_base)
    generate_scatter_plot(scatter_data, output_base)
    generate_regime_boxplot(rel_mae_by_enzyme_regime_split, relmae_output_base)

    formula_output_path = os.path.join(output_dir, "all_symbolic_formulas.txt")
    write_all_formulas(formula_registry, formula_output_path)

    if variant_key:
        print(f"Saved {variant_key} R² bar/scatter plots to {output_base}_*.png")
        print(f"Saved {variant_key} relMAE box plot to {relmae_output_base}_box.png")
    else:
        print(f"Saved R² bar/scatter plots to {output_base}_*.png")
        print(f"Saved relMAE box plot to {relmae_output_base}_box.png")
    print(f"Saved all formulas to {formula_output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate symbolic regression summary plots across enzyme models.")
    parser.add_argument('--root-dir', required=True, help='Path to top-level directory containing enzyme model subfolders')
    parser.add_argument('--dataset', required=False, help='Path to a specific dataset file (overrides default dataset paths)')
    parser.add_argument('--discovery-scales', type=str, required=True, help='JSON string or path to JSON with discovery scales')
    parser.add_argument('--variant', required=False, help='Restrict to PySR variant (sQSSA or tQSSA)')
    args = parser.parse_args()
    
    if args.discovery_scales.endswith('.json'):
        with open(args.discovery_scales) as f:
            discovery_scales = json.load(f)
    else:
        discovery_scales = json.loads(args.discovery_scales)

    main(args.root_dir, dataset_path=args.dataset, discovery_scales=discovery_scales, variant=args.variant)
