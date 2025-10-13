"""
This module provides functionality for symbolic regression using the PySR algorithm.
It evaluates different **biochemical regimes** by filtering the dataset based on feature ratios,
running PySR separately for each regime, and comparing the results against a Michaelis-Menten fit.

Usage:
    python pysr_regimes.py --dataset <path_to_dataset> --dataset_size <size> --features <feature_list> --temp_dir <path_to_temp_dir>
"""

import os
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
from collections import OrderedDict

from pysr import PySRRegressor
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import load_dataset as nn_load_dataset, train_model, evaluate_model
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pathlib import Path

from constants import PYSR_CONFIG
from plot_style import apply_cell_systems_style
from utils.seeding import resolve_seed, seed_everything

# Suppress all warnings
warnings.filterwarnings("ignore")
apply_cell_systems_style()

MODEL_LINE_ORDER = ["PySR", "Michaelis-Menten", "Neural Network"]
MODEL_COLORS = {
    "PySR": "#E69F00",              # orange
    "Michaelis-Menten": "#009E73",   # green
    "Neural Network": "#0072B2",     # blue
}
MM_REQUIRED_COLS = ["P_u", "k_off", "k_D", "k_cat", "tK"]

LOGGER = logging.getLogger(__name__)

# PySR configuration

# Numerical stability epsilon for log operations
EPS = 1e-20
LOG_SCALE_FLOOR = 1e-3

def file_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def michaelis_menten(P_u, k_off, k_D, k_cat, tK, k_inact=None):
    if k_inact is not None:
        denom = P_u + ((k_cat + k_off + k_inact) / (k_off * k_D))
    else:
        denom = P_u + ((k_cat + k_off) / (k_off * k_D))
    return (tK * P_u) / denom

def load_dataset(file_path, dataset_size=None, features=None):
    data = pd.read_csv(file_path)
    if features and features != "all":
        data = data[features.split(',')]
    return data.map(np.exp)


def _mm_components(df: pd.DataFrame):
    missing = [col for col in MM_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Michaelis-Menten evaluation: {missing}")
    arrays = [pd.to_numeric(df[col], errors='coerce').to_numpy() for col in MM_REQUIRED_COLS]
    k_inact = None
    if 'k_inact' in df.columns:
        raw = pd.to_numeric(df['k_inact'], errors='coerce').to_numpy()
        if np.isfinite(raw).any():
            k_inact = np.where(np.isfinite(raw), raw, 0.0)
    return (*arrays, k_inact)


def _mm_predict(df: pd.DataFrame):
    P_u, k_off, k_D, k_cat, tK, k_inact = _mm_components(df)
    return michaelis_menten(P_u, k_off, k_D, k_cat, tK, k_inact)


def _mm_components_row(row: pd.Series):
    required = ["k_off", "k_D", "k_cat", "tK"]
    missing = [col for col in required if col not in row.index or not np.isfinite(row[col])]
    if missing:
        raise ValueError(f"Row missing columns for Michaelis-Menten evaluation: {missing}")
    k_off = float(row['k_off'])
    k_D = float(row['k_D'])
    k_cat = float(row['k_cat'])
    tK = float(row['tK'])
    k_inact_val = row.get('k_inact')
    try:
        k_inact_val = float(k_inact_val)
    except (TypeError, ValueError):
        k_inact_val = np.nan
    k_inact = k_inact_val if np.isfinite(k_inact_val) else None
    return k_off, k_D, k_cat, tK, k_inact


def _mm_predict_row(row: pd.Series, pu_values):
    k_off, k_D, k_cat, tK, k_inact = _mm_components_row(row)
    return michaelis_menten(pu_values, k_off, k_D, k_cat, tK, k_inact)


def _mm_log_error(df: pd.DataFrame):
    target = pd.to_numeric(df['kcat_cg'], errors='coerce').to_numpy()
    mm_pred = _mm_predict(df)
    return np.abs(np.log(np.maximum(mm_pred, EPS)) - np.log(np.maximum(target, EPS)))

def run_pysr(X, y, model_path, output_file):
    run_dir = os.path.dirname(model_path)
    cached = (
        os.path.exists(model_path)
        or os.path.exists(os.path.join(run_dir, "hall_of_fame.csv"))
        or os.path.exists(os.path.join(run_dir, "equations.csv"))
    )
    if cached:
        print(f"Loading existing PySR model from run directory: {run_dir}")
        return PySRRegressor.from_file(run_directory=run_dir)
    print(f"Training new PySR model; run directory: {run_dir}")
    model = PySRRegressor(
        **PYSR_CONFIG,
        output_directory=os.path.dirname(run_dir),
        run_id="pysr",
    )
    model.fit(X, y)
    try:
        model.save()
    except Exception:
        pass
    return model
    
def save_pysr_formulas(model_dict, features, output_path):
    with open(output_path, "w") as f:
        for regime, models in model_dict.items():
            model = models['pysr']
            if model is None or model.equations_ is None or model.equations_.empty:
                continue

            # Get best formula (highest score)
            best_row = model.equations_.sort_values(by="score", ascending=False).iloc[0]
            formula = best_row["equation"]

            # Replace x0, x1, ... with actual variable names
            feature_names = features.split(',')
            for i, name in enumerate(feature_names):
                formula = formula.replace(f"x{i}", name)

            f.write(f"{regime}: {formula}    [score={best_row['score']:.4f}, loss={best_row['loss']:.4f}]\n")

def apply_log_space_gaussian_noise(df, std_multiplier):
    noisy_df = df.copy()
    target_col = noisy_df.columns[-1]

    positive_mask = noisy_df[target_col] > 0
    if not positive_mask.any():
        raise ValueError("No strictly positive targets available for log noise")

    # Work only on strictly positive entries to keep log well-defined.
    positive_vals = noisy_df.loc[positive_mask, target_col]
    log_vals = np.log(positive_vals)

    std = log_vals.std()
    noise = np.random.normal(loc=0, scale=std_multiplier * std, size=len(log_vals))
    noisy_df.loc[positive_mask, target_col] = np.exp(log_vals + noise)

    return noisy_df

def define_regimes_from_low_noise(base_df):
    return {
        "very_low_noise": base_df.copy(),
        "low_noise": apply_log_space_gaussian_noise(base_df, 1e-2),
        "medium_noise": apply_log_space_gaussian_noise(base_df, 1e-1),
        "high_noise": apply_log_space_gaussian_noise(base_df, 1.0),
    }

def define_regimes_from_full_dataset(base_df):
    return {
        "full_dataset": base_df.copy(),
        "full_low_noise": apply_log_space_gaussian_noise(base_df, 1e-2),
        "full_medium_noise": apply_log_space_gaussian_noise(base_df, 1e-1),
        "full_high_noise": apply_log_space_gaussian_noise(base_df, 1.0),
    }

REGIME_LABELS = {
    'very_low_noise': 'Very Low Noise',
    'low_noise': 'Low Noise',
    'medium_noise': 'Medium Noise',
    'high_noise': 'High Noise',
    'full_dataset': 'Full Dataset',
    'full_low_noise': 'Full + Low Noise',
    'full_medium_noise': 'Full + Medium Noise',
    'full_high_noise': 'Full + High Noise',
}

REGIME_ORDER = {
    'very_low_noise': 0,
    'low_noise': 1,
    'medium_noise': 2,
    'high_noise': 3,
    'full_dataset': 0,
    'full_low_noise': 1,
    'full_medium_noise': 2,
    'full_high_noise': 3,
}


def _ordered_regimes(model_dict):
    return sorted(
        model_dict.keys(),
        key=lambda r: (0 if r in REGIME_ORDER else 1, REGIME_ORDER.get(r, float('inf')), r)
    )


def _label_for_regime(regime):
    return REGIME_LABELS.get(regime, regime.replace('_', ' ').title())

def preprocess_data(X):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler())
    ])
    return pipeline.fit_transform(X), pipeline

def _get_eval_partition(models):
    eval_df = models.get('test_data')
    if eval_df is None:
        eval_df = models.get('data')

    eval_pre = models.get('test_preprocessed')
    if eval_pre is None:
        eval_pre = models.get('preprocessed')

    return eval_df, eval_pre


def _collect_error_distributions(model_dict):
    distributions = OrderedDict()
    regimes = _ordered_regimes(model_dict)
    for regime in regimes:
        models = model_dict.get(regime)
        if not models:
            continue

        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue

        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        regime_errors = {}

        if 'Neural Network' in MODEL_LINE_ORDER and 'nn' in models and data_pre is not None:
            try:
                y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
                regime_errors['Neural Network'] = np.abs(
                    np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS))
                )
            except Exception:
                pass

        if 'Michaelis-Menten' in MODEL_LINE_ORDER:
            try:
                y_pred_mm = _mm_predict(X)
                regime_errors['Michaelis-Menten'] = np.abs(
                    np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS))
                )
            except Exception:
                pass

        if 'PySR' in MODEL_LINE_ORDER and 'pysr' in models:
            try:
                y_pred_pysr = models['pysr'].predict(X.values)
                regime_errors['PySR'] = np.abs(
                    np.log(np.maximum(y_pred_pysr, EPS)) - np.log(np.maximum(y_true, EPS))
                )
            except Exception:
                pass

        if regime_errors:
            distributions[regime] = regime_errors

    return distributions


def _errors_to_dataframe(error_dict):
    records = []
    for model_name, errors in error_dict.items():
        if errors is None:
            continue
        arr = np.asarray(errors, dtype=float).flatten()
        arr = arr[~np.isnan(arr)]
        records.extend({'Model': model_name, 'Log-MAE': value} for value in arr)
    return pd.DataFrame(records)


def _compute_line_stats(error_distributions):
    stats = {model: {'median': [], 'q1': [], 'q3': []} for model in MODEL_LINE_ORDER}
    for regime in error_distributions:
        regime_errors = error_distributions[regime]
        for model in MODEL_LINE_ORDER:
            arr = np.asarray(regime_errors.get(model, []), dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size:
                stats[model]['median'].append(np.median(arr))
                stats[model]['q1'].append(np.quantile(arr, 0.25))
                stats[model]['q3'].append(np.quantile(arr, 0.75))
            else:
                stats[model]['median'].append(np.nan)
                stats[model]['q1'].append(np.nan)
                stats[model]['q3'].append(np.nan)
    return stats


def _model_color_map():
    return {model: MODEL_COLORS.get(model, "#333333") for model in MODEL_LINE_ORDER}


def _lineplot_limits(stats):
    lowers, uppers = [], []
    positive_candidates = []
    for model in MODEL_LINE_ORDER:
        lowers.extend([v for v in stats[model]['q1'] if not np.isnan(v)])
        uppers.extend([v for v in stats[model]['q3'] if not np.isnan(v)])
        positive_candidates.extend([v for v in stats[model]['median'] if v > 0 and not np.isnan(v)])
        positive_candidates.extend([v for v in stats[model]['q1'] if v > 0 and not np.isnan(v)])
        positive_candidates.extend([v for v in stats[model]['q3'] if v > 0 and not np.isnan(v)])

    if not lowers or not uppers:
        return None, None

    ymin = min(lowers)
    ymax = max(uppers)
    if positive_candidates:
        min_positive = min(positive_candidates)
    else:
        min_positive = None

    if np.isclose(ymin, ymax):
        margin = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
        ymin -= margin
        ymax += margin
    else:
        margin = 0.05 * (ymax - ymin)
        ymin -= margin
        ymax += margin

    if min_positive is not None and ymin <= 0:
        ymin = min_positive * 0.8

    ymin = max(ymin, LOG_SCALE_FLOOR)
    if ymax <= ymin:
        ymax = ymin * 1.5

    return ymin, ymax


def _apply_log_floor(stats_map):
    for model in MODEL_LINE_ORDER:
        for key in ('median', 'q1', 'q3'):
            stats_map[model][key] = [
                val if (not np.isfinite(val) or val >= LOG_SCALE_FLOOR) else LOG_SCALE_FLOOR
                for val in stats_map[model][key]
            ]


def _render_noise_regime_lineplot(regimes, stats, output_path, template=False):
    if not regimes:
        return

    colors = _model_color_map()
    _apply_log_floor(stats)
    x = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(10, 5))

    ymin, ymax = _lineplot_limits(stats)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    debug_finite_min = []
    for model in MODEL_LINE_ORDER:
        combined = []
        for key in ('median', 'q1', 'q3'):
            combined.extend(stats[model][key])
        combined = np.asarray(combined, dtype=float)
        combined = combined[np.isfinite(combined)]
        if combined.size:
            min_val = combined.min()
            debug_finite_min.append(min_val)
            print(f"[DEBUG noise lineplot] {model} min value: {min_val}")
    if any(val <= 0 for val in debug_finite_min):
        print("[DEBUG noise lineplot] Non-positive value detected, log scale will fail.")

    if not template:
        for model in MODEL_LINE_ORDER:
            medians = np.array(stats[model]['median'], dtype=float)
            if medians.size == 0 or np.all(np.isnan(medians)):
                continue
            q1 = np.array(stats[model]['q1'], dtype=float)
            q3 = np.array(stats[model]['q3'], dtype=float)
            ax.plot(x, medians, label=model, color=colors[model], marker='o', linewidth=2)
            ax.fill_between(x, q1, q3, color=colors[model], alpha=0.25)
        ax.legend(loc='upper left', frameon=False)
    else:
        handles = [Line2D([0], [0], color=colors[m], marker='o', linewidth=2, label=m) for m in MODEL_LINE_ORDER]
        ax.legend(handles=handles, loc='upper left', frameon=False)

    ax.set_xticks(x)
    all_values = []
    for model in MODEL_LINE_ORDER:
        all_values.extend(np.atleast_1d(stats[model]['median']).tolist())
        all_values.extend(np.atleast_1d(stats[model]['q1']).tolist())
        all_values.extend(np.atleast_1d(stats[model]['q3']).tolist())
    all_values = np.array(all_values, dtype=float)
    finite = all_values[np.isfinite(all_values)]
    positive = finite[finite > 0]
    if positive.size:
        if (finite <= 0).any():
            eps = max(LOG_SCALE_FLOOR, positive.min() * 1e-3)
            print(f"[DEBUG noise lineplot] Adding epsilon {eps} to non-positive values for log scale")
            for model in MODEL_LINE_ORDER:
                for key in ('median', 'q1', 'q3'):
                    stats[model][key] = [
                        val if (not np.isfinite(val) or val > eps) else eps
                        for val in stats[model][key]
                    ]
        ax.set_yscale('log')
        ax.set_ylim(bottom=LOG_SCALE_FLOOR)
    else:
        print("[DEBUG noise lineplot] No positive finite values detected; using linear scale")
        ax.set_yscale('linear')
    ax.set_xticklabels([_label_for_regime(r) for r in regimes], rotation=20, ha='right')
    ax.set_xlabel("Noise Regime", fontsize=12)
    ax.set_ylabel("Log-space MAE (on scale)", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_noise_regime_lineplot(error_distributions, output_dir):
    if not error_distributions:
        print("No regimes available for line plot.")
        return
    regimes = list(error_distributions.keys())
    stats = _compute_line_stats(error_distributions)
    output_path = os.path.join(output_dir, "shared/plots/log_mae_noise_regime_lineplot.png")
    _render_noise_regime_lineplot(regimes, stats, output_path, template=False)


def plot_noise_regime_lineplot_template(error_distributions, output_dir):
    if not error_distributions:
        print("No regimes available for line plot template.")
        return
    regimes = list(error_distributions.keys())
    stats = _compute_line_stats(error_distributions)
    output_path = os.path.join(output_dir, "shared/plots/log_mae_noise_regime_lineplot_template.png")
    _render_noise_regime_lineplot(regimes, stats, output_path, template=True)

def evaluate_models(data, features, output_dir, dataset_size, mode, seed: int) -> None:
    """Run symbolic-regression evaluations for each noise regime."""

    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    losses, model_dict = {}, {}
    shared_results_dir = Path(output_dir) / "shared/results/pysr"
    shared_plots_dir = Path(output_dir) / "shared/plots"
    shared_results_dir.mkdir(parents=True, exist_ok=True)
    shared_plots_dir.mkdir(parents=True, exist_ok=True)

    regimes = {}
    if mode == 'filtered':
        errors = _mm_log_error(data)
        base_df = data[errors < 0.01].reset_index(drop=True)
        regimes.update(define_regimes_from_low_noise(base_df))
    elif mode == 'full':
        regimes.update(define_regimes_from_full_dataset(data))

    for regime, filtered in regimes.items():
        filtered = filtered.reset_index(drop=True)

        if filtered.empty:
            print(f"No data for {regime} regime.")
            continue

        os.makedirs(os.path.join(output_dir, f"{regime}/processed"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/models/pysr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/models/nn"), exist_ok=True)

        filtered.to_csv(os.path.join(output_dir, f"{regime}/processed/filtered_data.csv"), index=False)

        capped_size = min(dataset_size, len(filtered)) if dataset_size else len(filtered)
        sample = (
            filtered.sample(n=capped_size, random_state=seed)
            if capped_size < len(filtered)
            else filtered.copy()
        )
        sample = sample.reset_index(drop=True)
        sample.to_csv(
            os.path.join(output_dir, f"{regime}/processed/filtered_data_model_sample.csv"),
            index=False,
        )

        if len(sample) < 2:
            print(f"Skipping {regime}: need at least 2 samples for train/test split.")
            continue

        test_size = max(1, int(np.ceil(len(sample) * 0.2)))
        if len(sample) - test_size < 1:
            print(f"Skipping {regime}: insufficient samples after applying test split.")
            continue

        train_df, test_df = train_test_split(
            sample, test_size=test_size, random_state=seed, shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(
            os.path.join(output_dir, f"{regime}/processed/filtered_data_train.csv"),
            index=False,
        )
        test_df.to_csv(
            os.path.join(output_dir, f"{regime}/processed/filtered_data_test.csv"),
            index=False,
        )

        X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
        X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values
        X_full, y_full = sample.iloc[:, :-1].values, sample.iloc[:, -1].values

        print(f"Evaluating {regime} with {len(train_df)} train / {len(test_df)} test samples")

        temp_file = os.path.join(output_dir, f"{regime}/models/pysr/hall_of_fame.csv")
        model_path = os.path.join(output_dir, f"{regime}/models/pysr/hall_of_fame.pkl")
        pysr_model = run_pysr(X_train, y_train, model_path, temp_file)
        y_pred_pysr = pysr_model.predict(X_test)
        log_mae = np.mean(
            np.abs(np.log(np.maximum(y_pred_pysr, EPS)) - np.log(np.maximum(y_test, EPS)))
        )
        losses[regime] = log_mae
        print(f"PySR Loss (test): {log_mae}")

        # Michaelis-Menten Model
        y_pred_mm = _mm_predict(test_df)
        mm_loss = np.mean(
            np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_test, EPS)))
        )
        losses[f"{regime}_mm"] = mm_loss
        print(f"Michaelis-Menten Loss (test): {mm_loss}")

        # Neural Network Model
        X_train_pre, pipeline = preprocess_data(X_train)
        X_test_pre = pipeline.transform(X_test)
        X_full_pre = pipeline.transform(X_full)
        train_pre = pd.DataFrame(
            np.column_stack([X_train_pre, np.log(y_train)]),
            columns=list(train_df.columns),
        )
        test_pre = pd.DataFrame(
            np.column_stack([X_test_pre, np.log(y_test)]),
            columns=list(test_df.columns),
        )
        full_pre = pd.DataFrame(
            np.column_stack([X_full_pre, np.log(y_full)]),
            columns=list(sample.columns),
        )

        nn_path = os.path.join(output_dir, f"{regime}/models/nn/model.pkl")
        if os.path.exists(nn_path):
            print(f"Loading existing NN model for {regime}")
            nn_model = train_model(
                train_pre,
                nn_path,
                verbose=False,
                retrain=False,
                seed=seed,
            )
        else:
            print(f"Training new NN model for {regime}")
            nn_model = train_model(
                train_pre,
                nn_path,
                verbose=False,
                seed=seed,
            )
        nn_loss, _ = evaluate_model(nn_model, test_pre)
        losses[f"{regime}_nn"] = nn_loss
        print(f"NN Loss (test): {nn_loss}")

        # Store models and data for the current regime
        model_dict[regime] = {
            'data': sample,
            'train_data': train_df,
            'test_data': test_df,
            'pysr': pysr_model,
            'mm': y_pred_mm,
            'nn': nn_model,
            'preprocessed': full_pre,
            'train_preprocessed': train_pre,
            'test_preprocessed': test_pre,
        }
        print()

    # Generate plots and save results
    plot_model_subregimes(model_dict, output_dir)
    plot_error_distributions(model_dict, output_dir)
    save_pysr_formulas(model_dict, features, str(shared_results_dir / "all_pysr_formulas.txt"))
    plot_input_error_correlation(model_dict, output_dir)
    plot_model_error_correlation(model_dict, output_dir)
    plot_nn_vs_mm_response_curves_linear(model_dict, output_dir)
    error_distributions = _collect_error_distributions(model_dict)
    plot_horizontal_boxplot_noise_regimes(model_dict, output_dir, error_distributions)
    plot_vertical_boxplot_noise_regimes(model_dict, output_dir, error_distributions)
    plot_noise_regime_lineplot(error_distributions, output_dir)
    plot_noise_regime_lineplot_template(error_distributions, output_dir)

def plot_error_distributions(model_dict, output_dir):
    """
    For each regime, plot a violin plot comparing Log-MAE errors for PySR, MM, and NN.
    Output is a grid of subplots, one per regime.
    """
    error_records = []

    for regime, models in model_dict.items():
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1].values

        # PySR
        y_pred_pysr = models['pysr'].predict(X)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'PySR', 'Regime': regime, 'Log-MAE': e} for e in err_pysr])

        # MM
        y_pred_mm = _mm_predict(data)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Regime': regime, 'Log-MAE': e} for e in err_mm])

        # NN
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'Neural Network', 'Regime': regime, 'Log-MAE': e} for e in err_nn])

    error_df = pd.DataFrame(error_records)

    regimes = [reg for reg in model_dict.keys() if reg in error_df['Regime'].unique()]
    if not regimes:
        print("No regimes with evaluable data for plotting.")
        return

    ncols = min(4, len(regimes))
    nrows = (len(regimes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, regime in enumerate(regimes):
        ax = axes[i]
        subset = error_df[error_df['Regime'] == regime]
        sns.violinplot(
            data=subset,
            x='Model',
            y='Log-MAE',
            ax=ax,
            inner='box',
            palette=[MODEL_COLORS[m] for m in MODEL_LINE_ORDER],
            order=MODEL_LINE_ORDER,
        )
        ax.set_title(regime)
        ax.set_xlabel('')
        if i % ncols == 0:
            ax.set_ylabel('Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)')
        else:
            ax.set_ylabel('')

    # Remove empty axes if any
    for j in range(len(regimes), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Log-space MAE Error Distributions by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "shared/plots/regime_loss_comparison.png"))

def plot_model_subregimes(model_dict, output_dir):
    regimes = list(model_dict.keys())
    models = ['pysr', 'mm', 'nn']
    model_labels = {'pysr': 'PySR', 'mm': 'Michaelis-Menten', 'nn': 'Neural Network'}
    model_colors = {'pysr': MODEL_COLORS['PySR'], 'mm': MODEL_COLORS['Michaelis-Menten'], 'nn': MODEL_COLORS['Neural Network']}

    n_rows, n_cols = len(regimes), len(models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), sharey=True)

    sampled_indices = {}
    for regime_name in regimes:
        eval_data, _ = _get_eval_partition(model_dict[regime_name])
        if eval_data is None or eval_data.empty:
            continue
        sample_size = min(len(eval_data), 600)
        sampled_indices[regime_name] = np.random.choice(
            eval_data.index.values,
            size=sample_size,
            replace=False,
        )

    for row_idx, regime_name in enumerate(regimes):
        eval_data, eval_pre = _get_eval_partition(model_dict[regime_name])
        indices = sampled_indices.get(regime_name)
        if eval_data is None or eval_data.empty or indices is None or len(indices) == 0:
            continue

        data = eval_data.loc[indices].copy()
        data_pre = eval_pre.loc[indices] if eval_pre is not None else None
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]
        y_mm = _mm_predict(data)

        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

            if model == 'pysr':
                y_pred = model_dict[regime_name]['pysr'].predict(X.values)
            elif model == 'mm':
                y_pred = y_mm
            elif model == 'nn':
                if data_pre is None:
                    continue
                data_pre_subset = data_pre.loc[data.index]
                y_pred = evaluate_model(model_dict[regime_name]['nn'], data_pre_subset)[1].detach().cpu().numpy().flatten()

            error = np.abs(np.log(np.maximum(y_pred, EPS)) - np.log(np.maximum(y_true, EPS)))

            ax.scatter(
                y_true, error, alpha=0.4, s=20,
                color=model_colors[model], label=model_labels[model],
                edgecolors='k', linewidths=0.2
            )

            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=11)
            if row_idx == 0:
                ax.set_title(model_labels[model], fontsize=13)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc='upper left')

            ax.set_xscale('log')

    fig.suptitle("Model Error Landscape by Regime and Model", fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "shared/plots/error_landscape.png")
    plt.savefig(output_path)

def plot_input_error_correlation(model_dict, output_dir):
    """
    Plot feature-error correlations for each regime in a grid.
    Also save an overall correlation across all regimes.
    """
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    all_error_df = []

    for idx, (regime, models) in enumerate(model_dict.items()):
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_df = pd.DataFrame()
        error_df['pysr_error'] = np.abs(np.log(np.maximum(models['pysr'].predict(X.values), EPS)) - np.log(np.maximum(y_true, EPS)))
        y_pred_mm = _mm_predict(data)
        error_df['mm_error'] = np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS)))
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        error_df['nn_error'] = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))

        for col in X.columns:
            error_df[col] = X[col].values

        corr = error_df.corr()[['pysr_error', 'mm_error', 'nn_error']].drop(['pysr_error', 'mm_error', 'nn_error'], axis=0)
        all_error_df.append(error_df)

        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")

    # Remove extra subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature-Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "shared/plots/feature_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    full_df = pd.concat(all_error_df, ignore_index=True)
    overall_corr = full_df.corr()[['pysr_error', 'mm_error', 'nn_error']].drop(['pysr_error', 'mm_error', 'nn_error'], axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Feature-Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/feature_error_correlation_overall.png"))
    plt.close()

def plot_model_error_correlation(model_dict, output_dir):
    """
    Plot model-model error correlations for each regime in a grid.
    Also save an overall correlation across all regimes.
    """
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    all_model_error_df = []

    for idx, (regime, models) in enumerate(model_dict.items()):
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        err_pysr = np.abs(np.log(np.maximum(models['pysr'].predict(X.values), EPS)) - np.log(np.maximum(y_true, EPS)))
        mm_pred = _mm_predict(data)
        err_mm = np.abs(np.log(np.maximum(mm_pred, EPS)) - np.log(np.maximum(y_true, EPS)))
        err_nn = np.abs(
            np.log(
                np.maximum(
                    evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten(),
                    EPS,
                )
            )
            - np.log(np.maximum(y_true, EPS))
        )

        df = pd.DataFrame({'PySR': err_pysr, 'MM': err_mm, 'NN': err_nn})
        all_model_error_df.append(df)

        corr = df.corr()
        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "shared/plots/model_error_correlation_grid.png"))
    plt.close()

    # Overall correlation
    df_all = pd.concat(all_model_error_df, ignore_index=True)
    overall_corr = df_all.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
    plt.title("Overall Model Error Correlation (All Regimes)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/model_error_correlation_overall.png"))
    plt.close()

def plot_nn_vs_mm_response_curves_linear(model_dict, output_dir, n_samples=10, n_pu_points=200):
    """
    For each regime:
    - Select n_samples covering low to high log-MAE.
    - Fix all inputs except P_u.
    - Sweep P_u linearly from 0.01*K_M to 100*K_M (per sample).
    - Compare NN predictions to Michaelis-Menten curve.
    - Plot in linear space with adaptive x/y limits.
    - Add vertical line at the original P_u value from the sample.
    """
    for regime, models in model_dict.items():
        print(f"Generating scaled response curves for {regime}...")

        data, preprocessed_data = _get_eval_partition(models)
        if data is None or data.empty:
            print(f"Skipping {regime}: no evaluation data available.")
            continue
        data = data.copy()
        model = models['nn']

        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]
        pu_index = X.columns.get_loc("P_u")

        _, pipeline = preprocess_data(X)

        y_pred_mm = _mm_predict(data)

        # Full NN prediction and error
        y_pred_nn = evaluate_model(model, preprocessed_data)[1].detach().cpu().numpy().flatten()
        log_mae = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))
        log_mae_mm = np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS)))

        # Select samples from across log-MAE distribution
        percentiles = np.linspace(0, 100, n_samples + 2)[1:-1]
        thresholds = np.percentile(log_mae, percentiles)
        chosen_indices = []
        for t in thresholds:
            idx = np.argmin(np.abs(log_mae - t))
            chosen_indices.append(idx)

        fig, axs = plt.subplots(2, n_samples // 2, figsize=(20, 10), sharex=False, sharey=False)
        axs = axs.flatten()

        for i, idx in enumerate(chosen_indices):
            fixed_sample = X.iloc[idx].copy()
            original_pu = fixed_sample["P_u"]

            pu_vals = np.linspace(0.01 * original_pu, 5 * original_pu, n_pu_points)

            # Sweep P_u
            varied_inputs = np.tile(fixed_sample.values, (n_pu_points, 1))
            varied_inputs[:, pu_index] = pu_vals

            # Preprocess inputs
            X_varied_inputs_pre = pipeline.transform(varied_inputs)
            X_varied_inputs_pre = torch.tensor(X_varied_inputs_pre, dtype=torch.float32)
            
            # NN prediction
            model.eval()
            with torch.no_grad():
                y_nn = np.exp(model(X_varied_inputs_pre))
            # MM prediction
            row = data.iloc[idx]
            if "tK" not in row or not np.isfinite(row["tK"]):
                raise ValueError(
                    "tK missing or non-finite for regime '{}' sample index {}".format(regime, idx)
                )
            y_mm = _mm_predict_row(row, pu_vals)

            # Plot
            ax = axs[i]
            ax.plot(pu_vals, y_nn, label="Neural Network", color=MODEL_COLORS['Neural Network'])
            ax.plot(pu_vals, y_mm, label="Michaelis-Menten", color=MODEL_COLORS['Michaelis-Menten'], linestyle='--')
            ax.scatter(original_pu, y_true[idx], color=MODEL_COLORS['Neural Network'], marker='o', s=100, label="Groundtruth kcat_cg")
            ax.axvline(original_pu, color='gray', linestyle=':', linewidth=1.5, label='Sampled P_u')  
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if i % (n_samples // 2) == 0:
                ax.set_ylabel("kcat_cg")
            ax.set_title(f"Sample {i+1} | Log-MAE: NN = {log_mae[idx]:.3f} - MM {log_mae_mm[idx]:.3f}")

        # Add suptitle slightly higher than default
        fig.suptitle(f"NN vs MM — Response Curves — {regime}", fontsize=16, y=1.07)

        # Add legend below title, but above subplots
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

        # Adjust tight layout to allow room for both title and legend
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # bottom, left, right, top

        # Save cleanly
        plt.savefig(os.path.join(output_dir, f"{regime}/plots/nn_vs_mm_response_curves_scaled.png"), bbox_inches="tight")
        plt.close()

def plot_horizontal_boxplot_noise_regimes(model_dict, output_dir, error_distributions=None):
    """
    Create a vertical stack of horizontal boxplots (1 per noise regime) showing
    Log-MAE for PySR, Michaelis-Menten, and Neural Network models.
    Styled for Cell Systems standards.
    """
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    if error_distributions is None:
        error_distributions = _collect_error_distributions(model_dict)

    regimes = list(error_distributions.keys())
    if not regimes:
        print("No regimes available for plotting.")
        return

    fig, axes = plt.subplots(nrows=len(regimes), ncols=1, figsize=(10, 2.8 * len(regimes)), sharex=True)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        errors = error_distributions.get(regime, {})
        error_df = _errors_to_dataframe(errors)
        if error_df.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            data=error_df,
            x="Log-MAE",
            y="Model",
            palette=MODEL_COLORS,
            orient="h",
            order=MODEL_LINE_ORDER,
            ax=ax
        )
        ax.set_title(_label_for_regime(regime), fontsize=14, weight='bold')
        ax.set_ylabel("")  # Avoid repeating "Model" label
        ax.tick_params(axis='both', labelsize=11)

    # Label only the bottom x-axis
    axes[-1].set_xlabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot.png"), dpi=300)
    plt.close()

    # No-outliers version
    fig, axes = plt.subplots(nrows=len(regimes), ncols=1, figsize=(10, 2.8 * len(regimes)), sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        errors = error_distributions.get(regime, {})
        error_df = _errors_to_dataframe(errors)
        if error_df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=error_df,
            x="Log-MAE",
            y="Model",
            palette=MODEL_COLORS,
            orient="h",
            order=MODEL_LINE_ORDER,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(_label_for_regime(regime), fontsize=14, weight='bold')
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=11)
    axes[-1].set_xlabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot_no_outliers.png"), dpi=300)
    plt.close()

def plot_vertical_boxplot_noise_regimes(model_dict, output_dir, error_distributions=None):
    """
    Create a single-row of vertical boxplots (1 per noise regime) showing
    Log-MAE for PySR, Michaelis-Menten, and Neural Network models.
    Each subplot has its own y-scale; style matches the horizontal version.
    """
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    if error_distributions is None:
        error_distributions = _collect_error_distributions(model_dict)

    regimes = list(error_distributions.keys())
    if not regimes:
        print("No regimes available for plotting (vertical).")
        return

    fig, axes = plt.subplots(nrows=1, ncols=len(regimes), figsize=(3.2 * len(regimes), 4.5), sharey=False)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        errors = error_distributions.get(regime, {})
        error_df = _errors_to_dataframe(errors)
        if error_df.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            data=error_df,
            x="Model",
            y="Log-MAE",
            palette=MODEL_COLORS,
            orient="v",
            order=MODEL_LINE_ORDER,
            ax=ax
        )
        ax.set_title(_label_for_regime(regime), fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_vertical_boxplot.png"), dpi=300)
    plt.close()

    # No-outliers version (vertical)
    fig, axes = plt.subplots(nrows=1, ncols=len(regimes), figsize=(3.2 * len(regimes), 4.5), sharey=False)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        errors = error_distributions.get(regime, {})
        error_df = _errors_to_dataframe(errors)
        if error_df.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=error_df,
            x="Model",
            y="Log-MAE",
            palette=MODEL_COLORS,
            orient="v",
            order=MODEL_LINE_ORDER,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(_label_for_regime(regime), fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Log-MAE", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_vertical_boxplot_no_outliers.png"), dpi=300)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate noise regimes using symbolic regression and compare with MM and NN.')
    parser.add_argument('--dataset', required=True, help='CSV dataset path')
    parser.add_argument('--dataset_size', type=int, help='Max samples to use')
    parser.add_argument('--features', type=str, help='Comma-separated list of features or "all"')
    parser.add_argument('--mode', choices=['filtered', 'full'], default='filtered', help='Noise regime definition to use')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducibility')
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dataset_path = Path(args.dataset)
    try:
        base_dir = dataset_path.parents[1]
    except IndexError:
        base_dir = dataset_path.parent
    output_dir = base_dir / f"noise_regimes_{args.mode}"
    evaluate_models(
        data,
        args.features,
        output_dir=str(output_dir),
        dataset_size=args.dataset_size,
        mode=args.mode,
        seed=seed,
    )

if __name__ == '__main__':
    main()
