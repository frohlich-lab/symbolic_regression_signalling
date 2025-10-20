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
from typing import Tuple

from sympy import symbols, lambdify
from pysr import PySRRegressor
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import load_dataset as nn_load_dataset, train_model, evaluate_model
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

from constants import PYSR_CONFIG
from plot_style import apply_cell_systems_style
from utils.seeding import resolve_seed, seed_everything

# Suppress all warnings
warnings.filterwarnings("ignore")
apply_cell_systems_style()

# Numerical stability constant for logs
EPS = 1e-20

MODEL_LINE_ORDER = ["PySR", "Michaelis-Menten", "Neural Network"]
MODEL_COLORS = {
    "PySR": "#E69F00",
    "Michaelis-Menten": "#009E73",
    "Neural Network": "#0072B2",
}
NN_DATASET_CAP = 20000
TARGET_COLUMN = "kcat_cg"

MM_REQUIRED_COLS = ["P_u", "k_off", "k_D", "k_cat", "tK"]

LOGGER = logging.getLogger(__name__)

# Define biochemical regimes (decoupled)
# Pair A: P_u vs tK
# Pair B: K_M vs P_u
REGIMES = {
    "pu_over_tk_high": lambda df, r: df[(df["P_u"] / df["tK"]) >= r],
    "pu_over_tk_low": lambda df, r: df[(df["tK"] / df["P_u"]) >= r],
    "km_over_pu_high": lambda df, r: df[(((df["k_off"] + df["k_cat"] + (df["k_inact"] if "k_inact" in df and not df["k_inact"].isnull().all() else 0)) / (df["k_D"] * df["k_off"])) / df["P_u"]) >= r],
    "km_over_pu_low": lambda df, r: df[(df["P_u"] / ((df["k_off"] + df["k_cat"] + (df["k_inact"] if "k_inact" in df and not df["k_inact"].isnull().all() else 0)) / (df["k_D"] * df["k_off"])) ) >= r],
}
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

def run_pysr(X, y, model_path, output_file):
    run_dir = os.path.dirname(model_path)
    # Consider the run cached if either a pickle or the CSV artifacts exist
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
    # Persist artifacts for future reuse
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

def _prepare_regime_sample(
    data: pd.DataFrame,
    dataset_size: int,
    seed: int,
    min_groups: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = data.copy().reset_index(drop=True)
    if dataset_size is None or dataset_size <= 0:
        return base, base

    oversample = dataset_size * max(min_groups, 1)
    symbolic_pool = (
        base.sample(n=oversample, random_state=seed).reset_index(drop=True)
        if len(base) > oversample else base
    )

    return symbolic_pool, base


def _build_nn_pool(
    base: pd.DataFrame,
    nn_cap: int,
    seed: int,
    exclude: pd.DataFrame = None,
) -> pd.DataFrame:
    if base.empty:
        return base
    pool = base.copy().reset_index(drop=True)
    if exclude is not None and not exclude.empty:
        common = [col for col in pool.columns if col in exclude.columns]
        if common:
            exclude_unique = exclude[common].drop_duplicates()
            merged = pool.merge(
                exclude_unique.assign(_mark=1),
                on=common,
                how='left',
            )
            pool = merged[merged['_mark'].isna()].drop(columns=['_mark']).reset_index(drop=True)
            if pool.empty:
                pool = base.copy().reset_index(drop=True)
    target = nn_cap if nn_cap and nn_cap > 0 else len(pool)
    replace = len(pool) < target
    return pool.sample(n=target, random_state=seed + 1, replace=replace).reset_index(drop=True)


def evaluate_models(data, features, output_dir, dataset_size, seed: int) -> None:
    seed_everything(seed)
    os.makedirs(output_dir, exist_ok=True)
    losses, model_dict = {}, {}

    for regime, filter_func in REGIMES.items():
        # Use a 1000x threshold for splits (≥1000 or ≤0.001)
        filtered = filter_func(data, 1000.0)
        symbolic_pool, nn_base = _prepare_regime_sample(
            filtered, dataset_size, seed, min_groups=len(REGIMES)
        )
        if symbolic_pool.empty:
            continue

        os.makedirs(os.path.join(output_dir, f"{regime}/processed"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/models/pysr"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, f"{regime}/models/nn"), exist_ok=True)

        symbolic_pool.to_csv(os.path.join(output_dir, f"{regime}/processed/filtered_data.csv"), index=False)

        capped_size = min(dataset_size, len(symbolic_pool)) if dataset_size else len(symbolic_pool)
        sample = (
            symbolic_pool.sample(n=capped_size, random_state=seed)
            if capped_size < len(symbolic_pool)
            else symbolic_pool.copy()
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

        y_pred_mm = _mm_predict(test_df)
        mm_loss = np.mean(
            np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_test, EPS)))
        )
        losses[f"{regime}_mm"] = mm_loss
        print(f"Michaelis-Menten Loss (test): {mm_loss}")

        nn_pool = _build_nn_pool(nn_base, NN_DATASET_CAP, seed, exclude=sample)
        nn_features = nn_pool.iloc[:, :-1]
        nn_target = nn_pool.iloc[:, -1]
        nn_X_train, nn_X_test, nn_y_train, nn_y_test = train_test_split(
            nn_features,
            nn_target,
            test_size=0.2,
            random_state=seed,
            shuffle=True,
        )

        nn_X_train_pre, nn_pipeline = preprocess_data(nn_X_train.values)
        nn_X_test_pre = nn_pipeline.transform(nn_X_test.values)
        nn_train_pre = pd.DataFrame(
            np.column_stack([nn_X_train_pre, np.log(nn_y_train.values)]),
            columns=list(nn_features.columns) + [TARGET_COLUMN],
        )
        nn_test_pre = pd.DataFrame(
            np.column_stack([nn_X_test_pre, np.log(nn_y_test.values)]),
            columns=list(nn_features.columns) + [TARGET_COLUMN],
        )

        train_pre = pd.DataFrame(
            np.column_stack([nn_pipeline.transform(train_df.iloc[:, :-1].values), np.log(y_train)]),
            columns=list(train_df.columns),
        )
        test_pre = pd.DataFrame(
            np.column_stack([nn_pipeline.transform(test_df.iloc[:, :-1].values), np.log(y_test)]),
            columns=list(test_df.columns),
        )
        full_pre = pd.DataFrame(
            np.column_stack([nn_pipeline.transform(sample.iloc[:, :-1].values), np.log(y_full)]),
            columns=list(sample.columns),
        )

        train_split = nn_train_pre.reset_index(drop=True)
        val_split = nn_test_pre.reset_index(drop=True)

        nn_path = os.path.join(output_dir, f"{regime}/models/nn/model.pth")
        if os.path.exists(nn_path):
            print(f"Loading existing NN model for {regime}")
            nn_model = train_model(
                train_split,
                val_split,
                nn_path,
                verbose=False,
                retrain=False,
                seed=seed,
            )
        else:
            print(f"Training new NN model for {regime}")
            nn_model = train_model(
                train_split,
                val_split,
                nn_path,
                verbose=False,
                seed=seed,
            )
        print(
            "NN splits | train: %d, val: %d, eval: %d (cap %d)",
            len(train_split),
            len(val_split),
            len(test_df),
            NN_DATASET_CAP,
        )
        nn_loss, _ = evaluate_model(nn_model, test_pre)
        losses[f"{regime}_nn"] = nn_loss
        print(f"NN Loss (test): {nn_loss}")

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

    plot_model_subregimes(model_dict, output_dir)
    plot_error_distributions(model_dict, output_dir)
    save_pysr_formulas(model_dict, features, os.path.join(output_dir,  "shared/results/pysr/all_pysr_formulas.txt"))
    plot_input_error_correlation(model_dict, output_dir)
    plot_model_error_correlation(model_dict, output_dir)
    plot_nn_vs_mm_response_curves_linear(model_dict, output_dir)
    plot_horizontal_boxplot_subregimes(model_dict, output_dir)
    plot_vertical_boxplot_subregimes(model_dict, output_dir)
    plot_kinetic_sanity_scatters(model_dict, output_dir, full_dataset=data)

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

    regimes = sorted(error_df['Regime'].unique())
    ncols = 3
    nrows = (len(regimes) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = axes.flatten()

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
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Log-space MAE Error Distributions by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "shared/plots/regime_loss_comparison.png"))

def plot_model_subregimes(model_dict, output_dir):

    models = ['pysr', 'mm', 'nn']
    # Use decoupled Pair A (P_u vs tK) only for this landscape
    regimes_per_column = {
        0: ['pu_over_tk_low'],   # P_u / tK < 1 — LEFT COLUMN
        1: ['pu_over_tk_high'],  # P_u / tK > 1 — RIGHT COLUMN
    }
    regime_markers = {
        'pu_over_tk_low': '^',
        'pu_over_tk_high': 'o',
    }
    cmap = cm.get_cmap('plasma', 256)  # Use a diverging colormap for better distinction
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), sharex=False, sharey=False)
    all_handles, all_labels = [], []
    regime_indices = {}
    for regime, models_entry in model_dict.items():
        eval_data, _ = _get_eval_partition(models_entry)
        if eval_data is None or eval_data.empty:
            continue
        sample_size = min(len(eval_data), 600)
        regime_indices[regime] = np.random.choice(
            eval_data.index.values,
            size=sample_size,
            replace=False,
        )

    # Collect all errors across all models and regimes for global vmin/vmax
    all_errors = []
    for model in models:
        for col in [0, 1]:
            for regime in regimes_per_column[col]:
                if regime not in model_dict:
                    continue
                indices = regime_indices.get(regime)
                if indices is None or len(indices) == 0:
                    continue

                eval_data, eval_pre = _get_eval_partition(model_dict[regime])
                data = eval_data.loc[indices]
                data_pre = eval_pre.loc[indices] if eval_pre is not None else None
                y_true = data.iloc[:, -1].values
                X = data.iloc[:, :-1]

                if model == 'pysr':
                    y_pred = model_dict[regime]['pysr'].predict(X.values)
                elif model == 'mm':
                    y_pred = _mm_predict(data)
                elif model == 'nn':
                    if data_pre is None:
                        continue
                    y_pred = evaluate_model(model_dict[regime]['nn'], data_pre)[1].detach().cpu().numpy().flatten()

                error = np.abs(np.log(np.maximum(y_pred, EPS)) - np.log(np.maximum(y_true, EPS)))
                if error.size:
                    all_errors.extend(error)

    if len(all_errors) == 0:
        print("Skipping error landscape plot: no matching regimes present.")
        return

    vmin = np.percentile(all_errors, 1)
    vmax = np.percentile(all_errors, 99)

    # Plotting
    for row, model in enumerate(models):
        for col in [0, 1]:
            ax = axes[row, col]
            subregimes = regimes_per_column[col]
            
            for regime in subregimes:
                if regime not in model_dict:
                    continue
                indices = regime_indices.get(regime)
                if indices is None or len(indices) == 0:
                    continue

                eval_data, eval_pre = _get_eval_partition(model_dict[regime])
                data = eval_data.loc[indices]
                data_pre = eval_pre.loc[indices] if eval_pre is not None else None
                y_true = data.iloc[:, -1].values
                X = data.iloc[:, :-1]

                if model == 'pysr':
                    y_pred = model_dict[regime]['pysr'].predict(X.values)
                elif model == 'mm':
                    y_pred = _mm_predict(data)
                elif model == 'nn':
                    if data_pre is None:
                        continue
                    y_pred = evaluate_model(model_dict[regime]['nn'], data_pre)[1].detach().cpu().numpy().flatten()

                error = np.abs(np.log(np.maximum(y_pred, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
                x_vals = (data["P_u"] / data["tK"]).values
                # Map Pair A subregimes to y scaling similar to prior convention
                if regime == 'pu_over_tk_high':
                    mm_vals = _mm_predict(data)
                    y_vals = mm_vals / data["P_u"]
                elif regime == 'pu_over_tk_low':
                    mm_vals = _mm_predict(data)
                    y_vals = mm_vals / data["tK"]
                else:
                    continue

                marker = regime_markers[regime]
                sc = ax.scatter(
                    x_vals, y_vals, c=error, cmap=cmap,
                    marker=marker, alpha=0.7, edgecolors='k', linewidths=0.2,
                    vmin=vmin, vmax=vmax, label=regime
                )

                if regime not in all_labels:
                    handle = plt.Line2D([0], [0], marker=marker, linestyle='',
                                        color='black', label=regime, markersize=8)
                    all_handles.append(handle)
                    all_labels.append(regime)

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel("P_u / tK")
            if col == 0:
                ylabel = r"K$_\mathsf{M}$ / tK"
            else:
                ylabel = r"K$_\mathsf{M}$ / P$_\mathsf{u}$"
            ax.set_ylabel(ylabel)

            regime_label = "< 0.001" if col == 0 else "≥ 1000"
            ax.set_title(f"{model.upper()} | P_u/tK {regime_label}")
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Per-row colorbar
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.72 - 0.30 * row, 0.015, 0.18])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"{model.upper()} Log-space MAE", fontsize=11)

    fig.legend(all_handles, all_labels, loc='upper center', ncol=4, frameon=False, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.9, 0.94])
    plt.savefig(os.path.join(output_dir, "shared/plots/error_landscape.png"))

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

        # Full NN prediction and error
        y_pred_nn = evaluate_model(model, preprocessed_data)[1].detach().cpu().numpy().flatten()
        log_mae = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))

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
            y_mm = _mm_predict_row(data.iloc[idx], pu_vals)

            # Plot
            ax = axs[i]
            ax.plot(pu_vals, y_nn, label="Neural Network", color=MODEL_COLORS['Neural Network'])
            ax.plot(pu_vals, y_mm, label="Michaelis-Menten", color=MODEL_COLORS['Michaelis-Menten'], linestyle='--')
            ax.scatter(
                original_pu,
                y_true[idx],
                color=MODEL_COLORS['Neural Network'],
                marker='o',
                s=100,
                label="Groundtruth kcat_cg",
            )
            ax.axvline(original_pu, color='gray', linestyle=':', linewidth=1.5, label='Sampled P_u')  
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if i % (n_samples // 2) == 0:
                ax.set_ylabel("Output")
            ax.set_title(f"Sample {i+1} | Log-MAE = {log_mae[idx]:.3f}")

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
    

def plot_horizontal_boxplot_subregimes(model_dict, output_dir):
    """
    Create a vertically stacked panel of horizontal boxplots, one per subregime,
    comparing log-MAE errors for PySR, Michaelis-Menten, Neural Network, and Theory.
    Styled for publication (Cell Systems standards).
    """
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    subregimes = ['pu_over_tk_high', 'pu_over_tk_low', 'km_over_pu_high', 'km_over_pu_low']
    subregime_labels = {
        'pu_over_tk_high': 'P_u / tK ≥ 1000',
        'pu_over_tk_low': 'P_u / tK ≤ 0.001',
        'km_over_pu_high': 'K_M / P_u ≥ 1000',
        'km_over_pu_low': 'K_M / P_u ≤ 0.001',
    }

    fig, axes = plt.subplots(nrows=len(subregimes), ncols=1, figsize=(10, 2.8 * len(subregimes)), sharex=True)

    for ax, regime in zip(axes, subregimes):
        if regime not in model_dict:
            continue

        models = model_dict[regime]
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_records = []

        # PySR
        y_pred_pysr = models['pysr'].predict(X.values)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'PySR', 'Log-MAE': e} for e in err_pysr])

        # Michaelis-Menten
        y_pred_mm = _mm_predict(X)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Log-MAE': e} for e in err_mm])

        # Neural Network
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))
        error_records.extend([{'Model': 'Neural Network', 'Log-MAE': e} for e in err_nn])

        # No "Theory" line for decoupled regimes

        # Plot for this regime
        error_df = pd.DataFrame(error_records)
        sns.boxplot(
            data=error_df,
            x="Log-MAE",
            y="Model",
            palette=MODEL_COLORS,
            orient="h",
            order=MODEL_LINE_ORDER,
            ax=ax
        )
        ax.set_title(subregime_labels[regime], fontsize=14, weight='bold')
        ax.set_ylabel("")  # remove repeated label
        ax.tick_params(axis='both', labelsize=11)

    # Only label bottom subplot's x-axis
    axes[-1].set_xlabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot.png"), dpi=300)
    plt.close()

    # Also save a no-outliers version of the horizontal boxplots
    fig, axes = plt.subplots(nrows=len(subregimes), ncols=1, figsize=(10, 2.8 * len(subregimes)), sharex=True)
    for ax, regime in zip(axes, subregimes):
        if regime not in model_dict:
            continue
        models = model_dict[regime]
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_records = []
        y_pred_pysr = models['pysr'].predict(X.values)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'PySR', 'Log-MAE': e} for e in err_pysr])

        y_pred_mm = _mm_predict(X)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Log-MAE': e} for e in err_mm])

        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Neural Network', 'Log-MAE': e} for e in err_nn])

        error_df = pd.DataFrame(error_records)
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
        ax.set_title(subregime_labels[regime], fontsize=14, weight='bold')
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=11)
    axes[-1].set_xlabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot_no_outliers.png"), dpi=300)
    plt.close()

def plot_kinetic_sanity_scatters(model_dict, output_dir, full_dataset=None):
    """
    Generate sanity scatter plots:
    - Plot A: x = P_u/tK, y = log-MAE of Michaelis-Menten, color = K_M/P_u
    - Plot B: x = K_M/P_u, y = log-MAE of Michaelis-Menten, color = K_M/P_u
    - Plot C: x = K_M/P_u, y = log-MAE of first-order approx (tK*P_u/K_M), color = K_M/P_u
    - Plot D: x = K_M/P_u, y = log-MAE of zero-order approx (tK), color = K_M/P_u
    Saved to shared/plots.
    """
    sns.set(style="whitegrid", font_scale=1.1, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    def get_full_df():
        if full_dataset is not None:
            return full_dataset.reset_index(drop=True)
        # Fallback to union of any available regime data
        frames = [v['data'] for k, v in model_dict.items() if 'data' in v]
        if not frames:
            return None
        return pd.concat(frames, axis=0, ignore_index=True).drop_duplicates()

    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)

    # Helper to compute K_M per row
    def compute_km(df):
        k_inact = df['k_inact'] if 'k_inact' in df.columns else 0.0
        KM = (df['k_cat'] + df['k_off'] + k_inact) / (df['k_D'] * df['k_off'])
        return KM

    # A: x = P_u/tK, y = MM log-MAE (use all samples) + aligned histogram below
    dfA = get_full_df()
    if dfA is not None and not dfA.empty:
        y_true = dfA.iloc[:, -1].values
        y_mm = _mm_predict(dfA)
        err_mm = np.abs(np.log(np.maximum(y_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        x_ratio = (dfA['P_u'] / dfA['tK']).values

        fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        # Scatter (top)
        axs[0].scatter(x_ratio, err_mm, color='tab:blue', s=14, alpha=0.6, edgecolors='none')
        axs[0].set_xscale('log', base=10)
        axs[0].set_yscale('log', base=10)
        axs[0].set_ylabel("Log-space MAE (MM)\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)")
        axs[0].set_title("MM Error vs P_u/tK")
        # Histogram (bottom)
        finite_x = x_ratio[np.isfinite(x_ratio) & (x_ratio > 0)]
        if finite_x.size > 0:
            min_exp = int(np.floor(np.log10(finite_x.min())))
            max_exp = int(np.ceil(np.log10(finite_x.max())))
            edges = 10.0 ** np.arange(min_exp, max_exp + 1)
            axs[1].hist(finite_x, bins=edges, color='gray', alpha=0.8)
        axs[1].set_xscale('log', base=10)
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("P_u / tK")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "shared/plots/sanity_scatter_mm_vs_pu_over_tk.png"), dpi=300)
        plt.close(fig)

    # B: x = K_M/P_u, y = MM log-MAE (use all samples) + aligned histogram below
    dfB = get_full_df()
    if dfB is not None and not dfB.empty:
        y_true = dfB.iloc[:, -1].values
        y_mm = _mm_predict(dfB)
        err_mm = np.abs(np.log(np.maximum(y_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        km = compute_km(dfB).values
        x_ratio = km / dfB['P_u'].values

        fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].scatter(x_ratio, err_mm, color='tab:blue', s=14, alpha=0.6, edgecolors='none')
        axs[0].set_xscale('log', base=10)
        axs[0].set_yscale('log', base=10)
        axs[0].set_ylabel("Log-space MAE (MM)\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)")
        axs[0].set_title("MM Error vs K_M/P_u")
        finite_x = x_ratio[np.isfinite(x_ratio) & (x_ratio > 0)]
        if finite_x.size > 0:
            min_exp = int(np.floor(np.log10(finite_x.min())))
            max_exp = int(np.ceil(np.log10(finite_x.max())))
            edges = 10.0 ** np.arange(min_exp, max_exp + 1)
            axs[1].hist(finite_x, bins=edges, color='gray', alpha=0.8)
        axs[1].set_xscale('log', base=10)
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("K_M / P_u")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "shared/plots/sanity_scatter_mm_vs_km_over_pu.png"), dpi=300)
        plt.close(fig)

        # C: First-order approx error vs K_M/P_u + aligned histogram
        y_first = (dfB['tK'].values * dfB['P_u'].values) / np.maximum(km, 1e-25)
        err_first = np.abs(np.log(np.maximum(y_first, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].scatter(x_ratio, err_first, color='tab:blue', s=14, alpha=0.6, edgecolors='none')
        axs[0].set_xscale('log', base=10)
        axs[0].set_yscale('log', base=10)
        axs[0].set_ylabel("Log-space MAE (First-order)\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)")
        axs[0].set_title("First-order Approx Error vs K_M/P_u")
        finite_x = x_ratio[np.isfinite(x_ratio) & (x_ratio > 0)]
        if finite_x.size > 0:
            min_exp = int(np.floor(np.log10(finite_x.min())))
            max_exp = int(np.ceil(np.log10(finite_x.max())))
            edges = 10.0 ** np.arange(min_exp, max_exp + 1)
            axs[1].hist(finite_x, bins=edges, color='gray', alpha=0.8)
        axs[1].set_xscale('log', base=10)
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("K_M / P_u")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "shared/plots/sanity_scatter_first_order_vs_km_over_pu.png"), dpi=300)
        plt.close(fig)

        # D: Zero-order approx error vs K_M/P_u + aligned histogram
        y_zero = dfB['tK'].values
        err_zero = np.abs(np.log(np.maximum(y_zero, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        fig, axs = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        axs[0].scatter(x_ratio, err_zero, color='tab:blue', s=14, alpha=0.6, edgecolors='none')
        axs[0].set_xscale('log', base=10)
        axs[0].set_yscale('log', base=10)
        axs[0].set_ylabel("Log-space MAE (Zero-order)\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)")
        axs[0].set_title("Zero-order Approx Error vs K_M/P_u")
        finite_x = x_ratio[np.isfinite(x_ratio) & (x_ratio > 0)]
        if finite_x.size > 0:
            min_exp = int(np.floor(np.log10(finite_x.min())))
            max_exp = int(np.ceil(np.log10(finite_x.max())))
            edges = 10.0 ** np.arange(min_exp, max_exp + 1)
            axs[1].hist(finite_x, bins=edges, color='gray', alpha=0.8)
        axs[1].set_xscale('log', base=10)
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("K_M / P_u")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "shared/plots/sanity_scatter_zero_order_vs_km_over_pu.png"), dpi=300)
        plt.close(fig)

def plot_vertical_boxplot_subregimes(model_dict, output_dir):
    """
    Create a single-row panel of vertical boxplots, one per subregime,
    comparing log-MAE errors for PySR, Michaelis-Menten, Neural Network, and Theory.
    Each subplot has its own y-scale; style matches the horizontal version.
    """
    sns.set(style="whitegrid", font_scale=1.2, rc={"axes.edgecolor": "black", "axes.linewidth": 1.0})

    subregimes = ['pu_over_tk_high', 'pu_over_tk_low', 'km_over_pu_high', 'km_over_pu_low']
    subregime_labels = {
        'pu_over_tk_high': 'P_u / tK ≥ 1000',
        'pu_over_tk_low': 'P_u / tK ≤ 0.001',
        'km_over_pu_high': 'K_M / P_u ≥ 1000',
        'km_over_pu_low': 'K_M / P_u ≤ 0.001',
    }

    fig, axes = plt.subplots(nrows=1, ncols=len(subregimes), figsize=(3.2 * len(subregimes), 4.5), sharey=False)
    if len(subregimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, subregimes):
        if regime not in model_dict:
            ax.set_visible(False)
            continue

        models = model_dict[regime]
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            ax.set_visible(False)
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]

        error_records = []

        # PySR
        y_pred_pysr = models['pysr'].predict(X.values)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'PySR', 'Log-MAE': e} for e in err_pysr])

        # Michaelis-Menten
        y_pred_mm = _mm_predict(X)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Log-MAE': e} for e in err_mm])

        # Neural Network
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Neural Network', 'Log-MAE': e} for e in err_nn])

        # No "Theory" line for decoupled regimes

        # Plot for this regime (vertical orientation)
        error_df = pd.DataFrame(error_records)
        sns.boxplot(
            data=error_df,
            x="Model",
            y="Log-MAE",
            palette=MODEL_COLORS,
            orient="v",
            order=MODEL_LINE_ORDER,
            ax=ax
        )
        ax.set_title(subregime_labels[regime], fontsize=14, weight='bold')
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
    fig, axes = plt.subplots(nrows=1, ncols=len(subregimes), figsize=(3.2 * len(subregimes), 4.5), sharey=False)
    if len(subregimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, subregimes):
        if regime not in model_dict:
            ax.set_visible(False)
            continue
        models = model_dict[regime]
        data, data_pre = _get_eval_partition(models)
        if data is None or data.empty:
            ax.set_visible(False)
            continue
        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]
        error_records = []
        y_pred_pysr = models['pysr'].predict(X.values)
        err_pysr = np.abs(np.log(np.maximum(y_pred_pysr, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'PySR', 'Log-MAE': e} for e in err_pysr])
        y_pred_mm = _mm_predict(X)
        err_mm = np.abs(np.log(np.maximum(y_pred_mm, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Michaelis-Menten', 'Log-MAE': e} for e in err_mm])
        y_pred_nn = evaluate_model(models['nn'], data_pre)[1].detach().cpu().numpy().flatten()
        err_nn = np.abs(np.log(np.maximum(y_pred_nn, 1e-25)) - np.log(np.maximum(y_true, 1e-25)))
        error_records.extend([{'Model': 'Neural Network', 'Log-MAE': e} for e in err_nn])
        # No "Theory" line for decoupled regimes
        error_df = pd.DataFrame(error_records)
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
        ax.set_title(subregime_labels[regime], fontsize=14, weight='bold')
        ax.tick_params(axis='both', labelsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
        if ax != axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Log-space MAE\nMean |ln(y_hat + ε) − ln(y + ε)| (ε=1e−20)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_vertical_boxplot_no_outliers.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate biochemical regimes using symbolic regression and compare with MM and NN.')
    parser.add_argument('--dataset', required=True, help='CSV dataset path')
    parser.add_argument('--dataset_size', type=int, help='Max samples to use')
    parser.add_argument('--features', type=str, help='Comma-separated list of features or "all"')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducibility')
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    dir_path = '/'.join(args.dataset.split('/')[:3])
    evaluate_models(
        data,
        args.features,
        output_dir=dir_path + '/kinetic_regimes/',
        dataset_size=args.dataset_size,
        seed=seed,
    )


if __name__ == '__main__':
    main()
