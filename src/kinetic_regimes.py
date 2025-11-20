# kinetic_regimes.py
"""
Symbolic regression across biochemical **ratio** regimes (e.g. P_u/tK and K_M/P_u),
evaluating PySR vs. Michaelis–Menten vs. NN for both sQSSA/tQSSA variants.

Matches the unified logging, metrics, and plotting style used elsewhere.

Usage:
    python kinetic_regimes.py \
        --dataset <path_to_dataset> \
        --dataset_size <size> \
        --features <feature_list|all> \
        --seed 42
"""

import os
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import train_model, evaluate_model

from constants import PYSR_CONFIG
from plot_style import apply_cell_systems_style
from utils.seeding import resolve_seed, seed_everything

# ── MM + Variant machinery ─────────────────────────────────────────────────────
from mm_models import (
    EPS as MM_EPS,
    TARGET_COLUMN as MM_TARGET_COLUMN,
    mm_predictions,
    mm_required_columns,
)
from regime_variants import (
    MODEL_COLOR_MAP,
    MODEL_COMBINATIONS,
    MODEL_LINE_ORDER,
    VARIANTS,
    augment_for_variant,
    collect_variant_predictions,
    collect_pysr_formula_lines,
    gather_model_errors,
    metric_keys,
    metric_spec,
    metric_errors,
    metric_summary,
    format_regime_message,
    format_variant_split,
    format_model_metrics,
    model_display_name,
    variant_suffix,
    regime_logger,
    aggregate_seed_error_statistics,
    clip_metric_values,
    metric_limits_with_margin,
)

# ── Style & logging ────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
apply_cell_systems_style()

LOGGER = regime_logger("pysr_regimes_ratio")
CURRENT_SEED_TAG: str = ""

def _emit(msg: str) -> None:
    prefix = CURRENT_SEED_TAG
    if prefix:
        first_line = msg.lstrip()
        if not first_line.startswith(prefix):
            msg = f"{prefix} {msg}"
    LOGGER.info(msg)
    print(msg)

# ── Constants ──────────────────────────────────────────────────────────────────
NN_DATASET_CAP = 20000
TARGET_COLUMN = MM_TARGET_COLUMN
EPS = MM_EPS
LOG_SCALE_FLOOR = 1e-3

# ── Kinetic ratio regimes ──────────────────────────────────────────────────────
REGIME_THRESHOLD = 1_000.0

REGIME_LABELS = {
    "pu_over_tk_high": "P_u / tK ≥ 1000",
    "pu_over_tk_low": "P_u / tK ≤ 0.001",
    "km_over_pu_high": "K_M / P_u ≥ 1000",
    "km_over_pu_low": "K_M / P_u ≤ 0.001",
}
REGIME_ORDER = {
    "pu_over_tk_high": 0,
    "pu_over_tk_low": 1,
    "km_over_pu_high": 2,
    "km_over_pu_low": 3,
}
def _label_for_regime(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime)

def _ordered_regimes(model_dict: Dict[str, Any]):
    return sorted(model_dict.keys(), key=lambda r: (REGIME_ORDER.get(r, float("inf")), r))

def _km_term(df: pd.DataFrame) -> pd.Series:
    k_inact = df["k_inact"] if "k_inact" in df.columns and not df["k_inact"].isnull().all() else 0.0
    numerator = df["k_off"] + df["k_cat"] + k_inact
    denominator = df["k_D"] * df["k_off"]
    return numerator / denominator

def _pu_over_tk(df: pd.DataFrame) -> pd.Series:
    return df["P_u"] / df["tK"]

def _km_over_pu(df: pd.DataFrame) -> pd.Series:
    return _km_term(df) / df["P_u"]

REGIMES = OrderedDict({
    "pu_over_tk_high":  lambda df: df[_pu_over_tk(df) >= REGIME_THRESHOLD],
    "pu_over_tk_low":   lambda df: df[(1.0 / _pu_over_tk(df)) >= REGIME_THRESHOLD],
    "km_over_pu_high":  lambda df: df[_km_over_pu(df) >= REGIME_THRESHOLD],
    "km_over_pu_low":   lambda df: df[(1.0 / _km_over_pu(df)) >= REGIME_THRESHOLD],
})

# ── IO & preprocessing ─────────────────────────────────────────────────────────
def load_dataset(file_path: str, dataset_size: Optional[int] = None, features: Optional[str] = None):
    data = pd.read_csv(file_path)
    if features and features != "all":
        cols = [c.strip() for c in features.split(",") if c.strip()]
        data = data[cols]
    # Source is log-space → map back to linear
    return data.map(np.exp)

def preprocess_data(X: np.ndarray):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler()),
    ])
    return pipeline.fit_transform(X), pipeline

# ── PySR runner & formula dumping ──────────────────────────────────────────────
def run_pysr(
    X,
    y,
    model_path: str,
    log_prefix: Optional[str] = None,
    config_override: Optional[Dict[str, object]] = None,
):
    model_path = Path(model_path)
    run_dir = model_path.parent
    base_dir = run_dir.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{log_prefix} PySR" if log_prefix else "[PySR]"
    run_id = run_dir.name

    candidates = [
        model_path,
        run_dir / "hall_of_fame.pkl",
        run_dir / "hall_of_fame.csv",
        run_dir / "equations.csv",
        run_dir / "model.pkl",
    ]
    load_target = next((path for path in candidates if path.exists()), None)
    if load_target is None and any(run_dir.iterdir()):
        load_target = run_dir

    if load_target is not None:
        _emit(f"{prefix} Loading existing model from {load_target}")
        return PySRRegressor.from_file(run_directory=str(run_dir))

    _emit(f"{prefix} Training new model in {run_dir}")
    config = dict(PYSR_CONFIG)
    if config_override:
        config.update(config_override)
    model = PySRRegressor(
        **config,
        output_directory=str(base_dir),
        run_id=run_id,
    )
    model.fit(X, y)
    try:
        model.save(str(model_path))
    except Exception:
        try:
            model.save()
        except Exception:
            pass
    return model

def save_pysr_formulas(model_dict, features, output_dir):
    out_dir = Path(output_dir) / "shared/results/pysr"
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_lines = collect_pysr_formula_lines(model_dict, features)
    combined: list[str] = []
    for variant_key, lines in variant_lines.items():
        if not lines:
            continue
        combined.extend(lines)
        (out_dir / f"all_pysr_formulas_{variant_key}.txt").write_text("\n".join(lines) + "\n")

    if combined:
        (out_dir / "all_pysr_formulas.txt").write_text("\n".join(combined) + "\n")

# ── Sampling helpers ───────────────────────────────────────────────────────────
def _prepare_regime_sample(
    data: pd.DataFrame,
    dataset_size: Optional[int],
    seed: int,
    min_groups: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = data.copy().reset_index(drop=True)
    if not dataset_size or dataset_size <= 0:
        return base, base
    oversample = dataset_size * max(min_groups, 1)
    symbolic_pool = base.sample(n=min(oversample, len(base)), random_state=seed, replace=False).reset_index(drop=True)
    return symbolic_pool, base

def _build_nn_pool(
    base: pd.DataFrame,
    nn_cap: int,
    seed: int,
    exclude: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if base.empty:
        return base
    pool = base.copy().reset_index(drop=True)
    if exclude is not None and not exclude.empty:
        common = [col for col in pool.columns if col in exclude.columns]
        if common:
            exclude_unique = exclude[common].drop_duplicates()
            merged = pool.merge(exclude_unique.assign(_mark=1), on=common, how='left')
            pool = merged[merged['_mark'].isna()].drop(columns=['_mark']).reset_index(drop=True)
            if pool.empty:
                pool = base.copy().reset_index(drop=True)
    target = nn_cap if nn_cap and nn_cap > 0 else len(pool)
    target = min(target, len(pool))
    return pool.sample(n=target, random_state=seed + 1, replace=False).reset_index(drop=True)

# ── Metric-aware distributions & plots ─────────────────────────────────────────
def _collect_error_distributions(
    model_dict: Dict[str, Dict[str, Any]]
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, List[float]]]],
]:
    """Returns aggregated error distributions and per-seed medians."""
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]] = OrderedDict()

    for regime in _ordered_regimes(model_dict):
        models = model_dict.get(regime)
        if not models:
            continue

        aggregated_errors = models.get("aggregated_errors", {})
        aggregated_seed_medians = models.get("aggregated_seed_medians", {})
        metric_maps = {mkey: {} for mkey in metric_keys()}
        median_maps = {mkey: {} for mkey in metric_keys()}

        for family, variant_key in MODEL_COMBINATIONS:
            display_name = model_display_name(family, variant_key)
            variant_errors = aggregated_errors.get(variant_key)
            if variant_errors:
                for mkey in metric_keys():
                    model_errors = variant_errors.get(mkey, {})
                    errs = model_errors.get(display_name)
                    if errs is None:
                        continue
                    metric_maps[mkey][display_name] = np.asarray(errs, dtype=float)
                    medians = (
                        aggregated_seed_medians
                        .get(variant_key, {})
                        .get(mkey, {})
                        .get(display_name)
                    )
                    if medians is not None:
                        median_maps[mkey][display_name] = list(medians)
                continue

            snapshot = collect_variant_predictions(models, variant_key, evaluate_model)
            if snapshot is None:
                continue
            preds = snapshot["predictions"].get(display_name)
            if preds is None:
                continue
            y_true = snapshot["y_true"]
            for mkey in metric_keys():
                errs = metric_errors(mkey, preds, y_true)
                metric_maps[mkey][display_name] = errs
                finite = errs[np.isfinite(errs)]
                median_value = float(np.median(finite)) if finite.size else float("nan")
                median_maps[mkey][display_name] = [median_value]

        if any(metric_maps[m] for m in metric_keys()):
            distributions[regime] = metric_maps
            seed_medians[regime] = median_maps

    return distributions, seed_medians

def _metric_slice(distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]], metric_key: str):
    out: Dict[str, Dict[str, np.ndarray]] = OrderedDict()
    for regime, m in distributions.items():
        if metric_key in m and m[metric_key]:
            out[regime] = m[metric_key]
    return out

def _errors_to_dataframe(error_dict: Dict[str, np.ndarray], metric_key: str) -> pd.DataFrame:
    records = []
    for model_name, errors in error_dict.items():
        arr = np.asarray(errors, dtype=float)
        arr = arr[np.isfinite(arr)]
        records.extend({"Model": model_name, metric_key: v} for v in arr)
    return pd.DataFrame.from_records(records)

def _combined_error_frame(metric_distributions: Dict[str, Dict[str, np.ndarray]], metric_key: str) -> pd.DataFrame:
    frames = []
    for regime, dist in metric_distributions.items():
        f = _errors_to_dataframe(dist, metric_key)
        if not f.empty:
            f["Regime"] = _label_for_regime(regime)
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Model", metric_key, "Regime"])

def _compute_line_stats(metric_distributions: Dict[str, Dict[str, np.ndarray]]):
    stats = {model: {"median": [], "q1": [], "q3": []} for model in MODEL_LINE_ORDER}
    for _, model_map in metric_distributions.items():
        for model in MODEL_LINE_ORDER:
            arr = np.asarray(model_map.get(model, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                stats[model]["median"].append(np.median(arr))
                stats[model]["q1"].append(np.quantile(arr, 0.25))
                stats[model]["q3"].append(np.quantile(arr, 0.75))
            else:
                stats[model]["median"].append(np.nan)
                stats[model]["q1"].append(np.nan)
                stats[model]["q3"].append(np.nan)
    return stats

def _lineplot_limits(stats, spec):
    if spec.floor is not None and spec.ceiling is not None:
        lower, upper = metric_limits_with_margin(spec.key)
        return lower, upper
    lowers, uppers, positives = [], [], []
    for model in MODEL_LINE_ORDER:
        lowers.extend([v for v in stats[model]["q1"] if np.isfinite(v)])
        uppers.extend([v for v in stats[model]["q3"] if np.isfinite(v)])
        if spec.y_scale == "log":
            positives.extend([v for v in stats[model]["median"] if np.isfinite(v) and v > 0])
            positives.extend([v for v in stats[model]["q1"] if np.isfinite(v) and v > 0])
            positives.extend([v for v in stats[model]["q3"] if np.isfinite(v) and v > 0])
    if not lowers or not uppers:
        return None, None
    ymin, ymax = min(lowers), max(uppers)
    if np.isclose(ymin, ymax):
        pad = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
        ymin -= pad; ymax += pad
    else:
        pad = 0.05 * (ymax - ymin)
        ymin -= pad; ymax += pad
    if spec.y_scale == "log":
        if positives and ymin <= 0:
            ymin = min(positives) * 0.8
        floor = spec.floor or LOG_SCALE_FLOOR
        ymin = max(ymin, floor)
        if ymax <= ymin:
            ymax = ymin * 1.5
    else:
        ymin = max(ymin, 0.0)
        if ymax <= ymin:
            ymax = ymin + 0.1 * max(ymin, 1.0)
    return ymin, ymax

def _plot_deviation_lineplot_for_metric(
    metric_key: str,
    metric_distributions: Dict[str, Dict[str, np.ndarray]],
    seed_medians: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    template: bool = False,
):
    if not metric_distributions:
        return
    spec = metric_spec(metric_key)
    plot_dir = Path(output_dir) / "shared/plots" / spec.folder
    plot_dir.mkdir(parents=True, exist_ok=True)

    regimes = list(metric_distributions.keys())
    stats = _compute_line_stats(metric_distributions)
    for model in MODEL_LINE_ORDER:
        for k in ("median", "q1", "q3"):
            stats_array = clip_metric_values(
                np.asarray(stats[model][k], dtype=float),
                metric_key,
            )
            if spec.y_scale == "log":
                floor = spec.floor or LOG_SCALE_FLOOR
                stats_array = np.where(
                    np.isfinite(stats_array) & (stats_array < floor),
                    floor,
                    stats_array,
                )
            stats[model][k] = stats_array

    ymin, ymax = _lineplot_limits(stats, spec)

    x = np.arange(len(regimes), dtype=float)
    x_labels = [_label_for_regime(r) for r in regimes]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for model in MODEL_LINE_ORDER:
        med = np.asarray(stats[model]["median"], dtype=float)
        q1 = np.asarray(stats[model]["q1"], dtype=float)
        q3 = np.asarray(stats[model]["q3"], dtype=float)
        color = MODEL_COLOR_MAP[model]
        if template:
            ax.plot(x, med, color=color, linestyle="--", alpha=0.3)
        else:
            ax.plot(x, med, marker="o", label=model, color=color)
            ax.fill_between(x, q1, q3, color=color, alpha=0.15)
            for idx, regime in enumerate(regimes):
                seed_values = np.asarray(
                    seed_medians.get(regime, {}).get(model, []),
                    dtype=float,
                )
                if seed_values.size == 0:
                    continue
                seed_values = seed_values[np.isfinite(seed_values)]
                if seed_values.size == 0:
                    continue
                seed_values = clip_metric_values(seed_values, metric_key)
                jitter = np.full(seed_values.shape, x[idx], dtype=float)
                ax.scatter(
                    jitter,
                    seed_values,
                    color=color,
                    alpha=0.4,
                    s=28,
                    linewidths=0,
                    zorder=4,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20, ha="right")
    ax.set_xlabel("Regime")
    ax.set_ylabel(spec.axis_label)
    ax.set_yscale(spec.y_scale)
    lower_limit, upper_limit = metric_limits_with_margin(metric_key)
    if lower_limit is not None and upper_limit is not None:
        ax.set_ylim(lower_limit, upper_limit)
    elif ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if not template:
        ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    suffix = "_template" if template else ""
    plt.savefig(plot_dir / f"{spec.filename_prefix}_regime_lineplot{suffix}.png", dpi=300)
    plt.close()

def plot_deviation_lineplots(
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_dir: str,
    template: bool = False,
):
    for mkey in metric_keys():
        metric_dist = _metric_slice(distributions, mkey)
        if not metric_dist:
            continue
        metric_seed_medians = {
            regime: seed_medians.get(regime, {}).get(mkey, {})
            for regime in metric_dist.keys()
        }
        _plot_deviation_lineplot_for_metric(
            mkey,
            metric_dist,
            metric_seed_medians,
            output_dir,
            template=template,
        )

def _plot_error_distributions(metric_key: str, metric_distributions: Dict[str, Dict[str, np.ndarray]], output_dir: str):
    df = _combined_error_frame(metric_distributions, metric_key)
    if df.empty:
        return
    regimes = list(metric_distributions.keys())
    ncols = min(4, len(regimes))
    nrows = int(np.ceil(len(regimes) / max(1, ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]
    spec = metric_spec(metric_key)
    metric_dir = Path(output_dir) / "shared/plots" / spec.folder
    metric_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df[metric_key] = clip_metric_values(df[metric_key].to_numpy(), metric_key)
    lower_limit, upper_limit = metric_limits_with_margin(metric_key)

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        subset = df[df["Regime"] == _label_for_regime(regime)]
        if subset.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(
            data=subset,
            x="Model",
            y=metric_key,
            ax=ax,
            inner="box",
            palette=palette,
            order=MODEL_LINE_ORDER,
        )
        ax.set_title(_label_for_regime(regime))
        ax.set_xlabel("")
        if idx % ncols == 0:
            ax.set_ylabel(spec.axis_label)
        else:
            ax.set_ylabel("")
        if spec.y_scale == "log":
            ax.set_yscale("log")
            if lower_limit is not None and upper_limit is not None:
                ax.set_ylim(lower_limit, upper_limit)
        elif lower_limit is not None and upper_limit is not None:
            ax.set_ylim(lower_limit, upper_limit)
    for j in range(len(regimes), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(f"{spec.label} Error Distributions by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(metric_dir / f"{spec.filename_prefix}_error_distributions.png")
    plt.close()

def plot_error_distributions(model_dict, output_dir):
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        _emit("No error distributions available; skipping violin plots")
        return
    for mkey in metric_keys():
        metric_map = _metric_slice(distributions, mkey)
        if not metric_map:
            continue
        _plot_error_distributions(mkey, metric_map, output_dir)

def plot_horizontal_boxplots(model_dict, output_dir):
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return
    regimes = list(distributions.keys())
    for mkey in metric_keys():
        metric_dist = _metric_slice(distributions, mkey)
        if not metric_dist:
            continue
        df = _combined_error_frame(metric_dist, mkey)
        if df.empty:
            continue
        spec = metric_spec(mkey)
        plot_dir = Path(output_dir) / "shared/plots" / spec.folder
        plot_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df[mkey] = clip_metric_values(df[mkey].to_numpy(), mkey)
        lower_limit, upper_limit = metric_limits_with_margin(mkey)
        floor = spec.floor or LOG_SCALE_FLOOR if spec.y_scale == "log" else None

        def _render(showfliers: bool, suffix: str):
            fig, axes = plt.subplots(nrows=len(regimes), ncols=1, figsize=(10, 2.6 * len(regimes)), sharex=True)
            axes = np.atleast_1d(axes)
            for ax, regime in zip(axes, regimes):
                subset = df[df["Regime"] == _label_for_regime(regime)]
                if subset.empty:
                    ax.set_visible(False); continue
                sns.boxplot(
                    data=subset,
                    x=mkey,
                    y="Model",
                    palette=[MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER],
                    orient="h",
                    order=MODEL_LINE_ORDER,
                    ax=ax,
                    showfliers=showfliers,
                )
                ax.set_title(_label_for_regime(regime), fontsize=14, weight="bold")
                ax.set_ylabel("")
                if spec.y_scale == "log":
                    ax.set_xscale("log")
                    if lower_limit is not None and upper_limit is not None:
                        ax.set_xlim(lower_limit, upper_limit)
                elif lower_limit is not None and upper_limit is not None:
                    ax.set_xlim(lower_limit, upper_limit)
            axes[-1].set_xlabel(spec.axis_label, fontsize=12)
            for ax in axes[:-1]: ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{spec.filename_prefix}_horizontal_boxplot{suffix}.png", dpi=300)
            plt.close()
        _render(True, ""); _render(False, "_no_outliers")

def plot_vertical_boxplots(model_dict, output_dir):
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return
    regimes = list(distributions.keys())
    for mkey in metric_keys():
        metric_dist = _metric_slice(distributions, mkey)
        if not metric_dist:
            continue
        df = _combined_error_frame(metric_dist, mkey)
        if df.empty:
            continue
        spec = metric_spec(mkey)
        plot_dir = Path(output_dir) / "shared/plots" / spec.folder
        plot_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df[mkey] = clip_metric_values(df[mkey].to_numpy(), mkey)
        lower_limit, upper_limit = metric_limits_with_margin(mkey)

        def _render(showfliers: bool, suffix: str):
            fig, axes = plt.subplots(nrows=1, ncols=len(regimes), figsize=(3.2 * len(regimes), 4.5), sharey=False)
            if len(regimes) == 1:
                axes = [axes]
            for ax, regime in zip(axes, regimes):
                subset = df[df["Regime"] == _label_for_regime(regime)]
                if subset.empty:
                    ax.set_visible(False); continue
                sns.boxplot(
                    data=subset,
                    x="Model",
                    y=mkey,
                    palette=[MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER],
                    orient="v",
                    order=MODEL_LINE_ORDER,
                    ax=ax,
                    showfliers=showfliers,
                )
                ax.set_title(_label_for_regime(regime), fontsize=14, weight="bold")
                if spec.y_scale == "log":
                    ax.set_yscale("log")
                    if lower_limit is not None and upper_limit is not None:
                        ax.set_ylim(lower_limit, upper_limit)
                for label in ax.get_xticklabels():
                    label.set_rotation(25); label.set_horizontalalignment("right")
                if ax != axes[0]:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(spec.axis_label, fontsize=12)
                if lower_limit is not None and upper_limit is not None and spec.y_scale != "log":
                    ax.set_ylim(lower_limit, upper_limit)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{spec.filename_prefix}_vertical_boxplot{suffix}.png", dpi=300)
            plt.close()
        _render(True, ""); _render(False, "_no_outliers")

# ── Shared landscape & correlation plots (unchanged semantics) ─────────────────
def plot_model_subregimes(model_dict, output_dir):
    regimes = list(REGIMES.keys())
    n_rows, n_cols = len(regimes), len(MODEL_COMBINATIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharey=True)

    rng = np.random.default_rng(42)
    sampled = {}
    for regime_name in regimes:
        regime_models = model_dict.get(regime_name, {})
        snapshot_any = None
        for _, variant_key in MODEL_COMBINATIONS:
            snapshot = collect_variant_predictions(regime_models, variant_key, evaluate_model)
            if snapshot is not None:
                snapshot_any = snapshot
                break
        if snapshot_any is None:
            continue
        idx = snapshot_any["features"].index
        sample_size = min(len(idx), 600)
        chosen = idx if len(idx) <= sample_size else rng.choice(idx, size=sample_size, replace=False)
        for _, variant_key in MODEL_COMBINATIONS:
            sampled[(regime_name, variant_key)] = chosen

    for row_idx, regime_name in enumerate(regimes):
        regime_models = model_dict.get(regime_name, {})
        for col_idx, (family, variant_key) in enumerate(MODEL_COMBINATIONS):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            display_name = model_display_name(family, variant_key)
            snapshot = collect_variant_predictions(regime_models, variant_key, evaluate_model)
            chosen = sampled.get((regime_name, variant_key))
            if snapshot is None or chosen is None or len(chosen) == 0:
                ax.set_axis_off(); continue

            predictions = snapshot["predictions"]
            data_subset = snapshot["data"].loc[chosen]
            positions = data_subset.index.map(snapshot["data"].index.get_loc).to_numpy()
            y_true = snapshot["y_true"][positions]
            pred = predictions.get(display_name)
            if pred is None:
                ax.set_axis_off(); continue
            pred = pred[positions]

            errors = np.abs(np.log(np.maximum(pred, EPS)) - np.log(np.maximum(y_true, EPS)))
            ax.scatter(
                y_true, errors, alpha=0.4, s=18,
                color=MODEL_COLOR_MAP[display_name],
                label=display_name, edgecolors="k", linewidths=0.2,
            )
            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('Log-space MAE |ln(ŷ+ε) − ln(y+ε)| (ε=1e−20)', fontsize=10)
            if row_idx == 0:
                ax.set_title(display_name, fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left")
            ax.set_xscale("log")

    fig.suptitle("Model Error Landscape by Regime and Model", fontsize=16, y=1.02)
    plt.tight_layout()
    Path(output_dir, "shared/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "shared/plots/error_landscape.png", dpi=300)
    plt.close()

def plot_input_error_correlation(model_dict, output_dir):
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    all_error_df = []
    filled = -1
    for idx, (regime, models) in enumerate(model_dict.items()):
        base_df, error_map = gather_model_errors(models, evaluate_model)
        if base_df is None or not error_map:
            continue
        error_df = pd.DataFrame(error_map)
        feature_cols = [c for c in base_df.columns if c != TARGET_COLUMN]
        for c in feature_cols:
            error_df[c] = base_df[c].values
        corr = error_df.corr()[list(error_map.keys())].drop(error_map.keys(), axis=0)
        all_error_df.append(error_df)
        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(_label_for_regime(regime))
        filled = idx

    for j in range(filled + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature–Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output_dir, "shared/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(output_dir) / "shared/plots/feature_error_correlation_grid.png")
    plt.close()

    if all_error_df:
        full_df = pd.concat(all_error_df, ignore_index=True)
        error_columns = [c for c in full_df.columns if c in MODEL_COLOR_MAP]
        overall_corr = full_df.corr()[error_columns].drop(error_columns, axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(overall_corr, annot=True, cmap="coolwarm", center=0)
        plt.title("Overall Feature–Error Correlation (All Regimes)")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "shared/plots/feature_error_correlation_overall.png")
        plt.close()

def plot_model_error_correlation(model_dict, output_dir):
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    all_model_error_df = []
    filled = -1
    for idx, (regime, models) in enumerate(model_dict.items()):
        _, error_map = gather_model_errors(models, evaluate_model)
        if not error_map:
            continue
        df = pd.DataFrame(error_map)
        all_model_error_df.append(df)
        corr = df.corr()
        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(_label_for_regime(regime))
        filled = idx

    for j in range(filled + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model–Model Error Correlations by Regime", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(Path(output_dir) / "shared/plots/model_error_correlation_grid.png")
    plt.close()

    if all_model_error_df:
        df_all = pd.concat(all_model_error_df, ignore_index=True)
        overall_corr = df_all.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(overall_corr, annot=True, cmap="coolwarm", center=0)
        plt.title("Overall Model–Model Error Correlation (All Regimes)")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "shared/plots/model_error_correlation_overall.png")
        plt.close()

def plot_nn_vs_mm_response_curves_linear(model_dict, output_dir, n_samples=10, n_pu_points=200):
    for regime, models in model_dict.items():
        _emit(f"Generating scaled response curves for {regime}...")
        data = models.get("test_data")
        if data is None or data.empty:
            data = models.get("data")
        pre = models.get("test_preprocessed")
        if pre is None:
            pre = models.get("preprocessed")
        if data is None or data.empty:
            _emit(f"Skipping {regime}: no evaluation data available.")
            continue
        model = models.get("nn")
        if model is None or "P_u" not in data.columns:
            _emit(f"Skipping {regime}: NN model or P_u missing")
            continue

        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]
        pu_index = X.columns.get_loc("P_u")

        _, pipeline = preprocess_data(X.values)

        y_pred_mm = mm_predictions(data)["sQSSA"]
        y_pred_nn = evaluate_model(model, pre)[1].detach().cpu().numpy().flatten()
        log_mae_nn = np.abs(np.log(np.maximum(y_pred_nn, EPS)) - np.log(np.maximum(y_true, EPS)))
        log_mae_mm = np.abs(np.log(np.maximum(y_pred_mm, EPS)) - np.log(np.maximum(y_true, EPS)))

        percentiles = np.linspace(0, 100, n_samples + 2)[1:-1]
        thresholds = np.percentile(log_mae_nn, percentiles)
        chosen = []
        for t in thresholds:
            idx = int(np.argmin(np.abs(log_mae_nn - t)))
            if idx not in chosen:
                chosen.append(idx)
        if not chosen:
            chosen = list(range(min(n_samples, len(log_mae_nn))))

        grid_cols = max(1, int(np.ceil(len(chosen) / 2)))
        grid_rows = max(1, int(np.ceil(len(chosen) / grid_cols)))
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(10 * grid_cols, 5 * grid_rows))
        axs = np.atleast_1d(axs).flatten()

        nn_label = model_display_name("nn", "sQSSA")
        mm_label = model_display_name("mm", "sQSSA")
        for i, idx in enumerate(chosen[: len(axs)]):
            fixed_sample = X.iloc[idx].copy()
            original_pu = fixed_sample["P_u"]
            pu_vals = np.linspace(0.01 * original_pu, 5 * original_pu, n_pu_points)

            varied_inputs = np.tile(fixed_sample.values, (n_pu_points, 1))
            varied_inputs[:, pu_index] = pu_vals
            X_var_pre = pipeline.transform(varied_inputs)
            X_var_pre = torch.tensor(X_var_pre, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                y_nn = np.exp(model(X_var_pre))
            base_row = data.iloc[[idx]].copy()
            mm_input = pd.DataFrame(np.tile(base_row.values, (n_pu_points, 1)), columns=base_row.columns)
            mm_input["P_u"] = pu_vals
            y_mm = mm_predictions(mm_input)["sQSSA"]

            ax = axs[i]
            ax.plot(pu_vals, y_nn, label=nn_label, color=MODEL_COLOR_MAP[nn_label])
            ax.plot(pu_vals, y_mm, label=mm_label, color=MODEL_COLOR_MAP[mm_label], linestyle="--")
            ax.scatter(original_pu, y_true[idx], color=MODEL_COLOR_MAP[nn_label], marker="o", s=80, label="Groundtruth kcat_cg")
            ax.axvline(original_pu, color="gray", linestyle=":", linewidth=1.2)
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if i % grid_cols == 0:
                ax.set_ylabel("kcat_cg")
            ax.set_title(f"Sample {i+1} | Log-MAE: NN={log_mae_nn[idx]:.3f} — MM={log_mae_mm[idx]:.3f}")

        fig.suptitle(f"NN vs MM — Response Curves — {_label_for_regime(regime)}", fontsize=16, y=1.05)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        regime_plot_dir = Path(output_dir) / regime / "plots"
        regime_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(regime_plot_dir / "nn_vs_mm_response_curves_scaled.png", bbox_inches="tight")
        plt.close(fig)

# ── Main evaluation ────────────────────────────────────────────────────────────
def _evaluate_models_single_seed(
    data,
    features,
    output_dir,
    dataset_size,
    seed: int,
    *,
    persist_outputs: bool = True,
) -> Dict[str, Dict[str, Any]]:
    resolved_seed = seed_everything(resolve_seed(seed))
    seed = resolved_seed
    global CURRENT_SEED_TAG
    previous_tag = CURRENT_SEED_TAG
    CURRENT_SEED_TAG = f"[seed={resolved_seed}]"
    Path(output_dir, "shared/plots").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "shared/results/pysr").mkdir(parents=True, exist_ok=True)

    model_dict: Dict[str, Dict[str, Any]] = OrderedDict()

    for regime, filter_func in REGIMES.items():
        filtered = filter_func(data)
        symbolic_pool, nn_base = _prepare_regime_sample(filtered, dataset_size, seed, min_groups=len(REGIMES))
        if symbolic_pool.empty:
            _emit(f"[{regime}] No data after filtering; skipping.")
            continue

        regime_dir = Path(output_dir) / regime
        (regime_dir / "processed").mkdir(parents=True, exist_ok=True)
        (regime_dir / "plots").mkdir(parents=True, exist_ok=True)

        if persist_outputs:
            symbolic_pool.to_csv(regime_dir / "processed/filtered_data.csv", index=False)

        capped_size = min(dataset_size, len(symbolic_pool)) if dataset_size else len(symbolic_pool)
        sample = (symbolic_pool.sample(n=capped_size, random_state=seed) if capped_size < len(symbolic_pool) else symbolic_pool.copy()).reset_index(drop=True)
        if persist_outputs:
            sample.to_csv(regime_dir / "processed/filtered_data_model_sample.csv", index=False)

        if len(sample) < 2:
            _emit(f"[{regime}] Insufficient samples (<2) for train/test split; skipping.")
            continue

        test_size = max(1, int(np.ceil(len(sample) * 0.2)))
        if len(sample) - test_size < 1:
            _emit(f"[{regime}] Insufficient train samples after split; skipping.")
            continue

        indices = np.arange(len(sample))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)

        _emit(
            format_regime_message(
                regime,
                _label_for_regime(regime),
                total_samples=len(sample),
                target_samples=capped_size,
                replace=False,
            )
        )

        regime_entry: Dict[str, Any] = {"data": sample, "train_indices": train_idx, "test_indices": test_idx}

        # Variant loop
        for variant_key, display_variant in VARIANTS.items():
            suffix = variant_suffix(variant_key)
            variant_sample = augment_for_variant(sample, variant_key)
            variant_train = variant_sample.iloc[train_idx].reset_index(drop=True)
            variant_test = variant_sample.iloc[test_idx].reset_index(drop=True)

            # Persist variant splits (unsuffixed kept for sQSSA convenience)
            if persist_outputs:
                if suffix:
                    variant_sample.to_csv(regime_dir / f"processed/filtered_data_model_sample{suffix}.csv", index=False)
                    variant_train.to_csv(regime_dir / f"processed/filtered_data_train{suffix}.csv", index=False)
                    variant_test.to_csv(regime_dir / f"processed/filtered_data_test{suffix}.csv", index=False)
                else:
                    variant_train.to_csv(regime_dir / "processed/filtered_data_train.csv", index=False)
                    variant_test.to_csv(regime_dir / "processed/filtered_data_test.csv", index=False)

            X_train = variant_train.iloc[:, :-1].values
            y_train = variant_train.iloc[:, -1].values
            X_test  = variant_test.iloc[:, :-1].values
            y_test  = variant_test.iloc[:, -1].values
            X_full  = variant_sample.iloc[:, :-1].values
            y_full  = variant_sample.iloc[:, -1].values

            _emit(format_variant_split(regime, display_variant, train=len(variant_train), test=len(variant_test)))

            # PySR
            pysr_dir = regime_dir / f"models/pysr{suffix}"
            pysr_dir.mkdir(parents=True, exist_ok=True)
            config_override = {"maxsize": 35} if variant_key == "tQSSA" else None
            pysr_model = run_pysr(
                X_train,
                y_train,
                str(pysr_dir / "hall_of_fame.pkl"),
                log_prefix=f"[{regime}] [{display_variant}]",
                config_override=config_override,
            )
            y_pred_pysr = np.asarray(pysr_model.predict(X_test), dtype=float)
            pysr_metrics = metric_summary(y_pred_pysr, y_test)
            _emit(format_model_metrics(regime, display_variant, model_display_name("pysr", variant_key), pysr_metrics))

            # Michaelis–Menten
            try:
                mm_variant_pred = np.asarray(mm_predictions(variant_test)[display_variant], dtype=float)
            except Exception as exc:
                _emit(f"[WARN] Failed to compute MM {display_variant}: {exc}")
                mm_variant_pred = np.zeros_like(y_test)
            mm_metrics = metric_summary(mm_variant_pred, y_test)
            _emit(format_model_metrics(regime, display_variant, model_display_name("mm", variant_key), mm_metrics))

            # Neural Network (cap + exclude overlap with SR sample)
            variant_nn_base = augment_for_variant(nn_base, variant_key)
            nn_pool = _build_nn_pool(variant_nn_base, NN_DATASET_CAP, seed, exclude=variant_sample)
            nn_features = nn_pool.iloc[:, :-1]
            nn_target = nn_pool.iloc[:, -1]
            if nn_features.empty or nn_target.empty:
                _emit(f"[{regime}] [{display_variant}] NN: insufficient pool")
                continue

            nn_X_train, nn_X_val, nn_y_train, nn_y_val = train_test_split(
                nn_features, nn_target, test_size=0.2, random_state=seed, shuffle=True
            )
            _emit(format_variant_split(regime, f"{display_variant} NN pool", train=len(nn_X_train), test=len(nn_X_val)))

            nn_X_train_pre, nn_pipeline = preprocess_data(nn_X_train.values)
            nn_X_val_pre = nn_pipeline.transform(nn_X_val.values)
            nn_train_pre = pd.DataFrame(
                np.column_stack([nn_X_train_pre, np.log(nn_y_train.values)]),
                columns=list(nn_features.columns) + [TARGET_COLUMN],
            )
            nn_val_pre = pd.DataFrame(
                np.column_stack([nn_X_val_pre, np.log(nn_y_val.values)]),
                columns=list(nn_features.columns) + [TARGET_COLUMN],
            )

            train_pre = pd.DataFrame(
                np.column_stack([nn_pipeline.transform(variant_train.iloc[:, :-1].values), np.log(y_train)]),
                columns=list(variant_train.columns),
            )
            test_pre = pd.DataFrame(
                np.column_stack([nn_pipeline.transform(variant_test.iloc[:, :-1].values), np.log(y_test)]),
                columns=list(variant_test.columns),
            )
            full_pre = pd.DataFrame(
                np.column_stack([nn_pipeline.transform(X_full), np.log(y_full)]),
                columns=list(variant_sample.columns),
            )

            nn_dir = regime_dir / f"models/nn{suffix}"
            nn_dir.mkdir(parents=True, exist_ok=True)
            nn_path = nn_dir / "model.pkl"
            retrain_flag = not nn_path.exists()
            nn_model_ref = train_model(nn_train_pre, nn_val_pre, str(nn_path), verbose=False, retrain=retrain_flag, seed=seed)

            _, nn_pred_tensor = evaluate_model(nn_model_ref, test_pre)
            nn_preds = nn_pred_tensor.detach().cpu().numpy().flatten()
            nn_metrics = metric_summary(nn_preds, y_test)
            _emit(format_model_metrics(regime, display_variant, model_display_name("nn", variant_key), nn_metrics))

            # Store artifacts
            regime_entry[f"data{suffix}"] = variant_sample
            regime_entry[f"train_data{suffix}"] = variant_train
            regime_entry[f"test_data{suffix}"] = variant_test
            regime_entry[f"pysr{suffix}"] = pysr_model
            regime_entry[f"nn{suffix}"] = nn_model_ref
            regime_entry[f"mm{suffix}"] = mm_variant_pred
            regime_entry[f"preprocessed{suffix}"] = full_pre.reset_index(drop=True)
            regime_entry[f"test_preprocessed{suffix}"] = test_pre.reset_index(drop=True)
            if suffix == "":
                regime_entry["pysr"] = pysr_model
                regime_entry["nn"] = nn_model_ref
                regime_entry["mm"] = mm_variant_pred

        model_dict[regime] = regime_entry
        _emit("")

    if not model_dict:
        _emit("No regimes produced results; skipping plotting.")
        CURRENT_SEED_TAG = previous_tag
        return OrderedDict()

    CURRENT_SEED_TAG = previous_tag
    return model_dict


def evaluate_models(
    data,
    features,
    output_dir,
    dataset_size,
    seed: int,
    *,
    num_seeds: int = 3,
) -> None:
    seeds = [seed + offset for offset in range(max(1, num_seeds))]
    seed_model_dicts: List[Dict[str, Dict[str, Any]]] = []

    for idx, run_seed in enumerate(seeds):
        persist = idx == 0
        run_output_dir = (
            output_dir
            if persist
            else str(Path(output_dir) / "_multi_seed" / f"seed_{idx + 1}")
        )
        _emit(
            f"[multi-seed] run {idx + 1}/{len(seeds)} | output_dir={run_output_dir}"
        )
        model_dict = _evaluate_models_single_seed(
            data,
            features,
            run_output_dir,
            dataset_size,
            run_seed,
            persist_outputs=persist,
        )
        if model_dict:
            seed_model_dicts.append(model_dict)

    if not seed_model_dicts:
        _emit("No regimes produced results; skipping plotting.")
        return

    aggregated_errors, seed_medians_map = aggregate_seed_error_statistics(
        seed_model_dicts,
        evaluate_model,
    )

    base_model_dict = seed_model_dicts[0]
    for regime, models in base_model_dict.items():
        models["aggregated_errors"] = aggregated_errors.get(regime, {})
        models["aggregated_seed_medians"] = seed_medians_map.get(regime, {})
        models["seed_values"] = seeds

    plot_model_subregimes(base_model_dict, output_dir)
    plot_error_distributions(base_model_dict, output_dir)
    plot_input_error_correlation(base_model_dict, output_dir)
    plot_model_error_correlation(base_model_dict, output_dir)
    plot_nn_vs_mm_response_curves_linear(base_model_dict, output_dir)

    error_distributions, seed_medians = _collect_error_distributions(base_model_dict)
    plot_horizontal_boxplots(base_model_dict, output_dir)
    plot_vertical_boxplots(base_model_dict, output_dir)
    plot_deviation_lineplots(error_distributions, seed_medians, output_dir, template=False)
    plot_deviation_lineplots(error_distributions, seed_medians, output_dir, template=True)
    save_pysr_formulas(base_model_dict, features, output_dir)
    _emit(
        f"Kinetic ratio regimes completed across {len(seeds)} seed runs. Outputs written to {output_dir}"
    )

# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate kinetic ratio regimes with PySR/MM/NN (sQSSA/tQSSA-aware), unified logging & plots.")
    parser.add_argument('--dataset', required=True, help='CSV dataset path')
    parser.add_argument('--dataset_size', type=int, help='Max samples to use')
    parser.add_argument('--features', type=str, help='Comma-separated list of features or \"all\"')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed for reproducibility')
    parser.add_argument('--num_seeds', type=int, default=3, dest='num_seeds', help='Number of random seeds to evaluate per model (>=1)')
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset(args.dataset, args.dataset_size, args.features)

    dataset_path = Path(args.dataset)
    try:
        base_dir = dataset_path.parents[1]
    except IndexError:
        base_dir = dataset_path.parent
    output_dir = base_dir / "kinetic_regimes"

    evaluate_models(
        data,
        args.features,
        output_dir=str(output_dir),
        dataset_size=args.dataset_size,
        seed=seed,
        num_seeds=max(1, args.num_seeds),
    )

if __name__ == "__main__":
    main()
