# timepoint_regimes.py
"""
Benchmark PySR, Michaelis–Menten, and NN models across timepoint groups,
supporting sQSSA and tQSSA variants in the same way as mm_deviation and the
updated noise_regimes script.

Usage:
    python timepoint_regimes.py \
        --dataset <path_to_csv> \
        --dataset_size <max_rows_per_time_group> \
        --features <comma_separated_feature_list|all> \
        --time_column time \
        --seed 42
"""
import os
import re
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional, List

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

# mm + variants
from mm_models import (
    EPS as MM_EPS,
    TARGET_COLUMN as MM_TARGET_COLUMN,
    mm_predictions,
    sqssa_only,
)

from regime_variants import (
    MODEL_COLOR_MAP,
    MODEL_COMBINATIONS,
    MODEL_FAMILIES,
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

warnings.filterwarnings("ignore")
apply_cell_systems_style()

# ── Constants ──────────────────────────────────────────────────────────────────
TIME_COLUMN = "time"
TARGET_COLUMN = MM_TARGET_COLUMN
EPS = MM_EPS
LOG_SCALE_FLOOR = 1e-3
NN_DATASET_CAP = 20000
MAX_TIME_GROUPS = 4

# ── Logging (aligned with dataset_size_regimes) ────────────────────────────────
LOGGER = regime_logger("timepoint_regimes")
REGIME_LABEL_CACHE: Dict[str, str] = {}
CURRENT_SEED_TAG: str = ""


def _emit(message: str) -> None:
    prefix = CURRENT_SEED_TAG
    if prefix:
        first_line = message.lstrip()
        if not first_line.startswith(prefix):
            message = f"{prefix} {message}"
    LOGGER.info(message)
    print(message)

# ── Helpers: formatting & grouping ─────────────────────────────────────────────
def _format_number(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "?"
    return f"{value:.3g}"

def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug.lower() or "time_group"

def _label_for_regime(regime: str) -> str:
    return REGIME_LABEL_CACHE.get(regime, regime.replace("_", " ").title())

def _assign_time_groups(
    data: pd.DataFrame,
    time_column: str,
    max_groups: int = MAX_TIME_GROUPS,
) -> Tuple[pd.DataFrame, list[dict]]:
    series = pd.to_numeric(data[time_column], errors="coerce")
    valid_mask = series.notna()
    if not valid_mask.any():
        raise ValueError("No valid time values available for grouping.")

    working = data.loc[valid_mask].copy()
    unique_count = series[valid_mask].nunique()
    n_groups = min(max_groups, unique_count)

    if n_groups <= 1:
        working["time_group"] = 0
        times = series[valid_mask]
        low_val = float(times.min())
        high_val = float(times.max())
        label = f"Group 1 ({_format_number(low_val)}–{_format_number(high_val)})"
        info = [{
            "id": 0,
            "label": label,
            "dirname": f"group_1_{_slugify(f'{_format_number(low_val)}_{_format_number(high_val)}')}",
            "center": float(times.median()),
        }]
        return working, info

    codes, _bins = pd.qcut(
        series[valid_mask],
        q=n_groups,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    codes = codes.astype(int)
    working["time_group"] = codes.reindex(working.index, fill_value=np.nan).astype(int)

    group_infos = []
    n_assigned_groups = int(codes.max()) + 1
    for idx in range(n_assigned_groups):
        mask = working["time_group"] == idx
        if not mask.any():
            continue
        times = series[valid_mask][mask]
        low_val = float(times.min())
        high_val = float(times.max())
        center_val = float(times.median())
        label = f"Group {idx + 1} ({_format_number(low_val)}–{_format_number(high_val)})"
        dirname = f"group_{idx + 1}_{_slugify(f'{_format_number(low_val)}_{_format_number(high_val)}')}"
        group_infos.append({
            "id": idx,
            "label": label,
            "dirname": dirname,
            "center": center_val,
        })

    return working, group_infos

# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset_with_time(path: str, features: Optional[str], time_column: str = TIME_COLUMN) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if time_column not in raw.columns:
        raise KeyError(f"Column '{time_column}' not found in dataset")

    if features and features != "all":
        requested = [col.strip() for col in features.split(",") if col.strip()]
    else:
        requested = [col for col in raw.columns if col != time_column]

    if TARGET_COLUMN not in requested:
        requested.append(TARGET_COLUMN)

    feature_cols = [col for col in requested if col not in (TARGET_COLUMN, time_column)]
    ordered_cols = [time_column] + feature_cols + [TARGET_COLUMN]

    data = raw[ordered_cols].copy()
    # Values are expected in log-space; map back to linear (match other scripts)
    for col in ordered_cols:
        if col == time_column:
            continue
        data[col] = np.exp(pd.to_numeric(data[col], errors="coerce"))

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data

# ── Preprocess ─────────────────────────────────────────────────────────────────
def preprocess_data(X: np.ndarray):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler()),
    ])
    return pipeline.fit_transform(X), pipeline

# ── PySR runner ────────────────────────────────────────────────────────────────
def run_pysr(
    X,
    y,
    model_path,
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

# ── NN pool (aligned with dataset_size_regimes) ────────────────────────────────
def _build_nn_pool(
    base: pd.DataFrame,
    cap: int,
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
    target = cap if cap and cap > 0 else len(pool)
    replace = len(pool) < target
    return pool.sample(n=target, replace=replace, random_state=seed + 1).reset_index(drop=True)

# ── Error distributions (metric-agnostic, matching dataset_size_regimes) ──────
def _ordered_regimes(model_dict: Dict[str, Dict[str, Any]]):
    # Preserve insertion order of the OrderedDict that we build in evaluate_models
    return list(model_dict.keys())

def _collect_error_distributions(
    model_dict: Dict[str, Dict[str, Any]]
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, float],
    Dict[str, Dict[str, Dict[str, List[float]]]],
]:
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
    centers: Dict[str, float] = OrderedDict()
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]] = OrderedDict()

    for regime in _ordered_regimes(model_dict):
        models = model_dict.get(regime)
        if not models:
            continue

        aggregated_errors = models.get("aggregated_errors", {})
        aggregated_seed_medians = models.get("aggregated_seed_medians", {})
        metric_maps = {metric_key: {} for metric_key in metric_keys()}
        median_maps = {metric_key: {} for metric_key in metric_keys()}

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
            centers[regime] = models.get("center")
            seed_medians[regime] = median_maps

    return distributions, centers, seed_medians

def _metric_slice(
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    metric_key: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    metric_dist: Dict[str, Dict[str, np.ndarray]] = OrderedDict()
    for regime, metric_map in distributions.items():
        model_map = metric_map.get(metric_key, {})
        if model_map:
            metric_dist[regime] = model_map
    return metric_dist

def _errors_to_dataframe(error_dict, metric_key):
    records = []
    for model_name, errors in error_dict.items():
        arr = np.asarray(errors, dtype=float)
        arr = arr[np.isfinite(arr)]
        records.extend({"Model": model_name, metric_key: value} for value in arr)
    return pd.DataFrame(records)

def _combined_error_frame(metric_distributions: Dict[str, Dict[str, np.ndarray]], metric_key: str) -> pd.DataFrame:
    frames = []
    for label, dist in metric_distributions.items():
        frame = _errors_to_dataframe(dist, metric_key)
        if not frame.empty:
            frame["Time"] = label
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["Model", metric_key, "Time"])

def _compute_line_stats(error_distributions):
    stats = {model: {"median": [], "q1": [], "q3": []} for model in MODEL_LINE_ORDER}
    for _, regime_errors in error_distributions.items():
        for model in MODEL_LINE_ORDER:
            arr = np.asarray(regime_errors.get(model, []), dtype=float)
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
    ymin = min(lowers)
    ymax = max(uppers)
    if np.isclose(ymin, ymax):
        margin = 0.1 * (abs(ymax) if ymax != 0 else 1.0)
        ymin -= margin
        ymax += margin
    else:
        margin = 0.05 * (ymax - ymin)
        ymin -= margin
        ymax += margin
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

# ── Plots: identical look/feel to dataset_size_regimes ─────────────────────────
def _plot_timepoint_lineplot_for_metric(
    metric_key: str,
    metric_distributions: Dict[str, Dict[str, np.ndarray]],
    centers: Dict[str, float],
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
        for key in ("median", "q1", "q3"):
            stats_array = clip_metric_values(
                np.asarray(stats[model][key], dtype=float),
                metric_key,
            )
            if spec.y_scale == "log":
                floor = spec.floor or LOG_SCALE_FLOOR
                stats_array = np.where(
                    np.isfinite(stats_array) & (stats_array < floor),
                    floor,
                    stats_array,
                )
            stats[model][key] = stats_array
    ymin, ymax = _lineplot_limits(stats, spec)

    # Use equally spaced group positions on the x-axis regardless of raw time centers.
    x = np.arange(len(regimes), dtype=float)
    x_labels = [_label_for_regime(r) for r in regimes]
    x_label_text = "Time Group"

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for model in MODEL_LINE_ORDER:
        median = np.asarray(stats[model]["median"], dtype=float)
        q1 = np.asarray(stats[model]["q1"], dtype=float)
        q3 = np.asarray(stats[model]["q3"], dtype=float)
        color = MODEL_COLOR_MAP[model]
        if not template:
            ax.plot(x, median, marker="o", label=model, color=color)
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
        else:
            ax.plot(x, median, color=color, linestyle="--", alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15, ha="right")
    ax.set_xlabel(x_label_text)
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

    suffix = "_template" if template else ""
    plt.tight_layout()
    filename = f"{spec.filename_prefix}_timepoint_lineplot{suffix}.png"
    plt.savefig(plot_dir / filename, dpi=300)
    plt.close()

def plot_timepoint_lineplots(
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    centers: Dict[str, float],
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
        _plot_timepoint_lineplot_for_metric(
            mkey,
            metric_dist,
            centers,
            metric_seed_medians,
            output_dir,
            template=template,
        )

def _plot_error_distributions(metric_key, metric_distributions, output_dir):
    df = _combined_error_frame(metric_distributions, metric_key)
    if df.empty:
        LOGGER.info("No %s errors for violin/box plots", metric_key)
        return

    regimes = list(metric_distributions.keys())
    ncols = min(4, len(regimes))
    nrows = int(np.ceil(len(regimes) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]
    spec = metric_spec(metric_key)
    metric_dir = Path(output_dir) / "shared/plots" / spec.folder
    metric_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df[metric_key] = clip_metric_values(df[metric_key].to_numpy(), metric_key)
    lower_limit, upper_limit = metric_limits_with_margin(metric_key)

    # Violin plots per time group
    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        subset = df[df["Time"] == regime]
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
        ax.set_title(regime)
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
    for idx in range(len(regimes), len(axes)):
        plt.delaxes(axes[idx])

    plt.suptitle(f"{spec.label} Error Distributions by Time Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(metric_dir / f"{spec.filename_prefix}_error_distributions.png")
    plt.close()

def plot_error_distributions(model_dict, output_dir):
    distributions, _, _ = _collect_error_distributions(model_dict)
    if not distributions:
        LOGGER.info("No error distributions available; skipping violin plots")
        return
    for mkey in metric_keys():
        metric_map = _metric_slice(distributions, mkey)
        if not metric_map:
            continue
        _plot_error_distributions(mkey, metric_map, output_dir)

def plot_horizontal_boxplots(model_dict, output_dir):
    distributions, _, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return
    regimes = list(distributions.keys())
    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]

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

        value_col = mkey
        df = df.copy()
        df[value_col] = clip_metric_values(df[value_col].to_numpy(), mkey)
        lower_limit, upper_limit = metric_limits_with_margin(mkey)
        floor = spec.floor or LOG_SCALE_FLOOR if spec.y_scale == "log" else None

        def _render(showfliers: bool, suffix: str):
            fig, axes = plt.subplots(len(regimes), 1, figsize=(10, 2.8 * len(regimes)), sharex=True)
            axes = np.atleast_1d(axes)
            for ax, regime in zip(axes, regimes):
                subset = df[df["Time"] == regime]
                if subset.empty:
                    ax.set_visible(False)
                    continue
                sns.boxplot(
                    data=subset,
                    x=value_col,
                    y="Model",
                    palette=palette,
                    orient="h",
                    order=MODEL_LINE_ORDER,
                    ax=ax,
                    showfliers=showfliers,
                )
                ax.set_title(regime, fontsize=14, weight="bold")
                ax.set_ylabel("")
                if spec.y_scale == "log":
                    ax.set_xscale("log")
                    if lower_limit is not None and upper_limit is not None:
                        ax.set_xlim(lower_limit, upper_limit)
                elif lower_limit is not None and upper_limit is not None:
                    ax.set_xlim(lower_limit, upper_limit)
            axes[-1].set_xlabel(spec.axis_label, fontsize=12)
            for ax in axes[:-1]:
                ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{spec.filename_prefix}_horizontal_boxplot{suffix}.png", dpi=300)
            plt.close()

        _render(True, "")
        _render(False, "_no_outliers")

def plot_vertical_boxplots(model_dict, output_dir):
    distributions, _, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return
    regimes = list(distributions.keys())
    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]

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

        value_col = mkey
        df = df.copy()
        df[value_col] = clip_metric_values(df[value_col].to_numpy(), mkey)
        lower_limit, upper_limit = metric_limits_with_margin(mkey)
        finite_vals = df[value_col].to_numpy()
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size and (lower_limit is None or upper_limit is None):
            global_min = finite_vals.min()
            global_max = finite_vals.max()
            if np.isclose(global_min, global_max):
                pad = max(global_min * 0.1, 1e-6)
                global_min -= pad
                global_max += pad
        else:
            global_min = global_max = None

        def _render(showfliers: bool, suffix: str):
            fig, axes = plt.subplots(1, len(regimes), figsize=(3.2 * len(regimes), 4.5), sharey=False)
            axes = np.atleast_1d(axes)
            for ax, regime in zip(axes, regimes):
                subset = df[df["Time"] == regime]
                if subset.empty:
                    ax.set_visible(False)
                    continue
                sns.boxplot(
                    data=subset,
                    x="Model",
                    y=value_col,
                    palette=palette,
                    orient="v",
                    order=MODEL_LINE_ORDER,
                    ax=ax,
                    showfliers=showfliers,
                )
                ax.set_title(regime, fontsize=14, weight="bold")
                if spec.y_scale == "log":
                    ax.set_yscale("log")
                    if lower_limit is not None and upper_limit is not None:
                        ax.set_ylim(lower_limit, upper_limit)
                for label in ax.get_xticklabels():
                    label.set_rotation(25)
                    label.set_horizontalalignment("right")
                if ax != axes[0]:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(spec.axis_label, fontsize=12)
                if lower_limit is not None and upper_limit is not None and spec.y_scale != "log":
                    ax.set_ylim(lower_limit, upper_limit)
                elif global_min is not None and global_max is not None:
                    ax.set_ylim(global_min, global_max)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{spec.filename_prefix}_vertical_boxplot{suffix}.png", dpi=300)
            plt.close()

        _render(True, "")
        _render(False, "_no_outliers")

# ── Extra plots (kept, but now metric-agnostic plumbing coexists) ─────────────
def plot_model_subregimes(model_dict, output_dir):
    regimes = list(model_dict.keys())
    n_rows, n_cols = len(regimes), len(MODEL_COMBINATIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.6 * n_rows), sharey=True)

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
        idx = snapshot_any['features'].index
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
                ax.set_axis_off()
                continue

            predictions = snapshot['predictions']
            data_subset = snapshot['data'].loc[chosen]
            positions = data_subset.index.map(snapshot['data'].index.get_loc).to_numpy()
            y_true = snapshot['y_true'][positions]
            pred = predictions.get(display_name)
            if pred is None:
                ax.set_axis_off()
                continue
            pred = pred[positions]

            errors = np.abs(
                np.log(np.maximum(pred, EPS)) - np.log(np.maximum(y_true, EPS))
            )
            ax.scatter(
                y_true,
                errors,
                alpha=0.4,
                s=18,
                color=MODEL_COLOR_MAP[display_name],
                label=display_name,
                edgecolors='k',
                linewidths=0.2,
            )

            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel('Log-space MAE |ln(ŷ+ε) − ln(y+ε)| (ε=1e−20)', fontsize=10)
            if row_idx == 0:
                ax.set_title(display_name, fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc='upper left')
            ax.set_xscale('log')

    fig.suptitle("Model Error Landscape by Time Group and Model", fontsize=16, y=1.02)
    plt.tight_layout()
    out = Path(output_dir) / "shared/plots/error_landscape.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
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
        feature_cols = [col for col in base_df.columns if col != TARGET_COLUMN]
        for col in feature_cols:
            error_df[col] = base_df[col].values

        corr = error_df.corr()[list(error_map.keys())].drop(error_map.keys(), axis=0)
        all_error_df.append(error_df)

        ax = axes[idx]
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")
        filled = idx

    for j in range(filled + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature–Error Correlations by Time Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(output_dir) / "shared/plots/feature_error_correlation_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()

    if all_error_df:
        full_df = pd.concat(all_error_df, ignore_index=True)
        error_columns = [col for col in full_df.columns if col in MODEL_COLOR_MAP]
        overall_corr = full_df.corr()[error_columns].drop(error_columns, axis=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
        plt.title("Overall Feature–Error Correlation (All Time Groups)")
        plt.tight_layout()
        out = Path(output_dir) / "shared/plots/feature_error_correlation_overall.png"
        plt.savefig(out, dpi=300)
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
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f"{regime}")
        filled = idx

    for j in range(filled + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model–Model Error Correlations by Time Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(output_dir) / "shared/plots/model_error_correlation_grid.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()

    if all_model_error_df:
        df_all = pd.concat(all_model_error_df, ignore_index=True)
        overall_corr = df_all.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(overall_corr, annot=True, cmap='coolwarm', center=0)
        plt.title("Overall Model–Model Error Correlation (All Time Groups)")
        plt.tight_layout()
        out = Path(output_dir) / "shared/plots/model_error_correlation_overall.png"
        plt.savefig(out, dpi=300)
        plt.close()

def save_pysr_formulas(model_dict, features, output_dir):
    out_dir = Path(output_dir) / "shared/results/pysr"
    out_dir.mkdir(parents=True, exist_ok=True)

    variant_lines = collect_pysr_formula_lines(model_dict, features)
    combined: list[str] = []
    for variant_key, lines in variant_lines.items():
        if not lines:
            continue
        combined.extend(lines)
        variant_path = out_dir / f"all_pysr_formulas_{variant_key}.txt"
        with variant_path.open("w") as handle:
            for line in lines:
                handle.write(line + "\n")

    if combined:
        with (out_dir / "all_pysr_formulas.txt").open("w") as handle:
            for line in combined:
                handle.write(line + "\n")

# ── Evaluation ─────────────────────────────────────────────────────────────────
def _evaluate_models_single_seed(
    data: pd.DataFrame,
    output_dir: str,
    dataset_size: int,
    seed: int,
    time_column: str = TIME_COLUMN,
    *,
    persist_outputs: bool = True,
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    resolved_seed = seed_everything(resolve_seed(seed))
    seed = resolved_seed
    global CURRENT_SEED_TAG
    previous_tag = CURRENT_SEED_TAG
    CURRENT_SEED_TAG = f"[seed={resolved_seed}]"
    REGIME_LABEL_CACHE.clear()

    # Optional overall cap (balance across groups)
    working_data = data
    if dataset_size:
        total_target = dataset_size * MAX_TIME_GROUPS
        if len(data) > total_target:
            working_data = data.sample(n=total_target, random_state=seed).reset_index(drop=True)
        else:
            working_data = data.reset_index(drop=True)

    grouped_data, group_infos = _assign_time_groups(working_data, time_column)
    if not group_infos:
        LOGGER.warning("No time groups could be formed; skipping evaluation.")
        CURRENT_SEED_TAG = previous_tag
        return OrderedDict()

    model_dict: Dict[str, Dict[str, Any]] = OrderedDict()

    for info in group_infos:
        label = info["label"]
        dirname = info["dirname"]
        group_rows = grouped_data[grouped_data["time_group"] == info["id"]].reset_index(drop=True)
        if len(group_rows) < 2:
            LOGGER.warning("Skipping %s: need at least 2 samples", label)
            continue

        group_dir = Path(output_dir) / dirname
        (group_dir / "processed").mkdir(parents=True, exist_ok=True)
        (group_dir / "plots").mkdir(parents=True, exist_ok=True)
        (group_dir / "models/pysr").mkdir(parents=True, exist_ok=True)
        (group_dir / "models/nn").mkdir(parents=True, exist_ok=True)

        # Drop grouping columns, keep features+target in linear space
        base_no_time = group_rows.drop(columns=["time_group", time_column]).reset_index(drop=True)

        # Cap per-group sample for the variant-aware symbolic modeling
        capped_size = min(dataset_size, len(base_no_time)) if dataset_size else len(base_no_time)
        capped_size = max(2, capped_size)
        sample = (
            base_no_time.sample(n=capped_size, random_state=seed)
            if capped_size < len(base_no_time)
            else base_no_time.copy()
        ).reset_index(drop=True)

        if persist_outputs:
            sample.to_csv(group_dir / "processed/model_sample.csv", index=False)

        _emit(
            format_regime_message(
                dirname,
                label,
                total_samples=len(sample),
                target_samples=capped_size,
                replace=False,
            )
        )

        # Shared train/test indices
        indices = np.arange(len(sample))
        test_size = max(1, int(np.ceil(len(sample) * 0.2)))
        if len(sample) - test_size < 1:
            LOGGER.warning("Skipping %s: insufficient train samples after split", label)
            _emit(f"[{dirname}] {label}: insufficient samples for train/test split")
            continue
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)

        # NN pool baseline (pre-variant)
        nn_base = base_no_time.copy()

        _emit(f"[{dirname}] Center time ≈ {_format_number(info.get('center'))} | Sample={len(sample)}")

        # Registry for this time-group
        regime_entry: Dict[str, Any] = {
            "data": sample,
            "center": info.get("center"),
            "label": label,
        }
        REGIME_LABEL_CACHE[dirname] = label

        # Variant-aware loop
        for variant_key, display_variant in VARIANTS.items():
            suffix = variant_suffix(variant_key)

            # Variant augmentation (adds/remaps columns per variant policy)
            variant_sample = augment_for_variant(sample, variant_key)
            variant_train = variant_sample.iloc[train_idx].reset_index(drop=True)
            variant_test = variant_sample.iloc[test_idx].reset_index(drop=True)

            # Persist splits
            if persist_outputs:
                if suffix:
                    variant_sample.to_csv(group_dir / f"processed/model_sample{suffix}.csv", index=False)
                    variant_train.to_csv(group_dir / f"processed/train{suffix}.csv", index=False)
                    variant_test.to_csv(group_dir / f"processed/test{suffix}.csv", index=False)
                else:
                    variant_train.to_csv(group_dir / "processed/train.csv", index=False)
                    variant_test.to_csv(group_dir / "processed/test.csv", index=False)

            X_train = variant_train.iloc[:, :-1].values
            y_train = variant_train.iloc[:, -1].values
            X_test = variant_test.iloc[:, :-1].values
            y_test = variant_test.iloc[:, -1].values
            X_full = variant_sample.iloc[:, :-1].values
            y_full = variant_sample.iloc[:, -1].values

            _emit(
                format_variant_split(
                    dirname,
                    display_variant,
                    train=len(variant_train),
                    test=len(variant_test),
                )
            )

            # ── PySR (variant-aware)
            pysr_dir = group_dir / f"models/pysr{suffix}"
            pysr_dir.mkdir(parents=True, exist_ok=True)
            pysr_path = pysr_dir / "hall_of_fame.pkl"
            config_override = {"maxsize": 35} if variant_key == "tQSSA" else None
            pysr_model = run_pysr(
                X_train,
                y_train,
                str(pysr_path),
                log_prefix=f"[{dirname}] [{display_variant}]",
                config_override=config_override,
            )
            y_pred_pysr = np.asarray(pysr_model.predict(X_test), dtype=float)
            pysr_metrics = metric_summary(y_pred_pysr, y_test)
            _emit(
                format_model_metrics(
                    dirname,
                    display_variant,
                    model_display_name("pysr", variant_key),
                    pysr_metrics,
                )
            )

            # ── MM (variant-aware)
            try:
                mm_variant_pred = np.asarray(mm_predictions(variant_test)[display_variant], dtype=float)
            except Exception as exc:
                _emit(f"[WARN] Failed to compute MM {display_variant}: {exc}")
                mm_variant_pred = np.zeros_like(y_test)
            mm_metrics = metric_summary(mm_variant_pred, y_test)
            _emit(
                format_model_metrics(
                    dirname,
                    display_variant,
                    model_display_name("mm", variant_key),
                    mm_metrics,
                )
            )

            # ── NN (variant-aware)
            variant_nn_base = augment_for_variant(nn_base, variant_key)
            nn_pool = _build_nn_pool(variant_nn_base, NN_DATASET_CAP, seed, exclude=variant_sample)
            nn_features = nn_pool.iloc[:, :-1]
            nn_target = nn_pool.iloc[:, -1]
            if nn_features.empty or nn_target.empty:
                _emit(f"[{dirname}] [{display_variant}] NN: insufficient pool")
                continue

            nn_X_train, nn_X_val, nn_y_train, nn_y_val = train_test_split(
                nn_features, nn_target, test_size=0.2, random_state=seed, shuffle=True
            )

            _emit(
                format_variant_split(
                    dirname,
                    f"{display_variant} NN pool",
                    train=len(nn_X_train),
                    val=len(nn_X_val),
                )
            )

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

            nn_dir = group_dir / f"models/nn{suffix}"
            nn_dir.mkdir(parents=True, exist_ok=True)
            nn_path = nn_dir / "model.pkl"
            retrain_flag = not nn_path.exists()
            nn_model_ref = train_model(nn_train_pre, nn_val_pre, str(nn_path), verbose=False, retrain=retrain_flag, seed=seed)

            _, nn_pred_tensor = evaluate_model(nn_model_ref, test_pre)
            nn_preds = nn_pred_tensor.detach().cpu().numpy().flatten()
            nn_metrics = metric_summary(nn_preds, y_test)
            _emit(
                format_model_metrics(
                    dirname,
                    display_variant,
                    model_display_name("nn", variant_key),
                    nn_metrics,
                )
            )

            # ── Store per-variant artifacts
            regime_entry[f'data{suffix}'] = variant_sample
            regime_entry[f'train_data{suffix}'] = variant_train
            regime_entry[f'test_data{suffix}'] = variant_test
            regime_entry[f'pysr{suffix}'] = pysr_model
            regime_entry[f'nn{suffix}'] = nn_model_ref
            regime_entry[f'mm{suffix}'] = mm_variant_pred
            regime_entry[f'preprocessed{suffix}'] = full_pre.reset_index(drop=True)
            regime_entry[f'test_preprocessed{suffix}'] = test_pre.reset_index(drop=True)

            if suffix == "":
                regime_entry['pysr'] = pysr_model
                regime_entry['nn'] = nn_model_ref
                regime_entry['mm'] = mm_variant_pred

        model_dict[label] = regime_entry
        _emit("")

    if not model_dict:
        LOGGER.warning("No valid timegroups processed; skipping plots")
        CURRENT_SEED_TAG = previous_tag
        return OrderedDict()

    CURRENT_SEED_TAG = previous_tag
    return model_dict


def evaluate_models(
    data: pd.DataFrame,
    output_dir: str,
    dataset_size: int,
    seed: int,
    time_column: str = TIME_COLUMN,
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
            run_output_dir,
            dataset_size,
            run_seed,
            time_column=time_column,
            persist_outputs=persist,
        )
        if model_dict:
            seed_model_dicts.append(model_dict)

    if not seed_model_dicts:
        LOGGER.warning("No valid timegroups processed; skipping plots")
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

    error_distributions, centers, seed_medians = _collect_error_distributions(base_model_dict)

    plot_model_subregimes(base_model_dict, output_dir)
    plot_error_distributions(base_model_dict, output_dir)
    plot_input_error_correlation(base_model_dict, output_dir)
    plot_model_error_correlation(base_model_dict, output_dir)

    plot_horizontal_boxplots(base_model_dict, output_dir)
    plot_vertical_boxplots(base_model_dict, output_dir)
    plot_timepoint_lineplots(error_distributions, centers, seed_medians, output_dir, template=False)
    plot_timepoint_lineplots(error_distributions, centers, seed_medians, output_dir, template=True)

    save_pysr_formulas(base_model_dict, features=None, output_dir=output_dir)
    _emit(
        f"Timepoint regimes completed across {len(seeds)} seed runs. Outputs written to {output_dir}"
    )

# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark models by timepoint using PySR, MM, and NN (sQSSA/tQSSA-aware)."
    )
    parser.add_argument("--dataset", required=True, help="Path to merged dynamic dataset (CSV)")
    parser.add_argument("--dataset_size", type=int, help="Maximum samples per timepoint group")
    parser.add_argument("--features", type=str, help="Comma-separated list of features or 'all'")
    parser.add_argument("--time_column", type=str, default=TIME_COLUMN, help="Column representing time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_seeds", type=int, default=3, dest="num_seeds", help="Number of random seeds to evaluate per model (>=1)")
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset_with_time(args.dataset, args.features, args.time_column)

    dataset_path = Path(args.dataset)
    try:
        base_dir = dataset_path.parents[1]
    except IndexError:
        base_dir = dataset_path.parent
    output_dir = base_dir / "timepoint_regimes"

    evaluate_models(
        data,
        output_dir=str(output_dir),
        dataset_size=args.dataset_size or len(data),
        seed=seed,
        time_column=args.time_column,
        num_seeds=max(1, args.num_seeds),
    )

if __name__ == "__main__":
    main()
