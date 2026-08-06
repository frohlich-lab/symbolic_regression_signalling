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
import json
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional, List

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
PROJECT_ROOT = SRC_ROOT.parent
os.environ.setdefault(
    "PYTHON_JULIAPKG_PROJECT",
    str((PROJECT_ROOT / ".snakemake" / "juliapkg" / Path(__file__).stem).resolve()),
)
os.environ.setdefault(
    "JULIA_DEPOT_PATH",
    str((PROJECT_ROOT / ".snakemake" / "julia_depot").resolve()),
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import train_model, evaluate_model, load_model_checkpoint
from constants import PYSR_CONFIG, pysr_operator_config
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
    canonicalize_variant_keys,
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
    baseline_label,
    baseline_labels,
    variant_plot_context,
    variant_suffix,
    has_persisted_splits,
    regime_logger,
    aggregate_seed_error_statistics,
    clip_metric_values,
    metric_limits_with_margin,
    selected_variants_context,
    write_pysr_formula_files,
)

warnings.filterwarnings("ignore")
apply_cell_systems_style()

# ── Constants ──────────────────────────────────────────────────────────────────
TIME_COLUMN = "time"
TARGET_COLUMN = MM_TARGET_COLUMN
EPS = MM_EPS
LOG_SCALE_FLOOR = 1e-3
LOG_EPS = 1e-20


def _warn_on_bad_values(values: np.ndarray, context: str) -> None:
    arr = np.asarray(values, dtype=float)
    nonfinite = np.count_nonzero(~np.isfinite(arr))
    nonpositive = np.count_nonzero(arr <= 0)
    if nonfinite or nonpositive:
        _emit(
            f"[warn] {context}: {nonfinite} non-finite, {nonpositive} non-positive values; "
            f"clipping to {LOG_EPS} before log transform."
        )


def _safe_log_values(values: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    _warn_on_bad_values(arr, context)
    safe = np.nan_to_num(arr, nan=LOG_EPS, posinf=LOG_EPS, neginf=LOG_EPS)
    return np.log(np.clip(safe, LOG_EPS, None))
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


def _dedupe_baseline_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        base_label = baseline_label(label)
        if base_label in seen:
            continue
        seen.add(base_label)
        unique_handles.append(handle)
        unique_labels.append(base_label)
    ax.legend(unique_handles, unique_labels, frameon=False)

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
    X = np.asarray(X, dtype=float)
    _warn_on_bad_values(X, "NN feature matrix")

    def _safe_log(values: np.ndarray) -> np.ndarray:
        safe = np.nan_to_num(values, nan=LOG_EPS, posinf=LOG_EPS, neginf=LOG_EPS)
        return np.log(np.clip(safe, LOG_EPS, None))

    pipeline = Pipeline(
        [
            ("log_transform", FunctionTransformer(_safe_log, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(X), pipeline

# ── PySR runner ────────────────────────────────────────────────────────────────
def run_pysr(
    X,
    y,
    model_path,
    log_prefix: Optional[str] = None,
    config_override: Optional[Dict[str, object]] = None,
    *,
    plots_only: bool = False,
    seed: Optional[int] = None,
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

    if plots_only and load_target is None:
        raise FileNotFoundError(f"No existing PySR artifacts found at {run_dir} for plots-only mode")

    if load_target is not None:
        _emit(f"{prefix} Loading existing model from {load_target}")
        return PySRRegressor.from_file(run_directory=str(run_dir))

    _emit(f"{prefix} Training new model in {run_dir}")
    config = dict(PYSR_CONFIG)
    if config_override:
        config.update(config_override)
    if seed is not None:
        config["random_state"] = seed
    # Enforce deterministic PySR runs (serial execution + fixed seed).
    config["deterministic"] = True
    config["parallelism"] = "serial"
    config["procs"] = 0
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


def _cache_path(regime_dir: Path) -> Path:
    return regime_dir / "processed" / "predictions_cache.json"


def load_prediction_cache(regime_dir: Path) -> dict:
    cache_file = _cache_path(regime_dir)
    if not cache_file.exists():
        return {}
    try:
        with cache_file.open("r") as handle:
            raw = json.load(handle)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_prediction_cache(regime_dir: Path, cache: dict) -> None:
    cache_file = _cache_path(regime_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w") as handle:
        json.dump(cache, handle)

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
    Dict[str, Dict[str, Dict[str, List[float]]]],
]:
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
    centers: Dict[str, float] = OrderedDict()
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]] = OrderedDict()
    seed_means: Dict[str, Dict[str, Dict[str, List[float]]]] = OrderedDict()

    for regime in _ordered_regimes(model_dict):
        models = model_dict.get(regime)
        if not models:
            continue

        aggregated_errors = models.get("aggregated_errors", {})
        aggregated_seed_medians = models.get("aggregated_seed_medians", {})
        aggregated_seed_means = models.get("aggregated_seed_means", {})
        metric_maps = {metric_key: {} for metric_key in metric_keys()}
        median_maps = {metric_key: {} for metric_key in metric_keys()}
        mean_maps = {metric_key: {} for metric_key in metric_keys()}

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
                    means = (
                        aggregated_seed_means
                        .get(variant_key, {})
                        .get(mkey, {})
                        .get(display_name)
                    )
                    if medians is not None:
                        median_maps[mkey][display_name] = list(medians)
                    if means is not None:
                        mean_maps[mkey][display_name] = list(means)
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
                mean_value = float(np.mean(finite)) if finite.size else float("nan")
                median_value = float(np.median(finite)) if finite.size else float("nan")
                mean_maps[mkey][display_name] = [mean_value]
                median_maps[mkey][display_name] = [median_value]

        if any(metric_maps[m] for m in metric_keys()):
            distributions[regime] = metric_maps
            centers[regime] = models.get("center")
            seed_medians[regime] = median_maps
            seed_means[regime] = mean_maps

    return distributions, centers, seed_medians, seed_means

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
        floor = spec.floor if spec.floor is not None else LOG_SCALE_FLOOR
        if floor is not None and ymin < floor:
            ymin = floor
        if spec.ceiling is not None:
            ymax = min(ymax, spec.ceiling * 1.1)
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
    seed_stats = {model: {"median": [], "q1": [], "q3": [], "min": [], "max": []} for model in MODEL_LINE_ORDER}
    for regime in regimes:
        for model in MODEL_LINE_ORDER:
            seed_values = np.asarray(seed_medians.get(regime, {}).get(model, []), dtype=float)
            seed_values = seed_values[np.isfinite(seed_values)]
            if seed_values.size:
                seed_values = clip_metric_values(seed_values, metric_key)
                seed_values = seed_values[np.isfinite(seed_values)]
            if seed_values.size == 0:
                seed_stats[model]["median"].append(np.nan)
                seed_stats[model]["q1"].append(np.nan)
                seed_stats[model]["q3"].append(np.nan)
                seed_stats[model]["min"].append(np.nan)
                seed_stats[model]["max"].append(np.nan)
            else:
                seed_stats[model]["median"].append(float(np.median(seed_values)))
                seed_stats[model]["q1"].append(float(np.quantile(seed_values, 0.25)))
                seed_stats[model]["q3"].append(float(np.quantile(seed_values, 0.75)))
                seed_stats[model]["min"].append(float(np.min(seed_values)))
                seed_stats[model]["max"].append(float(np.max(seed_values)))

    stats = {model: {"median": [], "q1": [], "q3": []} for model in MODEL_LINE_ORDER}
    for model in MODEL_LINE_ORDER:
        for key in ("median", "q1", "q3"):
            stats_array = clip_metric_values(
                np.asarray(seed_stats[model][key], dtype=float),
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

    lower_limit, upper_limit = metric_limits_with_margin(metric_key)

    def _render_lineplot(*, fixed_ylim: bool) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 6.0))

        for model in MODEL_LINE_ORDER:
            median = np.asarray(stats[model]["median"], dtype=float)
            min_vals = np.asarray(seed_stats[model]["min"], dtype=float)
            max_vals = np.asarray(seed_stats[model]["max"], dtype=float)
            color = MODEL_COLOR_MAP[model]
            is_orange = model.startswith("PySR")
            line_z = 4 if is_orange else 3
            band_z = 2 if is_orange else 1
            if not template:
                ax.plot(
                    x,
                    median,
                    marker="o",
                    markersize=12,
                    linewidth=3.0,
                    label=baseline_label(model),
                    color=color,
                    zorder=line_z,
                )
                if np.isfinite(min_vals).any() and np.isfinite(max_vals).any():
                    ax.fill_between(x, min_vals, max_vals, color=color, alpha=0.15, zorder=band_z)
            else:
                ax.plot(x, median, color=color, linestyle="--", alpha=0.3, linewidth=3.0, zorder=line_z)

        label_fs = 22
        tick_fs = 19
        legend_fs = 21
        legend_title_fs = 22
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha="right", fontsize=tick_fs)
        ax.set_xlabel(x_label_text, fontsize=label_fs)
        ax.set_ylabel(spec.axis_label, fontsize=label_fs)
        ax.set_yscale(spec.y_scale)
        if fixed_ylim and spec.y_scale == "log":
            lower, upper = 1e-2, 1e2
            pad = 2.0
            ax.set_ylim(lower / pad, upper * pad)
        elif ymin is not None and ymax is not None:
            if spec.y_scale == "log":
                ax.set_ylim(ymin, ymax * 1.15)
            else:
                ax.set_ylim(ymin, ymax * 1.05)
        elif lower_limit is not None and upper_limit is not None:
            ax.set_ylim(lower_limit, upper_limit)
        if not template and not fixed_ylim:
            _dedupe_baseline_legend(ax)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(legend.get_title().get_text(), prop={"size": legend_title_fs})
                for text in legend.get_texts():
                    text.set_fontsize(legend_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        suffix = "_template" if template else ""
        fixed_suffix = "_fixed_ylim" if fixed_ylim else ""
        plt.tight_layout()
        filename = f"{spec.filename_prefix}_timepoint_lineplot{fixed_suffix}{suffix}.png"
        plt.savefig(plot_dir / filename, dpi=300)
        plt.close()

    _render_lineplot(fixed_ylim=False)
    _render_lineplot(fixed_ylim=True)

def plot_timepoint_lineplots(
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    centers: Dict[str, float],
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]],
    seed_means: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_dir: str,
    template: bool = False,
):
    for mkey in metric_keys():
        metric_dist = _metric_slice(distributions, mkey)
        if not metric_dist:
            continue
        metric_seed_values = {
            regime: seed_medians.get(regime, {}).get(mkey, {})
            for regime in metric_dist.keys()
        }
        _plot_timepoint_lineplot_for_metric(
            mkey,
            metric_dist,
            centers,
            metric_seed_values,
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
        ax.set_xticklabels(baseline_labels(MODEL_LINE_ORDER))
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

    plt.suptitle(f"{spec.label} Error Distributions by Time Group", fontsize=32)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(metric_dir / f"{spec.filename_prefix}_error_distributions.png")
    plt.close()

def plot_error_distributions(model_dict, output_dir):
    distributions, _, _, _ = _collect_error_distributions(model_dict)
    if not distributions:
        LOGGER.info("No error distributions available; skipping violin plots")
        return
    for mkey in metric_keys():
        metric_map = _metric_slice(distributions, mkey)
        if not metric_map:
            continue
        _plot_error_distributions(mkey, metric_map, output_dir)

def plot_horizontal_boxplots(model_dict, output_dir):
    distributions, _, _, _ = _collect_error_distributions(model_dict)
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
                ax.set_yticklabels(baseline_labels(MODEL_LINE_ORDER))
                ax.set_title(regime, fontsize=28, weight="bold")
                ax.set_ylabel("")
                if spec.y_scale == "log":
                    ax.set_xscale("log")
                    if lower_limit is not None and upper_limit is not None:
                        ax.set_xlim(lower_limit, upper_limit)
                elif lower_limit is not None and upper_limit is not None:
                    ax.set_xlim(lower_limit, upper_limit)
            axes[-1].set_xlabel(spec.axis_label, fontsize=24)
            for ax in axes[:-1]:
                ax.set_xlabel("")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{spec.filename_prefix}_horizontal_boxplot{suffix}.png", dpi=300)
            plt.close()

        _render(True, "")
        _render(False, "_no_outliers")

def plot_vertical_boxplots(model_dict, output_dir):
    distributions, _, _, _ = _collect_error_distributions(model_dict)
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
                ax.set_xticklabels(baseline_labels(MODEL_LINE_ORDER))
                ax.set_title(regime, fontsize=28, weight="bold")
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
                    ax.set_ylabel(spec.axis_label, fontsize=24)
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
            display_label = baseline_label(display_name)
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
                label=display_label,
                edgecolors='k',
                linewidths=0.2,
            )

            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=20)
            if col_idx == 0:
                ax.set_ylabel('Log-space MAE |ln(ŷ+ε) − ln(y+ε)| (ε=1e−20)', fontsize=20)
            if row_idx == 0:
                ax.set_title(display_label, fontsize=22)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc='upper left')
            ax.set_xscale('log')

    fig.suptitle("Model Error Landscape by Time Group and Model", fontsize=32, y=1.02)
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

    plt.suptitle("Feature–Error Correlations by Time Group", fontsize=32)
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

    plt.suptitle("Model–Model Error Correlations by Time Group", fontsize=32)
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
    variant_lines = collect_pysr_formula_lines(model_dict, features)
    write_pysr_formula_files(output_dir, variant_lines)

# ── Evaluation ─────────────────────────────────────────────────────────────────
def _evaluate_models_single_seed(
    data: pd.DataFrame,
    output_dir: str,
    dataset_size: int,
    seed: int,
    time_column: str = TIME_COLUMN,
    *,
    persist_outputs: bool = True,
    plots_only: bool = False,
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
        variant_cache = load_prediction_cache(group_dir)

        # Drop grouping columns, keep features+target in linear space
        base_no_time = group_rows.drop(columns=["time_group", time_column]).reset_index(drop=True)

        # Cap per-group sample for the variant-aware symbolic modeling
        if plots_only:
            sample_path = group_dir / "processed/model_sample.csv"
            train_path = group_dir / "processed/train.csv"
            test_path = group_dir / "processed/test.csv"
            if not sample_path.exists() or not train_path.exists() or not test_path.exists():
                LOGGER.warning("Missing persisted splits for plots-only mode in %s", group_dir)
                continue
            sample = pd.read_csv(sample_path)
            base_train = pd.read_csv(train_path)
            base_test = pd.read_csv(test_path)
        else:
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
        if plots_only:
            train_idx = test_idx = None
        else:
            indices = np.arange(len(sample))
            test_size = max(1, int(np.ceil(len(sample) * 0.2)))
            if len(sample) - test_size < 1:
                LOGGER.warning("Skipping %s: insufficient train samples after split", label)
                _emit(f"[{dirname}] {label}: insufficient samples for train/test split")
                continue
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
            base_train = sample.iloc[train_idx].reset_index(drop=True)
            base_test = sample.iloc[test_idx].reset_index(drop=True)
            if persist_outputs:
                base_train.to_csv(group_dir / "processed/train.csv", index=False)
                base_test.to_csv(group_dir / "processed/test.csv", index=False)

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
            cached_variant = variant_cache.get(variant_key, {})

            # Variant augmentation (adds/remaps columns per variant policy)
            if plots_only:
                sample_path = group_dir / f"processed/model_sample{suffix}.csv" if suffix else group_dir / "processed/model_sample.csv"
                train_path = group_dir / f"processed/train{suffix}.csv" if suffix else group_dir / "processed/train.csv"
                test_path = group_dir / f"processed/test{suffix}.csv" if suffix else group_dir / "processed/test.csv"
                if not sample_path.exists() or not train_path.exists() or not test_path.exists():
                    LOGGER.warning("Missing variant splits for plots-only mode: %s", sample_path)
                    continue
                variant_sample = pd.read_csv(sample_path)
                variant_train = pd.read_csv(train_path)
                variant_test = pd.read_csv(test_path)
            else:
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
            config_override = dict(pysr_operator_config(variant_key))
            if not config_override:
                config_override = None
            if plots_only:
                cached_preds = cached_variant.get("predictions", {})
                cached_y_true = cached_variant.get("y_true")
                if cached_preds and cached_y_true is not None:
                    pysr_model = None
                    y_pred_pysr = np.asarray(
                        cached_preds.get(model_display_name("pysr", variant_key), []),
                        dtype=float,
                    )
                    y_test = np.asarray(cached_y_true, dtype=float)
                else:
                    _emit(
                        f"[{dirname}] [{display_variant}] Cached predictions missing; loading PySR model to rebuild cache."
                    )
                    pysr_model = run_pysr(
                        X_train,
                        y_train,
                        str(pysr_path),
                        log_prefix=f"[{dirname}] [{display_variant}]",
                        config_override=config_override,
                        plots_only=True,
                        seed=seed,
                    )
                    y_pred_pysr = np.asarray(pysr_model.predict(X_test), dtype=float)
                    cached_variant["y_true"] = y_test.tolist()
                    preds = cached_variant.get("predictions", {})
                    preds[model_display_name("pysr", variant_key)] = y_pred_pysr.tolist()
                    cached_variant["predictions"] = preds
            else:
                pysr_model = run_pysr(
                    X_train,
                    y_train,
                    str(pysr_path),
                    log_prefix=f"[{dirname}] [{display_variant}]",
                    config_override=config_override,
                    plots_only=plots_only,
                    seed=seed,
                )
                y_pred_pysr = np.asarray(pysr_model.predict(X_test), dtype=float)
                cached_variant["y_true"] = y_test.tolist()
                preds = cached_variant.get("predictions", {})
                preds[model_display_name("pysr", variant_key)] = y_pred_pysr.tolist()
                cached_variant["predictions"] = preds
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
            mm_variant_pred = None
            if plots_only:
                cached_mm = cached_variant.get("predictions", {}).get(model_display_name("mm", variant_key))
                if cached_mm is not None:
                    mm_variant_pred = np.asarray(cached_mm, dtype=float)
            if mm_variant_pred is None:
                try:
                    mm_variant_pred = np.asarray(mm_predictions(variant_test)[display_variant], dtype=float)
                except Exception as exc:
                    _emit(f"[WARN] Failed to compute MM {display_variant}: {exc}")
                    mm_variant_pred = np.zeros_like(y_test)
                preds = cached_variant.get("predictions", {})
                preds[model_display_name("mm", variant_key)] = mm_variant_pred.tolist()
                cached_variant["predictions"] = preds
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
                np.column_stack([nn_X_train_pre, _safe_log_values(nn_y_train.values, "NN train targets")]),
                columns=list(nn_features.columns) + [TARGET_COLUMN],
            )
            nn_val_pre = pd.DataFrame(
                np.column_stack([nn_X_val_pre, _safe_log_values(nn_y_val.values, "NN val targets")]),
                columns=list(nn_features.columns) + [TARGET_COLUMN],
            )

            train_pre = pd.DataFrame(
                np.column_stack(
                    [
                        nn_pipeline.transform(variant_train.iloc[:, :-1].values),
                        _safe_log_values(y_train, "Variant train targets"),
                    ]
                ),
                columns=list(variant_train.columns),
            )
            test_pre = pd.DataFrame(
                np.column_stack(
                    [
                        nn_pipeline.transform(variant_test.iloc[:, :-1].values),
                        _safe_log_values(y_test, "Variant test targets"),
                    ]
                ),
                columns=list(variant_test.columns),
            )
            full_pre = pd.DataFrame(
                np.column_stack([nn_pipeline.transform(X_full), _safe_log_values(y_full, "Full targets")]),
                columns=list(variant_sample.columns),
            )

            nn_dir = group_dir / f"models/nn{suffix}"
            nn_dir.mkdir(parents=True, exist_ok=True)
            nn_path = nn_dir / "model.pkl"
            if plots_only:
                cached_nn = cached_variant.get("predictions", {}).get(model_display_name("nn", variant_key))
                if cached_nn is not None:
                    nn_model_ref = None
                    if nn_path.exists():
                        nn_model_ref = load_model_checkpoint(
                            str(nn_path),
                            input_dim=nn_train_pre.shape[1] - 1,
                        )
                    nn_preds = np.asarray(cached_nn, dtype=float)
                else:
                    if not nn_path.exists():
                        raise FileNotFoundError(
                            f"Missing NN checkpoint for plots-only mode: {nn_path}"
                        )
                    _emit(
                        f"[{dirname}] [{display_variant}] Cached predictions missing; loading NN checkpoint to rebuild cache."
                    )
                    nn_model_ref = load_model_checkpoint(
                        str(nn_path),
                        input_dim=nn_train_pre.shape[1] - 1,
                    )
                    _, nn_pred_tensor = evaluate_model(nn_model_ref, test_pre)
                    nn_preds = nn_pred_tensor.detach().cpu().numpy().flatten()
                    preds = cached_variant.get("predictions", {})
                    preds[model_display_name("nn", variant_key)] = nn_preds.tolist()
                    cached_variant["predictions"] = preds
                    cached_variant["y_true"] = y_test.tolist()
            else:
                retrain_flag = not plots_only or not nn_path.exists()
                nn_model_ref = train_model(nn_train_pre, nn_val_pre, str(nn_path), verbose=False, retrain=retrain_flag, seed=seed)

                _, nn_pred_tensor = evaluate_model(nn_model_ref, test_pre)
                nn_preds = nn_pred_tensor.detach().cpu().numpy().flatten()
                preds = cached_variant.get("predictions", {})
                preds[model_display_name("nn", variant_key)] = nn_preds.tolist()
                cached_variant["predictions"] = preds
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
            regime_entry[f'cached_predictions{suffix}'] = {
                k: np.asarray(v, dtype=float) for k, v in cached_variant.get("predictions", {}).items()
            }
            regime_entry[f'cached_y_true{suffix}'] = np.asarray(
                cached_variant.get("y_true", y_test.tolist()), dtype=float
            )

            if suffix == "":
                regime_entry['pysr'] = pysr_model
                regime_entry['nn'] = nn_model_ref
                regime_entry['mm'] = mm_variant_pred
                regime_entry['cached_predictions'] = regime_entry[f'cached_predictions{suffix}']
                regime_entry['cached_y_true'] = regime_entry[f'cached_y_true{suffix}']

        model_dict[label] = regime_entry
        _emit("")
        if persist_outputs:
            save_prediction_cache(group_dir, variant_cache)

    if not model_dict:
        LOGGER.warning("No valid timegroups processed; skipping plots")
        CURRENT_SEED_TAG = previous_tag
        return OrderedDict()

    CURRENT_SEED_TAG = previous_tag
    return model_dict


def _render_variant_plots(model_dict: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Write per-variant plot bundles under output_dir/variants/<variant>."""
    for variant_key in VARIANTS.keys():
        variant_dir = Path(output_dir) / "variants" / variant_key.lower()
        with variant_plot_context(variant_key):
            plot_model_subregimes(model_dict, variant_dir)
            plot_error_distributions(model_dict, variant_dir)
            plot_input_error_correlation(model_dict, variant_dir)
            plot_model_error_correlation(model_dict, variant_dir)
            plot_horizontal_boxplots(model_dict, variant_dir)
            plot_vertical_boxplots(model_dict, variant_dir)
            variant_distributions, centers, seed_medians, seed_means = _collect_error_distributions(model_dict)
            plot_timepoint_lineplots(variant_distributions, centers, seed_medians, seed_means, variant_dir, template=False)
            plot_timepoint_lineplots(variant_distributions, centers, seed_medians, seed_means, variant_dir, template=True)
        _emit(f"[plots] Saved {variant_key} variant outputs to {variant_dir}")


def evaluate_models(
    data: pd.DataFrame,
    output_dir: str,
    dataset_size: int,
    seed: int,
    time_column: str = TIME_COLUMN,
    *,
    num_seeds: int = 3,
    plots_only: bool = False,
    render_plots: bool = True,
) -> None:
    seeds = [seed + offset for offset in range(max(1, num_seeds))]
    seed_model_dicts: List[Dict[str, Dict[str, Any]]] = []

    for idx, run_seed in enumerate(seeds):
        # Always persist per-seed splits/predictions so plots-only can reuse them.
        persist = True
        run_output_dir = (
            output_dir
            if num_seeds <= 1
            else str(Path(output_dir) / "_multi_seed" / f"seed_{idx + 1}")
        )
        if plots_only and not has_persisted_splits(Path(run_output_dir)):
            _emit(
                f"[multi-seed] skipping run {idx + 1}/{len(seeds)} | output_dir={run_output_dir} (missing persisted splits)"
            )
            continue
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
            plots_only=plots_only,
        )
        if model_dict:
            seed_model_dicts.append(model_dict)

    if not seed_model_dicts:
        LOGGER.warning("No valid timegroups processed; skipping plots")
        return

    base_model_dict = seed_model_dicts[0]
    # Save per-seed PySR formulas under output_dir/_multi_seed/seed_{i}
    for idx, model_dict in enumerate(seed_model_dicts):
        seed_dir = Path(output_dir) / "_multi_seed" / f"seed_{idx + 1}"
        save_pysr_formulas(model_dict, features=None, output_dir=seed_dir)
    if render_plots:
        aggregated_errors, seed_medians_map, seed_means_map = aggregate_seed_error_statistics(
            seed_model_dicts,
            evaluate_model,
        )

        for regime, models in base_model_dict.items():
            models["aggregated_errors"] = aggregated_errors.get(regime, {})
            models["aggregated_seed_medians"] = seed_medians_map.get(regime, {})
            models["aggregated_seed_means"] = seed_means_map.get(regime, {})
            models["seed_values"] = seeds

        error_distributions, centers, seed_medians, seed_means = _collect_error_distributions(base_model_dict)

        plot_model_subregimes(base_model_dict, output_dir)
        plot_error_distributions(base_model_dict, output_dir)
        plot_input_error_correlation(base_model_dict, output_dir)
        plot_model_error_correlation(base_model_dict, output_dir)

        plot_horizontal_boxplots(base_model_dict, output_dir)
        plot_vertical_boxplots(base_model_dict, output_dir)
        plot_timepoint_lineplots(error_distributions, centers, seed_medians, seed_means, output_dir, template=False)
        plot_timepoint_lineplots(error_distributions, centers, seed_medians, seed_means, output_dir, template=True)

        _render_variant_plots(base_model_dict, output_dir)
    # Keep existing shared output at output_dir for downstream rules
    save_pysr_formulas(base_model_dict, features=None, output_dir=output_dir)
    message = f"Timepoint regimes completed across {len(seeds)} seed runs. Outputs written to {output_dir}"
    if not render_plots:
        message = f"{message} (plots disabled)"
    _emit(message)

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
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()), help="Subset of variants to run, e.g. tQSSA or sQSSA tQSSA")
    parser.add_argument("--output-dir", type=str, help="Override output root directory")
    parser.add_argument("--plots-only", action="store_true", help="Do not train; load existing models and render plots only")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation during evaluation")
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset_with_time(args.dataset, args.features, args.time_column)
    selected_variants = canonicalize_variant_keys(args.variants)

    dataset_path = Path(args.dataset)
    try:
        base_dir = dataset_path.parents[1]
    except IndexError:
        base_dir = dataset_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "timepoint_regimes"

    with selected_variants_context(selected_variants):
        evaluate_models(
            data,
            output_dir=str(output_dir),
            dataset_size=args.dataset_size or len(data),
            seed=seed,
            time_column=args.time_column,
            num_seeds=max(1, args.num_seeds),
            plots_only=args.plots_only,
            render_plots=not args.no_plots,
        )

if __name__ == "__main__":
    main()
