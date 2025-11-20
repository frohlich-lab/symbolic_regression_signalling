"""Evaluate model performance across dataset-size regimes (sQSSA/tQSSA variants)."""

import os
import argparse
import warnings
import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import train_model, evaluate_model
from constants import PYSR_CONFIG
from plot_style import apply_cell_systems_style
from utils.seeding import resolve_seed, seed_everything

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
    mm_variant_prediction,
    variant_suffix,
    regime_logger,
    aggregate_seed_error_statistics,
    clip_metric_values,
    metric_limits_with_margin,
)

warnings.filterwarnings("ignore")
apply_cell_systems_style()

LOGGER = regime_logger("dataset_size_regimes")
CURRENT_SEED_TAG: str = ""


def _emit(message: str) -> None:
    prefix = CURRENT_SEED_TAG
    if prefix:
        first_line = message.lstrip()
        if not first_line.startswith(prefix):
            message = f"{prefix} {message}"
    LOGGER.info(message)
    print(message)

SIZE_GROUPS = OrderedDict(
    {
        "size_10pct": {"label": "Dataset Size 10%", "multiplier": 0.10},
        "size_50pct": {"label": "Dataset Size 50%", "multiplier": 0.50},
        "size_100pct": {"label": "Dataset Size 100%", "multiplier": 1.0},
        "size_200pct": {"label": "Dataset Size 200%", "multiplier": 2.0},
    }
)

REGIME_LABELS = {key: meta["label"] for key, meta in SIZE_GROUPS.items()}
REGIME_ORDER = {key: idx for idx, key in enumerate(SIZE_GROUPS.keys())}

TARGET_COLUMN = "kcat_cg"
LOG_SCALE_FLOOR = 1e-3


def _ordered_regimes(model_dict: Dict[str, Dict[str, object]]):
    return sorted(model_dict.keys(), key=lambda r: (REGIME_ORDER.get(r, float("inf")), r))


def _label_for_regime(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def load_dataset(path: str, features: Optional[str] = None) -> pd.DataFrame:
    data = pd.read_csv(path)
    if features and features != "all":
        cols = [col.strip() for col in features.split(",") if col.strip()]
        data = data[cols]
    return data.map(np.exp)


def run_pysr(X, y, model_path, log_prefix: Optional[str] = None, config_override: Optional[Dict[str, object]] = None):
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


def preprocess_data(X: np.ndarray):
    pipeline = Pipeline(
        [
            ("log_transform", FunctionTransformer(np.log, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(X), pipeline


def _build_nn_pool(base: pd.DataFrame, cap: int, seed: int, exclude: pd.DataFrame = None) -> pd.DataFrame:
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
                how="left",
            )
            pool = merged[merged["_mark"].isna()].drop(columns=["_mark"]).reset_index(drop=True)
            if pool.empty:
                pool = base.copy().reset_index(drop=True)
    target = cap if cap and cap > 0 else len(pool)
    replace = len(pool) < target
    return pool.sample(n=target, replace=replace, random_state=seed + 1).reset_index(drop=True)


def _collect_error_distributions(
    model_dict: Dict[str, Dict[str, object]]
) -> tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, List[float]]]],
]:
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
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
                for metric_key in metric_keys():
                    model_errors = variant_errors.get(metric_key, {})
                    errs = model_errors.get(display_name)
                    if errs is None:
                        continue
                    metric_maps[metric_key][display_name] = np.asarray(errs, dtype=float)
                    medians = (
                        aggregated_seed_medians
                        .get(variant_key, {})
                        .get(metric_key, {})
                        .get(display_name)
                    )
                    if medians is not None:
                        median_maps[metric_key][display_name] = list(medians)
                continue

            snapshot = collect_variant_predictions(models, variant_key, evaluate_model)
            if snapshot is None:
                continue
            preds = snapshot["predictions"].get(display_name)
            if preds is None:
                continue
            y_true = snapshot["y_true"]
            for metric_key in metric_keys():
                errs = metric_errors(metric_key, preds, y_true)
                metric_maps[metric_key][display_name] = errs
                finite = errs[np.isfinite(errs)]
                median_value = float(np.median(finite)) if finite.size else float("nan")
                median_maps[metric_key][display_name] = [median_value]

        if any(metric_maps[m] for m in metric_keys()):
            distributions[regime] = metric_maps
            seed_medians[regime] = median_maps

    return distributions, seed_medians


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
            frame["Regime"] = label
            frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["Model", metric_key, "Regime"])


def _compute_line_stats(error_distributions):
    stats = {model: {"median": [], "q1": [], "q3": []} for model in MODEL_LINE_ORDER}
    for regime in _ordered_regimes(error_distributions):
        regime_errors = error_distributions[regime]
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


def _apply_log_floor(stats):
    for model in MODEL_LINE_ORDER:
        for key in ("median", "q1", "q3"):
            stats[model][key] = [
                val if (not np.isfinite(val) or val >= LOG_SCALE_FLOOR) else LOG_SCALE_FLOOR
                for val in stats[model][key]
            ]


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


def _plot_dataset_size_lineplot(
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

    regimes = _ordered_regimes(metric_distributions)
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

    x = np.arange(len(regimes), dtype=float)
    x_labels = []
    for regime in regimes:
        mult = SIZE_GROUPS.get(regime, {}).get("multiplier")
        if mult is None:
            x_labels.append(regime)
        else:
            x_labels.append(f"{mult * 100:.0f}%")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

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
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Dataset Size Regime")
    ax.set_ylabel(spec.axis_label)
    ax.set_yscale(spec.y_scale)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    if not template:
        ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    suffix = "_template" if template else ""
    plt.tight_layout()
    filename = f"{spec.filename_prefix}_dataset_size_lineplot{suffix}.png"
    plt.savefig(plot_dir / filename, dpi=300)
    plt.close()


def plot_dataset_size_lineplots(
    distributions: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    seed_medians: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_dir: str,
    template: bool = False,
):
    for metric_key in metric_keys():
        metric_dist = _metric_slice(distributions, metric_key)
        if not metric_dist:
            continue
        metric_seed_medians = {
            regime: seed_medians.get(regime, {}).get(metric_key, {})
            for regime in metric_dist.keys()
        }
        _plot_dataset_size_lineplot(
            metric_key,
            metric_dist,
            metric_seed_medians,
            output_dir,
            template=template,
        )


def _plot_error_distributions(metric_key, metric_distributions, output_dir):
    df = _combined_error_frame(metric_distributions, metric_key)
    if df.empty:
        LOGGER.info("No %s errors for violin plots", metric_key)
        return

    regimes = _ordered_regimes(metric_distributions)
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

    for idx, regime in enumerate(regimes):
        ax = axes[idx]
        subset = df[df["Regime"] == regime]
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

    for idx in range(len(regimes), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"{spec.label} Error Distributions by Dataset Size", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(metric_dir / f"{spec.filename_prefix}_error_distributions.png")
    plt.close()


def plot_error_distributions(model_dict, output_dir):
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        LOGGER.info("No error distributions available; skipping violin plots")
        return

    for metric_key in metric_keys():
        metric_map = _metric_slice(distributions, metric_key)
        if not metric_map:
            continue
        _plot_error_distributions(metric_key, metric_map, output_dir)


def plot_model_subregimes(model_dict, output_dir):
    regimes = _ordered_regimes(model_dict)
    n_rows, n_cols = len(regimes), len(MODEL_COMBINATIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharey=True)

    rng = np.random.default_rng(42)
    sampled = {}
    for regime_name in regimes:
        for variant_key in VARIANTS.keys():
            snapshot = collect_variant_predictions(model_dict.get(regime_name, {}), variant_key, evaluate_model)
            if snapshot is None:
                continue
            idx = snapshot["features"].index
            sample_size = min(len(idx), 600)
            chosen = idx if len(idx) <= sample_size else rng.choice(idx, size=sample_size, replace=False)
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

            predictions = snapshot["predictions"]
            data_subset = snapshot["data"].loc[chosen]
            positions = data_subset.index.map(snapshot["data"].index.get_loc).to_numpy()
            y_true = snapshot["y_true"][positions]

            pred = predictions.get(display_name)
            if pred is None:
                ax.set_axis_off()
                continue
            pred = pred[positions]

            errors = np.abs(
                np.log(np.maximum(pred, 1e-20)) - np.log(np.maximum(y_true, 1e-20))
            )

            ax.scatter(
                y_true,
                errors,
                alpha=0.4,
                s=18,
                color=MODEL_COLOR_MAP[display_name],
                label=display_name,
                edgecolors="k",
                linewidths=0.2,
            )

            if row_idx == len(regimes) - 1:
                ax.set_xlabel("Groundtruth kcat_cg", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)| (ε=1e−20)", fontsize=10)
            if row_idx == 0:
                ax.set_title(display_name, fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, loc="upper left")

            ax.set_xscale("log")

        fig.text(
            0.02,
            (n_rows - row_idx - 0.5) / n_rows,
            _label_for_regime(regime_name),
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )

    fig.suptitle("Model Error Landscape by Dataset Size", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    shared_plot_dir = Path(output_dir) / "shared/plots"
    shared_plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(shared_plot_dir / "error_landscape.png")
    plt.close()


def plot_input_error_correlation(model_dict, output_dir):
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

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
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(_label_for_regime(regime))
        filled = idx

    for j in range(filled + 1, len(axes)):
        fig.delaxes(axes[j])

    if not all_error_df:
        plt.close(fig)
        return

    plt.suptitle("Feature-Error Correlations by Dataset Size", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    shared_plot_dir = Path(output_dir) / "shared/plots"
    shared_plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(shared_plot_dir / "feature_error_correlation_grid.png")
    plt.close()

    full_df = pd.concat(all_error_df, ignore_index=True)
    error_columns = [col for col in full_df.columns if col in MODEL_COLOR_MAP]
    overall_corr = full_df.corr()[error_columns].drop(error_columns, axis=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(overall_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Overall Feature-Error Correlation (All Dataset Sizes)")
    plt.tight_layout()
    plt.savefig(shared_plot_dir / "feature_error_correlation_overall.png")
    plt.close()


def plot_model_error_correlation(model_dict, output_dir):
    n = len(model_dict)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

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

    if not all_model_error_df:
        plt.close(fig)
        return

    plt.suptitle("Model Error Correlations by Dataset Size", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    shared_plot_dir = Path(output_dir) / "shared/plots"
    shared_plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(shared_plot_dir / "model_error_correlation_grid.png")
    plt.close()

    df_all = pd.concat(all_model_error_df, ignore_index=True)
    overall_corr = df_all.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(overall_corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Overall Model Error Correlation (All Dataset Sizes)")
    plt.tight_layout()
    plt.savefig(shared_plot_dir / "model_error_correlation_overall.png")
    plt.close()


def plot_nn_vs_mm_response_curves_linear(model_dict, output_dir, n_samples=10, n_pu_points=200):
    for regime, models in model_dict.items():
        LOGGER.info("Generating scaled response curves for %s...", regime)
        data = models.get("test_data")
        if data is None or data.empty:
            data = models.get("data")
        preprocessed = models.get("test_preprocessed")
        if preprocessed is None or (hasattr(preprocessed, "empty") and preprocessed.empty):
            preprocessed = models.get("preprocessed")
        nn_model = models.get("nn")
        if (
            data is None
            or data.empty
            or preprocessed is None
            or nn_model is None
            or "P_u" not in data.columns
        ):
            LOGGER.warning("Skipping %s: missing evaluation artifacts", regime)
            continue

        y_true = data.iloc[:, -1].values
        X = data.iloc[:, :-1]
        pu_index = X.columns.get_loc("P_u")

        _, pipeline = preprocess_data(X.values)

        nn_preds = evaluate_model(nn_model, preprocessed)[1].detach().cpu().numpy().flatten()
        log_mae_vals = np.abs(
            np.log(np.maximum(nn_preds, 1e-20)) - np.log(np.maximum(y_true, 1e-20))
        )
        if len(log_mae_vals) == 0:
            continue

        percentiles = np.linspace(0, 100, n_samples + 2)[1:-1]
        thresholds = np.percentile(log_mae_vals, percentiles)
        chosen_indices = []
        for threshold in thresholds:
            idx = int(np.argmin(np.abs(log_mae_vals - threshold)))
            if idx not in chosen_indices:
                chosen_indices.append(idx)
        if not chosen_indices:
            chosen_indices = list(range(min(n_samples, len(log_mae_vals))))

        grid_cols = max(1, int(np.ceil(len(chosen_indices) / 2)))
        grid_rows = max(1, int(np.ceil(len(chosen_indices) / grid_cols)))
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(10 * grid_cols, 5 * grid_rows))
        axs = np.atleast_1d(axs).flatten()

        nn_label = model_display_name("nn", "sQSSA")
        mm_label = model_display_name("mm", "sQSSA")
        for plot_idx, sample_idx in enumerate(chosen_indices[: len(axs)]):
            fixed_sample = X.iloc[sample_idx].copy()
            original_pu = fixed_sample["P_u"]
            pu_vals = np.linspace(0.01 * original_pu, 5 * original_pu, n_pu_points)

            varied_inputs = np.tile(fixed_sample.values, (n_pu_points, 1))
            varied_inputs[:, pu_index] = pu_vals

            X_varied_inputs_pre = pipeline.transform(varied_inputs)
            X_varied_inputs_pre = torch.tensor(X_varied_inputs_pre, dtype=torch.float32)

            nn_model.eval()
            with torch.no_grad():
                y_nn = np.exp(nn_model(X_varied_inputs_pre))

            base_row = data.iloc[[sample_idx]].copy()
            mm_input = pd.DataFrame(
                np.tile(base_row.values, (n_pu_points, 1)),
                columns=base_row.columns,
            )
            mm_input["P_u"] = pu_vals
            y_mm = mm_variant_prediction(mm_input, "sQSSA")

            ax = axs[plot_idx]
            ax.plot(pu_vals, y_nn, label=nn_label, color=MODEL_COLOR_MAP[nn_label])
            ax.plot(pu_vals, y_mm, label=mm_label, color=MODEL_COLOR_MAP[mm_label], linestyle="--")
            ax.scatter(
                original_pu,
                y_true[sample_idx],
                color=MODEL_COLOR_MAP[nn_label],
                marker="o",
                s=80,
                label="Groundtruth kcat_cg",
            )
            ax.axvline(original_pu, color="gray", linestyle=":", linewidth=1.2)
            ax.set_xlim(0.01 * original_pu, 5 * original_pu)
            ax.set_xlabel("P_u")
            if plot_idx % grid_cols == 0:
                ax.set_ylabel("Output")
            ax.set_title(f"Sample {plot_idx + 1} | Log-MAE = {log_mae_vals[sample_idx]:.3f}")

        fig.suptitle(f"NN vs MM — Response Curves — {_label_for_regime(regime)}", fontsize=16, y=1.05)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        shared_plot_dir = Path(output_dir) / regime / "plots"
        shared_plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(shared_plot_dir / "nn_vs_mm_response_curves_scaled.png", bbox_inches="tight")
        plt.close(fig)


def plot_horizontal_boxplots(model_dict, output_dir):
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return

    regimes = _ordered_regimes(distributions)
    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]

    for metric_key in metric_keys():
        metric_dist = _metric_slice(distributions, metric_key)
        if not metric_dist:
            continue
        df = _combined_error_frame(metric_dist, metric_key)
        if df.empty:
            continue
        spec = metric_spec(metric_key)
        plot_dir = Path(output_dir) / "shared/plots" / spec.folder
        plot_dir.mkdir(parents=True, exist_ok=True)

        value_col = metric_key
        df = df.copy()
        df[value_col] = clip_metric_values(df[value_col].to_numpy(), metric_key)
        lower_limit, upper_limit = metric_limits_with_margin(metric_key)
        floor = spec.floor or LOG_SCALE_FLOOR if spec.y_scale == "log" else None

        def _render(showfliers: bool, suffix: str):
            fig, axes = plt.subplots(len(regimes), 1, figsize=(10, 2.8 * len(regimes)), sharex=True)
            axes = np.atleast_1d(axes)
            for ax, regime in zip(axes, regimes):
                subset = df[df["Regime"] == regime]
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
                ax.set_title(_label_for_regime(regime), fontsize=14, weight="bold")
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
    distributions, _ = _collect_error_distributions(model_dict)
    if not distributions:
        return

    regimes = _ordered_regimes(distributions)
    palette = [MODEL_COLOR_MAP[m] for m in MODEL_LINE_ORDER]

    for metric_key in metric_keys():
        metric_dist = _metric_slice(distributions, metric_key)
        if not metric_dist:
            continue
        df = _combined_error_frame(metric_dist, metric_key)
        if df.empty:
            continue
        spec = metric_spec(metric_key)
        plot_dir = Path(output_dir) / "shared/plots" / spec.folder
        plot_dir.mkdir(parents=True, exist_ok=True)

        value_col = metric_key
        df = df.copy()
        df[value_col] = clip_metric_values(df[value_col].to_numpy(), metric_key)
        lower_limit, upper_limit = metric_limits_with_margin(metric_key)
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
                subset = df[df["Regime"] == regime]
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
                ax.set_title(_label_for_regime(regime), fontsize=14, weight="bold")
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


def _evaluate_models_single_seed(
    data: pd.DataFrame,
    features: Optional[str],
    output_dir: str,
    seed: int,
    pysr_base_size: int,
    nn_base_size: int,
    *,
    persist_outputs: bool = True,
) -> Dict[str, Dict[str, object]]:
    os.makedirs(output_dir, exist_ok=True)
    resolved_seed = seed_everything(resolve_seed(seed))
    global CURRENT_SEED_TAG
    previous_tag = CURRENT_SEED_TAG
    CURRENT_SEED_TAG = f"[seed={resolved_seed}]"
    try:
        model_dict: Dict[str, Dict[str, object]] = OrderedDict()
        base_len = len(data)

        def emit_grouped(lines: List[str]) -> None:
            if not lines:
                return
            head, *tail = lines
            if tail:
                _emit(head + "\n  " + "\n  ".join(tail))
            else:
                _emit(head)

        for regime_key, meta in SIZE_GROUPS.items():
            label = meta["label"]
            multiplier = meta["multiplier"]
            desired_n = max(2, int(np.round(pysr_base_size * multiplier)))
            replace = desired_n > base_len
            sample = (
                data.sample(n=desired_n, random_state=seed, replace=replace)
                .reset_index(drop=True)
            )

            regime_dir = Path(output_dir) / regime_key
            (regime_dir / "processed").mkdir(parents=True, exist_ok=True)
            (regime_dir / "models/pysr").mkdir(parents=True, exist_ok=True)
            (regime_dir / "models/nn").mkdir(parents=True, exist_ok=True)
            (regime_dir / "plots").mkdir(parents=True, exist_ok=True)

            full_path = regime_dir / "processed/full_dataset.csv"
            if persist_outputs:
                sample.to_csv(full_path, index=False)

            _emit(
                format_regime_message(
                    regime_key,
                    label,
                    total_samples=len(sample),
                    target_samples=desired_n,
                    replace=replace,
                )
            )

            test_size = max(1, int(np.ceil(len(sample) * 0.2)))
            if len(sample) - test_size < 1:
                LOGGER.warning("Skipping %s: insufficient samples for train/test split", label)
                _emit(f"[{regime_key}] {label}: insufficient samples for train/test split")
                continue

            indices = np.arange(len(sample))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, shuffle=True)
            base_train = sample.iloc[train_idx].reset_index(drop=True)
            base_test = sample.iloc[test_idx].reset_index(drop=True)

            if persist_outputs:
                base_train.to_csv(regime_dir / "processed/train.csv", index=False)
                base_test.to_csv(regime_dir / "processed/test.csv", index=False)

            regime_entry: Dict[str, object] = {
                "data": sample,
                "train_data": base_train,
                "test_data": base_test,
            }

            for variant_key, display_variant in VARIANTS.items():
                suffix = variant_suffix(variant_key)
                variant_sample = augment_for_variant(sample, variant_key)
                variant_train = variant_sample.iloc[train_idx].reset_index(drop=True)
                variant_test = variant_sample.iloc[test_idx].reset_index(drop=True)

                if persist_outputs:
                    if suffix:
                        variant_sample.to_csv(regime_dir / f"processed/model_sample{suffix}.csv", index=False)
                        variant_train.to_csv(regime_dir / f"processed/train{suffix}.csv", index=False)
                        variant_test.to_csv(regime_dir / f"processed/test{suffix}.csv", index=False)
                    else:
                        variant_sample.to_csv(regime_dir / "processed/model_sample.csv", index=False)

                X_train = variant_train.iloc[:, :-1].values
                y_train = variant_train.iloc[:, -1].values
                X_test = variant_test.iloc[:, :-1].values
                y_test = variant_test.iloc[:, -1].values
                X_full = variant_sample.iloc[:, :-1].values
                y_full = variant_sample.iloc[:, -1].values

                variant_lines = [
                    format_variant_split(
                        regime_key,
                        display_variant,
                        train=len(variant_train),
                        test=len(variant_test),
                    )
                ]

                pysr_dir = regime_dir / f"models/pysr{suffix}"
                pysr_dir.mkdir(parents=True, exist_ok=True)
                pysr_model_path = pysr_dir / "hall_of_fame.pkl"
                config_override = {"maxsize": 35} if variant_key == "tQSSA" else None
                pysr_model = run_pysr(
                    X_train,
                    y_train,
                    str(pysr_model_path),
                    log_prefix=f"[{regime_key}] [{display_variant}]",
                    config_override=config_override,
                )
                pysr_pred = np.asarray(pysr_model.predict(X_test), dtype=float)
                pysr_metrics = metric_summary(pysr_pred, y_test)
                variant_lines.append(
                    format_model_metrics(
                        regime_key,
                        display_variant,
                        model_display_name("pysr", variant_key),
                        pysr_metrics,
                    )
                )

                mm_pred = np.asarray(mm_variant_prediction(variant_test, variant_key), dtype=float)
                mm_metrics = metric_summary(mm_pred, y_test)
                variant_lines.append(
                    format_model_metrics(
                        regime_key,
                        display_variant,
                        model_display_name("mm", variant_key),
                        mm_metrics,
                    )
                )

                variant_nn_base = augment_for_variant(data, variant_key)
                target_nn_cap = max(2, int(np.round(nn_base_size * multiplier)))
                nn_pool = _build_nn_pool(variant_nn_base, target_nn_cap, seed, exclude=variant_sample)
                nn_features = nn_pool.iloc[:, :-1]
                nn_target = nn_pool.iloc[:, -1]
                if nn_features.empty or nn_target.empty:
                    LOGGER.warning("    Skipping NN (%s): insufficient pool", display_variant)
                    variant_lines.append(f"[{regime_key}] [{display_variant}] NN: insufficient pool")
                    emit_grouped(variant_lines)
                    continue

                nn_X_train, nn_X_val, nn_y_train, nn_y_val = train_test_split(
                    nn_features,
                    nn_target,
                    test_size=0.2,
                    random_state=seed,
                    shuffle=True,
                )
                variant_lines.append(
                    format_variant_split(
                        regime_key,
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

                train_split = nn_train_pre.reset_index(drop=True)
                val_split = nn_val_pre.reset_index(drop=True)

                nn_dir = regime_dir / f"models/nn{suffix}"
                nn_dir.mkdir(parents=True, exist_ok=True)
                nn_path = nn_dir / "model.pkl"
                if len(train_split) < 2 or len(val_split) < 1:
                    LOGGER.warning("    Skipping NN (%s): insufficient NN split", display_variant)
                    variant_lines.append(f"[{regime_key}] [{display_variant}] NN: insufficient NN split")
                    emit_grouped(variant_lines)
                    continue

                nn_model = train_model(train_split, val_split, str(nn_path), verbose=False, seed=seed)
                _, nn_pred_tensor = evaluate_model(nn_model, test_pre)
                nn_preds = nn_pred_tensor.detach().cpu().numpy().flatten()
                nn_metrics = metric_summary(nn_preds, y_test)
                variant_lines.append(
                    format_model_metrics(
                        regime_key,
                        display_variant,
                        model_display_name("nn", variant_key),
                        nn_metrics,
                    )
                )

                regime_entry[f"data{suffix}"] = variant_sample
                regime_entry[f"train_data{suffix}"] = variant_train
                regime_entry[f"test_data{suffix}"] = variant_test
                regime_entry[f"pysr{suffix}"] = pysr_model
                regime_entry[f"nn{suffix}"] = nn_model
                regime_entry[f"mm{suffix}"] = mm_pred
                regime_entry[f"preprocessed{suffix}"] = full_pre.reset_index(drop=True)
                regime_entry[f"test_preprocessed{suffix}"] = test_pre.reset_index(drop=True)

                if suffix == "":
                    regime_entry["pysr"] = pysr_model
                    regime_entry["nn"] = nn_model
                    regime_entry["mm"] = mm_pred

                emit_grouped(variant_lines)

            model_dict[regime_key] = regime_entry

        if not model_dict:
            LOGGER.warning("No dataset size regimes produced results; aborting")
            return OrderedDict()

        return model_dict
    finally:
        CURRENT_SEED_TAG = previous_tag


def evaluate_models(
    data: pd.DataFrame,
    features: Optional[str],
    output_dir: str,
    seed: int,
    pysr_base_size: int,
    nn_base_size: int,
    *,
    num_seeds: int = 3,
) -> None:
    seeds = [seed + offset for offset in range(max(1, num_seeds))]
    seed_model_dicts: List[Dict[str, Dict[str, object]]] = []

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
            run_seed,
            pysr_base_size,
            nn_base_size,
            persist_outputs=persist,
        )
        if model_dict:
            seed_model_dicts.append(model_dict)

    if not seed_model_dicts:
        LOGGER.warning("No dataset size regimes produced results; aborting")
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
    plot_horizontal_boxplots(base_model_dict, output_dir)
    plot_vertical_boxplots(base_model_dict, output_dir)
    error_distributions, seed_medians = _collect_error_distributions(base_model_dict)
    plot_dataset_size_lineplots(error_distributions, seed_medians, output_dir, template=False)
    plot_dataset_size_lineplots(error_distributions, seed_medians, output_dir, template=True)
    save_pysr_formulas(base_model_dict, features, output_dir)
    _emit(
        f"Dataset-size regimes completed across {len(seeds)} seed runs. Outputs written to {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate models across dataset-size regimes")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV")
    parser.add_argument("--features", type=str, help="Comma-separated list of features or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pysr-base-size", type=int, default=3000, dest="pysr_base_size", help="Baseline sample count previously used for PySR training (e.g., 3000)")
    parser.add_argument("--nn-base-size", type=int, default=20000, dest="nn_base_size", help="Baseline pool size previously used for NN training (e.g., 20000)")
    parser.add_argument("--num-seeds", type=int, default=3, dest="num_seeds", help="Number of random seeds to evaluate per model (>=1)")
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset(args.dataset, args.features)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(args.dataset)), "dataset_size_regimes")
    evaluate_models(
        data,
        args.features,
        output_dir,
        seed,
        pysr_base_size=args.pysr_base_size,
        nn_base_size=args.nn_base_size,
        num_seeds=max(1, args.num_seeds),
    )


if __name__ == "__main__":
    main()
