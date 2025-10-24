"""Benchmark PySR, Michaelis-Menten, and NN models across timepoint groups."""

import os
import argparse
import warnings
import logging
import re
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from nn_model import train_model, evaluate_model
from constants import PYSR_CONFIG
from plot_style import apply_cell_systems_style
from utils.seeding import resolve_seed, seed_everything

warnings.filterwarnings("ignore")
apply_cell_systems_style()

LOGGER = logging.getLogger(__name__)

TIME_COLUMN = "time"
TARGET_COLUMN = "kcat_cg"
EPS = 1e-20
LOG_SCALE_FLOOR = 1e-3
MODEL_LINE_ORDER = ["PySR", "Michaelis-Menten", "Neural Network"]
MODEL_COLORS = {
    "PySR": "#E69F00",
    "Michaelis-Menten": "#009E73",
    "Neural Network": "#0072B2",
}
NN_DATASET_CAP = 20000
MAX_TIME_GROUPS = 4


def _format_number(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "?"
    return f"{value:.3g}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug.lower() or "time_group"


def _assign_time_groups(data: pd.DataFrame, time_column: str, max_groups: int = MAX_TIME_GROUPS):
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


def _build_nn_pool(base: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if base.empty:
        return base
    target = cap if cap and cap > 0 else len(base)
    replace = len(base) < target
    return base.sample(n=target, replace=replace, random_state=seed + 1).reset_index(drop=True)


def load_dataset_with_time(path: str, features: str, time_column: str = TIME_COLUMN) -> pd.DataFrame:
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
    for col in ordered_cols:
        if col == time_column:
            continue
        data[col] = np.exp(pd.to_numeric(data[col], errors="coerce"))

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def michaelis_menten(P_u, k_off, k_D, k_cat, tK, k_inact=None):
    if k_inact is not None:
        denom = P_u + ((k_cat + k_off + k_inact) / (k_off * k_D))
    else:
        denom = P_u + ((k_cat + k_off) / (k_off * k_D))
    return (tK * P_u) / denom


def _mm_components(df: pd.DataFrame):
    required = ["P_u", "k_off", "k_D", "k_cat", "tK"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Michaelis-Menten evaluation: {missing}")
    arrays = [pd.to_numeric(df[col], errors="coerce").to_numpy() for col in required]
    k_inact = None
    if "k_inact" in df.columns:
        raw = pd.to_numeric(df["k_inact"], errors="coerce").to_numpy()
        if np.isfinite(raw).any():
            k_inact = np.where(np.isfinite(raw), raw, 0.0)
    return (*arrays, k_inact)


def _mm_predict(df: pd.DataFrame):
    P_u, k_off, k_D, k_cat, tK, k_inact = _mm_components(df)
    return michaelis_menten(P_u, k_off, k_D, k_cat, tK, k_inact)


def run_pysr(X, y, model_path):
    run_dir = os.path.dirname(model_path)
    cached = (
        os.path.exists(model_path)
        or os.path.exists(os.path.join(run_dir, "hall_of_fame.csv"))
        or os.path.exists(os.path.join(run_dir, "equations.csv"))
    )
    if cached:
        LOGGER.info("Loading existing PySR model from %s", run_dir)
        return PySRRegressor.from_file(run_directory=run_dir)
    LOGGER.info("Training new PySR model in %s", run_dir)
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


def preprocess_data(X: np.ndarray):
    pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log, validate=True)),
        ("scaler", StandardScaler()),
    ])
    return pipeline.fit_transform(X), pipeline


def _log_mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.abs(np.log(np.maximum(y_pred, EPS)) - np.log(np.maximum(y_true, EPS)))


def _collect_error_distributions(model_dict: Dict[str, Dict[str, Any]]):
    distributions = OrderedDict()
    centers = OrderedDict()
    for group_label, models in model_dict.items():
        data = models.get("test_data")
        if data is None or data.empty:
            continue
        y_true = data.iloc[:, -1].to_numpy()
        X = data.iloc[:, :-1]
        regime_errors: Dict[str, np.ndarray] = {}

        y_pred_mm = _mm_predict(data)
        regime_errors["Michaelis-Menten"] = _log_mae(y_pred_mm, y_true)

        pysr_model = models.get("pysr")
        if pysr_model is not None:
            try:
                y_pred_pysr = pysr_model.predict(X.to_numpy())
                regime_errors["PySR"] = _log_mae(y_pred_pysr, y_true)
            except Exception:
                LOGGER.exception("PySR prediction failed for group=%s", group_label)

        nn_model = models.get("nn")
        test_pre = models.get("test_preprocessed")
        if nn_model is not None and test_pre is not None:
            try:
                preds = evaluate_model(nn_model, test_pre)[1]
                y_pred_nn = preds.detach().cpu().numpy().flatten()
                regime_errors["Neural Network"] = _log_mae(y_pred_nn, y_true)
            except Exception:
                LOGGER.exception("NN prediction failed for group=%s", group_label)

        if regime_errors:
            distributions[group_label] = regime_errors
            centers[group_label] = models.get("center")
    return distributions, centers


def _errors_to_dataframe(distributions: Dict[str, np.ndarray], time_label: str) -> pd.DataFrame:
    records = []
    for model_name, errors in distributions.items():
        arr = np.asarray(errors, dtype=float).flatten()
        arr = arr[np.isfinite(arr)]
        records.extend({"Model": model_name, "Log-MAE": val, "Time": time_label} for val in arr)
    return pd.DataFrame(records)


def _combined_error_frame(error_distributions: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    frames = []
    for label, dists in error_distributions.items():
        frames.append(_errors_to_dataframe(dists, label))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["Model", "Log-MAE", "Time"])


def plot_timepoint_horizontal_boxplots(error_distributions, output_dir):
    if not error_distributions:
        return
    df = _combined_error_frame(error_distributions)
    if df.empty:
        return
    ordered_times = list(error_distributions.keys())

    fig, axes = plt.subplots(len(ordered_times), 1, figsize=(10, 2.5 * len(ordered_times)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, time_label in zip(axes, ordered_times):
        subset = df[df["Time"] == time_label]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=subset,
            x="Log-MAE",
            y="Model",
            palette=MODEL_COLORS,
            orient="h",
            order=MODEL_LINE_ORDER,
            ax=ax,
        )
        ax.set_title(time_label, fontsize=13, weight="bold")
    axes[-1].set_xlabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)|", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(len(ordered_times), 1, figsize=(10, 2.5 * len(ordered_times)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, time_label in zip(axes, ordered_times):
        subset = df[df["Time"] == time_label]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=subset,
            x="Log-MAE",
            y="Model",
            palette=MODEL_COLORS,
            orient="h",
            order=MODEL_LINE_ORDER,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(time_label, fontsize=13, weight="bold")
    axes[-1].set_xlabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)|", fontsize=12)
    for ax in axes[:-1]:
        ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_horizontal_boxplot_no_outliers.png"), dpi=300)
    plt.close()


def plot_timepoint_vertical_boxplots(error_distributions, output_dir):
    if not error_distributions:
        return
    df = _combined_error_frame(error_distributions)
    if df.empty:
        return
    ordered_times = list(error_distributions.keys())
    fig, axes = plt.subplots(1, len(ordered_times), figsize=(3.4 * len(ordered_times), 4.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, time_label in zip(axes, ordered_times):
        subset = df[df["Time"] == time_label]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=subset,
            x="Model",
            y="Log-MAE",
            palette=MODEL_COLORS,
            orient="v",
            order=MODEL_LINE_ORDER,
            ax=ax,
        )
        ax.set_title(time_label, fontsize=13, weight="bold")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment("right")
    axes[0].set_ylabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)|", fontsize=12)
    for ax in axes[1:]:
        ax.set_ylabel("")
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_vertical_boxplot.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, len(ordered_times), figsize=(3.4 * len(ordered_times), 4.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, time_label in zip(axes, ordered_times):
        subset = df[df["Time"] == time_label]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=subset,
            x="Model",
            y="Log-MAE",
            palette=MODEL_COLORS,
            orient="v",
            order=MODEL_LINE_ORDER,
            ax=ax,
            showfliers=False,
        )
        ax.set_title(time_label, fontsize=13, weight="bold")
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment("right")
    axes[0].set_ylabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)|", fontsize=12)
    for ax in axes[1:]:
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared/plots/log_mae_vertical_boxplot_no_outliers.png"), dpi=300)
    plt.close()


def _compute_line_stats(error_distributions):
    stats = {model: {"median": [], "q1": [], "q3": []} for model in MODEL_LINE_ORDER}
    for regime_key, regime_errors in error_distributions.items():
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


def _lineplot_limits(stats):
    lowers, uppers, positives = [], [], []
    for model in MODEL_LINE_ORDER:
        lowers.extend([v for v in stats[model]["q1"] if np.isfinite(v)])
        uppers.extend([v for v in stats[model]["q3"] if np.isfinite(v)])
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
    if positives and ymin <= 0:
        ymin = min(positives) * 0.8
    ymin = max(ymin, LOG_SCALE_FLOOR)
    if ymax <= ymin:
        ymax = ymin * 1.5
    return ymin, ymax


def plot_timepoint_lineplot(error_distributions, centers, output_dir, template=False):
    if not error_distributions:
        return
    ordered_labels = list(error_distributions.keys())
    stats = _compute_line_stats(error_distributions)
    _apply_log_floor(stats)
    has_finite = any(
        np.isfinite(val)
        for model in MODEL_LINE_ORDER
        for val in stats[model]["median"]
    )

    use_log_scale = False
    if centers:
        x = np.array([
            centers.get(label, idx)
            if centers.get(label) is not None else idx
            for idx, label in enumerate(ordered_labels)
        ], dtype=float)
        positive_mask = x > 0
        if not np.all(positive_mask):
            if positive_mask.any():
                min_positive = float(np.min(x[positive_mask]))
                replacement = max(min_positive * 0.5, np.finfo(float).tiny)
            else:
                replacement = LOG_SCALE_FLOOR
            x = np.where(positive_mask, x, replacement)
        if np.all(x > 0) and has_finite:
            use_log_scale = True
    else:
        x = np.arange(len(ordered_labels), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ymin, ymax = _lineplot_limits(stats)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    drew_lines = False
    if not template:
        for model in MODEL_LINE_ORDER:
            med = np.array(stats[model]["median"], dtype=float)
            if med.size == 0 or np.all(np.isnan(med)):
                continue
            q1 = np.array(stats[model]["q1"], dtype=float)
            q3 = np.array(stats[model]["q3"], dtype=float)
            ax.plot(x, med, label=model, color=MODEL_COLORS[model], marker="o", linewidth=2)
            ax.fill_between(x, q1, q3, color=MODEL_COLORS[model], alpha=0.25)
            drew_lines = True
        ax.legend(loc="upper left", frameon=False)
    else:
        handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], marker="o", linewidth=2, label=m) for m in MODEL_LINE_ORDER]
        ax.legend(handles=handles, loc="upper left", frameon=False)

    all_vals = []
    for model in MODEL_LINE_ORDER:
        all_vals.extend(stats[model]["median"])
        all_vals.extend(stats[model]["q1"])
        all_vals.extend(stats[model]["q3"])
    all_vals = np.array(all_vals, dtype=float)
    finite = all_vals[np.isfinite(all_vals)]
    positive = finite[finite > 0]
    if positive.size:
        if (finite <= 0).any():
            eps = max(LOG_SCALE_FLOOR, positive.min() * 1e-3)
            for model in MODEL_LINE_ORDER:
                for key in ("median", "q1", "q3"):
                    stats[model][key] = [
                        val if (not np.isfinite(val) or val > eps) else eps
                        for val in stats[model][key]
                    ]
        ax.set_yscale("log")
        ax.set_ylim(bottom=LOG_SCALE_FLOOR)
    else:
        ax.set_yscale("linear")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_labels, rotation=20, ha="right")
    if template or not drew_lines:
        use_log_scale = False
    if use_log_scale:
        try:
            ax.set_xscale("log")
            left = max(np.min(x) * 0.8, np.finfo(float).tiny)
            right = np.max(x) * 1.2
            if right <= left:
                right = left * 10.0
            ax.set_xlim(left=left, right=right)
        except ValueError:
            use_log_scale = False
    if use_log_scale:
        ax.set_xlabel("Time", fontsize=12)
    else:
        ax.set_xlabel("Time Group", fontsize=12)
    ax.set_ylabel("Log-space MAE |ln(ŷ+ε) − ln(y+ε)|", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plot_name = "log_mae_timepoint_lineplot_template.png" if template else "log_mae_timepoint_lineplot.png"
    os.makedirs(os.path.join(output_dir, "shared/plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"shared/plots/{plot_name}"), dpi=300)
    plt.close()


def save_pysr_formulas(model_dict, output_dir):
    lines = []
    for group_label, models in model_dict.items():
        pysr_model = models.get("pysr")
        feature_names = models.get("feature_names", [])
        if pysr_model is None or getattr(pysr_model, "equations_", None) is None:
            continue
        equations = pysr_model.equations_
        if equations is None or equations.empty:
            continue
        best = equations.sort_values(by="score", ascending=False).iloc[0]
        formula = str(best["equation"])
        for idx, name in enumerate(feature_names):
            formula = formula.replace(f"x{idx}", name)
        lines.append(f"{group_label}: {formula}    [score={best['score']:.4f}, loss={best['loss']:.4f}]")
    os.makedirs(os.path.join(output_dir, "shared/results/pysr"), exist_ok=True)
    with open(os.path.join(output_dir, "shared/results/pysr/all_pysr_formulas.txt"), "w") as f:
        for line in lines:
            f.write(line + "\n")


def evaluate_models(data: pd.DataFrame, output_dir: str, dataset_size: int, seed: int, time_column: str = TIME_COLUMN):
    os.makedirs(output_dir, exist_ok=True)
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
        return

    model_dict: Dict[str, Dict[str, Any]] = OrderedDict()

    for info in group_infos:
        label = info["label"]
        group_rows = grouped_data[grouped_data["time_group"] == info["id"]].reset_index(drop=True)
        if len(group_rows) < 2:
            LOGGER.warning("Skipping %s: need at least 2 samples", label)
            continue

        group_dir = os.path.join(output_dir, info["dirname"])
        os.makedirs(os.path.join(group_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(group_dir, "models/pysr"), exist_ok=True)
        os.makedirs(os.path.join(group_dir, "models/nn"), exist_ok=True)
        os.makedirs(os.path.join(group_dir, "plots"), exist_ok=True)

        group_rows.to_csv(os.path.join(group_dir, "processed/full_group.csv"), index=False)

        base_no_time = group_rows.drop(columns=["time_group", time_column]).reset_index(drop=True)
        capped_size = min(dataset_size, len(base_no_time)) if dataset_size else len(base_no_time)
        capped_size = max(2, capped_size)
        symbolic_sample = (
            base_no_time.sample(n=capped_size, random_state=seed)
            if capped_size < len(base_no_time)
            else base_no_time
        ).reset_index(drop=True)

        nn_pool = _build_nn_pool(base_no_time, NN_DATASET_CAP, seed)

        sample = symbolic_sample.copy()
        sample.to_csv(os.path.join(group_dir, "processed/model_sample.csv"), index=False)

        LOGGER.info(
            "Processing %s | total rows=%d | sample size=%d",
            label,
            len(group_rows),
            len(sample),
        )

        Xy = sample.copy()
        if len(Xy) < 2:
            LOGGER.warning("Skipping %s: insufficient rows after preprocessing", label)
            continue

        test_size = max(1, int(np.ceil(len(Xy) * 0.2)))
        if len(Xy) - test_size < 1:
            LOGGER.warning("Skipping %s: insufficient train samples", label)
            continue

        train_df, test_df = train_test_split(Xy, test_size=test_size, random_state=seed, shuffle=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(os.path.join(group_dir, "processed/train.csv"), index=False)
        test_df.to_csv(os.path.join(group_dir, "processed/test.csv"), index=False)

        print(f"Evaluating {label} with {len(train_df)} train / {len(test_df)} test samples")

        if len(train_df) < 2:
            LOGGER.warning("Skipping %s: need at least 2 training samples", label)
            continue

        feature_names = [col for col in train_df.columns if col != TARGET_COLUMN]
        X_train, y_train = train_df[feature_names].to_numpy(), train_df[TARGET_COLUMN].to_numpy()
        X_test, y_test = test_df[feature_names].to_numpy(), test_df[TARGET_COLUMN].to_numpy()
        X_full, y_full = Xy[feature_names].to_numpy(), Xy[TARGET_COLUMN].to_numpy()

        pysr_path = os.path.join(group_dir, "models/pysr/hall_of_fame.pkl")
        pysr_model = run_pysr(X_train, y_train, pysr_path)
        y_pred_pysr = pysr_model.predict(X_test)
        pysr_loss = _log_mae(y_pred_pysr, y_test).mean()
        LOGGER.info("%s PySR log-MAE: %.5f", label, pysr_loss)
        print(f"PySR Loss (test): {pysr_loss}")

        y_pred_mm = _mm_predict(test_df)
        mm_loss = _log_mae(y_pred_mm, y_test).mean()
        LOGGER.info("%s MM log-MAE: %.5f", label, mm_loss)
        print(f"Michaelis-Menten Loss (test): {mm_loss}")

        nn_target = nn_pool[TARGET_COLUMN]
        nn_features = nn_pool.drop(columns=[TARGET_COLUMN])
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

        # Ensure evaluation set is disjoint from NN train/validation pool
        eval_rows = sample.copy()
        eval_rows['_tmp_idx'] = np.arange(len(eval_rows))
        nn_pool['_tmp_idx'] = np.arange(len(nn_pool))
        overlap = nn_pool.merge(
            eval_rows,
            how='inner',
            on=feature_names,
            suffixes=('_nn', '_eval'),
        )
        if not overlap.empty:
            nn_pool = nn_pool.drop(index=overlap['_tmp_idx_nn']).reset_index(drop=True)
        nn_pool = nn_pool.drop(columns=['_tmp_idx'], errors='ignore')
        eval_rows = eval_rows.drop(columns=['_tmp_idx'], errors='ignore')
        if not nn_pool_eval_overlap.empty:
            nn_pool = nn_pool.drop(index=nn_pool_eval_overlap).reset_index(drop=True)
            nn_features = nn_pool.drop(columns=[TARGET_COLUMN])
            nn_target = nn_pool[TARGET_COLUMN]
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
            np.column_stack([nn_pipeline.transform(train_df[feature_names].values), np.log(y_train)]),
            columns=list(train_df.columns),
        )
        test_pre = pd.DataFrame(
            np.column_stack([nn_pipeline.transform(test_df[feature_names].values), np.log(y_test)]),
            columns=list(test_df.columns),
        )
        full_pre = pd.DataFrame(
            np.column_stack([nn_pipeline.transform(sample[feature_names].values), np.log(y_full)]),
            columns=list(sample.columns),
        )

        train_split = nn_train_pre.reset_index(drop=True)
        val_split = nn_test_pre.reset_index(drop=True)

        nn_path = os.path.join(group_dir, "models/nn/model.pkl")
        nn_model = train_model(train_split, val_split, nn_path, verbose=False, seed=seed)
        print(
            "NN splits | train: %d, val: %d, eval: %d (cap %d)",
            len(train_split),
            len(val_split),
            len(test_df),
            NN_DATASET_CAP,
        )
        nn_loss, nn_preds = evaluate_model(nn_model, test_pre)
        LOGGER.info("%s NN log-MAE: %.5f", label, nn_loss)
        print(f"NN Loss (test): {nn_loss}")

        model_dict[label] = {
            "pysr": pysr_model,
            "mm": y_pred_mm,
            "nn": nn_model,
            "data": Xy,
            "train_data": train_df,
            "test_data": test_df,
            "preprocessed": full_pre,
            "test_preprocessed": test_pre,
            "feature_names": feature_names,
            "nn_preds": nn_preds,
            "center": info.get("center"),
        }

    error_distributions, centers = _collect_error_distributions(model_dict)
    if not error_distributions:
        LOGGER.warning("No valid timepoints processed; skipping plots")
        return

    plot_timepoint_horizontal_boxplots(error_distributions, output_dir)
    plot_timepoint_vertical_boxplots(error_distributions, output_dir)
    plot_timepoint_lineplot(error_distributions, centers, output_dir, template=False)
    plot_timepoint_lineplot(error_distributions, centers, output_dir, template=True)
    save_pysr_formulas(model_dict, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Benchmark models by timepoint using PySR, MM, and NN.")
    parser.add_argument("--dataset", required=True, help="Path to merged dynamic dataset")
    parser.add_argument("--dataset_size", type=int, help="Maximum samples per timepoint")
    parser.add_argument("--features", type=str, help="Comma-separated list of features or 'all'")
    parser.add_argument("--time_column", type=str, default=TIME_COLUMN, help="Column representing time")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed = seed_everything(resolve_seed(args.seed))
    data = load_dataset_with_time(args.dataset, args.features, args.time_column)
    root_dir = os.path.dirname(os.path.dirname(args.dataset))
    output_dir = os.path.join(root_dir, "timepoint_regimes")

    evaluate_models(
        data,
        output_dir=output_dir,
        dataset_size=args.dataset_size or len(data),
        seed=seed,
        time_column=args.time_column,
    )


if __name__ == "__main__":
    main()
