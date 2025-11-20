"""Shared helpers for handling sQSSA/tQSSA model variants across regime scripts."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple
import re

import numpy as np
import pandas as pd

from mm_models import (
    EPS as MM_EPS,
    TARGET_COLUMN as MM_TARGET_COLUMN,
    mm_predictions,
    sqssa_only,
)

# Variant definitions and display names
VARIANTS = OrderedDict([
    ("sQSSA", "sQSSA"),
    ("tQSSA", "tQSSA"),
])

MODEL_FAMILIES = OrderedDict([
    ("pysr", "PySR"),
    ("nn", "Neural Network"),
    ("mm", "Michaelis-Menten"),
])

MODEL_COMBINATIONS = [
    ("pysr", "sQSSA"),
    ("pysr", "tQSSA"),
    ("nn", "sQSSA"),
    ("nn", "tQSSA"),
    ("mm", "sQSSA"),
    ("mm", "tQSSA"),
]

MODEL_LINE_ORDER = [
    f"{MODEL_FAMILIES[family]} ({VARIANTS[variant]})"
    for family, variant in MODEL_COMBINATIONS
]

MODEL_COLORS = {
    ("pysr", "sQSSA"): "#E69F00",
    ("pysr", "tQSSA"): "#F6C85F",
    ("nn", "sQSSA"): "#0072B2",
    ("nn", "tQSSA"): "#4C9CDC",
    ("mm", "sQSSA"): "#009E73",
    ("mm", "tQSSA"): "#33BBAA",
}

MODEL_COLOR_MAP = {
    f"{MODEL_FAMILIES[family]} ({VARIANTS[variant]})": MODEL_COLORS[(family, variant)]
    for family, variant in MODEL_COMBINATIONS
}


def variant_suffix(variant_key: str) -> str:
    return "" if variant_key == "sQSSA" else f"_{variant_key}"


def model_display_name(family: str, variant_key: str) -> str:
    return f"{MODEL_FAMILIES[family]} ({VARIANTS[variant_key]})"


def regime_logger(name: str = "regime") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def augment_for_variant(df: pd.DataFrame, variant_key: str) -> pd.DataFrame:
    if variant_key == "sQSSA":
        return df.copy().reset_index(drop=True)
    if variant_key == "tQSSA":
        augmented = df.copy().reset_index(drop=True)
        if "P_u" not in augmented.columns:
            raise KeyError("Column 'P_u' required for tQSSA augmentation is missing")
        augmented["P_u"] = augmented["P_u"] + augmented[MM_TARGET_COLUMN]
        if augmented.columns[-1] != MM_TARGET_COLUMN:
            feature_cols = [col for col in augmented.columns if col != MM_TARGET_COLUMN]
            augmented = augmented[feature_cols + [MM_TARGET_COLUMN]]
        return augmented
    raise ValueError(f"Unknown variant key '{variant_key}'")


def log_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.log(np.maximum(pred, MM_EPS))
                - np.log(np.maximum(truth, MM_EPS))
            )
        )
    )


def relative_errors(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(truth), MM_EPS)
    return np.abs(pred - truth) / denom


def relative_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(relative_errors(pred, truth)) * 100.0)


def mm_variant_prediction(df: pd.DataFrame, variant_key: str) -> np.ndarray:
    if variant_key == "sQSSA":
        return sqssa_only(df)
    mm_input = df.copy()
    if "P_u" not in mm_input.columns:
        raise KeyError("Column 'P_u' required for tQSSA prediction is missing")
    mm_input["P_u"] = np.maximum(
        mm_input["P_u"] - mm_input[MM_TARGET_COLUMN],
        MM_EPS,
    )
    return mm_predictions(mm_input)[VARIANTS[variant_key]]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    axis_label: str
    folder: str
    filename_prefix: str
    y_scale: str
    floor: Optional[float] = None
    ceiling: Optional[float] = None


METRIC_SPECS = OrderedDict([
    (
        "log_mae",
        MetricSpec(
            key="log_mae",
            label="Log-space MAE",
            axis_label="Log-space MAE |ln(ŷ+ε) − ln(y+ε)| (ε=1e−20)",
            folder="log_mae",
            filename_prefix="log_mae",
            y_scale="log",
            floor=1e-3,
        ),
    ),
])

METRIC_SPECS["relative_mae"] = MetricSpec(
    key="relative_mae",
    label="Relative MAE",
    axis_label="Relative MAE (%)",
    folder="relative_mae",
    filename_prefix="relative_mae",
    y_scale="log",
    floor=1e-3,
    ceiling=1e2,
)


def metric_keys() -> Iterable[str]:
    return METRIC_SPECS.keys()


def metric_spec(metric_key: str) -> MetricSpec:
    return METRIC_SPECS[metric_key]


def metric_bounds(metric_key: str) -> Tuple[Optional[float], Optional[float]]:
    spec = metric_spec(metric_key)
    return spec.floor, spec.ceiling


def metric_limits_with_margin(
    metric_key: str,
    *,
    lower_margin: float = 0.1,
    upper_margin: float = 0.1,
) -> Tuple[Optional[float], Optional[float]]:
    spec = metric_spec(metric_key)
    floor = spec.floor
    ceiling = spec.ceiling
    lower = floor
    upper = ceiling
    if floor is not None:
        if spec.y_scale == "log":
            lower = floor / (1 + lower_margin)
        else:
            lower = max(0.0, floor * (1 - lower_margin))
    if ceiling is not None:
        if spec.y_scale == "log":
            upper = ceiling * (1 + upper_margin)
        else:
            upper = ceiling * (1 + upper_margin)
    return lower, upper


def clip_metric_values(values: np.ndarray, metric_key: str) -> np.ndarray:
    spec = metric_spec(metric_key)
    arr = np.asarray(values, dtype=float)
    if spec.floor is not None:
        arr = np.where(np.isfinite(arr) & (arr < spec.floor), spec.floor, arr)
    if spec.ceiling is not None:
        arr = np.where(np.isfinite(arr) & (arr > spec.ceiling), spec.ceiling, arr)
    return arr


def metric_errors(metric_key: str, preds: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if metric_key == "log_mae":
        return np.abs(
            np.log(np.maximum(preds, MM_EPS))
            - np.log(np.maximum(truth, MM_EPS))
        )
    if metric_key == "relative_mae":
        return relative_errors(preds, truth) * 100.0
    raise ValueError(f"Unknown metric '{metric_key}'")


def metric_score(metric_key: str, preds: np.ndarray, truth: np.ndarray) -> float:
    if metric_key == "log_mae":
        return log_mae(preds, truth)
    if metric_key == "relative_mae":
        return relative_mae(preds, truth)
    raise ValueError(f"Unknown metric '{metric_key}'")


def metric_statistics(metric_key: str, preds: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    errors = metric_errors(metric_key, preds, truth)
    finite = errors[np.isfinite(errors)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
    }


def metric_summary(preds: np.ndarray, truth: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {
        metric_key: metric_statistics(metric_key, preds, truth)
        for metric_key in metric_keys()
    }


def format_regime_message(
    regime_key: str,
    label: str,
    *,
    total_samples: Optional[int] = None,
    target_samples: Optional[int] = None,
    replace: Optional[bool] = None,
) -> str:
    parts = [f"[{regime_key}] {label}"]
    if total_samples is not None:
        parts.append(f"samples={total_samples}")
    if target_samples is not None:
        parts.append(f"target={target_samples}")
    if replace is not None:
        parts.append(f"replace={replace}")
    return " | ".join(parts)


def format_variant_split(
    regime_key: str,
    variant_label: str,
    *,
    train: Optional[int] = None,
    val: Optional[int] = None,
    test: Optional[int] = None,
) -> str:
    counts = []
    if train is not None:
        counts.append(f"train={train}")
    if val is not None:
        counts.append(f"val={val}")
    if test is not None:
        counts.append(f"test={test}")
    count_str = ", ".join(counts) if counts else "splits=unknown"
    return f"[{regime_key}] [{variant_label}] splits: {count_str}"


def format_model_metrics(
    regime_key: str,
    variant_label: str,
    model_label: str,
    metrics: Dict[str, object],
) -> str:
    ordered = []

    def _format_value(value: Optional[float]) -> str:
        if value is None:
            return "nan"
        if isinstance(value, (int, float)) and not np.isfinite(value):
            return "nan"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "nan"

    for metric_key in metric_keys():
        if metric_key not in metrics:
            continue
        value = metrics[metric_key]
        if isinstance(value, dict):
            mean = _format_value(value.get("mean"))
            median = _format_value(value.get("median"))
            ordered.append(f"{metric_key}(mean={mean}, median={median})")
        else:
            ordered.append(f"{metric_key}={_format_value(value)}")

    metrics_str = ", ".join(ordered) if ordered else "no metrics"
    return f"[{regime_key}] [{variant_label}] {model_label}: {metrics_str}"



def collect_variant_predictions(
    models: Dict[str, object],
    variant_key: str,
    evaluate_fn,
):
    suffix = variant_suffix(variant_key)
    eval_df = models.get(f'test_data{suffix}')
    if eval_df is None or eval_df.empty:
        eval_df = models.get(f'data{suffix}')
    if eval_df is None or eval_df.empty:
        return None

    eval_pre = models.get(f'test_preprocessed{suffix}')
    if eval_pre is None or (hasattr(eval_pre, "empty") and eval_pre.empty):
        eval_pre = models.get(f'preprocessed{suffix}')
    X = eval_df.iloc[:, :-1]
    y_true = eval_df.iloc[:, -1].values

    predictions: Dict[str, np.ndarray] = {}

    pysr_model = models.get(f"pysr{suffix}")
    if pysr_model is not None:
        try:
            predictions[model_display_name("pysr", variant_key)] = pysr_model.predict(X.values)
        except Exception:
            pass

    try:
        predictions[model_display_name("mm", variant_key)] = mm_variant_prediction(eval_df, variant_key)
    except Exception:
        pass

    nn_model_ref = models.get(f"nn{suffix}")
    if nn_model_ref is not None and eval_pre is not None:
        try:
            preds = (
                evaluate_fn(nn_model_ref, eval_pre)[1]
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )
            predictions[model_display_name("nn", variant_key)] = preds
        except Exception:
            pass

    if not predictions:
        return None

    return {
        'data': eval_df,
        'features': X,
        'preprocessed': eval_pre,
        'y_true': y_true,
        'predictions': predictions,
    }


def aggregate_seed_error_statistics(
    model_dicts: Iterable[Dict[str, Dict[str, object]]],
    evaluate_fn,
) -> tuple[
    Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
]:
    aggregated: Dict[str, Dict[str, Dict[str, Dict[str, List[np.ndarray]]]]] = OrderedDict()
    seed_medians: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = OrderedDict()

    for model_dict in model_dicts:
        for regime, models in model_dict.items():
            for family, variant_key in MODEL_COMBINATIONS:
                display_name = model_display_name(family, variant_key)
                snapshot = collect_variant_predictions(models, variant_key, evaluate_fn)
                if snapshot is None:
                    continue
                preds = snapshot['predictions'].get(display_name)
                if preds is None:
                    continue
                y_true = snapshot['y_true']
                for metric_key in metric_keys():
                    errors = np.asarray(metric_errors(metric_key, preds, y_true), dtype=float)
                    aggregated.setdefault(regime, OrderedDict()).setdefault(variant_key, OrderedDict()).setdefault(metric_key, OrderedDict()).setdefault(display_name, []).append(errors)
                    finite = errors[np.isfinite(errors)]
                    median_value = float(np.median(finite)) if finite.size else float("nan")
                    seed_medians.setdefault(regime, OrderedDict()).setdefault(variant_key, OrderedDict()).setdefault(metric_key, OrderedDict()).setdefault(display_name, []).append(median_value)

    collapsed: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]] = OrderedDict()
    for regime, variant_map in aggregated.items():
        collapsed_variant: Dict[str, Dict[str, Dict[str, np.ndarray]]] = OrderedDict()
        for variant_key, metric_map in variant_map.items():
            collapsed_metric: Dict[str, Dict[str, np.ndarray]] = OrderedDict()
            for metric_key, model_map in metric_map.items():
                collapsed_model: Dict[str, np.ndarray] = OrderedDict()
                for model_name, error_list in model_map.items():
                    parts: List[np.ndarray] = []
                    for err in error_list:
                        arr = np.asarray(err, dtype=float).ravel()
                        if arr.size:
                            parts.append(arr)
                    if not parts:
                        continue
                    collapsed_model[model_name] = np.concatenate(parts)
                if collapsed_model:
                    collapsed_metric[metric_key] = collapsed_model
            if collapsed_metric:
                collapsed_variant[variant_key] = collapsed_metric
        if collapsed_variant:
            collapsed[regime] = collapsed_variant

    filtered_seed_medians: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = OrderedDict()
    for regime, variant_map in seed_medians.items():
        regime_seed: Dict[str, Dict[str, Dict[str, List[float]]]] = OrderedDict()
        for variant_key, metric_map in variant_map.items():
            variant_seed: Dict[str, Dict[str, List[float]]] = OrderedDict()
            for metric_key, model_map in metric_map.items():
                model_seed: Dict[str, List[float]] = OrderedDict()
                for model_name, values in model_map.items():
                    if not values:
                        continue
                    model_seed[model_name] = values
                if model_seed:
                    variant_seed[metric_key] = model_seed
            if variant_seed:
                regime_seed[variant_key] = variant_seed
        if regime_seed:
            filtered_seed_medians[regime] = regime_seed

    return collapsed, filtered_seed_medians


def gather_model_errors(
    models: Dict[str, object],
    evaluate_fn,
) -> tuple[pd.DataFrame | None, Dict[str, np.ndarray]]:
    base_df = models.get('test_data')
    if base_df is None or base_df.empty:
        base_df = models.get('data')
    if base_df is None or base_df.empty:
        return None, {}

    base_index = base_df.index
    errors: Dict[str, np.ndarray] = {}

    for family, variant_key in MODEL_COMBINATIONS:
        name = model_display_name(family, variant_key)
        snapshot = collect_variant_predictions(models, variant_key, evaluate_fn)
        if snapshot is None:
            continue
        preds = snapshot['predictions'].get(name)
        if preds is None:
            continue
        data_idx = snapshot['data'].index
        try:
            positions = data_idx.map(base_index.get_loc).to_numpy()
        except KeyError:
            continue

        err_vec = np.full(len(base_index), np.nan)
        err_vals = np.abs(
            np.log(np.maximum(preds, MM_EPS))
            - np.log(np.maximum(snapshot['y_true'], MM_EPS))
        )
        err_vec[positions] = err_vals
        errors[name] = err_vec

    return base_df, errors


def _feature_names_for_formula(
    models: Dict[str, object],
    suffix: str,
    features_hint: Optional[str],
    pysr_model,
    raw_formula: str,
) -> list[str]:
    data_frame = models.get(f'data{suffix}')
    if data_frame is None or getattr(data_frame, "empty", False):
        data_frame = models.get('data')

    feature_names: list[str] = []
    if data_frame is not None and not getattr(data_frame, "empty", False):
        cols = list(data_frame.columns)
        if cols:
            feature_names = cols[:-1]

    if not feature_names and features_hint and features_hint != "all":
        feature_names = [col.strip() for col in features_hint.split(',') if col.strip()]

    if not feature_names:
        matches = sorted({int(idx) for idx in re.findall(r"x(\d+)", str(raw_formula))})
        if matches:
            feature_names = [f"x{i}" for i in range(max(matches) + 1)]
        else:
            n_features = getattr(pysr_model, "n_features_in_", 0)
            feature_names = [f"x{i}" for i in range(n_features)]

    return feature_names


def collect_pysr_formula_lines(
    model_dict: Dict[str, Dict[str, object]],
    features_hint: Optional[str] = None,
) -> Dict[str, list[str]]:
    lines: Dict[str, list[str]] = {variant_key: [] for variant_key in VARIANTS.keys()}

    for regime, models in model_dict.items():
        for variant_key in VARIANTS.keys():
            suffix = variant_suffix(variant_key)
            pysr_model = models.get(f"pysr{suffix}")
            if pysr_model is None:
                continue
            equations = getattr(pysr_model, "equations_", None)
            if equations is None or equations.empty:
                continue

            best_row = equations.sort_values(by="score", ascending=False).iloc[0]
            raw_formula = str(best_row["equation"])
            feature_names = _feature_names_for_formula(
                models,
                suffix,
                features_hint,
                pysr_model,
                raw_formula,
            )

            formula = raw_formula
            for idx, name in enumerate(feature_names):
                formula = re.sub(rf"\bx{idx}\b", name, formula)

            lines[variant_key].append(
                f"{regime}: {formula}    [score={best_row['score']:.4f}, loss={best_row['loss']:.4f}]"
            )

    return lines


__all__ = [
    "VARIANTS",
    "MODEL_FAMILIES",
    "MODEL_COMBINATIONS",
    "MODEL_LINE_ORDER",
    "MODEL_COLOR_MAP",
    "MODEL_COLORS",
    "variant_suffix",
    "model_display_name",
    "augment_for_variant",
    "log_mae",
    "relative_errors",
    "relative_mae",
    "mm_variant_prediction",
    "metric_keys",
    "metric_spec",
    "metric_errors",
    "metric_score",
    "metric_statistics",
    "metric_summary",
    "format_regime_message",
    "format_variant_split",
    "format_model_metrics",
    "regime_logger",
    "aggregate_seed_error_statistics",
    "collect_variant_predictions",
    "gather_model_errors",
    "collect_pysr_formula_lines",
    "MM_EPS",
    "MM_TARGET_COLUMN",
    "metric_bounds",
    "metric_limits_with_margin",
    "clip_metric_values",
]
