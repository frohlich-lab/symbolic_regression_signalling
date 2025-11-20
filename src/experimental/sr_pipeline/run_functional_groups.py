"""Symbolic regression over functional marker groups.

This script mirrors the SR exploration notebook logic for part 3 by
training PySR models on balanced trajectories grouped by functional
markers. It supports two feature configurations "gfp" (GFP-only
features) and "all" (all available dynamical features) and reports test
metrics only, per user request.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pysr import PySRRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from utils.seeding import seed_everything
from experimental.data_prep.balancing import balance_by_order_of_magnitude

# Columns that should not be passed to the regression model as inputs.
EXCLUDE_COLUMNS = {"p-ERK1-2_dt", "p-MEK1-2_dt", "marker", "timepoint", "GFP_bin"}


LOGGER = logging.getLogger("sr.functional_groups")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        LOGGER.info("%s [%d]", stage, current)
        return
    percent = (current / total) * 100.0
    LOGGER.info("%s [%d/%d | %.1f%%]", stage, current, total, percent)


@dataclass
class GroupResult:
    group_name: str
    markers: Tuple[str, ...]
    feature_mode: str
    model_name: str
    formula: Optional[str]
    test_log_mae: Optional[float]
    test_log_r2: Optional[float]
    test_mae: Optional[float]
    test_relative_mae: Optional[float]
    test_log_relative_mae: Optional[float]
    test_r2: Optional[float]
    test_samples: int
    train_samples: int
    balanced_test_log_mae: Optional[float] = None
    balanced_test_log_r2: Optional[float] = None
    balanced_test_mae: Optional[float] = None
    balanced_test_relative_mae: Optional[float] = None
    balanced_test_log_relative_mae: Optional[float] = None
    balanced_test_r2: Optional[float] = None
    balanced_test_samples: Optional[int] = None
    target_space: Literal["log", "linear"] = "log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PySR over functional marker groups.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the CSV file produced from the perturbation trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where reports and plots will be written.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("snapshot", "per_minute"),
        default="snapshot",
        help="Source dataset format. 'snapshot' expects the fit snapshot CSV; 'per_minute' uses the dense per-minute fit grid.",
    )
    parser.add_argument(
        "--per-minute-max-time",
        type=float,
        default=30.0,
        help="When --dataset-mode per_minute is used, keep rows with timepoint ≤ this value (default: 30).",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="*",
        choices=("gfp", "all"),
        default=("gfp", "all"),
        help="Feature modes to run. 'gfp' keeps GFP-only columns; 'all' keeps every feature except exclusions.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pysr", "linreg"),
        default=("pysr",),
        help="Regression models to evaluate (default: pysr). Specify 'linreg' for the linear baseline.",
    )
    parser.add_argument(
        "--include-unbalanced-baseline",
        action="store_true",
        default=True,
        help="Also train/evaluate each model on the raw (unbalanced) split for comparison.",
    )
    parser.add_argument(
        "--disable-balancing",
        action="store_true",
        help="Skip order-of-magnitude balancing (use the raw splits as-is).",
    )
    parser.add_argument(
        "--gfp-columns",
        nargs="*",
        default=("GFP",),
        help="Columns to keep when --feature-modes includes 'gfp'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test evaluation (default: 0.2 for an 80/20 split).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and resampling.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=500,
        help="Minimum samples per order-of-magnitude bin when balancing the dataset.",
    )
    parser.add_argument(
        "--max-bin-samples",
        type=int,
        default=400,
        help="Maximum samples per order-of-magnitude bin when balancing the dataset.",
    )
    parser.add_argument(
        "--log10-cutoff",
        type=float,
        default=-3.0,
        help="Cutoff used for the signed log transform (matches the notebook default).",
    )
    parser.add_argument(
        "--marker-groups-json",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default marker groups. Should map group names to marker lists.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=200,
        help="Skip groups with fewer combined samples than this threshold after preprocessing.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="PySR niterations parameter (default mirrors notebook).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="PySR population size per generation.",
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=30,
        help="PySR populations (parallel demes).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=20,
        help="Maximum expression size for PySR.",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=0.8,
        help="Parsimony coefficient for PySR (complexity penalty).",
    )
    parser.add_argument(
        "--binary-operators",
        nargs="*",
        default=("+", "-", "*", "/"),
        help="Binary operators exposed to PySR.",
    )
    parser.add_argument(
        "--unary-operators",
        nargs="*",
        default=(),
        help="Optional unary operators for PySR (e.g. exp, log).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="PySR verbosity level (default silences per-iteration logging).",
    )
    parser.add_argument(
        "--annealing",
        action="store_true",
        default=True,
        help="Enable PySR simulated annealing (on by default).",
    )
    parser.add_argument(
        "--no-annealing",
        action="store_false",
        dest="annealing",
        help="Disable PySR simulated annealing.",
    )
    parser.add_argument(
        "--batching",
        action="store_true",
        default=True,
        help="Enable PySR mini-batching (on by default).",
    )
    parser.add_argument(
        "--no-batching",
        action="store_false",
        dest="batching",
        help="Disable PySR mini-batching.",
    )
    parser.add_argument(
        "--group-definitions-csv",
        type=Path,
        default=Path("data/experimental/processed/functional_groups.csv"),
        help="CSV file listing functional groups with member markers (columns: group, members).",
    )
    return parser.parse_args()


def load_marker_groups(
    dataset: pd.DataFrame,
    json_path: Optional[Path],
    csv_path: Optional[Path],
) -> Dict[str, List[str]]:
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(f"Group definitions CSV not found: {csv_path}")
        meta = pd.read_csv(csv_path)
        if "group" not in meta.columns or "members" not in meta.columns:
            raise ValueError(
                "Group definitions CSV must contain 'group' and 'members' columns"
            )
        groups: Dict[str, List[str]] = {}
        available_markers = set(dataset["marker"].unique())
        for _, row in meta.iterrows():
            name = str(row["group"]).strip()
            raw_members = row["members"]
            if pd.isna(name) or pd.isna(raw_members):
                continue
            try:
                parsed = ast.literal_eval(str(raw_members))
            except Exception:
                parsed = [m.strip() for m in str(raw_members).strip("[]").split(",") if m.strip()]
            members = [str(m).strip() for m in parsed if str(m).strip() in available_markers]
            if members:
                groups[name] = members
        if not groups:
            raise ValueError(
                "No functional groups found after intersecting CSV definitions with dataset markers"
            )
        return groups

    if json_path is not None:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {str(k): list(v) for k, v in payload.items()}
    raise ValueError("A group definitions CSV or JSON must be provided to load marker groups.")


def signed_log(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    epsilon = 10.0 ** log10_cutoff
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * (np.log10(np.abs(arr) + epsilon) - log10_cutoff)


def inverse_signed_log(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    epsilon = 10.0 ** log10_cutoff
    arr = np.asarray(values, dtype=float)
    magnitude = np.abs(arr)
    base = np.power(10.0, magnitude + log10_cutoff) - epsilon
    base = np.maximum(base, 0.0)
    return np.sign(arr) * base


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log10_cutoff: float,
) -> Dict[str, Optional[float]]:
    log_y_true = signed_log(y_true, log10_cutoff)
    log_y_pred = signed_log(y_pred, log10_cutoff)

    metrics: Dict[str, Optional[float]] = {
        "test_log_mae": float(mean_absolute_error(log_y_true, log_y_pred)),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_log_r2": None,
        "test_r2": None,
        "test_relative_mae": None,
        "test_log_relative_mae": None,
    }

    if len(np.unique(log_y_pred)) > 1:
        metrics["test_log_r2"] = float(r2_score(log_y_true, log_y_pred))

    if len(np.unique(y_pred)) > 1:
        metrics["test_r2"] = float(r2_score(y_true, y_pred))

    abs_true = np.abs(y_true)
    nonzero_mask = abs_true > 1e-8
    if np.any(nonzero_mask):
        rel_errors = np.abs(y_pred[nonzero_mask] - y_true[nonzero_mask]) / abs_true[nonzero_mask]
        metrics["test_relative_mae"] = float(np.mean(rel_errors))

    log_epsilon = 10.0 ** log10_cutoff
    log_true = np.log10(np.abs(y_true) + log_epsilon)
    log_pred = np.log10(np.abs(y_pred) + log_epsilon)
    finite_mask = np.isfinite(log_true) & np.isfinite(log_pred)
    if np.any(finite_mask):
        log_true_safe = log_true[finite_mask]
        log_diff = np.abs(log_pred[finite_mask] - log_true_safe)
        denom = np.maximum(np.abs(log_true_safe), 1e-8)
        rel_log_errors = log_diff / denom
        metrics["test_log_relative_mae"] = float(np.mean(rel_log_errors))

    return metrics


def _format_linear_formula(
    pipeline: Pipeline,
    feature_names: Sequence[str],
) -> str:
    scaler = pipeline.named_steps.get("scaler")
    reg: LinearRegression = pipeline.named_steps["regressor"]

    if scaler is None:
        coeffs = reg.coef_
        intercept = reg.intercept_
    else:
        scale = np.where(np.asarray(scaler.scale_) == 0.0, 1.0, scaler.scale_)
        mean = np.asarray(scaler.mean_)
        coeffs = reg.coef_ / scale
        intercept = reg.intercept_ - np.sum((reg.coef_ * mean) / scale)

    parts = [f"{intercept:.4f}"]
    for name, coef in zip(feature_names, coeffs):
        parts.append(f" {coef:+.4f} * {name}")
    return "".join(parts)


def sanitize_feature_names(columns: Iterable[str]) -> List[str]:
    renamed: List[str] = []
    for col in columns:
        clean = col
        clean = clean.replace("-", "_")
        clean = clean.replace(" ", "_")
        clean = clean.replace("(", "")
        clean = clean.replace(")", "")
        clean = clean.replace("/", "_")
        if clean and clean[0].isdigit():
            clean = f"f_{clean}"
        renamed.append(clean)
    return renamed


def select_features(
    df: pd.DataFrame, feature_mode: str, gfp_columns: Sequence[str]
) -> List[str]:
    candidate_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    if feature_mode == "gfp":
        return [c for c in candidate_columns if c in gfp_columns]
    if feature_mode == "all":
        return candidate_columns
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def train_group_models(
    data: pd.DataFrame,
    group_label: str,
    markers: Sequence[str],
    feature_mode: str,
    feature_columns: List[str],
    test_size: float,
    random_state: int,
    log10_cutoff: float,
    min_bin_samples: int,
    max_bin_samples: int,
    include_unbalanced: bool,
    *,
    run_pysr: bool,
    run_linreg: bool,
    sr_kwargs: Optional[Dict[str, object]] = None,
    apply_balancing: bool = True,
) -> List[GroupResult]:
    subset = data[data["marker"].isin(markers)].copy()
    if subset.empty:
        return []

    subset = subset.dropna(subset=feature_columns + ["p-ERK1-2_dt"])
    if subset.empty:
        return []

    X = subset[feature_columns].copy()
    y = subset["p-ERK1-2_dt"].astype(float)

    if len(y) < 5 or X.shape[1] == 0:
        return []

    sanitized_names = sanitize_feature_names(X.columns)
    X.columns = sanitized_names

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = subset.iloc[train_idx].copy()
    test_df_raw = subset.iloc[test_idx].copy()

    if apply_balancing:
        train_df = balance_by_order_of_magnitude(
            train_df,
            target_column="p-ERK1-2_dt",
            log10_cutoff=log10_cutoff,
            min_samples=min_bin_samples,
            max_samples=max_bin_samples,
            random_state=random_state,
        )
        if train_df.empty:
            return []
        test_df_balanced = balance_by_order_of_magnitude(
            test_df_raw,
            target_column="p-ERK1-2_dt",
            log10_cutoff=log10_cutoff,
            min_samples=min_bin_samples,
            max_samples=max_bin_samples,
            random_state=random_state,
        )
    else:
        test_df_balanced = pd.DataFrame(columns=test_df_raw.columns)

    def _make_xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x_local = frame[feature_columns].copy()
        x_local.columns = sanitized_names
        y_local = frame["p-ERK1-2_dt"].astype(float)
        return x_local, y_local

    X_train, y_train = _make_xy(train_df)
    X_test_raw, y_test_raw = _make_xy(test_df_raw)
    X_test_bal, y_test_bal = _make_xy(test_df_balanced) if not test_df_balanced.empty else (pd.DataFrame(columns=sanitized_names), pd.Series(dtype=float))

    if len(y_train) == 0 or (len(y_test_raw) == 0 and len(y_test_bal) == 0):
        return []

    log_loss_expr = (
        f"my_loss(x,y)=(sign(x)*(log10(abs(x)+10^{log10_cutoff})+{abs(log10_cutoff)})"
        f"-sign(y)*(log10(abs(y)+10^{log10_cutoff})+{abs(log10_cutoff)}))^2"
    )
    linear_loss_expr = "my_loss(x,y)=(x - y)^2"

    results: List[GroupResult] = []

    def _unbalanced_result_block() -> List[GroupResult]:
        """Train/eval on raw (unbalanced) split for comparison."""
        sub_results: List[GroupResult] = []
        X_train_raw, y_train_raw = _make_xy(subset.iloc[train_idx])
        X_test_raw_only, y_test_raw_only = _make_xy(subset.iloc[test_idx])
        if run_pysr:
            try:
                model = PySRRegressor(elementwise_loss=log_loss_expr, **(sr_kwargs or {}))
                model.fit(X_train_raw, y_train_raw)
                y_pred = model.predict(X_test_raw_only)
                metrics = _compute_regression_metrics(y_test_raw_only.values, y_pred, log10_cutoff)
                formula = str(model.sympy()) if model.equations_ is not None else None
            except Exception:
                metrics = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }
                formula = None
            sub_results.append(
                GroupResult(
                    group_name=group_label,
                    markers=tuple(markers),
                    feature_mode=feature_mode,
                    model_name="PySR (raw)",
                    formula=formula,
                    test_log_mae=metrics["test_log_mae"],
                    test_log_r2=metrics["test_log_r2"],
                    test_mae=metrics["test_mae"],
                    test_relative_mae=metrics["test_relative_mae"],
                    test_log_relative_mae=metrics["test_log_relative_mae"],
                    test_r2=metrics["test_r2"],
                    test_samples=len(y_test_raw_only),
                    train_samples=len(y_train_raw),
                )
            )
        if run_linreg:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ])
            try:
                X_train_log = signed_log(X_train_raw.values, log10_cutoff)
                X_test_log = signed_log(X_test_raw_only.values, log10_cutoff)
                y_train_log = signed_log(y_train_raw.values, log10_cutoff)
                pipeline.fit(X_train_log, y_train_log)
                y_pred_log = pipeline.predict(X_test_log)
                y_pred_lin = inverse_signed_log(y_pred_log, log10_cutoff)
                metrics_lin = _compute_regression_metrics(y_test_raw_only.values, y_pred_lin, log10_cutoff)
                log_feature_names = [f"log10_{name}" for name in sanitized_names]
                formula_lin = _format_linear_formula(pipeline, log_feature_names)
            except Exception:
                metrics_lin = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }
                formula_lin = None
            sub_results.append(
                GroupResult(
                    group_name=group_label,
                    markers=tuple(markers),
                    feature_mode=feature_mode,
                    model_name="Linear Regression (raw)",
                    formula=formula_lin,
                    test_log_mae=metrics_lin["test_log_mae"],
                    test_log_r2=metrics_lin["test_log_r2"],
                    test_mae=metrics_lin["test_mae"],
                    test_relative_mae=metrics_lin["test_relative_mae"],
                    test_log_relative_mae=metrics_lin["test_log_relative_mae"],
                    test_r2=metrics_lin["test_r2"],
                    test_samples=len(y_test_raw_only),
                    train_samples=len(y_train_raw),
                )
            )
        return sub_results

    if run_pysr:
        pysr_model = PySRRegressor(
            elementwise_loss=log_loss_expr,
            **(sr_kwargs or {}),
        )

        try:
            pysr_model.fit(X_train, y_train)

            if len(y_test_raw) > 0:
                y_pred_test_raw = pysr_model.predict(X_test_raw)
                metrics_raw = _compute_regression_metrics(
                    y_test_raw.values,
                    y_pred_test_raw,
                    log10_cutoff,
                )
            else:
                metrics_raw = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }

            if len(y_test_bal) > 0:
                y_pred_test_bal = pysr_model.predict(X_test_bal)
                metrics_bal = _compute_regression_metrics(
                    y_test_bal.values,
                    y_pred_test_bal,
                    log10_cutoff,
                )
            else:
                metrics_bal = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }
            formula = str(pysr_model.sympy()) if pysr_model.equations_ is not None else None
        except Exception:
            metrics_raw = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            metrics_bal = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula = None

        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="PySR",
                formula=formula,
                test_log_mae=metrics_raw["test_log_mae"],
                test_log_r2=metrics_raw["test_log_r2"],
                test_mae=metrics_raw["test_mae"],
                test_relative_mae=metrics_raw["test_relative_mae"],
                test_log_relative_mae=metrics_raw["test_log_relative_mae"],
                test_r2=metrics_raw["test_r2"],
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                balanced_test_log_mae=metrics_bal["test_log_mae"],
                balanced_test_log_r2=metrics_bal["test_log_r2"],
                balanced_test_mae=metrics_bal["test_mae"],
                balanced_test_relative_mae=metrics_bal["test_relative_mae"],
                balanced_test_log_relative_mae=metrics_bal["test_log_relative_mae"],
                balanced_test_r2=metrics_bal["test_r2"],
                balanced_test_samples=len(y_test_bal),
            )
        )

    if run_linreg:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        try:
            X_train_log = signed_log(X_train.values, log10_cutoff)
            X_test_raw_log = signed_log(X_test_raw.values, log10_cutoff) if len(y_test_raw) > 0 else None
            X_test_bal_log = signed_log(X_test_bal.values, log10_cutoff) if len(y_test_bal) > 0 else None
            y_train_log = signed_log(y_train.values, log10_cutoff)

            pipeline.fit(X_train_log, y_train_log)
            if X_test_raw_log is not None:
                y_pred_log_space = pipeline.predict(X_test_raw_log)
                y_pred_lin = inverse_signed_log(y_pred_log_space, log10_cutoff)
                metrics_lin_raw = _compute_regression_metrics(
                    y_test_raw.values,
                    y_pred_lin,
                    log10_cutoff,
                )
            else:
                metrics_lin_raw = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }

            if X_test_bal_log is not None and len(y_test_bal) > 0:
                y_pred_bal_log = pipeline.predict(X_test_bal_log)
                y_pred_bal_lin = inverse_signed_log(y_pred_bal_log, log10_cutoff)
                metrics_lin_bal = _compute_regression_metrics(
                    y_test_bal.values,
                    y_pred_bal_lin,
                    log10_cutoff,
                )
            else:
                metrics_lin_bal = {
                    "test_log_mae": None,
                    "test_mae": None,
                    "test_log_r2": None,
                    "test_r2": None,
                    "test_relative_mae": None,
                    "test_log_relative_mae": None,
                }
            log_feature_names = [f"log10_{name}" for name in sanitized_names]
            formula_lin = _format_linear_formula(pipeline, log_feature_names)
        except Exception:
            metrics_lin_raw = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            metrics_lin_bal = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula_lin = None

        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="Linear Regression",
                formula=formula_lin,
                test_log_mae=metrics_lin_raw["test_log_mae"],
                test_log_r2=metrics_lin_raw["test_log_r2"],
                test_mae=metrics_lin_raw["test_mae"],
                test_relative_mae=metrics_lin_raw["test_relative_mae"],
                test_log_relative_mae=metrics_lin_raw["test_log_relative_mae"],
                test_r2=metrics_lin_raw["test_r2"],
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                balanced_test_log_mae=metrics_lin_bal["test_log_mae"],
                balanced_test_log_r2=metrics_lin_bal["test_log_r2"],
                balanced_test_mae=metrics_lin_bal["test_mae"],
                balanced_test_relative_mae=metrics_lin_bal["test_relative_mae"],
                balanced_test_log_relative_mae=metrics_lin_bal["test_log_relative_mae"],
                balanced_test_r2=metrics_lin_bal["test_r2"],
                balanced_test_samples=len(y_test_bal),
            )
        )

    if include_unbalanced:
        results.extend(_unbalanced_result_block())

    # Linear-space baselines on raw (unbalanced) split
    X_train_raw = X.iloc[train_idx]
    X_test_raw_only = X.iloc[test_idx]
    y_train_raw = y.iloc[train_idx]
    y_test_raw_only = y.iloc[test_idx]

    # PySR with squared loss in linear space
    if run_pysr:
        try:
            pysr_lin = PySRRegressor(
                elementwise_loss=linear_loss_expr,
                **(sr_kwargs or {}),
            )
            pysr_lin.fit(X_train_raw, y_train_raw)
            y_pred_raw = pysr_lin.predict(X_test_raw_only)
            metrics_lin = _compute_regression_metrics(y_test_raw_only.values, y_pred_raw, log10_cutoff)
            formula_lin = str(pysr_lin.sympy()) if pysr_lin.equations_ is not None else None
        except Exception:
            metrics_lin = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula_lin = None
        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="PySR (linear raw)",
                formula=formula_lin,
                test_log_mae=metrics_lin["test_log_mae"],
                test_log_r2=metrics_lin["test_log_r2"],
                test_mae=metrics_lin["test_mae"],
                test_relative_mae=metrics_lin["test_relative_mae"],
                test_log_relative_mae=metrics_lin["test_log_relative_mae"],
                test_r2=metrics_lin["test_r2"],
                test_samples=len(y_test_raw_only),
                train_samples=len(y_train_raw),
                target_space="linear",
            )
        )

        # PySR with uniformized target
        try:
            qt = QuantileTransformer(output_distribution="uniform", random_state=random_state)
            y_train_q = qt.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
            y_test_q = qt.transform(y_test_raw_only.values.reshape(-1, 1)).ravel()

            pysr_lin_q = PySRRegressor(
                elementwise_loss=linear_loss_expr,
                **(sr_kwargs or {}),
            )
            pysr_lin_q.fit(X_train_raw, y_train_q)
            y_pred_q = pysr_lin_q.predict(X_test_raw_only)
            y_pred_lin_q = qt.inverse_transform(np.asarray(y_pred_q).reshape(-1, 1)).ravel()
            metrics_lin_q = _compute_regression_metrics(y_test_raw_only.values, y_pred_lin_q, log10_cutoff)
            formula_lin_q = str(pysr_lin_q.sympy()) if pysr_lin_q.equations_ is not None else None
        except Exception:
            metrics_lin_q = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula_lin_q = None
        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="PySR (linear uniform target)",
                formula=formula_lin_q,
                test_log_mae=metrics_lin_q["test_log_mae"],
                test_log_r2=metrics_lin_q["test_log_r2"],
                test_mae=metrics_lin_q["test_mae"],
                test_relative_mae=metrics_lin_q["test_relative_mae"],
                test_log_relative_mae=metrics_lin_q["test_log_relative_mae"],
                test_r2=metrics_lin_q["test_r2"],
                test_samples=len(y_test_raw_only),
                train_samples=len(y_train_raw),
                target_space="linear",
            )
        )

    # Linear regression on raw features/targets
    if run_linreg:
        try:
            pipeline_raw = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ])
            X_train_raw = X.iloc[train_idx]
            X_test_raw_only = X.iloc[test_idx]
            y_train_raw = y.iloc[train_idx]
            y_test_raw_only = y.iloc[test_idx]
            pipeline_raw.fit(X_train_raw.values, y_train_raw.values)
            y_pred_lin = pipeline_raw.predict(X_test_raw_only.values)
            metrics_linreg_raw = _compute_regression_metrics(y_test_raw_only.values, y_pred_lin, log10_cutoff)
            formula_lin_raw = _format_linear_formula(pipeline_raw, sanitized_names)
        except Exception:
            metrics_linreg_raw = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula_lin_raw = None
        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="Linear Regression (linear raw)",
                formula=formula_lin_raw,
                test_log_mae=metrics_linreg_raw["test_log_mae"],
                test_log_r2=metrics_linreg_raw["test_log_r2"],
                test_mae=metrics_linreg_raw["test_mae"],
                test_relative_mae=metrics_linreg_raw["test_relative_mae"],
                test_log_relative_mae=metrics_linreg_raw["test_log_relative_mae"],
                test_r2=metrics_linreg_raw["test_r2"],
                test_samples=len(y_test_raw_only),
                train_samples=len(y_train_raw),
                target_space="linear",
            )
        )

        # Linear regression with uniformized target (QuantileTransformer) to reduce skew
        try:
            qt = QuantileTransformer(output_distribution="uniform", random_state=random_state)
            y_train_q = qt.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
            y_test_q = qt.transform(y_test_raw_only.values.reshape(-1, 1)).ravel()

            pipeline_q = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ])
            pipeline_q.fit(X_train_raw.values, y_train_q)
            y_pred_q = pipeline_q.predict(X_test_raw_only.values)
            # invert back to original scale
            y_pred_lin_q = qt.inverse_transform(y_pred_q.reshape(-1, 1)).ravel()
            metrics_linreg_q = _compute_regression_metrics(y_test_raw_only.values, y_pred_lin_q, log10_cutoff)
            formula_lin_q = _format_linear_formula(pipeline_q, sanitized_names)
        except Exception:
            metrics_linreg_q = {
                "test_log_mae": None,
                "test_mae": None,
                "test_log_r2": None,
                "test_r2": None,
                "test_relative_mae": None,
                "test_log_relative_mae": None,
            }
            formula_lin_q = None
        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="Linear Regression (linear uniform target)",
                formula=formula_lin_q,
                test_log_mae=metrics_linreg_q["test_log_mae"],
                test_log_r2=metrics_linreg_q["test_log_r2"],
                test_mae=metrics_linreg_q["test_mae"],
                test_relative_mae=metrics_linreg_q["test_relative_mae"],
                test_log_relative_mae=metrics_linreg_q["test_log_relative_mae"],
                test_r2=metrics_linreg_q["test_r2"],
                test_samples=len(y_test_raw_only),
                train_samples=len(y_train_raw),
                target_space="linear",
            )
        )

    return results


def format_results_text(results: Sequence[GroupResult]) -> str:
    lines: List[str] = []
    for res in results:
        lines.append(f"Group: {res.group_name}")
        lines.append(f"Markers: {', '.join(res.markers)}")
        lines.append(f"Feature mode: {res.feature_mode}")
        lines.append(f"Model: {res.model_name}")
        lines.append(f"Train samples: {res.train_samples}")
        lines.append(f"Test samples (raw): {res.test_samples}")
        lines.append(f"Test samples (balanced): {res.balanced_test_samples if res.balanced_test_samples is not None else 'N/A'}")
        lines.append(f"Formula: {res.formula if res.formula is not None else 'N/A'}")
        lines.append(
            "Test log MAE: "
            + (f"{res.test_log_mae:.4f}" if res.test_log_mae is not None else "N/A")
        )
        lines.append(
            "Test log R2: "
            + (f"{res.test_log_r2:.4f}" if res.test_log_r2 is not None else "N/A")
        )
        lines.append(
            "Test MAE: "
            + (f"{res.test_mae:.4f}" if res.test_mae is not None else "N/A")
        )
        lines.append(
            "Test relative MAE: "
            + (f"{res.test_relative_mae:.4f}" if res.test_relative_mae is not None else "N/A")
        )
        lines.append(
            "Test log-relative MAE: "
            + (f"{res.test_log_relative_mae:.4f}" if res.test_log_relative_mae is not None else "N/A")
        )
        lines.append(
            "Test R2: "
            + (f"{res.test_r2:.4f}" if res.test_r2 is not None else "N/A")
        )
        lines.append(
            "Balanced test log MAE: "
            + (f"{res.balanced_test_log_mae:.4f}" if res.balanced_test_log_mae is not None else "N/A")
        )
        lines.append(
            "Balanced test log R2: "
            + (f"{res.balanced_test_log_r2:.4f}" if res.balanced_test_log_r2 is not None else "N/A")
        )
        lines.append(
            "Balanced test MAE: "
            + (f"{res.balanced_test_mae:.4f}" if res.balanced_test_mae is not None else "N/A")
        )
        lines.append(
            "Balanced test relative MAE: "
            + (f"{res.balanced_test_relative_mae:.4f}" if res.balanced_test_relative_mae is not None else "N/A")
        )
        lines.append(
            "Balanced test log-relative MAE: "
            + (f"{res.balanced_test_log_relative_mae:.4f}" if res.balanced_test_log_relative_mae is not None else "N/A")
        )
        lines.append(
            "Balanced test R2: "
            + (f"{res.balanced_test_r2:.4f}" if res.balanced_test_r2 is not None else "N/A")
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = output_dir / "reports"
    summary_dir = reports_dir / "summary"
    formulas_dir = reports_dir / "formulas"
    for path in (reports_dir, summary_dir, formulas_dir):
        path.mkdir(parents=True, exist_ok=True)

    seed_everything(args.random_state)
    LOGGER.info("Starting functional group symbolic regression")
    LOGGER.info("Loading dataset from %s", args.dataset)
    data = pd.read_csv(args.dataset)
    if args.dataset_mode == "per_minute":
        if "timepoint" not in data.columns:
            raise ValueError("Per-minute dataset must include a 'timepoint' column.")
        data = data.copy()
        data["timepoint"] = pd.to_numeric(data["timepoint"], errors="coerce")
        data = data[data["timepoint"].notna()]
        data = data[data["timepoint"] <= float(args.per_minute_max_time)]
        LOGGER.info(
            "Per-minute dataset mode enabled: keeping %d rows with timepoint ≤ %.1f",
            len(data),
            args.per_minute_max_time,
        )
    data.columns = data.columns.str.replace("_fit$", "", regex=True)

    required_columns = {"marker", "GFP_bin", "p-ERK1-2_dt"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    LOGGER.info("Dropping rows with missing target derivatives")
    before_rows = len(data)
    data = data.dropna(subset=["p-ERK1-2_dt"])
    LOGGER.info("Retained %d/%d rows after target drop", len(data), before_rows)
    data.loc[
        data["p-ERK1-2_dt"].abs() < 10.0 ** args.log10_cutoff,
        "p-ERK1-2_dt",
    ] = 0.0

    if args.dataset_mode == "per_minute":
        marker_list = sorted(data["marker"].dropna().unique().tolist())
        if not marker_list:
            raise ValueError("No markers available in per-minute dataset.")
        marker_groups = {marker: [marker] for marker in marker_list}
        LOGGER.info(
            "Per-minute mode: treating each marker as its own group (%d markers).",
            len(marker_groups),
        )
    else:
        LOGGER.info("Loading functional group definitions from %s", args.group_definitions_csv)
        marker_groups = load_marker_groups(
            data,
            args.marker_groups_json,
            args.group_definitions_csv,
        )
        if not marker_groups:
            raise ValueError("No marker groups available after intersecting with dataset markers.")
        LOGGER.info("Running on %d marker groups", len(marker_groups))

    models_requested = tuple(dict.fromkeys(args.models))
    run_pysr = "pysr" in models_requested
    run_linreg = "linreg" in models_requested
    if not (run_pysr or run_linreg):
        raise ValueError("At least one regression model must be selected via --models.")

    sr_kwargs: Optional[Dict[str, object]] = None
    if run_pysr:
        sr_kwargs = {
            "niterations": args.max_iterations,
            "population_size": args.population_size,
            "populations": args.populations,
            "maxsize": args.max_size,
            "parsimony": args.parsimony,
            "binary_operators": list(args.binary_operators),
            "unary_operators": list(args.unary_operators),
            "batching": args.batching,
            "annealing": args.annealing,
            "verbosity": args.verbosity,
        }

    all_results: List[GroupResult] = []
    total_modes = len(args.feature_modes)
    for mode_idx, feature_mode in enumerate(args.feature_modes, start=1):
        log_progress("Feature modes", mode_idx, total_modes)
        LOGGER.info("Selecting features for mode '%s'", feature_mode)
        feature_columns = select_features(data, feature_mode, args.gfp_columns)
        if not feature_columns:
            LOGGER.warning("No usable features for mode '%s'. Skipping.", feature_mode)
            continue

        mode_results: List[GroupResult] = []
        total_groups = len(marker_groups)
        LOGGER.info(
            "Evaluating %d marker groups for mode '%s'", total_groups, feature_mode
        )
        for group_idx, (group_name, markers) in enumerate(
            marker_groups.items(), start=1
        ):
            log_progress(f"{feature_mode} groups", group_idx, total_groups)
            results = train_group_models(
                data,
                group_name,
                markers,
                feature_mode,
                feature_columns,
                args.test_size,
                args.random_state,
                args.log10_cutoff,
                args.min_bin_samples,
                args.max_bin_samples,
                args.include_unbalanced_baseline,
                run_pysr=run_pysr,
                run_linreg=run_linreg,
                sr_kwargs=sr_kwargs,
                apply_balancing=not args.disable_balancing,
            )
            if not results:
                LOGGER.info(
                    "Skipping group '%s' for mode '%s' (insufficient data)",
                    group_name,
                    feature_mode,
                )
                continue

            for result in results:
                def _fmt(v: Optional[float]) -> str:
                    return f"{v:.4f}" if v is not None else "N/A"

                LOGGER.info(
                    "%-4s | %-28s | %-17s | train=%5d | test raw=%5d | test bal=%5s | "
                    "logR2 raw/bal=%8s | MAE raw/bal=%8s",
                    feature_mode,
                    group_name[:28],
                    result.model_name,
                    result.train_samples,
                    result.test_samples,
                    str(result.balanced_test_samples) if result.balanced_test_samples is not None else "N/A",
                    f"{_fmt(result.test_log_r2)}/{_fmt(result.balanced_test_log_r2)}",
                    f"{_fmt(result.test_mae)}/{_fmt(result.balanced_test_mae)}",
                )
                mode_results.append(result)

            # Tabular recap per group (both models and all metrics)
            def _fmt_table(v: Optional[float]) -> str:
                return f"{v:>8.4f}" if v is not None else f"{'N/A':>8}"

            table_lines = []
            table_lines.append(f"Mode={feature_mode} | Group={group_name} | markers={len(markers)}")
            table_lines.append(
                "+----------------------------+---------+------------+------------+-----------------+-----------------+-----------------+-------------------+----------------------+"
            )
            table_lines.append(
                "| Model                      | n_train | n_test_raw | n_test_bal |   logR2 raw/bal |    R2 raw/bal   |     MAE raw/bal |  relMAE raw/bal   | logRelMAE raw/bal    |"
            )
            table_lines.append(
                "+----------------------------+---------+------------+------------+-----------------+-----------------+-----------------+-------------------+----------------------+"
            )
            for res in results:
                table_lines.append(
                    "| {model:<26} | {n_train:7d} | {n_test_raw:10d} | {n_test_bal:10} | {r2_raw}/{r2_bal} | {r2_lin_raw}/{r2_lin_bal} | {mae_raw}/{mae_bal} | {rel_raw}/{rel_bal} | {logrel_raw}/{logrel_bal} |".format(
                        model=res.model_name[:26],
                        n_train=res.train_samples,
                        n_test_raw=res.test_samples,
                        n_test_bal=str(res.balanced_test_samples) if res.balanced_test_samples is not None else "N/A",
                        r2_raw=_fmt_table(res.test_log_r2),
                        r2_bal=_fmt_table(res.balanced_test_log_r2),
                        r2_lin_raw=_fmt_table(res.test_r2),
                        r2_lin_bal=_fmt_table(res.balanced_test_r2),
                        mae_raw=_fmt_table(res.test_mae),
                        mae_bal=_fmt_table(res.balanced_test_mae),
                        rel_raw=_fmt_table(res.test_relative_mae),
                        rel_bal=_fmt_table(res.balanced_test_relative_mae),
                        logrel_raw=_fmt_table(res.test_log_relative_mae),
                        logrel_bal=_fmt_table(res.balanced_test_log_relative_mae),
                    )
                )
            table_lines.append(
                "+----------------------------+---------+------------+------------+-----------------+-----------------+-----------------+-------------------+----------------------+"
            )
            LOGGER.info("\n".join(table_lines))

        if not mode_results:
            LOGGER.warning(
                "No successful regressions for feature mode '%s'", feature_mode
            )
            continue

        text_output = format_results_text(mode_results)
        text_path = formulas_dir / f"{feature_mode}.txt"
        text_path.write_text(text_output, encoding="utf-8")
        LOGGER.info("Saved formula report to %s", text_path)
        all_results.extend(mode_results)

    if not all_results:
        LOGGER.error("No successful symbolic regressions were produced")
        raise RuntimeError("No successful symbolic regressions were produced. Check preprocessing settings.")

    summary_df = pd.DataFrame(
        [
            {
                "group_name": res.group_name,
                "markers": list(res.markers),
                "feature_mode": res.feature_mode,
                "model": res.model_name,
                "formula": res.formula,
                "test_log_mae": res.test_log_mae,
                "test_log_r2": res.test_log_r2,
                "test_mae": res.test_mae,
                "test_relative_mae": res.test_relative_mae,
                "test_log_relative_mae": res.test_log_relative_mae,
                "test_r2": res.test_r2,
                "train_samples": res.train_samples,
                "test_samples_raw": res.test_samples,
                "test_samples_balanced": res.balanced_test_samples,
                "balanced_test_log_mae": res.balanced_test_log_mae,
                "balanced_test_log_r2": res.balanced_test_log_r2,
                "balanced_test_mae": res.balanced_test_mae,
                "balanced_test_relative_mae": res.balanced_test_relative_mae,
                "balanced_test_log_relative_mae": res.balanced_test_log_relative_mae,
                "balanced_test_r2": res.balanced_test_r2,
                "target_space": res.target_space,
                "primary_test_r2": res.test_log_r2 if res.target_space == "log" else res.test_r2,
                "primary_balanced_test_r2": (
                    res.balanced_test_log_r2 if res.target_space == "log" else res.balanced_test_r2
                ),
            }
            for res in all_results
        ]
    )
    summary_path = summary_dir / "functional_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote summary CSV to %s", summary_path)
    LOGGER.info("Symbolic regression pipeline completed")


if __name__ == "__main__":
    main()
