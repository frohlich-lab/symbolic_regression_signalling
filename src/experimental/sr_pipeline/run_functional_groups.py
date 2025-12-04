"""Symbolic regression over functional marker groups.

The goal here is intentionally simple: for each marker (or group of markers),
train one or more regressors on the preprocessed trajectories, record the
metrics, and emit plots/reports that downstream scripts can use. Preprocessing
is handled upstream; this file only touches training/evaluation.
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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.seeding import seed_everything

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

RUN_SEED: Optional[int] = None

def log_progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        if RUN_SEED is not None:
            LOGGER.info("[seed=%s] %s [%d]", RUN_SEED, stage, current)
        else:
            LOGGER.info("%s [%d]", stage, current)
        return
    percent = (current / total) * 100.0
    if RUN_SEED is not None:
        LOGGER.info("[seed=%s] %s [%d/%d | %.1f%%]", RUN_SEED, stage, current, total, percent)
    else:
        LOGGER.info("%s [%d/%d | %.1f%%]", stage, current, total, percent)


@dataclass
class GroupResult:
    group_name: str
    markers: Tuple[str, ...]
    feature_mode: str
    model_name: str
    formula: Optional[str]
    test_relative_mae: Optional[float]
    test_r2: Optional[float]
    test_samples: int
    train_samples: int
    train_relative_mae: Optional[float] = None
    train_r2: Optional[float] = None
    target_space: Literal["linear"] = "linear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PySR over functional marker groups.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the CSV file produced from the perturbation trajectories.",
    )
    parser.add_argument(
        "--per-minute-dataset",
        type=Path,
        default=None,
        help="Optional per-minute CSV. When provided, both snapshot (dataset) and per-minute runs are executed sequentially.",
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
        default="per_minute",
        help="Source dataset format. Default per-minute to mirror notebook strategy.",
    )
    parser.add_argument(
        "--per-minute-max-time",
        type=float,
        default=60.0,
        help="When --dataset-mode per_minute is used, keep rows with timepoint ≤ this value (default: 60).",
    )
    parser.add_argument(
        "--per-minute-sampling-strategy",
        choices=("max_time", "early_plus_sparse_late"),
        default="early_plus_sparse_late",
        help="Subsample per-minute data for training. Matches notebook default of early_plus_sparse_late.",
    )
    parser.add_argument(
        "--late-sample-window",
        nargs=2,
        type=float,
        default=(30.0, 60.0),
        metavar=("START", "END"),
        help="Window used by early_plus_sparse_late strategy (keep all ≤ START; sample within START-END).",
    )
    parser.add_argument(
        "--late-sample-points",
        type=int,
        default=15,
        help="Number of late-window points to keep per marker for early_plus_sparse_late.",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=(0.0, 5.0, 10.0, 15.0, 30.0, 60.0),
        help="Measured timepoints; metrics on per-minute runs are restricted to these if present.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of seeds; when provided, the pipeline runs once per seed and aggregates summaries.",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="*",
        choices=("all",),
        default=("all",),
        help="Feature modes to run. Only 'all' is supported to align with the notebook strategy.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("pysr", "linreg"),
        default=("pysr", "linreg"),
        help="Regression models to evaluate (default: pysr and linreg).",
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
        default=400,
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


def _compute_regression_metrics(
    y_true_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
) -> Dict[str, Optional[float]]:
    def _rel_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite):
            return None
        y_true_f = y_true[finite]
        y_pred_f = y_pred[finite]
        denom = np.maximum(np.abs(y_true_f), 1e-2)
        return float(np.mean(np.abs(y_pred_f - y_true_f) / denom))

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.sum(finite) < 2:
            return None
        y_true_f = y_true[finite]
        y_pred_f = y_pred[finite]
        if len(np.unique(y_pred_f)) <= 1:
            return None
        return float(r2_score(y_true_f, y_pred_f))

    return {
        "train_r2": _r2(y_true_train, y_pred_train),
        "test_r2": _r2(y_true_test, y_pred_test),
        "train_relative_mae": _rel_mae(y_true_train, y_pred_train),
        "test_relative_mae": _rel_mae(y_true_test, y_pred_test),
    }


def _binwise_r2(
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
) -> Optional[float]:
    """Compute mean R2 across GFP_bin trajectories (per-bin R2, then average)."""
    if "GFP_bin" not in frame.columns:
        return None
    r2_vals: List[float] = []
    for _, g in frame.groupby("GFP_bin"):
        y_true = pd.to_numeric(g[target_col], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if dataset_mode == "per_minute" and "timepoint" in g.columns:
            measured_mask = np.isin(g["timepoint"].to_numpy(), measured_timepoints)
            mask &= measured_mask
        if mask.sum() < 2:
            continue
        y_true_f = y_true[mask]
        y_pred_f = y_pred[mask]
        if len(np.unique(y_pred_f)) <= 1:
            continue
        den = np.sum((y_true_f - np.mean(y_true_f)) ** 2)
        if den <= 0:
            continue
        # Clamp negatives to zero to avoid penalizing below-baseline fits when averaging
        r2_vals.append(max(0.0, 1.0 - np.sum((y_true_f - y_pred_f) ** 2) / den))
    return float(np.mean(r2_vals)) if r2_vals else None


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


def select_features(df: pd.DataFrame, feature_mode: str) -> List[str]:
    candidate_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    if feature_mode == "all":
        return candidate_columns
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def choose_bin_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[Optional[set], Optional[set]]:
    bins = df["GFP_bin"].dropna().unique()
    if len(bins) < 2:
        return None, None
    rng = np.random.default_rng(random_state)
    shuffled = list(bins)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    n_test = min(len(shuffled) - 1, n_test)
    test_bins = set(shuffled[:n_test])
    train_bins = set(shuffled[n_test:])
    return train_bins, test_bins


def apply_per_minute_sampling(
    data: pd.DataFrame,
    strategy: str,
    max_time: float,
    late_window: Tuple[float, float],
    late_points: int,
    random_state: int,
) -> pd.DataFrame:
    if "timepoint" not in data.columns:
        raise ValueError("Per-minute dataset must include a 'timepoint' column.")
    df = data.copy()
    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    df = df[df["timepoint"].notna()]
    if strategy == "max_time":
        return df[df["timepoint"] <= max_time]
    if strategy == "early_plus_sparse_late":
        early_cutoff, late_max = late_window
        rng = np.random.default_rng(random_state)
        sampled_parts: List[pd.DataFrame] = []
        group_keys = ["marker"]
        if "GFP_bin" in df.columns:
            group_keys.append("GFP_bin")
        for _, group in df.groupby(group_keys):
            early = group[group["timepoint"] <= early_cutoff]
            late_candidates = group[
                (group["timepoint"] > early_cutoff) & (group["timepoint"] <= late_max)
            ]
            if len(late_candidates) > late_points:
                keep_idx = rng.choice(
                    late_candidates.index.to_numpy(), size=late_points, replace=False
                )
                late = late_candidates.loc[keep_idx]
            else:
                late = late_candidates
            sampled_parts.append(pd.concat([early, late], axis=0))
        return pd.concat(sampled_parts, axis=0) if sampled_parts else df.iloc[:0]
    raise ValueError(f"Unknown per-minute sampling strategy: {strategy}")


def train_group_models(
    data: pd.DataFrame,
    group_label: str,
    markers: Sequence[str],
    feature_mode: str,
    feature_columns: List[str],
    test_size: float,
    random_state: int,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
    *,
    run_pysr: bool,
    run_linreg: bool,
    sr_kwargs: Optional[Dict[str, object]] = None,
    trajectory_records: Optional[List[Dict[str, object]]] = None,
    log_prefix: str = "",
) -> List[GroupResult]:
    subset = data[data["marker"].isin(markers)].copy()
    if subset.empty:
        if log_prefix:
            LOGGER.info("%s: no rows for marker(s) %s", log_prefix, markers)
        return []

    subset = subset.dropna(subset=feature_columns + ["p-ERK1-2_dt"])
    if subset.empty:
        if log_prefix:
            LOGGER.info("%s: all rows dropped due to missing features/target", log_prefix)
        return []

    X = subset[feature_columns].copy()
    y = subset["p-ERK1-2_dt"].astype(float)

    if len(y) < 5 or X.shape[1] == 0:
        if log_prefix:
            LOGGER.info("%s: not enough samples (%d) or features (%d)", log_prefix, len(y), X.shape[1])
        return []

    sanitized_names = sanitize_feature_names(X.columns)
    X.columns = sanitized_names

    train_bins, test_bins = choose_bin_split(subset, test_size=test_size, random_state=random_state)
    if not train_bins or not test_bins:
        if log_prefix:
            LOGGER.info("%s: cannot split bins into train/test; skipping", log_prefix)
        return []
    subset = subset[subset["GFP_bin"].isin(train_bins | test_bins)].copy()
    train_df = subset[subset["GFP_bin"].isin(train_bins)].copy()
    test_df_raw = subset[subset["GFP_bin"].isin(test_bins)].copy()

    def _make_xy(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        x_local = frame[feature_columns].copy()
        x_local.columns = sanitized_names
        y_local = frame["p-ERK1-2_dt"].astype(float)
        return x_local, y_local

    X_train, y_train = _make_xy(train_df)
    X_test_raw, y_test_raw = _make_xy(test_df_raw)

    if len(y_train) == 0 or len(y_test_raw) == 0:
        return []

    results: List[GroupResult] = []

    def _record_preds(frame: pd.DataFrame, preds: np.ndarray, model_name: str, phase: str) -> None:
        if trajectory_records is None:
            return
        safe_time = frame["timepoint"] if "timepoint" in frame.columns else pd.Series([np.nan] * len(frame))
        safe_gfp = frame["GFP_bin"] if "GFP_bin" in frame.columns else pd.Series([np.nan] * len(frame))
        safe_pe = frame["p-ERK1-2"] if "p-ERK1-2" in frame.columns else pd.Series([np.nan] * len(frame))
        for idx, (_t, _gfp, _y_true, _y_pred, _pe) in enumerate(
            zip(
                safe_time.to_numpy(),
                safe_gfp.to_numpy(),
                frame["p-ERK1-2_dt"].to_numpy(),
                preds,
                safe_pe.to_numpy(),
            )
        ):
            trajectory_records.append(
                {
                    "marker": group_label,
                    "model": model_name,
                    "dataset_mode": dataset_mode,
                    "phase": phase,
                    "timepoint": _t,
                    "GFP_bin": _gfp,
                    "p-ERK1-2_dt_true": _y_true,
                    "p-ERK1-2_dt_pred": _y_pred,
                    "p-ERK1-2": _pe,
                }
            )

    if run_pysr:
        # Import lazily to avoid triggering Julia init when this module is imported for metrics-only scripts.
        from pysr import PySRRegressor

        pysr_model = PySRRegressor(**(sr_kwargs or {}))

        try:
            pysr_model.fit(X_train, y_train)

            y_pred_train = pysr_model.predict(X_train)
            y_true_train_eval = y_train.values
            y_pred_train_eval = y_pred_train
            _record_preds(train_df, y_pred_train, "PySR", "train")

            if len(y_test_raw) > 0:
                y_pred_test_raw = pysr_model.predict(X_test_raw)
                y_true_eval = y_test_raw.values
                y_pred_eval = y_pred_test_raw
                _record_preds(test_df_raw, y_pred_test_raw, "PySR", "test")
                if dataset_mode == "per_minute":
                    measured_mask = np.isin(test_df_raw["timepoint"].to_numpy(), measured_timepoints)
                    if measured_mask.any():
                        y_true_eval = y_true_eval[measured_mask]
                        y_pred_eval = y_pred_eval[measured_mask]
                metrics_raw = _compute_regression_metrics(
                    y_true_train_eval,
                    y_pred_train_eval,
                    y_true_eval,
                    y_pred_eval,
                )
                train_df["__y_pred__"] = y_pred_train
                test_df_raw["__y_pred__"] = y_pred_test_raw
                metrics_raw["train_r2"] = _binwise_r2(
                    train_df,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                )
                metrics_raw["test_r2"] = _binwise_r2(
                    test_df_raw,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                )
            else:
                metrics_raw = _compute_regression_metrics(
                    y_true_train_eval,
                    y_pred_train_eval,
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )
            formula = str(pysr_model.sympy()) if pysr_model.equations_ is not None else None
        except Exception:
            metrics_raw = {
                "train_r2": None,
                "test_r2": None,
                "train_relative_mae": None,
                "test_relative_mae": None,
            }
            formula = None

        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="PySR",
                formula=formula,
                test_relative_mae=metrics_raw["test_relative_mae"],
                test_r2=metrics_raw["test_r2"],
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                train_relative_mae=metrics_raw["train_relative_mae"],
                train_r2=metrics_raw["train_r2"],
            )
        )

    if run_linreg:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ])
        try:
            pipeline.fit(X_train.values, y_train.values)
            y_pred_train = pipeline.predict(X_train.values)
            _record_preds(train_df, y_pred_train, "Linear Regression", "train")
            if len(y_test_raw) > 0:
                y_pred_lin = pipeline.predict(X_test_raw.values)
                y_true_eval = y_test_raw.values
                y_pred_eval = y_pred_lin
                _record_preds(test_df_raw, y_pred_lin, "Linear Regression", "test")
                if dataset_mode == "per_minute":
                    measured_mask = np.isin(test_df_raw["timepoint"].to_numpy(), measured_timepoints)
                    if measured_mask.any():
                        y_true_eval = y_true_eval[measured_mask]
                        y_pred_eval = y_pred_eval[measured_mask]
                metrics_lin_raw = _compute_regression_metrics(
                    y_train.values,
                    y_pred_train,
                    y_true_eval,
                    y_pred_eval,
                )
                train_df["__y_pred__"] = y_pred_train
                test_df_raw["__y_pred__"] = y_pred_lin
                metrics_lin_raw["train_r2"] = _binwise_r2(
                    train_df,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                )
                metrics_lin_raw["test_r2"] = _binwise_r2(
                    test_df_raw,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                )
            else:
                metrics_lin_raw = _compute_regression_metrics(
                    y_train.values,
                    y_pred_train,
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )
            formula_lin = _format_linear_formula(pipeline, sanitized_names)
        except Exception:
            metrics_lin_raw = {
                "train_r2": None,
                "test_r2": None,
                "train_relative_mae": None,
                "test_relative_mae": None,
            }
            formula_lin = None

        results.append(
            GroupResult(
                group_name=group_label,
                markers=tuple(markers),
                feature_mode=feature_mode,
                model_name="Linear Regression",
                formula=formula_lin,
                test_relative_mae=metrics_lin_raw["test_relative_mae"],
                test_r2=metrics_lin_raw["test_r2"],
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                train_relative_mae=metrics_lin_raw["train_relative_mae"],
                train_r2=metrics_lin_raw["train_r2"],
            )
        )
    return results


def format_results_text(results: Sequence[GroupResult]) -> str:
    lines: List[str] = []
    for res in results:
        lines.append(f"Group: {res.group_name}")
        lines.append(f"Markers: {', '.join(res.markers)}")
        if hasattr(res, "dataset_mode"):
            lines.append(f"Dataset mode: {res.dataset_mode}")
        lines.append(f"Feature mode: {res.feature_mode}")
        lines.append(f"Model: {res.model_name}")
        lines.append(f"Train samples: {res.train_samples}")
        lines.append(f"Test samples (raw): {res.test_samples}")
        lines.append(f"Formula: {res.formula if res.formula is not None else 'N/A'}")
        lines.append(
            "Train R2: "
            + (f"{res.train_r2:.4f}" if res.train_r2 is not None else "N/A")
        )
        lines.append(
            "Train relative MAE: "
            + (f"{res.train_relative_mae:.4f}" if res.train_relative_mae is not None else "N/A")
        )
        lines.append(
            "Test R2: "
            + (f"{res.test_r2:.4f}" if res.test_r2 is not None else "N/A")
        )
        lines.append(
            "Test relative MAE: "
            + (f"{res.test_relative_mae:.4f}" if res.test_relative_mae is not None else "N/A")
        )
        lines.append("")
    return "\n".join(lines)


def run_pipeline(args: argparse.Namespace) -> Path:
    global RUN_SEED
    RUN_SEED = args.random_state
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Allow callers to point directly at a seed reports directory (e.g., .../reports/seed_42).
    if output_dir.name.startswith("seed_") and output_dir.parent.name == "reports":
        reports_dir = output_dir
    elif output_dir.name == "reports":
        reports_dir = output_dir
    else:
        reports_dir = output_dir / "reports"
    summary_dir = reports_dir / "summary"
    formulas_dir = reports_dir / "formulas"
    metrics_dir = reports_dir / "metrics"
    for path in (reports_dir, summary_dir, formulas_dir):
        path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.random_state)
    LOGGER.info(
        "Starting functional group symbolic regression | random_state=%s | dataset_mode=%s | feature_modes=%s",
        args.random_state,
        args.dataset_mode,
        ", ".join(args.feature_modes),
    )

    def _prepare_dataset(csv_path: Path, mode: str) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset for mode {mode} not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Strip _fit suffixes so training/plotting use the same names everywhere.
        df.columns = df.columns.str.replace("_fit$", "", regex=True)
        df["p-ERK1-2_dt"] = pd.to_numeric(df.get("p-ERK1-2_dt"), errors="coerce")
        df["GFP_bin"] = pd.to_numeric(df.get("GFP_bin"), errors="coerce")
        if "timepoint" in df.columns:
            df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
        return df

    # Always process snapshot first, then per-minute (if provided) for consistent logging/output ordering.
    run_configs: List[Tuple[str, Path]] = []
    if args.per_minute_dataset is not None:
        run_configs = [("snapshot", args.dataset), ("per_minute", args.per_minute_dataset)]
    else:
        run_configs = [(args.dataset_mode, args.dataset)]

    all_results: List[GroupResult] = []
    trajectories_by_mode: Dict[str, List[Dict[str, object]]] = {"snapshot": [], "per_minute": []}

    for run_mode, dataset_path in run_configs:
        LOGGER.info("Loading dataset (%s) from %s", run_mode, dataset_path)
        data = _prepare_dataset(dataset_path, run_mode)
        if run_mode == "per_minute":
            data = apply_per_minute_sampling(
                data,
                strategy=args.per_minute_sampling_strategy,
                max_time=args.per_minute_max_time,
                late_window=tuple(args.late_sample_window),
                late_points=args.late_sample_points,
                random_state=args.random_state,
            )
            LOGGER.info(
                "Per-minute dataset mode enabled: strategy=%s | rows=%d",
                args.per_minute_sampling_strategy,
                len(data),
            )
        data.columns = data.columns.str.replace("_fit$", "", regex=True)

        required_columns = {"marker", "GFP_bin", "p-ERK1-2_dt"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        if "marker" in data.columns:
            data["marker"] = data["marker"].astype(str)

        LOGGER.info("Dropping rows with missing target derivatives")
        before_rows = len(data)
        data = data.dropna(subset=["p-ERK1-2_dt"])
        LOGGER.info("Retained %d/%d rows after target drop", len(data), before_rows)

        marker_list = sorted(data["marker"].dropna().unique().tolist())
        if not marker_list:
            LOGGER.warning("No markers available after preprocessing for mode %s.", run_mode)
            continue
        marker_groups = {marker: [marker] for marker in marker_list}
        LOGGER.info(
            "Running per-marker regression (%d markers) for mode %s (group definitions ignored).",
            len(marker_groups),
            run_mode,
        )

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

        total_modes = len(args.feature_modes)
        for mode_idx, feature_mode in enumerate(args.feature_modes, start=1):
            log_progress(f"{run_mode} feature modes", mode_idx, total_modes)
            LOGGER.info("Selecting features for mode '%s' (%s)", feature_mode, run_mode)
            feature_columns = select_features(data, feature_mode)
            if not feature_columns:
                LOGGER.warning("No usable features for mode '%s'. Skipping.", feature_mode)
                continue

            mode_results: List[GroupResult] = []
            total_groups = len(marker_groups)
            LOGGER.info(
                "Evaluating %d markers for mode '%s' (%s)", total_groups, feature_mode, run_mode
            )
            for group_idx, (group_name, markers) in enumerate(
                marker_groups.items(), start=1
            ):
                log_progress(f"{run_mode}/{feature_mode}", group_idx, total_groups)
                results = train_group_models(
                    data,
                    group_name,
                    markers,
                    feature_mode,
                    feature_columns,
                    args.test_size,
                    args.random_state,
                    args.measured_timepoints,
                    run_mode,
                    run_pysr=run_pysr,
                    run_linreg=run_linreg,
                    sr_kwargs=sr_kwargs,
                    trajectory_records=trajectories_by_mode[run_mode],
                    log_prefix=f"{run_mode}/{feature_mode}/{group_name}",
                )
                if not results:
                    LOGGER.info(
                        "Skipping marker '%s' for mode '%s' (%s) (insufficient data)",
                        group_name,
                        feature_mode,
                        run_mode,
                    )
                    continue

                # Structured block per marker
                block_lines: List[str] = []
                prefix = f"[seed={RUN_SEED}] " if RUN_SEED is not None else ""
                block_lines.append(f"{prefix}[{run_mode}/{feature_mode}] Marker={group_name}")
                for result in results:
                    setattr(result, "dataset_mode", run_mode)
                    mode_results.append(result)
                    block_lines.append(f"  Model: {result.model_name}")
                    block_lines.append(f"    Train samples: {result.train_samples}")
                    block_lines.append(f"    Test samples : {result.test_samples}")
                    block_lines.append(
                        f"    Train R2     : {result.train_r2:.4f}" if result.train_r2 is not None else "    Train R2     : N/A"
                    )
                    block_lines.append(
                        f"    Test R2      : {result.test_r2:.4f}" if result.test_r2 is not None else "    Test R2      : N/A"
                    )
                    block_lines.append(
                        f"    Train relMAE : {result.train_relative_mae:.4f}"
                        if result.train_relative_mae is not None
                        else "    Train relMAE : N/A"
                    )
                    block_lines.append(
                        f"    Test relMAE  : {result.test_relative_mae:.4f}"
                        if result.test_relative_mae is not None
                        else "    Test relMAE  : N/A"
                    )
                LOGGER.info("\n".join(block_lines))

            if not mode_results:
                LOGGER.warning(
                    "No successful regressions for feature mode '%s' (%s)", feature_mode, run_mode
                )
                continue

            text_output = format_results_text(mode_results)
            text_path = formulas_dir / f"{feature_mode}_{run_mode}.txt"
            text_path.write_text(text_output, encoding="utf-8")
            LOGGER.info("Saved formula report to %s", text_path)
            all_results.extend(mode_results)

        # Persist trajectories per mode after processing feature modes
        if trajectories_by_mode.get(run_mode):
            traj_df = pd.DataFrame(trajectories_by_mode[run_mode])
            traj_out = metrics_dir / f"predicted_trajectories_{run_mode}.csv"
            traj_df.to_csv(traj_out, index=False)
            LOGGER.info("Saved predicted trajectories to %s", traj_out)

    if not all_results:
        LOGGER.error("No successful symbolic regressions were produced")
        raise RuntimeError("No successful symbolic regressions were produced. Check preprocessing settings.")

    summary_df = pd.DataFrame(
        [
            {
                "group_name": res.group_name,
                "markers": list(res.markers),
                "feature_mode": res.feature_mode,
                "dataset_mode": getattr(res, "dataset_mode", None),
                "model": res.model_name,
                "formula": res.formula,
                "test_relative_mae": res.test_relative_mae,
                "test_log_relative_mae": res.test_relative_mae,  # legacy compatibility
                "test_mae": None,
                "test_log_mae": None,
                "test_r2": res.test_r2,
                "test_log_r2": res.test_r2,  # legacy compatibility
                "train_relative_mae": res.train_relative_mae,
                "train_log_relative_mae": res.train_relative_mae,
                "train_mae": None,
                "train_log_mae": None,
                "train_r2": res.train_r2,
                "train_log_r2": res.train_r2,
                "train_samples": res.train_samples,
                "test_samples_raw": res.test_samples,
                "target_space": res.target_space,
                "primary_test_r2": res.test_r2,
            }
            for res in all_results
        ]
    )
    summary_path = summary_dir / "functional_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote summary CSV to %s", summary_path)
    LOGGER.info("Symbolic regression pipeline completed | random_state=%s", args.random_state)
    return summary_path


def main() -> None:
    args = parse_args()
    # If no seeds provided, run three seeds: base random_state and the next two ints.
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.random_state, args.random_state + 1, args.random_state + 2]
    LOGGER.info("Running seeds in order: %s", seeds)
    summary_paths: List[Path] = []
    for seed in seeds:
        seed_dir = args.output_dir / "reports" / f"seed_{seed}"
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_state = seed
        seed_args.output_dir = seed_dir
        summary_paths.append(run_pipeline(seed_args))

    if len(seeds) > 1:
        combined_dir = args.output_dir / "summary"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "functional_group_summary_all_seeds.csv"
        frames: List[pd.DataFrame] = []
        for seed, path in zip(seeds, summary_paths):
            df_seed = pd.read_csv(path)
            df_seed["seed"] = seed
            frames.append(df_seed)
        if frames:
            combined_df = pd.concat(frames, ignore_index=True)
            combined_df.to_csv(combined_path, index=False)
            LOGGER.info("Wrote combined multi-seed summary to %s", combined_path)

            keys = [c for c in ["group_name", "dataset_mode", "feature_mode", "model"] if c in combined_df.columns]
            numeric_cols = [c for c in combined_df.select_dtypes(include=[np.number]).columns if c not in ("seed",)]
            agg_mean = combined_df.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()
            if "formula" in combined_df.columns:
                formulas = combined_df.groupby(keys, dropna=False)["formula"].first().reset_index()
                agg_mean = agg_mean.merge(formulas, on=keys, how="left")
            # Persist mean-over-seeds summary as canonical for metrics.
            canonical_summary = args.output_dir / "reports" / "summary" / "functional_group_summary.csv"
            canonical_summary.parent.mkdir(parents=True, exist_ok=True)
            agg_mean.to_csv(canonical_summary, index=False)
            LOGGER.info("Wrote mean-over-seeds summary to %s", canonical_summary)

            # Also expose a single-seed summary (first seed) for trajectory-dependent plots.
            seed_summary_path = args.output_dir / "reports" / "summary" / "functional_group_summary_seed.csv"
            pd.read_csv(summary_paths[0]).to_csv(seed_summary_path, index=False)
            LOGGER.info("Saved first-seed summary to %s", seed_summary_path)

            # Combine formula reports across seeds into canonical paths
            formulas_dir = args.output_dir / "reports" / "formulas"
            formulas_dir.mkdir(parents=True, exist_ok=True)
            for mode in ("snapshot", "per_minute"):
                combined_formula_lines: List[str] = []
                for seed in seeds:
                    seed_file = args.output_dir / "reports" / f"seed_{seed}" / "formulas" / f"all_{mode}.txt"
                    if seed_file.exists():
                        combined_formula_lines.append(f"# seed={seed}\\n")
                        combined_formula_lines.extend(seed_file.read_text().splitlines())
                        combined_formula_lines.append("")
                if combined_formula_lines:
                    out_formula = formulas_dir / f"all_{mode}.txt"
                    out_formula.write_text("\\n".join(combined_formula_lines))
                    LOGGER.info("Wrote combined formula report to %s", out_formula)
                else:
                    LOGGER.warning("No formula files found to combine for mode %s", mode)
    else:
        # Single seed: ensure canonical files point to the seed outputs
        seed_summary = summary_paths[0]
        canonical_summary = args.output_dir / "reports" / "summary" / "functional_group_summary.csv"
        canonical_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(seed_summary).to_csv(canonical_summary, index=False)
        seed_summary_path = args.output_dir / "reports" / "summary" / "functional_group_summary_seed.csv"
        pd.read_csv(seed_summary).to_csv(seed_summary_path, index=False)
        formulas_dir = args.output_dir / "reports" / "formulas"
        formulas_dir.mkdir(parents=True, exist_ok=True)
        for mode in ("snapshot", "per_minute"):
            seed_formula = args.output_dir / "reports" / f"seed_{seeds[0]}" / "formulas" / f"all_{mode}.txt"
            if seed_formula.exists():
                out_formula = formulas_dir / f"all_{mode}.txt"
                out_formula.write_text(seed_formula.read_text())
                LOGGER.info("Copied formula report to %s", out_formula)


if __name__ == "__main__":
    main()
