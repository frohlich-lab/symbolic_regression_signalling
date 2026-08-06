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
import re
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import lsq_linear

from experimental.sr_pipeline.metrics import binwise_r2, binwise_r2_stats
from experimental.sr_pipeline.seeding import canonicalize_seeds, seed_all

# Columns that should not be passed to the regression model as inputs.
EXCLUDE_COLUMNS = {"p-ERK1-2_dt", "p-MEK1-2_dt", "marker", "timepoint", "GFP_bin"}
# Relative MAE denominator floor; tuned for ~0.1 percentage-point precision.
REL_MAE_EPS = 1e-1


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


def _is_runs_layout(base: Path) -> bool:
    """
    Return True when the output base looks like the new experimental layout:
    data/experimental/runs with aggregated/ and seeds/ children.
    """
    return base.name == "runs" or (base / "seeds").exists() or (base / "aggregated").exists()


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


def _run_module(cmd: List[str], description: str) -> None:
    """Run a Python module as a subprocess with logging."""
    try:
        LOGGER.info("Running %s", description)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        LOGGER.warning("Failed to run %s: %s", description, exc)


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
    # per-bin stats (equal weight per bin)
    test_r2_bin_mean: Optional[float] = None
    test_r2_bin_median: Optional[float] = None
    test_r2_bin_count: Optional[int] = None
    train_relative_mae: Optional[float] = None
    train_r2: Optional[float] = None
    train_r2_bin_mean: Optional[float] = None
    train_r2_bin_median: Optional[float] = None
    train_r2_bin_count: Optional[int] = None
    best_loss: Optional[float] = None
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
        default=700,
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
        "--linear-stability-penalty",
        type=float,
        default=1000.0,
        help="Penalty coefficient for positive slope in linear p direction.",
    )
    parser.add_argument(
        "--inverse-stability-penalty",
        type=float,
        default=1000.0,
        help="Penalty coefficient for negative slope in linear 1/p direction.",
    )
    parser.add_argument(
        "--disable-linear-stability-penalty",
        action="store_true",
        help="Set the linear-p stability penalty coefficient to 0.",
    )
    parser.add_argument(
        "--disable-inverse-stability-penalty",
        action="store_true",
        help="Set the inverse-p stability penalty coefficient to 0.",
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
        denom = np.maximum(np.abs(y_true_f), REL_MAE_EPS)
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

    # Dynamic pERK feature index for the custom PySR loss (Julia is 1-based).
    perk_name = sanitize_feature_names(["p_ERK1_2"])[0]
    perk_idx_1b: Optional[int] = None
    if perk_name in sanitized_names:
        perk_idx_1b = sanitized_names.index(perk_name) + 1
    else:
        # fallback: pick the first ERK-like feature if p_ERK1_2 is missing
        fallback = next((n for n in sanitized_names if "ERK" in n.upper()), None)
        if fallback is not None:
            perk_idx_1b = sanitized_names.index(fallback) + 1

    gfp_name = sanitize_feature_names(["GFP"])[0]
    gfp_idx_1b: Optional[int] = None
    if gfp_name in sanitized_names:
        gfp_idx_1b = sanitized_names.index(gfp_name) + 1
    else:
        fallback = next((n for n in sanitized_names if "GFP" in n.upper()), None)
        if fallback is not None:
            gfp_idx_1b = sanitized_names.index(fallback) + 1

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
                    "seed": RUN_SEED,
                }
            )

    if run_pysr:
        # Import lazily to avoid triggering Julia init when this module is imported for metrics-only scripts.
        from pysr import PySRRegressor

        sr_kwargs_local = dict(sr_kwargs or {})
        if perk_idx_1b is not None and "loss_function" in sr_kwargs_local:
            loss_fn = sr_kwargs_local["loss_function"]
            if isinstance(loss_fn, str) and "REQUIRED_IDX" in loss_fn:
                sr_kwargs_local["loss_function"] = re.sub(
                    r"REQUIRED_IDX\s*=\s*\d+",
                    f"REQUIRED_IDX = {perk_idx_1b}",
                    loss_fn,
                )
        elif "loss_function" in (sr_kwargs or {}) and log_prefix:
            LOGGER.warning(
                "%s: could not determine pERK feature index; leaving REQUIRED_IDX unchanged in loss_function",
                log_prefix,
            )

        if gfp_idx_1b is not None and "loss_function" in sr_kwargs_local:
            loss_fn = sr_kwargs_local["loss_function"]
            if isinstance(loss_fn, str) and "REQUIRED_IDX_GFP" in loss_fn:
                sr_kwargs_local["loss_function"] = re.sub(
                    r"REQUIRED_IDX_GFP\s*=\s*\d+",
                    f"REQUIRED_IDX_GFP = {gfp_idx_1b}",
                    loss_fn,
                )
        elif "loss_function" in (sr_kwargs or {}) and log_prefix:
            LOGGER.warning(
                "%s: could not determine GFP feature index; leaving REQUIRED_IDX_GFP unchanged in loss_function",
                log_prefix,
            )

        pysr_model = PySRRegressor(**sr_kwargs_local)
        best_loss: Optional[float] = None

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
                train_stats = binwise_r2_stats(
                    train_df,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                    clamp_negative=True,
                )
                test_stats = binwise_r2_stats(
                    test_df_raw,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                    clamp_negative=True,
                )
                if train_stats:
                    metrics_raw["train_r2"] = train_stats["mean"]
                    metrics_raw["train_r2_bin_mean"] = train_stats["mean"]
                    metrics_raw["train_r2_bin_median"] = train_stats["median"]
                    metrics_raw["train_r2_bin_count"] = train_stats["count"]
                if test_stats:
                    metrics_raw["test_r2"] = test_stats["mean"]
                    metrics_raw["test_r2_bin_mean"] = test_stats["mean"]
                    metrics_raw["test_r2_bin_median"] = test_stats["median"]
                    metrics_raw["test_r2_bin_count"] = test_stats["count"]
            else:
                metrics_raw = _compute_regression_metrics(
                    y_true_train_eval,
                    y_pred_train_eval,
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )
            formula = str(pysr_model.sympy()) if pysr_model.equations_ is not None else None
            equations = getattr(pysr_model, "equations_", None)
            if equations is not None and len(equations) > 0:
                try:
                    best_loss = float(equations.iloc[0]["loss"])
                except Exception:
                    best_loss = None
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
                test_r2_bin_mean=metrics_raw.get("test_r2_bin_mean"),
                test_r2_bin_median=metrics_raw.get("test_r2_bin_median"),
                test_r2_bin_count=metrics_raw.get("test_r2_bin_count"),
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                train_relative_mae=metrics_raw["train_relative_mae"],
                train_r2=metrics_raw["train_r2"],
                train_r2_bin_mean=metrics_raw.get("train_r2_bin_mean"),
                train_r2_bin_median=metrics_raw.get("train_r2_bin_median"),
                train_r2_bin_count=metrics_raw.get("train_r2_bin_count"),
                best_loss=best_loss,
            )
        )

    if run_linreg:
        try:
            # Standardise features then fit constrained least squares with pERK coefficient <= 0.
            scaler = StandardScaler()
            X_train_np = X_train.values
            X_test_np = X_test_raw.values if len(y_test_raw) > 0 else np.empty((0, X_train_np.shape[1]))
            Xtr = scaler.fit_transform(X_train_np)
            Xte = scaler.transform(X_test_np) if len(y_test_raw) > 0 else X_test_np

            perk_name = sanitize_feature_names(["p_ERK1_2"])[0]
            if perk_name not in sanitized_names:
                # fallback: first ERK-like feature
                perk_name = next((n for n in sanitized_names if "ERK" in n), perk_name)
            if perk_name not in sanitized_names:
                raise RuntimeError(f"Could not find pERK feature in LR features. perk_name={perk_name}")
            perk_idx = sanitized_names.index(perk_name)

            Xtr_aug = np.column_stack([Xtr, np.ones(len(y_train))])
            lb = np.full(Xtr_aug.shape[1], -np.inf)
            ub = np.full(Xtr_aug.shape[1], np.inf)
            ub[perk_idx] = 0.0  # enforce non-positive pERK coefficient

            res = lsq_linear(Xtr_aug, y_train.values, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
            coef_scaled = res.x[:-1]
            intercept_scaled = float(res.x[-1])

            y_pred_train = Xtr @ coef_scaled + intercept_scaled
            y_pred_test = Xte @ coef_scaled + intercept_scaled if len(y_test_raw) > 0 else np.array([])

            # Unscale coefficients for readability/formula
            scale = np.where(np.asarray(scaler.scale_) == 0.0, 1.0, np.asarray(scaler.scale_))
            mean = np.asarray(scaler.mean_)
            coef_unscaled = coef_scaled / scale
            intercept_unscaled = intercept_scaled - np.sum((coef_scaled * mean) / scale)

            if len(y_test_raw) > 0:
                y_true_eval = y_test_raw.values
                y_pred_eval = y_pred_test
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
                test_df_raw["__y_pred__"] = y_pred_test
                test_stats = binwise_r2_stats(
                    test_df_raw,
                    "p-ERK1-2_dt",
                    "__y_pred__",
                    measured_timepoints,
                    dataset_mode,
                    clamp_negative=True,
                )
                if test_stats:
                    metrics_lin_raw["test_r2"] = test_stats["mean"]
                    metrics_lin_raw["test_r2_bin_mean"] = test_stats["mean"]
                    metrics_lin_raw["test_r2_bin_median"] = test_stats["median"]
                    metrics_lin_raw["test_r2_bin_count"] = test_stats["count"]
            else:
                metrics_lin_raw = _compute_regression_metrics(
                    y_train.values,
                    y_pred_train,
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )

            train_df["__y_pred__"] = y_pred_train
            train_stats = binwise_r2_stats(
                train_df,
                "p-ERK1-2_dt",
                "__y_pred__",
                measured_timepoints,
                dataset_mode,
                clamp_negative=True,
            )
            if train_stats:
                metrics_lin_raw["train_r2"] = train_stats["mean"]
                metrics_lin_raw["train_r2_bin_mean"] = train_stats["mean"]
                metrics_lin_raw["train_r2_bin_median"] = train_stats["median"]
                metrics_lin_raw["train_r2_bin_count"] = train_stats["count"]

            def _format_from_coeffs(coef_vec: np.ndarray, intercept_val: float) -> str:
                parts = [f"{intercept_val:.4f}"]
                for name, c in zip(sanitized_names, coef_vec):
                    parts.append(f" {c:+.4f} * {name}")
                return "".join(parts)

            formula_lin = _format_from_coeffs(coef_unscaled, intercept_unscaled)
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
                test_r2_bin_mean=metrics_lin_raw.get("test_r2_bin_mean"),
                test_r2_bin_median=metrics_lin_raw.get("test_r2_bin_median"),
                test_r2_bin_count=metrics_lin_raw.get("test_r2_bin_count"),
                test_samples=len(y_test_raw),
                train_samples=len(y_train),
                train_relative_mae=metrics_lin_raw["train_relative_mae"],
                train_r2=metrics_lin_raw["train_r2"],
                train_r2_bin_mean=metrics_lin_raw.get("train_r2_bin_mean"),
                train_r2_bin_median=metrics_lin_raw.get("train_r2_bin_median"),
                train_r2_bin_count=metrics_lin_raw.get("train_r2_bin_count"),
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
    # Allow callers to point directly at a seed directory in the new layout (…/seeds/seed_<id>)
    # or a legacy reports/seed_<id> path.
    if output_dir.parent.name == "seeds":
        reports_dir = output_dir
    elif output_dir.name.startswith("seed_") and output_dir.parent.name == "reports":
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
    trajectories_dir = output_dir / "trajectories"
    plots_dir = output_dir / "plots"
    overlays_dir = plots_dir / "overlays"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.random_state)
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
    dataset_by_mode: Dict[str, Path] = {mode: path for mode, path in run_configs}

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
            linear_penalty = (
                0.0
                if args.disable_linear_stability_penalty
                else float(args.linear_stability_penalty)
            )
            inverse_penalty = (
                0.0
                if args.disable_inverse_stability_penalty
                else float(args.inverse_stability_penalty)
            )
            LOGGER.info(
                "Custom loss stability penalties | linear=%s | inverse=%s",
                linear_penalty,
                inverse_penalty,
            )
            loss_required_pERK = r"""
            import SymbolicRegression: Dataset, eval_tree_array
            import Statistics: count

            const REQUIRED_IDX = 2
            const REQUIRED_IDX_GFP = 1
            const MISSING_PENALTY = 1000.0

            const DEP_TOL = 1e-4
            const REDUNDANT_PENALTY = 1000.0

            const WRONGSIGN_P_LINEAR = __WRONGSIGN_P_LINEAR__
            const WRONGSIGN_INV_LINEAR = __WRONGSIGN_INV_LINEAR__

            const EPS_REL = 1e-3
            const LIN_TOL_REL = 1e-3

            @inline relu_pos(x) = x > 0 ? x : 0.0
            @inline relu_neg(x) = x < 0 ? -x : 0.0

            function _loss_core(tree, X, y, options)
                f0, ok0 = eval_tree_array(tree, X, options)
                if !ok0
                    return eltype(y)(Inf)
                end
                base = sum((f0 .- y).^2) / length(y)

                has_required = any(n -> n.degree == 0 && !n.constant && n.feature == REQUIRED_IDX, tree)
                has_gfp = any(n -> n.degree == 0 && !n.constant && n.feature == REQUIRED_IDX_GFP, tree)
                if !has_required || !has_gfp
                    return base + eltype(y)(MISSING_PENALTY)
                end

                p = view(X, REQUIRED_IDX, :)
                eps = EPS_REL .* (abs.(p) .+ 1.0)

                Xplus  = copy(X)
                Xminus = copy(X)
                @inbounds Xplus[REQUIRED_IDX, :]  .= p .+ eps
                @inbounds Xminus[REQUIRED_IDX, :] .= p .- eps

                fplus,  okp = eval_tree_array(tree, Xplus,  options)
                fminus, okm = eval_tree_array(tree, Xminus, options)
                if !(okp && okm)
                    return eltype(y)(Inf)
                end

                delta = abs.(fplus .- fminus)
                rel = delta ./ (abs.(f0) .+ 1.0)
                dep = sum(rel) / length(rel)
                if dep < DEP_TOL
                    return base + eltype(y)(REDUNDANT_PENALTY)
                end

                # Require non-trivial dependence on GFP as well (not just symbol presence).
                g = view(X, REQUIRED_IDX_GFP, :)
                eps_g = EPS_REL .* (abs.(g) .+ 1.0)

                Xplus_g  = copy(X)
                Xminus_g = copy(X)
                @inbounds Xplus_g[REQUIRED_IDX_GFP, :]  .= g .+ eps_g
                @inbounds Xminus_g[REQUIRED_IDX_GFP, :] .= g .- eps_g

                fplus_g, okpg = eval_tree_array(tree, Xplus_g,  options)
                fminus_g, okmg = eval_tree_array(tree, Xminus_g, options)
                if !(okpg && okmg)
                    return eltype(y)(Inf)
                end

                delta_g = abs.(fplus_g .- fminus_g)
                rel_g = delta_g ./ (abs.(f0) .+ 1.0)
                dep_g = sum(rel_g) / length(rel_g)
                if dep_g < DEP_TOL
                    return base + eltype(y)(REDUNDANT_PENALTY)
                end

                # Rule A: penalise positive slope in roughly-linear p direction
                slope_p = (fplus .- fminus) ./ (2 .* eps)
                fmid_p = 0.5 .* (fplus .+ fminus)
                lin_err_p = abs.(f0 .- fmid_p)
                scale_p = abs.(f0) .+ abs.(fmid_p) .+ 1.0
                linearish_p = lin_err_p .<= (LIN_TOL_REL .* scale_p)

                pen_p = 0.0
                nlin_p = count(linearish_p)
                if nlin_p > 0
                    pos_mass = sum(relu_pos.(slope_p[linearish_p])) / nlin_p
                    pen_p = WRONGSIGN_P_LINEAR * pos_mass
                end

                # Rule B: penalise negative slope in roughly-linear 1/p direction
                q0     = 1.0 ./ p
                qplus  = 1.0 ./ (p .+ eps)
                qminus = 1.0 ./ (p .- eps)
                dq = qplus .- qminus

                safe = isfinite.(q0) .& isfinite.(qplus) .& isfinite.(qminus) .& (abs.(dq) .> 1e-12)

                slope_q = similar(slope_p)
                @inbounds slope_q .= 0.0
                @inbounds slope_q[safe] .= (fplus[safe] .- fminus[safe]) ./ dq[safe]

                fhat_q = similar(f0)
                @inbounds fhat_q .= f0
                @inbounds fhat_q[safe] .= fminus[safe] .+ (fplus[safe] .- fminus[safe]) .* ((q0[safe] .- qminus[safe]) ./ dq[safe])

                lin_err_q = abs.(f0 .- fhat_q)
                scale_q = abs.(f0) .+ abs.(fhat_q) .+ 1.0
                linearish_q = safe .& (lin_err_q .<= (LIN_TOL_REL .* scale_q))

                pen_q = 0.0
                nlin_q = count(linearish_q)
                if nlin_q > 0
                    neg_mass = sum(relu_neg.(slope_q[linearish_q])) / nlin_q
                    pen_q = WRONGSIGN_INV_LINEAR * neg_mass
                end

                return base + eltype(y)(pen_p + pen_q)
            end

            function loss_with_required_pERK(tree, dataset::Dataset{T,L}, options)::L where {T,L}
                return L(_loss_core(tree, dataset.X, dataset.y, options))
            end

            function loss_with_required_pERK(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
                X = idx === nothing ? dataset.X : dataset.X[:, idx]
                y = idx === nothing ? dataset.y : view(dataset.y, idx)
                return L(_loss_core(tree, X, y, options))
            end
            """
            loss_required_pERK = loss_required_pERK.replace(
                "__WRONGSIGN_P_LINEAR__", format(linear_penalty, ".16g")
            ).replace(
                "__WRONGSIGN_INV_LINEAR__", format(inverse_penalty, ".16g")
            )

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
                "random_state": args.random_state,
                # Force deterministic evolution (single-process, no multithreading).
                "deterministic": True,
                # PySR >=0.16: prefer explicit serial execution to avoid nondeterminism.
                "parallelism": "serial",
                "procs": 0,

                # 👇 New bits
                "loss_function": """println(">>> Custom loss LOADED")""" + loss_required_pERK,
                # optional but a bit safer / clearer for custom loss:
                "loss_scale": "log",
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
                        f"    Train R2     : {result.train_r2:.4f}"
                        if result.train_r2 is not None
                        else "    Train R2     : N/A"
                    )
                    block_lines.append(
                        f"    Test R2      : {result.test_r2:.4f}"
                        if result.test_r2 is not None
                        else "    Test R2      : N/A"
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
                    # 👇 New line: print the formula in the log
                    block_lines.append(
                        f"    Formula      : {result.formula}" if result.formula is not None else "    Formula      : N/A"
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
            traj_filename = f"predicted_trajectories_{run_mode}.csv"
            traj_out = trajectories_dir / traj_filename
            legacy_traj_out = metrics_dir / traj_filename
            # Skip emitting snapshot trajectories to keep PySR snapshot outputs minimal.
            if run_mode != "snapshot":
                traj_df.to_csv(traj_out, index=False)
                if legacy_traj_out != traj_out:
                    try:
                        shutil.copy(traj_out, legacy_traj_out)
                    except Exception:
                        traj_df.to_csv(legacy_traj_out, index=False)
                LOGGER.info("Saved predicted trajectories to %s", traj_out)
            else:
                LOGGER.info("Skipping predicted trajectories export for snapshot mode.")

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
            "test_r2_bin_mean": res.test_r2_bin_mean,
            "test_r2_bin_median": res.test_r2_bin_median,
            "test_r2_bin_count": res.test_r2_bin_count,
            "train_relative_mae": res.train_relative_mae,
            "train_log_relative_mae": res.train_relative_mae,
            "train_mae": None,
            "train_log_mae": None,
            "train_r2": res.train_r2,
            "train_log_r2": res.train_r2,
            "train_r2_bin_mean": res.train_r2_bin_mean,
            "train_r2_bin_median": res.train_r2_bin_median,
            "train_r2_bin_count": res.train_r2_bin_count,
            "train_samples": res.train_samples,
            "test_samples_raw": res.test_samples,
            "target_space": res.target_space,
            "primary_test_r2": res.test_r2,
            "seed": RUN_SEED,
            }
            for res in all_results
        ]
    )
    summary_path = summary_dir / "functional_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote summary CSV to %s", summary_path)
    LOGGER.info("Symbolic regression pipeline completed | random_state=%s", args.random_state)

    # Integration metrics + trajectories per dataset mode (feeds aggregated metrics)
    for run_mode, dataset_path in dataset_by_mode.items():
        if run_mode == "snapshot":
            LOGGER.info("Skipping integration metrics/trajectories for snapshot mode.")
            continue
        metrics_out = metrics_dir / f"marker_integration_metrics_{run_mode}.csv"
        traj_filename = f"marker_integration_trajectories_{run_mode}.csv"
        traj_out = trajectories_dir / traj_filename
        legacy_traj_out = metrics_dir / traj_filename
        sr_traj = trajectories_dir / f"predicted_trajectories_{run_mode}.csv"
        sr_traj_legacy = metrics_dir / f"predicted_trajectories_{run_mode}.csv"
        if not sr_traj.exists() and sr_traj_legacy.exists():
            sr_traj = sr_traj_legacy
        cmd = [
            sys.executable,
            "-m",
            "experimental.sr_pipeline.compute_marker_integration",
            "--dataset",
            str(dataset_path),
            "--summary",
            str(summary_path),
            "--output",
            str(metrics_out),
            "--trajectories-output",
            str(traj_out),
            "--dataset-mode",
            run_mode,
            "--test-size",
            str(args.test_size),
            "--random-state",
            str(args.random_state),
            "--seed",
            str(args.random_state),
            "--measured-timepoints",
            *[str(t) for t in args.measured_timepoints],
        ]
        if sr_traj.exists():
            cmd.extend(["--sr-trajectories", str(sr_traj)])
        _run_module(cmd, f"integration metrics ({run_mode})")
        if traj_out.exists() and legacy_traj_out != traj_out:
            try:
                shutil.copy(traj_out, legacy_traj_out)
            except Exception:
                pass

        # Per-seed overlay plots per model
        if not sr_traj.exists():
            continue
        integ_traj = traj_out
        integ_metrics = metrics_out
        base_args = [
            sys.executable,
            "-m",
            "experimental.sr_pipeline.plot_marker_overlays",
            "--dataset",
            str(dataset_path),
            "--summary",
            str(summary_path),
            "--output-dir",
            str(overlays_dir),
            "--predicted-trajectories",
            str(sr_traj),
            "--dataset-mode",
            run_mode,
            "--measured-timepoints",
            *[str(t) for t in args.measured_timepoints],
            "--markers-per-fig",
            "0",
        ]
        if integ_traj.exists():
            base_args.extend(["--integration-trajectories", str(integ_traj)])
        if integ_metrics.exists():
            base_args.extend(["--integration-metrics", str(integ_metrics)])
        for model in ("PySR", "Linear Regression"):
            cmd_overlay = base_args + ["--model", model]
            _run_module(cmd_overlay, f"overlay plots ({run_mode}, {model})")
    return summary_path


def main() -> None:
    args = parse_args()
    seeds = canonicalize_seeds(args.random_state, args.seeds)
    LOGGER.info("Running seeds in order: %s", seeds)
    runs_layout = _is_runs_layout(args.output_dir)
    seeds_root = args.output_dir / "seeds" if runs_layout else args.output_dir / "reports"
    aggregated_root = args.output_dir / "aggregated" if runs_layout else args.output_dir / "reports"
    aggregated_reports = aggregated_root / "reports" if runs_layout else aggregated_root
    aggregated_summary_dir = aggregated_reports / "summary"
    aggregated_formulas_dir = aggregated_reports / "formulas"
    aggregated_metrics_dir = aggregated_root / "metrics"
    aggregated_plots_dir = aggregated_root / "plots"
    aggregated_trajectories_dir = aggregated_root / "trajectories"
    seeds_root.mkdir(parents=True, exist_ok=True)
    aggregated_summary_dir.mkdir(parents=True, exist_ok=True)
    aggregated_formulas_dir.mkdir(parents=True, exist_ok=True)
    aggregated_metrics_dir.mkdir(parents=True, exist_ok=True)
    aggregated_plots_dir.mkdir(parents=True, exist_ok=True)
    aggregated_trajectories_dir.mkdir(parents=True, exist_ok=True)
    summary_paths: List[Path] = []
    for seed in seeds:
        seed_dir = seeds_root / f"seed_{seed}"
        seed_args = argparse.Namespace(**vars(args))
        seed_args.random_state = seed
        seed_args.output_dir = seed_dir
        seed_all(seed)
        summary_paths.append(run_pipeline(seed_args))

    if len(seeds) > 1:
        combined_dir = aggregated_summary_dir
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
            canonical_summary = aggregated_summary_dir / "functional_group_summary.csv"
            canonical_summary.parent.mkdir(parents=True, exist_ok=True)
            agg_mean.to_csv(canonical_summary, index=False)
            LOGGER.info("Wrote mean-over-seeds summary to %s", canonical_summary)

            seed_summary_path = aggregated_summary_dir / "functional_group_summary_seed.csv"
            pd.read_csv(summary_paths[0]).to_csv(seed_summary_path, index=False)
            LOGGER.info("Saved first-seed summary to %s", seed_summary_path)

            formulas_dir = aggregated_formulas_dir
            for mode in ("snapshot", "per_minute"):
                combined_formula_lines: List[str] = []
                for seed in seeds:
                    seed_file = seeds_root / f"seed_{seed}" / "formulas" / f"all_{mode}.txt"
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
        plot_args = [
            sys.executable,
            "-m",
            "experimental.sr_pipeline.plot_metrics_summary",
            "--summary",
            str(canonical_summary),
            "--output-dir",
            str(aggregated_plots_dir),
            "--group-definitions-csv",
            str(args.group_definitions_csv),
            "--integration-metrics-dir",
            str(aggregated_metrics_dir),
            "--snapshot-dataset",
            str(args.dataset),
            "--measured-timepoints",
            *[str(t) for t in args.measured_timepoints],
        ]
        if args.per_minute_dataset is not None:
            plot_args.extend(["--per-minute-dataset", str(args.per_minute_dataset)])
        _run_module(plot_args, "aggregated metrics summary plots")
    else:
        seed_summary = summary_paths[0]
        canonical_summary = aggregated_summary_dir / "functional_group_summary.csv"
        canonical_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(seed_summary).to_csv(canonical_summary, index=False)
        seed_summary_path = aggregated_summary_dir / "functional_group_summary_seed.csv"
        pd.read_csv(seed_summary).to_csv(seed_summary_path, index=False)
        formulas_dir = aggregated_formulas_dir
        for mode in ("snapshot", "per_minute"):
            seed_formula = seeds_root / f"seed_{seeds[0]}" / "formulas" / f"all_{mode}.txt"
            if seed_formula.exists():
                out_formula = formulas_dir / f"all_{mode}.txt"
                out_formula.write_text(seed_formula.read_text())
                LOGGER.info("Copied formula report to %s", out_formula)
        plot_args = [
            sys.executable,
            "-m",
            "experimental.sr_pipeline.plot_metrics_summary",
            "--summary",
            str(canonical_summary),
            "--output-dir",
            str(aggregated_plots_dir),
            "--group-definitions-csv",
            str(args.group_definitions_csv),
            "--integration-metrics-dir",
            str(aggregated_metrics_dir),
            "--snapshot-dataset",
            str(args.dataset),
            "--measured-timepoints",
            *[str(t) for t in args.measured_timepoints],
        ]
        if args.per_minute_dataset is not None:
            plot_args.extend(["--per-minute-dataset", str(args.per_minute_dataset)])
        _run_module(plot_args, "aggregated metrics summary plots")


if __name__ == "__main__":
    main()
