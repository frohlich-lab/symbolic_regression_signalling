"""Generate processed time-course derivatives and fits from raw trajectories.

Recreates the logic from the `data_exploration.ipynb` notebook while giving
outputs names that match the marker SR context. By default the script
emits:

* ``markers_time_series.csv``
* ``markers_feature_matrix.csv``
* ``markers_fit_snapshot.csv``
* ``markers_per_minute_fit.csv``

The prefix can be customised with ``--output-prefix``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

try:
    from scipy.optimize import least_squares
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    least_squares = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

DEFAULT_PROTEINS = [
    "p-ERK1-2",
    "p-MEK1-2",
    "p-RAF",
    "p-p90RSK",
    "p-MAPKAPK2",
    "p-PDK1",
    "p-MKK3-6",
]

STATS_PROTEINS = {"p-ERK1-2", "p-MEK1-2"}

LOGGER = logging.getLogger("data_prep.marker_inputs")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed time-course datasets from raw measurements.")
    parser.add_argument(
        "--time-course",
        type=Path,
        required=True,
        help="Path to the raw time course CSV (complete_dat.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experimental/processed"),
        help="Directory where processed CSVs will be written.",
    )
    parser.add_argument(
        "--gfp-bins",
        type=int,
        default=50,
        help="Number of GFP quantile bins per marker (default 50).",
    )
    parser.add_argument(
        "--skip-rebin",
        action="store_true",
        help="(Deprecated) Kept for backwards compatibility; existing bins are reused automatically.",
    )
    parser.add_argument(
        "--force-rebin",
        action="store_true",
        help="Ignore any existing GFP_bin column and recompute quantile bins.",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.95,
        help="Minimum p-ERK1-2 fit R² required to keep a bin (default: 0.95).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum non-NaN timepoints required before fitting (default 5).",
    )
    parser.add_argument(
        "--joint-lam",
        type=float,
        default=1.0,
        help="Smoothness penalty λ applied between neighbouring GFP bins during the joint fit (default 1.0).",
    )
    parser.add_argument(
        "--extra-times",
        type=float,
        nargs="*",
        default=(1.0, 3.0),
        help="Additional timepoints (minutes) to evaluate fits for the extra CSV.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="markers",
        help="Prefix used when naming the emitted CSV files (default: markers).",
    )
    return parser.parse_args()


def rise_and_fall(t: np.ndarray, B: float, A_T: float, tau_T: float, tau_S: float) -> np.ndarray:
    return B + A_T * (1 - np.exp(-t / tau_T)) * np.exp(-t / tau_S)


def rise_and_fall_derivative(t: np.ndarray, B: float, A_T: float, tau_T: float, tau_S: float) -> np.ndarray:
    term1 = (A_T / tau_T) * np.exp(-t / tau_T) * np.exp(-t / tau_S)
    term2 = (A_T / tau_S) * (1 - np.exp(-t / tau_T)) * np.exp(-t / tau_S)
    return term1 - term2


def assign_gfp_bins(df: pd.DataFrame, bins: int) -> pd.Series:
    def _bin_column(col: pd.Series) -> pd.Series:
        unique_values = col.dropna().unique()
        if len(unique_values) < 2:
            return pd.Series(np.nan, index=col.index)
        quantiles = min(bins, len(unique_values))
        try:
            return pd.qcut(col, q=quantiles, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=col.index)

    binned = df.groupby("marker", group_keys=False)["GFP"].apply(_bin_column)
    return binned.astype("Int64")


def initialise_columns(df: pd.DataFrame, proteins: Sequence[str]) -> None:
    for protein in proteins:
        df[f"{protein}_fit"] = np.nan
        if protein in STATS_PROTEINS:
            df[f"{protein}_derivative"] = np.nan
            df[f"{protein}_dt"] = np.nan
            df[f"{protein}_max"] = np.nan
            df[f"{protein}_min"] = np.nan
            df[f"{protein}_avg"] = np.nan
            df[f"{protein}_range"] = np.nan


def initialise_extra_frame(base: pd.DataFrame, proteins: Sequence[str]) -> pd.DataFrame:
    extra = base[["GFP"]].copy()
    for protein in proteins:
        extra[f"{protein}_fit"] = np.nan
        if protein in STATS_PROTEINS:
            extra[f"{protein}_dt"] = np.nan
            extra[f"{protein}_min"] = np.nan
    return extra


def _clip_exp(z: np.ndarray) -> np.ndarray:
    """Keep exp() evaluations numerically stable."""
    return np.exp(np.clip(z, -50.0, 50.0))


def _rise_fall_model(t: np.ndarray, B: float, A: float, tau_T: float, tau_S: float) -> np.ndarray:
    tau_T = max(tau_T, 1e-3)
    tau_S = max(tau_S, 1e-3)
    e1 = _clip_exp(-t / tau_T)
    e2 = _clip_exp(-t / tau_S)
    return B + A * (1.0 - e1) * e2


def _rise_fall_derivative(t: np.ndarray, B: float, A: float, tau_T: float, tau_S: float) -> np.ndarray:
    tau_T = max(tau_T, 1e-3)
    tau_S = max(tau_S, 1e-3)
    e1 = _clip_exp(-t / tau_T)
    e2 = _clip_exp(-t / tau_S)
    return A * e2 * (e1 / tau_T - (1.0 - e1) / tau_S)


def _p0_rise_fall(t: np.ndarray, y: np.ndarray) -> Tuple[List[float], Tuple[List[float], List[float]], int]:
    B0 = float(np.nanmin(y))
    A0 = max(float(np.nanmax(y) - B0), 1e-3)
    return [B0, A0, 5.0, 10.0], ([-np.inf, 0.0, 3.0, 1e-3], [np.inf, np.inf, 1e3, 1e3]), 4


def _dlogistic_with_tail(
    t: np.ndarray, B: float, A1: float, A2: float, t1: float, t2: float, k: float, m: float
) -> np.ndarray:
    """Difference of logistics with a modest linear tail."""
    k = max(k, 1e-6)
    s1 = 1.0 / (1.0 + _clip_exp(-(t - t1) * k))
    s2 = 1.0 / (1.0 + _clip_exp(-(t - t2) * k))
    return B + A1 * s1 - A2 * s2 + m * t


def _dlogistic_with_tail_derivative(
    t: np.ndarray, B: float, A1: float, A2: float, t1: float, t2: float, k: float, m: float
) -> np.ndarray:
    k = max(k, 1e-6)
    s1 = 1.0 / (1.0 + _clip_exp(-(t - t1) * k))
    s2 = 1.0 / (1.0 + _clip_exp(-(t - t2) * k))
    ds1 = k * s1 * (1.0 - s1)
    ds2 = k * s2 * (1.0 - s2)
    return A1 * ds1 - A2 * ds2 + m


def _p0_dlogistic(t: np.ndarray, y: np.ndarray) -> Tuple[List[float], Tuple[List[float], List[float]], int]:
    B0 = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    span = max(ymax - B0, 1e-3)
    t_med = float(np.median(t))
    p0 = [B0, 0.7 * span, 0.7 * span, t_med * 0.7, t_med * 1.3, 0.2, 0.0]
    slope_max = 0.05
    lower = [-np.inf, 0.0, 0.0, -np.inf, -np.inf, 0.02, -slope_max]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf, 0.6, slope_max]
    return p0, (lower, upper), 7


PREDICTORS: Dict[str, Callable[..., np.ndarray]] = {
    "dlogistic": _dlogistic_with_tail,
    "rise_fall": _rise_fall_model,
}
PREDICTOR_DERIVS: Dict[str, Callable[..., np.ndarray]] = {
    "dlogistic": _dlogistic_with_tail_derivative,
    "rise_fall": _rise_fall_derivative,
}
INIT_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[List[float], Tuple[List[float], List[float]], int]]] = {
    "dlogistic": _p0_dlogistic,
    "rise_fall": _p0_rise_fall,
}


def _evaluate_fit(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> Tuple[float, float, float]:
    """Return RSS, R², and AICc for a single trajectory."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rss = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.nanmean(y_true)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    n = len(y_true)
    if n <= k + 1 or rss <= 0:
        aicc = float("inf")
    else:
        aicc = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1)
    return rss, r2, aicc


def _evaluate_fit_multi(
    all_y_true: List[np.ndarray],
    all_y_pred: List[np.ndarray],
    k_total: int,
) -> Tuple[float, float, float]:
    """Global RSS, R², and AICc across multiple bins."""
    y_true = np.concatenate([np.asarray(yb, dtype=float).ravel() for yb in all_y_true])
    y_pred = np.concatenate([np.asarray(yp, dtype=float).ravel() for yp in all_y_pred])
    rss = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.nanmean(y_true)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
    n = len(y_true)
    if n <= k_total + 1 or rss <= 0:
        aicc = float("inf")
    else:
        aicc = n * np.log(rss / n) + 2 * k_total + (2 * k_total * (k_total + 1)) / max(n - k_total - 1, 1)
    return rss, r2, aicc


def _joint_residuals(
    params_flat: np.ndarray,
    all_t: List[np.ndarray],
    all_y: List[np.ndarray],
    model_fn: Callable[..., np.ndarray],
    k: int,
    lam: float,
) -> np.ndarray:
    """Residual vector for joint optimisation with smoothness penalty."""
    residuals: List[np.ndarray] = []
    B = len(all_t)
    for b in range(B):
        theta = params_flat[b * k : (b + 1) * k]
        y_pred = model_fn(all_t[b], *theta)
        residuals.append(y_pred - all_y[b])
    if lam > 0.0 and B > 1:
        sqrt_lam = float(np.sqrt(lam))
        for b in range(B - 1):
            theta_b = params_flat[b * k : (b + 1) * k]
            theta_next = params_flat[(b + 1) * k : (b + 2) * k]
            residuals.append(sqrt_lam * (theta_next - theta_b))
    return np.concatenate([r.ravel() for r in residuals])


def fit_models_for_protein_joint(
    all_t: List[np.ndarray],
    all_y: List[np.ndarray],
    lam: float,
    model_names: Sequence[str] = ("dlogistic", "rise_fall"),
) -> Dict[str, object]:
    """Fit every GFP bin jointly for a single protein and return the best model."""
    if not all_t:
        return {}

    candidates: List[Dict[str, object]] = []
    for name in model_names:
        init_fn = INIT_FUNCS[name]
        try:
            p0_list: List[float] = []
            lower_list: List[float] = []
            upper_list: List[float] = []
            k_dim: Optional[int] = None
            for t_b, y_b in zip(all_t, all_y):
                p0_b, bounds_b, k_b = init_fn(t_b, y_b)
                if k_dim is None:
                    k_dim = k_b
                elif k_dim != k_b:
                    raise RuntimeError("Parameter mismatch while stacking bins.")
                p0_list.extend(p0_b)
                lo, hi = bounds_b
                lower_list.extend(lo)
                upper_list.extend(hi)
            if k_dim is None:
                continue
            result = least_squares(
                _joint_residuals,
                np.asarray(p0_list, dtype=float),
                bounds=(np.asarray(lower_list, dtype=float), np.asarray(upper_list, dtype=float)),
                args=(all_t, all_y, PREDICTORS[name], k_dim, lam),
                loss="huber",
                f_scale=1.0,
                method="trf",
                max_nfev=1000,
            )
            params_flat = result.x
            preds: List[np.ndarray] = []
            for b in range(len(all_t)):
                theta = params_flat[b * k_dim : (b + 1) * k_dim]
                preds.append(PREDICTORS[name](all_t[b], *theta))
            _, r2_global, aicc_global = _evaluate_fit_multi(all_y, preds, k_total=k_dim * len(all_t))
            r2_per_bin: List[float] = []
            for y_true_b, y_pred_b in zip(all_y, preds):
                _, r2_b, _ = _evaluate_fit(y_true_b, y_pred_b, k_dim)
                r2_per_bin.append(float(r2_b))
            candidates.append(
                {
                    "name": name,
                    "params_flat": params_flat,
                    "k": k_dim,
                    "r2": float(r2_global),
                    "aicc": float(aicc_global),
                    "r2_per_bin": r2_per_bin,
                }
            )
        except Exception as exc:
            warnings.warn(f"Joint {name} fit failed: {exc}", RuntimeWarning)

    if not candidates:
        return {}

    candidates.sort(key=lambda d: d["r2"], reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        eps = 1e-3
        tied = [cand for cand in candidates if abs(cand["r2"] - candidates[0]["r2"]) < eps]
        if len(tied) > 1:
            best = min(tied, key=lambda d: d["aicc"])
    return best


def fit_marker_bins_joint(
    marker: str,
    gfp_bins: Sequence[int],
    data_per_bin: Dict[int, pd.DataFrame],
    timepoints: np.ndarray,
    proteins: Sequence[str],
    min_points: int,
    extra_times: np.ndarray,
    dense_times: np.ndarray,
    lam: float,
) -> Tuple[
    Dict[int, Dict[str, dict]],
    Dict[int, List[Tuple[str, float, float]]],
    Dict[int, List[Tuple[str, float, float]]],
    Dict[int, Dict[str, Dict[str, float]]],
]:
    """Jointly fit every GFP bin for all proteins of a marker."""
    if least_squares is None:
        warnings.warn(
            f"SciPy unavailable; skipping fits for marker {marker}. Original error: {_SCIPY_IMPORT_ERROR}",
            RuntimeWarning,
        )
        return {}, {}, {}, {}

    x_full = np.asarray(timepoints, dtype=float)
    extended_times = np.unique(np.concatenate([x_full, extra_times]).astype(float))
    dense_times_arr = np.unique(np.asarray(dense_times, dtype=float))

    updates_per_bin: Dict[int, Dict[str, dict]] = {}
    extra_updates_per_bin: Dict[int, List[Tuple[str, float, float]]] = {}
    dense_updates_per_bin: Dict[int, List[Tuple[str, float, float]]] = {}
    metadata_per_bin: Dict[int, Dict[str, Dict[str, float]]] = {}

    for protein in proteins:
        bin_times: List[np.ndarray] = []
        bin_values: List[np.ndarray] = []
        bin_ids: List[int] = []
        observed_by_bin: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for gfp_bin in gfp_bins:
            df_bin = data_per_bin[int(gfp_bin)]
            y_series = df_bin[protein].to_numpy(dtype=float)
            valid = np.isfinite(y_series)
            if valid.sum() < min_points:
                continue
            t_valid = x_full[valid]
            y_valid = y_series[valid]
            if y_valid.size == 0:
                continue
            bin_times.append(t_valid)
            bin_values.append(y_valid)
            bin_ids.append(int(gfp_bin))
            observed_by_bin[int(gfp_bin)] = (t_valid, y_valid)

        if not bin_times:
            continue

        best = fit_models_for_protein_joint(bin_times, bin_values, lam=lam)
        if not best:
            continue

        model_name = str(best["name"])
        params_flat = np.asarray(best["params_flat"], dtype=float)
        k_dim = int(best["k"])
        r2_global = float(best["r2"])
        r2_per_bin = best.get("r2_per_bin", [])
        model_fn = PREDICTORS[model_name]
        deriv_fn = PREDICTOR_DERIVS[model_name]

        for idx, gfp_bin in enumerate(bin_ids):
            theta = params_flat[idx * k_dim : (idx + 1) * k_dim]
            t_valid, y_valid = observed_by_bin[gfp_bin]
            curve_main = model_fn(t_valid, *theta)
            deriv_main = deriv_fn(t_valid, *theta)
            curve_extended = model_fn(extended_times, *theta)
            deriv_extended = deriv_fn(extended_times, *theta)
            curve_dense = model_fn(dense_times_arr, *theta)
            deriv_dense = deriv_fn(dense_times_arr, *theta)

            updates_per_bin.setdefault(gfp_bin, {}).setdefault(protein, {})
            for t_val, y_fit, d_val in zip(t_valid, curve_main, deriv_main):
                updates_per_bin[gfp_bin][protein][float(t_val)] = (float(y_fit), float(d_val))

            stats = (
                float(np.max(y_valid)),
                float(np.min(y_valid)),
                float(np.mean(y_valid)),
                float(np.max(y_valid) - np.min(y_valid)),
            )
            if protein in STATS_PROTEINS:
                updates_per_bin[gfp_bin][protein]["stats"] = stats

            extra_updates = extra_updates_per_bin.setdefault(gfp_bin, [])
            for t_val, y_fit, d_val in zip(extended_times, curve_extended, deriv_extended):
                extra_updates.append((f"{protein}_fit", float(t_val), float(y_fit)))
                if protein in STATS_PROTEINS:
                    extra_updates.append((f"{protein}_dt", float(t_val), float(d_val)))
                    extra_updates.append((f"{protein}_min", float(t_val), float(stats[1])))

            dense_updates = dense_updates_per_bin.setdefault(gfp_bin, [])
            for t_val, y_fit, d_val in zip(dense_times_arr, curve_dense, deriv_dense):
                dense_updates.append((f"{protein}_fit", float(t_val), float(y_fit)))
                if protein in STATS_PROTEINS:
                    dense_updates.append((f"{protein}_dt", float(t_val), float(d_val)))
                    dense_updates.append((f"{protein}_min", float(t_val), float(stats[1])))

            meta_prot = metadata_per_bin.setdefault(gfp_bin, {}).setdefault(protein, {})
            meta_prot["model"] = model_name
            meta_prot["r2"] = float(r2_per_bin[idx]) if idx < len(r2_per_bin) else float("nan")
            meta_prot["r2_global"] = r2_global
            meta_prot["lam"] = float(lam)

    return updates_per_bin, extra_updates_per_bin, dense_updates_per_bin, metadata_per_bin


def apply_updates(
    time_trajectories: pd.DataFrame,
    extra_fit: pd.DataFrame,
    marker: str,
    gfp_bin: int,
    updates: dict,
    extra_updates: Iterable[Tuple[str, float, float]],
    model_names: Optional[Dict[str, str]] = None,
    r2_scores: Optional[Dict[str, float]] = None,
) -> None:
    for protein, values in updates.items():
        for timepoint, payload in values.items():
            if timepoint == "stats":
                max_v, min_v, avg_v, range_v = payload
                idx_slice = (marker, gfp_bin, slice(None))
                time_trajectories.loc[idx_slice, f"{protein}_max"] = max_v
                time_trajectories.loc[idx_slice, f"{protein}_min"] = min_v
                time_trajectories.loc[idx_slice, f"{protein}_avg"] = avg_v
                time_trajectories.loc[idx_slice, f"{protein}_range"] = range_v
                if model_names and protein in model_names:
                    time_trajectories.loc[idx_slice, f"{protein}_fit_model"] = model_names[protein]
                if r2_scores and protein in r2_scores:
                    time_trajectories.loc[idx_slice, f"{protein}_fit_r2"] = r2_scores[protein]
                continue
            fit_val, deriv_val = payload
            idx = (marker, gfp_bin, timepoint)
            time_trajectories.loc[idx, f"{protein}_fit"] = fit_val
            if protein in STATS_PROTEINS:
                time_trajectories.loc[idx, f"{protein}_derivative"] = deriv_val
                time_trajectories.loc[idx, f"{protein}_dt"] = deriv_val
            if model_names and protein in model_names:
                idx_slice = (marker, gfp_bin, slice(None))
                time_trajectories.loc[idx_slice, f"{protein}_fit_model"] = model_names[protein]
            if r2_scores and protein in r2_scores:
                idx_slice = (marker, gfp_bin, slice(None))
                time_trajectories.loc[idx_slice, f"{protein}_fit_r2"] = r2_scores[protein]

    for column, timepoint, value in extra_updates:
        idx = (marker, gfp_bin, timepoint)
        extra_fit.loc[idx, column] = value


def apply_extra_updates(
    frame: pd.DataFrame,
    marker: str,
    gfp_bin: int,
    extra_updates: Iterable[Tuple[str, float, float]],
) -> None:
    for column, timepoint, value in extra_updates:
        idx = (marker, gfp_bin, timepoint)
        frame.loc[idx, column] = value


def fill_extra_gfp_means(extra_fit: pd.DataFrame, fill_times: Sequence[float]) -> None:
    df_reset = extra_fit.reset_index()

    for t in fill_times:
        mask = df_reset["timepoint"].eq(t)
        for (marker, gfp_bin), group in df_reset[mask].groupby(["marker", "GFP_bin"]):
            mean_gfp = df_reset[
                (df_reset["marker"].eq(marker))
                & (df_reset["GFP_bin"].eq(gfp_bin))
                & df_reset["GFP"].notna()
            ]["GFP"].mean()
            idx = (
                df_reset["marker"].eq(marker)
                & df_reset["GFP_bin"].eq(gfp_bin)
                & df_reset["timepoint"].eq(t)
            )
            df_reset.loc[idx, "GFP"] = mean_gfp

    df_reset.sort_values(["marker", "GFP_bin", "timepoint"], inplace=True)
    df_reset.reset_index(drop=True, inplace=True)
    extra_fit.update(df_reset.set_index(["marker", "GFP_bin", "timepoint"]))


def build_filtered(time_trajectories: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "GFP",
        "p-ERK1-2",
        "p-MEK1-2",
        "p-RAF",
        "p-p90RSK",
        "p-MAPKAPK2",
        "p-PDK1",
        "p-MKK3-6",
    ]
    stat_cols = []
    for protein in ["p-ERK1-2", "p-MEK1-2"]:
        for suffix in ["_dt", "_min", "_max", "_avg", "_range"]:
            stat_cols.append(f"{protein}{suffix}")
    fit_cols = [f"{protein}_fit" for protein in base_cols[1:]]
    model_cols = [f"{protein}_fit_model" for protein in base_cols[1:]]
    r2_cols = [f"{protein}_fit_r2" for protein in base_cols[1:]]
    columns = base_cols + stat_cols + fit_cols + model_cols + r2_cols
    filtered = time_trajectories[columns].reset_index()
    return filtered[["marker", "GFP_bin", "timepoint"] + columns]


def run_processing(args: argparse.Namespace) -> None:
    LOGGER.info("Loading time-course data from %s", args.time_course)
    time_course = pd.read_csv(args.time_course)
    if {"marker", "GFP", "timepoint"} - set(time_course.columns):
        missing = {"marker", "GFP", "timepoint"} - set(time_course.columns)
        raise ValueError(f"time course data is missing required columns: {sorted(missing)}")

    time_course = time_course.copy()
    time_course["timepoint"] = time_course["timepoint"].astype(float)
    if args.skip_rebin:
        LOGGER.warning("--skip-rebin is deprecated; existing bins are now reused automatically.")

    if "GFP_bin" in time_course.columns and not args.force_rebin:
        LOGGER.info("Reusing existing GFP_bin assignments from input.")
        time_course = time_course.dropna(subset=["GFP_bin"])
        time_course["GFP_bin"] = time_course["GFP_bin"].astype(float).round().astype(int)
    else:
        LOGGER.info("Assigning GFP bins with %d quantiles", args.gfp_bins)
        time_course["GFP_bin"] = assign_gfp_bins(time_course, args.gfp_bins)
        time_course = time_course.dropna(subset=["GFP_bin"])  # drop markers lacking variation
        time_course["GFP_bin"] = time_course["GFP_bin"].astype(int)

    group_keys = {"marker", "GFP_bin", "timepoint"}
    numeric_cols = [
        col
        for col in time_course.select_dtypes(include=[np.number]).columns
        if col not in group_keys
    ]
    time_trajectories = (
        time_course.groupby(["marker", "GFP_bin", "timepoint"])[numeric_cols]
        .mean()
        .sort_index()
    )

    raw_timepoints = np.sort(time_course["timepoint"].dropna().unique())
    marker_bins = time_trajectories.index.droplevel("timepoint").unique()
    # Ensure every marker/bin has rows allocated for all measured timepoints so
    # later .loc updates never extend the index out of lexsorted order.
    base_index = pd.MultiIndex.from_tuples(
        [
            (marker, gfp_bin, float(t))
            for marker, gfp_bin in marker_bins
            for t in raw_timepoints
        ],
        names=time_trajectories.index.names,
    )
    time_trajectories = time_trajectories.reindex(base_index).sort_index()

    initialise_columns(time_trajectories, DEFAULT_PROTEINS)
    extra_fit = initialise_extra_frame(time_trajectories, DEFAULT_PROTEINS)
    max_time = float(np.nanmax(raw_timepoints))
    min_time = float(np.nanmin(raw_timepoints))
    dense_times = np.arange(np.floor(min_time), np.ceil(max_time) + 1.0, 1.0)
    dense_fit = initialise_extra_frame(time_trajectories, DEFAULT_PROTEINS)

    markers = time_trajectories.index.get_level_values("marker").unique()
    extra_times = np.array(args.extra_times, dtype=float)
    extended_timepoints = np.unique(np.concatenate([raw_timepoints, extra_times]))
    dense_timepoints = np.unique(np.concatenate([raw_timepoints, dense_times]))
    # Preallocate rows for every marker/bin at all requested times to keep the
    # MultiIndex lexsorted and avoid assignment warnings downstream.
    full_index = pd.MultiIndex.from_tuples(
        [
            (marker, gfp_bin, float(t))
            for marker, gfp_bin in marker_bins
            for t in extended_timepoints
        ],
        names=time_trajectories.index.names,
    )
    extra_fit = extra_fit.reindex(full_index).sort_index()
    dense_index = pd.MultiIndex.from_tuples(
        [
            (marker, gfp_bin, float(t))
            for marker, gfp_bin in marker_bins
            for t in dense_timepoints
        ],
        names=time_trajectories.index.names,
    )
    dense_fit = dense_fit.reindex(dense_index).sort_index()

    total_markers = len(markers)
    LOGGER.info("Processing %d markers", total_markers)
    for marker in tqdm(markers, desc="Markers", unit="marker", leave=False):
        marker_slice = time_trajectories.loc[marker]
        gfp_bins = list(marker_slice.index.get_level_values("GFP_bin").unique())
        if not gfp_bins:
            LOGGER.info("Marker %s has no GFP bins after preprocessing", marker)
            continue
        bin_data = {
            int(gfp_bin): marker_slice.loc[gfp_bin].reindex(raw_timepoints)
            for gfp_bin in gfp_bins
        }
        (
            updates_per_bin,
            extra_updates_per_bin,
            dense_updates_per_bin,
            metadata_per_bin,
        ) = fit_marker_bins_joint(
            marker,
            [int(g) for g in gfp_bins],
            bin_data,
            raw_timepoints,
            DEFAULT_PROTEINS,
            args.min_points,
            extra_times,
            dense_timepoints,
            lam=float(args.joint_lam),
        )

        applied_bins = 0
        for gfp_bin in gfp_bins:
            gfp_bin = int(gfp_bin)
            updates = updates_per_bin.get(gfp_bin)
            if not updates:
                continue
            meta_bin = metadata_per_bin.get(gfp_bin, {})
            erk_meta = meta_bin.get("p-ERK1-2")
            erk_r2 = erk_meta.get("r2") if erk_meta else None
            if erk_r2 is None or not np.isfinite(erk_r2) or erk_r2 < args.min_r2:
                LOGGER.debug(
                    "Skipping marker %s bin %s due to low p-ERK1-2 R²=%.3f (threshold %.3f)",
                    marker,
                    gfp_bin,
                    erk_r2 if erk_r2 is not None else float("nan"),
                    args.min_r2,
                )
                continue

            model_names = {protein: meta.get("model", "") for protein, meta in meta_bin.items()}
            r2_scores = {protein: meta.get("r2", float("nan")) for protein, meta in meta_bin.items()}

            apply_updates(
                time_trajectories,
                extra_fit,
                marker,
                gfp_bin,
                updates,
                extra_updates_per_bin.get(gfp_bin, []),
                model_names=model_names,
                r2_scores=r2_scores,
            )
            apply_extra_updates(dense_fit, marker, gfp_bin, dense_updates_per_bin.get(gfp_bin, []))
            applied_bins += 1

        if applied_bins:
            collected_r2 = [
                meta.get("r2")
                for bin_meta in metadata_per_bin.values()
                for meta in bin_meta.values()
                if meta.get("r2") is not None and np.isfinite(meta.get("r2", float("nan")))
            ]
            avg_r2 = float(np.mean(collected_r2)) if collected_r2 else float("nan")
            LOGGER.debug(
                "Applied joint fits for marker %s (bins=%d, avg_R²=%s, lam=%.2f)",
                marker,
                applied_bins,
                f"{avg_r2:.3f}" if np.isfinite(avg_r2) else "nan",
                float(args.joint_lam),
            )

    LOGGER.info("Filling extra GFP means for timepoints: %s", args.extra_times)
    fill_extra_gfp_means(extra_fit, args.extra_times)
    LOGGER.info("Filling per-minute GFP means up to %.1f minutes", max_time)
    fill_extra_gfp_means(dense_fit, dense_times)

    dest_dir = args.output_dir / args.output_prefix
    dest_dir.mkdir(parents=True, exist_ok=True)

    for frame_name, frame in {
        "time_trajectories": time_trajectories,
        "extra_fit": extra_fit,
        "dense_fit": dense_fit,
    }.items():
        overlap = set(frame.columns).intersection(frame.index.names)
        if overlap:
            raise ValueError(
                f"{frame_name} has columns that overlap index names: {sorted(overlap)}"
            )

    LOGGER.info("Writing functional-group datasets to %s", dest_dir)
    time_df = time_trajectories.reset_index()
    filtered_df = build_filtered(time_trajectories)
    extra_df = (
        extra_fit.reset_index().sort_values(["marker", "GFP_bin", "timepoint"])
    )
    dense_df = (
        dense_fit.reset_index().sort_values(["marker", "GFP_bin", "timepoint"])
    )
    drop_cols = [protein for protein in DEFAULT_PROTEINS if protein in extra_df.columns]
    if drop_cols:
        LOGGER.info("Dropping raw measurement columns from extra output: %s", drop_cols)
        extra_df = extra_df.drop(columns=drop_cols)
        dense_df = dense_df.drop(columns=drop_cols)

    output_payload = {
        "markers_time_series.csv": time_df,
        "markers_feature_matrix.csv": filtered_df,
        "markers_fit_snapshot.csv": extra_df,
        "markers_per_minute_fit.csv": dense_df,
    }

    for filename, frame in output_payload.items():
        path = dest_dir / filename
        frame.to_csv(path, index=False)
        LOGGER.info("Wrote %s", path)
    LOGGER.info("Finished generating functional group inputs")


if __name__ == "__main__":
    run_processing(parse_args())
