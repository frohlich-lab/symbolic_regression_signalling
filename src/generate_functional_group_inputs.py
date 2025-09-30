"""Generate processed time-course derivatives and fits from raw trajectories.

Recreates the logic from the `data_exploration.ipynb` notebook while giving
outputs names that match the functional-group SR context. By default the script
emits:

* ``functional_groups_time_trajectories.csv``
* ``functional_groups_filtered_features.csv``
* ``functional_groups_extra_fit_trajectories.csv``

The prefix can be customised with ``--output-prefix``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import warnings

try:
    from scipy.optimize import curve_fit
except Exception as exc:  # pragma: no cover - handled gracefully at runtime
    curve_fit = None  # type: ignore[assignment]
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
    "p-S6",
]

STATS_PROTEINS = {"p-ERK1-2", "p-MEK1-2", "p-S6"}


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
        "--min-points",
        type=int,
        default=5,
        help="Minimum non-NaN timepoints required before fitting (default 5).",
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
        default="functional_groups",
        help="Prefix used when naming the emitted CSV files (default: functional_groups).",
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


def fit_marker_bin(
    marker: str,
    gfp_bin: int,
    gfp_bin_data: pd.DataFrame,
    timepoints: np.ndarray,
    proteins: Sequence[str],
    min_points: int,
    extra_times: np.ndarray,
) -> Tuple[dict, List[Tuple[str, float, float]]]:
    updates = {}
    extra_updates: List[Tuple[str, float, float]] = []

    if curve_fit is None:
        warnings.warn(
            "SciPy is unavailable; skipping curve fitting for marker-bin combinations.",
            RuntimeWarning,
        )
        return updates, extra_updates

    x_full = timepoints
    for protein in proteins:
        y_full = gfp_bin_data[protein].to_numpy(dtype=float)
        valid = ~np.isnan(y_full)
        if valid.sum() < min_points:
            continue

        x_data = x_full[valid]
        y_data = y_full[valid]

        if np.allclose(y_data.max() - y_data.min(), 0, atol=1e-6):
            continue

        B0 = float(y_data.min())
        A_T0 = float(y_data.max() - y_data.min())
        tau_T0 = 5.0
        tau_S0 = 10.0
        p0 = [B0, max(A_T0, 1e-3), tau_T0, tau_S0]

        try:
            popt, _ = curve_fit(
                rise_and_fall,
                x_data,
                y_data,
                p0=p0,
                maxfev=10000,
                bounds=([-np.inf, 0.0, 1e-3, 1e-3], [np.inf, np.inf, 1e3, 1e3]),
            )
        except Exception:
            continue

        curve_main = rise_and_fall(x_full, *popt)
        deriv_main = rise_and_fall_derivative(x_full, *popt)

        extended_times = np.unique(np.concatenate([x_full, extra_times]))
        curve_extended = rise_and_fall(extended_times, *popt)
        deriv_extended = rise_and_fall_derivative(extended_times, *popt)

        updates.setdefault(protein, {})
        for idx, t in enumerate(x_full):
            updates[protein][t] = (curve_main[idx], deriv_main[idx])

        stats = (
            float(np.max(y_data)),
            float(np.min(y_data)),
            float(np.mean(y_data)),
            float(np.max(y_data) - np.min(y_data)),
        )

        protein_min = stats[1]

        for idx, t in enumerate(extended_times):
            extra_updates.append(
                (
                    protein,
                    float(t),
                    curve_extended[idx],
                )
            )
            if protein in STATS_PROTEINS:
                extra_updates.append(
                    (
                        f"{protein}_dt",
                        float(t),
                        deriv_extended[idx],
                    )
                )
                extra_updates.append(
                    (
                        f"{protein}_min",
                        float(t),
                        protein_min,
                    )
                )
        if protein in STATS_PROTEINS:
            updates[protein]["stats"] = stats

    return updates, extra_updates


def apply_updates(
    time_trajectories: pd.DataFrame,
    extra_fit: pd.DataFrame,
    marker: str,
    gfp_bin: int,
    updates: dict,
    extra_updates: Iterable[Tuple[str, float, float]],
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
                continue
            fit_val, deriv_val = payload
            idx = (marker, gfp_bin, timepoint)
            time_trajectories.loc[idx, f"{protein}_fit"] = fit_val
            if protein in STATS_PROTEINS:
                time_trajectories.loc[idx, f"{protein}_derivative"] = deriv_val
                time_trajectories.loc[idx, f"{protein}_dt"] = deriv_val

    for column, timepoint, value in extra_updates:
        idx = (marker, gfp_bin, timepoint)
        extra_fit.loc[idx, column] = value


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
    columns = base_cols + stat_cols + fit_cols
    filtered = time_trajectories[columns].reset_index()
    return filtered[["marker", "GFP_bin", "timepoint"] + columns]


def run_processing(args: argparse.Namespace) -> None:
    time_course = pd.read_csv(args.time_course)
    if {"marker", "GFP", "timepoint"} - set(time_course.columns):
        missing = {"marker", "GFP", "timepoint"} - set(time_course.columns)
        raise ValueError(f"time course data is missing required columns: {sorted(missing)}")

    time_course = time_course.copy()
    time_course["timepoint"] = time_course["timepoint"].astype(float)
    time_course["GFP_bin"] = assign_gfp_bins(time_course, args.gfp_bins)
    time_course = time_course.dropna(subset=["GFP_bin"])  # drop markers lacking variation
    time_course["GFP_bin"] = time_course["GFP_bin"].astype(int)

    numeric_cols = time_course.select_dtypes(include=[np.number]).columns
    time_trajectories = (
        time_course.groupby(["marker", "GFP_bin", "timepoint"])[numeric_cols]
        .mean()
        .sort_index()
    )

    initialise_columns(time_trajectories, DEFAULT_PROTEINS)
    extra_fit = initialise_extra_frame(time_trajectories, DEFAULT_PROTEINS)

    markers = time_trajectories.index.get_level_values("marker").unique()
    raw_timepoints = np.sort(time_course["timepoint"].dropna().unique())
    extra_times = np.array(args.extra_times, dtype=float)

    for marker in markers:
        marker_slice = time_trajectories.loc[marker]
        for gfp_bin in marker_slice.index.get_level_values("GFP_bin").unique():
            bin_slice = marker_slice.loc[gfp_bin]
            bin_slice = bin_slice.reindex(raw_timepoints)
            updates, extra_updates = fit_marker_bin(
                marker,
                int(gfp_bin),
                bin_slice,
                raw_timepoints,
                DEFAULT_PROTEINS,
                args.min_points,
                extra_times,
            )
            if not updates and not extra_updates:
                continue
            apply_updates(
                time_trajectories,
                extra_fit,
                marker,
                int(gfp_bin),
                updates,
                extra_updates,
            )

    fill_extra_gfp_means(extra_fit, args.extra_times)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base = args.output_prefix
    time_trajectories.reset_index().to_csv(
        output_dir / f"{base}_time_trajectories.csv",
        index=False,
    )
    build_filtered(time_trajectories).to_csv(
        output_dir / f"{base}_filtered_features.csv",
        index=False,
    )
    extra_fit.reset_index().sort_values(["marker", "GFP_bin", "timepoint"]).to_csv(
        output_dir / f"{base}_extra_fit_trajectories.csv",
        index=False,
    )


if __name__ == "__main__":
    run_processing(parse_args())
