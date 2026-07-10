"""Generate functional group SR plots from stored summary results."""
from __future__ import annotations

import argparse
import ast
import logging
import math
import shutil
from pathlib import Path
import sys
from posixpath import basename
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.data_prep.utils import signed_log

LOGGER = logging.getLogger("experimental.plot_functional_groups")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

try:
    from upsetplot import UpSet, from_contents
    HAS_UPSETPLOT = True
except Exception:
    HAS_UPSETPLOT = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create diagnostic plots for functional group symbolic regression results.")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to the functional_group_sr_summary.csv file produced by run_functional_groups.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--group-definitions-csv",
        type=Path,
        default=Path("data/experimental/processed/functional_groups.csv"),
        help="CSV file containing functional group metadata (columns: group, members, direction, confidence).",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="log_r2_bar",
        help="Base filename (without extension) for the emitted plots.",
    )
    parser.add_argument(
        "--alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the bar chart (optional).",
    )
    parser.add_argument(
        "--relative-basename",
        type=str,
        default="relative_mae_scatter",
        help="Base filename (without extension) for the relative error scatter plot.",
    )
    parser.add_argument(
        "--relative-alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the relative error scatter (optional).",
    )
    parser.add_argument(
        "--r2-basename",
        type=str,
        default="log_r2_scatter_size",
        help="Base filename (without extension) for the log R² scatter plot.",
    )
    parser.add_argument(
        "--r2-alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the log R² scatter plot (optional).",
    )
    parser.add_argument(
        "--r2-confidence-basename",
        type=str,
        default="log_r2_scatter_confidence",
        help="Base filename for the log R² scatter coloured by confidence.",
    )
    parser.add_argument(
        "--r2-confidence-alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the confidence-coloured scatter (optional).",
    )
    parser.add_argument(
        "--r2-direction-basename",
        type=str,
        default="log_r2_scatter_direction",
        help="Base filename for the log R² scatter coloured by direction.",
    )
    parser.add_argument(
        "--r2-direction-alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the direction-coloured scatter (optional).",
    )
    parser.add_argument(
        "--r2-confidence-high-basename",
        type=str,
        default="log_r2_scatter_confidence_high",
        help="Base filename for the log R² scatter coloured by confidence (confidence ≥ threshold).",
    )
    parser.add_argument(
        "--comparison-gfp-basename",
        type=str,
        default="functional_group_model_comparison_gfp",
        help="Base filename for the PySR vs Linear Regression scatter (GFP feature mode).",
    )
    parser.add_argument(
        "--comparison-all-basename",
        type=str,
        default="functional_group_model_comparison_all",
        help="Base filename for the PySR vs Linear Regression scatter (all features).",
    )
    parser.add_argument(
        "--comparison-gfp-high-basename",
        type=str,
        default="functional_group_model_comparison_gfp_high",
        help="Base filename for the PySR vs Linear Regression scatter restricted to high-performing groups (GFP mode).",
    )
    parser.add_argument(
        "--comparison-all-high-basename",
        type=str,
        default="functional_group_model_comparison_all_high",
        help="Base filename for the PySR vs Linear Regression scatter restricted to high-performing groups (all features).",
    )
    parser.add_argument(
        "--r2-confidence-high-alias-basename",
        type=str,
        default=None,
        help="Legacy base filename for the high-confidence scatter (optional).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=4.0,
        help="Minimum confidence value to include when plotting the high-confidence scatter.",
    )
    parser.add_argument(
        "--variant",
        choices=("legacy", "combined", "fresh", "per_minute"),
        default="legacy",
        help="Which group set to visualise: legacy (default), combined (legacy + curated), fresh only, or per_minute for marker-level runs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PySR",
        help="Model name to visualise when the summary includes multiple models (default: PySR).",
    )
    parser.add_argument(
        "--upset-gfp-basename",
        type=str,
        default="upset_gfp",
        help="Base filename for the UpSet plot of GFP-only features (constituents by direction).",
    )
    parser.add_argument(
        "--upset-all-basename",
        type=str,
        default="upset_all",
        help="Base filename for the UpSet plot of GFP + neighbouring nodes features (constituents by direction).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset CSV (functional_groups_fit_snapshot.csv) used to evaluate PySR fits.",
    )
    parser.add_argument(
        "--log10-cutoff",
        type=float,
        default=-3.0,
        help="Signed-log cutoff applied when plotting trajectory fits (matches training default).",
    )
    parser.add_argument(
        "--fit-groups",
        nargs="*",
        default=(),
        help="Group names for which to render PySR trajectory fit grids (log space).",
    )
    parser.add_argument(
        "--fit-grid-columns",
        type=int,
        default=5,
        help="Number of subplot columns when rendering trajectory fit grids.",
    )
    parser.add_argument(
        "--fit-output-subdir",
        type=str,
        default="pysr/fits",
        help="Relative subdirectory (under --output-dir) for PySR trajectory fit plots.",
    )
    return parser.parse_args()


def _sanitize_column_name(label: str) -> str:
    clean = label.replace("-", "_")
    clean = clean.replace(" ", "_")
    clean = clean.replace("(", "")
    clean = clean.replace(")", "")
    clean = clean.replace("/", "_")
    if clean and clean[0].isdigit():
        clean = f"f_{clean}"
    return clean


def _sanitize_column_map(columns: Iterable[str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for col in columns:
        base = _sanitize_column_name(str(col))
        counter = counts.get(base, 0)
        if counter == 0:
            new_name = base
        else:
            new_name = f"{base}_{counter}"
        counts[base] = counter + 1
        rename[str(col)] = new_name
    return rename


def _load_sr_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    rename = _sanitize_column_map(df.columns)
    df = df.rename(columns=rename)
    fit_renames: Dict[str, str] = {}
    for col in list(df.columns):
        if col.endswith("_fit"):
            base = col[:-4]
            if base and base not in df.columns:
                fit_renames[col] = base
    if fit_renames:
        df = df.rename(columns=fit_renames)
    if "marker" in df.columns:
        df["marker"] = df["marker"].astype(str)
    if "GFP_bin" in df.columns:
        df["GFP_bin"] = pd.to_numeric(df["GFP_bin"], errors="coerce")
    if "timepoint" in df.columns:
        df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    return df

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

def sanity_check_predictions(df: pd.DataFrame,
                             target_cols=("p-ERK1-2_dt", "p_ERK1_2_dt"),
                             pred_col: str = "pysr_prediction",
                             log10_cutoff: float = -3.0,
                             group_name: str | None = None,
                             marker: str | None = None) -> None:
    # pick first present target col
    if isinstance(target_cols, (list, tuple)):
        present = [c for c in target_cols if c in df.columns]
        if not present:
            raise KeyError(f"No target column found. Tried {target_cols}. "
                           f"Available columns (first 25): {list(df.columns[:25])}")
        target_col = present[0]
    else:
        target_col = target_cols
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not in DataFrame columns.")

    if pred_col not in df.columns:
        raise KeyError(f"Prediction column '{pred_col}' not in DataFrame columns.")

    eps = 10.0 ** log10_cutoff
    subset = df[[target_col, pred_col]].copy()

    # ensure numeric
    subset[target_col] = pd.to_numeric(subset[target_col], errors="coerce")
    subset[pred_col]   = pd.to_numeric(subset[pred_col], errors="coerce")

    n_total = len(subset)
    n_nan = subset.isna().sum().to_dict()
    print(f"\n[Sanity check] group={group_name or '?'} marker={marker or '?'} "
          f"rows={n_total} NaNs={n_nan} target_col={target_col}")

    mask = np.isfinite(subset[target_col]) & np.isfinite(subset[pred_col])
    if mask.sum() < 5:
        print("  ⚠️  Too few finite samples for reliable stats.")
        return

    y = subset.loc[mask, target_col].to_numpy()
    yhat = subset.loc[mask, pred_col].to_numpy()

    # tiny→zero (match training)
    y[np.abs(y) < eps] = 0.0
    yhat[np.abs(yhat) < eps] = 0.0

    r2_lin = r2_score(y, yhat)
    mae_lin = mean_absolute_error(y, yhat)

    def sgn_log(v): return np.sign(v) * (np.log10(np.abs(v) + eps) - log10_cutoff)
    y_log, yhat_log = sgn_log(y), sgn_log(yhat)
    r2_log = r2_score(y_log, yhat_log)
    mae_log = mean_absolute_error(y_log, yhat_log)

    print(f"  Linear-space  : R²={r2_lin:6.3f} | MAE={mae_lin:9.3e} "
          f"| y∈[{y.min():.2e},{y.max():.2e}]  ŷ∈[{yhat.min():.2e},{yhat.max():.2e}]")
    print(f"  Signed-logspace: R²={r2_log:6.3f} | MAE={mae_log:9.3e} "
          f"| log10_cutoff={log10_cutoff} ({eps:.1e})")

    if not np.isfinite(r2_lin):
        print("  ⚠️  Non-finite linear R² → check formula eval or variable mapping.")
    if np.allclose(yhat, 0) or np.std(yhat) < 1e-8:
        print("  ⚠️  Prediction array is (almost) constant.")

def _evaluate_pysr_formula(dataset: pd.DataFrame, formula: str) -> np.ndarray:
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("PySR formula is empty; cannot evaluate trajectory fit.")

    # ---- make parsing robust ----
    # 1) normalize power operator
    safe = formula.replace("^", "**")
    # 2) disallow unknown names turning into free symbols by mistake
    #    (e.g., "relu" without a mapping). You can extend allowed funcs here.
    allowed = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "exp": sp.exp, "log": sp.log, "abs": sp.Abs}
    try:
        expr = sp.sympify(safe, locals=allowed)  # only allow known funcs
    except Exception as e:
        raise ValueError(f"Failed to parse PySR formula: {e}\nformula={formula}")

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if not symbols:
        value = float(expr)
        return np.full(len(dataset), value, dtype=float)

    arrays = []
    for s in symbols:
        name = s.name
        if name not in dataset.columns:
            raise KeyError(
                f"Formula references '{name}', but dataset columns are missing it. "
                f"Have: {list(dataset.columns[:10])}..."
            )
        arrays.append(pd.to_numeric(dataset[name], errors="coerce").to_numpy())

    func = sp.lambdify(symbols, expr, modules={"numpy": np, **allowed})
    out = np.asarray(func(*arrays), dtype=float)
    return out


def _markers_for_group(metadata: pd.DataFrame, group_name: str) -> List[str]:
    if "group" not in metadata.columns:
        return []
    subset = metadata[metadata["group"].astype(str) == str(group_name)]
    if subset.empty:
        return []
    if "members_list" in subset.columns and subset["members_list"].notna().any():
        first = subset["members_list"].dropna().iloc[0]
        return list(first)
    if "members" in subset.columns:
        raw = subset["members"].dropna().astype(str).iloc[0]
        try:
            parsed = ast.literal_eval(raw)
            return [str(m).strip() for m in parsed]
        except Exception:
            return [m.strip().strip("'\"") for m in raw.strip("[]").split(",") if m.strip()]
    return []

def pick_formula_row(summary: pd.DataFrame, group_name: str, prefer="best") -> pd.Series:
    rows = summary[summary["group_name"].astype(str) == str(group_name)].copy()
    rows = rows[rows["model"].str.lower() == "pysr"] if "model" in rows.columns else rows
    if rows.empty:
        raise ValueError(f"No PySR rows for group '{group_name}'")

    if prefer in ("gfp","all"):
        rows = rows[rows["feature_mode"].str.lower()==prefer]
        if rows.empty:
            raise ValueError(f"No rows for mode '{prefer}' in group '{group_name}'")
        return rows.iloc[0]

    # prefer == "best"
    if "test_log_r2" in rows.columns:
        return rows.sort_values("test_log_r2", ascending=False).iloc[0]
    return rows.iloc[0]

def _r2_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if y.size < 2:
        return np.nan
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)

def sanitize_feature_names(columns):
    renamed = []
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

def plot_pysr_fit_grids(
    dataset: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    groups: Iterable[str],
    output_dir: Path,
    *,
    log10_cutoff: float,
    grid_columns: int,
) -> None:
    # ---- 1) Determine target column (support both raw and sanitized) ----
    raw_target = "p-ERK1-2_dt"
    san_target = "p_ERK1_2_dt"
    if raw_target in dataset.columns:
        target_column = raw_target
    elif san_target in dataset.columns:
        target_column = san_target
    else:
        LOGGER.warning(
            "Dataset missing target derivative column ('%s' or '%s'); skipping PySR fit plots",
            raw_target, san_target,
        )
        return

    # ---- 2) Quick set of groups that exist in summary ----
    available_groups = {str(g) for g in summary.get("group_name", pd.Series(dtype=str)).unique()}

    for group_name in groups:
        if str(group_name) not in available_groups:
            LOGGER.warning("Summary has no PySR entry for group '%s'; skipping", group_name)
            continue

        # Use the GFP-mode formula for this group (as per your intention)
        formula_row = summary[
            (summary["group_name"].astype(str) == str(group_name))
            & (summary.get("feature_mode", pd.Series(dtype=str)).astype(str).str.lower() == "gfp")
        ]
        if formula_row.empty:
            LOGGER.warning(
                "No PySR GFP-mode formula found for group '%s'; skipping trajectory fit plot",
                group_name,
            )
            continue
        formula = str(formula_row.iloc[0]["formula"])

        # Markers for this group
        markers = _markers_for_group(metadata, str(group_name))
        if not markers:
            LOGGER.warning("Metadata missing marker list for group '%s'; skipping", group_name)
            continue

        # --- One figure per marker ---
        for marker in markers:
            # ---- 3) Slice marker data and SANITIZE COLUMN NAMES to match training ----
            marker_subset_raw = dataset[dataset.get("marker", "").astype(str) == str(marker)].copy()
            if marker_subset_raw.empty:
                LOGGER.warning("Dataset has no rows for group '%s' marker '%s'; skipping", group_name, marker)
                continue

            # Keep a mapping to rename columns exactly as in training
            sanitized_names = sanitize_feature_names(marker_subset_raw.columns)
            rename_map = dict(zip(marker_subset_raw.columns, sanitized_names))
            marker_subset = marker_subset_raw.rename(columns=rename_map)

            # After sanitization, set the correct target column name
            if raw_target in marker_subset.columns:
                tgt_col = raw_target
            elif san_target in marker_subset.columns:
                tgt_col = san_target
            else:
                LOGGER.warning(
                    "After sanitizing, no target column found for group '%s' marker '%s'; skipping",
                    group_name, marker,
                )
                continue

            # ---- 4) Verify formula variables are present in the sanitized DF ----
            try:
                import sympy as sp
                expr = sp.sympify(formula)
                vars_in_formula = {str(s) for s in expr.free_symbols}
            except Exception as exc:
                LOGGER.warning("Failed to parse PySR formula for '%s': %s", group_name, exc)
                continue

            missing = [v for v in vars_in_formula if v not in marker_subset.columns]
            if missing:
                # Helpful message to show what columns we actually have
                sample_cols = ", ".join(list(marker_subset.columns)[:8]) + ("..." if marker_subset.shape[1] > 8 else "")
                LOGGER.warning(
                    "Formula variables %s not found in sanitized columns for group '%s' marker '%s'. "
                    "Ensure feature names match training (e.g., 'p-ERK1-2' -> 'p_ERK1_2'). "
                    "Have columns: %s. Skipping.",
                    missing, group_name, marker, sample_cols,
                )
                continue

            # Evaluate PySR model on the SANITIZED DataFrame
            try:
                predictions = _evaluate_pysr_formula(marker_subset, formula)  # expects sanitized columns
            except Exception as exc:
                LOGGER.warning("Failed to evaluate PySR formula for group '%s' marker '%s': %s",
                               group_name, marker, exc)
                continue

            marker_subset["pysr_prediction"] = predictions


            marker_subset.replace([np.inf, -np.inf], np.nan, inplace=True)

            sanity_check_predictions(
                marker_subset,
                target_cols=("p-ERK1-2_dt", "p_ERK1_2_dt"),
                pred_col="pysr_prediction",
                log10_cutoff=log10_cutoff,
                group_name=group_name,
                marker=marker,
            )

            # Require these columns to exist in sanitized form for plotting
            required_cols = {tgt_col, "pysr_prediction", "GFP_bin", "timepoint"}
            missing_req = [c for c in required_cols if c not in marker_subset.columns]
            if missing_req:
                LOGGER.warning(
                    "Missing required columns %s after sanitization for group '%s' marker '%s'; skipping",
                    missing_req, group_name, marker,
                )
                continue

            marker_subset = marker_subset.dropna(subset=list(required_cols))
            if marker_subset.empty:
                LOGGER.warning(
                    "After dropping NaNs there are no samples left for group '%s' marker '%s'; skipping",
                    group_name, marker,
                )
                continue

            # ---- 5) Compute signed-log transforms (target + prediction) ----
            marker_subset["target_log"] = signed_log(marker_subset[tgt_col].to_numpy(), log10_cutoff)
            marker_subset["prediction_log"] = signed_log(
                marker_subset["pysr_prediction"].to_numpy(), log10_cutoff
            )
            marker_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
            marker_subset = marker_subset.dropna(subset=["target_log", "prediction_log"])
            if marker_subset.empty:
                LOGGER.warning(
                    "No finite log-space samples remain for group '%s' marker '%s'; skipping",
                    group_name, marker,
                )
                continue

            # ---- 6) GFP bins present for this marker ----
            try:
                bins = sorted({int(b) for b in marker_subset["GFP_bin"].dropna().unique()})
            except Exception:
                bins = []
            if not bins:
                LOGGER.warning("Group '%s' marker '%s' has no valid GFP bins; skipping", group_name, marker)
                continue

            # ---- 7) Choose a p-ERK value column (linear, sanitized names) ----
            value_col = None
            for candidate in ("p_ERK1_2", "p_ERK1_2_obs", "p_ERK1_2_measured", "p_ERK1_2_fit"):
                if candidate in marker_subset.columns:
                    value_col = candidate
                    break
            if value_col is None:
                # Try raw names as a fallback
                for candidate in ("p-ERK1-2", "p-ERK1-2_obs", "p-ERK1-2_measured", "p-ERK1-2_fit"):
                    if candidate in marker_subset_raw.columns:
                        # bring it in (sanitized name already present if we rename)
                        raw_vals = marker_subset_raw[candidate].values
                        marker_subset["p_ERK1_2_fallback"] = raw_vals
                        value_col = "p_ERK1_2_fallback"
                        break
            if value_col is None:
                LOGGER.warning(
                    "Dataset for group '%s' marker '%s' is missing a p-ERK1/2 column; skipping",
                    group_name, marker,
                )
                continue
            if "timepoint" not in marker_subset.columns:
                LOGGER.warning("Dataset missing 'timepoint' column; skipping group '%s' marker '%s'",
                               group_name, marker)
                continue

            # ---- 8) Build per-bin curves (median over timepoints) ----
            bin_curves: Dict[int, pd.DataFrame] = {}
            for bin_id in bins:
                bin_slice = marker_subset[marker_subset["GFP_bin"] == bin_id].copy()
                if bin_slice.empty:
                    continue
                aggregated = (
                    bin_slice.groupby("timepoint")[["target_log", "prediction_log", value_col]]
                    .median()
                    .reset_index()
                    .sort_values("timepoint")
                )
                aggregated = aggregated.replace([np.inf, -np.inf], np.nan).dropna(subset=[value_col])
                if aggregated.empty:
                    continue

                # x-axis: signed log of -pERK (as in your original code)
                neg_values = -aggregated[value_col].to_numpy(dtype=float)
                signed_neg = signed_log(neg_values, log10_cutoff)
                if np.unique(signed_neg).size < 2:
                    continue
                aggregated["neg_value_signed_log"] = signed_neg
                aggregated["timepoint"] = aggregated["timepoint"].astype(float)
                bin_curves[bin_id] = aggregated

            if not bin_curves:
                LOGGER.warning("Group '%s' marker '%s' has no usable curve data; skipping", group_name, marker)
                continue

            # ---- 9) Axis limits and layout ----
            all_values = np.concatenate([df["neg_value_signed_log"].to_numpy(float) for df in bin_curves.values()])
            all_targets = np.concatenate([df["target_log"].to_numpy(float) for df in bin_curves.values()])
            all_predictions = np.concatenate([df["prediction_log"].to_numpy(float) for df in bin_curves.values()])
            x_span = all_values.max() - all_values.min()
            x_pad = max(x_span * 0.05, 0.1)
            import math
            y_values = np.concatenate([all_targets, all_predictions])
            y_pad = max((np.nanmax(y_values) - np.nanmin(y_values)) * 0.08, 0.15)

            n_panels = len(bin_curves)
            n_cols = max(1, min(grid_columns, n_panels))
            n_rows = math.ceil(n_panels / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 2.6))
            axes_arr = np.atleast_1d(axes).flatten()
            for ax in axes_arr[n_panels:]:
                ax.axis("off")

            legend_handles = None
            legend_labels = None

            # ---- 10) Draw panels ----
            for idx, (bin_id, curve_df) in enumerate(bin_curves.items()):
                ax = axes_arr[idx]

                # per-trajectory (bin) R² in log space
                r2_val = _r2_score(curve_df["target_log"], curve_df["prediction_log"])
                r2_text = f"{r2_val:.3f}" if np.isfinite(r2_val) else "NA"

                obs_line = ax.plot(
                    curve_df["neg_value_signed_log"], curve_df["target_log"],
                    color="#1b9e77", linewidth=1.8, label="Observed",
                )[0]
                ax.scatter(
                    curve_df["neg_value_signed_log"], curve_df["target_log"],
                    color="#1b9e77", s=16, zorder=3,
                )
                pred_line = ax.plot(
                    curve_df["neg_value_signed_log"], curve_df["prediction_log"],
                    color="#d95f02", linewidth=1.8, linestyle="--", label="PySR",
                )[0]
                ax.scatter(
                    curve_df["neg_value_signed_log"], curve_df["prediction_log"],
                    color="#d95f02", s=16, zorder=3,
                )

                if legend_handles is None or legend_labels is None:
                    legend_handles = [obs_line, pred_line]
                    legend_labels = ["Observed", "PySR"]

                tp_labels = ", ".join(f"{t:g}" for t in sorted(curve_df["timepoint"].unique()))
                ax.set_title(
                    f"GFP bin {bin_id} • R²={r2_text} (n={len(curve_df)})\nΔt: {tp_labels}",
                    fontsize=9.8
                )
                ax.grid(True, linewidth=0.45, alpha=0.35)
                ax.set_facecolor("#fafafa")
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)

            for idx_ax, ax in enumerate(axes_arr[:n_panels]):
                row_idx = idx_ax // n_cols
                col_idx = idx_ax % n_cols
                ax.set_xlabel("Signed log10(-p-ERK1/2)" if row_idx == n_rows - 1 else "")
                ax.set_ylabel("Signed log10 d/dt p-ERK1/2" if col_idx == 0 else "")
                ax.tick_params(labelsize=9.5)

            # Optional: set consistent limits if you want tighter visuals
            # for ax in axes_arr[:n_panels]:
            #     ax.set_xlim(all_values.min() - x_pad, all_values.max() + x_pad)
            #     ax.set_ylim(np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)

            fig.tight_layout(rect=(0.05, 0.1, 0.97, 0.92))
            if legend_handles and legend_labels:
                fig.legend(
                    legend_handles, legend_labels,
                    loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985), frameon=False,
                )

            fig.suptitle(f"{group_name} • {marker} • PySR fit vs observed (log space)", fontsize=14, y=0.999)

            group_slug = _sanitize_column_name(str(group_name).lower())
            marker_slug = _sanitize_column_name(str(marker).lower())
            png_path = output_dir / f"{group_slug}_{marker_slug}.png"
            svg_path = output_dir / f"{group_slug}_{marker_slug}.svg"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=400, bbox_inches="tight")
            fig.savefig(svg_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            LOGGER.info(
                "Wrote PySR trajectory fit plots for group '%s' marker '%s' → %s, %s",
                group_name, marker, png_path.name, svg_path.name,
            )


def _write_alias(png_path: Path, svg_path: Path, alias_basename: Optional[str]) -> None:
    if not alias_basename:
        return
    alias_png = png_path.with_name(f"{alias_basename}.png")
    alias_svg = svg_path.with_name(f"{alias_basename}.svg")
    try:
        alias_png.parent.mkdir(parents=True, exist_ok=True)
        alias_svg.parent.mkdir(parents=True, exist_ok=True)
        if alias_png != png_path:
            shutil.copyfile(png_path, alias_png)
        if alias_svg != svg_path:
            shutil.copyfile(svg_path, alias_svg)
    except Exception as exc:
        LOGGER.warning("Failed to write alias copies %s / %s: %s", alias_png.name, alias_svg.name, exc)

def _prettify_label(raw: str) -> str:
    label = raw.replace("_", " ")
    tokens = label.split()
    pretty: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower == "and":
            pretty.append("&")
        elif token.islower():
            pretty.append(token.capitalize())
        else:
            pretty.append(token)
    return " ".join(pretty)


def _load_group_metadata(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Group definitions CSV not found: {csv_path}")
    meta = pd.read_csv(csv_path)
    if "group" not in meta.columns:
        raise ValueError("Group definitions CSV must include a 'group' column")
    if "members" in meta.columns:
        def _parse_members(raw: object) -> List[str]:
            if pd.isna(raw):
                return []
            text = str(raw)
            try:
                parsed = ast.literal_eval(text)
                return [str(m).strip() for m in parsed]
            except Exception:
                return [m.strip().strip("'\"") for m in text.strip("[]").split(",") if m.strip()]

        meta["members_list"] = meta["members"].apply(_parse_members)
    return meta


def _filter_summary(summary: pd.DataFrame, metadata: pd.DataFrame, variant: str) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata["group"] = metadata["group"].astype(str)

    merged = summary.merge(
        metadata,
        left_on="group_name",
        right_on="group",
        how="left",
        suffixes=("", "_meta"),
    )

    if variant == "per_minute":
        return merged.copy()

    missing_meta = merged["group"].isna()
    if missing_meta.any():
        LOGGER.warning(
            "%d groups in summary missing metadata entries",
            int(missing_meta.sum()),
        )

    if "variant" in metadata.columns:
        if variant == "legacy":
            filtered = merged[merged["variant"].fillna("legacy") == "legacy"].copy()
        elif variant == "combined":
            filtered = merged.copy()
        elif variant == "fresh":
            filtered = merged[merged["variant"].fillna("") == "fresh"].copy()
        else:
            raise ValueError(f"Unsupported variant: {variant}")
    else:
        if variant != "legacy":
            LOGGER.warning(
                "Group metadata has no 'variant' column; returning all groups for variant '%s'",
                variant,
            )
        filtered = merged.copy()

    return filtered


def _plot_kde_marginal(
    ax: plt.Axes,
    values: np.ndarray,
    limits: tuple[float, float],
    orientation: str,
    log_scale: bool,
) -> None:
    finite = np.isfinite(values)
    data = values[finite]
    if log_scale:
        data = data[data > 0]
    if data.size < 2:
        ax.grid(False)
        if orientation == "top":
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylabel("")
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        return

    if log_scale:
        low, high = np.log10(limits[0]), np.log10(limits[1])
        grid = np.linspace(low, high, 256)
        kde = gaussian_kde(np.log10(data))
        density = kde(grid)
        coords = 10 ** grid
    else:
        coords = np.linspace(limits[0], limits[1], 256)
        kde = gaussian_kde(data)
        density = kde(coords)

    peak = float(density.max()) if density.size else 0.0
    if peak > 0:
        density = density / peak

    color = "#2c7fb8"
    if orientation == "top":
        ax.fill_between(coords, density, color=color, alpha=0.45)
        ax.plot(coords, density, color=color, linewidth=1.3)
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_ylabel("Density", fontsize=11)
    else:
        ax.fill_betweenx(coords, density, color=color, alpha=0.45)
        ax.plot(density, coords, color=color, linewidth=1.3)
        ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Density", fontsize=11)

    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _first_numeric(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    drop = numeric.dropna()
    return float(drop.iloc[0]) if not drop.empty else np.nan


def _group_metadata(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {}
    if "train_samples" in df.columns:
        agg_spec["train_samples"] = ("train_samples", "max")
    if "test_samples" in df.columns:
        agg_spec["test_samples"] = ("test_samples", "max")

    confidence_col = next(
        (col for col in ("confidence_first", "confidence") if col in df.columns),
        None,
    )
    if confidence_col is not None:
        agg_spec["confidence"] = (confidence_col, _first_numeric)

    direction_col = next(
        (col for col in ("direction_first", "direction") if col in df.columns),
        None,
    )
    if direction_col is not None:
        agg_spec["direction"] = (direction_col, _first_numeric)

    if not agg_spec:
        return pd.DataFrame(index=df.groupby("pretty_group").size().index)

    grouped = df.groupby("pretty_group").agg(**agg_spec)
    return grouped


def plot_log_r2(summary: pd.DataFrame, output_dir: Path, basename: str, variant: str, alias_basename: Optional[str] = None) -> None:
    df = summary.dropna(subset=["test_log_r2"]).copy()
    if df.empty:
        LOGGER.warning("Summary contains no valid log R² values; skipping plot generation")
        return

    df["test_log_r2"] = df["test_log_r2"].clip(lower=0.0)

    df["pretty_group"] = df["group_name"].apply(_prettify_label)

    marker_lookup: Dict[str, List[str]] = {}
    for _, row in df[["pretty_group", "markers"]].drop_duplicates("pretty_group").iterrows():
        markers_raw = row["markers"]
        parsed: List[str]
        if isinstance(markers_raw, (list, tuple)):
            parsed = [str(m) for m in markers_raw]
        else:
            try:
                parsed = [str(m).strip() for m in ast.literal_eval(str(markers_raw))]
            except Exception:
                cleaned = str(markers_raw).strip("[]")
                parsed = [m.strip().strip("'\"") for m in cleaned.split(",") if m.strip()]
        marker_lookup[row["pretty_group"]] = parsed

    pivot = df.pivot_table(index="pretty_group", columns="feature_mode", values="test_log_r2")
    modes = sorted(pivot.columns.tolist())

    if "gfp" in pivot.columns:
        order = pivot["gfp"].fillna(-np.inf).sort_values(ascending=False).index
    else:
        order = pivot.max(axis=1).fillna(-np.inf).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    groups = list(pivot.index)

    a4_width_in = 8.27
    a4_height_in = 11.69
    fig_width = max(1.3 * a4_width_in / 2, 8.0)
    base_height = (3 / 5) * a4_height_in
    fig_height = base_height

    plt.rcParams.update({
        "font.family": "Helvetica",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.titlepad": 13,
    })

    if variant == "combined":
        fig_height = max(base_height * 1.1, len(groups) * 0.5)
    elif variant == "fresh":
        fig_height = max(base_height * 0.9, len(groups) * 0.45)
    else:
        fig_height = max(base_height, len(groups) * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    y = np.arange(len(groups))
    height = 0.35 if len(modes) == 2 else 0.6
    mode_labels = {
        "gfp": "GFP",
        "all": "GFP + neighbouring nodes",
    }
    mode_colors = {
        "all": "#1f77b4",
        "gfp": "#ff7f0e",
    }

    max_val = float(np.nanmax(pivot.values)) if pivot.size else 1.0

    for idx, mode in enumerate(modes):
        mode_values = pivot[mode].values
        offset = (idx - (len(modes) - 1) / 2) * height
        if len(modes) == 2:
            if mode == "gfp":
                offset = -abs(offset)
            else:
                offset = abs(offset)
        label = mode_labels.get(mode, mode)
        color = mode_colors.get(mode)
        ax.barh(y + offset, mode_values, height=height, label=label, color=color)
        for y_pos, value in zip(y, mode_values):
            ax.text(
                value + max_val * 0.02,
                y_pos + offset,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                color="#333333",
            )

    if "train_samples" not in df.columns:
        df["train_samples"] = np.nan
    if "test_samples" not in df.columns:
        df["test_samples"] = np.nan

    samples = df.groupby("pretty_group")[["train_samples", "test_samples"]].first()
    y_labels: List[str] = []

    def _safe_count(val: float) -> int:
        if pd.isna(val):
            return 0
        try:
            return int(val)
        except Exception:
            return 0

    for group in groups:
        if group in samples.index:
            train = _safe_count(samples.loc[group, "train_samples"])
            test = _safe_count(samples.loc[group, "test_samples"])
        else:
            train = test = 0
        lines = [group]
        proteins = marker_lookup.get(group, [])
        if len(proteins) > 2:
            lines.append(", ".join(proteins))
        # For per-marker per-minute runs, skip train/test annotations if values are missing.
        if not (variant == "per_minute" and train == 0 and test == 0):
            lines.append(f"train={train:,} test={test:,}")
        y_labels.append("\n".join(lines))

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test R² (log$_{10}$ space)")
    ax.set_title("")
    ax.legend(title="Perturbational features", loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False, prune=None))

    ax.set_xlim(0, max(max_val * 1.2, 0.5))

    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    LOGGER.info("Wrote plots: %s, %s", png_path.name, svg_path.name)
    _write_alias(png_path, svg_path, alias_basename)


def plot_relative_mae_scatter(
    summary: pd.DataFrame, output_dir: Path, basename: str, variant: str, alias_basename: Optional[str] = None
) -> None:
    metric_column = "test_log_relative_mae" if "test_log_relative_mae" in summary.columns else None
    if metric_column is None:
        if "test_relative_mae" in summary.columns:
            metric_column = "test_relative_mae"
            LOGGER.warning(
                "Summary missing 'test_log_relative_mae'; falling back to linear relative MAE"
            )
        else:
            LOGGER.warning(
                "Summary missing relative MAE metrics entirely; rerun run_functional_groups.py"
            )
            return

    df = summary.dropna(subset=[metric_column]).copy()
    if df.empty:
        LOGGER.warning(
            "Summary contains no relative MAE values; skipping scatter plot"
        )
        return

    df["pretty_group"] = df["group_name"].apply(_prettify_label)

    pivot = df.pivot_table(
        index="pretty_group", columns="feature_mode", values=metric_column
    )
    required_modes = ["gfp", "all"]
    missing_modes = [mode for mode in required_modes if mode not in pivot.columns]
    if missing_modes:
        LOGGER.warning(
            "Cannot plot relative MAE scatter; missing modes: %s",
            ", ".join(missing_modes),
        )
        return

    scatter_df = pivot.dropna(subset=required_modes)
    if scatter_df.empty:
        LOGGER.warning(
            "Relative MAE scatter has no overlapping groups across modes; skipping"
        )
        return

    if metric_column == "test_log_relative_mae":
        scatter_df = (scatter_df * 100.0).clip(lower=1e-6)
        positive_mask = (scatter_df["gfp"] > 0) & (scatter_df["all"] > 0)
        axis_label = "Log-relative MAE (%)"
        use_log_axes = True
    else:
        positive_mask = (scatter_df["gfp"] > 0) & (scatter_df["all"] > 0)
        axis_label = "Relative MAE"
        use_log_axes = True
    if not positive_mask.all():
        dropped_groups = scatter_df.index[~positive_mask].tolist()
        LOGGER.warning(
            "Skipping groups with non-positive relative MAE: %s",
            ", ".join(dropped_groups[:5]) + ("..." if len(dropped_groups) > 5 else ""),
        )
        scatter_df = scatter_df.loc[positive_mask]

    if scatter_df.empty:
        LOGGER.warning("All groups had non-positive relative errors; skipping scatter plot")
        return

    grouped_meta = _group_metadata(df)
    zero_series = pd.Series(0.0, index=grouped_meta.index)
    train_series = grouped_meta.get("train_samples", zero_series).fillna(0.0)
    test_series = grouped_meta.get("test_samples", zero_series).fillna(0.0)
    scatter_samples = (train_series + test_series).reindex(scatter_df.index)

    if scatter_samples.isna().any():
        scatter_samples = scatter_samples.fillna(scatter_samples.dropna().median())

    sizes = scatter_samples.values.astype(float)
    if sizes.size == 0:
        LOGGER.warning("No sample size information available; skipping scatter plot")
        return

    min_size = float(np.min(sizes))
    max_size = float(np.max(sizes))
    if max_size == min_size:
        norm = plt.Normalize(vmin=min_size - 1.0, vmax=max_size + 1.0)
    else:
        norm = plt.Normalize(vmin=min_size, vmax=max_size)
    cmap = LinearSegmentedColormap.from_list("group_size_gradient", ["#d8e9ff", "#001f3f"])

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlepad": 12,
    })

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    x_vals = scatter_df["gfp"].values
    y_vals = scatter_df["all"].values
    min_limit = float(min(np.min(x_vals), np.min(y_vals)))
    max_limit = float(max(np.max(x_vals), np.max(y_vals)))
    if not np.isfinite(min_limit) or not np.isfinite(max_limit):
        LOGGER.warning("Encountered non-finite relative MAE values; skipping scatter plot")
        return

    if use_log_axes:
        epsilon = 1e-6
        min_limit = max(min_limit * 0.8, epsilon)
        max_limit = max(max_limit * 1.1, min_limit * 1.2)
        if max_limit <= min_limit:
            max_limit = min_limit * 1.5
    else:
        span = max_limit - min_limit
        margin = span * 0.05 if span > 0 else max(1.0, max_limit * 0.1)
        min_limit = max(0.0, min_limit - margin)
        max_limit = max_limit + margin
        if max_limit <= min_limit:
            max_limit = min_limit + max(1.0, margin)

    scatter = ax_joint.scatter(
        x_vals,
        y_vals,
        s=180,
        c=sizes,
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors="#1a1a1a",
        linewidths=0.4,
    )

    ax_joint.plot(
        [min_limit, max_limit],
        [min_limit, max_limit],
        linestyle="--",
        color="#4d4d4d",
        linewidth=1.0,
    )

    threshold = 0.7
    if min_limit <= threshold <= max_limit:
        ax_joint.axhline(threshold, color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)
        ax_joint.axvline(threshold, color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)

    if use_log_axes:
        ax_joint.set_xscale("log")
        ax_joint.set_yscale("log")
        ax_top.set_xscale("log")
        ax_right.set_yscale("log")
    ax_joint.set_xlim(min_limit, max_limit)
    ax_joint.set_ylim(min_limit, max_limit)

    _plot_kde_marginal(ax_top, x_vals, (min_limit, max_limit), "top", use_log_axes)
    _plot_kde_marginal(ax_right, y_vals, (min_limit, max_limit), "right", use_log_axes)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_joint, ax_top, ax_right], pad=0.025, fraction=0.04)
    cbar.set_label("Group size (samples)")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, _: f"{int(val):,}" if val >= 1 else f"{val:.1f}")
    )

    ax_joint.set_xlabel(f"{axis_label} (GFP features)")
    ax_joint.set_ylabel(f"{axis_label} (GFP + perturbation features)")
    ax_joint.set_title("")
    if use_log_axes:
        ax_joint.grid(True, which="both", linestyle="--", alpha=0.4)
    else:
        ax_joint.grid(True, linestyle="--", alpha=0.4)
    ax_joint.set_aspect("equal", adjustable="box")

    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    LOGGER.info("Wrote relative MAE scatter plots: %s, %s", png_path.name, svg_path.name)
    _write_alias(png_path, svg_path, alias_basename)


def plot_log_r2_scatter(
    summary: pd.DataFrame,
    output_dir: Path,
    basename: str,
    variant: str,
    color_mode: str = "size",
    confidence_min: Optional[float] = None,
    alias_basename: Optional[str] = None,
) -> None:
    df = summary.dropna(subset=["test_log_r2"]).copy()
    if df.empty:
        LOGGER.warning("Summary contains no log-space R² values; skipping scatter plot")
        return

    if confidence_min is not None:
        if "confidence_first" in df.columns:
            conf_numeric = pd.to_numeric(df["confidence_first"], errors="coerce")
        elif "confidence" in df.columns:
            conf_numeric = pd.to_numeric(df["confidence"], errors="coerce")
        else:
            conf_numeric = None
        if conf_numeric is not None:
            df = df[conf_numeric >= confidence_min]
            if df.empty:
                LOGGER.warning(
                    "No groups meet the confidence threshold %.2f; skipping plot %s",
                    confidence_min,
                    basename,
                )
                return
        else:
            LOGGER.warning(
                "Confidence threshold provided but no confidence column found; skipping filter"
            )

    df["pretty_group"] = df["group_name"].apply(_prettify_label)

    pivot = df.pivot_table(
        index="pretty_group", columns="feature_mode", values="test_log_r2"
    )
    required_modes = ["gfp", "all"]
    missing_modes = [mode for mode in required_modes if mode not in pivot.columns]
    if missing_modes:
        LOGGER.warning(
            "Cannot plot log-space R² scatter; missing modes: %s",
            ", ".join(missing_modes),
        )
        return

    scatter_df = pivot.dropna(subset=required_modes)
    if scatter_df.empty:
        LOGGER.warning(
            "Log-space R² scatter has no overlapping groups across modes; skipping"
        )
        return

    original_min = float(np.min(scatter_df.values))
    original_max = float(np.max(scatter_df.values))
    scatter_df = scatter_df.clip(lower=0.0, upper=1.0)
    if original_min < 0.0 or original_max > 1.0:
        LOGGER.warning(
            "Clipped log-space R² values outside [0, 1] range (min=%.3f, max=%.3f)",
            original_min,
            original_max,
        )

    grouped_meta = _group_metadata(df)
    zero_series = pd.Series(0.0, index=grouped_meta.index)
    train_series = grouped_meta.get("train_samples", zero_series).fillna(0.0)
    test_series = grouped_meta.get("test_samples", zero_series).fillna(0.0)
    scatter_samples = (train_series + test_series).reindex(scatter_df.index)

    if scatter_samples.isna().any():
        scatter_samples = scatter_samples.fillna(scatter_samples.dropna().median())

    sizes = scatter_samples.values.astype(float)
    if sizes.size == 0:
        LOGGER.warning("No sample size information available; skipping scatter plot")
        return

    min_size = float(np.min(sizes))
    max_size = float(np.max(sizes))
    if max_size == min_size:
        size_norm = plt.Normalize(vmin=min_size - 1.0, vmax=max_size + 1.0)
    else:
        size_norm = plt.Normalize(vmin=min_size, vmax=max_size)

    confidence = grouped_meta.get("confidence")
    if confidence is not None:
        confidence = confidence.reindex(scatter_df.index)
    direction = grouped_meta.get("direction")
    if direction is not None:
        direction = direction.reindex(scatter_df.index)
    color_values = None
    cmap = None
    norm = None
    discrete_legend_handles: List[Patch] = []
    confidence_ticks: Optional[List[float]] = None

    if color_mode == "size":
        color_values = sizes
        cmap = plt.get_cmap("magma_r")
        norm = size_norm
        colorbar_label = "Group size (samples)"
    elif color_mode == "confidence":
        if confidence is None:
            LOGGER.warning("Confidence metadata missing; falling back to size colouring")
            color_values = sizes
            cmap = plt.get_cmap("magma_r")
            norm = size_norm
            colorbar_label = "Group size (samples)"
        else:
            conf_values = pd.to_numeric(confidence, errors="coerce")
            finite_conf = conf_values[np.isfinite(conf_values)]
            if finite_conf.empty:
                LOGGER.warning("Confidence values are all NaN; defaulting to size colouring")
                color_values = sizes
                cmap = plt.get_cmap("magma_r")
                norm = size_norm
                colorbar_label = "Group size (samples)"
            else:
                min_conf, max_conf = float(finite_conf.min()), float(finite_conf.max())
                unique_conf = sorted(finite_conf.unique())
                if len(unique_conf) <= 1:
                    norm = plt.Normalize(vmin=min_conf - 0.5, vmax=max_conf + 0.5)
                    cmap = plt.get_cmap("viridis")
                else:
                    cmap = plt.get_cmap("viridis", len(unique_conf))
                    norm = plt.Normalize(vmin=min_conf - 0.5, vmax=max_conf + 0.5)
                color_values = (
                    conf_values.reindex(scatter_df.index).astype(float).to_numpy()
                )
                colorbar_label = "Confidence"
                confidence_ticks = unique_conf
    elif color_mode == "direction":
        if direction is None:
            LOGGER.warning("Direction metadata missing; defaulting to size colouring")
            color_values = sizes
            cmap = plt.get_cmap("magma_r")
            norm = size_norm
            colorbar_label = "Group size (samples)"
        else:
            dir_values = pd.to_numeric(direction, errors="coerce")
            palette = { -1: "#d73027", 0: "#8c8c8c", 1: "#4575b4" }
            labels = { -1: "Brake (-1)", 0: "Neutral (0)", 1: "Driver (+1)" }
            mapped_colors = []
            for val in dir_values:
                if np.isnan(val):
                    mapped_colors.append("#bdbdbd")
                else:
                    mapped_colors.append(palette.get(int(val), "#bdbdbd"))
            color_values = mapped_colors
            cmap = None
            norm = None
            present_dirs = set(dir_values.dropna().astype(int).tolist())
            for key, label in labels.items():
                mapped = palette.get(key)
                if mapped is not None and key in present_dirs:
                    discrete_legend_handles.append(Patch(color=mapped, label=label))
            if dir_values.isna().any():
                discrete_legend_handles.append(Patch(color="#bdbdbd", label="Unspecified"))
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlepad": 12,
    })

    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    x_vals = scatter_df["gfp"].values
    y_vals = scatter_df["all"].values

    scatter_kwargs = {
        "x": x_vals,
        "y": y_vals,
        "s": 180,
        "alpha": 0.7,
        "edgecolors": "#1a1a1a",
        "linewidths": 0.4,
    }
    if cmap is not None and norm is not None:
        scatter_kwargs.update({"c": color_values, "cmap": cmap, "norm": norm})
    else:
        scatter_kwargs.update({"c": color_values})
    ax_joint.scatter(**scatter_kwargs)

    ax_joint.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#4d4d4d", linewidth=1.0)

    threshold = 0.7
    ax_joint.axhline(threshold, color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)
    ax_joint.axvline(threshold, color="#c0392b", linestyle="-", linewidth=1.2, alpha=0.8)

    _plot_kde_marginal(ax_top, x_vals, (0.0, 1.0), "top", False)
    _plot_kde_marginal(ax_right, y_vals, (0.0, 1.0), "right", False)

    if cmap is not None and norm is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_joint, ax_top, ax_right], pad=0.025, fraction=0.04)
        cbar.set_label(colorbar_label)
        if color_mode == "confidence" and confidence_ticks:
            cbar.set_ticks(confidence_ticks)
            cbar.set_ticklabels(
                [str(int(t)) if float(t).is_integer() else f"{t:g}" for t in confidence_ticks]
            )
        if color_mode == "size":
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, prune="both"))
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda val, _: f"{int(val):,}" if val >= 1 else f"{val:.1f}"
                )
            )
    else:
        if discrete_legend_handles:
            ax_joint.legend(handles=discrete_legend_handles, loc="lower right", title="Direction")

    ax_joint.set_xlabel("Log-space R² (GFP features)")
    ax_joint.set_ylabel("Log-space R² (GFP + perturbation features)")
    ax_joint.set_title("")
    ax_joint.set_xlim(0.0, 1.0)
    ax_joint.set_ylim(0.0, 1.0)
    ax_joint.grid(True, linestyle="--", alpha=0.4)
    ax_joint.set_aspect("equal", adjustable="box")

    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    LOGGER.info("Wrote log-space R² scatter plots: %s, %s", png_path.name, svg_path.name)
    _write_alias(png_path, svg_path, alias_basename)
    _write_alias(png_path, svg_path, alias_basename)


def _prepare_model_comparison_data(
    summary: pd.DataFrame,
    feature_mode: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "feature_mode" not in summary.columns:
        LOGGER.warning("Cannot plot model comparison scatter; summary missing 'feature_mode'")
        return np.array([]), np.array([])
    if "model" not in summary.columns:
        LOGGER.warning("Cannot plot model comparison scatter; summary missing 'model'")
        return np.array([]), np.array([])
    if metric not in summary.columns:
        LOGGER.warning("Cannot plot model comparison scatter; summary missing '%s'", metric)
        return np.array([]), np.array([])

    subset = summary.copy()
    subset["feature_mode"] = subset["feature_mode"].astype(str)
    target_mode = feature_mode.lower()
    subset = subset[subset["feature_mode"].str.lower() == target_mode]
    if subset.empty:
        LOGGER.warning(
            "No rows available for model comparison in feature mode '%s'; skipping",
            feature_mode,
        )
        return np.array([]), np.array([])

    pivot = subset.pivot_table(
        index="group_name",
        columns="model",
        values=metric,
        aggfunc=_first_numeric,
    )
    required_models = ["PySR", "Linear Regression"]
    missing = [m for m in required_models if m not in pivot.columns]
    if missing:
        LOGGER.warning(
            "Skipping model comparison scatter for feature mode '%s'; missing models: %s",
            feature_mode,
            ", ".join(missing),
        )
        return np.array([]), np.array([])

    pivot = pivot.dropna(subset=required_models)
    if pivot.empty:
        LOGGER.warning(
            "No overlapping groups with finite values for comparison in feature mode '%s'",
            feature_mode,
        )
        return np.array([]), np.array([])

    x_vals = pivot["Linear Regression"].to_numpy(dtype=float)
    x_vals = np.clip(x_vals, 0.0, None)
    y_vals = pivot["PySR"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[finite_mask]
    y_vals = y_vals[finite_mask]
    if x_vals.size == 0:
        LOGGER.warning(
            "All comparison values are non-finite for feature mode '%s'",
            feature_mode,
        )
    return x_vals, y_vals


def plot_model_comparison_scatter(
    summary: pd.DataFrame,
    output_dir: Path,
    basename: str,
    feature_mode: str,
    metric: str = "test_log_r2",
) -> None:
    x_vals, y_vals = _prepare_model_comparison_data(summary, feature_mode, metric)
    if x_vals.size == 0:
        return

    overall_min = min(float(np.min(x_vals)), float(np.min(y_vals)))
    overall_max = max(float(np.max(x_vals)), float(np.max(y_vals)))
    padding = max((overall_max - overall_min) * 0.05, 1e-3)
    axis_min = overall_min - padding
    axis_max = overall_max + padding

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(x_vals, y_vals, s=52, color="#2c7fb8", alpha=0.85, edgecolor="none")
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#444444", linestyle="--", linewidth=1.2)

    try:
        rho = float(np.corrcoef(x_vals, y_vals)[0, 1])
    except Exception:
        rho = np.nan

    mode_label = feature_mode.upper()
    ax.set_title(
        f"{mode_label} mode • log R\u00b2 (PySR vs Linear Regression)\n\u03c1 = {rho:.2f}",
        fontsize=13,
    )
    ax.set_xlabel("Log R\u00b2 (Linear Regression)", fontsize=12)
    ax.set_ylabel("Log R\u00b2 (PySR)", fontsize=12)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.annotate(
        f"n = {x_vals.size}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=11,
        color="#333333",
    )

    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Wrote model comparison scatter (%s mode): %s, %s",
        feature_mode,
        png_path.name,
        svg_path.name,
    )


def plot_model_comparison_scatter_high(
    summary: pd.DataFrame,
    output_dir: Path,
    basename: str,
    feature_mode: str,
    metric: str = "test_log_r2",
    threshold: float = 0.7,
) -> None:
    x_vals, y_vals = _prepare_model_comparison_data(summary, feature_mode, metric)
    if x_vals.size == 0:
        return

    mask = (x_vals > threshold) | (y_vals > threshold)
    x_subset = x_vals[mask]
    y_subset = y_vals[mask]
    if x_subset.size == 0:
        LOGGER.info(
            "No groups exceed threshold %.2f in feature mode '%s'; skipping high-performance plot",
            threshold,
            feature_mode,
        )
        return

    overall_min = min(float(np.min(x_subset)), float(np.min(y_subset)))
    overall_max = max(float(np.max(x_subset)), float(np.max(y_subset)))
    padding = max((overall_max - overall_min) * 0.05, 1e-3)
    axis_min = overall_min - padding
    axis_max = overall_max + padding

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(x_subset, y_subset, s=52, color="#2c7fb8", alpha=0.85, edgecolor="none")
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="#444444", linestyle="--", linewidth=1.2)

    trend_handle = None
    try:
        slope, intercept = np.polyfit(x_subset, y_subset, 1)
        reg_x = np.array([axis_min, axis_max])
        reg_y = slope * reg_x + intercept
        (trend_handle,) = ax.plot(
            reg_x,
            reg_y,
            color="#c0392b",
            linewidth=1.4,
            label=f"Trend (m={slope:.2f}, b={intercept:.2f})",
        )
    except Exception as exc:
        LOGGER.warning("Failed to compute regression trend line for high subset: %s", exc)

    try:
        rho = float(np.corrcoef(x_subset, y_subset)[0, 1])
    except Exception:
        rho = np.nan

    mode_label = feature_mode.upper()
    ax.set_title(
        f"{mode_label} mode • log R\u00b2 (PySR vs Linear Regression)\n"
        f"\u03c1 = {rho:.2f} | threshold > {threshold:.2f}",
        fontsize=13,
    )
    ax.set_xlabel("Log R\u00b2 (Linear Regression)", fontsize=12)
    ax.set_ylabel("Log R\u00b2 (PySR)", fontsize=12)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.annotate(
        f"n = {x_subset.size}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=11,
        color="#333333",
    )
    if trend_handle is not None:
        ax.legend(loc="lower right", fontsize=10)

    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Wrote high-threshold model comparison scatter (%s mode): %s, %s",
        feature_mode,
        png_path.name,
        svg_path.name,
    )


def _direction_label(val: float | int | None) -> Optional[str]:
    if val is None or pd.isna(val):
        return None
    ival = int(np.sign(val))  # clamp to -1, 0, 1 just in case
    return { -1: "Brake (-1)", 0: "Indifferent (0)", 1: "Driver (+1)" }.get(ival)

def _build_direction_sets_for_mode(
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_mode: str,
) -> Dict[str, set]:
    """
    Returns a dict mapping direction labels to sets of member names that appear
    in groups of that direction, restricted to rows present for the given feature_mode.
    """
    # Keep only rows that actually appear for this mode in the summary
    if "feature_mode" in summary.columns:
        mode_summary = summary[summary["feature_mode"] == feature_mode].copy()
    else:
        mode_summary = summary.copy()

    # Merge in 'members_list' and 'direction' columns from metadata if not already present
    if {"members_list", "direction"}.issubset(mode_summary.columns):
        merged = mode_summary.copy()
    else:
        md = metadata.copy()
        if "members_list" not in md.columns:
            # Ensure members_list is present (should be after _load_group_metadata)
            md = _load_group_metadata(metadata)  # type: ignore[arg-type]

        merged = mode_summary.merge(
            md[["group", "members_list", "direction"]],
            left_on="group_name", right_on="group", how="left"
        )

    # Build sets of markers per direction
    dir_sets: Dict[str, set] = {"Driver (+1)": set(), "Brake (-1)": set(), "Indifferent (0)": set()}
    for _, row in merged.iterrows():
        dlabel = _direction_label(row.get("direction"))
        members = row.get("members_list")
        if dlabel is None or not isinstance(members, (list, tuple)):
            continue
        for m in members:
            if isinstance(m, str) and m.strip():
                dir_sets[dlabel].add(m.strip())

    # Drop empty sets so UpSet doesn’t draw empty columns
    return {k: v for k, v in dir_sets.items() if v}

def plot_upset_groups_with_r2(
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    basename: str,
    feature_mode: str,  # "gfp" or "all"
) -> None:
    """UpSet-style plot with genes (rows) split into Driver→Brake→Indifferent and groups (cols) sorted by log-R²."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Parse members into lists ----------
    md = metadata.copy()
    if "members_list" not in md.columns:
        if "members" not in md.columns:
            LOGGER.warning("Metadata missing 'members_list'/'members'; skipping %s", basename)
            return
        def _parse_members(raw: object) -> list[str]:
            if pd.isna(raw):
                return []
            txt = str(raw)
            try:
                out = ast.literal_eval(txt)
                return [str(m).strip() for m in out]
            except Exception:
                return [m.strip().strip("'\"") for m in txt.strip("[]").split(",") if m.strip()]
        md["members_list"] = md["members"].apply(_parse_members)
    if "group" not in md.columns:
        LOGGER.warning("Metadata missing 'group'; skipping %s", basename)
        return

    # ---------- Build membership matrix genes×groups ----------
    mem_long = (
        md.explode("members_list")
          .dropna(subset=["members_list"])
          .rename(columns={"members_list": "gene"})
    )
    if mem_long.empty:
        LOGGER.warning("No gene memberships; skipping %s", basename)
        return

    membership_df = (
        mem_long.assign(val=1)
                .pivot_table(index="gene", columns="group",
                             values="val", aggfunc="max", fill_value=0)
                .astype(bool)
                .sort_index()
    )
    if membership_df.empty:
        LOGGER.warning("Membership matrix empty; skipping %s", basename)
        return

    # ---------- Directions per gene ----------
    if "direction" in mem_long.columns:
        directions = (
            pd.to_numeric(mem_long["direction"], errors="coerce")
              .groupby(mem_long["gene"]).mean()
              .reindex(membership_df.index)
              .fillna(0.0)
        )
    else:
        directions = pd.Series(0.0, index=membership_df.index)

    def _dir_label(v: float) -> str:
        try:
            s = int(np.sign(float(v)))
        except Exception:
            s = 0
        return {+1: "Driver(+1)", -1: "Brake(-1)", 0: "Indifferent(0)"}[s]
    dir_labels = directions.map(_dir_label)

    # ---------- Per-group R² for selected feature_mode ----------
    if "feature_mode" in summary.columns:
        sub = summary[summary["feature_mode"] == feature_mode]
    else:
        sub = summary
    if "test_log_r2" in sub.columns:
        r2_per_group = (
            sub.dropna(subset=["test_log_r2"])
               .groupby("group_name")["test_log_r2"]
               .max()
               .clip(0.0, 1.0)
        )
    else:
        r2_per_group = None

    # Keep only groups present in membership; sort by R² desc (or by size if R² missing)
    if r2_per_group is not None and not r2_per_group.empty:
        common = [g for g in r2_per_group.index if g in membership_df.columns]
        if not common:
            LOGGER.warning("No overlapping groups between membership and R²; skipping %s", basename)
            return
        membership_df = membership_df[common]
        r2_per_group = r2_per_group.reindex(common)
        order_cols = r2_per_group.sort_values(ascending=False).index.tolist()
        membership_df = membership_df[order_cols]
        r2_per_group = r2_per_group.reindex(order_cols)
        bar_vals = r2_per_group.to_numpy()
        bar_label = "R² (log space)"
    else:
        order_cols = membership_df.sum(0).sort_values(ascending=False).index.tolist()
        membership_df = membership_df[order_cols]
        bar_vals = membership_df.sum(0).to_numpy()
        bar_label = "Genes per group"

    n_groups = membership_df.shape[1]

    # ---------- Block order & alphabetical within blocks ----------
    blocks: list[tuple[str, list[str]]] = []
    for label in ("Driver(+1)", "Brake(-1)", "Indifferent(0)"):  # TOP → BOTTOM
        genes_block = dir_labels[dir_labels == label].index
        if len(genes_block) == 0:
            continue
        sub_idx = sorted(genes_block, key=lambda s: s.lower())
        blocks.append((label, sub_idx))
    ordered_genes = [g for _, glist in blocks for g in glist]
    if not ordered_genes:
        LOGGER.warning("No genes to plot after block split; skipping %s", basename)
        return
    membership_df = membership_df.loc[ordered_genes]

    # ---------- Styling ----------
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 9,
    })

    # Wider figure so bar labels don’t collide; roomier rows
    fig_w = max(12, min(24, n_groups * 0.45))
    fig_h = max(8, min(30, len(ordered_genes) * 0.32))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 6.8], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_grid = fig.add_subplot(gs[1, 0], sharex=ax_bar)

    # ---------- TOP BARS (tall; no top/right spines) ----------
    x = np.arange(n_groups)
    ax_bar.bar(x, bar_vals, color="#333333", edgecolor="#333333", linewidth=0.5)
    ax_bar.set_ylabel(bar_label, labelpad=10)
    ax_bar.set_xticks([])
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    y_max = float(bar_vals.max() if len(bar_vals) else 1.0)
    headroom = 0.05 * (y_max if y_max > 0 else 1.0)
    for i, v in enumerate(bar_vals):
        ax_bar.text(i, v + headroom, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#333333", clip_on=False)

    # ---------- GRID (UpSet-style) ----------
    vals = membership_df.values
    genes_per_block = [len(glist) for _, glist in blocks]
    gap = 3.0  # whitespace between blocks

    # y-positions with gaps (NO invert_yaxis; first block stays on top)
    offsets = np.cumsum([0] + [n + gap for n in genes_per_block[:-1]])
    y_positions = np.concatenate([
        np.arange(len(glist)) + off for off, (_, glist) in zip(offsets, blocks)
    ])

    # Zebra banding
    for k, y in enumerate(y_positions):
        ax_grid.add_patch(plt.Rectangle(
            (-0.5, y - 0.5), n_groups, 1.0,
            facecolor=("#f4f4f4" if k % 2 else "#ffffff"),
            edgecolor="none", zorder=0))

    # Empty dots first (light grey), then filled (black)
    xx = np.repeat(np.arange(n_groups), len(y_positions))
    yy = np.tile(y_positions, n_groups)
    ax_grid.scatter(xx, yy, s=52, facecolors="#e0e0e0", edgecolors="none", zorder=1)

    rows_true, cols_true = np.where(vals)
    ax_grid.scatter(cols_true, y_positions[rows_true], s=62,
                    facecolors="#111111", edgecolors="none", zorder=3)

    # Vertical connectors per column, but NOT across blocks
    block_bounds = np.cumsum([0] + genes_per_block)  # boundaries in row indices (pre-gap)
    for j in range(vals.shape[1]):
        rows = np.where(vals[:, j])[0]
        if rows.size < 2:
            continue
        for b in range(len(block_bounds) - 1):
            start, stop = block_bounds[b], block_bounds[b + 1]
            seg = rows[(rows >= start) & (rows < stop)]
            if seg.size >= 2:
                y0 = y_positions[seg.min()] + 0.12
                y1 = y_positions[seg.max()] - 0.12
                if y1 > y0:
                    ax_grid.vlines(j, y0, y1, colors="#333333", linewidth=1.3, zorder=2)

    # Axes cosmetics
    ax_grid.set_xlim(-0.5, n_groups - 0.5)
    ax_grid.set_ylim(y_positions.max() + 0.5, -0.5)  # TOP at smaller y, BOTTOM at larger y
    ax_grid.set_xticks([])                            # hide group names
    ax_grid.set_yticks(y_positions)
    ax_grid.set_yticklabels(membership_df.index, fontsize=9)
    for sp in ax_grid.spines.values():
        sp.set_visible(False)
    ax_grid.grid(False)

    # Vertical block labels to the LEFT of gene names (no overlap)
    # Offset far enough left; adjust -4.5 if needed.
    left_x = -4.5
    for (label, glist), off in zip(blocks, offsets):
        start, stop = off, off + len(glist)
        mid_y = (start + stop - 1) / 2.0
        ax_grid.text(left_x, mid_y, label, rotation=90,
                     va="center", ha="center", fontsize=11, color="#333333", clip_on=False)

    # space for ylabels & block labels
    fig.subplots_adjust(left=0.22, right=0.98, top=0.93)
    fig.suptitle(f"Genes × Groups (UpSet-style) • mode: {feature_mode}", y=0.98)

    # ---------- Save ----------
    png = output_dir / f"{basename}.png"
    svg = output_dir / f"{basename}.svg"
    png.parent.mkdir(parents=True, exist_ok=True)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote UpSet-style plot: %s, %s", png.name, svg.name)

def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading summary from %s", args.summary)
    summary_all = pd.read_csv(args.summary)
    if "model" in summary_all.columns:
        summary_all["model"] = summary_all["model"].astype(str)
        available_models = sorted(summary_all["model"].unique())
        model_filter = args.model.strip().lower()
        summary = summary_all[summary_all["model"].str.lower() == model_filter].copy()
        if summary.empty:
            raise ValueError(
                f"Summary contains no rows for model '{args.model}'. Available models: {available_models}"
            )
    else:
        summary = summary_all.copy()
        available_models = []

    metadata = _load_group_metadata(args.group_definitions_csv)
    filtered = _filter_summary(summary, metadata, args.variant)
    if filtered.empty:
        LOGGER.warning("No rows available for variant '%s'; skipping plot", args.variant)
        return
    LOGGER.info("Generating %s plot in %s", args.variant, output_dir)
    plot_log_r2(filtered, output_dir, args.basename, args.variant, alias_basename=args.alias_basename)
    plot_relative_mae_scatter(filtered, output_dir, args.relative_basename, args.variant, alias_basename=args.relative_alias_basename)
    plot_log_r2_scatter(filtered, output_dir, args.r2_basename, args.variant, color_mode="size", alias_basename=args.r2_alias_basename)
    plot_log_r2_scatter(filtered, output_dir, args.r2_confidence_basename, args.variant, color_mode="confidence", alias_basename=args.r2_confidence_alias_basename)
    plot_log_r2_scatter(filtered, output_dir, args.r2_direction_basename, args.variant, color_mode="direction", alias_basename=args.r2_direction_alias_basename)
    plot_log_r2_scatter(
        filtered,
        output_dir,
        args.r2_confidence_high_basename,
        args.variant,
        color_mode="confidence",
        confidence_min=args.confidence_threshold,
        alias_basename=args.r2_confidence_high_alias_basename,
    )
    plot_upset_groups_with_r2(filtered, metadata, output_dir, basename=args.upset_gfp_basename, feature_mode="gfp")
    plot_upset_groups_with_r2(filtered, metadata, output_dir, basename=args.upset_all_basename, feature_mode="all")

    dataset_for_fits: Optional[pd.DataFrame] = None
    if args.fit_groups:
        if args.dataset is None:
            LOGGER.warning(
                "No dataset provided via --dataset; skipping PySR trajectory fit plots for requested groups."
            )
        else:
            try:
                dataset_for_fits = _load_sr_dataset(args.dataset)
            except FileNotFoundError as exc:
                LOGGER.warning("%s; skipping trajectory fit plots", exc)
            except Exception as exc:
                LOGGER.warning("Failed to load dataset '%s': %s", args.dataset, exc)

    if dataset_for_fits is not None and args.fit_groups:
        try:
            grid_cols = int(args.fit_grid_columns)
        except Exception:
            grid_cols = 5
        grid_cols = grid_cols if grid_cols > 0 else 5
        fit_output_dir = output_dir / args.fit_output_subdir
        plot_pysr_fit_grids(
            dataset_for_fits,
            filtered,
            metadata,
            args.fit_groups,
            fit_output_dir,
            log10_cutoff=args.log10_cutoff,
            grid_columns=grid_cols,
        )

    if "model" in summary_all.columns:
        comparison_filtered = _filter_summary(summary_all, metadata, args.variant)
        model_set = set(
            comparison_filtered.get("model", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
        )
        required_models = {"PySR", "Linear Regression"}
        if required_models.issubset(model_set):
            plot_model_comparison_scatter(
                comparison_filtered,
                output_dir,
                args.comparison_gfp_basename,
                feature_mode="gfp",
            )
            plot_model_comparison_scatter(
                comparison_filtered,
                output_dir,
                args.comparison_all_basename,
                feature_mode="all",
            )
            plot_model_comparison_scatter_high(
                comparison_filtered,
                output_dir,
                args.comparison_gfp_high_basename,
                feature_mode="gfp",
                threshold=0.7,
            )
            plot_model_comparison_scatter_high(
                comparison_filtered,
                output_dir,
                args.comparison_all_high_basename,
                feature_mode="all",
                threshold=0.7,
            )
        else:
            LOGGER.info(
                "Skipping model comparison scatter for variant '%s'; available models: %s",
                args.variant,
                ", ".join(sorted(model_set)) if model_set else "none",
            )

if __name__ == "__main__":
    main()
