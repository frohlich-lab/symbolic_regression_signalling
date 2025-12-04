"""Generate marker-level overlay plots from SR summaries."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.sr_pipeline.run_functional_groups import EXCLUDE_COLUMNS, sanitize_feature_names
from experimental.sr_pipeline.compute_marker_integration import evaluate_formula

LOGGER = logging.getLogger("experimental.marker_overlay")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create overlay/integration plots per marker from SR summary.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV used for SR (snapshot or per-minute).")
    parser.add_argument("--summary", type=Path, required=True, help="functional_group_summary.csv from SR run.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write plots/metrics.")
    parser.add_argument(
        "--predicted-trajectories",
        type=Path,
        default=None,
        help="Optional CSV of predicted trajectories from training (predicted_trajectories_{mode}.csv).",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("snapshot", "per_minute"),
        default="snapshot",
        help="Controls metric filtering (per-minute restricts to measured timepoints).",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=(0.0, 5.0, 10.0, 15.0, 30.0, 60.0),
        help="Measured timepoints; per-minute metrics restricted to these.",
    )
    parser.add_argument("--model", choices=("PySR", "Linear Regression"), default="PySR", help="Model to plot.")
    parser.add_argument(
        "--filter-dataset-mode",
        type=str,
        default=None,
        help="If summary contains a dataset_mode column, filter rows to this mode (e.g., snapshot or per_minute).",
    )
    parser.add_argument("--log10-cutoff", type=float, default=-3.0, help="Signed-log cutoff for R2 on dt.")
    parser.add_argument(
        "--fig-base", type=str, default="marker_overlay", help="Base filename (png/pdf) for overlay figure."
    )
    parser.add_argument("--metrics-csv", type=str, default="marker_overlay_metrics.csv", help="Metrics CSV filename.")
    parser.add_argument(
        "--metrics-output-dir",
        type=Path,
        default=None,
        help="Optional directory for metrics CSV (defaults to output-dir).",
    )
    parser.add_argument(
        "--integration-trajectories",
        type=Path,
        default=None,
        help="Optional precomputed integration trajectories CSV (from compute_marker_integration.py).",
    )
    parser.add_argument(
        "--integration-metrics",
        type=Path,
        default=None,
        help="Optional precomputed integration metrics CSV (from compute_marker_integration.py).",
    )
    parser.add_argument(
        "--markers-per-fig",
        type=int,
        default=0,
        help="Max markers per figure; 0 means all markers in one figure (default).",
    )
    parser.add_argument(
        "--per-seed-overlays",
        action="store_true",
        help="If summary/predictions include a seed column, generate overlays per seed into seed_<seed>/ subdirs.",
    )
    return parser.parse_args()


def _signed_log(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    eps = 10.0 ** log10_cutoff
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * (np.log10(np.abs(arr) + eps) - log10_cutoff)


def evaluate_formula(formula: str, features: pd.DataFrame) -> np.ndarray:
    expr = sp.sympify(formula)
    symbols = sorted({str(s) for s in expr.free_symbols})
    missing = [s for s in symbols if s not in features.columns]
    if missing:
        # Allow formulas that reference features not present in the current dataset by
        # adding zero-filled placeholder columns. This mirrors the pipeline behavior
        # where some operators may have been pruned per marker.
        for m in missing:
            features[m] = 0.0
    if not symbols:
        return np.full(len(features), float(expr))
    func = sp.lambdify(symbols, expr, "numpy")
    arrays = [features[s].to_numpy(dtype=float) for s in symbols]
    vals = func(*arrays)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def make_overlay_plots(
    dataset: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    # Keep only selected model rows
    summary = summary[summary["model"] == args.model].copy()
    if args.filter_dataset_mode and "dataset_mode" in summary.columns:
        summary = summary[summary["dataset_mode"] == args.filter_dataset_mode].copy()
    if summary.empty:
        raise ValueError(f"No rows for model {args.model} in summary.")

    markers = sorted(summary["group_name"].unique().tolist())
    if not markers:
        raise ValueError("No markers/groups available for overlay plotting.")

    # Prepare dataset if we need to evaluate formulas
    dataset = dataset.copy()
    dataset.columns = dataset.columns.str.replace("_fit$", "", regex=True)
    if "marker" in dataset.columns:
        dataset["marker"] = dataset["marker"].astype(str)
    if "GFP_bin" in dataset.columns:
        dataset["GFP_bin"] = pd.to_numeric(dataset["GFP_bin"], errors="coerce")
    if "timepoint" in dataset.columns:
        dataset["timepoint"] = pd.to_numeric(dataset["timepoint"], errors="coerce")

    candidate_cols = [c for c in dataset.columns if c not in EXCLUDE_COLUMNS]
    sanitized = sanitize_feature_names(candidate_cols)
    col_map = dict(zip(candidate_cols, sanitized))
    dataset.rename(columns=col_map, inplace=True)
    target_col = sanitize_feature_names(["p-ERK1-2_dt"])[0]
    dataset[target_col] = pd.to_numeric(dataset["p-ERK1-2_dt"], errors="coerce")

    # Optional predicted trajectories (avoids re-evaluating formulas)
    pred_df_all = None
    if args.predicted_trajectories:
        pred_df_all = pd.read_csv(args.predicted_trajectories)
        if "model" in pred_df_all.columns:
            pred_df_all = pred_df_all[pred_df_all["model"] == args.model]
        if "dataset_mode" in pred_df_all.columns:
            pred_df_all = pred_df_all[pred_df_all["dataset_mode"] == args.dataset_mode]
        if pred_df_all.empty:
            pred_df_all = None

    # Optional precomputed integration data
    integration_df_all = None
    if args.integration_trajectories:
        integration_df_all = pd.read_csv(args.integration_trajectories)
        integration_df_all = integration_df_all[integration_df_all["model"] == args.model]
        if "dataset_mode" in integration_df_all.columns:
            integration_df_all = integration_df_all[integration_df_all["dataset_mode"] == args.dataset_mode]
        if integration_df_all.empty:
            integration_df_all = None
    integration_metrics_df_all = None
    if args.integration_metrics:
        integration_metrics_df_all = pd.read_csv(args.integration_metrics)
        integration_metrics_df_all = integration_metrics_df_all[integration_metrics_df_all["model"] == args.model]
        if "dataset_mode" in integration_metrics_df_all.columns:
            integration_metrics_df_all = integration_metrics_df_all[integration_metrics_df_all["dataset_mode"] == args.dataset_mode]
        if integration_metrics_df_all.empty:
            integration_metrics_df_all = None

    seed_col = None
    for cand in ("seed", "random_state"):
        if cand in summary.columns:
            seed_col = cand
            break

    def _run_for_seed(seed_value: Optional[object]) -> None:
        if seed_value is not None and seed_col:
            summary_seed = summary[summary[seed_col] == seed_value].copy()
        else:
            summary_seed = summary.copy()
        if summary_seed.empty:
            LOGGER.warning("No summary rows for seed=%s; skipping overlay.", seed_value)
            return
        # Select a consistent seed from predictions/integration to avoid mixing seeds.
        def _pick_seed(df_all: Optional[pd.DataFrame]) -> Optional[object]:
            if df_all is None:
                return None
            for col in ("seed", "random_state"):
                if col in df_all.columns:
                    seeds = [s for s in df_all[col].dropna().unique().tolist()]
                    return sorted(seeds, key=str)[0] if seeds else None
            return None

        chosen_seed = seed_value
        if chosen_seed is None:
            chosen_seed = _pick_seed(pred_df_all) or _pick_seed(integration_df_all) or _pick_seed(integration_metrics_df_all)

        pred_df = pred_df_all
        if pred_df_all is not None:
            for col in ("seed", "random_state"):
                if col in pred_df_all.columns and chosen_seed is not None:
                    pred_df = pred_df_all[pred_df_all[col] == chosen_seed].copy()
                    break
            if seed_col and seed_col in pred_df_all.columns and seed_value is not None:
                pred_df = pred_df_all[pred_df_all[seed_col] == seed_value].copy()
            if pred_df is not None and pred_df.empty:
                pred_df = None

        integration_df = integration_df_all
        if integration_df_all is not None:
            for col in ("seed", "random_state"):
                if col in integration_df_all.columns and chosen_seed is not None:
                    integration_df = integration_df_all[integration_df_all[col] == chosen_seed].copy()
                    break
            if seed_col and seed_col in integration_df_all.columns and seed_value is not None:
                integration_df = integration_df_all[integration_df_all[seed_col] == seed_value].copy()
            if integration_df is not None and integration_df.empty:
                integration_df = None

        integration_metrics_df = integration_metrics_df_all
        if integration_metrics_df_all is not None:
            for col in ("seed", "random_state"):
                if col in integration_metrics_df_all.columns and chosen_seed is not None:
                    integration_metrics_df = integration_metrics_df_all[integration_metrics_df_all[col] == chosen_seed].copy()
                    break
            if seed_col and seed_col in integration_metrics_df_all.columns and seed_value is not None:
                integration_metrics_df = integration_metrics_df_all[integration_metrics_df_all[seed_col] == seed_value].copy()
            if integration_metrics_df is not None and integration_metrics_df.empty:
                integration_metrics_df = None

        if args.per_seed_overlays and seed_value is not None:
            seed_out = base_output_dir / f"seed_{seed_value}"
        else:
            seed_out = base_output_dir
        seed_out.mkdir(parents=True, exist_ok=True)

        _make_overlay_plots_core(
            dataset,
            summary_seed,
            pred_df,
            integration_df,
            integration_metrics_df,
            args,
            seed_out,
            sanitized,
            target_col,
        )

    if args.per_seed_overlays and seed_col:
        seeds = summary[seed_col].dropna().unique().tolist()
        for sval in seeds:
            _run_for_seed(sval)
    else:
        _run_for_seed(None)


def _make_overlay_plots_core(
    dataset: pd.DataFrame,
    summary: pd.DataFrame,
    pred_df: Optional[pd.DataFrame],
    integration_df: Optional[pd.DataFrame],
    integration_metrics_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
    output_dir: Path,
    sanitized_features: List[str],
    target_col: str,
) -> None:
    if integration_metrics_df is not None and "group_name" not in integration_metrics_df.columns and "marker" in integration_metrics_df.columns:
        integration_metrics_df["group_name"] = integration_metrics_df["marker"]

    markers = sorted(summary["group_name"].unique().tolist())
    if not markers:
        LOGGER.warning("No markers/groups available for overlay plotting.")
        return

    def _plot_dt_curves(pred: pd.DataFrame) -> None:
        """Plot dt true vs predicted per marker/bin (per-minute)."""
        if pred is None or pred.empty or args.dataset_mode != "per_minute" or args.model != "PySR":
            return
        plot_dir = output_dir / "dt_curves" / "pysr" / "per_minute"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for marker in sorted(pred["marker"].unique()):
            sub = pred[pred["marker"] == marker]
            bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
            if not bins:
                continue
            ncols = 3
            nrows = int(np.ceil(len(bins) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
            for ax, b in zip(axes.flat, bins):
                g = sub[sub["GFP_bin"] == b].copy()
                g.sort_values("timepoint", inplace=True)
                ax.plot(
                    g["timepoint"],
                    g["p-ERK1-2_dt_true"],
                    label="dt true",
                    marker="o",
                    markersize=5,
                    color="#2c3e50",
                )
                ax.plot(
                    g["timepoint"],
                    g["p-ERK1-2_dt_pred"],
                    label="dt pred",
                    marker="x",
                    markersize=5,
                    linestyle="--",
                    color="#e74c3c",
                )
                ax.set_title(f"GFP bin {b}")
                ax.set_xlabel("Time")
                ax.set_ylabel("p-ERK1/2 dt")
                ax.grid(True, linestyle="--", alpha=0.3)
            # Hide unused axes
            for ax in axes.flat[len(bins) :]:
                ax.axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")
            fig.suptitle(f"{marker} | PySR per-minute dt fit")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            for ext in ("png", "svg"):
                fig.savefig(plot_dir / f"{marker}_dt_curves.{ext}", dpi=200, bbox_inches="tight")
            plt.close(fig)

    target_col = sanitize_feature_names(["p-ERK1-2_dt"])[0]
    metrics_rows: List[Dict[str, object]] = []
    chunk_size = args.markers_per_fig if args.markers_per_fig and args.markers_per_fig > 0 else len(markers)
    chunk_size = max(1, int(chunk_size))
    ncols = 5
    num_chunks = int(np.ceil(len(markers) / chunk_size))

    def _obs_pe_array(frame: pd.DataFrame, length: int) -> np.ndarray:
        for col in ("obs_pERK1_2", "p_ERK1_2", "p-ERK1-2"):
            if col in frame.columns:
                return frame[col].to_numpy(dtype=float)
        return np.full(length, np.nan, dtype=float)

    for chunk_idx in range(0, len(markers), chunk_size):
        chunk = markers[chunk_idx : chunk_idx + chunk_size]
        fig, axes = plt.subplots(
            len(chunk),
            ncols,
            figsize=(4.2 * ncols, 3.1 * len(chunk)),
            squeeze=False,
        )

        for i, marker in enumerate(chunk):
            row = summary[summary["group_name"] == marker].iloc[0]
            formula = row["formula"]
            if formula in (None, "N/A"):
                LOGGER.warning("Skipping marker %s (missing formula).", marker)
                continue
            train_r2 = float(row["train_r2"]) if "train_r2" in row and pd.notna(row["train_r2"]) else None
            test_r2 = float(row["test_r2"]) if "test_r2" in row and pd.notna(row["test_r2"]) else None
            def _fmt_r2(val: Optional[float]) -> str:
                return f"{val:.2f}" if val is not None and np.isfinite(val) else "NA"
            integ_train_r2 = integ_test_r2 = ode_train_r2 = ode_test_r2 = None
            if integration_metrics_df is not None:
                key_col = "group_name" if "group_name" in integration_metrics_df.columns else "marker"
                mrow = integration_metrics_df[integration_metrics_df[key_col] == marker]
                if not mrow.empty:
                    mrow = mrow.iloc[0]
                    integ_train_r2 = (
                        float(mrow["integ_r2_median_train"])
                        if "integ_r2_median_train" in mrow and pd.notna(mrow["integ_r2_median_train"])
                        else None
                    )
                    integ_test_r2 = (
                        float(mrow["integ_r2_median"])
                        if "integ_r2_median" in mrow and pd.notna(mrow["integ_r2_median"])
                        else None
                    )
                    ode_train_r2 = (
                        float(mrow["ode_integ_r2_median_train"])
                        if "ode_integ_r2_median_train" in mrow and pd.notna(mrow["ode_integ_r2_median_train"])
                        else None
                    )
                    ode_test_r2 = (
                        float(mrow.get("ode_integ_r2_median") or mrow.get("integrated_ode_r2"))
                        if (
                            ("ode_integ_r2_median" in mrow and pd.notna(mrow["ode_integ_r2_median"]))
                            or ("integrated_ode_r2" in mrow and pd.notna(mrow["integrated_ode_r2"]))
                        )
                        else None
                    )

            if pred_df is not None:
                sub_pred = pred_df[pred_df["marker"] == marker].copy()
                if sub_pred.empty:
                    LOGGER.warning("No predicted trajectories for marker %s; falling back to raw dataset + formula.", marker)
                    sub_pred = None
            else:
                sub_pred = None

            sub = dataset[dataset["marker"] == marker].copy()
            sub = sub.dropna(subset=[target_col, "GFP_bin", "timepoint"])
            if sub.empty and sub_pred is None:
                LOGGER.warning("Marker %s has no usable rows.", marker)
                continue

            sub_int = None
            if integration_df is not None:
                sub_int = integration_df[integration_df["marker"] == marker].copy()
                if sub_int.empty:
                    LOGGER.warning("No integration trajectories for marker %s; leaving integrated panels blank.", marker)
                    sub_int = None

            # Always have dt predictions for plotting the derivative panels
            if sub_pred is not None:
                sub_pred = sub_pred.dropna(subset=["timepoint", "GFP_bin"])
                preds_dt_series = sub_pred["p-ERK1-2_dt_pred"].to_numpy()
                sub_pred["__y_pred__"] = preds_dt_series
            else:
                feats = sub[[c for c in sanitized_features if c in sub.columns]].copy()
                feats = feats.apply(pd.to_numeric, errors="coerce")
                preds_dt = evaluate_formula(formula, feats)
                sub["__y_pred__"] = preds_dt

            # Integration per bin (for plotting only; metrics handled separately)
            integrate_on_the_fly = integration_df is None
            use_precomputed = integration_df is not None and sub_int is not None
            bins_source = sub_pred if sub_pred is not None else sub
            bins = (
                sorted(bins_source["GFP_bin"].dropna().unique().tolist())
                if not use_precomputed
                else sorted(sub_int["GFP_bin"].dropna().unique().tolist())
            )
            cmap = plt.colormaps.get_cmap("viridis").resampled(max(1, len(bins)))
            ax_pred_dt, ax_obs_dt, ax_pred_int, ax_pred_int_ode, ax_obs_int = axes[i]

            for cidx, b in enumerate(bins):
                if integrate_on_the_fly:
                    source = sub_pred if sub_pred is not None else sub
                    g = source[source["GFP_bin"] == b].copy()
                    g.sort_values("timepoint", inplace=True)
                    t = g["timepoint"].to_numpy(dtype=float)
                    dt_pred = g["__y_pred__"].to_numpy(dtype=float)
                    if sub_pred is not None:
                        obs_dt = (
                            g["p-ERK1-2_dt_true"].to_numpy(dtype=float)
                            if "p-ERK1-2_dt_true" in g
                            else np.full_like(t, np.nan, dtype=float)
                        )
                        obs_pe = _obs_pe_array(g, len(t))
                    else:
                        obs_dt = g[target_col].to_numpy(dtype=float)
                        obs_pe = _obs_pe_array(g, len(t))
                    mask = np.isfinite(t) & np.isfinite(dt_pred) & np.isfinite(obs_dt)
                    if mask.sum() < 2:
                        continue
                    t, dt_pred, obs_dt, obs_pe = t[mask], dt_pred[mask], obs_dt[mask], obs_pe[mask]
                    order = np.argsort(t)
                    t, dt_pred, obs_dt, obs_pe = t[order], dt_pred[order], obs_dt[order], obs_pe[order]
                    integ = np.full_like(t, np.nan, dtype=float)
                    integ[0] = obs_pe[0] if len(obs_pe) > 0 else 0.0
                    for j in range(1, len(t)):
                        dt = t[j] - t[j - 1]
                        if np.isfinite(dt_pred[j - 1]) and np.isfinite(integ[j - 1]):
                            integ[j] = integ[j - 1] + dt * dt_pred[j - 1]
                elif use_precomputed:
                    g_full = sub_int[sub_int["GFP_bin"] == b].copy()
                    g_full.sort_values("timepoint", inplace=True)
                    phases = (
                        sorted(g_full["phase"].dropna().unique().tolist())
                        if "phase" in g_full.columns
                        else [None]
                    )
                    for ph in phases:
                        g = g_full if ph is None else g_full[g_full["phase"] == ph].copy()
                        if g.empty:
                            continue
                        t = g["timepoint"].to_numpy(dtype=float)
                        dt_pred = g["pred_dt"].to_numpy(dtype=float)
                        dt_pred_ode = g["pred_dt_ode"].to_numpy(dtype=float) if "pred_dt_ode" in g.columns else None
                        obs_dt = g["obs_dt"].to_numpy(dtype=float)
                        obs_pe = g["obs_pERK1_2"].to_numpy(dtype=float)
                        integ = g["pred_integrated"].to_numpy(dtype=float)
                        integ_ode = (
                            g["pred_integrated_ode"].to_numpy(dtype=float)
                            if "pred_integrated_ode" in g.columns
                            else None
                        )
                        mask_time = np.isfinite(t)
                        if mask_time.sum() < 2:
                            continue
                        order = np.argsort(t[mask_time])
                        t = t[mask_time][order]
                        dt_pred = dt_pred[mask_time][order]
                        obs_dt = obs_dt[mask_time][order]
                        obs_pe = obs_pe[mask_time][order]
                        integ = integ[mask_time][order]
                        if dt_pred_ode is not None:
                            dt_pred_ode = dt_pred_ode[mask_time][order]
                        if integ_ode is not None:
                            integ_ode = integ_ode[mask_time][order]
                        linestyle = "-" if str(ph).lower().startswith("train") else "--"
                        label_suffix = f"{ph}" if ph is not None else None
                        label = f"{b}{' | ' + label_suffix if label_suffix else ''}"
                        color = cmap(cidx / max(1, len(bins) - 1))

                        mask_dt_pred = np.isfinite(dt_pred)
                        if mask_dt_pred.any():
                            ax_pred_dt.plot(t[mask_dt_pred], dt_pred[mask_dt_pred], color=color, label=label, linestyle=linestyle)
                        mask_obs_dt = np.isfinite(obs_dt)
                        if mask_obs_dt.any():
                            ax_obs_dt.plot(t[mask_obs_dt], obs_dt[mask_obs_dt], color=color, linestyle=linestyle)
                        mask_integ = np.isfinite(integ)
                        if mask_integ.any():
                            ax_pred_int.plot(t[mask_integ], integ[mask_integ], color=color, linestyle=linestyle)
                        mask_obs_pe = np.isfinite(obs_pe)
                        if mask_obs_pe.any():
                            ax_obs_int.plot(
                                t[mask_obs_pe],
                                obs_pe[mask_obs_pe],
                                color=color,
                                linestyle=linestyle,
                            )
                        if integ_ode is not None:
                            mask_integ_ode = np.isfinite(integ_ode)
                            if mask_integ_ode.any():
                                ax_pred_int_ode.plot(
                                    t[mask_integ_ode],
                                    integ_ode[mask_integ_ode],
                                    color=color,
                                    linestyle=linestyle,
                                )
                            else:
                                ax_pred_int_ode.text(0.5, 0.5, "ODE\nnot available", ha="center", va="center")
                        else:
                            ax_pred_int_ode.text(0.5, 0.5, "ODE\nnot available", ha="center", va="center")
                    continue
                else:
                    # No integration available; just plot dt panels
                    source = sub_pred if sub_pred is not None else sub
                    g = source[source["GFP_bin"] == b].copy()
                    g.sort_values("timepoint", inplace=True)
                    t = g["timepoint"].to_numpy(dtype=float)
                    dt_pred = g["__y_pred__"].to_numpy(dtype=float)
                    if sub_pred is not None:
                        obs_dt = (
                            g["p-ERK1-2_dt_true"].to_numpy(dtype=float)
                            if "p-ERK1-2_dt_true" in g
                            else np.full_like(t, np.nan, dtype=float)
                        )
                    else:
                        obs_dt = g[target_col].to_numpy(dtype=float)
                    obs_pe = _obs_pe_array(g, len(t))
                    mask = np.isfinite(t) & np.isfinite(dt_pred) & np.isfinite(obs_dt)
                    if mask.sum() < 2:
                        continue
                    t, dt_pred, obs_dt, obs_pe = t[mask], dt_pred[mask], obs_dt[mask], obs_pe[mask]
                    order = np.argsort(t)
                    t, dt_pred, obs_dt, obs_pe = t[order], dt_pred[order], obs_dt[order], obs_pe[order]
                    color = cmap(cidx / max(1, len(bins) - 1))
                    mask_plot = np.isfinite(t) & np.isfinite(dt_pred)
                    if mask_plot.any():
                        ax_pred_dt.plot(t[mask_plot], dt_pred[mask_plot], color=color, label=str(b))
                    mask_obs_plot = np.isfinite(t) & np.isfinite(obs_dt)
                    if mask_obs_plot.any():
                        ax_obs_dt.plot(t[mask_obs_plot], obs_dt[mask_obs_plot], color=color, linestyle="--")
                    mask_obs_pe_plot = np.isfinite(t) & np.isfinite(obs_pe)
                    if mask_obs_pe_plot.any():
                        ax_obs_int.plot(t[mask_obs_pe_plot], obs_pe[mask_obs_pe_plot], color=color, linestyle="--")
                    continue

                meas_mask = (
                    np.isin(t, args.measured_timepoints)
                    if args.dataset_mode == "per_minute"
                    else np.ones_like(t, dtype=bool)
                )
                if meas_mask.sum() > 1 and np.isfinite(integ[meas_mask]).sum() > 1:
                    obs_pe_eval = _obs_pe_array(g, len(g))
                    if len(obs_pe_eval) == len(g):
                        obs_pe_eval = obs_pe_eval[mask][order]
                    try:
                        r2 = np.corrcoef(obs_pe_eval[meas_mask], integ[meas_mask])[0, 1] ** 2
                        integ_r2_vals.append(float(r2))
                    except Exception:
                        pass

                color = cmap(cidx / max(1, len(bins) - 1))

                # Plot predicted/observed dt only where finite
                mask_plot = np.isfinite(t) & np.isfinite(dt_pred)
                if mask_plot.any():
                    ax_pred_dt.plot(t[mask_plot], dt_pred[mask_plot], color=color, label=str(b))
                mask_obs_plot = np.isfinite(t) & np.isfinite(obs_dt)
                if mask_obs_plot.any():
                    ax_obs_dt.plot(t[mask_obs_plot], obs_dt[mask_obs_plot], color=color, linestyle="--")

                # Plot integrated trajectories only where finite
                mask_integ_plot = np.isfinite(t) & np.isfinite(integ)
                if mask_integ_plot.any():
                    ax_pred_int.plot(t[mask_integ_plot], integ[mask_integ_plot], color=color)
                obs_pe_plot = _obs_pe_array(g, len(g))
                if len(obs_pe_plot) == len(g):
                    obs_pe_plot = obs_pe_plot[mask][order]
                mask_obs_pe_plot = np.isfinite(t) & np.isfinite(obs_pe_plot[: len(t)])
                if mask_obs_pe_plot.any():
                    ax_obs_int.plot(
                        t[mask_obs_pe_plot],
                        obs_pe_plot[: len(t)][mask_obs_pe_plot],
                        color=color,
                        linestyle="--",
                    )

                if use_precomputed and "pred_integrated_ode" in g.columns:
                    mask_integ_ode_plot = np.isfinite(t) & np.isfinite(integ_ode)
                    if mask_integ_ode_plot.any():
                        ax_pred_int_ode.plot(t[mask_integ_ode_plot], integ_ode[mask_integ_ode_plot], color=color)
                else:
                    ax_pred_int_ode.text(0.5, 0.5, "ODE\nnot available", ha="center", va="center")

            metrics_rows.append({"marker": marker, "model": args.model})

            ax_pred_dt.set_title(
                f"{marker} predicted dt (R2 train={_fmt_r2(train_r2)}, test={_fmt_r2(test_r2)})"
            )
            ax_obs_dt.set_title("Observed dt")
            ax_pred_int.set_title(
                f"Integrated pred (R2 tr={_fmt_r2(integ_train_r2)}, te={_fmt_r2(integ_test_r2)})"
            )
            ax_pred_int_ode.set_title(
                f"ODE integrated pred (R2 tr={_fmt_r2(ode_train_r2)}, te={_fmt_r2(ode_test_r2)})"
            )
            ax_obs_int.set_title("Observed p-ERK1-2")
            for ax in (ax_pred_dt, ax_obs_dt, ax_pred_int, ax_pred_int_ode, ax_obs_int):
                ax.set_xlabel("time")
            ax_pred_dt.set_ylabel("p-ERK1-2_dt")
            if i == 0 and bins:
                ax_pred_dt.legend(
                    title="GFP_bin", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small"
                )

        fig.tight_layout()
        chunk_suffix = f"_{chunk_idx // chunk_size:02d}" if num_chunks > 1 else ""
        fig_base = f"{args.fig_base}{chunk_suffix}"
        for ext in ("png", "svg"):
            out_path = output_dir / f"{fig_base}.{ext}"
            fig.savefig(out_path, dpi=220)
            # Always write the legacy/base name for the first chunk to satisfy pipelines.
            if chunk_idx == 0 and chunk_suffix:
                base_path = output_dir / f"{args.fig_base}.{ext}"
                fig.savefig(base_path, dpi=220)
        plt.close(fig)

    # Per-marker dt curves (per-minute PySR) using predicted trajectories
    _plot_dt_curves(pred_df)

    metrics_df = integration_metrics_df.copy() if integration_metrics_df is not None else pd.DataFrame(metrics_rows)
    metrics_dir = Path(args.metrics_output_dir) if args.metrics_output_dir else output_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / args.metrics_csv
    metrics_df.to_csv(metrics_path, index=False)


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    summary = pd.read_csv(args.summary)
    make_overlay_plots(dataset, summary, args)


if __name__ == "__main__":
    main()
