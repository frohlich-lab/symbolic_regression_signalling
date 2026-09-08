"""SelectKBest sweeps for per-minute linear regression models.

The script re-runs the per-minute linear regression baseline with a
SelectKBest(f_regression) stage, sweeping K from the full feature set
down to K=2. For each K and marker it records:
- dt R2 on train/test splits
- feature-driven integrated R2
- ODE-style integrated R2
- coefficient- and variance-based feature importance

Outputs:
- metrics CSV (per marker/K/split)
- coefficient and variance importance CSVs
- boxplots for dt and ODE R2 vs K
- ribbon charts of mean feature importance vs K (coef + variance) with
  overlaid mean test dt R2.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import lsq_linear

# Register src/ on the path the way the sibling scripts do, rather than relying on
# PYTHONPATH being set by the caller: without it the rule that runs this script
# fails at import unless it happens to be launched from a shell that exports it.
_repo_src = Path(__file__).resolve().parents[3]
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from pipelines.experimental.sr_pipeline.metrics import binwise_r2  # noqa: E402

from pipelines.experimental.sr_pipeline.compute_marker_integration import (
    evaluate_formula,
    integrate_marker_ode,
    integrate_single_marker,
    make_formula_function,
    RAW_PERK_COL,
    attach_raw_perk,
    prepare_sr_dataset,
)
from pipelines.experimental.sr_pipeline.run_markers import (
    EXCLUDE_COLUMNS,
    apply_per_minute_sampling,
    choose_bin_split,
    sanitize_feature_names,
    _compute_regression_metrics,
)
from pipelines.experimental.sr_pipeline.seeding import canonicalize_seeds, seed_all

MEASURED_TIMEPOINTS = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "figure.dpi": 200,
    }
)
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SelectKBest sweep for per-minute linear regression.")
    parser.add_argument(
        "--raw-dataset", default=None,
        help="Raw binned measurements; when given the integrated trajectory is scored "
             "against these rather than the fitted curve.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experimental/processed/markers/markers_per_minute_fit.csv"),
        help="Per-minute CSV used for training/evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write plots and CSVs.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=2,
        help="Minimum K to evaluate (inclusive).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Optional maximum K; defaults to all features.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for GFP-bin split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/sampling.")
    parser.add_argument(
        "--test-split-policy",
        choices=("random_bins", "top_gfp_bins"),
        default="random_bins",
        help=(
            "GFP-bin train/test split policy, mirroring run_markers. 'random_bins' (default) "
            "holds out a random fraction of bins (in-distribution); 'top_gfp_bins' holds out the "
            "highest-GFP bins as an out-of-distribution dose-extrapolation test."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(42, 43, 44),
        help="Seeds to run; results are aggregated across seeds for plots.",
    )
    parser.add_argument(
        "--per-minute-max-time",
        type=float,
        default=60.0,
        help="Keep rows with timepoint ≤ this value before sampling.",
    )
    parser.add_argument(
        "--per-minute-sampling-strategy",
        choices=("max_time", "early_plus_sparse_late"),
        default="early_plus_sparse_late",
        help="Downsampling strategy mirroring the SR pipeline.",
    )
    parser.add_argument(
        "--late-sample-window",
        nargs=2,
        type=float,
        default=(30.0, 60.0),
        metavar=("START", "END"),
        help="Window used by early_plus_sparse_late.",
    )
    parser.add_argument(
        "--late-sample-points",
        type=int,
        default=15,
        help="Late-window samples per marker/bin for early_plus_sparse_late.",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_TIMEPOINTS,
        help="Measured timepoints to retain for per-minute evaluation.",
    )
    return parser.parse_args()


def _build_linear_expr(intercept: float, coef_map: Dict[str, float]) -> sp.Expr:
    expr = sp.Float(intercept)
    for name, coef in coef_map.items():
        expr += sp.Float(coef) * sp.Symbol(name)
    return expr


def _fit_constrained_selectk(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int,
    perk_name: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float], List[str]]:
    """
    Fit SelectK + linear regression with constraint pERK coefficient <= 0.
    Returns train/test preds, intercept (unscaled), coef map (unscaled), selected names.
    """
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(X_train)
    Xte_scaled = scaler.transform(X_test)

    # Always include pERK in the selected set when available.
    perk_in_features = perk_name in feature_cols
    if k >= len(feature_cols):
        support = np.ones(len(feature_cols), dtype=bool)
        selected_names = list(feature_cols)
        Xtr_sel = Xtr_scaled
        Xte_sel = Xte_scaled
    else:
        if perk_in_features:
            base_features = [f for f in feature_cols if f != perk_name]
            select_k = max(1, min(len(base_features), k - 1))
            selector = SelectKBest(score_func=f_regression, k=select_k)
            selector.fit(Xtr_scaled[:, [feature_cols.index(f) for f in base_features]], y_train)
            base_support = selector.get_support()
            base_selected = [f for f, keep in zip(base_features, base_support) if keep]
            selected_names = [perk_name] + base_selected
        else:
            selector_k = max(1, k)
            selector = SelectKBest(score_func=f_regression, k=selector_k)
            selector.fit(Xtr_scaled, y_train)
            support_mask = selector.get_support()
            selected_names = [name for name, keep in zip(feature_cols, support_mask) if keep]

        # Build support mask aligned to feature_cols
        support = np.zeros(len(feature_cols), dtype=bool)
        for name in selected_names:
            idx = feature_cols.index(name)
            support[idx] = True
        indices = [i for i, keep in enumerate(support) if keep]
        Xtr_sel = Xtr_scaled[:, indices]
        Xte_sel = Xte_scaled[:, indices]

    Xtr_aug = np.column_stack([Xtr_sel, np.ones(len(y_train))])
    lb = np.full(Xtr_aug.shape[1], -np.inf)
    ub = np.full(Xtr_aug.shape[1], np.inf)
    if perk_name in selected_names:
        perk_idx = selected_names.index(perk_name)
        ub[perk_idx] = 0.0  # enforce non-positive pERK coefficient

    res = lsq_linear(Xtr_aug, y_train, bounds=(lb, ub), lsmr_tol="auto", verbose=0)
    coef_scaled = res.x[:-1]
    intercept_scaled = float(res.x[-1])

    preds_train = Xtr_sel @ coef_scaled + intercept_scaled
    preds_test = Xte_sel @ coef_scaled + intercept_scaled if len(Xte_sel) > 0 else np.array([])

    scale_all = np.asarray(scaler.scale_)
    mean_all = np.asarray(scaler.mean_)
    scale_sel = np.where(scale_all[support] == 0.0, 1.0, scale_all[support])
    mean_sel = mean_all[support]

    coef_unscaled = coef_scaled / scale_sel
    intercept_unscaled = intercept_scaled - np.sum((coef_scaled * mean_sel) / scale_sel)
    coef_map = dict(zip(selected_names, coef_unscaled))

    return preds_train, preds_test, float(intercept_unscaled), coef_map, selected_names


def _plot_boxplot(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    output: Path,
    title: str,
) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="k", y=metric_col, hue="split", palette="Set2")
    plt.xlabel("K (features)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def _plot_ribbon(
    importance_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    importance_type: str,
    output: Path,
    title: str,
) -> None:
    data = importance_df[importance_df["importance_type"] == importance_type].copy()
    if data.empty:
        return
    mean_imp = (
        data.groupby(["k", "feature"])["normalized_importance"]
        .mean()
        .reset_index()
    )
    pivot = mean_imp.pivot(index="k", columns="feature", values="normalized_importance").fillna(0.0)
    pivot = pivot.sort_index()
    if pivot.empty:
        return
    feature_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    plt.figure(figsize=(10, 5))
    plt.stackplot(pivot.index, pivot[feature_order].T, labels=feature_order, alpha=0.8)

    mean_ode_r2 = (
        metric_df[metric_df["split"] == "test"]
        .groupby("k")["ode_integ_r2_median"]
        .mean()
        .reindex(pivot.index)
    )
    mean_ode_r2 = mean_ode_r2.clip(lower=0.0)
    plt.plot(mean_ode_r2.index, mean_ode_r2.values, color="black", linewidth=2, label="mean test ODE integrated R2")
    plt.xlabel("K (features)")
    plt.ylabel("Mean normalized importance")
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def main() -> None:
    args = parse_args()

    seeds = canonicalize_seeds(args.random_state, args.seeds)

    raw = pd.read_csv(args.dataset)
    full_data, target_col = prepare_sr_dataset(raw)
    if getattr(args, "raw_dataset", None):
        full_data = attach_raw_perk(full_data, args.raw_dataset)
    metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    total_steps = 0
    for seed in seeds:
        seed_all(seed)
        sampled = apply_per_minute_sampling(
            raw,
            strategy=args.per_minute_sampling_strategy,
            max_time=args.per_minute_max_time,
            late_window=tuple(args.late_sample_window),
            late_points=args.late_sample_points,
            random_state=seed,
        )
        data, target_col_sampled = prepare_sr_dataset(sampled)
        # assert consistent target column naming
        target_col = target_col_sampled
        data = data.dropna(subset=[target_col])
        feature_cols = [c for c in data.columns
                    if c not in EXCLUDE_COLUMNS and c != target_col and c != RAW_PERK_COL]
        if len(feature_cols) < 2:
            continue
        max_k = min(args.max_k or len(feature_cols), len(feature_cols))
        min_k = max(2, args.min_k)
        k_values = list(range(max_k, min_k - 1, -1))
        k_values = sorted(set(k_values), reverse=True)
        markers = sorted(data["marker"].dropna().unique().tolist())
        total_steps += len(markers) * len(k_values)
    if total_steps == 0:
        raise ValueError("No markers or K values available for the SelectK sweep.")

    def _progress(step_idx: int, marker: str, k_val: int) -> None:
        pct = (step_idx / total_steps) * 100.0
        print(f"[SelectK] {step_idx}/{total_steps} ({pct:5.1f}%) marker={marker} k={k_val}", flush=True)

    def _clamp_r2(val: Optional[float]) -> float:
        if val is None or not np.isfinite(val):
            return np.nan
        return float(max(0.0, val))

    step_counter = 0
    for seed in seeds:
        seed_all(seed)
        sampled = apply_per_minute_sampling(
            raw,
            strategy=args.per_minute_sampling_strategy,
            max_time=args.per_minute_max_time,
            late_window=tuple(args.late_sample_window),
            late_points=args.late_sample_points,
            random_state=seed,
        )
        data, target_col = prepare_sr_dataset(sampled)
        data = data.dropna(subset=[target_col])
        feature_cols = [c for c in data.columns
                    if c not in EXCLUDE_COLUMNS and c != target_col and c != RAW_PERK_COL]
        if len(feature_cols) < 2:
            continue
        max_k = min(args.max_k or len(feature_cols), len(feature_cols))
        min_k = max(2, args.min_k)
        k_values = list(range(max_k, min_k - 1, -1))
        k_values = sorted(set(k_values), reverse=True)
        markers = sorted(data["marker"].dropna().unique().tolist())

        for marker in markers:
            subset = data[data["marker"] == marker].copy()
            subset = subset.dropna(subset=feature_cols + [target_col])
            if subset.empty:
                continue
            train_bins, test_bins = choose_bin_split(
                subset,
                test_size=args.test_size,
                random_state=seed,
                split_policy=args.test_split_policy,
            )
            if not train_bins or not test_bins:
                continue
            subset = subset[subset["GFP_bin"].isin(train_bins | test_bins)].copy()
            train_df = subset[subset["GFP_bin"].isin(train_bins)].copy()
            test_df = subset[subset["GFP_bin"].isin(test_bins)].copy()
            if train_df.empty or test_df.empty:
                continue

            for k in k_values:
                step_counter += 1
                _progress(step_counter, marker, k)
                X_train = train_df[feature_cols].copy()
                X_test = test_df[feature_cols].copy()
                y_train = train_df[target_col].astype(float).to_numpy()
                y_test = test_df[target_col].astype(float).to_numpy()

                perk_name = sanitize_feature_names(["p_ERK1_2"])[0]
                preds_train, preds_test, intercept, coef_map, selected = _fit_constrained_selectk(
                    X_train,
                    y_train,
                    X_test,
                    feature_cols,
                    k,
                    perk_name,
                )

                # Build full-grid frames for integration (use full dataset, but keep train/test bin split)
                full_marker = full_data[full_data["marker"] == marker].copy()
                full_marker = full_marker.dropna(subset=[target_col, "GFP_bin", "timepoint"])
                full_train = full_marker[full_marker["GFP_bin"].isin(train_bins)].copy()
                full_test = full_marker[full_marker["GFP_bin"].isin(test_bins)].copy()

                def _add_preds(df_full: pd.DataFrame) -> pd.DataFrame:
                    if df_full.empty:
                        return df_full
                    df_full = df_full.copy()
                    # ensure all selected features exist
                    for feat in selected:
                        if feat not in df_full.columns:
                            df_full[feat] = 0.0
                    vals = np.full(len(df_full), intercept, dtype=float)
                    for feat, coef in coef_map.items():
                        vals += coef * pd.to_numeric(df_full[feat], errors="coerce").to_numpy(dtype=float)
                    df_full["__y_pred__"] = vals
                    return df_full

                train_with_pred = train_df.copy()
                train_with_pred["__y_pred__"] = preds_train
                test_with_pred = test_df.copy()
                test_with_pred["__y_pred__"] = preds_test

                full_train_with_pred = _add_preds(full_train)
                full_test_with_pred = _add_preds(full_test)

                dt_r2_train = _clamp_r2(
                    binwise_r2(
                        train_with_pred,
                        target_col=target_col,
                        pred_col="__y_pred__",
                        measured_timepoints=args.measured_timepoints,
                        dataset_mode="per_minute",
                        clamp_negative=True,
                    )
                )
                dt_r2_test = _clamp_r2(
                    binwise_r2(
                        test_with_pred,
                        target_col=target_col,
                        pred_col="__y_pred__",
                        measured_timepoints=args.measured_timepoints,
                        dataset_mode="per_minute",
                        clamp_negative=True,
                    )
                )

                integ_train, _ = integrate_single_marker(
                    full_train_with_pred if not full_train_with_pred.empty else train_with_pred,
                    target_col,
                    args.measured_timepoints,
                    "per_minute",
                )
                integ_test, _ = integrate_single_marker(
                    full_test_with_pred if not full_test_with_pred.empty else test_with_pred,
                    target_col,
                    args.measured_timepoints,
                    "per_minute",
                )

                expr = _build_linear_expr(intercept, coef_map)
                try:
                    formula_fn_jax, symbols = make_formula_function(expr, backend="jax")
                    ode_train, _ = integrate_marker_ode(
                        full_train_with_pred if not full_train_with_pred.empty else train_with_pred,
                        target_col,
                        args.measured_timepoints,
                        "per_minute",
                        formula_fn_jax,
                        symbols,
                        "p_ERK1_2",
                        restrict_to_measured=False,
                        include_trajectories=False,
                        phase="train",
                        log_label=f"{marker} train",
                    )
                    ode_test, _ = integrate_marker_ode(
                        full_test_with_pred if not full_test_with_pred.empty else test_with_pred,
                        target_col,
                        args.measured_timepoints,
                        "per_minute",
                        formula_fn_jax,
                        symbols,
                        "p_ERK1_2",
                        restrict_to_measured=False,
                        include_trajectories=False,
                        phase="test",
                        log_label=f"{marker} test",
                    )
                except Exception:
                    ode_train, ode_test = (
                        {
                            "ode_integ_r2_median": np.nan,
                            "ode_integ_rel_mae_mean_bins": np.nan,
                            "ode_integ_rel_mae_median_bins": np.nan,
                        },
                        {
                            "ode_integ_r2_median": np.nan,
                            "ode_integ_rel_mae_mean_bins": np.nan,
                            "ode_integ_rel_mae_median_bins": np.nan,
                        },
                    )

                metrics_rows.extend(
                    [
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "split": "train",
                            "dt_r2": dt_r2_train,
                            "dt_rel_mae_mean_bins": integ_train.get("dt_rel_mae_mean_bins"),
                            "dt_rel_mae_median_bins": integ_train.get("dt_rel_mae_median_bins"),
                            "integ_r2_median": integ_train.get("integ_r2_median"),
                            "integ_rel_mae_mean_bins": integ_train.get("integ_rel_mae_mean_bins"),
                            "integ_rel_mae_median_bins": integ_train.get("integ_rel_mae_median_bins"),
                            "ode_integ_r2_median": ode_train.get("ode_integ_r2_median"),
                            "ode_integ_rel_mae_mean_bins": ode_train.get("ode_integ_rel_mae_mean_bins"),
                            "ode_integ_rel_mae_median_bins": ode_train.get("ode_integ_rel_mae_median_bins"),
                        },
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "split": "test",
                            "dt_r2": dt_r2_test,
                            "dt_rel_mae_mean_bins": integ_test.get("dt_rel_mae_mean_bins"),
                            "dt_rel_mae_median_bins": integ_test.get("dt_rel_mae_median_bins"),
                            "integ_r2_median": integ_test.get("integ_r2_median"),
                            "integ_rel_mae_mean_bins": integ_test.get("integ_rel_mae_mean_bins"),
                            "integ_rel_mae_median_bins": integ_test.get("integ_rel_mae_median_bins"),
                            "ode_integ_r2_median": ode_test.get("ode_integ_r2_median"),
                            "ode_integ_rel_mae_mean_bins": ode_test.get("ode_integ_rel_mae_mean_bins"),
                            "ode_integ_rel_mae_median_bins": ode_test.get("ode_integ_rel_mae_median_bins"),
                        },
                    ]
                )

                total_coef = float(sum(abs(v) for v in coef_map.values()))
                for feat in feature_cols:
                    weight = float(abs(coef_map.get(feat, 0.0)))
                    importance_rows.append(
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "feature": feat,
                            "importance_type": "coef",
                            "raw_importance": weight,
                            "normalized_importance": (weight / total_coef) if total_coef > 0 else np.nan,
                        }
                    )

                grouped = subset.groupby("GFP_bin")[feature_cols].mean()
                raw_variance: Dict[str, float] = {feat: 0.0 for feat in feature_cols}
                if not grouped.empty and selected:
                    for feat, coef in coef_map.items():
                        vals = grouped[feat].to_numpy(dtype=float)
                        contrib = vals * coef
                        contrib = contrib[np.isfinite(contrib)]
                        if contrib.size > 0:
                            raw_variance[feat] = float(max(0.0, np.var(contrib)))
                    total_raw = float(sum(raw_variance.values()))
                else:
                    total_raw = 0.0
                for feat in feature_cols:
                    raw_val = raw_variance.get(feat, 0.0)
                    importance_rows.append(
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "feature": feat,
                            "importance_type": "variance",
                            "raw_importance": raw_val,
                            "normalized_importance": (raw_val / total_raw) if total_raw > 0 else np.nan,
                        }
                    )

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = outdir / "select_k_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    agg_cols = [
        "dt_r2",
        "dt_rel_mae_mean_bins",
        "dt_rel_mae_median_bins",
        "integ_r2_median",
        "integ_rel_mae_mean_bins",
        "integ_rel_mae_median_bins",
        "ode_integ_r2_median",
        "ode_integ_rel_mae_mean_bins",
        "ode_integ_rel_mae_median_bins",
    ]
    agg_cols = [c for c in agg_cols if c in metrics_df.columns]
    agg_metrics = (
        metrics_df.groupby(["marker", "k", "split"], dropna=False)[agg_cols]
        .mean()
        .reset_index()
    )
    agg_metrics_path = outdir / "select_k_metrics_agg.csv"
    agg_metrics.to_csv(agg_metrics_path, index=False)

    importance_df = pd.DataFrame(importance_rows)
    importance_path = outdir / "select_k_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    agg_importance = (
        importance_df.groupby(["marker", "k", "feature", "importance_type"], dropna=False)[
            ["raw_importance", "normalized_importance"]
        ]
        .mean()
        .reset_index()
    )
    agg_importance_path = outdir / "select_k_importance_agg.csv"
    agg_importance.to_csv(agg_importance_path, index=False)

    # Add per-marker overall rows (mean of train/test where available) for plotting convenience (using seed-aggregated metrics).
    overall_rows: List[Dict[str, object]] = []
    for (marker, k), grp in agg_metrics.groupby(["marker", "k"]):
        overall_rows.append(
            {
                "marker": marker,
                "k": k,
                "split": "overall",
                "dt_r2": float(np.nanmean(grp["dt_r2"])) if not grp["dt_r2"].isna().all() else np.nan,
                "dt_rel_mae_mean_bins": float(np.nanmean(grp["dt_rel_mae_mean_bins"]))
                if "dt_rel_mae_mean_bins" in grp and not grp["dt_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "dt_rel_mae_median_bins": float(np.nanmean(grp["dt_rel_mae_median_bins"]))
                if "dt_rel_mae_median_bins" in grp and not grp["dt_rel_mae_median_bins"].isna().all()
                else np.nan,
                "integ_r2_median": float(np.nanmean(grp["integ_r2_median"]))
                if not grp["integ_r2_median"].isna().all()
                else np.nan,
                "integ_rel_mae_mean_bins": float(np.nanmean(grp["integ_rel_mae_mean_bins"]))
                if "integ_rel_mae_mean_bins" in grp and not grp["integ_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "integ_rel_mae_median_bins": float(np.nanmean(grp["integ_rel_mae_median_bins"]))
                if "integ_rel_mae_median_bins" in grp and not grp["integ_rel_mae_median_bins"].isna().all()
                else np.nan,
                "ode_integ_r2_median": float(np.nanmean(grp["ode_integ_r2_median"]))
                if not grp["ode_integ_r2_median"].isna().all()
                else np.nan,
                "ode_integ_rel_mae_mean_bins": float(np.nanmean(grp["ode_integ_rel_mae_mean_bins"]))
                if "ode_integ_rel_mae_mean_bins" in grp and not grp["ode_integ_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "ode_integ_rel_mae_median_bins": float(np.nanmean(grp["ode_integ_rel_mae_median_bins"]))
                if "ode_integ_rel_mae_median_bins" in grp and not grp["ode_integ_rel_mae_median_bins"].isna().all()
                else np.nan,
            }
        )
    metrics_plot_df = agg_metrics.copy()
    if overall_rows:
        metrics_plot_df = pd.concat([metrics_plot_df, pd.DataFrame(overall_rows)], axis=0, ignore_index=True)

    _plot_boxplot(metrics_plot_df, "dt_r2", "dt R²", outdir / "boxplot_dt_r2.png", "dt R² vs K (train/test/overall)")
    _plot_boxplot(
        metrics_plot_df,
        "integ_r2_median",
        "Feature-driven integrated R²",
        outdir / "boxplot_integ_r2.png",
        "Integrated R² vs K (train/test/overall)",
    )
    _plot_boxplot(
        metrics_plot_df,
        "ode_integ_r2_median",
        "ODE integrated R²",
        outdir / "boxplot_ode_r2.png",
        "ODE integrated R² vs K (train/test/overall)",
    )

    _plot_ribbon(
        agg_importance,
        metrics_plot_df,
        "coef",
        outdir / "ribbon_coef_importance.png",
        "Mean coefficient importance vs K",
    )
    _plot_ribbon(
        agg_importance,
        metrics_plot_df,
        "variance",
        outdir / "ribbon_variance_importance.png",
        "Mean variance-weighted importance vs K",
    )
    print(f"Wrote per-seed metrics to {metrics_path}")
    print(f"Wrote seed-aggregated metrics to {agg_metrics_path}")
    print(f"Wrote per-seed importance to {importance_path}")
    print(f"Wrote seed-aggregated importance to {agg_importance_path}")


if __name__ == "__main__":
    main()
