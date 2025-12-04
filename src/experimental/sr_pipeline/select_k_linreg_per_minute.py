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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.sr_pipeline.compute_marker_integration import (
    NoClipStateTransform,
    evaluate_formula,
    integrate_marker_ode,
    integrate_single_marker,
    make_formula_function,
    prepare_sr_dataset,
    ODE_DERIV_MARGIN_FRAC,
    ODE_DERIV_PERCENTILES,
    ODE_TSIT5_DERIV_CLIP,
)
from experimental.sr_pipeline.run_functional_groups import (
    EXCLUDE_COLUMNS,
    apply_per_minute_sampling,
    choose_bin_split,
    _compute_regression_metrics,
)

MEASURED_TIMEPOINTS = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SelectKBest sweep for per-minute linear regression.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experimental/processed/functional_groups/functional_groups_per_minute_fit.csv"),
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
    parser.add_argument(
        "--prefer-radau",
        action="store_true",
        help="Prefer SciPy Radau fallback for ODE integration when JAX is unavailable.",
    )
    return parser.parse_args()


def _compute_deriv_clip(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (-ODE_TSIT5_DERIV_CLIP, ODE_TSIT5_DERIV_CLIP)
    lo, hi = np.percentile(arr, [p * 100.0 for p in ODE_DERIV_PERCENTILES])
    span = hi - lo
    lo = lo - span * ODE_DERIV_MARGIN_FRAC
    hi = hi + span * ODE_DERIV_MARGIN_FRAC
    lo = float(np.clip(lo, -ODE_TSIT5_DERIV_CLIP, ODE_TSIT5_DERIV_CLIP))
    hi = float(np.clip(hi, -ODE_TSIT5_DERIV_CLIP, ODE_TSIT5_DERIV_CLIP))
    return lo, hi


def _extract_coefficients(
    pipeline: Pipeline,
    feature_names: Sequence[str],
) -> Tuple[float, Dict[str, float], List[str]]:
    scaler: Optional[StandardScaler] = pipeline.named_steps.get("scaler")
    selector: SelectKBest = pipeline.named_steps["selector"]
    reg: LinearRegression = pipeline.named_steps["regressor"]

    support = selector.get_support()
    selected = [name for name, keep in zip(feature_names, support) if keep]
    if scaler is None:
        coeffs = reg.coef_
        intercept = reg.intercept_
    else:
        scale_all = np.asarray(scaler.scale_)
        mean_all = np.asarray(scaler.mean_)
        scale = scale_all[support]
        mean = mean_all[support]
        scale = np.where(scale == 0.0, 1.0, scale)
        coeffs = reg.coef_ / scale
        intercept = reg.intercept_ - np.sum((reg.coef_ * mean) / scale)

    coef_map = dict(zip(selected, coeffs))
    return float(intercept), coef_map, selected


def _build_linear_expr(intercept: float, coef_map: Dict[str, float]) -> sp.Expr:
    expr = sp.Float(intercept)
    for name, coef in coef_map.items():
        expr += sp.Float(coef) * sp.Symbol(name)
    return expr


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
    plt.close()


def main() -> None:
    args = parse_args()

    seeds = list(dict.fromkeys(args.seeds)) if args.seeds else [args.random_state]

    raw = pd.read_csv(args.dataset)
    metrics_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    total_steps = 0
    for seed in seeds:
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
        feature_cols = [c for c in data.columns if c not in EXCLUDE_COLUMNS and c != target_col]
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

    def _binwise_r2(
        frame: pd.DataFrame,
        target_col: str,
        pred_col: str,
        measured_timepoints: Sequence[float],
        dataset_mode: str,
    ) -> float:
        """Compute mean R2 across GFP_bin trajectories (per-bin R2, then average)."""
        r2_vals: List[float] = []
        for _, g in frame.groupby("GFP_bin"):
            y_true = pd.to_numeric(g[target_col], errors="coerce").to_numpy()
            y_pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if dataset_mode == "per_minute":
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
            r2_vals.append(1.0 - np.sum((y_true_f - y_pred_f) ** 2) / den)
        return float(np.mean(r2_vals)) if r2_vals else np.nan

    step_counter = 0
    for seed in seeds:
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
        feature_cols = [c for c in data.columns if c not in EXCLUDE_COLUMNS and c != target_col]
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
            train_bins, test_bins = choose_bin_split(subset, test_size=args.test_size, random_state=seed)
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
                selector_k = "all" if k >= len(feature_cols) else k
                pipeline = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("selector", SelectKBest(score_func=f_regression, k=selector_k)),
                        ("regressor", LinearRegression()),
                    ]
                )
                X_train = train_df[feature_cols].copy()
                X_test = test_df[feature_cols].copy()
                y_train = train_df[target_col].astype(float).to_numpy()
                y_test = test_df[target_col].astype(float).to_numpy()
                pipeline.fit(X_train, y_train)
                preds_train = pipeline.predict(X_train)
                preds_test = pipeline.predict(X_test)

                train_with_pred = train_df.copy()
                train_with_pred["__y_pred__"] = preds_train
                test_with_pred = test_df.copy()
                test_with_pred["__y_pred__"] = preds_test

                dt_r2_train = _clamp_r2(
                    _binwise_r2(
                        train_with_pred,
                        target_col=target_col,
                        pred_col="__y_pred__",
                        measured_timepoints=args.measured_timepoints,
                        dataset_mode="per_minute",
                    )
                )
                dt_r2_test = _clamp_r2(
                    _binwise_r2(
                        test_with_pred,
                        target_col=target_col,
                        pred_col="__y_pred__",
                        measured_timepoints=args.measured_timepoints,
                        dataset_mode="per_minute",
                    )
                )

                integ_train, _ = integrate_single_marker(
                    train_with_pred,
                    target_col,
                    args.measured_timepoints,
                    "per_minute",
                )
                integ_test, _ = integrate_single_marker(
                    test_with_pred,
                    target_col,
                    args.measured_timepoints,
                    "per_minute",
                )

                intercept, coef_map, selected = _extract_coefficients(pipeline, feature_cols)
                expr = _build_linear_expr(intercept, coef_map)
                try:
                    formula_fn_np, symbols = make_formula_function(expr, backend="numpy")
                    deriv_clip = _compute_deriv_clip(evaluate_formula(expr, train_df[feature_cols].copy()))
                    ode_train, _ = integrate_marker_ode(
                        train_df.copy(),
                        target_col,
                        args.measured_timepoints,
                        "per_minute",
                        formula_fn_np,
                        symbols,
                        "p_ERK1_2",
                        formula_fn_fallback=formula_fn_np,
                        state_transform=NoClipStateTransform(),
                        deriv_clip=deriv_clip,
                        prefer_radau=args.prefer_radau,
                        include_trajectories=False,
                        phase="train",
                    )
                    ode_test, _ = integrate_marker_ode(
                        test_df.copy(),
                        target_col,
                        args.measured_timepoints,
                        "per_minute",
                        formula_fn_np,
                        symbols,
                        "p_ERK1_2",
                        formula_fn_fallback=formula_fn_np,
                        state_transform=NoClipStateTransform(),
                        deriv_clip=deriv_clip,
                        prefer_radau=args.prefer_radau,
                        include_trajectories=False,
                        phase="test",
                    )
                except Exception:
                    ode_train, ode_test = {"ode_integ_r2_median": np.nan}, {"ode_integ_r2_median": np.nan}

                metrics_rows.extend(
                    [
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "split": "train",
                            "dt_r2": dt_r2_train,
                            "integ_r2_median": integ_train.get("integ_r2_median"),
                            "ode_integ_r2_median": ode_train.get("ode_integ_r2_median"),
                        },
                        {
                            "marker": marker,
                            "k": k,
                            "seed": seed,
                            "split": "test",
                            "dt_r2": dt_r2_test,
                            "integ_r2_median": integ_test.get("integ_r2_median"),
                            "ode_integ_r2_median": ode_test.get("ode_integ_r2_median"),
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

    agg_metrics = (
        metrics_df.groupby(["marker", "k", "split"], dropna=False)[["dt_r2", "integ_r2_median", "ode_integ_r2_median"]]
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
                "integ_r2_median": float(np.nanmean(grp["integ_r2_median"]))
                if not grp["integ_r2_median"].isna().all()
                else np.nan,
                "ode_integ_r2_median": float(np.nanmean(grp["ode_integ_r2_median"]))
                if not grp["ode_integ_r2_median"].isna().all()
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
