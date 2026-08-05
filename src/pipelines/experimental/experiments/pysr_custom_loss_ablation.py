#!/usr/bin/env python3
"""Ablation study for PySR custom-loss stability penalties on experimental data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class AblationCondition:
    key: str
    label: str
    extra_flags: tuple[str, ...]


CONDITIONS: tuple[AblationCondition, ...] = (
    AblationCondition("normal", "Normal", ()),
    AblationCondition(
        "no_linear_stability",
        "No linear stability",
        ("--disable-linear-stability-penalty",),
    ),
    AblationCondition(
        "no_inverse_stability",
        "No inverse stability",
        ("--disable-inverse-stability-penalty",),
    ),
    AblationCondition(
        "no_stability_penalties",
        "No stability penalties",
        (
            "--disable-linear-stability-penalty",
            "--disable-inverse-stability-penalty",
        ),
    ),
)


DISPLAY_LABELS = {
    "normal": "Normal",
    "no_linear_stability": "No linear\nstability",
    "no_inverse_stability": "No inverse\nstability",
    "no_stability_penalties": "No stability\npenalties",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PySR custom-loss ablations and plot marker-level boxplots."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--per-minute-dataset", type=Path, required=True)
    parser.add_argument("--group-definitions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-markers-script",
        type=Path,
        default=Path("src/pipelines/experimental/sr_pipeline/run_markers.py"),
        help="Path to the PySR marker runner script.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=(0.0, 5.0, 10.0, 15.0, 30.0, 60.0),
    )
    parser.add_argument("--per-minute-max-time", type=float, default=60.0)
    parser.add_argument(
        "--per-minute-sampling-strategy",
        choices=("max_time", "early_plus_sparse_late"),
        default="early_plus_sparse_late",
    )
    parser.add_argument(
        "--late-sample-window",
        nargs=2,
        type=float,
        default=(30.0, 60.0),
        metavar=("START", "END"),
    )
    parser.add_argument("--late-sample-points", type=int, default=15)
    parser.add_argument("--max-iterations", type=int, default=700)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--populations", type=int, default=30)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--parsimony", type=float, default=0.8)
    parser.add_argument("--metric-column", type=str, default="ode_integ_r2_median")
    parser.add_argument("--metric-dataset-mode", type=str, default="per_minute")
    parser.add_argument("--metric-feature-mode", type=str, default="all")
    parser.add_argument(
        "--metric-source",
        choices=("integration", "summary"),
        default="integration",
        help="Read ablation metric from integration metrics CSV or summary CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path for marker-level metrics.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Output PNG path for the boxplot.",
    )
    parser.add_argument(
        "--output-plot-svg",
        type=Path,
        default=None,
        help="Output SVG path for the boxplot.",
    )
    return parser.parse_args()


def _summary_path(run_root: Path) -> Path:
    canonical = run_root / "reports" / "summary" / "marker_marker_summary.csv"
    if canonical.exists():
        return canonical
    candidates = sorted(run_root.rglob("marker_marker_summary.csv"))
    if not candidates:
        raise FileNotFoundError(f"No marker summary found under {run_root}")
    return candidates[0]


def _integration_metrics_path(run_root: Path, dataset_mode: str) -> Path:
    filename = f"marker_integration_metrics_{dataset_mode}.csv"
    candidates = sorted(run_root.rglob(filename))
    if not candidates:
        raise FileNotFoundError(
            f"No integration metrics file '{filename}' found under {run_root}"
        )
    return candidates[0]


def _build_command(
    args: argparse.Namespace,
    condition: AblationCondition,
    condition_run_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(args.run_markers_script),
        "--dataset",
        str(args.dataset),
        "--per-minute-dataset",
        str(args.per_minute_dataset),
        "--output-dir",
        str(condition_run_dir),
        "--group-definitions-csv",
        str(args.group_definitions_csv),
        "--models",
        "pysr",
        "--random-state",
        str(args.seed),
        "--seeds",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--measured-timepoints",
        *[str(t) for t in args.measured_timepoints],
        "--per-minute-max-time",
        str(args.per_minute_max_time),
        "--per-minute-sampling-strategy",
        str(args.per_minute_sampling_strategy),
        "--late-sample-window",
        str(args.late_sample_window[0]),
        str(args.late_sample_window[1]),
        "--late-sample-points",
        str(args.late_sample_points),
        "--max-iterations",
        str(args.max_iterations),
        "--population-size",
        str(args.population_size),
        "--populations",
        str(args.populations),
        "--max-size",
        str(args.max_size),
        "--parsimony",
        str(args.parsimony),
    ]
    cmd.extend(condition.extra_flags)
    return cmd


def _extract_metric_rows(
    summary: pd.DataFrame,
    condition: AblationCondition,
    metric_column: str,
    dataset_mode: str,
    feature_mode: str,
    seed: int,
) -> pd.DataFrame:
    if metric_column not in summary.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in summary.")
    df = summary.copy()
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower() == "pysr"]
    if "dataset_mode" in df.columns and dataset_mode:
        df = df[df["dataset_mode"].astype(str) == dataset_mode]
    if "feature_mode" in df.columns and feature_mode:
        df = df[df["feature_mode"].astype(str) == feature_mode]
    if "marker" not in df.columns:
        raise ValueError("Summary is missing required 'marker' column.")
    df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
    df = df[np.isfinite(df[metric_column])]
    if df.empty:
        raise ValueError(
            f"No rows left for condition '{condition.key}' after filtering metric/model/mode."
        )
    df = (
        df.groupby("marker", as_index=False)[metric_column]
        .mean()
        .rename(columns={metric_column: "metric_value"})
    )
    df["metric"] = metric_column
    df["condition"] = condition.key
    df["condition_label"] = condition.label
    df["seed"] = int(seed)
    return df[
        ["condition", "condition_label", "marker", "metric", "metric_value", "seed"]
    ]


def _extract_metric_rows_from_integration(
    metrics: pd.DataFrame,
    condition: AblationCondition,
    metric_column: str,
    dataset_mode: str,
    seed: int,
) -> pd.DataFrame:
    if metric_column not in metrics.columns:
        raise ValueError(f"Metric column '{metric_column}' not found in integration metrics.")
    df = metrics.copy()
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower() == "pysr"]
    if "dataset_mode" in df.columns and dataset_mode:
        df = df[df["dataset_mode"].astype(str) == dataset_mode]
    if "marker" not in df.columns:
        raise ValueError("Integration metrics are missing required 'marker' column.")
    df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce")
    df = df[np.isfinite(df[metric_column])]
    if df.empty:
        raise ValueError(
            f"No integration rows left for condition '{condition.key}' after filtering."
        )
    df = (
        df.groupby("marker", as_index=False)[metric_column]
        .mean()
        .rename(columns={metric_column: "metric_value"})
    )
    df["metric"] = metric_column
    df["condition"] = condition.key
    df["condition_label"] = condition.label
    df["seed"] = int(seed)
    return df[
        ["condition", "condition_label", "marker", "metric", "metric_value", "seed"]
    ]


def _format_metric_label(metric_name: str) -> str:
    if metric_name == "ode_integ_r2_median":
        return r"Median ODE-integrated $R^2$"
    if metric_name == "integ_r2_median":
        return r"Median integrated $R^2$"
    if metric_name.endswith("_r2") or metric_name.endswith("_r2_median"):
        return metric_name.replace("_", " ")
    return metric_name.replace("_", " ")


def _plot_boxplot(
    metrics_df: pd.DataFrame,
    output_png: Path,
    output_svg: Path | None,
    metric_name: str,
) -> None:
    order = [condition.key for condition in CONDITIONS]
    labels = [DISPLAY_LABELS.get(condition.key, condition.label) for condition in CONDITIONS]
    values = [
        metrics_df.loc[metrics_df["condition"] == condition, "metric_value"].to_numpy()
        for condition in order
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.4), facecolor="white")
    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.56,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(1.0)

    ax.set_facecolor("white")
    ax.set_ylabel(_format_metric_label(metric_name), fontsize=11)
    ax.set_xlabel("")
    ax.set_title("")
    ax.tick_params(axis="x", labelsize=9, length=0)
    ax.tick_params(axis="y", labelsize=10, direction="out", length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", alpha=0.8, linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if metric_name.endswith("_r2") or metric_name.endswith("_r2_median"):
        ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight", facecolor="white")
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _iter_conditions() -> Iterable[AblationCondition]:
    return CONDITIONS


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or (args.output_dir / "pysr_custom_loss_ablation_metrics.csv")
    output_plot = args.output_plot or (args.output_dir / "pysr_custom_loss_ablation_boxplot.png")
    output_plot_svg = args.output_plot_svg or (
        args.output_dir / "pysr_custom_loss_ablation_boxplot.svg"
    )

    all_rows: list[pd.DataFrame] = []
    for condition in _iter_conditions():
        condition_dir = args.output_dir / condition.key
        run_root = condition_dir / "run"
        run_root.mkdir(parents=True, exist_ok=True)
        cmd = _build_command(args, condition, run_root)
        print(f"[ablation] running {condition.key}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)

        if args.metric_source == "integration":
            integration_path = _integration_metrics_path(
                run_root, args.metric_dataset_mode
            )
            integration_df = pd.read_csv(integration_path)
            rows = _extract_metric_rows_from_integration(
                integration_df,
                condition,
                metric_column=args.metric_column,
                dataset_mode=args.metric_dataset_mode,
                seed=args.seed,
            )
            source_path = integration_path
        else:
            summary_path = _summary_path(run_root)
            summary_df = pd.read_csv(summary_path)
            rows = _extract_metric_rows(
                summary_df,
                condition,
                metric_column=args.metric_column,
                dataset_mode=args.metric_dataset_mode,
                feature_mode=args.metric_feature_mode,
                seed=args.seed,
            )
            source_path = summary_path
        all_rows.append(rows)
        print(
            f"[ablation] {condition.key}: collected {len(rows)} marker rows from {source_path}",
            flush=True,
        )

    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(output_csv, index=False)
    print(f"[ablation] wrote metrics CSV to {output_csv}", flush=True)
    _plot_boxplot(
        metrics_df=merged,
        output_png=output_plot,
        output_svg=output_plot_svg,
        metric_name=args.metric_column,
    )
    print(f"[ablation] wrote boxplot to {output_plot}", flush=True)
    if output_plot_svg is not None:
        print(f"[ablation] wrote boxplot (svg) to {output_plot_svg}", flush=True)


if __name__ == "__main__":
    main()
