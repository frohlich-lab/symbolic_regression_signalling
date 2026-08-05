"""Plot representative PySR per-minute p-ERK trajectories around selected R2 levels."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from textwrap import fill
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MEASURED_DEFAULT = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot representative PySR per-minute trajectories near selected R2 thresholds."
    )
    parser.add_argument("--trajectories", type=Path, required=True, help="marker_integration_trajectories_per_minute.csv")
    parser.add_argument(
        "--raw-dataset",
        type=Path,
        default=None,
        help="Optional markers_time_series.csv for plotting raw measured p-ERK crosses.",
    )
    parser.add_argument("--output-plot", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with selected examples.")
    parser.add_argument(
        "--targets",
        nargs="*",
        type=float,
        default=(0.50, 0.55, 0.60, 0.65, 0.70),
        help="Target measured-timepoint R2 values to match.",
    )
    parser.add_argument(
        "--override-example",
        action="append",
        default=[],
        help="Optional manual example in the form target,marker,gfp_bin,seed.",
    )
    parser.add_argument(
        "--examples-per-target",
        type=int,
        default=1,
        help="Number of examples to show for each target R2 level.",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_DEFAULT,
        help="Measured timepoints used for the threshold metric.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter.")
    parser.add_argument("--phase", type=str, default="test", help="Trajectory phase to inspect.")
    parser.add_argument("--model", type=str, default="PySR", help="Model name to inspect.")
    parser.add_argument("--dataset-mode", type=str, default="per_minute", help="Dataset mode to inspect.")
    return parser.parse_args()


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = np.asarray(y_true[mask], dtype=float)
    yp = np.asarray(y_pred[mask], dtype=float)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return max(0.0, 1.0 - ss_res / ss_tot)


def _choose_prediction_column(df: pd.DataFrame) -> str:
    for col in ("pred_integrated_ode", "pred_integrated"):
        if col in df.columns:
            return col
    raise KeyError("No integrated prediction column found in trajectories.")


def _summarize_bins(df: pd.DataFrame, measured_timepoints: Sequence[float], pred_col: str) -> pd.DataFrame:
    measured = set(float(t) for t in measured_timepoints)
    eval_df = df[df["timepoint"].isin(measured)].copy()
    group_cols = ["marker", "GFP_bin"]
    if "seed" in eval_df.columns:
        group_cols.append("seed")

    rows: List[Dict[str, object]] = []
    for keys, grp in eval_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        payload = dict(zip(group_cols, keys))
        y_obs = pd.to_numeric(grp["obs_pERK1_2"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(grp[pred_col], errors="coerce").to_numpy(dtype=float)
        r2 = _r2_score(y_obs, y_pred)
        if not np.isfinite(r2):
            continue
        payload["r2_measured"] = float(r2)
        payload["n_measured"] = int(grp["timepoint"].nunique())
        rows.append(payload)
    return pd.DataFrame(rows)


def _parse_overrides(items: Sequence[str]) -> Dict[float, List[Dict[str, object]]]:
    parsed: Dict[float, List[Dict[str, object]]] = {}
    for item in items:
        parts = [p.strip() for p in str(item).split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid override example '{item}'. Expected target,marker,gfp_bin,seed.")
        target_str, marker, gfp_bin_str, seed_str = parts
        parsed.setdefault(float(target_str), []).append(
            {
                "marker": marker,
                "GFP_bin": int(gfp_bin_str),
                "seed": int(seed_str),
            }
        )
    return parsed


def _select_examples(
    summary: pd.DataFrame,
    targets: Sequence[float],
    overrides: Dict[float, List[Dict[str, object]]] | None = None,
    examples_per_target: int = 1,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    remaining = summary.copy()
    rows = []
    overrides = overrides or {}
    used_keys = set()
    for target in targets:
        picked_count = 0
        for spec in overrides.get(float(target), []):
            match = remaining[
                (remaining["marker"] == spec["marker"])
                & (remaining["GFP_bin"] == spec["GFP_bin"])
                & (remaining["seed"] == spec["seed"])
            ]
            if match.empty:
                raise ValueError(
                    "Override example not found in available bins: "
                    f"{target},{spec['marker']},{spec['GFP_bin']},{spec['seed']}"
                )
            idx = match.index[0]
            picked = remaining.loc[idx].copy()
            key = (picked["marker"], int(picked["GFP_bin"]), int(picked["seed"]))
            if key in used_keys:
                continue
            picked["target_r2"] = float(target)
            picked["abs_delta"] = abs(float(picked["r2_measured"]) - float(target))
            picked["example_rank"] = picked_count + 1
            rows.append(picked)
            used_keys.add(key)
            remaining = remaining.drop(index=idx)
            picked_count += 1
            if picked_count >= examples_per_target:
                break

        while picked_count < examples_per_target and not remaining.empty:
            available = remaining[
                ~remaining.apply(
                    lambda r: (r["marker"], int(r["GFP_bin"]), int(r["seed"])) in used_keys,
                    axis=1,
                )
            ].copy()
            if available.empty:
                break
            idx = (available["r2_measured"] - float(target)).abs().idxmin()
            picked = available.loc[idx].copy()
            key = (picked["marker"], int(picked["GFP_bin"]), int(picked["seed"]))
            picked["target_r2"] = float(target)
            picked["abs_delta"] = abs(float(picked["r2_measured"]) - float(target))
            picked["example_rank"] = picked_count + 1
            rows.append(picked)
            used_keys.add(key)
            remaining = remaining.drop(index=idx)
            picked_count += 1
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target_r2", "example_rank"]).reset_index(drop=True)


def _write_placeholder(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 2.8))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_examples(
    df: pd.DataFrame,
    chosen: pd.DataFrame,
    pred_col: str,
    output: Path,
    measured_timepoints: Sequence[float],
    raw_df: pd.DataFrame | None = None,
    target_order: Sequence[float] | None = None,
    examples_per_target: int = 1,
) -> None:
    if chosen.empty:
        _write_placeholder(output, "No PySR per-minute test trajectories available for threshold sanity plot.")
        return

    targets = [float(t) for t in (target_order or sorted(chosen["target_r2"].unique().tolist()))]
    ncols = len(targets)
    nrows = max(1, int(examples_per_target))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 2.95 * nrows), squeeze=False, sharex=True)
    measured = set(float(t) for t in measured_timepoints)

    for row_idx in range(nrows):
        for col_idx, target in enumerate(targets):
            ax = axes[row_idx, col_idx]
            match = chosen[
                (np.isclose(chosen["target_r2"], float(target)))
                & (chosen["example_rank"] == row_idx + 1)
            ]
            if match.empty:
                ax.axis("off")
                continue
            row = match.iloc[0]

            sub = df[(df["marker"] == row["marker"]) & (df["GFP_bin"] == row["GFP_bin"])].copy()
            if "seed" in df.columns and "seed" in row.index:
                sub = sub[sub["seed"] == row["seed"]].copy()
            sub = sub[np.isfinite(pd.to_numeric(sub[pred_col], errors="coerce"))].copy()
            sub = sub.sort_values("timepoint")
            if sub.empty:
                ax.axis("off")
                continue

            t = pd.to_numeric(sub["timepoint"], errors="coerce").to_numpy(dtype=float)
            y_pred = pd.to_numeric(sub[pred_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(t) & np.isfinite(y_pred)
            if mask.sum() < 2:
                ax.axis("off")
                continue
            t = t[mask]
            y_pred = y_pred[mask]
            order = np.argsort(t)
            t = t[order]
            y_pred = y_pred[order]

            ax.plot(t, y_pred, color="black", linewidth=2.0, label="Integrated PySR prediction", zorder=2)

            if raw_df is not None:
                raw_sub = raw_df[(raw_df["marker"] == row["marker"]) & (raw_df["GFP_bin"] == row["GFP_bin"])].copy()
                raw_sub = raw_sub.sort_values("timepoint")
                t_obs = pd.to_numeric(raw_sub["timepoint"], errors="coerce").to_numpy(dtype=float)
                y_obs = pd.to_numeric(raw_sub["p-ERK1-2"], errors="coerce").to_numpy(dtype=float)
            else:
                metric_sub = sub[np.isin(sub["timepoint"], list(measured))].copy()
                t_obs = pd.to_numeric(metric_sub["timepoint"], errors="coerce").to_numpy(dtype=float)
                y_obs = pd.to_numeric(metric_sub["obs_pERK1_2"], errors="coerce").to_numpy(dtype=float)

            obs_mask = np.isfinite(t_obs) & np.isfinite(y_obs)
            t_obs = t_obs[obs_mask]
            y_obs = y_obs[obs_mask]
            obs_order = np.argsort(t_obs)
            t_obs = t_obs[obs_order]
            y_obs = y_obs[obs_order]
            ax.scatter(
                t_obs,
                y_obs,
                marker="x",
                color="black",
                s=34,
                linewidths=1.2,
                label="Raw measured p-ERK",
                zorder=4,
            )

            marker_label = fill(str(row["marker"]), width=18)
            seed_note = ""
            if "seed" in row.index and pd.notna(row["seed"]):
                seed_note = f" | seed {int(row['seed'])}"
            ax.set_title(
                f"R² = {row['r2_measured']:.2f}\n"
                f"{marker_label} | bin {int(row['GFP_bin'])}{seed_note}"
            )
            ax.set_xlabel("Time (min)")
            ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
            ax.grid(True, color="#dddddd", linewidth=0.7)

            y_all = np.concatenate([y_obs[np.isfinite(y_obs)], y_pred[np.isfinite(y_pred)]])
            if y_all.size:
                ymin = float(np.nanmin(y_all))
                ymax = float(np.nanmax(y_all))
                pad = 0.08 * max(1e-6, ymax - ymin)
                if math.isclose(ymin, ymax):
                    pad = max(pad, 0.1 * max(1.0, abs(ymin)))
                ax.set_ylim(ymin - pad, ymax + pad)

    first_ax = axes.flat[0]
    handles, labels = first_ax.get_legend_handles_labels()
    if handles:
        first_ax.legend(handles, labels, loc="lower right", frameon=False)

    for row_idx in range(nrows):
        axes[row_idx, 0].set_ylabel("p-ERK")
    fig.suptitle(
        "PySR per-minute test trajectories near the R2 threshold\n"
        "Three examples per threshold. Columns correspond to target R2 values 0.50, 0.55, 0.60, 0.65, and 0.70.\n"
        "Lines show the single-seed integrated PySR prediction and crosses show raw measured p-ERK.",
        y=1.14,
        fontsize=12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.trajectories).copy()
    if "model" in df.columns:
        df = df[df["model"] == args.model]
    if "dataset_mode" in df.columns:
        df = df[df["dataset_mode"] == args.dataset_mode]
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.lower() == args.phase.lower()]
    if args.seed is not None and "seed" in df.columns:
        df = df[pd.to_numeric(df["seed"], errors="coerce") == args.seed]

    raw_df = None
    if args.raw_dataset is not None and args.raw_dataset.exists():
        raw_df = pd.read_csv(args.raw_dataset).copy()

    pred_col = _choose_prediction_column(df)
    summary = _summarize_bins(df, args.measured_timepoints, pred_col)
    overrides = _parse_overrides(args.override_example)
    chosen = _select_examples(
        summary,
        args.targets,
        overrides=overrides,
        examples_per_target=max(1, int(args.examples_per_target)),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if chosen.empty:
        pd.DataFrame(
            columns=["target_r2", "example_rank", "r2_measured", "abs_delta", "marker", "GFP_bin", "seed", "n_measured"]
        ).to_csv(args.output_csv, index=False)
    else:
        chosen.to_csv(args.output_csv, index=False)

    plot_examples(
        df,
        chosen,
        pred_col,
        args.output_plot,
        args.measured_timepoints,
        raw_df=raw_df,
        target_order=args.targets,
        examples_per_target=max(1, int(args.examples_per_target)),
    )


if __name__ == "__main__":
    main()
