"""Compare PySR against the neural ODE baseline (full-feature MLP, no K sweep)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import matplotlib.transforms as mtransforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sympy as sp

# Global styling for publication-style figures
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


def _annotate_non_overlapping(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    *,
    min_r2: float = 0.6,
    fontsize: int = 12,
    point_keepout_px: float = 10.0,
    pad_px: float = 2.0,
    max_labels: int | None = None,
) -> None:
    """
    Directional offset placement with a penalty heuristic (no arrows).
    """
    d = df.copy()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]
    d = d[(d[[x_col, y_col]].max(axis=1) >= min_r2)]
    if d.empty:
        return
    d = d.sort_values(by=[y_col, x_col], ascending=False)
    if max_labels is not None:
        d = d.head(max_labels)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    placed: List[mtransforms.Bbox] = []

    dirs = [
        (1, 1, "left", "bottom"),
        (0, 1, "center", "bottom"),
        (-1, 1, "right", "bottom"),
        (1, 0, "left", "center"),
        (-1, 0, "right", "center"),
        (1, -1, "left", "top"),
        (0, -1, "center", "top"),
        (-1, -1, "right", "top"),
    ]
    radii = [8, 12, 16, 22, 30, 40, 55, 75]

    def expanded_bbox(t):
        b = t.get_window_extent(renderer=renderer)
        return mtransforms.Bbox.from_extents(b.x0 - pad_px, b.y0 - pad_px, b.x1 + pad_px, b.y1 + pad_px)

    def overlaps_any(b):
        return any(b.overlaps(bb) for bb in placed)

    def hits_own_point(b, x, y):
        px, py = ax.transData.transform((x, y))
        keep = mtransforms.Bbox.from_extents(px - point_keepout_px, py - point_keepout_px, px + point_keepout_px, py + point_keepout_px)
        return b.overlaps(keep)

    def outside_axes(b):
        return (b.x0 < ax_bbox.x0) or (b.x1 > ax_bbox.x1) or (b.y0 < ax_bbox.y0) or (b.y1 > ax_bbox.y1)

    for _, r in d.iterrows():
        x = float(r[x_col])
        y = float(r[y_col])
        label = str(r[label_col])

        best = None
        best_bbox = None
        best_score = None

        for rad in radii:
            for sx, sy, ha, va in dirs:
                dx, dy = sx * rad, sy * rad
                t = ax.annotate(
                    label,
                    xy=(x, y),
                    xycoords="data",
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=fontsize,
                    ha=ha,
                    va=va,
                    arrowprops=None,
                )
                fig.canvas.draw()
                b = expanded_bbox(t)

                if overlaps_any(b) or hits_own_point(b, x, y):
                    t.remove()
                    continue

                penalty = 0.0
                if outside_axes(b):
                    penalty += 1000.0
                penalty += (abs(dx) + abs(dy)) * 0.1
                penalty += max(0.0, ax_bbox.x0 - b.x0) + max(0.0, b.x1 - ax_bbox.x1)
                penalty += max(0.0, ax_bbox.y0 - b.y0) + max(0.0, b.y1 - ax_bbox.y1)

                if best_score is None or penalty < best_score:
                    if best is not None:
                        best.remove()
                    best = t
                    best_bbox = b
                    best_score = penalty
                else:
                    t.remove()
            if best is not None:
                break

        if best is None:
            best = ax.annotate(
                label,
                xy=(x, y),
                xycoords="data",
                xytext=(-90, -90),
                textcoords="offset points",
                fontsize=fontsize,
                ha="right",
                va="top",
                arrowprops=None,
            )
            fig.canvas.draw()
            best_bbox = expanded_bbox(best)

        placed.append(best_bbox)

MEASURED_DEFAULT = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PySR vs neural ODE baseline (per-minute).")
    parser.add_argument("--baseline-metrics", type=Path, required=True, help="neural_ode_metrics_agg.csv")
    parser.add_argument(
        "--pysr-metrics",
        type=Path,
        required=True,
        help="marker_integration_metrics_per_minute_all_seeds_mean.csv",
    )
    parser.add_argument("--output-panel-ode", type=Path, required=True, help="Path to save PySR vs neural ODE (ODE R²) scatter.")
    parser.add_argument("--output-panel-dt", type=Path, required=True, help="Path to save PySR vs neural ODE (dt R²) scatter.")
    parser.add_argument("--output-panel-ode-relmae", type=Path, required=False, help="Optional relMAE scatter (ODE integration).")
    parser.add_argument("--output-panel-dt-relmae", type=Path, required=False, help="Optional relMAE scatter (dt).")
    parser.add_argument(
        "--output-baseline",
        type=Path,
        required=False,
        help="Optional combined scatter showing dt/ODE points per marker.",
    )
    parser.add_argument(
        "--output-pysr-k-vs-r2",
        type=Path,
        required=False,
        help="Optional scatter of PySR effective k vs PySR ODE R².",
    )
    parser.add_argument(
        "--output-quadrant-bar",
        type=Path,
        required=False,
        help="Optional bar chart of marker counts per quadrant (PySR vs Neural ODE, R² threshold 0.6).",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_DEFAULT,
        help="Measured timepoints for per-minute filtering in binwise R² (kept for compatibility).",
    )
    parser.add_argument("--max-k", type=int, default=10, help="Upper clip for effective PySR complexity.")
    return parser.parse_args()


def _pysr_effective_k(formula: str, max_k: int) -> int:
    try:
        expr = sp.sympify(formula)
        syms = {s for s in expr.free_symbols if not str(s).startswith("__")}
        k = max(1, len(syms))
    except Exception:
        k = 1
    k = int(round(k))
    k = min(max_k, max(1, k))
    return k


def _load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["split"] = df["split"].str.lower()
    df = df[df["split"] == "test"]
    return df


def _load_pysr(path: Path, max_k: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df.get("model") == "PySR") & (df.get("dataset_mode") == "per_minute")].copy()
    if "formula" in df.columns:
        df["k_pysr"] = df["formula"].apply(lambda f: _pysr_effective_k(f, max_k))
    return df


def build_panel(
    baseline: pd.DataFrame,
    pysr: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "marker",
        "dt_r2",
        "ode_integ_r2_median",
        "dt_rel_mae_mean_bins",
        "ode_integ_rel_mae_median_bins",
        "k",
    ]
    base_cols = [c for c in base_cols if c in baseline.columns]
    base = baseline[base_cols].rename(
        columns={
            "dt_r2": "dt_r2_baseline",
            "ode_integ_r2_median": "ode_r2_baseline",
            "dt_rel_mae_mean_bins": "dt_rel_mae_baseline",
            "ode_integ_rel_mae_median_bins": "ode_rel_mae_baseline",
            "k": "k_baseline",
        }
    )
    py_cols = [
        "marker",
        "dt_r2",
        "ode_integ_r2_median",
        "dt_rel_mae_mean_bins",
        "ode_integ_rel_mae_median_bins",
        "k_pysr",
    ]
    py_cols = [c for c in py_cols if c in pysr.columns]
    pysr_df = pysr[py_cols].rename(
        columns={
            "dt_r2": "dt_r2_pysr",
            "ode_integ_r2_median": "ode_r2_pysr",
            "dt_rel_mae_mean_bins": "dt_rel_mae_pysr",
            "ode_integ_rel_mae_median_bins": "ode_rel_mae_pysr",
        }
    )
    panel = base.merge(pysr_df, on="marker", how="inner")
    return panel


def plot_panel(
    panel: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    output: Path,
    *,
    min_label_value: float = 0.6,
    max_floor: Optional[float] = 1.0,
    log_axes: bool = False,
) -> None:
    df = panel.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.8, 6.2))
    plt.scatter(df[x_col], df[y_col], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, color="black")
    if log_axes:
        eps = 1e-12
        x_vals = df[x_col].to_numpy(dtype=float)
        y_vals = df[y_col].to_numpy(dtype=float)
        x_pos = x_vals[x_vals > 0]
        y_pos = y_vals[y_vals > 0]
        if x_pos.size == 0 or y_pos.size == 0:
            return
        lim_min = float(np.nanmin([x_pos.min(), y_pos.min()]))
        lim_min = max(lim_min, eps)
        lim_max = float(np.nanmax([x_vals.max(), y_vals.max()]))
        if max_floor is not None:
            lim_max = max(lim_max, max_floor)
        lims = [lim_min, lim_max]
        plt.xscale("log")
        plt.yscale("log")
        plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0)
        plt.xlim(lims)
        plt.ylim(lims)
    else:
        lim_min = np.nanmin([df[x_col].min(), df[y_col].min(), 0.0])
        lim_max = np.nanmax([df[x_col].max(), df[y_col].max(), 0.0])
        if max_floor is not None:
            lim_max = max(lim_max, max_floor)
        lims = [lim_min, lim_max]
        plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0)
        plt.xlim(lims)
        plt.ylim(lims)
    _annotate_non_overlapping(plt.gca(), df, x_col, y_col, "marker", min_r2=min_label_value)
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel} (r={corr:.2f})")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_baseline_combo(panel: pd.DataFrame, output: Optional[Path]) -> None:
    if output is None:
        return
    df = panel.dropna(subset=["ode_r2_baseline", "ode_r2_pysr", "dt_r2_baseline", "dt_r2_pysr"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(7.0, 6.6))
    for _, r in df.iterrows():
        plt.plot(
            [r["ode_r2_baseline"], r["dt_r2_baseline"]],
            [r["ode_r2_pysr"], r["dt_r2_pysr"]],
            color="gray",
            alpha=0.35,
            linewidth=1.0,
        )
    plt.scatter(df["ode_r2_baseline"], df["ode_r2_pysr"], s=75, alpha=0.85, edgecolor="k", linewidth=0.4, color="black", label="ODE R²")
    plt.scatter(
        df["dt_r2_baseline"],
        df["dt_r2_pysr"],
        s=75,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.4,
        marker="s",
        color="black",
        label="dt R²",
    )
    lims = [
        np.nanmin([df[["ode_r2_baseline", "dt_r2_baseline"]].min().min(), df[["ode_r2_pysr", "dt_r2_pysr"]].min().min(), 0.0]),
        np.nanmax([df[["ode_r2_baseline", "dt_r2_baseline"]].max().max(), df[["ode_r2_pysr", "dt_r2_pysr"]].max().max(), 1.0]),
    ]
    plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1.0)
    plt.xlim(lims)
    plt.ylim(lims)
    _annotate_non_overlapping(
        plt.gca(),
        df,
        "ode_r2_baseline",
        "ode_r2_pysr",
        "marker",
        min_r2=0.6,
    )
    corr_ode = df[["ode_r2_baseline", "ode_r2_pysr"]].corr().iloc[0, 1]
    corr_dt = df[["dt_r2_baseline", "dt_r2_pysr"]].corr().iloc[0, 1]
    plt.xlabel("Neural ODE R²")
    plt.ylabel("PySR R²")
    plt.title(f"PySR vs Neural ODE (ODE r={corr_ode:.2f}, dt r={corr_dt:.2f})")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_pysr_k(panel: pd.DataFrame, output: Optional[Path]) -> None:
    if output is None:
        return
    df = panel.dropna(subset=["k_pysr", "ode_r2_pysr"]).copy()
    if df.empty:
        return
    plt.figure(figsize=(6.4, 5.0))
    plt.scatter(df["k_pysr"], df["ode_r2_pysr"], s=70, alpha=0.85, edgecolor="k", linewidth=0.4, color="black")
    _annotate_non_overlapping(plt.gca(), df, "k_pysr", "ode_r2_pysr", "marker", min_r2=0.6)
    corr = df[["k_pysr", "ode_r2_pysr"]].corr().iloc[0, 1]
    plt.xlabel("PySR effective complexity k")
    plt.ylabel("PySR R² (ODE)")
    plt.title(f"PySR complexity vs R² (r={corr:.2f})")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_quadrant_counts(panel: pd.DataFrame, output: Optional[Path], threshold: float = 0.6) -> None:
    if output is None:
        return
    df = panel.dropna(subset=["ode_r2_baseline", "ode_r2_pysr"]).copy()
    if df.empty:
        return
    # Focus only on markers where PySR underperforms (< threshold) and show Neural split as a stacked bar.
    psr_low = df["ode_r2_pysr"] < threshold
    neural_high = (df["ode_r2_baseline"] >= threshold) & psr_low
    neural_low = (df["ode_r2_baseline"] < threshold) & psr_low
    counts = {
        "Neural≥0.6": int(neural_high.sum()),
        "Neural<0.6": int(neural_low.sum()),
    }
    total = max(1, sum(counts.values()))

    plt.figure(figsize=(7, 2.2))
    y_pos = 0
    left = 0
    # Pastel blue/yellow to match the quadrant shading (slightly darker for contrast).
    colors = {"Neural≥0.6": "#f4e19a", "Neural<0.6": "#b8d7f0"}
    for label in ("Neural<0.6", "Neural≥0.6"):
        width = counts[label]
        plt.barh(y_pos, width, left=left, color=colors[label], edgecolor="none", height=0.45)
        left += width
    plt.xlim(0, total if total > 0 else 1)
    plt.yticks([])
    plt.xlabel("Marker count (PySR R² < {:.1f})".format(threshold))
    plt.title(f"PySR< {threshold:.1f}: Neural ODE R² split")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def main() -> None:
    args = parse_args()
    baseline = _load_baseline(args.baseline_metrics)
    pysr = _load_pysr(args.pysr_metrics, args.max_k)
    if baseline.empty or pysr.empty:
        print("No baseline or PySR rows available; nothing to plot.")
        return

    panel = build_panel(baseline, pysr)
    if panel.empty:
        print("No overlapping markers between baseline and PySR.")
        return

    plot_panel(
        panel,
        x_col="ode_r2_baseline",
        y_col="ode_r2_pysr",
        xlabel="Neural ODE R² (ODE)",
        ylabel="PySR R² (ODE)",
        output=args.output_panel_ode,
    )
    plot_panel(
        panel,
        x_col="dt_r2_baseline",
        y_col="dt_r2_pysr",
        xlabel="Neural ODE R² (dt)",
        ylabel="PySR R² (dt)",
        output=args.output_panel_dt,
    )
    if args.output_panel_ode_relmae:
        if not {"ode_rel_mae_baseline", "ode_rel_mae_pysr"}.issubset(panel.columns):
            raise KeyError(
                "Missing ODE relMAE columns in baseline or PySR metrics. "
                "Regenerate neural_ode_metrics_agg.csv and marker_integration metrics."
            )
        plot_panel(
            panel,
            x_col="ode_rel_mae_baseline",
            y_col="ode_rel_mae_pysr",
            xlabel="Neural ODE relMAE (ODE)",
            ylabel="PySR relMAE (ODE)",
            output=args.output_panel_ode_relmae,
            min_label_value=0.0,
            max_floor=None,
            log_axes=True,
        )
    if args.output_panel_dt_relmae:
        if not {"dt_rel_mae_baseline", "dt_rel_mae_pysr"}.issubset(panel.columns):
            raise KeyError(
                "Missing dt relMAE columns in baseline or PySR metrics. "
                "Regenerate neural_ode_metrics_agg.csv and marker_integration metrics."
            )
        plot_panel(
            panel,
            x_col="dt_rel_mae_baseline",
            y_col="dt_rel_mae_pysr",
            xlabel="Neural ODE relMAE (dt)",
            ylabel="PySR relMAE (dt)",
            output=args.output_panel_dt_relmae,
            min_label_value=0.0,
            max_floor=None,
            log_axes=True,
        )
    plot_baseline_combo(panel, args.output_baseline)
    plot_pysr_k(panel, args.output_pysr_k_vs_r2)
    plot_quadrant_counts(panel, args.output_quadrant_bar, threshold=0.6)


if __name__ == "__main__":
    main()
