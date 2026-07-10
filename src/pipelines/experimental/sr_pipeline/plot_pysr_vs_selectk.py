"""
Compare PySR ODE performance against SelectK Linear Regression at matched complexity.

Inputs:
- SelectK aggregated metrics (per marker/k/split), averaged across seeds.
- PySR integration metrics (averaged across seeds).

Outputs:
- Panel A: scatter of PySR vs LinReg R2 at PySR's effective feature count.
- Panel B: delta-complexity vs headroom plot showing how much k LinReg needs
  to match PySR and the R2 headroom relative to the best LinReg R2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import matplotlib.transforms as mtransforms

# Ensure project root is on sys.path before importing project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import seaborn as sns

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

from pipelines.experimental.sr_pipeline.metrics import coefficient_of_determination

PASTEL_GREEN = "#a7c8a1"
PASTEL_RED = "#e7a29c"

MEASURED_DEFAULT = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PySR vs SelectK Linear Regression complexity/performance.")
    parser.add_argument("--selectk-metrics", type=Path, required=True, help="select_k_metrics_agg.csv (averaged across seeds)")
    parser.add_argument(
        "--pysr-metrics",
        type=Path,
        required=True,
        help="marker_integration_metrics_per_minute_all_seeds_mean.csv",
    )
    parser.add_argument("--output-panel-a", type=Path, required=True, help="Path to save panel A PNG")
    parser.add_argument("--output-panel-b", type=Path, required=True, help="Path to save panel B PNG")
    parser.add_argument("--output-panel-a-box", type=Path, required=False, help="Optional path to save panel A box/violin PNG")
    parser.add_argument("--output-panel-a-dt", type=Path, required=False, help="Optional path to save panel A (dt R2) PNG")
    parser.add_argument("--output-panel-a-dt-box", type=Path, required=False, help="Optional path to save panel A box (dt R2) PNG")
    parser.add_argument("--output-panel-b-dt", type=Path, required=False, help="Optional path to save panel B (dt R2) PNG")
    parser.add_argument("--output-panel-a-relmae", type=Path, required=False, help="Optional path to save panel A (relMAE, ODE) PNG")
    parser.add_argument("--output-panel-a-relmae-box", type=Path, required=False, help="Optional path to save panel A box (relMAE, ODE) PNG")
    parser.add_argument("--output-panel-b-relmae", type=Path, required=False, help="Optional path to save panel B (relMAE, ODE) PNG")
    parser.add_argument("--output-panel-a-relmae-dt", type=Path, required=False, help="Optional path to save panel A (relMAE, dt) PNG")
    parser.add_argument("--output-panel-a-relmae-dt-box", type=Path, required=False, help="Optional path to save panel A box (relMAE, dt) PNG")
    parser.add_argument("--output-panel-b-relmae-dt", type=Path, required=False, help="Optional path to save panel B (relMAE, dt) PNG")
    parser.add_argument(
        "--output-linreg-compare-bar",
        type=Path,
        required=False,
        help="Optional stacked bar: LinReg vs PySR R² at matched complexity (PySR<0.6 subset).",
    )
    parser.add_argument(
        "--output-linreg-compare-bar-relmae",
        type=Path,
        required=False,
        help="Optional stacked bar: LinReg vs PySR relMAE at matched complexity.",
    )
    parser.add_argument("--output-pysr-k-vs-r2", type=Path, required=False, help="Optional scatter of PySR k vs PySR R² (ODE)")
    parser.add_argument(
        "--output-trajectory-all-k",
        type=Path,
        required=False,
        help="Optional plot: PySR R² vs LinReg R² across all k per marker (lines colored by k).",
    )
    parser.add_argument(
        "--output-baseline-pysr-vs-linreg",
        type=Path,
        required=False,
        help="Optional scatter: baseline LinReg vs PySR (dt and ODE) without SelectK.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=10,
        help="Upper clip for effective PySR complexity and LinReg k.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="marker_integration_trajectories_per_minute(_all_seeds).csv containing obs/pred dt values (PySR + LinReg).",
    )
    parser.add_argument(
        "--selectk-trajectories",
        type=Path,
        default=None,
        help="Optional SelectK trajectories with per-bin preds and a 'k' column; defaults to --trajectories LinReg rows when absent.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=None,
        help="Output PNG for marker × GFP-bin R² heatmaps (PySR vs LinReg @ k(PySR)); skipped when omitted.",
    )
    parser.add_argument(
        "--heatmap-phase",
        type=str,
        default="all",
        choices=("train", "test", "all"),
        help="Phase to use for binwise heatmaps (train/test/all).",
    )
    parser.add_argument(
        "--variability-dt-output",
        type=Path,
        default=None,
        help="Optional scatter: PySR R² vs ground-truth dt variability (per marker mean std across bins).",
    )
    parser.add_argument(
        "--variability-perk-output",
        type=Path,
        default=None,
        help="Optional scatter: PySR R² vs ground-truth pERK variability (per marker mean std across bins).",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_DEFAULT,
        help="Measured timepoints for per-minute filtering in binwise R².",
    )
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


def build_panel_data(
    selectk: pd.DataFrame,
    pysr: pd.DataFrame,
    max_k: int,
    metric_col: str,
    *,
    higher_is_better: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if metric_col not in selectk.columns:
        raise KeyError(f"SelectK metrics missing column '{metric_col}'. Regenerate select_k_metrics_agg.csv.")
    if metric_col not in pysr.columns:
        raise KeyError(f"PySR metrics missing column '{metric_col}'. Regenerate marker_integration_metrics_per_minute.")
    # PySR metrics
    pysr_subset = pysr[(pysr["model"] == "PySR") & (pysr["dataset_mode"] == "per_minute")].copy()
    pysr_subset = pysr_subset.dropna(subset=[metric_col])
    pysr_subset["k_pysr"] = pysr_subset["formula"].apply(lambda f: _pysr_effective_k(f, max_k))

    # SelectK (test split only)
    sel = selectk[selectk["split"].str.lower() == "test"].copy()
    sel = sel.dropna(subset=["k", metric_col])
    sel["k"] = sel["k"].astype(int)

    # Prepare LinReg lookup: marker -> {k: r2}
    lin_lookup: Dict[str, Dict[int, float]] = {}
    for marker, grp in sel.groupby("marker"):
        lin_lookup[marker] = {int(r["k"]): float(r[metric_col]) for _, r in grp.iterrows()}

    # Panel A dataframe
    rows_a: List[Dict[str, object]] = []
    for _, r in pysr_subset.iterrows():
        m = r["marker"]
        k_pysr = int(r["k_pysr"])
        pysr_r2 = float(r[metric_col])
        target_k = k_pysr if k_pysr >= 2 else 2  # SelectK starts at k>=2
        lin_map = lin_lookup.get(m, {})
        chosen_k = None
        chosen_r2 = np.nan
        if lin_map:
            avail = sorted([k for k, v in lin_map.items() if np.isfinite(v)])
            if avail:
                # first k >= target_k else highest available
                chosen_k = next((k for k in avail if k >= target_k), avail[-1])
                chosen_r2 = lin_map.get(chosen_k, np.nan)
        rows_a.append(
            {
                "marker": m,
                "k_pysr": k_pysr,
                "k_lin_used": chosen_k,
                "r2_pysr": pysr_r2,
                "r2_lin_match": chosen_r2,
                "pysr_k_is_one": k_pysr == 1,
            }
        )
    panel_a = pd.DataFrame(rows_a)

    # Panel B dataframe
    rows_b: List[Dict[str, object]] = []
    for _, r in pysr_subset.iterrows():
        m = r["marker"]
        k_pysr = int(r["k_pysr"])
        pysr_r2 = float(r[metric_col])
        lin_map = lin_lookup.get(m, {})
        if not lin_map:
            continue
        if higher_is_better:
            k_best, r2_best = max(lin_map.items(), key=lambda kv: kv[1])
            k_match = None
            r2_match = None
            for k in sorted(lin_map.keys()):
                if lin_map[k] >= pysr_r2:
                    k_match = k
                    r2_match = lin_map[k]
                    break
            if k_match is None:
                k_match = max_k
                r2_match = lin_map.get(k_best, np.nan)
                delta_c = k_match - k_pysr
                headroom = r2_best - pysr_r2
                status = "never_matches"
            else:
                delta_c = k_match - k_pysr
                headroom = r2_best - r2_match
                status = "matches"
        else:
            k_best, r2_best = min(lin_map.items(), key=lambda kv: kv[1])
            k_match = None
            r2_match = None
            for k in sorted(lin_map.keys()):
                if lin_map[k] <= pysr_r2:
                    k_match = k
                    r2_match = lin_map[k]
                    break
            if k_match is None:
                k_match = max_k
                r2_match = lin_map.get(k_best, np.nan)
                delta_c = k_match - k_pysr
                headroom = r2_match - r2_best
                status = "never_matches"
            else:
                delta_c = k_match - k_pysr
                headroom = r2_match - r2_best
                status = "matches"
        rows_b.append(
            {
                "marker": m,
                "k_pysr": k_pysr,
                "k_match": k_match,
                "k_best": k_best,
                "r2_pysr": pysr_r2,
                "r2_match": r2_match,
                "r2_best": r2_best,
                "delta_c": delta_c,
                "headroom": headroom,
                "status": status,
            }
        )
    panel_b = pd.DataFrame(rows_b)
    return panel_a, panel_b


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
    placed = []

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


def plot_panel_a(
    panel_a: pd.DataFrame,
    output: Path,
    ylabel: str,
    *,
    xlabel: str = "LinReg R² @ k = k(PySR)",
    min_label_value: float = 0.6,
    max_floor: Optional[float] = 1.0,
    log_axes: bool = False,
) -> None:
    df = panel_a.dropna(subset=["r2_pysr"]).copy()
    plt.figure(figsize=(6.6, 6.0))
    group_k1 = df[df["pysr_k_is_one"]]
    group_rest = df[~df["pysr_k_is_one"]]
    plt.scatter(group_rest["r2_lin_match"], group_rest["r2_pysr"], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, color="black", label="k(PySR) ≥ 2")
    if not group_k1.empty:
        plt.scatter(group_k1["r2_lin_match"], group_k1["r2_pysr"], s=90, alpha=0.9, edgecolor="k", linewidth=0.4, marker="^", color="black", label="k(PySR)=1 (LinReg k=2)")
    if log_axes:
        eps = 1e-12
        x_vals = df["r2_lin_match"].to_numpy(dtype=float)
        y_vals = df["r2_pysr"].to_numpy(dtype=float)
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
        plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
        plt.xlim(lims)
        plt.ylim(lims)
    else:
        # diagonal
        lim_min = np.nanmin([df["r2_lin_match"].min(), df["r2_pysr"].min(), 0])
        lim_max = np.nanmax([df["r2_lin_match"].max(), df["r2_pysr"].max(), 0])
        if max_floor is not None:
            lim_max = max(lim_max, max_floor)
        lims = [lim_min, lim_max]
        plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
        plt.xlim(lims)
        plt.ylim(lims)
    # rugs
    x0, x1 = plt.xlim()
    y0, y1 = plt.ylim()
    if not log_axes:
        for _, r in df.iterrows():
            plt.plot([r["r2_lin_match"], r["r2_lin_match"]], [y0, y0 + 0.01], color="gray", alpha=0.4, lw=0.8)
            plt.plot([x0, x0 + 0.01], [r["r2_pysr"], r["r2_pysr"]], color="gray", alpha=0.4, lw=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # correlation
    corr = df[["r2_lin_match", "r2_pysr"]].dropna().corr().iloc[0, 1]
    plt.title(f"Matched Complexity: PySR vs LinReg (r={corr:.2f})")
    _annotate_non_overlapping(plt.gca(), df, "r2_lin_match", "r2_pysr", "marker", min_r2=min_label_value)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_panel_a_box(
    panel_a: pd.DataFrame,
    output: Path,
    ylabel: str,
    title: str,
    *,
    log_y: bool = False,
) -> None:
    df = panel_a.dropna(subset=["r2_lin_match", "r2_pysr"]).copy()
    if df.empty:
        return
    vals = [df["r2_pysr"].dropna().to_numpy(), df["r2_lin_match"].dropna().to_numpy()]
    labels = ["PySR", "LinReg @ k(PySR)"]
    plt.figure(figsize=(5.0, 5.5))
    plt.boxplot(vals, vert=True, tick_labels=labels)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.ylabel(ylabel)
    plt.title(title)
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_panel_b(panel_b: pd.DataFrame, output: Path, ylabel: str) -> None:
    df = panel_b.copy()
    plt.figure(figsize=(7.2, 5.2))
    matches = df[df["status"] == "matches"]
    never = df[df["status"] == "never_matches"]
    plt.scatter(matches["delta_c"], matches["headroom"], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, color="black", label="matches")
    if not never.empty:
        plt.scatter(never["delta_c"], never["headroom"], s=90, alpha=0.9, marker="x", color="black", label="never match")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("ΔC = k_match - k(PySR)")
    plt.ylabel(ylabel)
    plt.title("LinReg complexity gap vs headroom")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_linreg_vs_pysr_bar(
    panel_a: pd.DataFrame,
    output: Optional[Path],
    *,
    higher_is_better: bool = True,
    title: str = "LinReg (matched-k) vs PySR R²",
) -> None:
    if output is None:
        return
    df = panel_a.dropna(subset=["r2_pysr", "r2_lin_match"]).copy()
    if df.empty:
        return
    if higher_is_better:
        lin_worse = df["r2_lin_match"] <= df["r2_pysr"]
        lin_better = df["r2_lin_match"] > df["r2_pysr"]
        labels = ("LinReg ≤ PySR", "LinReg > PySR")
    else:
        lin_worse = df["r2_lin_match"] >= df["r2_pysr"]
        lin_better = df["r2_lin_match"] < df["r2_pysr"]
        labels = ("LinReg ≥ PySR", "LinReg < PySR")
    counts = {
        labels[0]: int(lin_worse.sum()),
        labels[1]: int(lin_better.sum()),
    }
    total = max(1, sum(counts.values()))

    plt.figure(figsize=(7, 2.2))
    y_pos = 0
    left = 0
    colors = {labels[0]: PASTEL_GREEN, labels[1]: PASTEL_RED}
    for label in labels:
        width = counts[label]
        plt.barh(y_pos, width, left=left, color=colors[label], edgecolor="none", height=0.45)
        left += width
    plt.xlim(0, total if total > 0 else 1)
    plt.yticks([])
    plt.xlabel("Marker count")
    plt.title(title)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_pysr_k_vs_r2(panel_a: pd.DataFrame, output: Path) -> None:
    df = panel_a.dropna(subset=["k_pysr", "r2_pysr"]).copy()
    if df.empty or output is None:
        return
    plt.figure(figsize=(6.4, 5.2))
    plt.scatter(df["k_pysr"], df["r2_pysr"], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, color="black")
    _annotate_non_overlapping(plt.gca(), df, "k_pysr", "r2_pysr", "marker", min_r2=0.6)
    plt.xlabel("PySR effective complexity k")
    plt.ylabel("PySR R² (ODE)")
    corr = df[["k_pysr", "r2_pysr"]].corr().iloc[0, 1]
    plt.title(f"PySR complexity vs R² (r={corr:.2f})")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def plot_all_k_trajectory(selectk: pd.DataFrame, panel_a: pd.DataFrame, output: Path) -> None:
    if output is None:
        return
    # Only test split
    sel = selectk[selectk["split"].str.lower() == "test"].copy()
    if sel.empty:
        return
    pysr_lookup = {r["marker"]: r["r2_pysr"] for _, r in panel_a.iterrows() if pd.notna(r["r2_pysr"])}
    if not pysr_lookup:
        return
    plt.figure(figsize=(7, 6))
    cmap = plt.get_cmap("viridis")
    k_min = sel["k"].min()
    k_max = sel["k"].max()
    for marker, grp in sel.groupby("marker"):
        if marker not in pysr_lookup:
            continue
        grp = grp.dropna(subset=["k", "ode_integ_r2_median"])
        if grp.empty:
            continue
        grp = grp.copy()
        grp["k"] = pd.to_numeric(grp["k"], errors="coerce")
        grp = grp.sort_values("k")
        ks = grp["k"].to_numpy(dtype=float)
        r2_lin = grp["ode_integ_r2_median"].to_numpy(dtype=float)
        r2_pysr = pysr_lookup[marker]
        ys = np.full_like(r2_lin, r2_pysr)
        colors = [cmap((k_val - k_min) / (k_max - k_min + 1e-9)) for k_val in ks]
        plt.plot(r2_lin, ys, "-", lw=1.2, alpha=0.5)
        plt.scatter(r2_lin, ys, s=50, edgecolor="k", linewidth=0.3, c=colors, label=marker)
    plt.xlabel("LinReg R² (ODE)")
    plt.ylabel("PySR R² (ODE)")
    plt.title("PySR vs LinReg across k (color=k)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def plot_baseline_pysr_vs_linreg(pysr_metrics: pd.DataFrame, output: Path) -> None:
    if output is None:
        return
    df = pysr_metrics.copy()
    if df.empty:
        return
    df = df[df.get("dataset_mode", "per_minute") == "per_minute"]
    pysr_rows = df[df["model"] == "PySR"]
    lin_rows = df[df["model"] == "Linear Regression"]
    if pysr_rows.empty or lin_rows.empty:
        return
    cols = ["marker", "dt_r2", "ode_integ_r2_median"]
    pysr_rows = pysr_rows[cols].rename(columns={"dt_r2": "dt_r2_pysr", "ode_integ_r2_median": "ode_r2_pysr"})
    lin_rows = lin_rows[cols].rename(columns={"dt_r2": "dt_r2_lin", "ode_integ_r2_median": "ode_r2_lin"})
    merged = pysr_rows.merge(lin_rows, on="marker", how="inner")
    if merged.empty:
        return
    plt.figure(figsize=(6.5, 6.5))
    # connect dt/ODE points per marker
    for _, r in merged.iterrows():
        plt.plot(
            [r["ode_r2_lin"], r["dt_r2_lin"]],
            [r["ode_r2_pysr"], r["dt_r2_pysr"]],
            color="gray",
            alpha=0.4,
            linewidth=1.0,
        )
    # ODE
    plt.scatter(merged["ode_r2_lin"], merged["ode_r2_pysr"], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, color="black", label="ODE R²")
    # dt
    plt.scatter(merged["dt_r2_lin"], merged["dt_r2_pysr"], s=80, alpha=0.85, edgecolor="k", linewidth=0.4, marker="s", color="black", label="dt R²")
    lims = [
        np.nanmin([merged[["ode_r2_lin", "dt_r2_lin"]].min().min(), merged[["ode_r2_pysr", "dt_r2_pysr"]].min().min(), 0.0]),
        np.nanmax([merged[["ode_r2_lin", "dt_r2_lin"]].max().max(), merged[["ode_r2_pysr", "dt_r2_pysr"]].max().max(), 1.0]),
    ]
    plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    _annotate_non_overlapping(
        plt.gca(),
        merged,
        "ode_r2_lin",
        "ode_r2_pysr",
        "marker",
        min_r2=0.6,
    )
    corr_ode = merged[["ode_r2_lin", "ode_r2_pysr"]].corr().iloc[0, 1]
    corr_dt = merged[["dt_r2_lin", "dt_r2_pysr"]].corr().iloc[0, 1]
    plt.xlabel("LinReg R²")
    plt.ylabel("PySR R²")
    plt.title(f"Baseline PySR vs LinReg (ODE r={corr_ode:.2f}, dt r={corr_dt:.2f})")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.savefig(output.with_suffix(".svg"))
    plt.close()


def _binwise_r2_from_traj(
    traj: pd.DataFrame,
    measured_timepoints: Tuple[float, ...],
    phase: str,
    *,
    obs_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Compute per (marker, GFP_bin) R² from trajectory rows."""
    required_cols = {"marker", "GFP_bin", obs_col, pred_col}
    if not required_cols.issubset(traj.columns):
        return pd.DataFrame(columns=["marker", "GFP_bin", "r2"])
    df = traj.copy()
    if phase != "all" and "phase" in df.columns:
        df = df[df["phase"].str.lower() == phase.lower()]
    if "timepoint" in df.columns and measured_timepoints:
        df = df[df["timepoint"].isin(measured_timepoints)]
    rows: List[Dict[str, object]] = []
    for (marker, gbin), grp in df.groupby(["marker", "GFP_bin"]):
        y_true = pd.to_numeric(grp.get(obs_col), errors="coerce").to_numpy()
        y_pred = pd.to_numeric(grp.get(pred_col), errors="coerce").to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            continue
        r2 = coefficient_of_determination(y_true[mask], y_pred[mask])
        if np.isfinite(r2):
            rows.append({"marker": marker, "GFP_bin": gbin, "r2": max(0.0, float(r2))})
    return pd.DataFrame(rows)


def _select_lin_traj_for_marker(df_lin: pd.DataFrame, marker: str, target_k: int) -> pd.DataFrame:
    """Return LinReg/SelectK rows for marker, preferring exact k when available (k≥2)."""
    target_k = max(2, target_k)
    df_lin_marker = df_lin[df_lin["marker"] == marker].copy()
    if df_lin_marker.empty:
        return df_lin_marker
    if "k" in df_lin_marker.columns:
        try:
            df_lin_marker["k"] = pd.to_numeric(df_lin_marker["k"], errors="coerce")
            match = df_lin_marker[np.isclose(df_lin_marker["k"], target_k)]
            if not match.empty:
                return match
        except Exception:
            pass
    return df_lin_marker


def _build_heatmap_frames(
    traj_pysr: pd.DataFrame,
    traj_lin: pd.DataFrame,
    k_lookup: Dict[str, int],
    measured_timepoints: Tuple[float, ...],
    phase: str,
    *,
    obs_col: str,
    pred_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare pivoted marker × GFP_bin R² matrices for PySR and LinReg@k(PySR)."""
    pysr_rows: List[Dict[str, object]] = []
    lin_rows: List[Dict[str, object]] = []
    for marker, k_pysr in k_lookup.items():
        df_pysr_m = traj_pysr[traj_pysr["marker"] == marker].copy()
        if df_pysr_m.empty:
            continue
        lin_df_m = _select_lin_traj_for_marker(traj_lin, marker, k_pysr)
        pysr_r2 = _binwise_r2_from_traj(df_pysr_m, measured_timepoints, phase, obs_col=obs_col, pred_col=pred_col)
        lin_r2 = _binwise_r2_from_traj(lin_df_m, measured_timepoints, phase, obs_col=obs_col, pred_col=pred_col)
        if not pysr_r2.empty:
            pysr_rows.extend(pysr_r2.to_dict(orient="records"))
        if not lin_r2.empty:
            lin_rows.extend(lin_r2.to_dict(orient="records"))
    pysr_mat = pd.DataFrame(pysr_rows)
    lin_mat = pd.DataFrame(lin_rows)
    if pysr_mat.empty and lin_mat.empty:
        return pysr_mat, lin_mat
    # Ensure consistent GFP_bin ordering
    bins_p = pysr_mat["GFP_bin"].tolist() if "GFP_bin" in pysr_mat else []
    bins_l = lin_mat["GFP_bin"].tolist() if "GFP_bin" in lin_mat else []
    all_bins = sorted(set(bins_p + bins_l))
    marker_order = (
        pysr_mat.groupby("marker")["r2"].mean().sort_values(ascending=False).index.tolist()
        if not pysr_mat.empty
        else sorted(set(lin_mat["marker"]))  # fallback
    )
    pysr_pivot = (
        pysr_mat.pivot(index="marker", columns="GFP_bin", values="r2")
        .reindex(index=marker_order, columns=all_bins)
    )
    lin_pivot = (
        lin_mat.pivot(index="marker", columns="GFP_bin", values="r2")
        .reindex(index=marker_order, columns=all_bins)
    )
    return pysr_pivot, lin_pivot


def _binwise_variability(traj_df: pd.DataFrame, value_col: str, phase: str, measured_timepoints: Tuple[float, ...]) -> pd.DataFrame:
    df = traj_df.copy()
    if phase != "all" and "phase" in df.columns:
        df = df[df["phase"].str.lower() == phase.lower()]
    df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    if measured_timepoints:
        df = df[df["timepoint"].isin(measured_timepoints)]
    agg = (
        df.groupby(["marker", "GFP_bin"], dropna=False)[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.rename(columns={"std": f"{value_col}_std"}, inplace=True)
    return agg


def plot_pysr_vs_variability(traj_df: pd.DataFrame, panel_a: pd.DataFrame, output_dt: Path, output_pe: Path, measured_timepoints: Tuple[float, ...], phase: str) -> None:
    if traj_df is None or traj_df.empty:
        return
    # Only PySR rows
    df = traj_df[traj_df["model"] == "PySR"].copy()
    if df.empty:
        return
    # Inter-bin variability: L2 distance between GFP bin trajectories and baseline bin (e.g., 0), averaged over bins and timepoints, normalized by signal magnitude
    def _inter_bin_l2(value_col: str, out_col: str) -> pd.DataFrame:
        dlocal = df.copy()
        if phase != "all" and "phase" in dlocal.columns:
            dlocal = dlocal[dlocal["phase"].str.lower() == phase.lower()]
        dlocal["timepoint"] = pd.to_numeric(dlocal["timepoint"], errors="coerce")
        if measured_timepoints:
            dlocal = dlocal[dlocal["timepoint"].isin(measured_timepoints)]
        rows = []
        for marker, grp in dlocal.groupby("marker"):
            pivot = grp.pivot_table(index="GFP_bin", columns="timepoint", values=value_col, aggfunc="mean")
            if pivot.empty or 0.0 not in pivot.index:
                continue
            baseline = pivot.loc[0.0].to_numpy(dtype=float)
            dists = []
            for gfp, row in pivot.iterrows():
                if gfp == 0.0:
                    continue
                vec = row.to_numpy(dtype=float)
                if baseline.shape != vec.shape:
                    continue
                if not np.isfinite(baseline).any() or not np.isfinite(vec).any():
                    continue
                mask = np.isfinite(baseline) & np.isfinite(vec)
                if mask.sum() == 0:
                    continue
                d = np.sqrt(np.sum((baseline[mask] - vec[mask]) ** 2))
                dists.append(d)
            if not dists:
                continue
            mean_d = float(np.nanmean(dists))
            norm = np.nanstd(pd.to_numeric(grp[value_col], errors="coerce").to_numpy())
            norm_val = mean_d / norm if norm not in (0.0, np.nan) else np.nan
            rows.append({"marker": marker, out_col: norm_val})
        return pd.DataFrame(rows)

    dt_marker = _inter_bin_l2("obs_dt", "obs_dt_inter_bin_norm")
    pe_marker = _inter_bin_l2("obs_pERK1_2", "obs_pERK1_2_inter_bin_norm")

    merged_dt = panel_a.merge(dt_marker, on="marker", how="left")
    merged_pe = panel_a.merge(pe_marker, on="marker", how="left")

    def _scatter(df_local: pd.DataFrame, value_col: str, xlabel: str, output: Path):
        if output is None:
            return
        df_plot = df_local.dropna(subset=[value_col, "r2_pysr"])
        if df_plot.empty:
            return
        plt.figure(figsize=(6, 5))
        plt.scatter(df_plot[value_col], df_plot["r2_pysr"], s=60, alpha=0.8, edgecolor="k", linewidth=0.4)
        _annotate_non_overlapping(plt.gca(), df_plot, value_col, "r2_pysr", "marker", min_r2=0.6)
        plt.xlabel(xlabel)
        plt.ylabel("PySR R² (ODE)")
        corr = df_plot[[value_col, "r2_pysr"]].corr().iloc[0, 1]
        plt.title(f"PySR R² vs inter-bin variability ({phase}) (r={corr:.2f})")
        plt.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300)
        plt.close()

    _scatter(merged_dt, "obs_dt_inter_bin_norm", "Ground-truth dt inter-bin variability (norm.)", output_dt)
    _scatter(merged_pe, "obs_pERK1_2_inter_bin_norm", "Ground-truth pERK inter-bin variability (norm.)", output_pe)


def plot_heatmaps_side_by_side(
    pysr_pivot: pd.DataFrame,
    lin_pivot: pd.DataFrame,
    output: Path,
    *,
    title: str,
    note: Optional[str] = None,
) -> None:
    if pysr_pivot.empty and lin_pivot.empty:
        return
    vmax = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(pysr_pivot) * 0.3 + 2)))
    sns.heatmap(
        pysr_pivot,
        ax=axes[0],
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        cbar=True,
        cbar_kws={"label": "R²"},
    )
    axes[0].set_title("PySR: binwise R²")
    axes[0].set_xlabel("GFP bin")
    axes[0].set_ylabel("Marker")
    sns.heatmap(
        lin_pivot,
        ax=axes[1],
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        cbar=True,
        cbar_kws={"label": "R²"},
    )
    axes[1].set_title("LinReg @ k(PySR): binwise R²")
    axes[1].set_xlabel("GFP bin")
    axes[1].set_ylabel("")
    plt.suptitle(title)
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9, color="#444444")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    selectk = pd.read_csv(args.selectk_metrics)
    pysr = pd.read_csv(args.pysr_metrics)
    if "seed" in pysr.columns:
        keys = [k for k in ("marker", "model", "dataset_mode", "feature_mode") if k in pysr.columns]
        if keys:
            numeric_cols = pysr.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "seed"]
            pysr_mean = pysr.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()
            if "formula" in pysr.columns and "formula" not in pysr_mean.columns:
                first_formulas = pysr.groupby(keys, dropna=False)["formula"].first().reset_index()
                pysr_mean = pysr_mean.merge(first_formulas, on=keys, how="left")
            pysr = pysr_mean
            print("[plot_pysr_vs_selectk] Aggregated PySR metrics across seeds.", flush=True)
    panel_a, panel_b = build_panel_data(selectk, pysr, args.max_k, metric_col="ode_integ_r2_median")
    plot_panel_a(
        panel_a,
        args.output_panel_a,
        ylabel="PySR R² (ODE)",
        xlabel="LinReg R² @ k = k(PySR)",
        min_label_value=0.6,
        max_floor=1.0,
    )
    if args.output_panel_a_box:
        plot_panel_a_box(panel_a, args.output_panel_a_box, ylabel="R² (ODE)", title="Matched Complexity R² (ODE)")
    plot_panel_b(panel_b, args.output_panel_b, ylabel="Headroom = R²_best(LR) - R²_match")
    plot_linreg_vs_pysr_bar(panel_a, args.output_linreg_compare_bar, higher_is_better=True, title="LinReg (matched-k) vs PySR R²")
    if args.output_pysr_k_vs_r2:
        plot_pysr_k_vs_r2(panel_a, args.output_pysr_k_vs_r2)
    if args.output_baseline_pysr_vs_linreg:
        plot_baseline_pysr_vs_linreg(pysr, args.output_baseline_pysr_vs_linreg)

    if args.output_panel_a_dt or args.output_panel_a_dt_box or args.output_panel_b_dt:
        panel_a_dt, panel_b_dt = build_panel_data(selectk, pysr, args.max_k, metric_col="dt_r2")
        if args.output_panel_a_dt:
            plot_panel_a(
                panel_a_dt,
                args.output_panel_a_dt,
                ylabel="PySR R² (dt)",
                xlabel="LinReg R² @ k = k(PySR)",
                min_label_value=0.6,
                max_floor=1.0,
            )
        if args.output_panel_a_dt_box:
            plot_panel_a_box(panel_a_dt, args.output_panel_a_dt_box, ylabel="R² (dt)", title="Matched Complexity R² (dt)")
        if args.output_panel_b_dt:
            plot_panel_b(panel_b_dt, args.output_panel_b_dt, ylabel="Headroom = R²_best(LR) - R²_match (dt)")
        if args.output_trajectory_all_k:
            plot_all_k_trajectory(selectk, panel_a, args.output_trajectory_all_k)

    if args.output_panel_a_relmae or args.output_panel_b_relmae or args.output_panel_a_relmae_box or args.output_linreg_compare_bar_relmae:
        panel_a_rel, panel_b_rel = build_panel_data(
            selectk,
            pysr,
            args.max_k,
            metric_col="ode_integ_rel_mae_median_bins",
            higher_is_better=False,
        )
        if args.output_panel_a_relmae:
            plot_panel_a(
                panel_a_rel,
                args.output_panel_a_relmae,
                ylabel="PySR relMAE (ODE)",
                xlabel="LinReg relMAE @ k = k(PySR)",
                min_label_value=0.0,
                max_floor=None,
                log_axes=True,
            )
        if args.output_panel_a_relmae_box:
            plot_panel_a_box(
                panel_a_rel,
                args.output_panel_a_relmae_box,
                ylabel="relMAE (ODE)",
                title="Matched Complexity relMAE (ODE)",
                log_y=True,
            )
        if args.output_panel_b_relmae:
            plot_panel_b(panel_b_rel, args.output_panel_b_relmae, ylabel="Headroom = relMAE_match - relMAE_best")
        if args.output_linreg_compare_bar_relmae:
            plot_linreg_vs_pysr_bar(
                panel_a_rel,
                args.output_linreg_compare_bar_relmae,
                higher_is_better=False,
                title="LinReg (matched-k) vs PySR relMAE",
            )

    if args.output_panel_a_relmae_dt or args.output_panel_b_relmae_dt or args.output_panel_a_relmae_dt_box:
        panel_a_rel_dt, panel_b_rel_dt = build_panel_data(
            selectk,
            pysr,
            args.max_k,
            metric_col="dt_rel_mae_mean_bins",
            higher_is_better=False,
        )
        if args.output_panel_a_relmae_dt:
            plot_panel_a(
                panel_a_rel_dt,
                args.output_panel_a_relmae_dt,
                ylabel="PySR relMAE (dt)",
                xlabel="LinReg relMAE @ k = k(PySR)",
                min_label_value=0.0,
                max_floor=None,
                log_axes=True,
            )
        if args.output_panel_a_relmae_dt_box:
            plot_panel_a_box(
                panel_a_rel_dt,
                args.output_panel_a_relmae_dt_box,
                ylabel="relMAE (dt)",
                title="Matched Complexity relMAE (dt)",
                log_y=True,
            )
        if args.output_panel_b_relmae_dt:
            plot_panel_b(panel_b_rel_dt, args.output_panel_b_relmae_dt, ylabel="Headroom = relMAE_match - relMAE_best (dt)")

    # Variability scatters (PySR only, vs ground truth variability)
    if args.trajectories is not None and (args.variability_dt_output or args.variability_perk_output):
        try:
            traj_df = pd.read_csv(args.trajectories)
        except Exception:
            traj_df = None
        plot_pysr_vs_variability(
            traj_df,
            panel_a,
            args.variability_dt_output,
            args.variability_perk_output,
            measured_timepoints=tuple(args.measured_timepoints),
            phase=args.heatmap_phase,
        )

    if args.heatmap_output and args.trajectories:
        traj_all = pd.read_csv(args.trajectories)
        if "dataset_mode" in traj_all.columns:
            traj_all["dataset_mode"] = traj_all["dataset_mode"].astype(str)
            traj_all = traj_all[traj_all["dataset_mode"].str.lower() == "per_minute"]
        traj_all["model"] = traj_all["model"].astype(str)
        traj_pysr = traj_all[traj_all["model"].str.lower() == "pysr"].copy()
        if args.selectk_trajectories:
            traj_lin = pd.read_csv(args.selectk_trajectories)
            if "model" in traj_lin.columns:
                traj_lin["model"] = traj_lin["model"].astype(str)
        else:
            traj_lin = traj_all[traj_all["model"].str.lower().str.contains("linear")].copy()

        # Build marker -> k(PySR) lookup from pysr metrics
        pysr_subset = pysr[(pysr["model"] == "PySR") & (pysr["dataset_mode"] == "per_minute")].copy()
        k_lookup: Dict[str, int] = {}
        k_orig: Dict[str, int] = {}
        for _, row in pysr_subset.iterrows():
            try:
                orig_k = _pysr_effective_k(row["formula"], args.max_k)
                k_orig[row["marker"]] = orig_k
                k_lookup[row["marker"]] = max(2, orig_k)
            except Exception:
                continue
        markers_k1 = sorted([m for m, k in k_orig.items() if k == 1])
        if k_lookup and not traj_pysr.empty and not traj_lin.empty:
            # Prefer ODE-integrated columns; fallback to integrated/pred_dt if absent
            pred_col = "pred_integrated_ode"
            if pred_col not in traj_pysr.columns or pred_col not in traj_lin.columns:
                pred_col = "pred_integrated" if "pred_integrated" in traj_pysr.columns else "pred_dt"
            obs_col = "obs_pERK1_2" if "obs_pERK1_2" in traj_pysr.columns else "obs_dt"
            pysr_pivot, lin_pivot = _build_heatmap_frames(
                traj_pysr,
                traj_lin,
                k_lookup,
                tuple(args.measured_timepoints),
                args.heatmap_phase,
                obs_col=obs_col,
                pred_col=pred_col,
            )
            plot_heatmaps_side_by_side(
                pysr_pivot,
                lin_pivot,
                args.heatmap_output,
                title="Binwise ODE integrated R² (per-minute)",
                note=f"LinReg uses k=max(2,k(PySR)); markers with k(PySR)=1 forced to k=2: {', '.join(markers_k1)}" if markers_k1 else "LinReg uses k=max(2,k(PySR))",
            )
        else:
            print("Skipping heatmap: missing trajectories or k(PySR) lookup.")


if __name__ == "__main__":
    main()
