"""Figure S3 - sparse Neural ODE regulariser and hyperparameter selection.

(A) Held-out R^2 per overexpression context for the three sparse-Jacobian
    variants (L1, L21, PathReg), best of three seeds.
(B) L21 lambda sweep: mean in-distribution validation R^2 and mean held-out
    R^2, with the elbow rule that retains lambda = 3.
(C) Whether in-distribution fit quality tracks held-out performance across a
    hyperparameter grid, using the 72-arm PySR config sweep (the only grid in
    this study scored on both splits).

All inputs are the frozen run artefacts under
data/experimental/runs/{sparse_neural_ode,pysr_config_sweep}.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402

VARIANTS = [("l1", "L1"), ("l21_lam3", "L21"), ("pathreg", "PathReg")]


def heldout_by_marker(root: Path, variant: str) -> pd.Series:
    """Best-of-three-seeds held-out integrated R^2 per marker."""
    frames = [pd.read_csv(f) for f in
              sorted(glob.glob(str(root / variant / "seed_*" /
                                   "neural_ode_diffrax_metrics_agg.csv")))]
    if not frames:
        raise SystemExit(f"no metrics under {root / variant}")
    d = pd.concat(frames)
    d = d[d.split == "test"]
    return d.groupby("marker")["ode_integ_r2_median"].max()


def panel_a(ax, root: Path) -> pd.DataFrame:
    cols = {label: heldout_by_marker(root, key) for key, label in VARIANTS}
    df = pd.DataFrame(cols)
    labels = list(df.columns)
    data = [df[c].values for c in labels]

    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#111111", lw=1.1),
                    boxprops=dict(facecolor="#f2f4f4", edgecolor="#555555", lw=0.7),
                    whiskerprops=dict(color="#555555", lw=0.7),
                    capprops=dict(color="#555555", lw=0.7))
    del bp
    rng = np.random.default_rng(42)
    for i, v in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=4.5,
                   color=st.INK, alpha=0.55, linewidths=0, zorder=3)
    ax.axhline(0.6, color=st.ACCENT, ls=":", lw=0.8, zorder=1)
    ax.text(0.02, 0.615, "R² = 0.6", transform=ax.get_yaxis_transform(),
            fontsize=st.FS_TICK, color=st.ACCENT, va="bottom")

    # Medians go above each box; below the axis they collide with the tick labels.
    for i, c in enumerate(labels, start=1):
        ax.text(i, 1.11, f"{df[c].median():.2f}", ha="center",
                fontsize=st.FS_TICK, color="#555555")
    ax.text(0.5, 1.11, "median", ha="right", fontsize=st.FS_TICK,
            color="#555555")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.05, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Held-out integrated R²")
    ax.set_xlabel("Sparse-Jacobian penalty")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    return df


def panel_b(ax, summary: Path) -> None:
    d = pd.read_csv(summary).sort_values("lambda_jac")
    x = d.lambda_jac.values
    ax.plot(x, d.mean_val_r2, "-o", ms=3, color=st.INK, lw=1.0,
            label="in-distribution validation")
    ax.plot(x, d.mean_heldout_r2, "-s", ms=3, color=st.MUTED, lw=1.0,
            label="held-out (top-GFP bins)")

    # Elbow rule: largest lambda whose mean validation R2 is within 0.05 of the best.
    best = d.mean_val_r2.max()
    ax.axhspan(best - 0.05, best, color=st.GREEN_FILL, alpha=0.55, zorder=0,
               lw=0)
    ax.text(30, best - 0.046, "within 0.05\nof best", fontsize=st.FS_TICK,
            color="#4a7358", va="bottom", ha="right", linespacing=1.15)
    keep = d[d.mean_val_r2 >= best - 0.05].lambda_jac.max()
    yk = float(d.loc[d.lambda_jac == keep, "mean_val_r2"].iloc[0])
    ax.scatter([keep], [yk], s=42, facecolors="none", edgecolors=st.ACCENT,
               lw=1.2, zorder=4)
    ax.annotate(f"retained λ = {keep:g}", xy=(keep, yk),
                xytext=(keep * 0.62, yk + 0.13), fontsize=st.FS_TICK,
                color=st.ACCENT, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=st.ACCENT))

    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.set_ylim(0.15, 0.95)
    ax.set_xlabel("L21 Jacobian penalty λ")
    ax.set_ylabel("Mean R² across contexts")
    ax.legend(loc="lower left", handlelength=1.4, borderaxespad=0.2)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def panel_c(ax, table_s10: Path) -> tuple[float, float, int]:
    d = pd.read_csv(table_s10)
    rho, p = spearmanr(d.median_train_r2, d.median_heldout_r2)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.scatter(d.median_train_r2, d.median_heldout_r2, s=9, color=st.INK,
               alpha=0.7, linewidths=0, zorder=3)
    sel = d[d.train_rank == 1]
    ax.scatter(sel.median_train_r2, sel.median_heldout_r2, s=34,
               facecolors="none", edgecolors=st.ACCENT, lw=1.2, zorder=4)
    ax.annotate("selected\nconfiguration",
                xy=(float(sel.median_train_r2.iloc[0]),
                    float(sel.median_heldout_r2.iloc[0])),
                xytext=(0.985, 0.66), fontsize=st.FS_TICK, color=st.ACCENT,
                ha="right", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=st.ACCENT))
    ax.axhline(0.6, color=st.MUTED, ls=":", lw=0.8, zorder=1)
    ax.text(0.46, 0.61, "R² = 0.6", ha="left", va="bottom",
            fontsize=st.FS_TICK, color=st.MUTED)
    ax.text(0.03, 0.05, f"Spearman ρ = {rho:.2f}\n"
                        f"P = {p:.0e}, n = {len(d)} configurations",
            transform=ax.transAxes, va="bottom", fontsize=st.FS_TICK,
            color="#333333", linespacing=1.15)
    ax.set_xlabel("Median R² on training GFP bins")
    ax.set_ylabel("Median held-out R²")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.03, 0.78)
    return float(rho), float(p), len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nde-root", type=Path,
                    default=Path("data/experimental/runs/sparse_neural_ode"))
    ap.add_argument("--table-s10", type=Path,
                    default=Path("data/experimental/runs/pysr_config_sweep/"
                                 "metrics/table_s10_config_sweep.csv"))
    ap.add_argument("--with-regulariser-panel", action="store_true",
                    help="Restore the L1-vs-L21 boxplot as panel A. Off by default: the "
                         "comparison is no longer in the manuscript.")
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_s3_neural_ode_selection.png"))
    args = ap.parse_args()

    st.apply()
    # The regulariser comparison was dropped from the manuscript: its L1 arm has no
    # surviving per-fit metrics, so it cannot be put on the same validation-selected
    # seed rule as everything else. Keep the panel behind a flag rather than delete it.
    n_panels = 3 if args.with_regulariser_panel else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6.27 * n_panels / 3, 2.0))
    fig.subplots_adjust(wspace=0.42)

    df = None
    k = 0
    if args.with_regulariser_panel:
        df = panel_a(axes[k], args.nde_root); k += 1
    panel_b(axes[k], args.nde_root / "lambda_sweep_summary.csv"); k += 1
    rho, p, n = panel_c(axes[k], args.table_s10)

    for ax, letter in zip(axes, "ABC"):
        st.panel_letter(ax, letter)

    st.save(fig, args.output)

    if df is not None:
        print("\nRegulariser panel (best of 3 seeds, n = %d contexts):" % len(df))
        for c in df.columns:
            print(f"  {c:8s} median {df[c].median():.3f}  mean {df[c].mean():.3f}  "
                  f"n>=0.6 {(df[c] >= 0.6).sum()}")
    print(f"Config-grid panel: Spearman rho = {rho:.3f}, P = {p:.3g}, n = {n}")


if __name__ == "__main__":
    main()
