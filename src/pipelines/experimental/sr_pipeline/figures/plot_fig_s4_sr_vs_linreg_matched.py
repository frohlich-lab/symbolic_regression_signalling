"""Figure S4 - SR against a complexity-matched linear baseline.

Same axes as Fig. 4D but the linear baseline uses only k inputs (default 4,
the median number of variables in a successful PySR rate law) rather than all
ten, so the comparison is at matched model size. Matching is on the *number*
of inputs, not their identity: SelectKBest picks the k inputs per context.

Both axes are best-of-three-seeds held-out integrated R^2 on the identical
top-GFP-bin extrapolation split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402
from display_names import gene_labels  # noqa: E402

CONTROL_PREFIXES = ("FLAG-GFP", "untransfected")


def label_points(ax, xs, ys, labels, fontsize):
    """Greedy 8-direction placement; drops a label rather than overlap one."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    placed = []
    dirs = [(1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
            (1, -1, "left", "top"), (-1, -1, "right", "top"),
            (1, 0, "left", "center"), (-1, 0, "right", "center"),
            (0, 1, "center", "bottom"), (0, -1, "center", "top")]
    ax_bb = ax.get_window_extent(renderer=rend)
    order = sorted(zip(xs, ys, labels), key=lambda t: -max(t[0], t[1]))
    for x, y, lab in order:
        # Near the right edge, try the leftward offsets first or the label runs
        # off the panel.
        cand = sorted(dirs, key=lambda d: -d[0]) if x < 0.7 else \
            sorted(dirs, key=lambda d: d[0])
        for dx, dy, ha, va in cand:
            t = ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(dx * 3.2, dy * 3.2), ha=ha, va=va,
                            fontsize=fontsize, color="#333333")
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=rend).expanded(1.04, 1.12)
            if (not any(bb.overlaps(o) for o in placed)
                    and bb.x0 > ax_bb.x0 - 2 and bb.x1 < ax_bb.x1 + 2):
                placed.append(bb)
                break
            t.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linreg-metrics", type=Path,
                    default=Path("data/experimental/runs/linreg_ood/"
                                 "select_k_metrics_agg.csv"))
    ap.add_argument("--pysr-integ-csv", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/"
                                 "metrics/integrated_r2_all40.csv"))
    ap.add_argument("--linreg-k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_s4_sr_vs_linreg_matched.png"))
    args = ap.parse_args()

    lin = pd.read_csv(args.linreg_metrics)
    lin = lin[(lin.split == "test") & (lin.k == args.linreg_k)]
    lin_r2 = lin.groupby("marker")["ode_integ_r2_median"].max()

    integ = pd.read_csv(args.pysr_integ_csv)
    col = "ode_integ_r2_median" if "ode_integ_r2_median" in integ else "ode_r2"
    sr_r2 = integ.groupby("marker")[col].max()

    common = sorted(set(lin_r2.index) & set(sr_r2.index))
    df = pd.DataFrame({
        "marker": common,
        "lin": [max(0.0, float(lin_r2[m])) for m in common],
        "sr": [max(0.0, float(sr_r2[m])) for m in common],
    })
    df["control"] = df.marker.str.startswith(CONTROL_PREFIXES)
    th = args.threshold

    n_sr = int((df.sr > df.lin).sum())
    n_lin = int((df.lin > df.sr).sum())
    n_tie = len(df) - n_sr - n_lin
    pert = df[~df.control]
    w = wilcoxon(pert.sr, pert.lin)

    st.apply()
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.set_aspect("equal", adjustable="box")
    ax.fill_between([0, 1], [0, 1], 1, facecolor=st.GREEN_FILL,
                    edgecolor="none", zorder=0)
    ax.grid(True, zorder=1)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.9, zorder=2)
    ax.axhline(th, color=st.MUTED, ls=":", lw=0.8, zorder=2)
    ax.axvline(th, color=st.MUTED, ls=":", lw=0.8, zorder=2)

    for ctrl, mk, fc in ((False, "o", st.INK), (True, "^", "#ffffff")):
        s = df[df.control == ctrl]
        ax.scatter(s.lin, s.sr, s=22, marker=mk, facecolors=fc,
                   edgecolors=st.INK, linewidths=0.7, zorder=3)

    # Sits just above the diagonal in the empty mid-left of the green region;
    # the top-left corner is taken by the marker labels.
    ax.text(0.30, 0.40, "SR outperforms\nlinear regression", fontsize=st.FS_TICK,
            color="#4a7358", transform=ax.transAxes, va="bottom", ha="left",
            rotation=45, rotation_mode="anchor", linespacing=1.2)
    ax.text(0.97, 0.03,
            f"SR higher: {n_sr}/{len(df)}\nlinear higher: {n_lin}   tied at 0: {n_tie}\n"
            f"above R² = {th:g}: {int((df.sr >= th).sum())} vs {int((df.lin >= th).sum())}\n"
            f"32 perturbations: {int((pert.sr >= th).sum())} vs "
            f"{int((pert.lin >= th).sum())}, Wilcoxon P = {w.pvalue:.3f}",
            fontsize=st.FS_TICK, color="#333333", transform=ax.transAxes,
            ha="right", va="bottom", linespacing=1.35)

    lab = df[df[["lin", "sr"]].max(axis=1) >= th]
    label_points(ax, lab.lin.values, lab.sr.values, gene_labels(lab.marker).values,
                 st.FS_TICK - 0.5)

    handles = [plt.Line2D([], [], ls="none", marker="o", mfc=st.INK,
                          mec=st.INK, ms=4),
               plt.Line2D([], [], ls="none", marker="^", mfc="#ffffff",
                          mec=st.INK, ms=4.5)]
    ax.legend(handles, ["overexpression", "control"], loc="upper center",
              ncol=2, fontsize=st.FS_TICK, handlelength=1.0,
              bbox_to_anchor=(0.5, 1.10), columnspacing=1.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(f"Integrated linear regression fit (R²), {args.linreg_k} inputs")
    ax.set_ylabel("Integrated symbolic regression fit (R²)")

    st.save(fig, args.output)
    print(f"n={len(df)} SR higher={n_sr} linear higher={n_lin} tied={n_tie}")
    print(f"above {th}: SR {int((df.sr >= th).sum())} vs linear "
          f"{int((df.lin >= th).sum())}")
    print(f"32 perturbations: SR {int((pert.sr >= th).sum())} vs linear "
          f"{int((pert.lin >= th).sum())}, Wilcoxon P={w.pvalue:.4f}")


if __name__ == "__main__":
    main()
