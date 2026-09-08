"""Fig. 4 alternative: SR success as a dose-response in dynamical complexity.

Why this replaces the quadrant boxplots
---------------------------------------
The boxplot version gates on neural-ODE R2 > 0.6, then compares complexity between the
SR-solves and SR-fails groups. Three costs follow from that design: the gate discards
half the contexts (40 -> 17), the binary split puts 0.556 and 0.608 in opposite groups
so a single membership flip moves the p-value, and any context that fails at low
complexity reads as a counterexample.

This version drops the gate and the group comparison. It asks the monotone question
instead -- does the probability that SR recovers a generalising law fall as the
dynamics get more complex -- over every context, with complexity binned into terciles
so no cut point is chosen by hand.

Under a dose-response, a low-complexity failure is not a counterexample: the low
tercile's pass rate is about one half, so failures there are predicted. EGFR and MAP2K2
sit on identical complexity coordinates with opposite outcomes, which refutes a
deterministic reading and is simply residual variance under this one.

Complexity = interacting input pairs (the manuscript's existing measure). Dependency
count and the standardised mean of the two give the same trend; the numbers are printed.

SR R2 is the frozen production configuration with the seed chosen per marker by train
parsimony -- the printed figure's own rule, which uses no held-out data to select.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

SOLVED = "#2a78d6"   # palette slot 1
FAILED = "#eb6834"   # palette slot 2
BAR = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
THR = 0.6
CONTROLS = {f"untransfected{i}" for i in (1, 2, 3, 4)} | {f"FLAG-GFP{i}" for i in (1, 2, 3, 4)}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval -- behaves at k=0 and k=n, unlike the normal approximation."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def trend(c, ok):
    """Spearman trend of pass/fail across complexity terciles."""
    t = pd.qcut(c, 3, labels=[0, 1, 2], duplicates="drop")
    codes = pd.Categorical(t).codes
    r, p = stats.spearmanr(codes, ok)
    return codes, r, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-inputs", required=True,
                    help="--dump-inputs CSV from plot_parsimony_tradeoff.py (needs py_r2).")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.panel_inputs)
    d = d.dropna(subset=["py_r2", "nn_inter", "nn_pr"]).copy()
    d["ctl"] = d.marker.isin(CONTROLS)
    d["ok"] = (d.py_r2 > THR).astype(int)

    sets = [("All 40 contexts", d), ("32 perturbations", d[~d.ctl])]

    print("trend across complexity terciles:")
    for lab, sub in sets:
        for nm, c in (("pairs", sub.nn_inter), ("deps", sub.nn_pr),
                      ("z(pairs)+z(deps)", (stats.zscore(sub.nn_inter) + stats.zscore(sub.nn_pr)) / 2)):
            codes, r, p = trend(c, sub.ok)
            top = sub.ok[codes == 2]
            print(f"  {lab:<17} {nm:<17} rho={r:+.3f} p={p:.4f}  top tercile {int(top.sum())}/{len(top)}")

    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
        "axes.edgecolor": INK2, "axes.linewidth": 0.9,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.8),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- A: every context, complexity against the outcome it is meant to predict ----
    axA.axhline(THR, color=INK2, lw=0.9, ls="--", zorder=2)
    axA.annotate("SR success threshold", xy=(0.99, THR), xycoords=("axes fraction", "data"),
                 xytext=(0, 5), textcoords="offset points", ha="right", va="bottom",
                 fontsize=9, color=INK2)
    for ctl, marker in ((False, "o"), (True, "^")):
        s = d[d.ctl == ctl]
        axA.scatter(s.nn_inter, s.py_r2.clip(lower=0), s=64, marker=marker,
                    c=[SOLVED if v else FAILED for v in s.ok],
                    alpha=0.85, edgecolors="white", linewidths=1.1, zorder=3)
    # tercile boundaries -- the binning used in panel B, shown so it is not hidden
    for q in d.nn_inter.quantile([1 / 3, 2 / 3]):
        axA.axvline(q, color=GRID, lw=1.0, zorder=1)
    axA.set_xlabel("Interacting input pairs (neural ODE)")
    axA.set_ylabel("SR held-out R²")
    axA.set_title("Every context", loc="left", pad=8)
    axA.set_ylim(-0.04, 1.04)
    axA.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.55)
    axA.set_axisbelow(True)
    axA.legend(handles=[Line2D([], [], marker="o", ls="none", markersize=8, markerfacecolor=SOLVED,
                               markeredgecolor="white", label="SR recovers a law"),
                        Line2D([], [], marker="o", ls="none", markersize=8, markerfacecolor=FAILED,
                               markeredgecolor="white", label="SR fails"),
                        Line2D([], [], marker="^", ls="none", markersize=8, markerfacecolor="#b9b8b2",
                               markeredgecolor="white", label="control context")],
               loc="upper right", frameon=False, fontsize=9, handletextpad=0.5)

    # ---- B: the dose-response itself ----
    w = 0.36
    for i, (lab, sub) in enumerate(sets):
        codes, r, p = trend(sub.nn_inter, sub.ok)
        xs = np.arange(3) + (i - 0.5) * w
        rate, lo, hi, ns = [], [], [], []
        for t in range(3):
            y = sub.ok[codes == t]
            k, n = int(y.sum()), len(y)
            a, b = wilson(k, n)
            rate.append(k / n if n else 0); lo.append(a); hi.append(b); ns.append((k, n))
        col = BAR if i == 0 else "#9dc3ef"
        axB.bar(xs, rate, width=w, color=col, edgecolor="white", linewidth=1.4, zorder=3,
                label=f"{lab}  (trend p={p:.3f})")
        axB.errorbar(xs, rate, yerr=[np.array(rate) - lo, np.array(hi) - np.array(rate)],
                     fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
        for x, rt, (k, n) in zip(xs, rate, ns):
            axB.annotate(f"{k}/{n}", (x, rt), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=5)
    axB.set_xticks(range(3))
    axB.set_xticklabels(["low", "middle", "high"])
    axB.set_xlabel("Dynamical complexity (tercile)")
    axB.set_ylabel("Fraction of contexts where SR succeeds")
    axB.set_title("Success falls as the dynamics get more complex", loc="left", pad=8)
    axB.set_ylim(0, 1.0)
    axB.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.55)
    axB.set_axisbelow(True)
    axB.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout(w_pad=2.2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
