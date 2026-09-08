"""Figure S1 - ablation of the custom PySR loss penalties, on the OOD split.

Aggregates the 160 sharded fits (4 loss configurations x 40 overexpression
contexts, seed 42) produced by run_custom_loss_ablation_ood.py / the NEMO
stage-1+2 arrays, and draws the held-out integrated R^2 distribution per
configuration.

Supersedes the panel in draft v4, which was fitted on the in-distribution
random-bin split at max-iterations 700 / maximum complexity 20 and so reported
R^2 values on a different scale from everything else in the manuscript. This run
uses the selected configuration (1400 iterations, 30 populations of 30, maximum
complexity 26, parsimony 0.8) and the top-GFP-bin split.

Because every configuration is scored on the same 40 contexts, each ablation is
compared to the full-penalty run by a paired Wilcoxon signed-rank test rather
than by eyeballing the boxes.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402

# Order is deliberate: full penalties first, then progressively fewer, so the
# panel reads left-to-right as "what happens as you remove them".
CONDITIONS = [
    ("normal", "Both"),
    ("no_inverse_stability", "Inverse\nonly"),
    ("no_linear_stability", "Linear\nonly"),
    ("no_stability_penalties", "Neither"),
]
XLABEL = "Stability penalties retained"
METRIC = "ode_integ_r2_median"


def collect(fits_dir: Path) -> pd.DataFrame:
    """One row per (condition, context): held-out integrated R^2, seed 42."""
    rows = []
    for key, label in CONDITIONS:
        pattern = str(fits_dir / key / "*" / "reports" / "seed_42" / "metrics" /
                      "marker_integration_metrics_per_minute.csv")
        for path in sorted(glob.glob(pattern)):
            d = pd.read_csv(path)
            if "model" in d:
                d = d[d.model.astype(str).str.lower() == "pysr"]
            if "dataset_mode" in d:
                d = d[d.dataset_mode.astype(str) == "per_minute"]
            if d.empty or METRIC not in d:
                continue
            # A context that fails to integrate scores 0, not NaN: dropping it
            # would quietly grade the unstable configurations on their survivors.
            v = pd.to_numeric(d[METRIC], errors="coerce").fillna(0.0)
            rows.append({"condition": key, "condition_label": label,
                         "marker": str(d.marker.iloc[0]),
                         "metric": METRIC,
                         "metric_value": float(v.mean()),
                         "seed": 42})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/custom_loss_ablation_ood/fits"))
    ap.add_argument("--metrics-out", type=Path,
                    default=Path("data/experimental/runs/custom_loss_ablation_ood/"
                                 "pysr_custom_loss_ablation_ood_metrics.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_s1_loss_ablation.png"))
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()

    df = collect(args.fits_dir)
    if df.empty:
        raise SystemExit(f"no metrics under {args.fits_dir}")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.metrics_out, index=False)
    print(f"wrote {args.metrics_out} ({len(df)} rows)")

    wide = df.pivot_table(index="marker", columns="condition",
                          values="metric_value")
    keys = [k for k, _ in CONDITIONS if k in wide]
    labels = [lab for k, lab in CONDITIONS if k in wide]

    ref = keys[0]
    data = [wide[k].dropna().values for k in keys]
    # Under the out-of-distribution split most contexts score exactly 0, so the
    # median is 0 in every configuration and a box plot says nothing. The mean
    # and the count clearing the threshold are what separate the conditions.
    means = [float(v.mean()) for v in data]
    counts = [int((v >= args.threshold).sum()) for v in data]
    pvals = {}
    for k in keys[1:]:
        paired = wide[[ref, k]].dropna()
        pvals[k] = float(wilcoxon(paired[ref], paired[k]).pvalue)

    print(f"\n{'condition':26s} {'median':>7s} {'mean':>7s} {'n>=0.6':>7s}  "
          f"paired vs {ref}")
    for k, v in zip(keys, data):
        line = (f"{k:26s} {np.median(v):7.3f} {v.mean():7.3f} "
                f"{int((v >= args.threshold).sum()):7d}")
        if k in pvals:
            line += f"  P = {pvals[k]:.4f} (n=40)"
        print(line)

    st.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.27, 2.75),
                             gridspec_kw={"width_ratios": [1.75, 1]})
    fig.subplots_adjust(wspace=0.38)
    x = np.arange(1, len(keys) + 1)

    # ---- A: per-context distribution -----------------------------------
    ax = axes[0]
    rng = np.random.default_rng(42)
    for i, v in zip(x, data):
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=6,
                   color=st.INK, alpha=0.5, linewidths=0, zorder=3)
    ax.scatter(x, means, marker="D", s=26, facecolors=st.ACCENT,
               edgecolors="white", linewidths=0.6, zorder=5)
    for i, m in zip(x, means):
        ax.text(i + 0.22, m, f"{m:.2f}", va="center", fontsize=st.FS_TICK,
                color=st.ACCENT)
    ax.axhline(args.threshold, color=st.MUTED, ls=":", lw=0.8, zorder=1)
    ax.text(0.02, args.threshold + 0.02, f"R² = {args.threshold:g}",
            transform=ax.get_yaxis_transform(), fontsize=st.FS_TICK,
            color=st.MUTED, va="bottom")
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Held-out integrated R²")
    ax.set_xlim(0.4, len(keys) + 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=st.FS_TICK, linespacing=1.25)
    ax.set_xlabel(XLABEL)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend([plt.Line2D([], [], ls="none", marker="D", mfc=st.ACCENT,
                          mec="white", ms=4)], ["mean"], loc="lower right",
              fontsize=st.FS_TICK, handlelength=1.0, borderaxespad=0.3)

    # ---- B: contexts clearing the threshold ----------------------------
    ax = axes[1]
    bars = ax.bar(x, counts, width=0.6, color="#dfe5e5", edgecolor=st.INK,
                  linewidth=0.7, zorder=2)
    top = max(counts)
    for b, c, k in zip(bars, counts, keys):
        cx = b.get_x() + b.get_width() / 2
        ax.text(cx, c + 0.25, str(c), ha="center", fontsize=st.FS_TICK,
                color="#111111")
        if k in pvals:
            p = pvals[k]
            # One shared height, or the reader compares label positions instead
            # of bar heights.
            ax.text(cx, top + 1.6,
                    ("P = %.3f" % p) if p >= 1e-3 else "P < 0.001",
                    ha="center", fontsize=st.FS_TICK - 0.5,
                    color=st.ACCENT if p < 0.05 else st.MUTED)
    ax.set_ylim(0, top + 4.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=st.FS_TICK, linespacing=1.25)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(f"Contexts with R² ≥ {args.threshold:g}  (of 40)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.5, 1.005, "paired Wilcoxon vs “Both”", transform=ax.transAxes,
            ha="center", fontsize=st.FS_TICK - 0.5, color=st.MUTED)

    for a, letter in zip(axes, "AB"):
        st.panel_letter(a, letter, x=-0.16)

    st.save(fig, args.output)


if __name__ == "__main__":
    main()
