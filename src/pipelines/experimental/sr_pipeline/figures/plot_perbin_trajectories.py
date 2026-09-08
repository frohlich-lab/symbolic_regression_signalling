"""Per-GFP-bin trajectory panels: integrated prediction against the measured points.

The fan overlays every bin in one axes, which shows the dose gradient but hides how well any
single trajectory is reproduced. This draws a small square panel per selected bin -- the
ODE-integrated prediction as a coloured line through the measured timepoints, the observations as
black crosses, and the per-bin R2 in the corner -- so fit quality is legible bin by bin.

Bins are sampled evenly across the available range and coloured by dose (viridis, dark = lowest
GFP), so the panels also read as a dose series. R2 is computed on the measured timepoints only,
which is where the observations exist.

Usage:
  python plot_perbin_trajectories.py --traj <marker_integration_trajectories_per_minute.csv> \
      --output fig.png [--n-bins 4] [--measured 0 5 10 15 30 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True,
                    help="marker_integration_trajectories_per_minute.csv for one marker/seed.")
    ap.add_argument("--output", required=True, help="PNG path; a PDF is written alongside.")
    ap.add_argument("--n-bins", type=int, default=4,
                    help="How many GFP bins to show (arranged as a square-ish grid).")
    ap.add_argument("--measured", nargs="+", type=float,
                    default=[0.0, 5.0, 10.0, 15.0, 30.0, 60.0],
                    help="Timepoints at which observations exist.")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    d = pd.read_csv(args.traj)
    d = d[d["pred_integrated_ode"].notna()]
    if d.empty:
        raise SystemExit("no integrated predictions in this trajectory file")

    # measured timepoints only: that is where the crosses live and where R2 is meaningful
    d = d[d.timepoint.isin(args.measured)]
    bins = np.array(sorted(d.GFP_bin.unique()))
    if len(bins) == 0:
        raise SystemExit("no GFP bins after filtering to measured timepoints")

    # evenly spaced across the dose range, always including the lowest and highest
    idx = np.unique(np.linspace(0, len(bins) - 1, args.n_bins).round().astype(int))
    chosen = bins[idx]

    ncol = int(np.ceil(np.sqrt(len(chosen))))
    nrow = int(np.ceil(len(chosen) / ncol))
    FS = 25.3      # 23.0 + 10%
    mpl.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333", "axes.linewidth": 1.2,
    })

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(bins.min()), vmax=float(bins.max()))

    lo = min(d.obs_pERK1_2.min(), d.pred_integrated_ode.min())
    hi = max(d.obs_pERK1_2.max(), d.pred_integrated_ode.max())
    pad = 0.08 * (hi - lo if hi > lo else 1.0)

    for k, b in enumerate(chosen):
        ax = axes[k // ncol][k % ncol]
        h = d[d.GFP_bin == b].sort_values("timepoint")
        obs, pred = h.obs_pERK1_2.to_numpy(), h.pred_integrated_ode.to_numpy()
        ax.plot(h.timepoint, pred, "-o", color=cmap(norm(b)), lw=2.8, ms=5.5, zorder=3)
        ax.plot(h.timepoint, obs, "x", color="#111111", ms=9.0, mew=2.3, zorder=4)

        ss = float(((obs - obs.mean()) ** 2).sum())
        r2 = 1.0 - float(((obs - pred) ** 2).sum()) / ss if ss > 0 else np.nan
        ax.text(0.97, 0.95, f"R²={r2:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=FS - 0.5)

        ax.set_box_aspect(3 / 5)                  # 5:3 plotting area (w:h)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, color="#e0dfdb", lw=0.9, ls=(0, (1, 2)))
        ax.set_axisbelow(True)
        if k % ncol == 0:
            ax.set_ylabel("p-ERK (a.u.)")
        if k // ncol == nrow - 1:
            ax.set_xlabel("time (min)")

    for k in range(len(chosen), nrow * ncol):     # blank any unused cell
        axes[k // ncol][k % ncol].axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=FS + 1, y=0.99)
    fig.tight_layout()
    # sharex/sharey suppress the inner tick labels, so the cells can sit close; must follow
    # tight_layout, which overwrites wspace/hspace.
    fig.subplots_adjust(wspace=0.06, hspace=0.10)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  {out.name}: bins {list(chosen)} of {len(bins)}")


if __name__ == "__main__":
    main()
