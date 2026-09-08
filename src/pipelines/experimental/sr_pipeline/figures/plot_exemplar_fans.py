"""Fig. 5 centre panel: the measured p-ERK fan beside the integrated recovered law.

One curve per GFP bin, coloured by overexpression level. The left panel is what the model
was trained against; the right is the trajectory obtained by integrating the discovered ODE
with the other readouts supplied as exogenous inputs. Read them as a pair: the claim is that
the recovered law reproduces the ordered fan -- same peak timing, same decay, no crossings --
not merely that it scores a high R2, which a flat monotonic curve can also achieve.

Style follows sweeps/make_fan_plots.py exactly (square boxes, compressed layout, shared y,
no colourbar, 16 pt base), so the paper panel and the sweep diagnostic are visually the same
plot. Two deliberate differences:

  * Data source is the per-fit trajectory table that the per-bin panels also read, not a
    sweep shard. The sweep directory those earlier fans came from no longer exists, and
    reading one table for both panels of Fig. 5 keeps them describing the same fit.
  * The left panel draws `fit_pERK1_2`, the parametric fit, and is labelled as such. The
    diagnostic drew `obs_pERK1_2`; in this table that column is populated only at the six
    acquired timepoints, so it would render as a six-point polyline rather than a fan, and
    the Fig. 5 caption is explicit that the curves shown are the fits, not the raw readings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
# Helvetica first, as the sweep diagnostic does; Arial and DejaVu as fallbacks.
mpl.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
})
import matplotlib.pyplot as plt                                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, required=True,
                    help="marker_integration_trajectories_per_minute.csv for one fit.")
    ap.add_argument("--title", default="", help="Suptitle, e.g. 'PTPN7 | seed 43'.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.traj)
    need = {"GFP_bin", "timepoint", "fit_pERK1_2", "pred_integrated_ode"}
    if not need.issubset(d.columns):
        raise SystemExit(f"{args.traj} needs {sorted(need)}")
    d = d[d.pred_integrated_ode.notna()]
    if d.empty:
        raise SystemExit("no integrated predictions in this table")

    bins = sorted(d.GFP_bin.unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(bins), vmax=max(bins))

    # layout="compressed": set_box_aspect forces a square box inside each cell, and
    # tight_layout would leave the leftover width inside the cells, so shrinking wspace
    # would do nothing to the visible gap.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True, layout="compressed")
    for b in bins:
        h = d[d.GFP_bin == b].drop_duplicates("timepoint").sort_values("timepoint")
        c = cmap(norm(b))
        axes[0].plot(h.timepoint, h.fit_pERK1_2, color=c, lw=1.2)
        axes[1].plot(h.timepoint, h.pred_integrated_ode, color=c, lw=1.2)
    axes[0].set_title("Fitted p-ERK1-2")
    axes[1].set_title("ODE integrated pred")
    for ax in axes:
        ax.set_xlabel("time (min)")
        ax.set_box_aspect(1)          # square plotting area, independent of data range
        ax.set_xticks([0, 20, 40, 60])
    axes[0].set_ylabel("p-ERK (a.u.)")
    # The right panel's outward y ticks collide with the left panel's frame once the two sit
    # flush, and they carry no labels under sharey.
    axes[1].tick_params(axis="y", length=0)
    if args.title:
        fig.suptitle(args.title, fontsize=17)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight": set_box_aspect leaves the layout engine unable to account for the
    # axis labels, which were otherwise clipped at the bottom edge.
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {args.output.name}: {len(bins)} bins")


if __name__ == "__main__":
    main()
