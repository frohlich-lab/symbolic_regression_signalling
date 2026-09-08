"""Render ODE-integrated vs observed pERK fans for chosen sweep shards.

This is the diagnostic that tracks overlay quality: the predicted fan should reproduce
the observed viridis gradient (ordered by GFP bin, no crossings) with the same peak
timing and decay. R2 alone ranks flat monotonic fans above dynamically correct fits.

Usage:
  python make_fan_plots.py <sweep_dir> <out_dir> [arm:marker_slug:seed ...]

With no shard list, plots every shard that clears a basic quality bar.
"""
import os
import sys
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Helvetica proper (weights 300/400/700 are registered on this machine, so bold does not
# silently fall back); Arial and DejaVu kept as fallbacks for other systems.
matplotlib.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15,
})
import numpy as np
import pandas as pd

SWEEP = sys.argv[1].rstrip("/")
OUTDIR = sys.argv[2].rstrip("/")
WANTED = sys.argv[3:]
os.makedirs(OUTDIR, exist_ok=True)

TRAJ_GLOB = f"{SWEEP}/out_seed/*/*/s*/seeds/seed_*/metrics/marker_integration_trajectories_per_minute.csv"


def shard_id(path):
    parts = path.split(os.sep)
    return parts[-7], parts[-6], parts[-5].lstrip("s")  # arm, slug, seed


def plot_shard(path, arm, slug, seed):
    d = pd.read_csv(path)
    d = d[d["pred_integrated_ode"].notna()]
    if d.empty:
        return None
    bins = sorted(d["GFP_bin"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(bins), vmax=max(bins))

    # layout="compressed": set_box_aspect below forces a square box inside each subplot cell, and
    # tight_layout leaves the leftover width *inside* the cells -- so shrinking wspace does nothing
    # to the visible gap. The compressed engine shrinks the cells onto the fixed-aspect boxes.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True, layout="compressed")
    for b in bins:
        h = d[d["GFP_bin"] == b].sort_values("timepoint")
        c = cmap(norm(b))
        # data on the left, model on the right: the reader sees what was measured first
        axes[0].plot(h["timepoint"], h["obs_pERK1_2"], color=c, lw=1.2)
        axes[1].plot(h["timepoint"], h["pred_integrated_ode"], color=c, lw=1.2)
    axes[0].set_title("Observed p-ERK1-2")
    axes[1].set_title("ODE integrated pred")
    for ax in axes:
        ax.set_xlabel("time (min)")
        ax.set_box_aspect(1)   # square plotting area, independent of the data ranges
        ax.set_xticks([0, 20, 40, 60])
    axes[0].set_ylabel("p-ERK (a.u.)")   # same wording as the per-bin panels
    # the right panel's outward y ticks would collide with the left panel's frame once the two
    # sit flush, and they carry no labels under sharey, so drop them
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle(f"{slug} | {arm} | seed {seed}", fontsize=17)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)
    out = f"{OUTDIR}/{arm}__{slug}__s{seed}.png"
    # bbox_inches=tight: set_box_aspect leaves tight_layout unable to account for the axis
    # labels, which were being clipped at the bottom edge.
    fig.savefig(out, dpi=200, bbox_inches="tight")
    # vector alongside the raster, as every other paper-figure script does
    fig.savefig(Path(out).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


paths = sorted(glob.glob(TRAJ_GLOB))
print(f"found {len(paths)} trajectory files")

selected = []
for p in paths:
    arm, slug, seed = shard_id(p)
    key = f"{arm}:{slug}:{seed}"
    if WANTED and key not in WANTED:
        continue
    selected.append((p, arm, slug, seed))

if WANTED and not selected:
    print("none of the requested shards were found; requested:", WANTED)

for p, arm, slug, seed in selected:
    try:
        out = plot_shard(p, arm, slug, seed)
        print("wrote" if out else "empty", out or f"{arm}/{slug}/s{seed}")
    except Exception as exc:
        print(f"FAIL {arm}/{slug}/s{seed}: {exc}")

print(f"\n{len(selected)} shards plotted into {OUTDIR}")
