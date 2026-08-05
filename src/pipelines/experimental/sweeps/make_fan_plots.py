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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for b in bins:
        h = d[d["GFP_bin"] == b].sort_values("timepoint")
        c = cmap(norm(b))
        axes[0].plot(h["timepoint"], h["pred_integrated_ode"], color=c, lw=1.2)
        axes[1].plot(h["timepoint"], h["obs_pERK1_2"], color=c, lw=1.2)
    axes[0].set_title("ODE integrated pred")
    axes[1].set_title("Observed p-ERK1-2")
    for ax in axes:
        ax.set_xlabel("time (min)")
    axes[0].set_ylabel("pERK")
    fig.suptitle(f"{slug} | {arm} | seed {seed}", fontsize=11)
    fig.tight_layout()
    out = f"{OUTDIR}/{arm}__{slug}__s{seed}.png"
    fig.savefig(out, dpi=130)
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
