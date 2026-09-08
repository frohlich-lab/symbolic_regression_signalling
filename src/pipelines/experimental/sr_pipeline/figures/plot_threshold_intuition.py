"""Visual intuition for the R^2 = 0.6 cutoff: every trajectory, grouped.

Three blocks, one per candidate cutoff. Each block draws the PER_LEVEL held-out
GFP bins closest in R^2 to that value - one panel per curve, measurement and
prediction together, sorted by R^2.

The rule is a fixed count rather than a fixed band. Drawing every curve within
+/-0.05 was the most airtight version (78 panels, no selection at all) but was
unreadable at page width, and narrowing the band left the blocks lopsided
(18/7/12 at +/-0.02) because the R^2 distribution is not uniform. Nearest-N is
symmetric across the three levels, has no free parameter beyond N, and answers
exactly the question asked: what does a curve at this R^2 look like.

Selection was the flaw in every earlier version of this figure. Draw the bins
nearest the cutoff and you show its worst admitted case; draw quantiles of the
accepted population and you show curves far better than the cutoff under a
header naming the cutoff, which leaves a false impression no caption repairs.
Drawing every curve in the band removes the question: there is no selection rule
to interrogate because there is no selection rule.

Each panel is scaled to its own measured range, so panels are comparable and the
reader is judging shape agreement rather than absolute p-ERK level.

What it shows: the 0.6 block is visibly tighter than the 0.5 block and hard to
tell apart from the 0.7 block. That is the case for 0.6 being an adequate bar.
It is NOT a claim that 0.6 is optimal - eight criteria were tested, none single
it out, and several favour 0.7 or higher (docs/experimental_provenance.md).

Bins with less than MIN_FOLD measured change are excluded throughout: R^2 is
scale-free, so a near-flat readout can score well while showing nothing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402
from plot_fig_s2_threshold_examples import CONTROL_RE, collect  # noqa: E402

LEVELS = (0.5, 0.6, 0.7)
PER_LEVEL = 8     # curves nearest each level; symmetric, no discretion
CUTOFF = 0.6
MIN_FOLD = 1.5
NCOL = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_threshold_intuition.png"))
    args = ap.parse_args()

    df = collect(args.fits_dir).reset_index(drop=True)
    df = df[~df.marker.astype(str).str.contains(CONTROL_RE)]
    fold = df.obs.map(lambda o: float(o.max()) / float(o.min())
                      if float(o.min()) > 0 else 1.0)
    df = df[fold >= MIN_FOLD]

    blocks = []
    for lvl in LEVELS:
        band = (df.assign(d=(df.r2 - lvl).abs())
                  .nsmallest(PER_LEVEL, "d")
                  .sort_values("r2", ascending=False))
        blocks.append((lvl, band))
    rows = [int(np.ceil(len(b) / NCOL)) for _, b in blocks]
    # One spacer row between blocks: without it the last row's R² labels ran
    # into the next block's heading. It is given a fraction of a panel's height
    # via height_ratios - a full row of whitespace was far too much.
    SPACER = 0.42
    starts, r, heights = [], 0, []
    for i, n in enumerate(rows):
        starts.append(r)
        heights += [1.0] * n
        r += n
        if i < len(rows) - 1:
            heights.append(SPACER)
            r += 1
    total_rows = r

    st.apply()
    fig = plt.figure(figsize=(6.27, sum(heights) * 0.92 + 0.95))
    gs = GridSpec(total_rows, NCOL, figure=fig, height_ratios=heights,
                  hspace=0.55, wspace=0.24,
                  left=0.035, right=0.995, bottom=0.05, top=0.93)

    for (lvl, band), r0 in zip(blocks, starts):
        chosen = abs(lvl - CUTOFF) < 1e-9
        col = st.ACCENT if chosen else st.INK
        errs, first_ax = [], None
        for k, (_, b) in enumerate(band.iterrows()):
            ax = fig.add_subplot(gs[r0 + k // NCOL, k % NCOL])
            if first_ax is None:
                first_ax = ax
            lo = float(b.obs.min())
            rng = float(b.obs.max() - lo)
            ax.plot(b.t, (b.curve - lo) / rng, color=col, lw=0.9, zorder=3)
            ax.plot(b.t_obs, (b.obs - lo) / rng, "x", color="#b4553f", ms=2.4,
                    mew=0.75, zorder=4)
            ax.axhline(0, color=st.GRID, lw=0.5, zorder=1)
            ax.axhline(1, color=st.GRID, lw=0.5, zorder=1)
            ax.set_xlim(0, 60)
            ax.set_ylim(-0.55, 1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.text(0.5, -0.06, f"{b.r2:.2f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=st.FS_TICK - 0.8,
                    color=st.MUTED)
            errs.append(np.abs(np.interp(b.t_obs, b.t, b.curve) - b.obs).mean()
                        / rng * 100)

        top = first_ax.get_position().y1
        fig.text(0.035, top + 0.006, f"R² ≈ {lvl:g}", ha="left", va="bottom",
                 fontsize=st.FS_LABEL,
                 fontweight="bold" if chosen else "normal",
                 color=col if chosen else "#111111")
        fig.text(0.160, top + 0.007,
                 f"the {len(band)} curves nearest this value · typical miss "
                 f"{np.median(errs):.0f}% of range",
                 ha="left", va="bottom", fontsize=st.FS_TICK - 0.5,
                 color=st.ACCENT if chosen else st.MUTED)
        print(f"R2 ~ {lvl}: {len(band)} curves, typical miss "
              f"{np.median(errs):.1f}% of range")

    fig.text(0.5, 0.008,
             f"The {PER_LEVEL} held-out curves closest in R² to each value, "
             "sorted; crosses are measured p-ERK, lines the model prediction "
             "over 0–60 min. Each panel is scaled to its own measured range.",
             ha="center", va="bottom", fontsize=st.FS_TICK - 0.5,
             color=st.MUTED)

    st.save(fig, args.output)


if __name__ == "__main__":
    main()
