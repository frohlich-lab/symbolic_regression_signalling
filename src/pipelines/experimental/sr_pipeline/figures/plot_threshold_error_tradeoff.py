"""What each candidate cutoff buys: held-out prediction error against yield.

For every candidate cutoff, the typical error of everything that cutoff would
accept - median across held-out bins of the mean absolute residual, as a
percentage of that bin's measured dynamic range. Normalising per bin makes the
number comparable across contexts at different absolute p-ERK levels, and gives
it a reading a biologist can act on: "the accepted fits track the measured
trajectory to within X% of its own range".

**This is a smooth tradeoff, not a sweet spot, and must not be drawn as one.**
Error falls monotonically with the cutoff and the intervals overlap heavily
across 0.5-0.7; there is no knee. An earlier version of this figure binned fits
*marginally* (within +/-0.05 of each cutoff) and appeared to show a plateau at
0.55-0.75 - an artefact of bands holding only 3-5 fits, where the mean and
median disagree by 11 points. Always summarise cumulatively, and bootstrap over
fits rather than bins: bins within a fit are not independent.

What the panel supports is the accuracy claim - at 0.6 the accepted fits predict
held-out trajectories to within ~12% of their dynamic range - plus the yield
cost of tightening further. See docs/experimental_provenance.md for the six
criteria that were tested and did NOT single out 0.6.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402
from plot_fig_s2_threshold_examples import CONTROL_RE, collect  # noqa: E402

CHOSEN = 0.6
SEED = 20260807
LABEL_AT = (0.4, 0.5, 0.6, 0.7, 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--per-fit-csv", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/metrics/"
                                 "integrated_r2_per_fit.csv"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_threshold_error_tradeoff.png"))
    args = ap.parse_args()

    df = collect(args.fits_dir)
    df = df[~df.marker.astype(str).str.contains(CONTROL_RE)]
    fit_r2 = df.groupby(["marker", "seed"]).r2.median().rename("fit_r2")
    df = df.join(fit_r2, on=["marker", "seed"])

    rows = []
    for _, r in df.iterrows():
        rng = float(r.obs.max() - r.obs.min())
        if rng <= 0:
            continue
        pred = np.interp(r.t_obs, r.t, r.curve)
        rows.append((r.fit_r2, 100.0 * np.abs(pred - r.obs).mean() / rng,
                     f"{r.marker}|{r.seed}"))
    s = pd.DataFrame(rows, columns=["fit_r2", "err", "fit"])

    real = (pd.read_csv(args.per_fit_csv).query("~is_control")
            .groupby("marker").ode_integ_r2_median.max().to_numpy(float))

    grid = np.arange(0.40, 0.801, 0.01)
    rng = np.random.default_rng(SEED)
    med, lo, hi = [], [], []
    for c in grid:
        q = s[s.fit_r2 >= c]
        med.append(q.err.median())
        keys = q.fit.unique()
        # Resample fits, not bins: bins within a fit share a formula.
        boot = [q[q.fit.isin(rng.choice(keys, len(keys)))].err.median()
                for _ in range(args.n_boot)]
        a, b = np.percentile(boot, [2.5, 97.5])
        lo.append(a); hi.append(b)
    med, lo, hi = np.array(med), np.array(lo), np.array(hi)
    at = int(np.argmin(np.abs(grid - CHOSEN)))

    st.apply()
    fig, ax = plt.subplots(figsize=(3.9, 2.85))
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.185, top=0.80)

    ax.fill_between(grid, lo, hi, color=st.INK, alpha=0.16, lw=0, zorder=2)
    ax.plot(grid, med, color=st.INK, lw=1.6, zorder=3)
    ax.axvline(CHOSEN, color=st.ACCENT, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
    ax.plot([CHOSEN], [med[at]], "o", ms=5, color=st.ACCENT, zorder=6)
    ax.annotate(f"{med[at]:.0f}% of range", xy=(CHOSEN, med[at]),
                xytext=(0.645, med[at] + 4.6), fontsize=st.FS_LABEL,
                fontweight="bold", color=st.ACCENT,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=st.ACCENT))

    ax.set_xlim(0.40, 0.80)
    ax.set_ylim(0, 20)
    ax.set_xlabel("Acceptance cutoff (integrated R²)", fontsize=st.FS_LABEL)
    ax.set_ylabel("Held-out prediction error of\naccepted fits (% of measured range)",
                  fontsize=st.FS_LABEL, linespacing=1.25)

    # Yield along the top, so the cost of tightening is read off the same axis.
    for c in LABEL_AT:
        ax.text(c, 21.4, f"{int((real >= c).sum())}", ha="center", va="bottom",
                fontsize=st.FS_TICK,
                color=st.ACCENT if abs(c - CHOSEN) < 1e-9 else st.MUTED,
                fontweight="bold" if abs(c - CHOSEN) < 1e-9 else "normal")
    ax.text(0.60, 23.4, "contexts kept", ha="center", va="bottom",
            fontsize=st.FS_TICK, color=st.MUTED)

    st.save(fig, args.output)
    for c in LABEL_AT:
        i = int(np.argmin(np.abs(grid - c)))
        n_fits = s[s.fit_r2 >= c].fit.nunique()
        print(f"cutoff {c}: {int((real >= c).sum()):2d} contexts, {n_fits:2d} fits, "
              f"error {med[i]:.1f}% [{lo[i]:.0f}, {hi[i]:.0f}]")


if __name__ == "__main__":
    main()
