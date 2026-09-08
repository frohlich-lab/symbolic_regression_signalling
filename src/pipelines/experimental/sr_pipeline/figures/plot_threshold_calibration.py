"""Why R^2 = 0.6 - one panel, calibrated against a wrong-context null.

The null asks what integrated R^2 a trajectory earns when it is a real,
well-behaved SR fit belonging to a *different* context. It is built to mirror
the reported statistic exactly: per held-out bin, score a donor curve drawn from
another context; take the median across bins (the quantity the cutoff acts on);
take the best of three seeds (as the pipeline does). Skipping that last step
compares a single draw against a maximum of three and inflates the separation.

**The claim is the vertical gap at the cutoff, not where the null curve ends.**
Where it ends is set by the number of draws: at 128,000 draws it terminated at
R^2 = 0.63, which looked like a cliff exactly at the cutoff and was an artefact
of the resolution limit. At 1.92M draws the tail runs to 0.796. Read the gap at
0.6, which is stable; do not caption the endpoint.

This justifies 0.6 as safe, not optimal - the null floor alone would license any
cutoff above roughly 0.35. The companion argument, that nothing depends on the
exact value, is printed as a diagnostic for the Methods text rather than drawn,
to keep the panel to one idea.
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

MEASURED = [0.0, 5.0, 10.0, 15.0, 30.0, 60.0]
SEED = 20260807


def r2v(o: np.ndarray, p: np.ndarray) -> np.ndarray:
    ss = ((o - p) ** 2).sum(-1)
    tot = ((o - o.mean(-1, keepdims=True)) ** 2).sum(-1)
    return np.where(tot > 0, 1 - ss / tot, np.nan)


def wrong_context_null(fits_dir: Path, n_rep: int) -> np.ndarray:
    """Best-of-three-seeds integrated R^2 for trajectories from other contexts."""
    df = collect(fits_dir).reset_index(drop=True)
    df = df[~df.marker.astype(str).str.contains(CONTROL_RE)].reset_index(drop=True)
    obs = np.stack(df.obs.values)
    pred = np.stack([np.interp(MEASURED, r.t, r.curve) for _, r in df.iterrows()])
    marker = df.marker.astype(str).values
    rng = np.random.default_rng(SEED)

    per_fit: dict[tuple, np.ndarray] = {}
    for (m, s), idx in df.groupby(["marker", "seed"]).indices.items():
        pool = np.flatnonzero(marker != m)
        donor = rng.choice(pool, size=(n_rep, len(idx)))
        # Negative R^2 is clipped to 0 upstream, so the null must be clipped too.
        v = np.clip(r2v(obs[idx][None, :, :], pred[donor]), 0, None)
        per_fit[(m, s)] = np.median(v, axis=1)

    return np.stack([
        np.max([v for k, v in per_fit.items() if k[0] == m], axis=0)
        for m in np.unique(marker)
    ]).ravel()


def parsimony_vs_cutoff(panel_csv: Path, thetas: np.ndarray):
    """Median participation ratio of each group at each candidate cutoff."""
    p = pd.read_csv(panel_csv)
    pr, r2 = p.nn_pr.to_numpy(float), p.py_r2.to_numpy(float)
    out = []
    for th in thetas:
        ok = r2 >= th
        if ok.sum() < 3 or (~ok).sum() < 3:
            continue
        out.append((th, int(ok.sum()), np.median(pr[ok]), np.median(pr[~ok])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--per-fit-csv", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/metrics/"
                                 "integrated_r2_per_fit.csv"))
    ap.add_argument("--panel-csv", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/metrics/"
                                 "panel_inputs_32perturbations.csv"))
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--n-rep", type=int, default=60000,
                    help="Replicates per fit; total draws = n_rep x 32 contexts.")
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_threshold_calibration.png"))
    args = ap.parse_args()

    null = wrong_context_null(args.fits_dir, args.n_rep)
    real = (pd.read_csv(args.per_fit_csv).query("~is_control")
            .groupby("marker").ode_integ_r2_median.max().to_numpy(float))
    grid = np.linspace(0, 1, 1001)
    s_real = np.array([(real >= t).mean() for t in grid])
    s_null = np.array([(null >= t).mean() for t in grid])
    floor = 1.0 / len(null)

    p_real = float((real >= args.cutoff).mean())
    p_null = float((null >= args.cutoff).mean())
    ratio = p_real / p_null

    st.apply()
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    fig.subplots_adjust(left=0.215, right=0.97, bottom=0.19, top=0.95)

    ax.fill_between(grid, floor * 0.3, np.maximum(s_null, floor * 0.3),
                    color=st.MUTED, alpha=0.25, lw=0, zorder=1)
    ax.plot(grid, np.maximum(s_null, floor * 0.3), color=st.MUTED, lw=1.2,
            zorder=2)
    ax.plot(grid, np.maximum(s_real, 1e-12), color=st.INK, lw=1.6, zorder=3)
    ax.axvline(args.cutoff, color=st.ACCENT, lw=1.0, ls=(0, (4, 2.5)), zorder=4)

    # The one quantity the panel is making: the separation at the cutoff.
    xa = args.cutoff + 0.045
    ax.annotate("", xy=(xa, p_real), xytext=(xa, p_null),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color=st.ACCENT,
                                shrinkA=0, shrinkB=0), zorder=6)
    # Rounded: the ratio rests on ~50 null counts, so its Poisson error is ~15%
    # and a four-digit figure would be false precision.
    rounded = round(ratio, -int(np.floor(np.log10(ratio))))
    ax.text(xa + 0.03, np.sqrt(p_real * p_null), f"≈{rounded:,.0f}×",
            fontsize=st.FS_LABEL, fontweight="bold", color=st.ACCENT,
            ha="left", va="center")

    ax.set_yscale("log")
    ax.set_ylim(floor * 0.25, 3.2)
    ax.set_xlim(0, 1)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
    ax.set_yticklabels(["0.0001%", "0.01%", "1%", "100%"])
    ax.set_ylabel("Contexts clearing the cutoff", fontsize=st.FS_LABEL)
    ax.set_xlabel("Acceptance cutoff (integrated R²)", fontsize=st.FS_LABEL)
    ax.text(0.055, 0.60, "Real contexts", fontsize=st.FS_LABEL, color=st.INK,
            ha="left", va="bottom")
    ax.text(0.06, 1.0e-3, "Wrong-context null", fontsize=st.FS_LABEL,
            color=st.MUTED, ha="left", va="top")
    ax.text(args.cutoff - 0.018, 2.6, f"R² = {args.cutoff:g}",
            fontsize=st.FS_LABEL, color=st.ACCENT, ha="right", va="top")

    st.save(fig, args.output)
    print(f"null draws {len(null):,}  max {null.max():.3f}  "
          f"99.99th pct {np.percentile(null, 99.99):.3f}")
    print(f"at cutoff {args.cutoff}: real {p_real * 100:.1f}% "
          f"({int((real >= args.cutoff).sum())}/32), null {p_null * 100:.5f}% "
          f"({int((null >= args.cutoff).sum())}/{len(null):,}) -> {ratio:,.0f}x")
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"   cutoff {t}: real {(real >= t).sum():2d}/32, "
              f"null {(null >= t).mean() * 100:.5f}%")
    rows = parsimony_vs_cutoff(args.panel_csv, np.arange(0.35, 0.735, 0.05))
    print("insensitivity (for the Methods sentence, not drawn):")
    for th, n, s_, u_ in rows:
        print(f"   theta {th:.2f}: n_solved {n:2d}, PR solved {s_:.2f} "
              f"vs unsolved {u_:.2f}")


if __name__ == "__main__":
    main()
