"""Figure S2 - what the R^2 = 0.6 acceptance cutoff buys, fit by fit.

Shows the K perturbation fits immediately below and the K immediately above the
cutoff, sorted by marker-level integrated R^2 (the median across their held-out
GFP bins - the quantity the cutoff acts on). For each, the ODE-integrated p-ERK
trajectory of a representative held-out bin is drawn against the measured values
it was scored on.

Three choices make this an audit rather than an illustration:

  * **Perturbations only.** The 24 control fits cluster at R^2 0.60-0.73, so any
    selection reaching above the cutoff fills up with them. The old target-ladder
    version was 6/9 controls at and above 0.6 and 0/6 below, which made "cleared
    the cutoff" and "is a trivially easy control" the same visual category. The
    paper excludes controls from the headline rate for exactly this reason.
  * **No omissions.** Taking the K nearest fits either side selects a contiguous
    R^2 band, and every perturbation fit inside that band is drawn. Nothing in the
    window is dropped, so the panel cannot flatter the cutoff by picking.
  * **Comparable axes.** Each panel is scaled to its own measured range, so the
    prediction's error reads as a fraction of the dynamics it is meant to
    reproduce. On free axes a fit over a 1.2x fold range looked as convincing as
    one over 2x. The raw fold range is printed per panel so nothing is hidden.

Reads the frozen OOD run under data/experimental/runs/pysr_ood_final, so the
panel reflects the selected configuration and the top-GFP-bin split.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402
from display_names import gene_label  # noqa: E402

MEASURED = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
# Same rule as summarise_pysr_ood.CONTROL_RE: contexts with no construct.
CONTROL_RE = re.compile(r"untransfected|FLAG.?GFP", re.IGNORECASE)
NCOL = 3


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def collect(fits_dir: Path) -> pd.DataFrame:
    """Per (marker, seed, held-out bin) integrated R^2 plus the curve itself."""
    rows = []
    pattern = str(fits_dir / "*" / "seed_*" / "metrics" /
                  "marker_integration_trajectories_per_minute.csv")
    for path in sorted(glob.glob(pattern)):
        d = pd.read_csv(path)
        # The file interleaves the derivative-integration rows (pred_integrated,
        # ODE column NaN) with the true ODE-solver rows. Only the latter are what
        # the reported ode_integ_r2_median scores, so keep those.
        d = d[d.pred_integrated_ode.notna()]
        d = d[d.phase == "test"]
        if d.empty:
            continue
        marker = str(d.marker.iloc[0])
        seed = int(d.seed.iloc[0])
        for b, g in d.groupby("GFP_bin"):
            g = g.sort_values("timepoint")
            m = g.timepoint.isin(MEASURED)
            obs = g.loc[m, "obs_pERK1_2"].to_numpy(float)
            pred = g.loc[m, "pred_integrated_ode"].to_numpy(float)
            if len(obs) < 2 or not np.isfinite(pred).all():
                continue
            rows.append({
                "marker": marker, "seed": seed, "bin": int(b),
                "r2": r2(obs, pred),
                "t": g.timepoint.to_numpy(float),
                "curve": g.pred_integrated_ode.to_numpy(float),
                "t_obs": g.loc[m, "timepoint"].to_numpy(float),
                "obs": obs,
            })
    return pd.DataFrame(rows)


def pick(df: pd.DataFrame, cutoff: float, n_per_side: int) -> list[pd.Series]:
    """The n fits either side of the cutoff, ascending, controls excluded.

    The cutoff is applied to a fit's median R^2 across its held-out bins, so fits
    are ranked on that quantity rather than on individual bins; the bin drawn is
    the one sitting at that median. Selecting per-bin instead would show
    trajectories whose score is not the one the cutoff ever acts on.
    """
    keep = df[~df.marker.astype(str).str.contains(CONTROL_RE)]
    fit_score = keep.groupby(["marker", "seed"]).r2.median().rename("fit_r2")
    fit_score = fit_score.reset_index().sort_values("fit_r2")

    below = fit_score[fit_score.fit_r2 < cutoff].tail(n_per_side)
    above = fit_score[fit_score.fit_r2 >= cutoff].head(n_per_side)
    chosen = pd.concat([below, above])

    out = []
    for _, f in chosen.iterrows():
        bins = keep[(keep.marker == f.marker) & (keep.seed == f.seed)]
        # The bin closest to the fit's median score represents it.
        row = bins.iloc[int((bins.r2 - f.fit_r2).abs().to_numpy().argmin())].copy()
        row["fit_r2"] = float(f.fit_r2)
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--cutoff", type=float, default=0.6)
    ap.add_argument("--n-per-side", type=int, default=6)
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_s2_threshold_examples.png"))
    args = ap.parse_args()

    df = collect(args.fits_dir)
    print(f"{len(df)} held-out bins across "
          f"{df.groupby(['marker', 'seed']).ngroups} fits")
    chosen = pick(df, args.cutoff, args.n_per_side)
    scores = [r.fit_r2 for r in chosen]
    n_below = sum(s < args.cutoff for s in scores)

    # The selection is a contiguous band, so state its width and confirm that
    # every perturbation fit inside it is drawn - that is the claim the caption
    # makes and it should fail loudly rather than quietly if it stops holding.
    lo, hi = min(scores), max(scores)
    keep = df[~df.marker.astype(str).str.contains(CONTROL_RE)]
    in_band = keep.groupby(["marker", "seed"]).r2.median()
    in_band = in_band[(in_band >= lo) & (in_band <= hi)]
    assert len(in_band) == len(chosen), (
        f"band {lo:.3f}-{hi:.3f} holds {len(in_band)} fits but {len(chosen)} drawn")

    st.apply()
    nrow = int(np.ceil(len(chosen) / NCOL))
    fig, axes = plt.subplots(nrow, NCOL, figsize=(6.27, 1.24 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(wspace=0.18, hspace=0.72, left=0.10, right=0.98)

    for k, ax in enumerate(axes.ravel()):
        if k >= len(chosen):
            ax.axis("off")
            continue
        r = chosen[k]
        ok = r.fit_r2 >= args.cutoff
        # Scale to the measured range of this bin: the observations span 0-1 in
        # every panel, so a given vertical gap is the same fraction of the
        # dynamics everywhere and the panels can be compared by eye.
        lo_o, hi_o = float(r.obs.min()), float(r.obs.max())
        span = hi_o - lo_o
        norm = (lambda y: (np.asarray(y, float) - lo_o) / span) if span > 0 else \
            (lambda y: np.asarray(y, float) * 0.0)
        ax.axhline(0, color=st.GRID, lw=0.6, zorder=1)
        ax.axhline(1, color=st.GRID, lw=0.6, zorder=1)
        ax.plot(r.t, norm(r.curve), color=st.INK if ok else st.MUTED, lw=1.1,
                zorder=3)
        ax.plot(r.t_obs, norm(r.obs), "x", color=st.ACCENT, ms=3.2, mew=0.9,
                zorder=4)
        ax.set_title(f"R² = {r.fit_r2:.2f}\n{gene_label(r.marker)} · s{r.seed} "
                     f"· bin {r['bin']}",
                     fontsize=st.FS_TICK, pad=2.5, linespacing=1.2,
                     color="#111111" if ok else "#666666")
        # Fold range of the measurements the R² was computed on. Without it a
        # near-flat readout on a rescaled axis reads like a well-fitted one.
        ax.text(0.97, 0.04, f"{hi_o / lo_o:.1f}×" if lo_o > 0 else "",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=st.FS_TICK - 0.5, color=st.MUTED)
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 20, 40, 60])
        ax.set_ylim(-0.35, 1.45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["min", "max"])
        ax.tick_params(labelsize=st.FS_TICK, pad=1.5)
        if k % NCOL == 0:
            ax.set_ylabel("p-ERK\n(measured range)", fontsize=st.FS_LABEL,
                          linespacing=1.2)
        if k >= len(chosen) - NCOL:
            ax.set_xlabel("Time (min)", fontsize=st.FS_LABEL)

    # The cutoff falls on a row boundary by construction (n_per_side is a
    # multiple of NCOL), so it can be drawn as one rule across the figure.
    split_row = n_below // NCOL
    if n_below % NCOL == 0 and 0 < split_row < nrow:
        top = axes[split_row - 1, 0].get_position().y0
        bot = axes[split_row, 0].get_position().y1 + 0.052
        y = 0.5 * (top + bot)
        fig.add_artist(plt.Line2D([0.10, 0.98], [y, y], color=st.ACCENT,
                                  lw=0.9, ls=(0, (4, 2.5))))
        # Sat on the row above when placed at the left end; the right end of the
        # inter-row gap is the only clear space at this figure width.
        fig.text(0.975, y, f"R² = {args.cutoff:g} cutoff",
                 fontsize=st.FS_TICK, color=st.ACCENT, va="center", ha="right",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

    handles = [plt.Line2D([], [], color=st.INK, lw=1.1),
               plt.Line2D([], [], color=st.MUTED, lw=1.1),
               plt.Line2D([], [], color=st.ACCENT, ls="none", marker="x",
                          ms=3.2, mew=0.9)]
    fig.legend(handles,
               ["Accepted (≥ cutoff)", "Rejected (< cutoff)", "Measured p-ERK"],
               loc="lower center", ncol=3, fontsize=st.FS_TICK,
               bbox_to_anchor=(0.5, 0.008), handlelength=1.6)

    st.save(fig, args.output)
    print(f"perturbation fits only; band {lo:.3f}-{hi:.3f} contains "
          f"{len(in_band)} fits, all drawn ({n_below} below cutoff, "
          f"{len(chosen) - n_below} at or above)")
    for r in chosen:
        print("  {}/s{} fit R2={:.3f} (bin {} R2={:.3f}, fold {:.2f})".format(
            r.marker, r.seed, r.fit_r2, r["bin"], r.r2,
            float(r.obs.max()) / float(r.obs.min())))


if __name__ == "__main__":
    main()
