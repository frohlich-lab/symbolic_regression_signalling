"""Figure S2 - integrated SR trajectories near the R^2 acceptance threshold.

Selects fits whose marker-level integrated R^2 (the median across their held-out
GFP bins - the quantity the 0.6 cutoff acts on) sits closest to a ladder of
target values (0.50 ... 0.70), and plots the ODE-integrated p-ERK trajectory of
a representative held-out bin against the measured values it was scored on.
This is what the 0.6 cutoff is calibrated against.

Reads the frozen OOD run under data/experimental/runs/pysr_ood_final, so the
panel reflects the selected configuration and the top-GFP-bin split.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402

MEASURED = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
TARGETS = (0.50, 0.55, 0.60, 0.65, 0.70)


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


def pick(df: pd.DataFrame, n_per_target: int) -> list[list[pd.Series]]:
    """n examples per target, selected on the marker-level score.

    The 0.6 cutoff in the paper is applied to a fit's median R^2 across its
    held-out bins, so examples are chosen on that quantity rather than on
    individual bins; the bin drawn is the one sitting at that median. Choosing
    per-bin instead would show trajectories whose score is not the one the
    threshold ever acts on.
    """
    fit_score = df.groupby(["marker", "seed"]).r2.median().rename("fit_r2")
    out = []
    used_markers: set[str] = set()
    for tgt in TARGETS:
        cand = (fit_score.reset_index()
                .assign(dist=lambda x: (x.fit_r2 - tgt).abs())
                .sort_values("dist"))
        chosen: list[pd.Series] = []
        seen: set[str] = set()
        for _, f in cand.iterrows():
            if f.marker in seen or f.marker in used_markers:
                continue
            bins = df[(df.marker == f.marker) & (df.seed == f.seed)]
            # The bin closest to the fit's median score represents it.
            row = bins.iloc[int((bins.r2 - f.fit_r2).abs().to_numpy().argmin())]
            row = row.copy()
            row["fit_r2"] = float(f.fit_r2)
            chosen.append(row)
            seen.add(f.marker)
            if len(chosen) == n_per_target:
                break
        used_markers |= seen
        out.append(chosen)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--n-per-target", type=int, default=3)
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_s2_threshold_examples.png"))
    args = ap.parse_args()

    df = collect(args.fits_dir)
    print(f"{len(df)} held-out bins across "
          f"{df.groupby(['marker', 'seed']).ngroups} fits")
    cols = pick(df, args.n_per_target)

    st.apply()
    nrow, ncol = args.n_per_target, len(TARGETS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.09, 1.35 * nrow),
                             sharex=True)
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(wspace=0.34, hspace=0.42)

    for j, (tgt, chosen) in enumerate(zip(TARGETS, cols)):
        for i in range(nrow):
            ax = axes[i, j]
            if i >= len(chosen):
                ax.axis("off")
                continue
            r = chosen[i]
            ok = r.fit_r2 >= 0.6
            ax.plot(r.t, r.curve, color=st.INK if ok else st.MUTED, lw=1.1,
                    zorder=3)
            ax.plot(r.t_obs, r.obs, "x", color="#b4553f", ms=3.2, mew=0.9,
                    zorder=4)
            ax.set_title(f"R² = {r.fit_r2:.2f}\n{r.marker} · s{r.seed} · bin {r['bin']}",
                         fontsize=st.FS_TICK, pad=2.5, linespacing=1.2,
                         color="#111111" if ok else "#666666")
            ax.set_xlim(0, 60)
            ax.set_xticks([0, 20, 40, 60])
            ax.tick_params(labelsize=st.FS_TICK, pad=1.5)
            ax.locator_params(axis="y", nbins=4)
            if j == 0:
                ax.set_ylabel("p-ERK", fontsize=st.FS_LABEL)
            if i == nrow - 1:
                ax.set_xlabel("Time (min)", fontsize=st.FS_LABEL)

    # Column banner naming the target each column was drawn around.
    for j, tgt in enumerate(TARGETS):
        axes[0, j].annotate(f"target R² ≈ {tgt:.2f}", xy=(0.5, 1.42),
                            xycoords="axes fraction", ha="center",
                            fontsize=st.FS_LABEL, fontweight="bold",
                            color="#333333")

    handles = [plt.Line2D([], [], color=st.INK, lw=1.1),
               plt.Line2D([], [], color="#b4553f", ls="none", marker="x",
                          ms=3.2, mew=0.9)]
    fig.legend(handles, ["Integrated SR prediction", "Measured p-ERK"],
               loc="lower center", ncol=2, fontsize=st.FS_TICK,
               bbox_to_anchor=(0.5, -0.055), handlelength=1.6)

    st.save(fig, args.output)
    for tgt, chosen in zip(TARGETS, cols):
        got = ", ".join(
            "{}/s{} fit R2={:.3f} (bin {} R2={:.3f})".format(
                r.marker, r.seed, r.fit_r2, r["bin"], r.r2)
            for r in chosen)
        print(f"  target {tgt:.2f}: {got}")


if __name__ == "__main__":
    main()
