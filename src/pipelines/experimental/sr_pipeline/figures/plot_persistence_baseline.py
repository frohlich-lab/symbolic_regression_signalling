"""SR against a dose-independent persistence baseline, per context.

The OOD task holds out the top 20% of GFP bins and asks the model to
extrapolate to an unseen dose. The obvious null for that task is to assume the
dose does nothing: predict every held-out bin with the trajectory measured at
the highest *training* dose. That predictor uses no held-out information and is
available at training time, so it is a legitimate baseline rather than an
oracle.

It is also a strong one, because p-ERK trajectories vary smoothly with dose
(median R^2 between adjacent GFP bins is 0.96). Scoring it the same way as the
models - median across held-out bins, negatives clipped to 0 - it reaches a
median R^2 of 0.80 and clears 0.6 on 18 of 31 contexts, against 0.07 and 8/31
for PySR.

**This replaces the R^2 = 0.6 cutoff with a criterion that needs no cutoff.** A
context counts as extrapolated only if the model beats persistence on it: points
above the diagonal. That is the standard "beat the naive forecast" test, it is
per-context rather than global, and it cannot be tuned. Six contexts pass.

Read the diagonal, not the axes. Most of what a fixed R^2 cutoff counts as
"solved" is recovered by assuming the dose does nothing.
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
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supp_style as st  # noqa: E402
from display_names import gene_label  # noqa: E402

MEASURED = [0, 5, 10, 15, 30, 60]
CONTROL_RE = re.compile(r"untransfected|FLAG.?GFP", re.IGNORECASE)


def r2(o: np.ndarray, p: np.ndarray) -> float:
    ss = float(((o - p) ** 2).sum())
    tot = float(((o - o.mean()) ** 2).sum())
    return 1 - ss / tot if tot > 0 else np.nan


def persistence(fits_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(fits_dir / "*" / "seed_42" / "metrics" /
                                     "marker_integration_trajectories_per_minute.csv"))):
        d = pd.read_csv(path)
        d = d[d.pred_integrated_ode.notna()]
        marker = str(d.marker.iloc[0])
        if CONTROL_RE.search(marker):
            continue
        train, test = {}, {}
        for (phase, b), g in d.groupby(["phase", "GFP_bin"]):
            g = g.sort_values("timepoint")
            k = g.timepoint.isin(MEASURED)
            if k.sum() < len(MEASURED):
                continue
            (train if phase == "train" else test)[int(b)] = \
                g.loc[k, "obs_pERK1_2"].to_numpy(float)
        if not train or not test:
            continue
        donor = train[max(train)]          # nearest available dose
        v = [r2(test[b], donor) for b in sorted(test)]
        v = [x for x in v if np.isfinite(x)]
        if v:
            rows.append((marker, max(float(np.median(v)), 0.0)))
    return pd.DataFrame(rows, columns=["marker", "baseline"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits-dir", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/fits"))
    ap.add_argument("--per-fit-csv", type=Path,
                    default=Path("data/experimental/runs/pysr_ood_final/metrics/"
                                 "integrated_r2_per_fit.csv"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary/fig_persistence_baseline.png"))
    args = ap.parse_args()

    base = persistence(args.fits_dir)
    sr = (pd.read_csv(args.per_fit_csv).query("~is_control")
          .groupby("marker").ode_integ_r2_median.max().rename("sr"))
    j = base.join(sr, on="marker").dropna()
    wins = j[j.sr > j.baseline]
    w = wilcoxon(j.baseline, j.sr)

    st.apply()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig.subplots_adjust(left=0.185, right=0.97, bottom=0.155, top=0.90)

    ax.fill_between([-0.05, 1.05], [-0.05, 1.05], 1.05, color=st.GREEN_FILL,
                    alpha=0.55, lw=0, zorder=0)
    ax.plot([-0.05, 1.05], [-0.05, 1.05], color="#888888", lw=0.8, zorder=2)
    ax.scatter(j.baseline, j.sr, s=22, facecolor="white", edgecolor=st.INK,
               lw=0.9, zorder=3)
    ax.scatter(wins.baseline, wins.sr, s=24, color=st.ACCENT, zorder=4)
    for _, r in wins.iterrows():
        ax.annotate(gene_label(r.marker), (r.baseline, r.sr),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=st.FS_TICK - 0.5, color=st.ACCENT)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Persistence baseline (nearest training dose)",
                  fontsize=st.FS_LABEL)
    ax.set_ylabel("PySR, held-out R²", fontsize=st.FS_LABEL)
    ax.set_title(f"SR beats persistence on {len(wins)}/{len(j)} contexts",
                 fontsize=st.FS_LABEL, pad=6, loc="left")
    ax.text(0.055, 0.965, "SR adds\nover persistence", fontsize=st.FS_TICK - 0.5,
            color="#4a7a5c", va="top", linespacing=1.2)

    st.save(fig, args.output)
    print(f"contexts {len(j)}; SR median {j.sr.median():.3f} vs baseline "
          f"{j.baseline.median():.3f}; baseline >=0.6 on "
          f"{int((j.baseline >= 0.6).sum())}, SR on {int((j.sr >= 0.6).sum())}; "
          f"paired Wilcoxon p={w.pvalue:.4f}")
    print("SR wins:", ", ".join(f"{r.marker} ({r.sr:.2f} vs {r.baseline:.2f})"
                                for _, r in wins.iterrows()))


if __name__ == "__main__":
    main()
