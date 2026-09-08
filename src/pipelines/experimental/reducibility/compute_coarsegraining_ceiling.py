"""How much of the rate is explainable by ANY instantaneous function of the observed inputs.

SR fits dy/dt = f(y, u) -- an instantaneous, memoryless function of the measured variables.
That form is only available if p-ERK's rate really is a function of what was measured. If the
observable is a coarse-graining of a finer network whose hidden variables move on comparable
timescales, the effective rate carries memory, and then NO expression in these inputs fits it,
at any complexity. The obstruction is a property of the data, not of the search.

It is measurable without fitting anything. Take pairs of points that are near-neighbours in
input space: a function must assign them nearly equal rates. The extent to which their
measured rates disagree, extrapolated to zero separation, is the variance no function can
explain -- the Gamma (or delta) test of Stefansson, Koncar & Jones. Dividing by the target
variance turns it into a CEILING on the R^2 of any instantaneous model:

    ceiling = 1 - Gamma / var(dy/dt)

Two versions, matching SR's two ways of failing:

  ceiling_train  computed inside the training bins. A low value means no expression can fit
                 the training data -- the coarse-graining is incomplete there.
  ceiling_shift  held-out points are matched to their nearest TRAINING neighbours. A low
                 value means the map from inputs to rate differs between the training region
                 and the held-out high-GFP region, so a law fitted on train cannot transfer
                 however well it fits.

Neither uses SR, the neural ODE, or any fitted model, so using them to explain SR's failures
is not circular. And ceiling_train makes a falsifiable prediction: no context should show an
SR train R^2 above it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

FEATS = ["p-ERK1-2_fit", "GFP", "p-ERK1-2_min", "p-MEK1-2_fit", "p-MEK1-2_min",
         "p-RAF_fit", "p-p90RSK_fit", "p-MAPKAPK2_fit", "p-PDK1_fit", "p-MKK3-6_fit"]
TARGET = "p-ERK1-2_dt"
KMAX = 12
HELDOUT_FRAC = 0.2      # top_gfp_bins: the top 20% of GFP bins are held out


def _knn(A, B, kmax):
    """Indices of the kmax nearest rows of B for each row of A, by Euclidean distance."""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    order = np.argsort(d2, axis=1)[:, :kmax]
    return order, np.take_along_axis(d2, order, axis=1)


def gamma_test(X, y, kmax=KMAX, self_exclude=True, Xq=None, yq=None):
    """Gamma-test estimate of irreducible variance.

    Regresses the half mean-squared target difference of k-th neighbours (gamma) on their
    mean squared input distance (delta); the intercept estimates the variance no smooth
    function of X can explain. With Xq/yq given, query points are matched against X --
    that measures cross-region consistency instead of within-region noise.
    """
    A = X if Xq is None else Xq
    ya = y if yq is None else yq
    if len(A) < 5 or len(X) < kmax + 2:
        return np.nan
    order, d2 = _knn(A, X, kmax + (1 if self_exclude and Xq is None else 0))
    if self_exclude and Xq is None:
        order, d2 = order[:, 1:], d2[:, 1:]     # drop self-match
    deltas, gammas = [], []
    for k in range(order.shape[1]):
        deltas.append(d2[:, k].mean())
        gammas.append(0.5 * np.mean((ya - y[order[:, k]]) ** 2))
    deltas, gammas = np.asarray(deltas), np.asarray(gammas)
    if len(deltas) < 2 or np.ptp(deltas) <= 0:
        return np.nan
    slope, intercept = np.polyfit(deltas, gammas, 1)
    return float(max(intercept, 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/experimental/processed/functional_groups/"
                                         "markers_per_minute_fit.csv")
    ap.add_argument("--output", default="data/experimental/runs/reducibility/"
                                       "coarsegraining_ceiling.csv")
    ap.add_argument("--measured-timepoints", type=float, nargs="+",
                    default=[0, 5, 10, 15, 30, 60],
                    help="Restrict to real measurements; the per-minute grid is interpolated, "
                         "and interpolated points would understate the disagreement.")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[df.timepoint.isin(args.measured_timepoints)]
    rows = []
    for marker, g in df.groupby("marker"):
        g = g.dropna(subset=FEATS + [TARGET])
        if len(g) < 30:
            continue
        # top_gfp_bins split, reproduced from the bin ordering
        bins = np.sort(g.GFP_bin.unique())
        ncut = max(1, int(round(HELDOUT_FRAC * len(bins))))
        held = set(bins[-ncut:])
        tr, te = g[~g.GFP_bin.isin(held)], g[g.GFP_bin.isin(held)]

        mu, sd = tr[FEATS].mean().values, tr[FEATS].std().replace(0, 1).values
        Xtr = ((tr[FEATS].values - mu) / sd)
        ytr = tr[TARGET].values
        vtr = float(np.var(ytr))
        gam_tr = gamma_test(Xtr, ytr)
        ceil_tr = 1.0 - gam_tr / vtr if (vtr > 0 and gam_tr == gam_tr) else np.nan

        ceil_sh = np.nan
        if len(te) >= 5:
            Xte = ((te[FEATS].values - mu) / sd)
            yte = te[TARGET].values
            vte = float(np.var(np.r_[ytr, yte]))
            gam_sh = gamma_test(Xtr, ytr, self_exclude=False, Xq=Xte, yq=yte)
            ceil_sh = 1.0 - gam_sh / vte if (vte > 0 and gam_sh == gam_sh) else np.nan

        rows.append(dict(marker=marker, n_train=len(tr), n_heldout=len(te),
                         ceiling_train=ceil_tr, ceiling_shift=ceil_sh,
                         target_var=vtr))
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(out)} markers")
    print(out[["ceiling_train", "ceiling_shift"]].describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
