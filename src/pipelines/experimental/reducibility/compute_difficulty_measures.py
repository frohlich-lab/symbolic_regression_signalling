"""Complexity measures that track how HARD a rate law is, not how many inputs it has.

The manuscript's two measures -- effective dependency count and interacting-pair count --
are thresholded counts of the Jacobian and Hessian evaluated on TRAINING points. They
measure arity. They are blind to two things that determine whether SR recovers a law that
generalises:

  * functional form. A three-variable saturating law with a sign change is much harder
    than a six-variable linear one; arity scores the first as simpler.
  * behaviour off the training distribution. The held-out split is the top GFP bins, so a
    law that drifts as GFP rises will fail out of distribution no matter how few variables
    it uses. Measures evaluated only on training points cannot see this at all.

Measures computed here, all continuous and all normalised so they compare across contexts:

  nonlin        1 - R^2 of the best LINEAR model of f over the training inputs.
                Distance from the easiest expression class SR could possibly find.
  curv          mean |off-diagonal Hessian| / mean |f|. Interaction MAGNITUDE, the
                continuous counterpart of the thresholded pair count.
  cancel        mean(|J| . sigma_x) / mean|f|. How much the output relies on large terms
                cancelling; high values give SR a brutal loss landscape and were the
                diagnosed failure mode in several contexts.
  extrap_f      |f(GFP -> beyond train max) - f(GFP = train)| / sd(f on train). How far the
                law moves when pushed into the held-out region.
  extrap_j      ||J(extrapolated) - J(train)|| / ||J(train)||. Whether the law itself
                changes shape there, rather than merely taking larger values.
  gfp_drift     ||J(high-GFP bins) - J(low-GFP bins)|| / ||J(low)||, within training data.
                A shift measured without extrapolating at all.
  jac_pr        participation ratio of |J| -- continuous arity, kept as the control that
                should behave like the published measures.

Averaged over the three network seeds. Writes one row per marker.
"""
import glob
import json
import os
import sys

sys.path.insert(0, "src/pipelines/experimental/reducibility")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from _shared import _baseline

OUT = "data/experimental/runs/reducibility/difficulty_measures.csv"
HESS_N = 400
GFP_COL = 1          # X = [p_ERK, exo_cols...]; exo_cols[0] is GFP
EXTRAP_SD = 1.0      # push GFP this many training SDs past its training maximum


def measures(mdl, X, gfp_bin):
    f = np.asarray(jax.jit(jax.vmap(mdl))(jnp.asarray(X))).reshape(-1)
    absf = max(float(np.mean(np.abs(f))), 1e-9)
    sdf = max(float(np.std(f)), 1e-9)

    # --- distance from linearity: the easiest thing SR could find ---
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    resid = f - A @ coef
    nonlin = float(1.0 - (1.0 - resid.var() / max(f.var(), 1e-12)))
    nonlin = float(np.clip(resid.var() / max(f.var(), 1e-12), 0.0, 1.0))

    J = np.asarray(jax.jit(jax.vmap(jax.grad(mdl)))(jnp.asarray(X)))
    Jm = np.abs(J).mean(axis=0)

    # --- continuous arity (control) ---
    w = Jm / max(Jm.sum(), 1e-12)
    jac_pr = float(1.0 / max((w ** 2).sum(), 1e-12))

    # --- cancellation / conditioning ---
    cancel = float((Jm * X.std(axis=0)).sum() / absf)

    # --- interaction magnitude, not count ---
    sel = np.random.default_rng(0).choice(len(X), min(HESS_N, len(X)), replace=False)
    H = np.abs(np.asarray(jax.jit(jax.vmap(jax.hessian(mdl)))(jnp.asarray(X[sel])))).mean(axis=0)
    k = H.shape[0]
    off = H[np.triu_indices(k, 1)]
    curv = float(off.mean() / absf)

    # --- push GFP past its training range: does the law hold there? ---
    Xe = X.copy()
    Xe[:, GFP_COL] = X[:, GFP_COL].max() + EXTRAP_SD * X[:, GFP_COL].std()
    fe = np.asarray(jax.jit(jax.vmap(mdl))(jnp.asarray(Xe))).reshape(-1)
    extrap_f = float(np.mean(np.abs(fe - f)) / sdf)
    Je = np.asarray(jax.jit(jax.vmap(jax.grad(mdl)))(jnp.asarray(Xe)))
    nJ = max(float(np.linalg.norm(Jm)), 1e-9)
    extrap_j = float(np.linalg.norm(np.abs(Je).mean(axis=0) - Jm) / nJ)

    # --- shift measured inside the training data, no extrapolation ---
    med = np.median(gfp_bin)
    lo, hi = gfp_bin <= med, gfp_bin > med
    if lo.sum() > 5 and hi.sum() > 5:
        Jl = np.abs(J[lo]).mean(axis=0)
        Jh = np.abs(J[hi]).mean(axis=0)
        gfp_drift = float(np.linalg.norm(Jh - Jl) / max(np.linalg.norm(Jl), 1e-9))
    else:
        gfp_drift = np.nan

    return dict(nonlin=nonlin, curv=curv, cancel=cancel, extrap_f=extrap_f,
                extrap_j=extrap_j, gfp_drift=gfp_drift, jac_pr=jac_pr)


def main() -> None:
    nodb = _baseline()
    rows = []
    for jp in sorted(glob.glob(
            "data/experimental/runs/reducibility/l21_hl4_local/seed_*/models/*.json")):
        meta = json.loads(open(jp).read())
        base = jp[:-5]
        ep, tp = base + ".eqx", base + "_train.npz"
        if not (os.path.exists(ep) and os.path.exists(tp)):
            continue
        mdl = eqx.tree_deserialise_leaves(ep, nodb.RHS(
            in_dim=int(meta["in_dim"]), hidden_dim=int(meta["hidden_dim"]),
            hidden_layers=int(meta["hidden_layers"]), activation=str(meta["activation"]),
            key=jax.random.PRNGKey(0)))
        tr = dict(np.load(tp))
        y = (tr["ys"] - float(meta["y_mean"])) / float(meta["y_scale"])
        us = tr["us_scaled"]
        msk = tr["real_mask"].astype(bool)
        nb, nt = y.shape
        bin_idx = np.repeat(np.arange(nb)[:, None], nt, axis=1)[msk]
        X = np.concatenate([y[msk][:, None], us[msk]], axis=1)
        if len(X) < 20:
            continue
        m = measures(mdl, X, bin_idx)
        m.update(marker=meta["marker"], seed=int(meta["seed"]))
        rows.append(m)
        jax.clear_caches()

    d = pd.DataFrame(rows).drop_duplicates(["marker", "seed"])
    agg = d.drop(columns=["seed"]).groupby("marker").mean().reset_index()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    agg.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(agg)} markers x {agg.shape[1]-1} measures")
    print(agg.describe().T[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
