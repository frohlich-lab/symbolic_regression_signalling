"""Recompute network interaction counts against the OFF-DIAGONAL scale.

Thresholding off-diagonal curvature against the global Hessian maximum is not comparable
across function classes: symbolic laws with divisions carry huge diagonal curvature that
suppresses their cross-terms, while tanh networks do not. Scoring each law's off-diagonals
against its own off-diagonal maximum puts both on a common footing.
"""
import sys, json, glob, os
sys.path.insert(0, "src/pipelines/experimental/reducibility")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, equinox as eqx
from _shared import _baseline
nodb = _baseline(); rows = []
for jp in sorted(glob.glob("data/experimental/runs/reducibility/l21_hl4_local/seed_*/models/*.json")):
    meta = json.loads(open(jp).read()); base = jp[:-5]
    ep, tp = base + ".eqx", base + "_train.npz"
    if not (os.path.exists(ep) and os.path.exists(tp)): continue
    mdl = eqx.tree_deserialise_leaves(ep, nodb.RHS(
        in_dim=int(meta["in_dim"]), hidden_dim=int(meta["hidden_dim"]),
        hidden_layers=int(meta["hidden_layers"]), activation=str(meta["activation"]),
        key=jax.random.PRNGKey(0)))
    tr = dict(np.load(tp)); y = (tr["ys"] - float(meta["y_mean"])) / float(meta["y_scale"])
    us = tr["us_scaled"]; msk = tr["real_mask"].astype(bool).reshape(-1)
    X = np.concatenate([y.reshape(-1)[msk][:, None], us.reshape(-1, us.shape[-1])[msk]], axis=1)
    if X.size == 0: continue
    g = np.abs(np.asarray(jax.jit(jax.vmap(jax.grad(mdl)))(jnp.asarray(X)))).mean(axis=0)
    idx = np.where(g / max(g.max(), 1e-12) > 0.15)[0]; k = len(idx)
    sel = np.random.default_rng(0).choice(len(X), min(600, len(X)), replace=False)
    H = np.abs(np.asarray(jax.jit(jax.vmap(jax.hessian(mdl)))(jnp.asarray(X[sel])))).mean(axis=0)
    sub = H[np.ix_(idx, idx)] if k else np.zeros((0, 0))
    off = sub[np.triu_indices(k, 1)] if k > 1 else np.array([])
    omax = off.max() if off.size else 0.0
    rows.append(dict(marker=meta["marker"], seed=int(meta["seed"]), k=float(k),
                     p_offnorm=float((off > 0.25 * omax).sum()) if omax > 0 else 0.0))
    jax.clear_caches()
d = pd.DataFrame(rows).drop_duplicates(["marker", "seed"])
d.groupby("marker")["p_offnorm"].mean().rename("p0.25").reset_index().to_csv(
    "data/experimental/runs/reducibility/interactions_offnorm.csv", index=False)
print("markers", d.marker.nunique(), "| mean off-norm interaction count %.2f" % d.p_offnorm.mean())
