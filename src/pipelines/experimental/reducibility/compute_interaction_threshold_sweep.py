"""Interacting-pair counts across a range of off-diagonal thresholds.

Table S12's companion. compute_interaction_counts.py fixes the cut at 0.25 of each law's
largest off-diagonal Hessian entry; this reruns the same computation over a range of cuts so
the reported contrast can be shown not to depend on that choice.

Two thresholds are in play and they are not interchangeable. The Jacobian cut (0.15 of the
largest mean |df/dx|) selects WHICH inputs count as dependencies; the off-diagonal cut then
counts interacting pairs AMONG those inputs. Sweeping both at once would confound them, so
the Jacobian cut is held at the value used throughout and only the off-diagonal cut varies.

Normalisation matters here: off-diagonals are scored against each law's own largest
off-diagonal, not against the global Hessian maximum. Symbolic laws with divisions carry
huge diagonal curvature that would swamp their cross-terms under a global cut, while tanh
networks do not, so a global cut is not comparable across function classes.
"""
import sys, json, glob, os

sys.path.insert(0, "src/pipelines/experimental/reducibility")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, equinox as eqx      # noqa: E402
from _shared import _baseline                                                # noqa: E402

JAC_CUT = 0.15                                    # as used throughout
OFF_CUTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
OUT = "data/experimental/runs/reducibility/interactions_threshold_sweep.csv"

nodb = _baseline()
rows = []
paths = sorted(glob.glob("data/experimental/runs/reducibility/l21_hl4_local/"
                         "seed_*/models/*.json"))
print(f"{len(paths)} checkpoints")
for n, jp in enumerate(paths, 1):
    meta = json.loads(open(jp).read()); base = jp[:-5]
    ep, tp = base + ".eqx", base + "_train.npz"
    if not (os.path.exists(ep) and os.path.exists(tp)):
        continue
    mdl = eqx.tree_deserialise_leaves(ep, nodb.RHS(
        in_dim=int(meta["in_dim"]), hidden_dim=int(meta["hidden_dim"]),
        hidden_layers=int(meta["hidden_layers"]), activation=str(meta["activation"]),
        key=jax.random.PRNGKey(0)))
    tr = dict(np.load(tp)); y = (tr["ys"] - float(meta["y_mean"])) / float(meta["y_scale"])
    us = tr["us_scaled"]; msk = tr["real_mask"].astype(bool).reshape(-1)
    X = np.concatenate([y.reshape(-1)[msk][:, None],
                        us.reshape(-1, us.shape[-1])[msk]], axis=1)
    if X.size == 0:
        continue
    g = np.abs(np.asarray(jax.jit(jax.vmap(jax.grad(mdl)))(jnp.asarray(X)))).mean(axis=0)
    idx = np.where(g / max(g.max(), 1e-12) > JAC_CUT)[0]; k = len(idx)
    # same 600-point subsample and seed as the production script, so the 0.25 column
    # reproduces interactions_offnorm.csv rather than merely resembling it
    sel = np.random.default_rng(0).choice(len(X), min(600, len(X)), replace=False)
    H = np.abs(np.asarray(jax.jit(jax.vmap(jax.hessian(mdl)))(jnp.asarray(X[sel])))).mean(axis=0)
    sub = H[np.ix_(idx, idx)] if k else np.zeros((0, 0))
    off = sub[np.triu_indices(k, 1)] if k > 1 else np.array([])
    omax = off.max() if off.size else 0.0
    row = dict(marker=meta["marker"], seed=int(meta["seed"]), k=float(k))
    for c in OFF_CUTS:
        row[f"pairs_{c:g}"] = float((off > c * omax).sum()) if omax > 0 else 0.0
    rows.append(row)
    jax.clear_caches()
    if n % 20 == 0:
        print(f"  {n}/{len(paths)}")

d = pd.DataFrame(rows).drop_duplicates(["marker", "seed"])
d.to_csv(OUT, index=False)
print(f"wrote {OUT}: {d.marker.nunique()} markers x {d.seed.nunique()} seeds")
print(d[[f"pairs_{c:g}" for c in OFF_CUTS]].mean().round(2).to_string())
