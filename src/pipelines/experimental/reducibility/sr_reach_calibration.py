"""Map the reach of symbolic regression on ground truth: closure dimension x coupling.

Why
---
The biological result says symbolic recovery becomes unreliable as the closure dimension of
the dynamics rises, with recovery near zero above about four measured variables. On real data
that rests on ten to seventeen contexts, and a ceiling estimated from ten points is an
observation about one dataset, not a property of the method.

The ceiling is measurable directly. Hold the real input trajectories fixed — same markers,
same GFP bins, same measured timepoints, same top-GFP held-out split — and replace only the
state with one simulated from a rate law whose closure dimension is KNOWN. Then ask how often
symbolic regression recovers it. Ground truth, with as many replicates as we care to run.

The design turns on a second axis the existing PR calibration does not vary. Its ground-truth
law is additive in the drivers,

    dy/dt = -a y + sum_i b_i tanh(u_i)                                  (SEPARABLE)

so its interaction density is ~0. The real contexts have saturated density (~1.0: every driver
couples to every other), and the interaction terms a closed form must carry grow as the square
of the dimension only in that coupled regime. So we also simulate a coupled law of the
competitive-saturation form natural to enzyme kinetics,

    dy/dt = -a y + (sum_i b_i u_i) / (1 + sum_i c_i u_i)                (FULLY COUPLED)

whose cross-curvature is non-zero for every driver pair by construction.

The falsifiable prediction: if symbolic regression's reach is set by the number of interaction
terms rather than the number of variables, recovery should persist to high dimension on
additive laws and collapse at low dimension on coupled laws — with the coupled ceiling landing
near the ~4-variable ceiling seen in the biology. If both regimes collapse at the same
dimension, reach is limited by arity, the k^2 account is wrong, and the biological ceiling needs
a different explanation.

Deliberately free of jax and of the reducibility bundle machinery: it needs only the
per-minute CSV, so it runs in the same environment as PySR.

Usage:
    python sr_reach_calibration.py --marker EGFR --k-true 3 --coupling coupled \
        --sr-seed 42 --output-dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATS = ["GFP", "p-ERK1-2_min", "p-MEK1-2_fit", "p-MEK1-2_min", "p-RAF_fit",
         "p-p90RSK_fit", "p-MAPKAPK2_fit", "p-PDK1_fit", "p-MKK3-6_fit"]
STATE = "p_ERK1_2"
MEASURED = [0.0, 5.0, 10.0, 15.0, 30.0, 60.0]
HELDOUT_FRAC = 0.2

# Same operator basis and budget as the frozen production configuration, so the ceiling this
# measures is the one that applies to the biological analysis.
FROZEN = dict(niterations=1400, populations=30, population_size=30, maxsize=26,
              parsimony=0.8, binary_operators=["+", "-", "*", "/"], unary_operators=[])


def rhs_factory(driver_idx, w, c, decay, coupling):
    def rhs(state, u):
        if not driver_idx.size:
            return -decay * state
        ud = u[driver_idx]
        if coupling == "additive":
            drive = float(np.sum(w * np.tanh(ud)))
        else:
            den = 1.0 + float(np.sum(c * ud))
            if abs(den) < 0.25:
                den = 0.25 * (1.0 if den >= 0 else -1.0)
            drive = float(np.sum(w * ud)) / den
        return -decay * state + drive
    return rhs


def simulate_bin(t, U, rhs, substeps=10):
    """Sub-stepped Euler along one bin's own per-minute grid. Transparent by design: the law
    is smooth and mildly damped, and an auditable integrator keeps the ground truth checkable."""
    y = np.zeros_like(t)
    dy = np.zeros_like(t)
    state = 0.0
    y[0], dy[0] = state, rhs(state, U[0])
    for j in range(len(t) - 1):
        h = (t[j + 1] - t[j]) / substeps
        for s in range(substeps):
            frac = s / substeps
            u_row = U[j] + frac * (U[j + 1] - U[j])
            state = state + h * rhs(state, u_row)
        y[j + 1] = state
        dy[j + 1] = rhs(state, U[j + 1])
    return y, dy


def density_of(rhs, X, driver_idx):
    """Realised fraction of driver pairs with non-negligible cross-curvature, by finite
    differences on the ground-truth law — the same construct measured on the networks."""
    if driver_idx.size < 2:
        return np.nan
    x0 = X.mean(axis=0)
    h = 0.05 * (X.std(axis=0) + 1e-9)
    idx = 1 + driver_idx                       # column 0 is the state
    off = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ia, ib = idx[a], idx[b]
            v = []
            for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                x = x0.copy(); x[ia] += sa * h[ia]; x[ib] += sb * h[ib]
                v.append(rhs(x[0], x[1:]))
            off.append(abs((v[0] - v[1] - v[2] + v[3]) / (4 * h[ia] * h[ib])))
    off = np.asarray(off)
    return float((off > 0.25 * off.max()).mean()) if off.size and off.max() > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", required=True)
    ap.add_argument("--k-true", type=int, required=True)
    ap.add_argument("--coupling", choices=["additive", "coupled"], required=True)
    ap.add_argument("--sr-seed", type=int, default=42)
    ap.add_argument("--law-seed", type=int, default=0)
    ap.add_argument("--decay", type=float, default=0.15)
    ap.add_argument("--dataset", default="data/experimental/processed/functional_groups/"
                                         "markers_per_minute_fit.csv")
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.dataset)
    g = df[df.marker == a.marker].dropna(subset=FEATS).copy()
    if g.empty:
        raise SystemExit(f"no rows for {a.marker}")
    if a.k_true > len(FEATS):
        raise SystemExit(f"k_true {a.k_true} exceeds {len(FEATS)} inputs")

    # top_gfp_bins split, reproduced from the bin ordering
    bins = np.sort(g.GFP_bin.unique())
    ncut = max(1, int(round(HELDOUT_FRAC * len(bins))))
    held = set(bins[-ncut:])

    # scale inputs on the training bins only, so the held-out region stays genuinely unseen
    tr_rows = g[~g.GFP_bin.isin(held)]
    mu, sd = tr_rows[FEATS].mean().values, tr_rows[FEATS].std().replace(0, 1).values

    rng = np.random.default_rng(a.law_seed)
    driver_idx = (rng.choice(len(FEATS), size=a.k_true, replace=False) if a.k_true
                  else np.array([], dtype=int))
    w = rng.uniform(0.5, 1.5, size=a.k_true) * rng.choice([-1.0, 1.0], size=a.k_true)
    c = rng.uniform(0.2, 0.8, size=a.k_true)
    rhs = rhs_factory(driver_idx, w, c, a.decay, a.coupling)

    rows = {"train": [], "test": []}
    for b, gb in g.groupby("GFP_bin"):
        gb = gb.sort_values("timepoint")
        t = gb.timepoint.values.astype(float)
        U = (gb[FEATS].values - mu) / sd
        y, dy = simulate_bin(t, U, rhs)
        keep = np.isin(t, MEASURED)                      # score at real measurements only
        X = np.column_stack([y[keep], U[keep]])
        rows["test" if b in held else "train"].append((X, dy[keep]))
    if not rows["train"] or not rows["test"]:
        raise SystemExit("empty split")
    Xtr = np.vstack([x for x, _ in rows["train"]]); ytr = np.concatenate([d for _, d in rows["train"]])
    Xte = np.vstack([x for x, _ in rows["test"]]);  yte = np.concatenate([d for _, d in rows["test"]])

    dens = density_of(rhs, Xtr, driver_idx)
    names = [STATE] + [f.replace("-", "_").replace("_fit", "") for f in FEATS]

    from pysr import PySRRegressor
    m = PySRRegressor(deterministic=True, parallelism="serial", random_state=a.sr_seed,
                      progress=False, verbosity=0, temp_equation_file=True, **FROZEN)
    m.fit(Xtr, ytr, variable_names=names)

    def r2(yy, pp):
        yy = np.asarray(yy, float)
        pp = np.nan_to_num(np.asarray(pp, float), nan=0.0, posinf=0.0, neginf=0.0)
        v = float(np.var(yy))
        return float(1.0 - np.mean((yy - pp) ** 2) / v) if v > 0 else np.nan

    eq = m.equations_
    best = eq.iloc[int(eq.loss.idxmin())]
    row = dict(marker=a.marker, k_true=a.k_true, closure_dim=a.k_true + 1,
               coupling=a.coupling, density=dens, sr_seed=a.sr_seed, law_seed=a.law_seed,
               train_r2=r2(ytr, m.predict(Xtr)), test_r2=r2(yte, m.predict(Xte)),
               complexity=int(best.complexity), equation=str(best.equation),
               n_train=len(ytr), n_test=len(yte),
               drivers=";".join(str(i) for i in driver_idx))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    key = (f"{''.join(ch if ch.isalnum() else '_' for ch in a.marker)}"
           f"_k{a.k_true}_{a.coupling}_sr{a.sr_seed}_law{a.law_seed}")
    pd.DataFrame([row]).to_csv(out / f"{key}.csv", index=False)
    print(f"{a.marker} k_true={a.k_true} ({a.coupling}, density={dens:.2f}) "
          f"train R2={row['train_r2']:.3f} heldout R2={row['test_r2']:.3f} "
          f"complexity={row['complexity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
