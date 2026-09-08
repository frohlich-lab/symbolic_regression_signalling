"""Tables S12 and S13: sensitivity of the two complexity contrasts to their thresholds.

Both tables answer the same question at two levels of the same construction. S12 varies the
Jacobian cut that decides which inputs count as dependencies; S13 holds that cut at 0.15 and
varies the off-diagonal Hessian cut that decides which pairs of those inputs count as
interacting. Holding one while sweeping the other is the point -- sweeping both at once
confounds them, and the S13 caption says so.

Written as a rule output rather than by hand because the partition (which contexts the
network fits, which SR solves) has to be the SAME one the printed panel uses. It is read
from the Fig. 4 panel dump for exactly that reason, the way Table S8 already does: a
hand-maintained copy is how a table and the figure it sits beside come to disagree.

Two P-values per row, because the two are answering different questions and the captions
quote both:
  * gated   -- the contexts the network fits, accuracy-residualised exact permutation, the
               same test the Fig. 4 brackets print
  * all 40  -- every context, two-sided Mann-Whitney, which needs no accuracy covariate
               because it is not conditioned on the network generalising
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src/pipelines/experimental/sr_pipeline/figures")

# The residualised permutation lives with the figure that prints it, so the table cannot
# drift onto a different test than the brackets.
from plot_parsimony_tradeoff import _perm_p          # noqa: E402

JAC_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
OFF_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
JAC_HELD_AT = 0.15          # S13 holds the dependency cut here while sweeping pairs
HESS_SUBSAMPLE = 600        # rows the Hessian is averaged over, as in the panel's own CSV


def per_checkpoint(nn_dir: Path) -> pd.DataFrame:
    """Mean |dF/dx| and mean |d2F/dxi dxj| per (marker, seed), reduced to counts.

    One pass over the checkpoints produces every threshold in both sweeps: the Jacobian and
    Hessian do not depend on the cuts, only the counting does. Recomputing them per
    threshold would multiply an already slow step by fourteen.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from _shared import _baseline

    nodb = _baseline()
    rows = []
    for jp in sorted(glob.glob(str(nn_dir / "seed_*" / "models" / "*.json"))):
        meta = json.loads(Path(jp).read_text())
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
        msk = tr["real_mask"].astype(bool).reshape(-1)
        X = np.concatenate(
            [y.reshape(-1)[msk][:, None], us.reshape(-1, us.shape[-1])[msk]], axis=1)
        if X.size == 0:
            continue
        g = np.abs(np.asarray(jax.jit(jax.vmap(jax.grad(mdl)))(jnp.asarray(X)))).mean(axis=0)
        # A fresh generator per checkpoint, matching compute_interaction_counts.py: one
        # generator reused across the loop draws different rows per marker and the pair
        # counts stop matching the deposited interaction CSV.
        sel_rows = np.random.default_rng(0).choice(
            len(X), min(HESS_SUBSAMPLE, len(X)), replace=False)
        H = np.abs(np.asarray(
            jax.jit(jax.vmap(jax.hessian(mdl)))(jnp.asarray(X[sel_rows])))).mean(axis=0)

        row = {"marker": meta["marker"], "seed": int(meta["seed"])}
        gmax = max(g.max(), 1e-12)
        for t in JAC_THRESHOLDS:
            row[f"k_{t}"] = float((g / gmax > t).sum())
        # Pair counts are scored among the inputs the dependency criterion retains, so the
        # index set is fixed at JAC_HELD_AT and only the off-diagonal cut moves.
        idx = np.where(g / gmax > JAC_HELD_AT)[0]
        sub = H[np.ix_(idx, idx)] if len(idx) else np.zeros((0, 0))
        off = sub[np.triu_indices(len(idx), 1)] if len(idx) > 1 else np.array([])
        omax = off.max() if off.size else 0.0
        for t in OFF_THRESHOLDS:
            row[f"pairs_{t}"] = float((off > t * omax).sum()) if omax > 0 else 0.0
        row["k"] = float(len(idx))
        rows.append(row)
        jax.clear_caches()
    if not rows:
        raise SystemExit(f"no checkpoints under {nn_dir}")
    return pd.DataFrame(rows).drop_duplicates(["marker", "seed"])


def contrast(per_marker: pd.DataFrame, col: str, threshold: float) -> dict:
    """One table row: gated and all-40 group means with the P-value each caption quotes.

    Returned unrounded. S13's ratio is formed from these full-precision means and only then
    rounded -- dividing the two already-rounded means shifts it by 0.01 on two rows.
    """
    nn_ok = per_marker.nn_r2 > threshold
    py_ok = per_marker.py_r2 > threshold
    gated = per_marker[nn_ok]
    g_fail = (gated.py_r2 <= threshold).values
    a_fail = (~py_ok).values
    return {
        "Gated failed": float(gated.loc[g_fail, col].mean()),
        "Gated solved": float(gated.loc[~g_fail, col].mean()),
        "Gated P": float(_perm_p(gated[col].values, g_fail, gated.nn_r2.values)),
        "All-40 failed": float(per_marker.loc[a_fail, col].mean()),
        "All-40 solved": float(per_marker.loc[~a_fail, col].mean()),
        "All-40 P": float(stats.mannwhitneyu(
            per_marker.loc[a_fail, col].values,
            per_marker.loc[~a_fail, col].values).pvalue),
    }


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Two decimals on counts and ratios, four on P-values, as the captions print them."""
    out = df.copy()
    for c in out.columns:
        if c == "Threshold":
            continue
        out[c] = out[c].round(4 if c.endswith(" P") else 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nn-dir", type=Path, required=True,
                    help="Neural ODE run whose checkpoints the panel is drawn from.")
    ap.add_argument("--panel-inputs", type=Path, required=True,
                    help="Fig. 4 panel dump: supplies the SAME partition the figure prints.")
    ap.add_argument("--r2-threshold", type=float, default=0.6)
    ap.add_argument("--per-fit-csv", type=Path, required=True,
                    help="Per-(marker, seed) counts, deposited so both tables are auditable.")
    ap.add_argument("--table-s12", type=Path, required=True)
    ap.add_argument("--table-s13", type=Path, required=True)
    args = ap.parse_args()

    panel = pd.read_csv(args.panel_inputs)[["marker", "nn_r2", "py_r2"]]
    per_fit = per_checkpoint(args.nn_dir)
    args.per_fit_csv.parent.mkdir(parents=True, exist_ok=True)
    per_fit.to_csv(args.per_fit_csv, index=False)

    # Averaged over seeds first, then contrasted: the context is the unit of analysis, as
    # Methods states, so a marker with three seeds must not count three times.
    per_marker = per_fit.groupby("marker").mean(numeric_only=True).reset_index()
    per_marker = per_marker.merge(panel, on="marker")
    n_missing = per_fit.marker.nunique() - len(per_marker)
    if n_missing:
        print(f"  {n_missing} marker(s) had checkpoints but no panel row; dropped")

    s12 = pd.DataFrame([
        {"Threshold": t, **contrast(per_marker, f"k_{t}", args.r2_threshold)}
        for t in JAC_THRESHOLDS
    ])
    s13 = pd.DataFrame([
        {"Threshold": t, **contrast(per_marker, f"pairs_{t}", args.r2_threshold)}
        for t in OFF_THRESHOLDS
    ])
    # Ratio is the statistic the S13 caption leads with, so it is stored, not left to the
    # reader to divide two rounded means -- and it is formed before rounding.
    s13.insert(3, "Ratio", s13["Gated failed"] / s13["Gated solved"])

    for path, df in ((args.table_s12, rounded(s12)), (args.table_s13, rounded(s13))):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"  wrote {path}")
    print(f"  gated n={int((per_marker.nn_r2 > args.r2_threshold).sum())}  "
          f"all n={len(per_marker)}  checkpoints={len(per_fit)}")


if __name__ == "__main__":
    main()
