"""Participation ratio of the input Jacobian over the ten model inputs (as in Methods).

Walks every sparse-Neural-ODE variant (l1 / l21_lam3 / pathreg) x seed x marker
checkpoint and writes one tidy CSV. This is the "effective dependency count"
column of Table S8, so it reads the saved models rather than any metrics file --
the variant rules must have run with --save-models.

Usage:
    python compute_participation_ratios.py --nde-root <dir of l1/l21_lam3/pathreg> \
        --output participation_ratio_variants.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

# The repository's `src` on sys.path, derived from this file rather than hardcoded:
# figures/ -> sr_pipeline/ -> experimental/ -> pipelines/ -> src/
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from pipelines.experimental.sr_pipeline.neural_ode_diffrax_baseline import RHS  # noqa: E402

VARIANTS = ("l1", "l21_lam3", "pathreg")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nde-root", type=Path, required=True,
                    help="Directory holding the l1/ l21_lam3/ pathreg/ variant trees.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for variant in VARIANTS:
        for seed_dir in sorted((args.nde_root / variant).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            mdir = seed_dir / "models"
            for jp in sorted(mdir.glob("*.json")):
                safe = jp.stem
                ep, tp = mdir / f"{safe}.eqx", mdir / f"{safe}_train.npz"
                if not (ep.exists() and tp.exists()):
                    continue
                meta = json.loads(jp.read_text())
                model = eqx.tree_deserialise_leaves(str(ep), RHS(
                    in_dim=int(meta["in_dim"]), hidden_dim=int(meta["hidden_dim"]),
                    hidden_layers=int(meta["hidden_layers"]),
                    activation=str(meta["activation"]), key=jax.random.PRNGKey(0)))
                tr = dict(np.load(tp))
                y = (tr["ys"] - float(meta["y_mean"])) / float(meta["y_scale"])
                us = tr["us_scaled"]
                m = tr["real_mask"].astype(bool).reshape(-1)
                X = np.concatenate([y.reshape(-1)[m][:, None],
                                    us.reshape(-1, us.shape[-1])[m]], axis=1)
                if X.size == 0:
                    continue
                jac = np.asarray(jax.jit(jax.vmap(jax.grad(model)))(jnp.asarray(X)))
                names = ["p-ERK1-2"] + list(meta["exo_cols"])
                a = np.abs(jac).mean(axis=0)
                pr = float(a.sum() ** 2 / max((a ** 2).sum(), 1e-12))
                rows.append({"variant": variant, "seed": seed,
                             "marker": str(meta["marker"]),
                             "pr": pr,
                             "n_inputs": len(names)})
                print(variant, seed, meta["marker"], round(pr, 3), flush=True)

    if not rows:
        raise SystemExit(f"no {'/'.join(VARIANTS)} seed_*/models/*.json under "
                         f"{args.nde_root}; the variant rules need --save-models")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print("wrote", args.output, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
