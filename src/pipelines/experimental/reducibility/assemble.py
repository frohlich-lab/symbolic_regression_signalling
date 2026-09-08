"""Step 1 — assemble one per-(marker, seed, variant) table for every downstream test.

Joins three things that currently live apart, so nothing downstream has to
re-derive them:

  * PySR held-out performance, PER SEED  (pysr_ood_final/metrics/integrated_r2_all40.csv)
  * sparse-Neural-ODE held-out performance, PER SEED AND VARIANT
        (sparse_neural_ode/<variant>/seed_<s>/neural_ode_diffrax_metrics.csv, split == "test")
  * participation ratio, PER SEED AND VARIANT
        (sparse_neural_ode/participation_ratio_variants.csv)

Two things this fixes relative to the manuscript as drafted:

1. *Objective parity.* The SR/Neural-ODE comparison in the text pairs numbers
   that were produced by different objectives (SR is fit on derivative MSE, the
   Neural ODE on trajectory MSE). Both sources happen to also record an
   integrated / rolled-out held-out R^2, so this script carries a single
   `matched` metric for both sides — `ode_r2` for PySR, `ode_integ_r2_median`
   for the Neural ODE — which are the same quantity: R^2 of the integrated
   trajectory against the held-out p-ERK measurements. Every comparison
   downstream uses the matched pair; the native metrics are kept alongside so
   the effect of matching is auditable rather than assumed.

2. *Per-seed everything.* No best-of-seed reduction happens here. Reductions
   are the caller's choice and are made explicit in `quadrants.py`.

Usage:
    python assemble.py [--output-dir data/experimental/runs/reducibility]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    NDE_ROOT,
    PYSR_ROOT,
    SUCCESS_THRESHOLD,
    VARIANTS,
    default_outdir,
    write_csv,
)

PYSR_SEEDS = (42, 43, 44)


def load_pysr() -> pd.DataFrame:
    """Per-(marker, seed) PySR held-out performance, with the control flag."""
    all40 = pd.read_csv(PYSR_ROOT / "metrics" / "integrated_r2_all40.csv")
    keep = ["marker", "seed", "ode_r2", "ode_r2_train", "dt_r2"]
    out = all40[keep].rename(columns={
        "ode_r2": "pysr_ode_r2",            # held-out, integrated  <- matched metric
        "ode_r2_train": "pysr_ode_r2_train",
        "dt_r2": "pysr_dt_r2",              # held-out, derivative  <- native objective
    })

    per_fit = PYSR_ROOT / "metrics" / "integrated_r2_per_fit.csv"
    if per_fit.exists():
        pf = pd.read_csv(per_fit)
        cols = ["marker", "seed"] + [c for c in ("is_control", "formula") if c in pf.columns]
        out = out.merge(pf[cols].drop_duplicates(["marker", "seed"]),
                        on=["marker", "seed"], how="left")
    if "is_control" not in out.columns:
        out["is_control"] = np.nan
    return out


def load_node() -> pd.DataFrame:
    """Per-(variant, marker, seed) Neural-ODE held-out performance."""
    frames = []
    for variant in VARIANTS:
        for seed_dir in sorted((NDE_ROOT / variant).glob("seed_*")):
            path = seed_dir / "neural_ode_diffrax_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df = df[df["split"] == "test"].copy()
            if df.empty:
                continue
            df["variant"] = variant
            df["seed"] = int(seed_dir.name.split("_")[1])
            frames.append(df[[
                "variant", "seed", "marker",
                "ode_integ_r2_median",   # held-out, integrated <- matched metric
                "trajectory_r2_mean",    # held-out, rollout    <- native objective
                "dt_r2", "n_bins",
            ]])
    if not frames:
        raise SystemExit(f"no split=='test' metrics under {NDE_ROOT}/<variant>/seed_*/")
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={
        "ode_integ_r2_median": "node_ode_r2",
        "trajectory_r2_mean": "node_traj_r2",
        "dt_r2": "node_dt_r2",
    })


def load_pr() -> pd.DataFrame:
    path = NDE_ROOT / "participation_ratio_variants.csv"
    if not path.exists():
        raise SystemExit(
            f"{path} missing — run figures/compute_participation_ratios.py first")
    pr = pd.read_csv(path)
    return pr.rename(columns={"pr": "node_pr"})[
        ["variant", "seed", "marker", "node_pr", "n_inputs"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir())
    args = ap.parse_args(argv)

    pysr, node, pr = load_pysr(), load_node(), load_pr()

    master = (node
              .merge(pr, on=["variant", "seed", "marker"], how="outer")
              .merge(pysr, on=["marker", "seed"], how="left"))

    # Matched-objective success labels: same metric, same threshold, both sides.
    master["pysr_success"] = master["pysr_ode_r2"] >= SUCCESS_THRESHOLD
    master["node_success"] = master["node_ode_r2"] >= SUCCESS_THRESHOLD
    # Native-objective labels, for the sensitivity check in quadrants.py.
    master["pysr_success_native"] = master["pysr_dt_r2"] >= SUCCESS_THRESHOLD
    master["node_success_native"] = master["node_traj_r2"] >= SUCCESS_THRESHOLD

    master = master.sort_values(["variant", "marker", "seed"]).reset_index(drop=True)
    write_csv(master, args.output_dir / "reducibility_master.csv")

    # Coverage report — silent joins are how phantom n's get into manuscripts.
    print("\n--- coverage ---")
    print(f"markers            : {master['marker'].nunique()}")
    print(f"variants           : {sorted(master['variant'].dropna().unique())}")
    print(f"seeds              : {sorted(master['seed'].dropna().unique())}")
    for col in ("node_ode_r2", "node_pr", "pysr_ode_r2"):
        n_missing = int(master[col].isna().sum())
        print(f"missing {col:<18}: {n_missing} / {len(master)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
