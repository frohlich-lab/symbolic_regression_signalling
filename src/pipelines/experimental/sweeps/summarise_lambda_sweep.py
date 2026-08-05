"""Summarise the sparse Neural ODE lambda_jac sweep and apply the elbow rule (Fig. S3B).

The selection rule, fixed in the SI Methods: take the LARGEST lambda_jac whose mean
in-distribution validation R2 is within `--tolerance` of the best. Because a stronger
penalty yields a sparser model, this favours the sparsest network that is not measurably
worse, so the dependency counts reported downstream are conservative with respect to
sparsity.

Selection reads the `val` split only. The `test` split is the held-out highest-dose bins
and is reported here purely so the (weak) relationship between the two is visible -- in
this grid validation R2 is a poor predictor of held-out performance, which is why the
retained lambda should be read as a reasonable default rather than an optimum for
extrapolation.

One trap this script exists to avoid: `neural_ode_diffrax_metrics.csv` contains train,
val AND test rows. A bare `.max()` or `.mean()` over the file silently reports training
accuracy. Every aggregation below filters on `split` explicitly.

Usage:
    python summarise_lambda_sweep.py --sweep-dir <dir of lam_*/seed_*/> --output out.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

METRIC = "trajectory_r2_mean"


def parse_lambda(name: str) -> float | None:
    """`lam_3p0` -> 3.0, `lam_15p0` -> 15.0."""
    m = re.match(r"^lam_(\d+)(?:p(\d+))?$", name)
    if not m:
        return None
    whole, frac = m.group(1), m.group(2) or "0"
    return float(f"{whole}.{frac}")


def load(sweep_dir: str) -> pd.DataFrame:
    frames = []
    pattern = os.path.join(sweep_dir, "lam_*", "**", "neural_ode_diffrax_metrics.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(path, sweep_dir)
        lam = parse_lambda(rel.split(os.sep)[0])
        if lam is None:
            continue
        df = pd.read_csv(path)
        if "split" not in df.columns or METRIC not in df.columns:
            print(f"  skipping {path}: no 'split' or '{METRIC}' column")
            continue
        df["lambda_jac"] = lam
        frames.append(df)
    if not frames:
        raise SystemExit(f"no lam_*/**/neural_ode_diffrax_metrics.csv under {sweep_dir}")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Elbow width on mean validation R2 (SI Methods uses 0.05).")
    args = ap.parse_args(argv)

    fits = load(args.sweep_dir)

    rows = []
    for lam, group in fits.groupby("lambda_jac"):
        val = group[group.split == "val"]
        test = group[group.split == "test"]
        rows.append({
            "lambda_jac": lam,
            "n_contexts": group.marker.nunique(),
            "n_seeds": group.seed.nunique(),
            "mean_val_r2": val[METRIC].mean(),
            "median_val_r2": val[METRIC].median(),
            "mean_heldout_r2": test[METRIC].mean(),
            "median_heldout_r2": test[METRIC].median(),
        })
    table = pd.DataFrame(rows).sort_values("lambda_jac").reset_index(drop=True)

    best = table.mean_val_r2.max()
    within = table[table.mean_val_r2 >= best - args.tolerance]
    retained = within.lambda_jac.max()
    table["within_tolerance"] = table.mean_val_r2 >= best - args.tolerance
    table["retained"] = table.lambda_jac == retained

    table.round(6).to_csv(args.output, index=False)

    print("=== sparse Neural ODE lambda_jac sweep (L21 on the input Jacobian) ===")
    print(table.round(3).to_string(index=False))
    print(f"\n  best mean validation R2 : {best:.3f}")
    print(f"  elbow tolerance         : {args.tolerance:g}")
    print(f"  RETAINED lambda_jac     : {retained:g}  "
          f"(largest within tolerance of the best)")

    # Along this one-dimensional ladder both curves fall monotonically with lambda, so
    # validation and held-out R2 agree almost perfectly and selecting on validation
    # costs nothing. That does NOT generalise to the architecture grid, where validation
    # R2 tracks held-out performance only weakly (Fig. S3C) -- the two are separate
    # searches and only this one is safe to read off validation.
    if table.mean_heldout_r2.notna().sum() >= 3:
        rho = table.mean_val_r2.corr(table.mean_heldout_r2, method="spearman")
        print(f"  Spearman(mean val R2, mean held-out R2) along the ladder = {rho:.2f}")

    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
