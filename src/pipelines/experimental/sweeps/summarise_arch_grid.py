"""Summarise the sparse Neural ODE architecture grid (Fig. S3C / Table S9).

Companion to summarise_lambda_sweep.py, and it selects the same way: on the `val`
split only. The held-out highest-dose bins play no part in choosing an
architecture, which is the whole point of running this grid separately from the
reported results.

Unlike the lambda ladder there is no elbow rule here -- the grid is scored on mean
in-distribution validation R2 and the argmax is retained. Along the lambda ladder
validation R2 tracks held-out R2 closely enough that selecting on validation costs
nothing; across this grid it does not (that weak relationship IS Fig. S3C), so the
retained architecture should be read as a reasonable default rather than an optimum
for extrapolation.

Cell directories are named `hd<width>_hl<layers>_lr<rate>_<activation>` with `.`
written as `p`, matching the `lam_3p0` convention of the lambda sweep.

Usage:
    python summarise_arch_grid.py --grid-dir <dir of hd*_hl*_lr*_*/> --output out.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import pandas as pd

METRIC = "trajectory_r2_mean"

CELL = re.compile(r"^hd(?P<hd>\d+)_hl(?P<hl>\d+)_lr(?P<lr>[\dp]+)_(?P<act>[a-z]+)$")


def parse_cell(name: str) -> dict | None:
    """`hd64_hl4_lr0p003_tanh` -> the four hyperparameter values."""
    m = CELL.match(name)
    if not m:
        return None
    return {
        "cfg": name,
        "hidden_dim": int(m.group("hd")),
        "hidden_layers": int(m.group("hl")),
        "lr": float(m.group("lr").replace("p", ".")),
        "activation": m.group("act"),
    }


def load(grid_dir: str) -> pd.DataFrame:
    frames = []
    pattern = os.path.join(grid_dir, "*", "**", "neural_ode_diffrax_metrics.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        rel = os.path.relpath(path, grid_dir)
        cell = parse_cell(rel.split(os.sep)[0])
        if cell is None:
            continue
        df = pd.read_csv(path)
        # Same trap as the lambda sweep: this file holds train, val AND test rows, so a
        # bare mean over it reports training accuracy. Every aggregation below filters.
        if "split" not in df.columns or METRIC not in df.columns:
            print(f"  skipping {path}: no 'split' or '{METRIC}' column")
            continue
        for key, value in cell.items():
            df[key] = value
        frames.append(df)
    if not frames:
        raise SystemExit(f"no hd*_hl*_lr*_*/**/neural_ode_diffrax_metrics.csv "
                         f"under {grid_dir}")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    fits = load(args.grid_dir)

    rows = []
    for cfg, group in fits.groupby("cfg"):
        val = group[group.split == "val"]
        first = group.iloc[0]
        rows.append({
            "cfg": cfg,
            "hidden_dim": int(first.hidden_dim),
            "hidden_layers": int(first.hidden_layers),
            "lr": float(first.lr),
            "activation": str(first.activation),
            "mean_val_r2": val[METRIC].mean(),
            "median_val_r2": val[METRIC].median(),
            # Cells differ in how many contexts trained to completion, and a cell that
            # dropped contexts is not comparable to one that kept them all; Table S9
            # reports the count so that is visible rather than averaged away.
            "n_markers": val.marker.nunique(),
        })
    table = (pd.DataFrame(rows)
             .sort_values("mean_val_r2", ascending=False)
             .reset_index(drop=True))

    retained = table.iloc[0]
    table["retained"] = table.cfg == retained.cfg
    table.round(6).to_csv(args.output, index=False)

    print("=== sparse Neural ODE architecture grid (validation R2 only) ===")
    print(table.round(3).to_string(index=False))
    print(f"\n  RETAINED : hidden_dim={retained.hidden_dim} "
          f"hidden_layers={retained.hidden_layers} lr={retained.lr:g} "
          f"activation={retained.activation}  "
          f"(mean val R2 {retained.mean_val_r2:.3f})")
    print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
