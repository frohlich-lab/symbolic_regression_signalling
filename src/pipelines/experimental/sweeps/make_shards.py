"""Split the marker datasets into one dataset pair per context, for job-array sharding.

Writes `<sweep>/data/shards/<slug>_{per_minute,snapshot}.csv` plus a slug -> name table,
because context names contain spaces and parentheses that are unsafe in paths.

Two modes:

  --markers A B C     select these contexts straight out of the processed datasets.
                      Used for the configuration sweep, whose development set is a fixed
                      six contexts.
  --all               every context in the processed datasets. Used for the frozen-config
                      run, so the reported numbers come from one identical run rather
                      than the sweep's development contexts stitched to a later batch.

Usage:
    python make_shards.py <sweep_dir> --markers AKT3 ALPK2 DYRK2 ERBB2 PIP5K3 PTPN7
    python make_shards.py <sweep_dir> --all
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

DEFAULT_SOURCE = "data/experimental/processed/functional_groups"


def slugify(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", str(name))).strip("_")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", help="Sweep root; shards land in <sweep_dir>/data/shards.")
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE,
                    help="Directory holding markers_per_minute_fit.csv and "
                         "markers_fit_snapshot.csv.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--markers", nargs="+", help="Context names to shard.")
    group.add_argument("--all", action="store_true", help="Shard every context.")
    args = ap.parse_args(argv)

    sweep = args.sweep_dir.rstrip("/")
    shard_dir = f"{sweep}/data/shards"
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(f"{sweep}/arms", exist_ok=True)

    per_minute = pd.read_csv(f"{args.source_dir}/markers_per_minute_fit.csv")
    snapshot = pd.read_csv(f"{args.source_dir}/markers_fit_snapshot.csv")

    available = set(per_minute["marker"].dropna().astype(str).unique())
    if args.all:
        selected = sorted(available)
    else:
        selected = list(args.markers)
        unknown = [m for m in selected if m not in available]
        if unknown:
            raise SystemExit(
                f"context(s) not in {args.source_dir}: {unknown}\n"
                f"available: {sorted(available)}"
            )

    rows = []
    for marker in selected:
        slug = slugify(marker)
        pm = per_minute[per_minute["marker"].astype(str) == marker]
        sn = snapshot[snapshot["marker"].astype(str) == marker]
        pm.to_csv(f"{shard_dir}/{slug}_per_minute.csv", index=False)
        sn.to_csv(f"{shard_dir}/{slug}_snapshot.csv", index=False)
        rows.append((slug, marker, len(pm), len(sn)))
        print(f"{slug:14s} <- {marker!r:18s} per_minute={len(pm):6d} snapshot={len(sn):5d}")

    with open(f"{sweep}/arms/markers.tsv", "w") as fh:
        for slug, name, _, _ in rows:
            fh.write(f"{slug}\t{name}\n")

    print(f"\n{len(rows)} context shards -> {shard_dir}")
    print(f"slug table -> {sweep}/arms/markers.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
