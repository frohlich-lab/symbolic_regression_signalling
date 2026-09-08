"""Re-run the Fig. S1 loss-penalty ablation on the OOD split, sharded by context.

`pysr_custom_loss_ablation.py` drives run_markers.py once per loss condition over
all 40 contexts in a single process, which is ~40 sequential PySR searches per
condition. Here each (condition, context) pair is its own run_markers invocation
against a one-context dataset shard, so the 160 fits fan out over a process pool.
The shards come from sweeps/make_shards.py --all, the same mechanism the frozen
`pysr_ood_final` run used on SLURM.

Everything else matches the manuscript's selected configuration: 1400 iterations,
30 populations of 30, maximum complexity 26, parsimony 0.8, binary operators only,
top-GFP-bin split, seed 42.

Resumable: a (condition, context) pair whose integration metrics already exist is
skipped, so the pool can be killed and relaunched.

    python run_custom_loss_ablation_ood.py --workers 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# Mirrors CONDITIONS in pysr_custom_loss_ablation.py; kept here so a change to
# the ablation's condition set is a deliberate edit in both places rather than a
# silent divergence in what Fig. S1 shows.
CONDITIONS = {
    "normal": [],
    "no_linear_stability": ["--disable-linear-stability-penalty"],
    "no_inverse_stability": ["--disable-inverse-stability-penalty"],
    "no_stability_penalties": ["--disable-linear-stability-penalty",
                               "--disable-inverse-stability-penalty"],
}

SELECTED_CONFIG = ["--max-iterations", "1400", "--populations", "30",
                   "--population-size", "30", "--max-size", "26",
                   "--parsimony", "0.8"]

MEASURED = ["0", "5", "10", "15", "30", "60"]


def metrics_path(out_root: Path, condition: str, slug: str) -> Path:
    return (out_root / "fits" / condition / slug / "reports" / "seed_42" /
            "metrics" / "marker_integration_metrics_per_minute.csv")


def run_one(job: tuple[str, str, str, str]) -> tuple[str, str, float, str]:
    condition, slug, shard_dir, out_root_s = job
    out_root = Path(out_root_s)
    out_dir = out_root / "fits" / condition / slug
    log = out_root / "logs" / f"{condition}__{slug}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # PySR runs deterministic/serial, so each fit is one core. Pin the numeric
    # stacks to one thread each or ten concurrent fits oversubscribe the box.
    env.update({"JULIA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})

    cmd = [
        sys.executable,
        str(REPO / "src/pipelines/experimental/sr_pipeline/run_markers.py"),
        "--dataset", f"{shard_dir}/{slug}_snapshot.csv",
        "--per-minute-dataset", f"{shard_dir}/{slug}_per_minute.csv",
        "--output-dir", str(out_dir),
        "--group-definitions-csv",
        str(REPO / "data/experimental/processed/functional_groups.csv"),
        "--models", "pysr",
        "--random-state", "42", "--seeds", "42",
        "--test-size", "0.2", "--test-split-policy", "top_gfp_bins",
        "--measured-timepoints", *MEASURED,
        *SELECTED_CONFIG,
        *CONDITIONS[condition],
    ]

    t0 = time.time()
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              env=env, cwd=str(REPO))
    dt = time.time() - t0
    ok = "ok" if (proc.returncode == 0 and
                  metrics_path(out_root, condition, slug).exists()) else \
        f"FAILED(rc={proc.returncode})"
    return condition, slug, dt, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path,
                    default=REPO / "data/experimental/runs/custom_loss_ablation_ood")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    args = ap.parse_args()

    shard_dir = args.out_root / "data" / "shards"
    slugs = sorted({p.name.replace("_per_minute.csv", "")
                    for p in shard_dir.glob("*_per_minute.csv")})
    if not slugs:
        raise SystemExit(f"no shards in {shard_dir}; run make_shards.py --all first")

    jobs, skipped = [], 0
    for condition in args.conditions:
        for slug in slugs:
            if metrics_path(args.out_root, condition, slug).exists():
                skipped += 1
                continue
            jobs.append((condition, slug, str(shard_dir), str(args.out_root)))

    print(f"{len(slugs)} contexts x {len(args.conditions)} conditions = "
          f"{len(slugs) * len(args.conditions)} fits; {skipped} already done, "
          f"{len(jobs)} to run on {args.workers} workers", flush=True)

    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            condition, slug, dt, ok = fut.result()
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (len(jobs) - done) / 60
            print(f"[{done}/{len(jobs)}] {condition}/{slug} {ok} "
                  f"{dt / 60:.1f} min | elapsed {elapsed / 60:.0f} min | "
                  f"ETA {eta:.0f} min", flush=True)

    print(f"finished {done} fits in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
