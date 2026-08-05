"""Run stage 2 (ODE integration metrics) for many shards inside few processes.

Why this exists: invoking compute_marker_integration.py once per shard costs ~30 s,
almost all of it python + jax import and the diffrax JIT compile. Those are paid once
per PROCESS, so looping over shards inside one process amortises them and the marginal
cost per shard drops to the actual solve. Each worker handles a contiguous block.

Usage:
  python stage2_local.py <repo_root> <sweep_root> [n_workers] [limit]
"""
import os
import sys
import time
import multiprocessing as mp

REPO = sys.argv[1].rstrip("/")
SWEEP = sys.argv[2].rstrip("/")
N_WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 0

SEEDS = [42, 43, 44]
MEASURED = ["0", "5", "10", "15", "30", "60"]


def load_tables():
    with open(f"{SWEEP}/arms/arms.tsv") as fh:
        rows = [l.rstrip("\n") for l in fh][1:]
    arms = [r.split("\t") for r in rows if r.strip()]
    with open(f"{SWEEP}/arms/markers.tsv") as fh:
        mk = [l.rstrip("\n").split("\t")[0] for l in fh if l.strip()]
    return arms, mk


ARMS, MARKERS = load_tables()
PER_ARM = len(MARKERS) * len(SEEDS)
N_SHARDS = len(ARMS) * PER_ARM


def shard_spec(idx):
    arm_row = ARMS[idx // PER_ARM]
    rem = idx % PER_ARM
    return {
        "arm": arm_row[0],
        "test_size": arm_row[1],
        "slug": MARKERS[rem // len(SEEDS)],
        "seed": SEEDS[rem % len(SEEDS)],
    }


def worker(block):
    """Import once, then process a block of shards in-process."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    # XLA runs its own host thread pool that the *_NUM_THREADS vars do not control;
    # without pinning it, N workers each spawn N threads and oversubscribe the CPU.
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    ).strip()
    sys.path.insert(0, f"{REPO}/src")
    import importlib
    cmi = importlib.import_module("pipelines.experimental.sr_pipeline.compute_marker_integration")

    done = skipped = failed = 0
    for idx in block:
        s = shard_spec(idx)
        out = f"{SWEEP}/out_seed/{s['arm']}/{s['slug']}/s{s['seed']}"
        md = f"{out}/seeds/seed_{s['seed']}/metrics"
        summ = f"{out}/seeds/seed_{s['seed']}/summary/marker_summary.csv"
        traj = f"{md}/predicted_trajectories_per_minute.csv"
        outcsv = f"{md}/marker_integration_metrics_per_minute.csv"
        pm = f"{SWEEP}/data/shards/{s['slug']}_per_minute.csv"

        if not (os.path.exists(summ) and os.path.exists(traj)):
            skipped += 1
            continue
        if os.path.exists(outcsv) and os.path.getsize(outcsv) > 0:
            skipped += 1
            continue
        os.makedirs(f"{out}/aggregated/metrics", exist_ok=True)

        argv = [
            "compute_marker_integration",
            "--dataset", pm,
            "--summary", summ,
            "--sr-trajectories", traj,
            "--dataset-mode", "per_minute",
            "--output", outcsv,
            "--trajectories-output", f"{md}/marker_integration_trajectories_per_minute.csv",
            "--aggregate-output", f"{out}/aggregated/metrics/marker_integration_metrics_per_minute_all_seeds.csv",
            "--measured-timepoints", *MEASURED,
            "--test-size", s["test_size"],
            "--seed", str(s["seed"]),
        ]
        old_argv, old_stdout = sys.argv, sys.stdout
        try:
            sys.argv = argv
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                cmi.main()
            done += 1
        except SystemExit:
            done += 1
        except Exception:
            failed += 1
        finally:
            sys.argv, sys.stdout = old_argv, old_stdout
    return done, skipped, failed


def main():
    n = LIMIT if LIMIT else N_SHARDS
    idxs = list(range(n))
    blocks = [idxs[i::N_WORKERS] for i in range(N_WORKERS)]
    t0 = time.time()
    print(f"{n} shards, {N_WORKERS} workers, {len(ARMS)} arms x {len(MARKERS)} markers x {len(SEEDS)} seeds")
    with mp.Pool(N_WORKERS) as pool:
        res = pool.map(worker, blocks)
    d = sum(r[0] for r in res); s = sum(r[1] for r in res); f = sum(r[2] for r in res)
    el = time.time() - t0
    print(f"done={d} skipped={s} failed={f} in {el/60:.1f} min "
          f"({el/max(d,1):.1f} s/shard computed)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
