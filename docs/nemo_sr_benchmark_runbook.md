# SR-benchmark run on NEMO (login.nemo.thecrick.org)

Runbook for executing the symbolic-regression method comparison on the Crick NEMO
cluster. The four methods are **aifeynman, kan, pysr, dso** (pysindy is not used).
NEMO is x86 Linux, which is why we run there instead of the ARM Mac: AI-Feynman's
Fortran extensions and DSO's TensorFlow 1.14 stack build cleanly on x86 Linux but not
on Apple Silicon.

Status of the four methods (as of the fixing session):
- **kan** — verified working on `two_step_enzyme` locally (clean parsimonious formula).
- **pysr** — the always-working baseline.
- **aifeynman** — code fixed and runs clean end-to-end; needs the x86 Fortran build +
  `numpy<2` to actually produce a non-empty Pareto front (that's what this cluster run is for).
- **dso** — code + wiring fixed; needs source (`scripts/setup_dso.sh`) and the TF1.14/py3.7
  env, which only build on x86 Linux.

Target for this run: `enzyme_model: two_step_enzyme`, `data_type: dynamic`
(already set in `config.yaml`).

Expected DAG (validated locally with `snakemake -n`):
`aifeynman, kan, pysr(×2 variants), dso → get_best_formula → integrate_and_plot_results`
→ `data/two_step_enzyme/dynamic/sr_comparison/results/loss_comparison.csv`.

---

## 0. Get the code onto NEMO

```bash
# From the Mac: commit + push the branch (see the repo's normal remote).
# On NEMO:
ssh <user>@login.nemo.thecrick.org
cd /camp/…/symbolic_regression_signalling   # your working copy
git pull                                     # bring in the fixed common.smk, aifeynman_model.py, config.yaml, envs/aifeynman.yaml
```

The input data must be present on NEMO:
`data/two_step_enzyme/dynamic/processed/data_train.csv` and `data_test.csv`.
If they aren't there, rsync them from the Mac (they already exist locally) or
regenerate them with the upstream data-generation rules.

## 1. Driver environment (snakemake)

```bash
module load Anaconda3            # or the cluster's conda module; adjust to NEMO's actual module name
# One-time: create a driver env with snakemake + mamba (mamba speeds up --use-conda env builds)
conda create -n sr_driver -c conda-forge -c bioconda snakemake=7.32 mamba pandas -y
conda activate sr_driver
snakemake --version              # expect 7.32.x
```

Snakemake 7.32 matches what the repo was developed against. (8.x also builds the DAG
fine — verified — but 7.32 avoids CLI-flag surprises with `--use-conda`.)

## 2. Per-rule conda envs (built automatically by --use-conda)

Snakemake materializes one conda env per rule from `envs/*.yaml` into `.snakemake/conda/`.
The heavy/fragile ones are **aifeynman** and (optionally) **dso**.

Key fixes already baked in so these build correctly on Linux:
- `envs/aifeynman.yaml` now pins **`numpy>=1.22,<2`**. This is essential: aifeynman 2.0.x
  builds its Fortran extensions via `numpy.distutils` (removed in numpy 2.x) and calls
  several numpy 1.x-only APIs. With numpy 2.x the search silently returns an empty
  Pareto set and produces no formula.
- The aifeynman **runtime install** (in `common.smk` `symbolic_regression_rule` `install_cmds`)
  is: `pip install 'setuptools<60' wheel` → `export SDKROOT=$(xcrun … || echo)` → 
  `pip install --no-deps --no-build-isolation aifeynman`. On Linux `xcrun` is absent so
  `SDKROOT` is empty (harmless); the conda `gfortran` in the env compiles the Fortran.
  After the build, `.../site-packages/aifeynman/` should contain compiled `feynman_*`
  binaries — if that directory has **no** `feynman_*` files, brute-force search is a no-op
  and only polyfit results (if any) survive.

## 3. Run

```bash
cd /path/to/symbolic_regression_signalling
conda activate sr_driver

TARGET=data/two_step_enzyme/dynamic/sr_comparison/results/loss_comparison.csv

# Dry-run first to confirm the DAG (fast, no builds):
snakemake -n --use-conda --rerun-triggers mtime -c1 "$TARGET"

# Real run. --use-conda builds per-rule envs on first use (slow the first time,
# especially aifeynman's Fortran compile). --conda-frontend mamba speeds this up.
snakemake --use-conda --conda-frontend mamba --rerun-triggers mtime -c4 -p "$TARGET"
```

Notes:
- `--rerun-triggers mtime` prevents snakemake from re-running everything just because
  code/env definitions changed (otherwise pysr etc. re-run on provenance).
- Each SR method has a wall-clock cap of `timeout_duration: 1800` (30 min) in `config.yaml`.
- The per-method rule masks failures with `… || test -s {temp_file}`: if a method dies
  but left a non-empty temp file, the DAG continues. `aifeynman_model.py` also exits 0
  on internal failure (it salvages partial results in a `finally`), so a genuinely empty
  aifeynman result yields an empty temp file → `get_best_formula` reports "no formula"
  and that method is simply absent from `loss_comparison.csv`.

### SLURM (optional, for long runs)

Run the driver inside an allocation so the 30-min method timeouts fit:

```bash
# Interactive:
srun --partition=ncpu --cpus-per-task=4 --mem=16G --time=08:00:00 --pty bash
#   …then activate sr_driver and run the snakemake command above.

# Or let snakemake submit each rule as its own SLURM job (snakemake 7.32):
snakemake --use-conda --conda-frontend mamba --rerun-triggers mtime \
  --jobs 8 \
  --cluster "sbatch --partition=ncpu --cpus-per-task={threads} --mem=16G --time=01:00:00" \
  "$TARGET"
```
Adjust partition names / limits to NEMO's actual scheduler config.

## 4. Verify

```bash
cat data/two_step_enzyme/dynamic/sr_comparison/results/loss_comparison.csv
ls  data/two_step_enzyme/dynamic/sr_comparison/results/formula_*.txt
```

`loss_comparison.csv` should have one row per method that produced a formula. Each
`formula_<method>.txt` holds the extracted best expression (with `x0,x1,…` mapped back
to `P_u,k_off,k_D,k_cat,tK,k_inact`). If a method is missing, check its
`formula_<method>.txt` (empty → the method produced nothing) and the rule log.

## 5. DSO source (required before the run, since `dso` is in `models`)

`src/dso` is gitignored and empty in a fresh checkout. Fetch the source with the
helper script (run it on NEMO, before `snakemake`):

```bash
bash scripts/setup_dso.sh          # clones dso-org/deep-symbolic-optimization @ pinned commit
                                   # and places its inner project dir at src/dso
ls src/dso/setup.py src/dso/dso/__init__.py   # sanity check
```

The `dso` rule then builds `envs/dso.yaml` (Python 3.7 + TensorFlow 1.14 + numpy≤1.19)
and runs its `install_cmds` (`pip install absl-py==0.7.0 && pip install numpy==1.18 &&
pip install -e src/dso`). Those numpy/absl pins are **deliberate** TF1.14-compat
downgrades — do not "simplify" them away. DSO also compiles a small Cython/C extension,
so the env needs a C compiler (present by default on NEMO).

`dso_model.py` was fixed to not call `setup()` twice (DSO's `train()` already calls it).
If DSO is too heavy for the first pass, temporarily drop `dso` from `models:` in
`config.yaml`, get a `loss_comparison.csv` from aifeynman+kan+pysr, then add `dso` back.

---

## What was fixed to make this runnable (session summary)

- **AI-Feynman** (`src/sr_models/aifeynman_model.py`, rewritten):
  1. Neutralized the unguarded generalized-symmetry / gradient-decomposition stage
     (`identify_decompositions`) that crashed the whole run with `'int' object is not
     callable` before any solution was written.
  2. Fixed the solution path — AI-Feynman writes to `./results/` (CWD-relative), not
     `data/aifeynman/results/`.
  3. Normalized the native solution table (`test_err … complexity fit_err <formula>`) into
     the `Formula:`/`Error:` format the parser expects, and salvage it in a `finally`.
- **`get_best_formula.py`**: aifeynman extractor now uses `start_index=0, underscore=False`
  (aifeynman emits `x0,x1,…`, 0-indexed, no underscore).
- **`common.smk`** SR-comparison rules: removed the Linux-only `strace` wrapper and added
  `PYTHONPATH=src` (scripts import top-level `regime_variants`/`plot_style`/`mm_models`).
- **`config.yaml`**: `models: [aifeynman, kan, pysr, dso]`, `enzyme_model: two_step_enzyme`.
- **`envs/aifeynman.yaml`**: `numpy>=1.22,<2`.
- **DSO**: `scripts/setup_dso.sh` fetches source to `src/dso`; `dso_model.py` double-`setup()` fixed.
