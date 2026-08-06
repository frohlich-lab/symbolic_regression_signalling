#!/bin/bash
#SBATCH --job-name=mm_sweep
#SBATCH --output=logs/mm_sweep.out
#SBATCH --error=logs/mm_sweep.err
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

set -euo pipefail

module load Anaconda3
module load Coreutils
module load GCCcore/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snakemake

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Export it before submitting the job." >&2
  exit 1
fi

mkdir -p logs
SWEEP_DIR="sweep_runs"
rm -rf "$SWEEP_DIR"
mkdir -p "$SWEEP_DIR"

MAX_WORKERS="${SWEEP_MAX_WORKERS:-10}"
CPUS_PER_RUN="${SWEEP_CPUS_PER_RUN:-2}"
MAX_TRIALS="${SWEEP_MAX_TRIALS:-118}"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
echo "[$(timestamp)] Launching sweep (expects envs pysr_env, pysindy_env, dso_env, kan_env, aifeynman_env to exist)" | tee logs/env_setup.log
echo "[$(timestamp)] Configuration: workers=$MAX_WORKERS cpus_per_run=$CPUS_PER_RUN max_trials=$MAX_TRIALS" | tee -a logs/env_setup.log

python src/sr_sweep/agent.py \
  --dataset data/three_step_enzyme/dynamic/processed/data_merged.csv \
  --features "P_u,k_off,k_D,k_cat,tK,k_inact,kcat_cg" \
  --methods pysr,pysindy,dso,kan,aifeynman \
  --search-space sweeps/sr_default_search.yaml \
  --max-runtime-seconds 900 \
  --max-workers "$MAX_WORKERS" \
  --cpus-per-run "$CPUS_PER_RUN" \
  --max-trials "$MAX_TRIALS" \
  --wandb-mode online \
  --project symbolic_regression_signalling \
  --entity lab_frohlich \
  --sweep-id lab_frohlich/symbolic_regression_signalling/dm0r9umc \
  --output-dir "$SWEEP_DIR"
