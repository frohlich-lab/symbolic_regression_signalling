#!/bin/bash
#SBATCH --job-name=sr_sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --partition=ncpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

set -euo pipefail

# Load cluster modules / activate environment
ml Anaconda3
ml Coreutils
ml GCCcore/12.2.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snakemake

# Ensure WANDB_API_KEY is set in your environment before submitting the job.
# export WANDB_API_KEY="your-key-here"

# Create a dedicated output directory for this sweep run
SWEEP_DIR="sweep_runs_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR" logs

python src/sr_sweep/agent.py \
  --dataset data/three_step_enzyme/dynamic/processed/data_merged.csv \
  --features "P_u,k_off,k_D,k_cat,tK,k_inact,kcat_cg" \
  --methods pysr,pysindy,dso,kan,aifeynman \
  --search-space sweeps/sr_default_search.yaml \
  --max-runtime-seconds 900 \
  --max-workers 4 \
  --cpus-per-run 10 \
  --max-trials 40 \
  --wandb-mode online \
  --project symbolic_regression_signalling \
  --entity lab_frohlich \
  --sweep-id lab_frohlich/symbolic_regression_signalling/dm0r9umc \
  --output-dir "$SWEEP_DIR"
