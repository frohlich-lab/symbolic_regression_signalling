#!/bin/bash
#SBATCH --job-name=sr_benchmark          # Job name
#SBATCH --output=output_file.out          # Output file
#SBATCH --error=error_file.err            # Error file
#SBATCH --time=23:00:00                   # Time limit (HH:MM:SS)
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --partition=ncpu              # Partition (queue) name


# Load any modules or activate your environment here
ml Anaconda3
ml Coreutils
ml GCCcore/12.2.0

# Disable SSL verification for Git
export GIT_SSL_NO_VERIFY=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate snakemake

# Run your script or command
snakemake --unlock
snakemake --cores=all --use-conda