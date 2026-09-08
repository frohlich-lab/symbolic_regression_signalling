#!/usr/bin/env bash
# Submit the training-based reducibility analyses to NEMO as SLURM job arrays,
# at maximum parallelism.
#
# Granularity: one array task per *unit of work*, not per marker. Each step is
# expanded down to the finest split its own CLI already supports —
#
#   identifiability   (marker, seed)                    ->  120 tasks
#   ablation          (marker, seed)                    ->  120 tasks
#   latent            (marker, seed, latent_dim)        ->  480 tasks
#   calibration       (marker, seed, k_true)            ->  720 tasks
#   rescue            (marker, seed)                    ->  120 tasks
#   arch              (marker, seed, cfg)               -> 1440 tasks
#
# — because splitting finer than the marker costs nothing here. The XLA compile
# is per input-shape, so a task covering several k values recompiles for each
# one anyway; there is no amortisation to lose by splitting them apart.
#
# No throttle (`%N`) is applied: every task is eligible at once and SLURM
# schedules as nodes free up. Each task writes to its own output directory, so
# there is no CSV contention between concurrent tasks; `collect.py` merges them
# afterwards.
#
# These are tiny MLPs — CPU (`ncpu`) is the right partition, not a GPU.
#
# Usage (from the cluster checkout):
#   src/pipelines/experimental/reducibility/submit_nemo.sh all
#   src/pipelines/experimental/reducibility/submit_nemo.sh ablation
#
# Steps: identifiability | ablation | latent | calibration | rescue | arch | all
set -euo pipefail

STEP="${1:-all}"
REPO="${REPO:-/nemo/lab/froehlichf/home/users/pomeret/symbolic_regression_signalling}"
# sr_metrics2 is the env built for the ODE metrics work: jax[cpu] + diffrax +
# equinox + scikit-learn. pysr_env predates those and cannot run these scripts.
ENV_NAME="${ENV_NAME:-sr_metrics2}"
OUT="${OUT:-$REPO/data/experimental/runs/reducibility}"
LOGS="${LOGS:-$REPO/logs/reducibility}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-200}"
# Overridable: a delta/partial submission MUST use its own task dir, or it
# rewrites the files that already-queued arrays read by line index at
# runtime — silently remapping every pending task to the wrong work.
TASKDIR="${TASKDIR:-$OUT/_tasks}"

mkdir -p "$LOGS" "$OUT" "$TASKDIR"
cd "$REPO/src/pipelines/experimental/reducibility"

# --- marker list, derived from the dataset so it cannot drift ---------------
MARKER_LIST="${MARKER_LIST:-$OUT/markers.txt}"
if [ ! -s "$MARKER_LIST" ]; then
    echo "building marker list -> $MARKER_LIST"
    ml Anaconda3 >/dev/null 2>&1 || true
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    python -c "
from _shared import all_markers, load_raw, make_baseline_args
import pathlib
args = make_baseline_args(pathlib.Path('.'))
print('\n'.join(all_markers(load_raw(args))))" > "$MARKER_LIST"
    conda deactivate
fi
N_MARKERS=$(wc -l < "$MARKER_LIST")
echo "$N_MARKERS markers, seeds: $SEEDS"

# --- build a task file: one line = one task's extra CLI args ---------------
# Fields are TAB-separated so markers containing spaces — "PTPN2 (P1)",
# "DUSP10 (P2)" — survive intact. A space separator would silently split them
# into two bogus markers.
build_tasks() {
    local name="$1"; shift
    if [ -s "$TASKDIR/$name.tsv" ] && \
       squeue -u "$USER" -h -n "red_$name" 2>/dev/null | grep -q .; then
        echo "REFUSING: red_$name has live jobs reading $TASKDIR/$name.tsv." >&2
        echo "Cancel them first, or set TASKDIR to a fresh directory." >&2
        exit 1
    fi
    local variants=("$@")          # extra flag strings; use "" for none
    local f="$TASKDIR/$name.tsv"
    : > "$f"
    while IFS= read -r marker; do
        [ -n "$marker" ] || continue
        for seed in $SEEDS; do
            for v in "${variants[@]}"; do
                # slug: safe directory name for this task's private output dir
                slug=$(printf '%s_%s_%s' "$marker" "$seed" "$v" \
                       | tr -c 'A-Za-z0-9._-' '_')
                printf '%s\t%s\t%s\t%s\n' "$marker" "$seed" "$slug" "$v" >> "$f"
            done
        done
    done < <(cat "$MARKER_LIST"; echo)
    wc -l < "$f"
}

submit() {
    local name="$1" script="$2" hours="$3" mem="${4:-8G}"
    local f="$TASKDIR/$name.tsv"
    local n
    n=$(wc -l < "$f")
    [ "$n" -gt 0 ] || { echo "no tasks for $name" >&2; return 1; }
    sbatch --parsable \
        --job-name="red_$name" \
        --partition=ncpu \
        --array="1-$n" \
        --cpus-per-task=2 \
        --mem="$mem" \
        --time="$hours:00:00" \
        --output="$LOGS/${name}_%A_%a.out" \
        --error="$LOGS/${name}_%A_%a.err" \
        --wrap="
set -euo pipefail
ml Anaconda3
ml GCCcore/12.2.0
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
# Single-threaded BLAS: thousands of concurrent tasks each spawning threads
# would thrash the nodes, and these matrices are far too small to benefit.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $REPO/src/pipelines/experimental/reducibility
line=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $f)
marker=\$(printf '%s' \"\$line\" | cut -f1)
seed=\$(printf '%s' \"\$line\" | cut -f2)
slug=\$(printf '%s' \"\$line\" | cut -f3)
extra=\$(printf '%s' \"\$line\" | cut -f4)
echo \"task \${SLURM_ARRAY_TASK_ID}: marker='\$marker' seed=\$seed extra='\$extra'\"
# Private output dir per task -> no CSV contention across concurrent tasks.
python $script \
    --output-dir $OUT/$name/\$slug \
    --markers \"\$marker\" --seeds \$seed --epochs $EPOCHS --resume \$extra
"
}

# Per-step task expansion. The variant lists below are what make each step
# split finer than (marker, seed).
prep() {
    case "$1" in
        identifiability) build_tasks identifiability "" ;;
        ablation)        build_tasks ablation "" ;;
        rescue)          build_tasks rescue "--control-input random" ;;
        latent)          build_tasks latent \
                             "--latent-dims 0" "--latent-dims 1" \
                             "--latent-dims 2" "--latent-dims 3" ;;
        calibration)     build_tasks calibration \
                             "--k-true 0 --reps 3 --shuffle-control" \
                             "--k-true 1 --reps 3 --shuffle-control" \
                             "--k-true 2 --reps 3 --shuffle-control" \
                             "--k-true 3 --reps 3 --shuffle-control" \
                             "--k-true 4 --reps 3 --shuffle-control" \
                             "--k-true 6 --reps 3 --shuffle-control" ;;
        arch)
            # One task per architecture cell, so all 12 defensible cells run
            # concurrently rather than sequentially inside a task.
            local cfgs
            cfgs=$(ml Anaconda3 >/dev/null 2>&1; \
                   source "$(conda info --base)/etc/profile.d/conda.sh"; \
                   conda activate "$ENV_NAME"; \
                   python -c "
from arch_sweep import candidate_cells
print(' '.join(candidate_cells(False, 0.05)['cfg']))")
            local args=()
            for c in $cfgs; do args+=("--cfgs $c"); done
            build_tasks arch "${args[@]}" ;;
    esac
}

run_step() {
    local name="$1" script="$2" hours="$3" mem="${4:-8G}"
    local n; n=$(prep "$name")
    local id; id=$(submit "$name" "$script" "$hours" "$mem")
    echo "  $name: $n tasks -> job $id"
}

case "$STEP" in
    identifiability) run_step identifiability identifiability_baseline.py 4 ;;
    ablation)        run_step ablation        input_ablation.py           8 24G ;;
    latent)          run_step latent          latent_node.py              6 ;;
    calibration)     run_step calibration     pr_calibration.py           8 ;;
    rescue)          run_step rescue          rescue_demo.py             12 ;;
    arch)            run_step arch            arch_sweep.py               4 ;;
    all)
        echo "submitting all steps at full parallelism:"
        run_step identifiability identifiability_baseline.py 4
        run_step arch            arch_sweep.py               4
        run_step ablation        input_ablation.py           8 24G
        run_step latent          latent_node.py              6
        run_step calibration     pr_calibration.py           8
        run_step rescue          rescue_demo.py             12
        echo
        echo "watch:   squeue -u \$USER -o '%.18i %.14j %.2t %.10M %R' | head -30"
        echo "collect: python collect.py --root $OUT"
        ;;
    *) echo "unknown step: $STEP" >&2; exit 1 ;;
esac
