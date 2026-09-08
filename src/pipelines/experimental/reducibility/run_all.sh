#!/usr/bin/env bash
# Run the reducibility-diagnostic analyses.
#
#   ./run_all.sh                  re-analysis only (seconds; needs pandas/numpy)
#   ./run_all.sh --with-training  adds scoped calibration / ablation / latent runs
#
# The three training steps need jax + equinox + diffrax + sklearn. This script
# picks the first Snakemake conda env that has all four, since the repo has no
# single named env guaranteed to. Override with PYTHON=/path/to/python.
#
# NOTE (from experience in this repo): an active venv hijacks `python` inside
# Snakemake rules, so the interpreter is always invoked by absolute path here.
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(cd ../../../.. && pwd)"
OUT="${OUT:-$REPO_ROOT/data/experimental/runs/reducibility}"

# --- 0. resolve interpreters -------------------------------------------------
# find_python <module>...  -> first interpreter that imports all of them.
find_python() {
    local mods="$*"
    for cand in python "$REPO_ROOT"/.snakemake/conda/*_/bin/python; do
        command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ] || continue
        if "$cand" -c "
import importlib.util as u, sys
sys.exit(0 if all(u.find_spec(m) for m in '$mods'.split()) else 1)" >/dev/null 2>&1
        then echo "$cand"; return 0; fi
    done
    return 1
}

# pr_stats' mixed model needs statsmodels; the rest only needs pandas. Prefer an
# interpreter that has statsmodels so the mixed model is not silently skipped.
PLAIN_PYTHON="${PLAIN_PYTHON:-$(find_python pandas numpy || echo python)}"
STATS_PYTHON="${STATS_PYTHON:-$(find_python pandas numpy statsmodels || echo "$PLAIN_PYTHON")}"
if ! "$STATS_PYTHON" -c "import statsmodels" >/dev/null 2>&1; then
    echo "WARNING: no statsmodels found — the mixed model will be skipped." >&2
fi

WITH_TRAINING=0
[ "${1:-}" = "--with-training" ] && WITH_TRAINING=1

# --- 1-3. re-analysis (no training) ------------------------------------------
echo "=== 1. assemble master table ==="
"$PLAIN_PYTHON" assemble.py --output-dir "$OUT"

echo
echo "=== 2. SR x Neural-ODE quadrants ==="
"$PLAIN_PYTHON" quadrants.py --variant all --output-dir "$OUT"

echo
echo "=== 3a. participation-ratio contrast, all 40 contexts ==="
"$STATS_PYTHON" pr_stats.py --n-perm 20000 --output-dir "$OUT"

echo
echo "=== 3b. the manuscript's own subset (n=12/6) ==="
# perturbations only + conditioned on Neural-ODE success + best-of-seed
# reproduces the published group sizes exactly; this is the like-for-like retest.
"$STATS_PYTHON" pr_stats.py --condition-on-node-success --perturbations-only \
    --n-perm 20000 --output-dir "$OUT/manuscript_subset"

if [ "$WITH_TRAINING" -eq 0 ]; then
    echo
    echo "Re-analysis complete -> $OUT"
    echo "Add --with-training for the calibration / ablation / latent-state runs."
    exit 0
fi

# --- 4-6. training-based analyses --------------------------------------------
PYTHON="${PYTHON:-$(find_python jax equinox diffrax sklearn || true)}"
if [ -z "${PYTHON:-}" ]; then
    echo "ERROR: no environment with jax+equinox+diffrax+sklearn found." >&2
    echo "Set PYTHON=/path/to/python and re-run." >&2
    exit 1
fi
echo
echo "training interpreter: $PYTHON"

# Scoped by default. These are the knobs to widen once a scoped run looks sane:
#   MARKERS   contexts to run (default: a spread of SR-success and SR-fail)
#   SEEDS     random seeds
#   EPOCHS    training epochs (published run uses 200)
MARKERS="${MARKERS:-PTPN7 PIKFYVE PTPN2\ (P1) AKT3 ABL1 MAPK1}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-200}"

echo
echo "=== 4. PR calibration against known ground truth ==="
# --shuffle-control doubles the cost but is what isolates the correlation
# contribution, which is the whole point of the calibration.
"$PYTHON" pr_calibration.py \
    --output-dir "$OUT/calibration" \
    --markers $MARKERS --seeds $SEEDS --epochs "$EPOCHS" \
    --k-true 0 1 2 3 4 6 --reps 3 --shuffle-control --resume

echo
echo "=== 5. nested input ablation -> k* ==="
"$PYTHON" input_ablation.py \
    --output-dir "$OUT/ablation" \
    --markers $MARKERS --seeds $SEEDS --epochs "$EPOCHS" --resume

echo
echo "=== 6. latent-augmented Neural ODE -> hidden-state test ==="
"$PYTHON" latent_node.py \
    --output-dir "$OUT/latent" \
    --markers $MARKERS --seeds $SEEDS --epochs "$EPOCHS" \
    --latent-dims 0 1 2 3 --resume

echo
echo "=== 7. identifiability / observability head-to-head ==="
"$PYTHON" identifiability_baseline.py \
    --output-dir "$OUT/identifiability" \
    --markers $MARKERS --seeds $SEEDS --epochs "$EPOCHS" --resume

echo
echo "=== 8. rescue demonstration (drop -> diagnose -> restore) ==="
"$PYTHON" rescue_demo.py \
    --output-dir "$OUT/rescue" \
    --markers $MARKERS --seeds $SEEDS --epochs "$EPOCHS" \
    --control-input random --resume

echo
echo "=== 9. adjudicate the Path A claims ==="
"$STATS_PYTHON" collect.py --root "$OUT"

echo
echo "All steps complete -> $OUT"
