#!/usr/bin/env bash
set -euo pipefail

python -m py_compile \
  src/pipelines/experimental/data_prep/marker_inputs.py \
  src/pipelines/experimental/sr_pipeline/run_markers.py \
  src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
  src/pipelines/experimental/experiments/pysr_hyperparam_grid.py \
  src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
  src/pipelines/regimes/regime_variants.py \
  src/shared/constants.py

if rg -n "functional_groups|marker_set" src/pipelines workflow/rules | rg -v "archive|compat|legacy"; then
  echo "Found forbidden active naming tokens"
  exit 1
fi

pytest -q
