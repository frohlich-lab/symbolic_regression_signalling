#!/usr/bin/env bash
# Fetch the Deep Symbolic Optimization (DSO) source and place it at src/dso so
# that `pip install -e src/dso` and `from dso import DeepSymbolicRegressor` work.
#
# DSO is pinned to TensorFlow 1.14 / Python 3.7 (see envs/dso.yaml). It builds on
# x86 Linux (e.g. the NEMO cluster) but NOT on Apple Silicon — run this there.
#
# The upstream repo nests the installable project one level down:
#   deep-symbolic-optimization/           <- clone root
#     dso/                                <- project dir (has setup.py)   <-- becomes src/dso
#       dso/                              <- the importable `dso` package
# so we move the inner project dir to src/dso.
#
# src/dso is gitignored (per .gitignore), so nothing here is committed.
set -euo pipefail

REPO_URL="https://github.com/dso-org/deep-symbolic-optimization.git"
# Pin for reproducibility (HEAD as vendored 2026-07-10). Override with DSO_COMMIT=<sha>.
DSO_COMMIT="${DSO_COMMIT:-8348d5b08d1eef6170fdbfd222a492ded990ea12}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/src/dso"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [ -e "$DEST/setup.py" ]; then
    echo "src/dso already present (setup.py found) — skipping clone. Remove src/dso to re-fetch."
    exit 0
fi

echo "Cloning DSO into a temp dir…"
git clone "$REPO_URL" "$TMP/dso_repo"
git -C "$TMP/dso_repo" checkout "$DSO_COMMIT"

echo "Placing project dir at $DEST …"
mkdir -p "$REPO_ROOT/src"
rm -rf "$DEST"
mv "$TMP/dso_repo/dso" "$DEST"

# Sanity checks
test -f "$DEST/setup.py"        || { echo "ERROR: $DEST/setup.py missing"; exit 1; }
test -f "$DEST/dso/__init__.py" || { echo "ERROR: $DEST/dso/__init__.py missing"; exit 1; }
echo "DSO ready at src/dso (commit $DSO_COMMIT)."
echo "Next: snakemake --use-conda builds envs/dso.yaml and runs 'pip install -e src/dso' per the dso rule."
