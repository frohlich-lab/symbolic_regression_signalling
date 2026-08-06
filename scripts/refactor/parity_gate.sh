#!/usr/bin/env bash
# Guard rail for the flat-src -> src/pipelines migration: byte-compile the modules the
# migration touched, forbid the pre-migration naming, and run the tests.
set -euo pipefail

python -m py_compile \
  src/pipelines/experimental/data_prep/marker_inputs.py \
  src/pipelines/experimental/sr_pipeline/run_markers.py \
  src/pipelines/experimental/sr_pipeline/compute_marker_integration.py \
  src/pipelines/experimental/sr_pipeline/plot_marker_overlays.py \
  src/shared/regime_variants.py \
  src/shared/constants.py

# Every intra-repo import must name a real module and a real symbol. This is the check
# that would have caught tests/ still importing pipelines.regimes.regime_variants after
# that module moved to src/shared/ -- a grep over src/ alone missed it.
python - <<'PY'
import ast, sys
from pathlib import Path

SRC = Path("src")
LOCAL = {"shared", "pipelines", "sr_models", "sr_sweep", "synthetic"}
ROOTS = [SRC, Path("tests")]


def names_of(path):
    out = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.If):
            out |= {n.name for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return out


def as_module(mod):
    p = SRC / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    i = SRC / mod.replace(".", "/") / "__init__.py"
    return i if i.exists() else None


def is_namespace_pkg(mod):
    return (SRC / mod.replace(".", "/")).is_dir()


problems, checked = [], 0
for root in ROOTS:
    for py in sorted(root.rglob("*.py")):
        if any(x in py.parts for x in ("dso", "pysindy", "__pycache__")):
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.split(".")[0] not in LOCAL:
                    continue
                checked += 1
                target = as_module(node.module)
                if target is None:
                    if is_namespace_pkg(node.module):
                        base = SRC / node.module.replace(".", "/")
                        for a in node.names:
                            if not (base / f"{a.name}.py").exists() and not (base / a.name).is_dir():
                                problems.append(
                                    f"{py}:{node.lineno} '{a.name}' is not a submodule of {node.module}")
                    else:
                        problems.append(f"{py}:{node.lineno} no module '{node.module}'")
                    continue
                available = names_of(target)
                for a in node.names:
                    if a.name != "*" and a.name not in available:
                        problems.append(f"{py}:{node.lineno} '{a.name}' not defined in {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] not in LOCAL:
                        continue
                    checked += 1
                    if as_module(a.name) is None and not is_namespace_pkg(a.name):
                        problems.append(f"{py}:{node.lineno} no module '{a.name}'")

print(f"import check: {checked} intra-repo bindings")
if problems:
    for p in problems:
        print("  " + p)
    sys.exit(1)
print("import check: all resolve")
PY

# grep, not rg: ripgrep is not installed here, and because these checks sit in `if`
# conditions a missing binary made the `if` false -- so the gate reported success while
# two of its four checks had silently not run. `set -e` does not catch that either.
BARE='^[[:space:]]*(from (constants|regime_variants|plot_style|mm_models|nn_model|utils)(\.| import)|import (constants|regime_variants|plot_style|mm_models|nn_model))'

# No module may be imported by bare name any more -- shared code is addressed as
# `shared.<module>`, so a bare `from constants import ...` means a missed rewrite.
if grep -rEn "$BARE" src tests 2>/dev/null | grep -vE 'src/(dso|pysindy)/'; then
  echo "Found bare-name imports of shared modules; use shared.<module> instead"
  exit 1
fi

# The rename was functional_groups -> markers in CODE. Three exclusions, all of which
# this check needed and none of which it had (it had never actually run: `rg` is absent,
# so the `if` was always false):
#   - __pycache__, which matches as "Binary file ... matches"
#   - data paths, because the processed directory is genuinely named functional_groups
#     (config.yaml `output_prefix: functional_groups`), so those references are correct
#   - `marker_set` was in the original pattern but only ever matches a local variable
#     meaning "a set of markers", which is not the legacy naming
if grep -rEn "functional_groups" src/pipelines workflow/rules 2>/dev/null \
     --exclude-dir=__pycache__ \
     | grep -vE "archive|compat|legacy" \
     | grep -vE "data/|processed_dir|exp_processed"; then
  echo "Found forbidden active naming tokens"
  exit 1
fi

# Scoped to tests/: a bare `pytest -q` also collects the vendored suites under
# src/dso/ and src/pysindy/, which fail to even import (18 collection errors) and have
# nothing to do with this repo's code.
pytest tests/ -q
