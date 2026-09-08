"""Shared plumbing for the reducibility-diagnostic analyses.

Everything in this package answers one question: *how strong is the evidence
that SR failure marks a genuine absence of a compact low-dimensional rate law,
rather than a limitation of SR?* The scripts here reuse the paper's own
Neural-ODE machinery (`neural_ode_diffrax_baseline`) so that any number they
produce is directly comparable to Table S8 / Fig. S3.

The paper's OOD configuration is pinned in `PAPER_CFG` below, transcribed from
`workflow/rules/experimental.smk` (`_diffrax_sweep_params` +
`experimental_neural_ode_diffrax_ood_l21`) and `workflow/rules/common.smk`.
Keep it in sync with those rules; every script in this package inherits it so a
calibration run and the published run differ only in what is under test.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# reducibility/ -> experimental/ -> pipelines/ -> src/
SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

# jax / equinox / diffrax are imported lazily via `_jax()` and `_baseline()` so
# that the pure re-analysis scripts (assemble / quadrants / pr_stats, which only
# need pandas) run in any environment. Only the three training-based analyses
# need the pysr env.
_JAX_CACHE: dict = {}


def _jax():
    """Import and configure jax on first use."""
    if "jax" not in _JAX_CACHE:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        _JAX_CACHE["jax"], _JAX_CACHE["jnp"] = jax, jnp
    return _JAX_CACHE["jax"], _JAX_CACHE["jnp"]


def _baseline():
    """The published Neural-ODE module (pulls in jax/equinox/diffrax)."""
    if "nodb" not in _JAX_CACHE:
        _jax()
        from pipelines.experimental.sr_pipeline import (
            neural_ode_diffrax_baseline as nodb,
        )
        _JAX_CACHE["nodb"] = nodb
    return _JAX_CACHE["nodb"]


#: "p_ERK1_2" — hardcoded rather than imported so pandas-only scripts stay light.
#: Asserted against the baseline in `_baseline_perk_check()`.
PERK_NAME = "p_ERK1_2"
RUNS_ROOT = REPO_ROOT / "data" / "experimental" / "runs"
NDE_ROOT = RUNS_ROOT / "sparse_neural_ode"
PYSR_ROOT = RUNS_ROOT / "pysr_ood_final"

#: The published sparse-Neural-ODE OOD configuration (L21, lambda_jac = 3).
PAPER_CFG: Dict[str, object] = dict(
    dataset=REPO_ROOT / "data/experimental/processed/functional_groups/markers_per_minute_fit.csv",
    raw_dataset=str(REPO_ROOT / "data/experimental/raw/lun_2019.csv"),
    seeds=(42, 43, 44),
    measured_timepoints=[0.0, 5.0, 10.0, 15.0, 30.0, 60.0],
    per_minute_max_time=30.0,
    per_minute_sampling_strategy="early_plus_sparse_late",
    late_sample_window=(30.0, 60.0),
    late_sample_points=15,
    test_split_policy="top_gfp_bins",
    hidden_dim=64,
    hidden_layers=3,
    activation="tanh",
    lr=3e-3,
    epochs=200,
    patience=20,
    jac_reg=3.0,
    jac_reg_mode="l21",
    hess_reg=0.0,
    path_reg=0.0,
)

#: Held-out R^2 above which a context counts as "a compact law was recovered".
#: 0.6 is the paper's threshold; kept as a module constant so every script in
#: this package moves together if it is ever re-calibrated.
SUCCESS_THRESHOLD = 0.6

VARIANTS = ("l1", "l21_lam3", "pathreg")
REFERENCE_VARIANT = "l21_lam3"


# -----------------------------------------------------------------------------
# Baseline argument namespace
# -----------------------------------------------------------------------------

def make_baseline_args(output_dir: Path, **overrides) -> argparse.Namespace:
    """An argparse.Namespace equivalent to the published L21 OOD invocation.

    Built by asking the baseline's own parser for its defaults (so new options
    can't silently go missing), then applying PAPER_CFG and any overrides.
    """
    nodb = _baseline()
    assert nodb.PERK_NAME == PERK_NAME, (
        f"baseline renamed the p-ERK column to {nodb.PERK_NAME!r}; update PERK_NAME")
    # The baseline's parse_args() reads sys.argv directly, so borrow it rather
    # than duplicating its defaults here — that way a new option added upstream
    # is inherited instead of silently missing.
    saved_argv = sys.argv
    try:
        sys.argv = ["neural_ode_diffrax_baseline.py", "--output-dir", str(output_dir)]
        args = nodb.parse_args()
    finally:
        sys.argv = saved_argv
    for key, value in PAPER_CFG.items():
        setattr(args, key, value)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise KeyError(f"{key!r} is not a neural_ode_diffrax_baseline option")
        setattr(args, key, value)
    args.output_dir = Path(output_dir)
    return args


def load_raw(args: argparse.Namespace) -> pd.DataFrame:
    return pd.read_csv(args.dataset)


def all_markers(raw: pd.DataFrame) -> List[str]:
    from pipelines.experimental.sr_pipeline.compute_marker_integration import (
        prepare_sr_dataset,
    )
    base, _ = prepare_sr_dataset(raw)
    return sorted(base["marker"].dropna().unique().tolist())


def load_bundle(raw: pd.DataFrame, marker: str, seed: int,
                args: argparse.Namespace) -> Optional[Dict[str, object]]:
    """Rebuild the exact train/val/test bundle the published run used."""
    return _baseline()._per_minute_split(raw, marker, seed, args)


# -----------------------------------------------------------------------------
# Input subsetting  (for the nested-ablation analysis)
# -----------------------------------------------------------------------------

def subset_bundle(bundle: Dict[str, object],
                  keep_exo: Sequence[int]) -> Dict[str, object]:
    """Return a copy of `bundle` retaining only `keep_exo` exogenous inputs.

    p-ERK itself is always input 0 of the network (see `_predict_dt_at_rows`:
    the RHS is called on ``[y, *U_scaled]``), so it is never droppable — the
    state variable has to stay. `keep_exo` indexes into ``bundle["exo_cols"]``.

    Every array that carries an exogenous axis is subset consistently, so the
    result can be handed straight to `_train_marker` / `_split_metrics`.
    """
    keep = np.asarray(sorted(set(int(i) for i in keep_exo)), dtype=int)
    exo_cols = list(bundle["exo_cols"])
    if keep.size and (keep.min() < 0 or keep.max() >= len(exo_cols)):
        raise IndexError(f"keep_exo out of range for {len(exo_cols)} exogenous inputs")

    out = copy.copy(bundle)
    new_exo = [exo_cols[i] for i in keep]
    out["exo_cols"] = new_exo
    # feature_cols only feeds in_dim (= 1 + n_exo) and the perk_index bookkeeping.
    out["feature_cols"] = [PERK_NAME] + new_exo
    out["perk_index"] = 0
    out["u_mean"] = np.asarray(bundle["u_mean"])[keep]
    out["u_scale"] = np.asarray(bundle["u_scale"])[keep]

    for split in ("train", "val", "test"):
        pack = bundle.get(split)
        if pack is None:
            out[split] = None
            continue
        new_pack = dict(pack)
        us = np.asarray(pack["us"])
        new_pack["us"] = us[:, :, keep] if us.size else us[:, :, :0]
        new_pack["n_F"] = int(keep.size)
        new_trajs = []
        for tr in pack["trajs"]:
            new_tr = dict(tr)
            for field in ("U", "U_scaled"):
                if field in tr and np.asarray(tr[field]).size:
                    new_tr[field] = np.asarray(tr[field])[:, keep]
                elif field in tr:
                    new_tr[field] = np.asarray(tr[field])[:, :0]
            new_trajs.append(new_tr)
        new_pack["trajs"] = new_trajs
        out[split] = new_pack
    return out


# -----------------------------------------------------------------------------
# Jacobian diagnostics
# -----------------------------------------------------------------------------

def input_matrix(bundle: Dict[str, object], split: str = "train") -> np.ndarray:
    """The ``[y, *U_scaled]`` rows the RHS is actually evaluated on."""
    pack = bundle[split]
    y_mean, y_scale = float(bundle["y_mean"]), float(bundle["y_scale"])
    rows = []
    for tr in pack["trajs"]:
        y = (np.asarray(tr["y"]) - y_mean) / y_scale
        u = np.asarray(tr["U_scaled"])
        rows.append(np.concatenate([y[:, None], u], axis=1) if u.size else y[:, None])
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 1))


def participation_ratio(abs_jac_mean: np.ndarray) -> float:
    """PR = (sum |J_i|)^2 / sum |J_i|^2 — the paper's effective dependency count."""
    a = np.asarray(abs_jac_mean, dtype=float)
    denom = float((a ** 2).sum())
    return float(a.sum() ** 2 / max(denom, 1e-12))


def jacobian_summary(rhs, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean signed and absolute input Jacobian, plus the participation ratio."""
    jax, jnp = _jax()
    if X.size == 0:
        n = getattr(rhs, "in_dim", 1)
        nan = np.full(n, np.nan)
        return {"mean_jac": nan, "abs_jac": nan, "pr": float("nan")}
    jac = np.asarray(jax.jit(jax.vmap(jax.grad(rhs)))(jnp.asarray(X)))
    abs_jac = np.abs(jac).mean(axis=0)
    return {"mean_jac": jac.mean(axis=0), "abs_jac": abs_jac,
            "pr": participation_ratio(abs_jac)}


def input_names(bundle: Dict[str, object]) -> List[str]:
    return [PERK_NAME] + list(bundle["exo_cols"])


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows)", flush=True)


def default_outdir() -> Path:
    return RUNS_ROOT / "reducibility"
