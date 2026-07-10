"""Compute marker-level integration metrics from SR formulas and datasets.

This script evaluates each formula on the provided dataset, integrates predicted
derivatives per GFP bin, and reports dt/test R2 plus median integrated R2.

Two integration modes are recorded:
1) Feature-driven integration (existing): evaluate d/dt at the observed
   feature values and integrate forward (Euler).
2) ODE-style integration: treat p-ERK1-2 as the only unknown state, other
   features as known exogenous signals, and re-evaluate the derivative at
   each step using the currently integrated p-ERK1-2 value. When formulas
   do not include p-ERK1-2, both modes coincide.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sympy as sp

from jax import config as jax_config  # type: ignore
jax_config.update("jax_enable_x64", True)
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve, Kvaerno5  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.experimental.sr_pipeline.run_markers import (  # type: ignore
    EXCLUDE_COLUMNS,
    choose_bin_split,
    REL_MAE_EPS,
    sanitize_feature_names,
)
from pipelines.experimental.sr_pipeline.metrics import coefficient_of_determination  # type: ignore
from pipelines.experimental.sr_pipeline.seeding import seed_all
from utils.seeding import resolve_seed


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Tolerances / steps for the ODE-style integration (Kvaerno5 via diffrax)
ODE_ATOL = 1e-6
ODE_RTOL = 1e-4
ODE_DT0_MIN = 1e-3
ODE_DT0_SCALE = 0.2  # fraction of the smallest dt to start with
ODE_DTMAX = 1.0
ODE_MAX_STEPS = 200_000
ODE_KVAERNO5_SOLVER_KWARGS = {
    "rtol": ODE_RTOL,
    "atol": ODE_ATOL,
    "dtmax": ODE_DTMAX,
    "dt0": 1e-1,
    "max_steps": ODE_MAX_STEPS,
}

# Legacy derivative clipping helpers (used by SelectK scripts)
ODE_TSIT5_DERIV_CLIP = 1e6
ODE_DERIV_PERCENTILES = (0.01, 0.99)
ODE_DERIV_MARGIN_FRAC = 0.25

# Small epsilon for sanitising PySR formulas (avoid singularities)
PY_SR_SAFE_DIVISION_EPS = 1e-1


# -----------------------------------------------------------------------------
# Legacy compatibility helpers
# -----------------------------------------------------------------------------

class NoClipStateTransform:
    """Pass-through transform (compat shim for downstream scripts)."""

    name = "identity"
    actual_bounds = (np.nan, np.nan)
    state_bounds = (np.nan, np.nan)

    def clip_actual(self, values):
        return np.asarray(values, dtype=float)

    def clip_state(self, values):
        return np.asarray(values, dtype=float)

    def forward(self, values):
        return np.asarray(values, dtype=float)

    def inverse(self, values):
        return np.asarray(values, dtype=float)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute integration metrics from SR summary and dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CSV used for SR (snapshot or per-minute).",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="marker_summary.csv from SR run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV path for integration metrics.",
    )
    parser.add_argument(
        "--trajectories-output",
        type=Path,
        required=True,
        help="CSV path for integrated trajectories.",
    )
    parser.add_argument(
        "--sr-trajectories",
        type=Path,
        default=None,
        help=(
            "Optional predicted_trajectories CSV from SR "
            "(reuse its train/test split instead of re-splitting)."
        ),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("snapshot", "per_minute"),
        default="snapshot",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=(0.0, 5.0, 10.0, 15.0, 30.0, 60.0),
        help="Measured timepoints; per-minute metrics restricted to these.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=("PySR", "Linear Regression"),
        help="Models to include from the summary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed identifier to annotate outputs for downstream averaging.",
    )
    parser.add_argument(
        "--pysr-safe-division-eps",
        type=float,
        default=PY_SR_SAFE_DIVISION_EPS,
        help="ε used when sanitising PySR formulas (avoid singularities).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of GFP bins to hold out for test metrics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for bin split.",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help=(
            "Optional path to maintain aggregated metrics across seeds "
            "(mean over numeric columns)."
        ),
    )
    return parser.parse_args()


def _infer_aggregate_output(args: argparse.Namespace) -> Optional[Path]:
    """Default aggregate_output to runs/aggregated/metrics when output is under runs/seeds."""
    if args.aggregate_output is not None:
        return args.aggregate_output
    out = Path(args.output).resolve()
    for parent in out.parents:
        if parent.name.startswith("seed_") and parent.parent.name == "seeds":
            aggregated = parent.parent.parent / "aggregated" / "metrics"
            aggregated.mkdir(parents=True, exist_ok=True)
            return aggregated / out.name
    return None


# -----------------------------------------------------------------------------
# Sympy helpers
# -----------------------------------------------------------------------------

def _is_negative_number(value: sp.Basic) -> bool:
    if value.is_number:
        try:
            return float(value) < 0
        except Exception:
            return False
    return False


def _apply_safe_division(expr: sp.Expr, eps: float) -> sp.Expr:
    """
    Simple sanitiser: for negative powers x**(-k), replace x by |x| + eps
    to avoid exploding derivatives at zeros / sign flips.
    """
    eps_sym = sp.Float(eps)

    def _transform(node: sp.Expr) -> sp.Expr:
        if isinstance(node, sp.Pow):
            base = _transform(node.base)
            exp = _transform(node.exp)
            if _is_negative_number(exp):
                safe_base = sp.Abs(base) + eps_sym
                return sp.Pow(safe_base, exp)
            return sp.Pow(base, exp)
        if node.args:
            new_args = tuple(_transform(arg) for arg in node.args)
            if new_args == node.args:
                return node
            return node.func(*new_args)
        return node

    return _transform(expr)


def sanitize_formula(
    formula: sp.Expr | str,
    *,
    safe_division_eps: Optional[float] = None,
) -> sp.Expr:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    if safe_division_eps is not None and safe_division_eps > 0:
        expr = _apply_safe_division(expr, safe_division_eps)
    return expr


def evaluate_formula(formula: sp.Expr | str, features: pd.DataFrame) -> np.ndarray:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    symbols = sorted({str(s) for s in expr.free_symbols})
    # Missing features default to 0
    missing = [s for s in symbols if s not in features.columns]
    for m in missing:
        features[m] = 0.0
    if not symbols:
        return np.full(len(features), float(expr))
    func = sp.lambdify(symbols, expr, "numpy")
    arrays = [features[s].to_numpy(dtype=float) for s in symbols]
    vals = func(*arrays)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals[~np.isfinite(vals)] = np.nan
    return vals


def make_formula_function(
    formula: sp.Expr | str,
    backend: str = "numpy",
) -> Tuple[Callable[..., np.ndarray], List[str]]:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    func = sp.lambdify(symbols, expr, backend)
    return func, [s.name for s in symbols]


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns are present: {candidates}")


# -----------------------------------------------------------------------------
# Dataset prep
# -----------------------------------------------------------------------------

def prepare_sr_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Mirror the training-time sanitisation (strip `_fit`, normalise names).
    """
    df = dataset.copy()
    df.columns = df.columns.str.replace("_fit$", "", regex=True)

    candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    sanitized = sanitize_feature_names(candidate_cols)
    rename_map = dict(zip(candidate_cols, sanitized))
    df.rename(columns=rename_map, inplace=True)

    target_col = sanitize_feature_names(["p-ERK1-2_dt"])[0]
    df[target_col] = pd.to_numeric(
        df.get("p-ERK1-2_dt", df.get(target_col)),
        errors="coerce",
    )

    if "marker" in df.columns:
        df["marker"] = df["marker"].astype(str)
    if "GFP_bin" in df.columns:
        df["GFP_bin"] = pd.to_numeric(df["GFP_bin"], errors="coerce")
    if "timepoint" in df.columns:
        df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    return df, target_col


# -----------------------------------------------------------------------------
# Feature-driven integration (Euler on dt at observed features)
# -----------------------------------------------------------------------------

def integrate_single_marker(
    sub: pd.DataFrame,
    target_col: str,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
    *,
    restrict_to_measured: bool = False,
    include_trajectories: bool = False,
    phase: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Compute dt R² and integrated R² for one marker using feature-driven integration:
    integrate predicted dt forward in time per GFP bin (simple Euler).
    """
    traj_records: List[Dict[str, object]] = []
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    integ_r2_vals: List[float] = []
    integ_rel_mae_vals: List[float] = []
    dt_rel_mae_vals: List[float] = []
    dt_r2_vals: List[float] = []
    integrated_bins = 0

    obs_col = _pick_column(sub, (target_col, "p-ERK1-2_dt"))
    obs_pe_candidates = ("p_ERK1_2", "p-ERK1-2")

    for b in bins:
        g = sub[sub["GFP_bin"] == b].copy()
        g.sort_values("timepoint", inplace=True)

        t = g["timepoint"].to_numpy(dtype=float)
        dt_pred = g["__y_pred__"].to_numpy(dtype=float)
        obs_dt = g[obs_col].to_numpy(dtype=float)

        mask = np.isfinite(t) & np.isfinite(dt_pred) & np.isfinite(obs_dt)
        if mask.sum() < 2:
            continue

        t, dt_pred, obs_dt = t[mask], dt_pred[mask], obs_dt[mask]
        order = np.argsort(t)
        t, dt_pred, obs_dt = t[order], dt_pred[order], obs_dt[order]

        integ = np.full_like(t, np.nan, dtype=float)

        try:
            obs_pe_col = _pick_column(g, obs_pe_candidates)
            obs_pe = (
                g.loc[g.index[mask], obs_pe_col]
                .to_numpy(dtype=float)[order]
            )
            integ[0] = obs_pe[0] if len(obs_pe) > 0 else 0.0
        except KeyError:
            obs_pe = np.full_like(t, np.nan, dtype=float)
            integ[0] = 0.0

        # Simple forward Euler integration on predicted dt
        for j in range(1, len(t)):
            dt = t[j] - t[j - 1]
            if np.isfinite(dt_pred[j - 1]) and np.isfinite(integ[j - 1]):
                integ[j] = integ[j - 1] + dt * dt_pred[j - 1]

        use_mask = dataset_mode == "per_minute" and restrict_to_measured
        meas_mask = np.isin(t, measured_timepoints) if use_mask else np.ones_like(t, dtype=bool)

        # dt R² (where defined)
        if meas_mask.sum() > 1:
            try:
                r2_dt = coefficient_of_determination(obs_dt[meas_mask], dt_pred[meas_mask])
                if np.isfinite(r2_dt):
                    dt_r2_vals.append(float(r2_dt))
            except Exception:
                pass

        rel_dt_mask = meas_mask & np.isfinite(dt_pred) & np.isfinite(obs_dt)
        if np.any(rel_dt_mask):
            try:
                denom = np.maximum(np.abs(obs_dt[rel_dt_mask]), REL_MAE_EPS)
                rel_mae = float(np.mean(np.abs(dt_pred[rel_dt_mask] - obs_dt[rel_dt_mask]) / denom))
                if np.isfinite(rel_mae):
                    dt_rel_mae_vals.append(rel_mae)
            except Exception:
                pass

        # integrated R² vs observed p-ERK
        if (
            meas_mask.sum() > 1
            and np.isfinite(integ[meas_mask]).sum() > 1
            and np.isfinite(obs_pe[meas_mask]).sum() > 1
        ):
            try:
                r2 = coefficient_of_determination(obs_pe[meas_mask], integ[meas_mask])
                if np.isfinite(r2):
                    integ_r2_vals.append(float(r2))
                    integrated_bins += 1
            except Exception:
                pass

        rel_mask = meas_mask & np.isfinite(integ) & np.isfinite(obs_pe)
        if np.any(rel_mask):
            try:
                denom = np.maximum(np.abs(obs_pe[rel_mask]), REL_MAE_EPS)
                rel_mae = float(np.mean(np.abs(integ[rel_mask] - obs_pe[rel_mask]) / denom))
                if np.isfinite(rel_mae):
                    integ_rel_mae_vals.append(rel_mae)
            except Exception:
                pass

        if include_trajectories:
            for ti, pred_dt_i, obs_dt_i, integ_i, obs_pe_i in zip(
                t, dt_pred, obs_dt, integ, obs_pe
            ):
                traj_records.append(
                    {
                        "phase": phase,
                        "GFP_bin": b,
                        "timepoint": ti,
                        "pred_dt": pred_dt_i,
                        "obs_dt": obs_dt_i,
                        "pred_integrated": integ_i,
                        "obs_pERK1_2": obs_pe_i,
                    }
                )

    dt_r2_vals = [float(v) for v in dt_r2_vals if np.isfinite(v)]
    integ_r2_vals = [float(v) for v in integ_r2_vals if np.isfinite(v)]
    integ_rel_mae_vals = [float(v) for v in integ_rel_mae_vals if np.isfinite(v)]
    dt_rel_mae_vals = [float(v) for v in dt_rel_mae_vals if np.isfinite(v)]
    dt_r2_mean_bins = float(np.mean(dt_r2_vals)) if dt_r2_vals else np.nan
    dt_r2_median_bins = float(np.median(dt_r2_vals)) if dt_r2_vals else np.nan
    dt_rel_mae_mean_bins = float(np.mean(dt_rel_mae_vals)) if dt_rel_mae_vals else np.nan
    dt_rel_mae_median_bins = float(np.median(dt_rel_mae_vals)) if dt_rel_mae_vals else np.nan
    # Legacy column keeps the clamped mean for backward compatibility
    dt_r2_mean = max(0.0, dt_r2_mean_bins) if np.isfinite(dt_r2_mean_bins) else np.nan
    integ_r2_median = float(np.median(integ_r2_vals)) if integ_r2_vals else np.nan
    if np.isfinite(integ_r2_median):
        integ_r2_median = max(0.0, integ_r2_median)
    integ_rel_mae_mean_bins = float(np.mean(integ_rel_mae_vals)) if integ_rel_mae_vals else np.nan
    integ_rel_mae_median_bins = float(np.median(integ_rel_mae_vals)) if integ_rel_mae_vals else np.nan
    metrics = {
        "dt_r2": dt_r2_mean,
        "dt_r2_mean_bins": dt_r2_mean_bins,
        "dt_r2_median_bins": dt_r2_median_bins,
        "dt_r2_valid_bins": len(dt_r2_vals),
        "dt_rel_mae_mean_bins": dt_rel_mae_mean_bins,
        "dt_rel_mae_median_bins": dt_rel_mae_median_bins,
        "dt_rel_mae_valid_bins": len(dt_rel_mae_vals),
        "integ_r2_median": integ_r2_median,
        "integ_rel_mae_mean_bins": integ_rel_mae_mean_bins,
        "integ_rel_mae_median_bins": integ_rel_mae_median_bins,
        "integ_rel_mae_valid_bins": len(integ_rel_mae_vals),
        "bins": len(bins),
        "integrated_bins": integrated_bins,
    }
    return metrics, traj_records


# -----------------------------------------------------------------------------
# Jitted diffrax solver builder
# -----------------------------------------------------------------------------

def build_jitted_ode_solver(
    t_jnp: jnp.ndarray,
    formula_fn_jax: Callable[..., jnp.ndarray],
    symbol_names: Sequence[str],
    p_symbol: str,
):
    """
    Build a jitted solver for a fixed time grid t_jnp and symbol set.

    integrate_single_bin(p0, feature_mat) expects:
        - p0: scalar initial p-ERK1-2 (JAX scalar/array)
        - feature_mat: [n_syms, T] JAX array, where row i corresponds to symbol_names[i].
          The row for p_symbol is ignored (it will be overwritten logically via the state).
    """
    t_jnp = jnp.asarray(t_jnp)
    n_syms = len(symbol_names)
    symbol_index = {name: i for i, name in enumerate(symbol_names)}
    p_idx = symbol_index[p_symbol]

    # Choose initial step from this grid
    if t_jnp.shape[0] > 1:
        min_dt = jnp.min(jnp.diff(t_jnp))
        dt0 = jnp.maximum(ODE_DT0_MIN, min_dt * ODE_DT0_SCALE)
    else:
        dt0 = ODE_DT0_MIN

    @jax.jit
    def integrate_single_bin(p0: jnp.ndarray, feature_mat: jnp.ndarray) -> jnp.ndarray:
        # feature_mat: [n_syms, T]

        def ode_fn(ti, yi, feature_mat_local):
            # yi shape: (1,)
            p_val = yi[0]

            # Interpolate all symbol time series at time ti
            def interp_row(row):
                return jnp.interp(ti, t_jnp, row)

            exog_vals = jax.vmap(interp_row)(feature_mat_local)  # [n_syms]
            exog_vals = exog_vals.at[p_idx].set(p_val)

            # formula_fn_jax still expects positional args in symbol_names order
            deriv = formula_fn_jax(*[exog_vals[i] for i in range(n_syms)])
            deriv = jnp.asarray(deriv).reshape(())
            return jnp.asarray([deriv])

        sol = diffeqsolve(
            ODETerm(ode_fn),
            Kvaerno5(),
            t0=t_jnp[0],
            t1=t_jnp[-1],
            dt0=dt0,
            y0=jnp.asarray([p0]),
            saveat=SaveAt(ts=t_jnp),
            stepsize_controller=PIDController(
                atol=ODE_ATOL,
                rtol=ODE_RTOL,
                dtmax=ODE_DTMAX,
            ),
            args=feature_mat,
            max_steps=ODE_MAX_STEPS,
            throw=False,
        )
        # sol.ys: [T, 1]
        return sol.ys[:, 0]

    return integrate_single_bin


# -----------------------------------------------------------------------------
# ODE-style integration (p-ERK1-2 as state, others as exogenous, via diffrax)
# -----------------------------------------------------------------------------

def integrate_marker_ode(
    sub: pd.DataFrame,
    target_col: str,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
    formula_fn_jax: Callable[..., jnp.ndarray],
    symbol_names: Sequence[str],
    p_symbol: str,
    *,
    restrict_to_measured: bool = False,
    include_trajectories: bool = False,
    phase: Optional[str] = None,
    log_label: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Integrate p-ERK1-2 as the sole state; other features treated as known signals.

    Uses a jitted diffrax Kvaerno5 solver shared across all bins for this marker/split,
    assuming all bins share the same time grid (true for snapshot/per-minute here).
    """
    traj_records: List[Dict[str, object]] = []
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    if not bins:
        # Treat missing bins as a failed ODE run (R2 = 0)
        return {
            "ode_integ_r2_median": 0.0,
            "ode_integ_rel_mae_mean_bins": np.nan,
            "ode_integ_rel_mae_median_bins": np.nan,
            "ode_integ_rel_mae_valid_bins": 0,
            "ode_bins": 0,
            "ode_integrated_bins": 0,
        }, []

    integ_r2_vals: List[float] = []
    integ_rel_mae_vals: List[float] = []
    integrated_bins = 0
    used_solvers: set = set()

    obs_pe_candidates = ("p_ERK1_2", "p-ERK1-2")

    # -------------------------------------------------------------------------
    # Establish canonical time grid from the first bin
    # -------------------------------------------------------------------------
    first_bin = bins[0]
    g0 = sub[sub["GFP_bin"] == first_bin].copy()
    g0.sort_values("timepoint", inplace=True)
    t0 = g0["timepoint"].to_numpy(dtype=float)
    mask0 = np.isfinite(t0) & np.isfinite(g0[target_col].to_numpy(dtype=float))
    if mask0.sum() < 2:
        return {
            "ode_integ_r2_median": 0.0,
            "ode_integ_rel_mae_mean_bins": np.nan,
            "ode_integ_rel_mae_median_bins": np.nan,
            "ode_integ_rel_mae_valid_bins": 0,
            "ode_bins": len(bins),
            "ode_integrated_bins": 0,
        }, []

    t0 = t0[mask0]
    order0 = np.argsort(t0)
    t_grid_np = t0[order0]
    t_jnp = jnp.asarray(t_grid_np)

    # Build jitted solver for this time grid + formula
    integrate_single_bin = build_jitted_ode_solver(
        t_jnp, formula_fn_jax, symbol_names, p_symbol
    )
    used_solvers.add("diffrax_kvaerno5_jitted")

    n_syms = len(symbol_names)
    symbol_index = {name: i for i, name in enumerate(symbol_names)}
    p_idx = symbol_index[p_symbol]

    # -------------------------------------------------------------------------
    # Loop over bins, reuse the same jitted solver
    # -------------------------------------------------------------------------
    for b in bins:
        g = sub[sub["GFP_bin"] == b].copy()
        g.sort_values("timepoint", inplace=True)

        t = g["timepoint"].to_numpy(dtype=float)
        obs_dt = g[target_col].to_numpy(dtype=float)

        # observed p-ERK
        try:
            obs_pe_col = _pick_column(g, obs_pe_candidates)
            obs_pe = g[obs_pe_col].to_numpy(dtype=float)
        except KeyError:
            obs_pe = np.full_like(t, np.nan, dtype=float)

        mask = np.isfinite(t) & np.isfinite(obs_dt)
        if mask.sum() < 2:
            continue

        t = t[mask]
        obs_dt = obs_dt[mask]
        obs_pe = obs_pe[mask]

        order = np.argsort(t)
        t = t[order]
        obs_dt = obs_dt[order]
        obs_pe = obs_pe[order]

        # Ensure time grid matches the canonical grid; if not, skip ODE for this bin
        if t.shape != t_grid_np.shape or not np.allclose(t, t_grid_np):
            canon_str = np.array2string(t_grid_np, precision=3, separator=",")
            obs_str = np.array2string(t, precision=3, separator=",")
            print(
                f"[compute_marker_integration] {log_label or ''} bin {b}: "
                "time grid differs from canonical; skipping ODE integration for this bin. "
                f"canonical={canon_str} observed={obs_str}"
            )
            continue

        # Initial condition: first finite observed p-ERK, else mean, else 0
        if np.isfinite(obs_pe[0]):
            p0 = float(obs_pe[0])
        else:
            finite_pe = obs_pe[np.isfinite(obs_pe)]
            p0 = float(finite_pe.mean()) if finite_pe.size else 0.0

        # Build feature matrix [n_syms, T] for this bin, in symbol_names order
        feature_mat_np = np.zeros((n_syms, t.shape[0]), dtype=float)
        for i, name in enumerate(symbol_names):
            if name == p_symbol:
                # placeholder; p(t) is injected at runtime via the state
                feature_mat_np[i, :] = 0.0
            else:
                if name in g.columns:
                    arr = (
                        pd.to_numeric(g[name], errors="coerce")
                        .to_numpy(dtype=float)[mask][order]
                    )
                    feature_mat_np[i, :] = arr
                else:
                    feature_mat_np[i, :] = 0.0

        feature_mat = jnp.asarray(feature_mat_np)
        p0_j = jnp.asarray(p0)

        # Run jitted diffrax solver for this bin
        try:
            state_vals = np.asarray(integrate_single_bin(p0_j, feature_mat), dtype=float)
        except Exception as exc:
            print(
                f"[compute_marker_integration] {log_label or ''} bin {b}: "
                f"diffrax_kvaerno5 solver failed: {exc}"
            )
            continue

        if not np.isfinite(state_vals).any():
            continue

        integ = state_vals.copy()

        # R² on integrated trajectory vs observed p-ERK
        use_mask = dataset_mode == "per_minute" and restrict_to_measured
        meas_mask = np.isin(t, measured_timepoints) if use_mask else np.ones_like(t, dtype=bool)

        if (
            meas_mask.sum() > 1
            and np.isfinite(integ[meas_mask]).sum() > 1
            and np.isfinite(obs_pe[meas_mask]).sum() > 1
        ):
            try:
                r2 = coefficient_of_determination(obs_pe[meas_mask], integ[meas_mask])
                if np.isfinite(r2):
                    integ_r2_vals.append(float(r2))
                    integrated_bins += 1
            except Exception:
                pass

        rel_mask = meas_mask & np.isfinite(integ) & np.isfinite(obs_pe)
        if np.any(rel_mask):
            try:
                denom = np.maximum(np.abs(obs_pe[rel_mask]), REL_MAE_EPS)
                rel_mae = float(np.mean(np.abs(integ[rel_mask] - obs_pe[rel_mask]) / denom))
                if np.isfinite(rel_mae):
                    integ_rel_mae_vals.append(rel_mae)
            except Exception:
                pass

        if include_trajectories:
            # Recompute dt along trajectory (using JAX formula) for logging
            dt_preds: List[float] = []
            for ti, p_val in zip(t_grid_np, state_vals):
                ti_j = jnp.asarray(ti)
                p_j = jnp.asarray(p_val)

                # Build exog values at this ti from feature_mat
                def interp_row(row):
                    return jnp.interp(ti_j, t_jnp, row)

                exog_vals = jax.vmap(interp_row)(feature_mat)  # [n_syms]
                exog_vals = exog_vals.at[p_idx].set(p_j)

                deriv_eval = formula_fn_jax(*[exog_vals[i] for i in range(n_syms)])
                deriv_eval = float(jnp.asarray(deriv_eval).reshape(()))
                dt_preds.append(deriv_eval)

            dt_preds_arr = np.asarray(dt_preds, dtype=float)
            for ti, pred_dt_i, obs_dt_i, integ_i, obs_pe_i in zip(
                t_grid_np, dt_preds_arr, obs_dt, integ, obs_pe
            ):
                traj_records.append(
                    {
                        "phase": phase,
                        "GFP_bin": b,
                        "timepoint": ti,
                        "pred_dt_ode": pred_dt_i,
                        "obs_dt": obs_dt_i,
                        "pred_integrated_ode": integ_i,
                        "obs_pERK1_2": obs_pe_i,
                    }
                )

    ode_r2_median = float(np.median(integ_r2_vals)) if integ_r2_vals else 0.0
    if np.isfinite(ode_r2_median):
        ode_r2_median = max(0.0, ode_r2_median)
    ode_rel_mae_mean_bins = float(np.mean(integ_rel_mae_vals)) if integ_rel_mae_vals else np.nan
    ode_rel_mae_median_bins = float(np.median(integ_rel_mae_vals)) if integ_rel_mae_vals else np.nan
    metrics = {
        "ode_integ_r2_median": ode_r2_median,
        "ode_integ_rel_mae_mean_bins": ode_rel_mae_mean_bins,
        "ode_integ_rel_mae_median_bins": ode_rel_mae_median_bins,
        "ode_integ_rel_mae_valid_bins": len(integ_rel_mae_vals),
        "ode_bins": len(bins),
        "ode_integrated_bins": integrated_bins,
    }
    if log_label:
        solvers = ", ".join(sorted(used_solvers)) if used_solvers else "none"
        print(f"[compute_marker_integration] {log_label}: ODE solver(s) used -> {solvers}")
    return metrics, traj_records


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.aggregate_output = _infer_aggregate_output(args)

    run_seed = resolve_seed(args.seed if args.seed is not None else args.random_state, default=42)
    split_seed = resolve_seed(args.random_state, default=run_seed)
    seed_all(run_seed)

    raw_dataset = pd.read_csv(args.dataset)
    dataset, target_col = prepare_sr_dataset(raw_dataset)

    summary = pd.read_csv(args.summary)
    summary = summary[summary["model"].isin(args.models)]
    if "dataset_mode" in summary.columns:
        summary = summary[summary["dataset_mode"] == args.dataset_mode]
    if summary.empty:
        raise ValueError("No summary rows found after filtering.")

    metrics: List[Dict[str, object]] = []
    traj_records: List[Dict[str, object]] = []

    # Symbolic name for p-ERK
    p_symbol = sanitize_feature_names(["p-ERK1-2"])[0]

    # Optional reuse of SR train/test splits from predicted_trajectories
    sr_split_map: Dict[str, Dict[str, set]] = {}
    if args.sr_trajectories is not None:
        try:
            traj_df = pd.read_csv(args.sr_trajectories)
            traj_df["GFP_bin"] = pd.to_numeric(traj_df["GFP_bin"], errors="coerce")
            traj_df["dataset_mode"] = traj_df.get("dataset_mode", args.dataset_mode)
            for marker in traj_df["marker"].unique():
                sub = traj_df[
                    (traj_df["marker"] == marker)
                    & (traj_df["dataset_mode"] == args.dataset_mode)
                ]
                train_bins = set(
                    sub[sub["phase"] == "train"]["GFP_bin"].dropna().astype(int).tolist()
                )
                test_bins = set(
                    sub[sub["phase"] == "test"]["GFP_bin"].dropna().astype(int).tolist()
                )
                if train_bins and test_bins:
                    sr_split_map[str(marker)] = {"train": train_bins, "test": test_bins}
        except Exception as exc:
            print(
                f"[compute_marker_integration] Failed to load SR trajectories for split reuse: {exc}"
            )

    for _, row in summary.iterrows():
        marker = row["group_name"]
        formula = row["formula"]
        if formula in (None, "N/A"):
            continue

        sub = dataset[dataset["marker"] == marker].copy()
        sub = sub.dropna(subset=[target_col, "GFP_bin", "timepoint"])
        if sub.empty:
            continue

        safe_eps = args.pysr_safe_division_eps if row["model"] == "PySR" else None
        expr = sanitize_formula(formula, safe_division_eps=safe_eps)

        # Train/test split on GFP bins
        if marker in sr_split_map:
            train_bins = sr_split_map[marker]["train"]
            test_bins = sr_split_map[marker]["test"]
        else:
            train_bins, test_bins = choose_bin_split(
                sub, test_size=args.test_size, random_state=split_seed
            )

        if not train_bins or not test_bins:
            print(
                f"[compute_marker_integration] Skipping marker {marker}: "
                f"unable to split train/test bins."
            )
            continue

        train_raw = sub[sub["GFP_bin"].isin(train_bins)].copy()
        test_raw = sub[sub["GFP_bin"].isin(test_bins)].copy()

        # For now we don't do any extra sampling for per-minute mode
        train_proc = train_raw
        test_proc = test_raw

        model_label = row["model"]
        seed_label = args.seed if args.seed is not None else "NA"
        print(
            f"[compute_marker_integration] seed={seed_label} model={model_label} "
            f"mode={args.dataset_mode} {marker}: "
            f"train rows {len(train_raw)} -> {len(train_proc)} | "
            f"test rows {len(test_raw)} -> {len(test_proc)}"
        )

        if test_proc.empty or train_proc.empty:
            continue

        # Build JAX formula function once per marker
        formula_fn_jax, symbol_names = make_formula_function(expr, backend="jax")

        # Shared helper to evaluate feature-driven + ODE integration for a split
        def _eval_and_metrics(
            split_name: str, frame: pd.DataFrame
        ) -> Tuple[
            Dict[str, float],
            List[Dict[str, object]],
            Dict[str, float],
            List[Dict[str, object]],
        ]:
            restrict_mask = args.dataset_mode == "per_minute" and split_name == "test"

            feats_local = frame[
                [c for c in dataset.columns if c not in EXCLUDE_COLUMNS and c != target_col]
            ].copy()
            feats_local = feats_local.apply(pd.to_numeric, errors="coerce")

            preds_dt_local = evaluate_formula(expr, feats_local)
            frame = frame.copy()
            frame["__y_pred__"] = preds_dt_local

            metrics_vals, bin_trajs = integrate_single_marker(
                frame,
                target_col,
                args.measured_timepoints,
                args.dataset_mode,
                restrict_to_measured=restrict_mask,
                include_trajectories=True,
                phase=split_name,
            )

            # ODE-style integration on p-ERK via diffrax
            try:
                ode_metrics, ode_trajs = integrate_marker_ode(
                    frame,
                    target_col,
                    args.measured_timepoints,
                    args.dataset_mode,
                    formula_fn_jax,
                    symbol_names,
                    p_symbol,
                    restrict_to_measured=restrict_mask,
                    include_trajectories=True,
                    phase=split_name,
                    log_label=f"{marker} {model_label} seed={seed_label} {split_name}",
                )
            except Exception as exc:
                print(
                    f"[compute_marker_integration] {marker} {split_name}: "
                    f"ODE integration failed, skipping ODE metrics: {exc}"
                )
                ode_metrics, ode_trajs = {
                    "ode_integ_r2_median": 0.0,
                    "ode_integ_rel_mae_mean_bins": np.nan,
                    "ode_integ_rel_mae_median_bins": np.nan,
                    "ode_integ_rel_mae_valid_bins": 0,
                    "ode_bins": 0,
                    "ode_integrated_bins": 0,
                }, []

            return metrics_vals, bin_trajs, ode_metrics, ode_trajs

        metrics_train, traj_train, ode_train, traj_ode_train = _eval_and_metrics(
            "train", train_proc
        )
        metrics_test, traj_test, ode_test, traj_ode_test = _eval_and_metrics(
            "test", test_proc
        )

        metrics_row = {
            "marker": marker,
            "model": row["model"],
            "dataset_mode": args.dataset_mode,
            "seed": args.seed,
            "formula": formula,
            "dt_r2_train": metrics_train.get("dt_r2"),
            "dt_rel_mae_mean_bins_train": metrics_train.get("dt_rel_mae_mean_bins"),
            "dt_rel_mae_median_bins_train": metrics_train.get("dt_rel_mae_median_bins"),
            "dt_rel_mae_valid_bins_train": metrics_train.get("dt_rel_mae_valid_bins"),
            "integ_r2_median_train": metrics_train.get("integ_r2_median"),
            "integ_rel_mae_mean_bins_train": metrics_train.get("integ_rel_mae_mean_bins"),
            "integ_rel_mae_median_bins_train": metrics_train.get("integ_rel_mae_median_bins"),
            "integ_rel_mae_valid_bins_train": metrics_train.get("integ_rel_mae_valid_bins"),
            "ode_integ_r2_median_train": ode_train.get("ode_integ_r2_median"),
            "ode_integ_rel_mae_mean_bins_train": ode_train.get("ode_integ_rel_mae_mean_bins"),
            "ode_integ_rel_mae_median_bins_train": ode_train.get("ode_integ_rel_mae_median_bins"),
            "ode_integ_rel_mae_valid_bins_train": ode_train.get("ode_integ_rel_mae_valid_bins"),
        }
        metrics_row.update(metrics_test)
        metrics_row["ode_integ_r2_median"] = ode_test.get("ode_integ_r2_median")
        metrics_row["ode_integ_rel_mae_mean_bins"] = ode_test.get("ode_integ_rel_mae_mean_bins")
        metrics_row["ode_integ_rel_mae_median_bins"] = ode_test.get("ode_integ_rel_mae_median_bins")
        metrics_row["ode_integ_rel_mae_valid_bins"] = ode_test.get("ode_integ_rel_mae_valid_bins")
        metrics.append(metrics_row)

        # Merge trajectories with phase labels
        for payload in (
            traj_train + traj_test + traj_ode_train + traj_ode_test
        ):
            payload.update(
                {
                    "marker": marker,
                    "model": row["model"],
                    "dataset_mode": args.dataset_mode,
                    "seed": args.seed,
                }
            )
            traj_records.append(payload)

    # Write per-run metrics
    out_df = pd.DataFrame(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    model_note = ",".join(args.models) if getattr(args, "models", None) else "unknown"
    seed_note = args.seed if args.seed is not None else "NA"

    # Maintain combined + aggregated metrics across seeds if requested
    if args.aggregate_output is None:
        agg_all_path = args.output.with_name(args.output.stem + "_all_seeds.csv")
        agg_mean_path = agg_all_path.with_name(agg_all_path.stem + "_mean.csv")
    else:
        agg_all_path = args.aggregate_output
        agg_mean_path = agg_all_path.with_name(agg_all_path.stem + "_mean.csv")

    try:
        agg_all_path.parent.mkdir(parents=True, exist_ok=True)
        if agg_all_path.exists():
            existing = pd.read_csv(agg_all_path)
            combined = pd.concat([existing, out_df], ignore_index=True)
        else:
            combined = out_df.copy()
        combined.to_csv(agg_all_path, index=False)

        keys = [
            c
            for c in ["marker", "model", "dataset_mode"]
            if c in combined.columns
        ]
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "seed"]

        agg_mean = (
            combined.groupby(keys, dropna=False)[numeric_cols]
            .mean()
            .reset_index()
        )

        if "formula" in combined.columns and "formula" not in agg_mean.columns:
            first_formulas = (
                combined.groupby(keys, dropna=False)["formula"]
                .first()
                .reset_index()
            )
            agg_mean = agg_mean.merge(first_formulas, on=keys, how="left")

        agg_mean.to_csv(agg_mean_path, index=False)
        if args.aggregate_output is not None:
            canonical_mean = agg_mean_path.with_name(Path(args.output).name)
            agg_mean.to_csv(canonical_mean, index=False)
        print(f"Wrote combined seed metrics to {agg_all_path} | models={model_note}")
        print(f"Wrote mean-over-seeds metrics to {agg_mean_path} | models={model_note}")
    except Exception as exc:
        print(
            f"[compute_marker_integration] Failed to aggregate metrics across seeds: {exc}"
        )

    # Trajectories
    traj_df = pd.DataFrame(traj_records)
    args.trajectories_output.parent.mkdir(parents=True, exist_ok=True)
    traj_df.to_csv(args.trajectories_output, index=False)
    print(f"Wrote integration metrics to {args.output} | seed={seed_note} | models={model_note}")
    print(f"Wrote integrated trajectories to {args.trajectories_output} | seed={seed_note} | models={model_note}")


if __name__ == "__main__":
    main()
