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
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sympy as sp

from jax import config as jax_config  # type: ignore
jax_config.update("jax_enable_x64", True)
import jax.numpy as jnp  # type: ignore
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve, Kvaerno5  # type: ignore

try:
    from scipy.integrate import solve_ivp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    solve_ivp = None  # type: ignore

# Tolerances / initial step for the ODE-style integration (Kvaerno5)
ODE_TSIT5_ATOL = 1e-6
ODE_TSIT5_RTOL = 1e-4
ODE_TSIT5_DT0_MIN = 1e-3
ODE_TSIT5_DT0_SCALE = 0.2  # fraction of the smallest dt to start with
ODE_TSIT5_MAX_STEPS = 1e5
ODE_TSIT5_DERIV_CLIP = 1e6
ODE_TSIT5_STATE_MIN = 0.0
ODE_TSIT5_STATE_MAX = 1e6
ODE_FEATURE_CLIP = (0.01, 0.99)  # percentile clipping for ODE inputs
ODE_RADAU_RTOL = 1e-4
ODE_RADAU_ATOL = 1e-7
ODE_STATE_PERCENTILES = (0.01, 0.99)
ODE_STATE_MARGIN_FRAC = 0.05
ODE_DERIV_PERCENTILES = (0.01, 0.99)
ODE_DERIV_MARGIN_FRAC = 0.25
PY_SR_SAFE_DIVISION_EPS = 1e-3
STATE_TRANSFORM_CHOICES = ("identity", "log", "logit")
STATE_TRANSFORM_LOG_EPS = 1e-3
STATE_TRANSFORM_LOGIT_EPS = 1e-3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.sr_pipeline.run_functional_groups import (
    EXCLUDE_COLUMNS,
    choose_bin_split,
    sanitize_feature_names,
)
from experimental.sr_pipeline.metrics import coefficient_of_determination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute integration metrics from SR summary and dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="CSV used for SR (snapshot or per-minute).")
    parser.add_argument("--summary", type=Path, required=True, help="functional_group_summary.csv from SR run.")
    parser.add_argument("--output", type=Path, required=True, help="CSV path for integration metrics.")
    parser.add_argument("--trajectories-output", type=Path, required=True, help="CSV path for integrated trajectories.")
    parser.add_argument(
        "--sr-trajectories",
        type=Path,
        default=None,
        help="Optional predicted_trajectories CSV from SR (reuse its train/test split instead of re-splitting).",
    )
    parser.add_argument("--dataset-mode", choices=("snapshot", "per_minute"), default="snapshot")
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
        "--prefer-radau",
        action="store_true",
        help="Prefer SciPy Radau fallback for ODE-style integration (PySR) when available.",
    )
    parser.add_argument(
        "--ode-state-transform",
        choices=STATE_TRANSFORM_CHOICES,
        default="identity",
        help="Transform applied to the p-ERK1-2 state during ODE integration (identity, log, logit).",
    )
    parser.add_argument(
        "--ode-state-log-eps",
        type=float,
        default=STATE_TRANSFORM_LOG_EPS,
        help="ε added inside log(p + ε) when using the log transform.",
    )
    parser.add_argument(
        "--ode-state-logit-eps",
        type=float,
        default=STATE_TRANSFORM_LOGIT_EPS,
        help="ε for clipping scaled states before the logit transform.",
    )
    parser.add_argument(
        "--pysr-safe-division-eps",
        type=float,
        default=PY_SR_SAFE_DIVISION_EPS,
        help="ε added to denominators when sanitising PySR formulas.",
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
        help="Random seed used for bin split and late sampling.",
    )
    parser.add_argument(
        "--per-minute-max-time",
        type=float,
        default=60.0,
        help="Per-minute sampling: keep rows with timepoint ≤ this value.",
    )
    parser.add_argument(
        "--per-minute-sampling-strategy",
        choices=("max_time", "early_plus_sparse_late"),
        default="early_plus_sparse_late",
        help="Strategy for per-minute downsampling applied to the training split.",
    )
    parser.add_argument(
        "--late-sample-window",
        nargs=2,
        type=float,
        default=(30.0, 60.0),
        metavar=("START", "END"),
        help="Late window for early_plus_sparse_late sampling (inclusive).",
    )
    parser.add_argument(
        "--late-sample-points",
        type=int,
        default=15,
        help="Points to keep per marker/bin in the late window for early_plus_sparse_late sampling.",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Optional path to maintain aggregated metrics across seeds (mean over numeric columns).",
    )
    return parser.parse_args()


def _is_negative_number(value: sp.Basic) -> bool:
    if value.is_number:
        try:
            return float(value) < 0
        except Exception:
            return False
    return False


def _apply_safe_division(expr: sp.Expr, eps: float) -> sp.Expr:
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


def sanitize_formula(formula: sp.Expr | str, *, safe_division_eps: Optional[float] = None) -> sp.Expr:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    if safe_division_eps is not None and safe_division_eps > 0:
        expr = _apply_safe_division(expr, safe_division_eps)
    return expr


def _compute_percentile_interval(
    values: np.ndarray,
    percentiles: Tuple[float, float],
    margin_fraction: float,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    lo_pct = np.clip(percentiles[0], 0.0, 1.0) * 100.0
    hi_pct = np.clip(percentiles[1], 0.0, 1.0) * 100.0
    try:
        lo = float(np.nanpercentile(arr, lo_pct))
        hi = float(np.nanpercentile(arr, hi_pct))
    except Exception:
        return fallback
    if not np.isfinite(lo) or not np.isfinite(hi):
        return fallback
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    reference = span if span > 0 else max(abs(lo), abs(hi), 1.0)
    pad = reference * margin_fraction
    lo -= pad
    hi += pad
    if lo == hi:
        hi = lo + max(1.0, abs(lo)) * 0.1
    return float(lo), float(hi)


def _logit(prob: float) -> float:
    prob = float(np.clip(prob, 1e-9, 1 - 1e-9))
    return float(np.log(prob / (1.0 - prob)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class StateTransform:
    name: str
    actual_bounds: Tuple[float, float]
    log_eps: float = STATE_TRANSFORM_LOG_EPS
    logit_eps: float = STATE_TRANSFORM_LOGIT_EPS

    def __post_init__(self) -> None:
        name = (self.name or "identity").lower()
        lo, hi = self.actual_bounds
        if not np.isfinite(lo):
            lo = ODE_TSIT5_STATE_MIN
        if not np.isfinite(hi):
            hi = ODE_TSIT5_STATE_MAX
        if hi <= lo:
            hi = lo + max(1.0, abs(lo)) * 0.1
        self.actual_bounds = (float(lo), float(hi))
        self.name = name if name in STATE_TRANSFORM_CHOICES else "identity"
        self.scale = float(self.actual_bounds[1] - self.actual_bounds[0])
        if self.name == "identity":
            self.state_bounds = self.actual_bounds
        elif self.name == "log":
            max_arg = self.actual_bounds[1] + self.log_eps
            if max_arg <= 0:
                raise ValueError("Log transform requires positive p-ERK1-2 values.")
            min_arg = max(self.actual_bounds[0] + self.log_eps, self.log_eps)
            if min_arg <= 0:
                raise ValueError("Log transform requires positive lower bound.")
            self.state_bounds = (float(np.log(min_arg)), float(np.log(max_arg)))
        elif self.name == "logit":
            span = self.actual_bounds[1] - self.actual_bounds[0]
            if span <= 0:
                raise ValueError("Logit transform requires a positive span.")
            self.scale = float(span)
            self.state_bounds = (
                _logit(self.logit_eps),
                _logit(1.0 - self.logit_eps),
            )
        else:
            raise ValueError(f"Unsupported state transform '{self.name}'")

    def clip_actual(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.clip(arr, self.actual_bounds[0], self.actual_bounds[1])

    def clip_state(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.clip(arr, self.state_bounds[0], self.state_bounds[1])

    def forward(self, values: np.ndarray | float) -> np.ndarray:
        arr = self.clip_actual(values)
        if self.name == "identity":
            return arr
        if self.name == "log":
            return np.log(np.clip(arr + self.log_eps, self.log_eps, None))
        if self.name == "logit":
            scaled = (arr - self.actual_bounds[0]) / self.scale
            scaled = np.clip(scaled, self.logit_eps, 1.0 - self.logit_eps)
            return np.log(scaled / (1.0 - scaled))
        raise ValueError(f"Unsupported state transform '{self.name}'")

    def inverse(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if self.name == "identity":
            recovered = arr
        elif self.name == "log":
            recovered = np.exp(arr) - self.log_eps
        elif self.name == "logit":
            recovered = self.actual_bounds[0] + self.scale * _sigmoid(arr)
        else:
            raise ValueError(f"Unsupported state transform '{self.name}'")
        return self.clip_actual(recovered)


class NoClipStateTransform:
    """Pass-through transform used when state clipping is disabled."""

    name = "identity"
    actual_bounds = (np.nan, np.nan)
    state_bounds = (np.nan, np.nan)

    def clip_actual(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def clip_state(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def forward(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def inverse(self, values: np.ndarray | float) -> np.ndarray:
        return np.asarray(values, dtype=float)

def build_state_transform(
    kind: str,
    bounds: Tuple[float, float],
    *,
    log_eps: float,
    logit_eps: float,
) -> StateTransform:
    try:
        return StateTransform(kind, bounds, log_eps=log_eps, logit_eps=logit_eps)
    except ValueError as exc:
        print(
            f"[compute_marker_integration] Falling back to identity state transform for bounds {bounds}: {exc}"
        )
        return StateTransform("identity", bounds, log_eps=log_eps, logit_eps=logit_eps)


def evaluate_formula(formula: sp.Expr | str, features: pd.DataFrame) -> np.ndarray:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    symbols = sorted({str(s) for s in expr.free_symbols})
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


def make_formula_function(formula: sp.Expr | str, backend: str = "numpy") -> Tuple[Callable[..., np.ndarray], List[str]]:
    expr = formula if isinstance(formula, sp.Expr) else sp.sympify(formula)
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    func = sp.lambdify(symbols, expr, backend)
    return func, [s.name for s in symbols]


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns are present: {candidates}")


def prepare_sr_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Mirror the training-time sanitisation (strip `_fit`, normalise names)."""
    df = dataset.copy()
    df.columns = df.columns.str.replace("_fit$", "", regex=True)

    candidate_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    sanitized = sanitize_feature_names(candidate_cols)
    rename_map = dict(zip(candidate_cols, sanitized))
    df.rename(columns=rename_map, inplace=True)

    target_col = sanitize_feature_names(["p-ERK1-2_dt"])[0]
    df[target_col] = pd.to_numeric(df.get("p-ERK1-2_dt", df.get(target_col)), errors="coerce")

    if "marker" in df.columns:
        df["marker"] = df["marker"].astype(str)
    if "GFP_bin" in df.columns:
        df["GFP_bin"] = pd.to_numeric(df["GFP_bin"], errors="coerce")
    if "timepoint" in df.columns:
        df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    return df, target_col


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
    """Compute dt R² and integrated R² for one marker (feature-driven integration)."""
    traj_records: List[Dict[str, object]] = []
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    integ_r2_vals: List[float] = []
    integrated_bins = 0

    def _clip_array(arr: np.ndarray) -> np.ndarray:
        if feature_clip_percentiles is None or arr.size == 0:
            return arr
        lo, hi = feature_clip_percentiles
        lo_val = np.nanpercentile(arr, lo * 100.0)
        hi_val = np.nanpercentile(arr, hi * 100.0)
        return np.clip(arr, lo_val, hi_val)
    obs_col = _pick_column(sub, (target_col, "p-ERK1-2_dt"))
    obs_pe_candidates = ("p_ERK1_2", "p-ERK1-2")
    dt_r2_vals: List[float] = []

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
        obs_pe = None
        try:
            obs_pe_col = _pick_column(g, obs_pe_candidates)
            obs_pe = g.loc[g.index[mask], obs_pe_col].to_numpy(dtype=float)[order]
            integ[0] = obs_pe[0] if len(obs_pe) > 0 else 0.0
        except KeyError:
            integ[0] = 0.0

        for j in range(1, len(t)):
            dt = t[j] - t[j - 1]
            if np.isfinite(dt_pred[j - 1]) and np.isfinite(integ[j - 1]):
                integ[j] = integ[j - 1] + dt * dt_pred[j - 1]

        use_mask = dataset_mode == "per_minute" and restrict_to_measured
        meas_mask = np.isin(t, measured_timepoints) if use_mask else np.ones_like(t, dtype=bool)
        if meas_mask.sum() > 1:
            try:
                r2_dt = coefficient_of_determination(obs_dt[meas_mask], dt_pred[meas_mask])
                if np.isfinite(r2_dt):
                    dt_r2_vals.append(float(r2_dt))
            except Exception:
                pass
        if (
            meas_mask.sum() > 1
            and np.isfinite(integ[meas_mask]).sum() > 1
            and obs_pe is not None
        ):
            try:
                r2 = coefficient_of_determination(obs_pe[meas_mask], integ[meas_mask])
                if np.isfinite(r2):
                    integ_r2_vals.append(float(r2))
                    integrated_bins += 1
            except Exception:
                pass

        if include_trajectories:
            for ti, pred_dt_i, obs_dt_i, integ_i, obs_pe_i in zip(
                t, dt_pred, obs_dt, integ, obs_pe if obs_pe is not None else np.full_like(t, np.nan)
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

    median_integ_r2 = float(np.median(integ_r2_vals)) if integ_r2_vals else np.nan
    dt_r2 = float(np.mean(dt_r2_vals)) if dt_r2_vals else np.nan
    metrics = {
        "dt_r2": dt_r2,
        "integ_r2_median": median_integ_r2,
        "bins": len(bins),
        "integrated_bins": integrated_bins,
    }
    return metrics, traj_records


def integrate_marker_ode(
    sub: pd.DataFrame,
    target_col: str,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
    formula_fn: Callable[..., np.ndarray],
    symbol_names: Sequence[str],
    p_symbol: str,
    formula_fn_fallback: Optional[Callable[..., np.ndarray]] = None,
    *,
    feature_clip_percentiles: Optional[Tuple[float, float]] = None,
    anchor_alpha: float = 0.0,
    anchor_adaptive_scale: float = 0.0,
    state_transform: Optional[StateTransform] = None,
    deriv_clip: Optional[Tuple[float, float]] = None,
    prefer_radau: bool = False,
    restrict_to_measured: bool = False,
    include_trajectories: bool = False,
    phase: Optional[str] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Integrate p-ERK1-2 as the sole state; other features treated as known signals."""
    traj_records: List[Dict[str, object]] = []
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    integ_r2_vals: List[float] = []
    integrated_bins = 0
    state_transform = state_transform or StateTransform(
        "identity", (ODE_TSIT5_STATE_MIN, ODE_TSIT5_STATE_MAX)
    )
    deriv_clip = deriv_clip or (-ODE_TSIT5_DERIV_CLIP, ODE_TSIT5_DERIV_CLIP)

    def _clip_array(arr: np.ndarray) -> np.ndarray:
        if feature_clip_percentiles is None or arr.size == 0:
            return arr
        lo, hi = feature_clip_percentiles
        lo_val = np.nanpercentile(arr, lo * 100.0)
        hi_val = np.nanpercentile(arr, hi * 100.0)
        return np.clip(arr, lo_val, hi_val)

    def _to_float(val: np.ndarray | float) -> float:
        arr = np.asarray(val, dtype=float)
        return float(arr.reshape(-1)[0])

    def _run_radau(
        t_local: np.ndarray,
        current_state_local: float,
        mask_local: np.ndarray,
        order_local: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray], List[float], Optional[np.ndarray]]:
        if solve_ivp is None:
            return False, None, [], None
        feature_arrays_np: Dict[str, np.ndarray] = {}
        for name in symbol_names:
            if name == p_symbol:
                continue
            if name in g.columns:
                arr_np = (
                    pd.to_numeric(g.loc[g.index[mask_local], name], errors="coerce")
                    .to_numpy(dtype=float)[order_local]
                )
                feature_arrays_np[name] = _clip_array(arr_np)
            else:
                feature_arrays_np[name] = np.zeros_like(t_local, dtype=float)

        def _rhs(ti, yi):
            vals_eval: List[float] = []
            state_val = float(state_transform.clip_state(yi[0]))
            for name in symbol_names:
                if name == p_symbol:
                    vals_eval.append(state_val)
                else:
                    arr = feature_arrays_np.get(name)
                    vals_eval.append(float(np.interp(ti, np.asarray(t_local), np.asarray(arr))))
            deriv_val = float(
                formula_fn_fallback(*vals_eval) if formula_fn_fallback else formula_fn(*vals_eval)
            )
            deriv_val = float(np.clip(deriv_val, deriv_clip[0], deriv_clip[1]))
            return [deriv_val]

        sol = solve_ivp(
            _rhs,
            (t_local[0], t_local[-1]),
            [current_state_local],
            t_eval=t_local,
            method="Radau",
            rtol=ODE_RADAU_RTOL,
            atol=ODE_RADAU_ATOL,
        )
        state_vals = state_transform.clip_state(np.asarray(sol.y[0], dtype=float))
        integ_r = state_transform.inverse(state_vals)
        dt_preds_r: List[float] = []
        for ti, state_val in zip(t_local[:-1], state_vals[:-1]):
            vals_eval: List[float] = []
            for name in symbol_names:
                if name == p_symbol:
                    vals_eval.append(float(state_val))
                else:
                    vals_eval.append(float(np.interp(ti, np.asarray(t_local), np.asarray(feature_arrays_np[name]))))
            try:
                deriv_eval = float(
                    formula_fn_fallback(*vals_eval) if formula_fn_fallback else formula_fn(*vals_eval)
                )
                deriv_eval = float(np.clip(deriv_eval, deriv_clip[0], deriv_clip[1]))
            except Exception:
                deriv_eval = np.nan
            dt_preds_r.append(deriv_eval)
        return True, integ_r, dt_preds_r, state_vals

    for b in bins:
        g = sub[sub["GFP_bin"] == b].copy()
        g.sort_values("timepoint", inplace=True)
        t = g["timepoint"].to_numpy(dtype=float)
        obs_dt = g[target_col].to_numpy(dtype=float)
        obs_pe = None
        try:
            obs_pe_col = _pick_column(g, ("p_ERK1_2", "p-ERK1-2"))
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
        t, obs_dt, obs_pe = t[order], obs_dt[order], obs_pe[order]
        obs_pe = state_transform.clip_actual(obs_pe)

        if np.isfinite(obs_pe[0]):
            current_p = float(obs_pe[0])
        else:
            current_p = float(np.mean(state_transform.actual_bounds))
        current_p = float(state_transform.clip_actual(current_p))
        current_state = _to_float(state_transform.forward(current_p))
        integ = np.full_like(t, np.nan, dtype=float)
        state_history = np.full_like(t, np.nan, dtype=float)
        integ[0] = current_p
        state_history[0] = current_state
        dt_preds: List[float] = []
        solver_available = (
            jnp is not None
            and ODETerm is not None
            and SaveAt is not None
            and diffeqsolve is not None
            and Kvaerno5 is not None
            and PIDController is not None
        )

        radau_used = False
        if prefer_radau:
            success, integ_r, dt_preds_r, state_vals_r = _run_radau(t, current_state, mask, order)
            if success and integ_r is not None and state_vals_r is not None:
                integ = integ_r
                state_history = state_vals_r
                dt_preds = dt_preds_r
                radau_used = True

        if not radau_used and solver_available:
            try:
                t_jnp = jnp.asarray(t, dtype=float)
                feature_arrays: Dict[str, jnp.ndarray] = {}
                for name in symbol_names:
                    if name == p_symbol:
                        continue
                    if name in g.columns:
                        arr = (
                            pd.to_numeric(g.loc[g.index[mask], name], errors="coerce")
                            .to_numpy(dtype=float)[order]
                        )
                        arr = _clip_array(arr)
                        feature_arrays[name] = jnp.asarray(arr, dtype=float)

                def _ode_fn(ti, yi, _args):
                    vals: List[jnp.ndarray] = []
                    state_val = jnp.clip(yi[0], state_transform.state_bounds[0], state_transform.state_bounds[1])
                    for name in symbol_names:
                        if name == p_symbol:
                            vals.append(state_val)
                        else:
                            arr = feature_arrays.get(name)
                            if arr is None:
                                vals.append(jnp.array(0.0))
                            else:
                                vals.append(jnp.interp(ti, t_jnp, arr))
                    deriv = formula_fn(*vals)
                    deriv = jnp.clip(deriv, deriv_clip[0], deriv_clip[1])
                    return jnp.asarray([deriv], dtype=float)

                if len(t) > 1:
                    min_dt = jnp.min(jnp.diff(t_jnp))
                    dt0 = jnp.maximum(ODE_TSIT5_DT0_MIN, min_dt * ODE_TSIT5_DT0_SCALE)
                else:
                    dt0 = ODE_TSIT5_DT0_MIN
                sol = diffeqsolve(
                    ODETerm(_ode_fn),
                    Kvaerno5(),
                    t0=t_jnp[0],
                    t1=t_jnp[-1],
                    dt0=dt0,
                    y0=jnp.asarray([current_state], dtype=float),
                    saveat=SaveAt(ts=t_jnp),
                    stepsize_controller=PIDController(atol=ODE_TSIT5_ATOL, rtol=ODE_TSIT5_RTOL),
                    max_steps=ODE_TSIT5_MAX_STEPS,
                    throw=False,
                )
                state_vals = state_transform.clip_state(np.asarray(sol.ys[:, 0], dtype=float))
                integ = state_transform.inverse(state_vals)
                state_history = state_vals
                if not np.isfinite(integ).any():
                    raise ValueError("diffrax integration returned no finite values")
                dt_preds = []
                for ti, state_val in zip(t, state_vals):
                    vals_eval: List[float] = []
                    for name in symbol_names:
                        if name == p_symbol:
                            vals_eval.append(float(state_val))
                        else:
                            arr = feature_arrays.get(name)
                            if arr is None:
                                vals_eval.append(0.0)
                            else:
                                vals_eval.append(float(np.interp(ti, np.asarray(t), np.asarray(arr))))
                    try:
                        deriv_eval = float(formula_fn(*vals_eval))
                        deriv_eval = float(np.clip(deriv_eval, deriv_clip[0], deriv_clip[1]))
                        dt_preds.append(deriv_eval)
                    except Exception:
                        dt_preds.append(np.nan)
                if dt_preds:
                    dt_preds = dt_preds[:-1]
            except Exception:
                solver_available = False

        if not radau_used and not solver_available:
            success, integ_r, dt_preds_r, state_vals_r = _run_radau(t, current_state, mask, order)
            if success and integ_r is not None and state_vals_r is not None:
                integ = integ_r
                state_history = state_vals_r
                dt_preds = dt_preds_r
                radau_used = True

        if not radau_used:
            dt_preds = []
            integ[0] = current_p
            state_history[0] = current_state
            for j in range(len(t) - 1):
                row = g.iloc[mask.nonzero()[0][order[j]]]
                vals_fb: List[float] = []
                state_val = state_history[j]
                for name in symbol_names:
                    if name == p_symbol:
                        vals_fb.append(float(state_transform.clip_state(state_val)))
                    else:
                        val = float(row.get(name, 0.0))
                        vals_fb.append(val)
                try:
                    func_fb = formula_fn_fallback or formula_fn
                    deriv = float(func_fb(*vals_fb))
                    deriv = float(np.clip(deriv, deriv_clip[0], deriv_clip[1]))
                except Exception:
                    deriv = np.nan
                dt_preds.append(deriv)
                dt = t[j + 1] - t[j]
                if np.isfinite(deriv) and np.isfinite(state_val):
                    next_state = float(state_transform.clip_state(state_val + dt * deriv))
                    state_history[j + 1] = next_state
                    integ[j + 1] = _to_float(state_transform.inverse(next_state))

        if (anchor_alpha > 0 or anchor_adaptive_scale > 0) and np.isfinite(obs_pe).any():
            span = max(state_transform.actual_bounds[1] - state_transform.actual_bounds[0], 1e-6)
            for idx, ti in enumerate(t):
                if dataset_mode == "per_minute" and not np.isin(ti, measured_timepoints):
                    continue
                if not (np.isfinite(obs_pe[idx]) and np.isfinite(integ[idx])):
                    continue
                diff = abs(obs_pe[idx] - integ[idx])
                alpha_local = anchor_alpha
                if anchor_adaptive_scale > 0:
                    alpha_local += anchor_adaptive_scale * (diff / span)
                alpha_local = float(np.clip(alpha_local, 0.0, 1.0))
                if alpha_local <= 0:
                    continue
                blended = (1.0 - alpha_local) * integ[idx] + alpha_local * obs_pe[idx]
                integ[idx] = float(state_transform.clip_actual(blended))
                state_history[idx] = _to_float(state_transform.forward(integ[idx]))

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

        if include_trajectories:
            dt_preds_arr = np.asarray(dt_preds + [np.nan], dtype=float)
            for ti, pred_dt_i, obs_dt_i, integ_i, obs_pe_i in zip(
                t, dt_preds_arr, obs_dt, integ, obs_pe
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

    metrics = {
        "ode_integ_r2_median": float(np.median(integ_r2_vals)) if integ_r2_vals else np.nan,
        "ode_bins": len(bins),
        "ode_integrated_bins": integrated_bins,
    }
    return metrics, traj_records


def main() -> None:
    args = parse_args()
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

    p_symbol = sanitize_feature_names(["p-ERK1-2"])[0]

    sr_split_map: Dict[str, Dict[str, set]] = {}
    if args.sr_trajectories is not None:
        try:
            traj_df = pd.read_csv(args.sr_trajectories)
            traj_df["GFP_bin"] = pd.to_numeric(traj_df["GFP_bin"], errors="coerce")
            traj_df["dataset_mode"] = traj_df.get("dataset_mode", args.dataset_mode)
            for marker in traj_df["marker"].unique():
                sub = traj_df[(traj_df["marker"] == marker) & (traj_df["dataset_mode"] == args.dataset_mode)]
                train_bins = set(sub[sub["phase"] == "train"]["GFP_bin"].dropna().astype(int).tolist())
                test_bins = set(sub[sub["phase"] == "test"]["GFP_bin"].dropna().astype(int).tolist())
                if train_bins and test_bins:
                    sr_split_map[str(marker)] = {"train": train_bins, "test": test_bins}
        except Exception as exc:
            print(f"[compute_marker_integration] Failed to load SR trajectories for split reuse: {exc}")

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
        if marker in sr_split_map:
            train_bins = sr_split_map[marker]["train"]
            test_bins = sr_split_map[marker]["test"]
        else:
            train_bins, test_bins = choose_bin_split(sub, test_size=args.test_size, random_state=args.random_state)
        if not train_bins or not test_bins:
            print(f"[compute_marker_integration] Skipping marker {marker}: unable to split train/test bins.")
            continue

        train_raw = sub[sub["GFP_bin"].isin(train_bins)].copy()
        test_raw = sub[sub["GFP_bin"].isin(test_bins)].copy()

        if args.dataset_mode == "per_minute":
            train_proc = train_raw
            test_proc = test_raw
        else:
            train_proc = train_raw
            test_proc = test_raw

        print(
            f"[compute_marker_integration] {marker}: train rows {len(train_raw)} -> {len(train_proc)} | "
            f"test rows {len(test_raw)} -> {len(test_proc)}"
        )

        if test_proc.empty or train_proc.empty:
            continue

        obs_p_candidates = (p_symbol, "p_ERK1_2", "p-ERK1-2")
        obs_p_series = None
        for col in obs_p_candidates:
            if col in train_proc.columns:
                obs_p_series = pd.to_numeric(train_proc[col], errors="coerce")
                break
        if obs_p_series is None:
            obs_p_series = pd.Series(np.nan, index=train_proc.index, dtype=float)
        obs_p_values = obs_p_series.to_numpy(dtype=float)
        state_bounds = _compute_percentile_interval(
            obs_p_values,
            ODE_STATE_PERCENTILES,
            ODE_STATE_MARGIN_FRAC,
            (ODE_TSIT5_STATE_MIN, ODE_TSIT5_STATE_MAX),
        )
        state_transform = NoClipStateTransform()

        def _eval_and_metrics(split_name: str, frame: pd.DataFrame) -> Tuple[Dict[str, float], List[Dict[str, object]], Dict[str, float], List[Dict[str, object]]]:
            restrict_mask = args.dataset_mode == "per_minute" and split_name == "test"
            feats_local = frame[
                [c for c in dataset.columns if c not in EXCLUDE_COLUMNS and c != target_col]
            ].copy()
            feats_local = feats_local.apply(pd.to_numeric, errors="coerce")
            preds_dt_raw_local = evaluate_formula(expr, feats_local)
            preds_dt_local = np.clip(preds_dt_raw_local, deriv_clip[0], deriv_clip[1])
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
            try:
                if jnp is not None:
                    formula_fn_jax, symbols = make_formula_function(expr, backend="jax")
                    formula_fn_np, _ = make_formula_function(expr, backend="numpy")
                    ode_formula_fn = formula_fn_jax
                    fallback_formula_fn = formula_fn_np
                else:
                    ode_formula_fn, symbols = make_formula_function(expr, backend="numpy")
                    fallback_formula_fn = ode_formula_fn
                ode_metrics, ode_trajs = integrate_marker_ode(
                    frame,
                    target_col,
                    args.measured_timepoints,
                    args.dataset_mode,
                    ode_formula_fn,
                    symbols,
                    p_symbol,
                    formula_fn_fallback=fallback_formula_fn,
                    feature_clip_percentiles=None,
                    anchor_alpha=0.0,
                    anchor_adaptive_scale=0.0,
                    state_transform=state_transform,
                    deriv_clip=deriv_clip,
                    prefer_radau=(row["model"] == "PySR" and args.prefer_radau),
                    restrict_to_measured=restrict_mask,
                    include_trajectories=True,
                    phase=split_name,
                )
            except Exception:
                ode_metrics, ode_trajs = {"ode_integ_r2_median": np.nan, "ode_bins": 0, "ode_integrated_bins": 0}, []
            return metrics_vals, bin_trajs, ode_metrics, ode_trajs

        # Derivative clip from training split
        feats_train = train_proc[
            [c for c in dataset.columns if c not in EXCLUDE_COLUMNS and c != target_col]
        ].copy()
        feats_train = feats_train.apply(pd.to_numeric, errors="coerce")
        preds_dt_raw_train = evaluate_formula(expr, feats_train)
        deriv_clip = _compute_percentile_interval(
            preds_dt_raw_train,
            ODE_DERIV_PERCENTILES,
            ODE_DERIV_MARGIN_FRAC,
            (-ODE_TSIT5_DERIV_CLIP, ODE_TSIT5_DERIV_CLIP),
        )

        metrics_train, traj_train, ode_train, traj_ode_train = _eval_and_metrics("train", train_proc)
        metrics_test, traj_test, ode_test, traj_ode_test = _eval_and_metrics("test", test_proc)

        metrics_row = {
            "marker": marker,
            "model": row["model"],
            "dataset_mode": args.dataset_mode,
            "seed": args.seed,
            "dt_r2_train": metrics_train.get("dt_r2"),
            "integ_r2_median_train": metrics_train.get("integ_r2_median"),
            "ode_integ_r2_median_train": ode_train.get("ode_integ_r2_median"),
            "ode_state_min": state_transform.actual_bounds[0],
            "ode_state_max": state_transform.actual_bounds[1],
            "ode_state_transform": state_transform.name,
            "ode_deriv_min": deriv_clip[0],
            "ode_deriv_max": deriv_clip[1],
        }
        metrics_row.update(metrics_test)
        metrics_row.update(
            {
                "ode_integ_r2_median": ode_test.get("ode_integ_r2_median"),
            }
        )
        metrics.append(metrics_row)

        # Merge trajectories with phase labels
        for payload in traj_train + traj_test + traj_ode_train + traj_ode_test:
            payload.update(
                {
                    "marker": marker,
                    "model": row["model"],
                    "dataset_mode": args.dataset_mode,
                    "seed": args.seed,
                }
            )
            traj_records.append(payload)

    out_df = pd.DataFrame(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # maintain combined + aggregated metrics across seeds if requested
    if args.aggregate_output is None:
        agg_all_path = args.output.with_name(args.output.stem + "_all_seeds.csv")
        agg_mean_path = args.output.with_name(args.output.stem + "_agg_mean.csv")
    else:
        agg_all_path = args.aggregate_output
        agg_mean_path = args.aggregate_output.with_name(args.aggregate_output.stem + "_mean.csv")
    try:
        if agg_all_path.exists():
            existing = pd.read_csv(agg_all_path)
            combined = pd.concat([existing, out_df], ignore_index=True)
        else:
            combined = out_df.copy()
        combined.to_csv(agg_all_path, index=False)

        keys = [c for c in ["marker", "model", "dataset_mode"] if c in combined.columns]
        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "seed"]
        agg_mean = combined.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()
        if "formula" in combined.columns and "formula" not in agg_mean.columns:
            first_formulas = combined.groupby(keys, dropna=False)["formula"].first().reset_index()
            agg_mean = agg_mean.merge(first_formulas, on=keys, how="left")
        agg_mean.to_csv(agg_mean_path, index=False)
        print(f"Wrote combined seed metrics to {agg_all_path}")
        print(f"Wrote mean-over-seeds metrics to {agg_mean_path}")
    except Exception as exc:
        print(f"[compute_marker_integration] Failed to aggregate metrics across seeds: {exc}")

    traj_df = pd.DataFrame(traj_records)
    args.trajectories_output.parent.mkdir(parents=True, exist_ok=True)
    traj_df.to_csv(args.trajectories_output, index=False)
    print(f"Wrote integration metrics to {args.output}")
    print(f"Wrote integrated trajectories to {args.trajectories_output}")


if __name__ == "__main__":
    main()
