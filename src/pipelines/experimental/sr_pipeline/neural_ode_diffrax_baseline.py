"""
True Neural ODE baseline for per-marker p-ERK1-2 trajectories.

Counterpart to `neural_ode_baseline_per_minute.py`. That script trains an
MLP on dt MSE and integrates with forward Euler at inference — dt R² is
fine (0.7-0.9 test) but `integ_r2_median` collapses to ~0 on most markers
because fitting velocity does not imply fitting trajectories.

This script trains a real Neural ODE: MLP RHS, integrated by Dopri5
through diffrax, with backprop through the solver and a trajectory MSE
loss directly on p-ERK1-2(t). Exogenous features are linearly interpolated
in time per GFP-bin so the adaptive solver can query them between observed
rows.

Run
---
Uses `pysr_env` (already has jax + diffrax + equinox).

    python src/pipelines/experimental/sr_pipeline/neural_ode_diffrax_baseline.py \
        --output-dir data/experimental/runs/neural_ode_diffrax/_seeds/seed_42 \
        --seeds 42

The Snakemake rule `experimental_neural_ode_diffrax_baseline` drives this
per seed, then `neural_ode_diffrax_promote.py` reshapes the per-seed CSVs
into the canonical NN-baseline schema under
`{runs_root}/neural_ode_diffrax/full/`.

Outputs (per invocation)
------------------------
- <output>/neural_ode_diffrax_metrics.csv     (per marker x seed x split)
- <output>/neural_ode_diffrax_metrics_agg.csv (mean across seeds)
- <output>/training_history.csv                (loss history per marker)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU is plenty on tiny MLPs.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import diffrax

_FDTYPE = jnp.float64

SRC_ROOT = Path(__file__).resolve().parents[3]  # src/
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.experimental.sr_pipeline.compute_marker_integration import (
    attach_raw_perk,
    prepare_sr_dataset,
)
from pipelines.experimental.sr_pipeline.run_markers import (
    EXCLUDE_COLUMNS,
    apply_per_minute_sampling,
    sanitize_feature_names,
)
from pipelines.experimental.sr_pipeline.metrics import (
    binwise_r2,
    coefficient_of_determination,
)


MEASURED_TIMEPOINTS = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
PERK_NAME = sanitize_feature_names(["p-ERK1-2"])[0]  # "p_ERK1_2"
RAW_PERK_COL = "p_ERK1_2_raw"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True Neural ODE per-marker (diffrax).")
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/experimental/processed/functional_groups/markers_per_minute_fit.csv"
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=(42, 43, 44))
    p.add_argument("--markers", nargs="*", type=str, default=None,
                   help="Optional subset of markers (default: all).")
    p.add_argument("--marker-limit", type=int, default=None,
                   help="If set, cap to first N markers (sorted) after filtering.")
    # Sampling (mirror pipeline defaults)
    p.add_argument("--per-minute-max-time", type=float, default=60.0)
    p.add_argument("--per-minute-sampling-strategy",
                   choices=("max_time", "early_plus_sparse_late"),
                   default="early_plus_sparse_late")
    p.add_argument("--late-sample-window", nargs=2, type=float, default=(30.0, 60.0))
    p.add_argument("--late-sample-points", type=int, default=15)
    p.add_argument("--measured-timepoints", nargs="*", type=float,
                   default=MEASURED_TIMEPOINTS)
    # Split
    p.add_argument("--eval-only", default=None, metavar="MODELS_DIR",
                   help="Skip training: reload the checkpoints written by a previous run with "
                        "--save-models and re-score them. The split is deterministic, so the "
                        "rebuilt bundle is the one the model was trained on; use this to "
                        "re-evaluate under a changed metric without retraining.")
    p.add_argument("--raw-dataset", default=None,
                   help="Raw binned measurements; when given the rollout is scored against "
                        "these rather than the fitted curve.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    # Model / training
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--activation", choices=("tanh", "relu", "softplus"), default="tanh")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--solver", choices=("dopri5", "tsit5", "heun"), default="dopri5")
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--dt0", type=float, default=0.5,
                   help="Initial step size for the adaptive solver.")
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument(
        "--test-split-policy",
        choices=("random_bins", "top_gfp_bins"),
        default="random_bins",
        help="GFP-bin split for train/val/test. random_bins interleaves "
             "(model can interpolate over dose). top_gfp_bins assigns the "
             "highest-GFP bins to test/val, lower bins to train -> true "
             "extrapolation OOD.",
    )
    p.add_argument("--save-models", action="store_true",
                   help="Save per-marker model + metadata + train inputs to "
                        "<output-dir>/models/ for downstream causal analysis.")
    p.add_argument("--jac-reg", type=float, default=0.0,
                   help="Penalty on the network's Jacobian w.r.t. exogenous "
                        "features, mean over training inputs. 0 disables (default). "
                        "Encourages a sparse set of causal drivers without "
                        "penalising the state self-feedback.")
    p.add_argument("--jac-reg-mode", choices=("l1", "l21"), default="l1",
                   help="Form of the Jacobian penalty: 'l1' element-wise "
                        "|grad|, or 'l21' group-sparse (per-feature L2 over "
                        "inputs, L1 over features) for whole-feature pruning.")
    p.add_argument("--hess-reg", type=float, default=0.0,
                   help="Penalty on off-diagonal entries of the Hessian "
                        "(pairwise gating terms). 0 disables (default).")
    p.add_argument("--hess-reg-mode", choices=("l1", "l21"), default="l21",
                   help="Form of the Hessian penalty: 'l1' or 'l21'. "
                        "Default l21 = group-sparse over feature pairs, "
                        "drives entire (i,j) interactions to zero.")
    p.add_argument("--path-reg", type=float, default=0.0,
                   help="Aliee et al. 2022 PATHREG penalty on the product "
                        "of absolute weight magnitudes along input-to-output "
                        "paths. Architecture-aware feature pruning: a "
                        "feature i is silenced only when every path from "
                        "input-column i of layer 1 to the output is near 0. "
                        "Cheaper and stronger than Jacobian L21 for feature "
                        "selection. 0 disables (default).")
    # Comparison
    p.add_argument("--pipeline-metrics-agg", type=Path, default=None,
                   help="Optional pipeline neural_ode_metrics_agg.csv to merge "
                        "into a side-by-side comparison CSV.")
    p.add_argument("--pysr-metrics", type=Path, default=None,
                   help="Optional marker_integration_metrics_per_minute.csv to "
                        "merge PySR integrated R² into the comparison.")
    p.add_argument("--tag", type=str, default="diffrax")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Data prep -- per-bin trajectories
# -----------------------------------------------------------------------------

def _split_bins_three(
    bins: Sequence[float], test_size: float, val_size: float,
    rng: np.random.Generator, policy: str = "random_bins",
) -> Tuple[Optional[set], Optional[set], Optional[set]]:
    """
    Three-way GFP-bin split.

    policy:
      - "random_bins"   : shuffle, take first n_test as test, next n_val as val.
                          Interleaved OOD: the model can interpolate over dose.
      - "top_gfp_bins"  : ascending sort by GFP. The HIGHEST n_test bins go to
                          test, the next-highest n_val to val, the rest to
                          train. True extrapolation OOD: the model is trained
                          on low-to-mid doses and evaluated on the highest doses
                          it has never seen.
    """
    uniq = sorted({float(b) for b in bins if pd.notna(b)})
    if len(uniq) < 3:
        return None, None, None
    n = len(uniq)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    if n_test + n_val >= n:
        n_test = max(1, min(n - 2, n_test))
        n_val = max(1, min(n - 1 - n_test, n_val))
    if n_test + n_val >= n:
        return None, None, None

    if policy == "random_bins":
        shuffled = uniq.copy()
        rng.shuffle(shuffled)
        test_bins = set(shuffled[:n_test])
        val_bins = set(shuffled[n_test:n_test + n_val])
        train_bins = set(shuffled[n_test + n_val:])
    elif policy == "top_gfp_bins":
        test_bins = set(uniq[-n_test:])
        val_bins = set(uniq[-(n_test + n_val):-n_test])
        train_bins = set(uniq[:-(n_test + n_val)])
    else:
        raise ValueError(f"Unknown test split policy: {policy}")
    return train_bins, val_bins, test_bins


def _build_bin_trajectory(
    g: pd.DataFrame, target_col: str, exo_cols: Sequence[str],
) -> Optional[Dict[str, np.ndarray]]:
    """Return arrays for a single (marker, GFP_bin). None if too sparse."""
    g = g.sort_values("timepoint").copy()
    t = pd.to_numeric(g["timepoint"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g[PERK_NAME], errors="coerce").to_numpy(dtype=float)
    # The fitted curve drives the rollout; the raw measurements score it. Absent the raw
    # column the two coincide, so behaviour without --raw-dataset is unchanged.
    y_eval = (pd.to_numeric(g[RAW_PERK_COL], errors="coerce").to_numpy(dtype=float)
              if RAW_PERK_COL in g.columns else y.copy())
    dt_obs = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
    if len(exo_cols) > 0:
        U = g[list(exo_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        U = np.zeros((len(t), 0), dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & np.all(np.isfinite(U), axis=1) if U.size else np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    t, y, U = t[mask], y[mask], U[mask]
    dt_obs = dt_obs[mask]
    order = np.argsort(t)
    t, y, U, dt_obs, y_eval = t[order], y[order], U[order], dt_obs[order], y_eval[order]

    # Need strictly increasing t for diffrax interpolation.
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, y, U, dt_obs, y_eval = t[keep], y[keep], U[keep], dt_obs[keep], y_eval[keep]
    if len(t) < 3:
        return None

    return {
        "t": t.astype(np.float64),
        "y": y.astype(np.float64),
        "y_eval": y_eval.astype(np.float64),
        "U": U.astype(np.float64),
        "dt_obs": dt_obs.astype(np.float64),
    }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class RHS(eqx.Module):
    layers: list
    activation: str = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int,
                 activation: str, key: jax.Array):
        keys = jax.random.split(key, hidden_layers + 1)
        layers: list = []
        dim = in_dim
        for i in range(hidden_layers):
            layers.append(eqx.nn.Linear(dim, hidden_dim, key=keys[i]))
            dim = hidden_dim
        layers.append(eqx.nn.Linear(dim, 1, key=keys[-1]))
        self.layers = layers
        self.activation = activation
        self.in_dim = in_dim

    def _act(self, x):
        if self.activation == "tanh":
            return jnp.tanh(x)
        if self.activation == "relu":
            return jax.nn.relu(x)
        return jax.nn.softplus(x)

    def __call__(self, inp: jnp.ndarray) -> jnp.ndarray:
        x = inp
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x).squeeze(-1)


def _solver_for(name: str):
    if name == "dopri5":
        return diffrax.Dopri5()
    if name == "tsit5":
        return diffrax.Tsit5()
    return diffrax.Heun()


def _make_integrate_fn(
    solver_name: str, rtol: float, atol: float, dt0: float, max_steps: int,
):
    """Return a function that integrates a SINGLE trajectory (used inside vmap)."""
    solver = _solver_for(solver_name)
    controller = diffrax.PIDController(rtol=rtol, atol=atol)

    def integrate_one(
        rhs: RHS,
        y0: jnp.ndarray,                # scalar
        t: jnp.ndarray,                  # (T,) strictly increasing
        y_mean: jnp.ndarray,             # scalar
        y_scale: jnp.ndarray,            # scalar
        u_table_scaled: jnp.ndarray,     # (T, F)  pre-scaled exogenous
    ):
        interp = diffrax.LinearInterpolation(ts=t, ys=u_table_scaled)

        def f(time, state, _args):
            u_t = interp.evaluate(time)
            y_scaled = (state - y_mean) / y_scale
            inp = jnp.concatenate([jnp.atleast_1d(y_scaled), u_t])
            return rhs(inp)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=dt0,
            y0=y0,
            saveat=diffrax.SaveAt(ts=t),
            stepsize_controller=controller,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            throw=False,
        )
        return sol.ys  # (T,)

    return integrate_one


# -----------------------------------------------------------------------------
# Hand-rolled Adam (no optax in this env)
# -----------------------------------------------------------------------------

def _adam_init(params):
    leaves = jax.tree_util.tree_leaves(eqx.filter(params, eqx.is_inexact_array))
    m = jax.tree_util.tree_map(jnp.zeros_like, leaves)
    v = jax.tree_util.tree_map(jnp.zeros_like, leaves)
    # `step` must be a traced array, not a Python int. As a Python int it is a
    # STATIC argument to eqx.filter_jit, so incrementing it invalidates the
    # compile cache on every call and the whole diffrax solve is re-traced each
    # gradient step (measured: 6 calls -> 6 traces). As an array it compiles
    # once. The arithmetic below is unchanged, bias correction included.
    return {"step": jnp.zeros((), dtype=jnp.int32), "m": m, "v": v}


def _adam_apply(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    state = dict(state)
    step = state["step"] + 1
    state["step"] = step

    p_arrays, p_treedef = jax.tree_util.tree_flatten(
        eqx.filter(params, eqx.is_inexact_array)
    )
    g_arrays, _ = jax.tree_util.tree_flatten(
        eqx.filter(grads, eqx.is_inexact_array)
    )

    new_m, new_v, new_p = [], [], []
    for p, g, m_i, v_i in zip(p_arrays, g_arrays, state["m"], state["v"]):
        if g is None:
            new_m.append(m_i)
            new_v.append(v_i)
            new_p.append(p)
            continue
        m_new = b1 * m_i + (1 - b1) * g
        v_new = b2 * v_i + (1 - b2) * (g * g)
        s_f = step.astype(m_new.dtype) if hasattr(step, "astype") else step
        m_hat = m_new / (1 - b1 ** s_f)
        v_hat = v_new / (1 - b2 ** s_f)
        update = lr * (m_hat / (jnp.sqrt(v_hat) + eps) + wd * p)
        new_p.append(p - update)
        new_m.append(m_new)
        new_v.append(v_new)
    state["m"] = new_m
    state["v"] = new_v
    new_params_inexact = jax.tree_util.tree_unflatten(p_treedef, new_p)
    # Merge updated inexact arrays back into params (keeps static fields).
    new_params = eqx.combine(
        new_params_inexact,
        eqx.filter(params, eqx.is_inexact_array, inverse=True),
    )
    return new_params, state


# -----------------------------------------------------------------------------
# Per-marker training
# -----------------------------------------------------------------------------

def _per_minute_split(
    raw: pd.DataFrame, marker: str, seed: int, args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    sampled = apply_per_minute_sampling(
        raw,
        strategy=args.per_minute_sampling_strategy,
        max_time=args.per_minute_max_time,
        late_window=tuple(args.late_sample_window),
        late_points=args.late_sample_points,
        random_state=seed,
    )
    data, target_col = prepare_sr_dataset(sampled)
    if getattr(args, "raw_dataset", None):
        data = attach_raw_perk(data, args.raw_dataset)
    data = data[data["marker"] == marker].copy()
    if data.empty:
        return None
    data = data.dropna(subset=[target_col, "GFP_bin", "timepoint", PERK_NAME])
    if data.empty:
        return None

    feature_cols = [c for c in data.columns
                    if c not in EXCLUDE_COLUMNS and c != target_col and c != RAW_PERK_COL]
    if PERK_NAME not in feature_cols:
        return None
    exo_cols = [c for c in feature_cols if c != PERK_NAME]

    rng = np.random.default_rng(seed)
    splits = _split_bins_three(
        data["GFP_bin"].dropna().unique(), args.test_size, args.val_size, rng,
        policy=args.test_split_policy,
    )
    if splits[0] is None:
        return None
    train_bins, val_bins, test_bins = splits

    def _make_trajs(bins: set) -> List[Dict[str, np.ndarray]]:
        out: List[Dict[str, np.ndarray]] = []
        for b in sorted(bins):
            g = data[data["GFP_bin"] == b]
            traj = _build_bin_trajectory(g, target_col, exo_cols)
            if traj is None:
                continue
            traj["bin"] = float(b)
            out.append(traj)
        return out

    train_trajs = _make_trajs(train_bins)
    val_trajs = _make_trajs(val_bins)
    test_trajs = _make_trajs(test_bins)
    if not train_trajs or not val_trajs or not test_trajs:
        return None

    # Standardise inputs using TRAIN statistics only.
    if exo_cols:
        U_train = np.concatenate([t["U"] for t in train_trajs], axis=0)
        u_mean = U_train.mean(axis=0)
        u_std = U_train.std(axis=0)
        u_std = np.where(u_std < 1e-6, 1.0, u_std)
    else:
        u_mean = np.zeros((0,), dtype=np.float64)
        u_std = np.ones((0,), dtype=np.float64)

    y_train = np.concatenate([t["y"] for t in train_trajs], axis=0)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-6:
        y_std = 1.0

    def _attach_scaled(trajs):
        for tr in trajs:
            if tr["U"].size:
                tr["U_scaled"] = ((tr["U"] - u_mean) / u_std).astype(np.float64)
            else:
                tr["U_scaled"] = np.zeros((len(tr["t"]), 0), dtype=np.float64)
        return trajs

    _attach_scaled(train_trajs)
    _attach_scaled(val_trajs)
    _attach_scaled(test_trajs)

    # Pad each split to a common T_max (PER SPLIT — smaller compile cost) so
    # diffrax JIT-compiles once and we can vmap the integrator over bins.
    measured_set = set(float(t) for t in args.measured_timepoints)

    def _stack_split(trajs):
        if not trajs:
            return None
        n_real = np.array([len(tr["t"]) for tr in trajs], dtype=np.int32)
        T_max = int(n_real.max())
        n_F = trajs[0]["U_scaled"].shape[1]
        B = len(trajs)
        ts = np.zeros((B, T_max), dtype=np.float64)
        ys = np.zeros((B, T_max), dtype=np.float64)
        us = np.zeros((B, T_max, n_F), dtype=np.float64)
        dts = np.full((B, T_max), np.nan, dtype=np.float64)
        loss_mask = np.zeros((B, T_max), dtype=bool)
        real_mask = np.zeros((B, T_max), dtype=bool)
        for i, tr in enumerate(trajs):
            n = len(tr["t"])
            ts[i, :n] = tr["t"]
            ys[i, :n] = tr["y"]
            dts[i, :n] = tr["dt_obs"]
            if n_F:
                us[i, :n] = tr["U_scaled"]
            real_mask[i, :n] = True
            # Pad t with tiny increments past the real end so LinearInterp stays
            # monotone and the adaptive solver burns negligible steps on the
            # tail (padded segment has constant u -> RHS ~constant).
            if n < T_max:
                last_t = float(tr["t"][-1])
                pad_t = last_t + (np.arange(T_max - n, dtype=np.float64) + 1.0) * 1e-3
                ts[i, n:] = pad_t
                ys[i, n:] = tr["y"][-1]
                if n_F:
                    us[i, n:] = tr["U_scaled"][-1]
            # Loss only on real + measured timepoints.
            in_meas = np.isin(ts[i], np.array(sorted(measured_set)))
            loss_mask[i] = real_mask[i] & in_meas
        # Integrate ONLY to the last real timepoint per bin.
        t_end = np.array([float(tr["t"][-1]) for tr in trajs], dtype=np.float64)
        y0 = np.array([float(tr["y"][0]) for tr in trajs], dtype=np.float64)
        # Save points are the padded ts; we'll mask the saved values for those
        # that lie past the real end (solver returns NaN at unsolved points).
        return {
            "ts": ts, "ys": ys, "us": us, "dts": dts,
            "loss_mask": loss_mask, "real_mask": real_mask,
            "t_end": t_end, "y0": y0,
            "n_F": n_F, "B": B, "T_max": T_max,
            "trajs": trajs,
        }

    train_pack = _stack_split(train_trajs)
    val_pack = _stack_split(val_trajs)
    test_pack = _stack_split(test_trajs)
    if train_pack is None or val_pack is None or test_pack is None:
        return None

    return {
        "target_col": target_col,
        "feature_cols": feature_cols,
        "exo_cols": exo_cols,
        "perk_index": feature_cols.index(PERK_NAME),
        "train": train_pack,
        "val": val_pack,
        "test": test_pack,
        "u_mean": u_mean.astype(np.float64),
        "u_scale": u_std.astype(np.float64),
        "y_mean": np.float64(y_mean),
        "y_scale": np.float64(y_std),
    }


def _batched_trajectory_loss(
    rhs: RHS, pack: Dict[str, jnp.ndarray], integrate_one,
    y_mean: jnp.ndarray, y_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Mean per-bin MSE on measured + real timepoints (vmapped over bins)."""

    def one(y0, ts, ys, us, mask):
        y_pred = integrate_one(rhs, y0, ts, y_mean, y_scale, us)
        sq = (y_pred - ys) ** 2
        sq = jnp.where(mask, sq, 0.0)
        n = jnp.maximum(jnp.sum(mask), 1)
        return jnp.sum(sq) / n

    losses = jax.vmap(one)(
        pack["y0"], pack["ts"], pack["ys"], pack["us"], pack["loss_mask"],
    )
    return jnp.mean(losses)


def _pack_to_jax(pack: Dict[str, object]) -> Dict[str, jnp.ndarray]:
    return {
        "ts": jnp.asarray(pack["ts"]),
        "ys": jnp.asarray(pack["ys"]),
        "us": jnp.asarray(pack["us"]),
        "loss_mask": jnp.asarray(pack["loss_mask"]),
        "real_mask": jnp.asarray(pack["real_mask"]),
        "y0": jnp.asarray(pack["y0"]),
    }


def _jac_penalty(
    rhs: RHS, pack: Dict[str, jnp.ndarray],
    y_mean: jnp.ndarray, y_scale: jnp.ndarray,
    mode: str = "l1",
) -> jnp.ndarray:
    """
    Penalty on ∂f/∂x_exo over real training inputs.

    Restricted to the exogenous features (input index ≥ 1) so the state
    self-feedback ∂f/∂y at index 0 is left untouched — driving that to zero
    would damage the dynamics' stability without telling us anything about
    causal structure.

    mode:
      "l1"  — mean |grad_{n,j}| over (input, feature) pairs. Element-wise
              sparsifying — shrinks individual gradient values but doesn't
              strongly couple all inputs of one feature together.
      "l21" — sum_j sqrt(mean_n grad_{n,j}^2). Group-sparse over features:
              drives entire features to zero across all inputs. This is the
              standard "group lasso" for feature selection.
    """
    ys_scaled = (pack["ys"] - y_mean) / y_scale            # (B, T)
    inputs = jnp.concatenate(
        [ys_scaled[..., None], pack["us"]], axis=-1
    )                                                      # (B, T, F+1)
    F1 = inputs.shape[-1]
    flat = inputs.reshape(-1, F1)                          # (N, F+1)
    grads = jax.vmap(jax.grad(rhs))(flat)                  # (N, F+1)
    g_exo = grads[:, 1:]                                   # (N, F)
    m = pack["real_mask"].reshape(-1).astype(g_exo.dtype)  # (N,)
    n_real = jnp.maximum(jnp.sum(m), 1.0)
    F = g_exo.shape[-1]
    if mode == "l1":
        weighted = jnp.abs(g_exo) * m[:, None]
        return jnp.sum(weighted) / (n_real * F)
    # l21: per-feature RMS over inputs, averaged across features
    sq = (g_exo ** 2) * m[:, None]
    per_feat_rms = jnp.sqrt(jnp.sum(sq, axis=0) / n_real + 1e-12)  # (F,)
    return jnp.mean(per_feat_rms)


def _path_penalty(rhs: RHS) -> jnp.ndarray:
    """
    PATHREG (Aliee et al. 2022): L1 norm of the product of absolute layer
    weights from input to output.

    For an L-layer MLP with weight matrices W_1, ..., W_L, the path strength
    from input feature i to the (scalar) output is the sum of |W_L · ... · W_1|
    over all (h_1, ..., h_{L-1}) hidden-unit chains. In matrix form this
    equals (|W_L| · |W_{L-1}| · ... · |W_1|)[0, i].

    Sums over the EXOGENOUS feature columns only (skipping input index 0,
    the state self-loop). Mean over features for scale stability vs the
    other regularisers.
    """
    M = None
    for lin in rhs.layers:
        W = jnp.abs(lin.weight)
        M = W if M is None else W @ M  # accumulate from input toward output
    # M now has shape (out_dim=1, in_dim=F+1)
    F1 = M.shape[-1]
    return jnp.sum(M[0, 1:]) / max(F1 - 1, 1)


def _hess_penalty(
    rhs: RHS, pack: Dict[str, jnp.ndarray],
    y_mean: jnp.ndarray, y_scale: jnp.ndarray,
    mode: str = "l21",
) -> jnp.ndarray:
    """
    Penalty on off-diagonal entries of the Hessian d²f/dx_i dx_j.

    Targets gating sparsity: drives pairwise interaction terms to zero so
    only the dominant gating structure survives. Diagonal entries (self-
    curvature) are excluded.

    mode:
      "l1"  — mean |H_{n,i,j}| over (input, off-diagonal pair).
      "l21" — sum_{i≠j} sqrt(mean_n H_{n,i,j}^2). Group-sparse over
              feature pairs: drives entire (i,j) interactions to zero.
    """
    ys_scaled = (pack["ys"] - y_mean) / y_scale
    inputs = jnp.concatenate(
        [ys_scaled[..., None], pack["us"]], axis=-1
    )
    F1 = inputs.shape[-1]
    flat = inputs.reshape(-1, F1)
    H = jax.vmap(jax.hessian(rhs))(flat)                   # (N, F1, F1)
    m = pack["real_mask"].reshape(-1).astype(H.dtype)
    n_real = jnp.maximum(jnp.sum(m), 1.0)
    off_mask = ~jnp.eye(F1, dtype=bool)                    # (F1, F1)
    n_off = float(F1 * (F1 - 1))
    if mode == "l1":
        weighted = jnp.abs(H) * m[:, None, None]
        weighted_off = jnp.where(off_mask[None], weighted, 0.0)
        return jnp.sum(weighted_off) / (n_real * n_off)
    # l21
    sq = (H ** 2) * m[:, None, None]
    per_pair_rms = jnp.sqrt(jnp.sum(sq, axis=0) / n_real + 1e-12)  # (F1, F1)
    per_pair_off = jnp.where(off_mask, per_pair_rms, 0.0)
    return jnp.sum(per_pair_off) / n_off


def _train_marker(
    marker: str, seed: int, args: argparse.Namespace,
    bundle: Dict[str, object],
) -> Tuple[RHS, Dict[str, float], List[Dict[str, float]]]:
    feature_cols = bundle["feature_cols"]
    in_dim = len(feature_cols)
    key = jax.random.PRNGKey(seed)
    rhs = RHS(in_dim, args.hidden_dim, args.hidden_layers, args.activation, key)

    integrate_one = _make_integrate_fn(
        args.solver, args.rtol, args.atol, args.dt0, args.max_steps,
    )

    y_mean = jnp.asarray(bundle["y_mean"])
    y_scale = jnp.asarray(bundle["y_scale"])

    train_jax = _pack_to_jax(bundle["train"])
    val_jax = _pack_to_jax(bundle["val"])

    jac_reg = float(args.jac_reg)
    jac_mode = args.jac_reg_mode
    hess_reg = float(args.hess_reg)
    hess_mode = args.hess_reg_mode
    path_reg = float(args.path_reg)

    @eqx.filter_jit
    def loss_fn(rhs_local, pack):
        # Validation/early-stop uses trajectory MSE only — keeps val loss
        # comparable across regularisation strengths.
        return _batched_trajectory_loss(
            rhs_local, pack, integrate_one, y_mean, y_scale,
        )

    @eqx.filter_jit
    def step_fn(rhs_local, opt_state, pack):
        def total(r):
            loss = _batched_trajectory_loss(r, pack, integrate_one, y_mean, y_scale)
            if jac_reg > 0:
                loss = loss + jac_reg * _jac_penalty(r, pack, y_mean, y_scale, jac_mode)
            if hess_reg > 0:
                loss = loss + hess_reg * _hess_penalty(r, pack, y_mean, y_scale, hess_mode)
            if path_reg > 0:
                loss = loss + path_reg * _path_penalty(r)
            return loss
        loss, grads = eqx.filter_value_and_grad(total)(rhs_local)
        rhs_new, opt_state = _adam_apply(
            rhs_local, grads, opt_state, lr=args.lr, wd=args.weight_decay,
        )
        return rhs_new, opt_state, loss

    opt_state = _adam_init(rhs)
    best_val = float("inf")
    best_rhs = rhs
    history: List[Dict[str, float]] = []
    bad = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        rhs, opt_state, loss = step_fn(rhs, opt_state, train_jax)
        tr_loss = float(loss)
        val_loss = float(loss_fn(rhs, val_jax))
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})

        if not math.isfinite(val_loss):
            break
        if val_loss + args.min_delta < best_val:
            best_val = val_loss
            best_rhs = jax.tree_util.tree_map(
                lambda x: x.copy() if hasattr(x, "copy") else x, rhs,
            )
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    elapsed = time.time() - t0
    info = {
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "wall_seconds": elapsed,
    }
    return best_rhs, info, history


# -----------------------------------------------------------------------------
# Eval (rebuilds frames with predicted dt, runs same metrics as pipeline)
# -----------------------------------------------------------------------------

def _load_trained_rhs(
    models_root: Path, marker: str, seed: int, bundle: Dict[str, object],
) -> Optional[RHS]:
    """Reload a checkpoint written with --save-models, shaped by its own metadata."""
    base = models_root / f"seed_{seed}" / "models"
    if not base.exists():
        base = models_root / "models"
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(marker)).strip("_")
    weights, meta_path = base / f"{stem}.eqx", base / f"{stem}.json"
    if not (weights.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if int(meta["in_dim"]) != len(bundle["feature_cols"]):
        raise SystemExit(
            f"{marker} seed {seed}: checkpoint expects {meta['in_dim']} inputs but the rebuilt "
            f"split has {len(bundle['feature_cols'])}; the two do not describe the same run")
    skeleton = RHS(in_dim=int(meta["in_dim"]), hidden_dim=int(meta["hidden_dim"]),
                   hidden_layers=int(meta["hidden_layers"]),
                   activation=str(meta["activation"]), key=jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(str(weights), skeleton)


def _predict_dt_at_rows(
    rhs: RHS, bundle: Dict[str, object], trajs: List[Dict[str, np.ndarray]],
) -> List[np.ndarray]:
    """Predict instantaneous dt at the observed (y, u) rows of each trajectory."""
    y_mean = float(bundle["y_mean"])
    y_scale = float(bundle["y_scale"])
    out: List[np.ndarray] = []
    for tr in trajs:
        y = (tr["y"] - y_mean) / y_scale
        u = tr["U_scaled"]
        if u.size:
            inp = np.concatenate([y[:, None], u], axis=1)
        else:
            inp = y[:, None]
        inp_j = jnp.asarray(inp)
        pred = jax.vmap(rhs)(inp_j)
        out.append(np.asarray(pred))
    return out


def _rollout(
    rhs: RHS, bundle: Dict[str, object], pack: Dict[str, object],
    args: argparse.Namespace,
) -> List[np.ndarray]:
    integrate_one = _make_integrate_fn(
        args.solver, args.rtol, args.atol, args.dt0, args.max_steps,
    )
    y_mean = jnp.asarray(bundle["y_mean"])
    y_scale = jnp.asarray(bundle["y_scale"])
    ts = jnp.asarray(pack["ts"])
    ys = jnp.asarray(pack["ys"])
    us = jnp.asarray(pack["us"])
    y0 = jnp.asarray(pack["y0"])

    def one(y0_i, ts_i, us_i):
        return integrate_one(rhs, y0_i, ts_i, y_mean, y_scale, us_i)

    rollout_batched = jax.jit(jax.vmap(one))(y0, ts, us)
    rollout_np = np.asarray(rollout_batched)
    # Slice each bin back down to its real length.
    return [rollout_np[i, : len(tr["t"])] for i, tr in enumerate(pack["trajs"])]


def _split_metrics(
    rhs: RHS, bundle: Dict[str, object], pack: Optional[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, float]:
    if pack is None or not pack["trajs"]:
        return {
            "dt_r2": np.nan, "integ_r2_median": np.nan,
            "ode_integ_r2_median": np.nan, "trajectory_r2_mean": np.nan,
            "trajectory_rmse_measured": np.nan, "n_bins": 0,
        }
    trajs = pack["trajs"]
    measured = np.asarray(args.measured_timepoints, dtype=float)
    dt_preds = _predict_dt_at_rows(rhs, bundle, trajs)
    rollouts = _rollout(rhs, bundle, pack, args)

    # dt R² per bin (on measured timepoints only — match binwise_r2 with
    # dataset_mode="per_minute").
    dt_r2_vals: List[float] = []
    integ_r2_vals: List[float] = []
    ode_r2_vals: List[float] = []
    traj_r2_vals: List[float] = []
    rmse_vals: List[float] = []
    for tr, dt_p, roll in zip(trajs, dt_preds, rollouts):
        t = tr["t"]
        y = tr["y"]
        y_eval = tr.get("y_eval", y)
        dt_obs = tr["dt_obs"]
        meas_mask = np.isin(t, measured)
        if meas_mask.sum() >= 2:
            r2 = coefficient_of_determination(dt_obs[meas_mask], dt_p[meas_mask])
            if np.isfinite(r2):
                dt_r2_vals.append(max(0.0, r2))
            # Trajectory R² (Neural ODE rollout vs observed p-ERK)
            r2_t = coefficient_of_determination(y_eval[meas_mask], roll[meas_mask])
            if np.isfinite(r2_t):
                traj_r2_vals.append(max(0.0, r2_t))
                ode_r2_vals.append(max(0.0, r2_t))
            rmse = float(np.sqrt(np.mean((y_eval[meas_mask] - roll[meas_mask]) ** 2)))
            if np.isfinite(rmse):
                rmse_vals.append(rmse)

        # Feature-driven Euler integration (matches pipeline integ_r2_median)
        if len(t) >= 2:
            integ = np.full_like(t, np.nan, dtype=float)
            integ[0] = y[0]
            state = float(y[0])
            for j in range(len(t) - 1):
                dt_step = t[j + 1] - t[j]
                state = state + dt_step * float(dt_p[j])
                integ[j + 1] = state
            if meas_mask.sum() >= 2:
                r2_i = coefficient_of_determination(y_eval[meas_mask], integ[meas_mask])
                if np.isfinite(r2_i):
                    integ_r2_vals.append(max(0.0, r2_i))

    def _med(xs):
        return float(np.median(xs)) if xs else float("nan")

    return {
        "dt_r2": float(np.mean(dt_r2_vals)) if dt_r2_vals else float("nan"),
        "integ_r2_median": _med(integ_r2_vals),
        "ode_integ_r2_median": _med(ode_r2_vals),
        "trajectory_r2_mean": float(np.mean(traj_r2_vals)) if traj_r2_vals else float("nan"),
        "trajectory_rmse_measured": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
        "n_bins": len(pack["trajs"]),
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[NeuralODE-diffrax] JAX backend: {jax.default_backend()}", flush=True)
    print(f"[NeuralODE-diffrax] Dataset: {args.dataset}", flush=True)
    raw = pd.read_csv(args.dataset)

    # Discover marker list from the prepared dataset.
    base_data, _ = prepare_sr_dataset(raw)
    markers = sorted(base_data["marker"].dropna().unique().tolist())
    if args.markers:
        wanted = set(args.markers)
        markers = [m for m in markers if m in wanted]
    if args.marker_limit:
        markers = markers[: args.marker_limit]
    print(f"[NeuralODE-diffrax] Markers ({len(markers)}): {markers[:8]}{'...' if len(markers) > 8 else ''}",
          flush=True)

    # Per-marker incremental output: append each marker's rows to this CSV so
    # results survive crashes / OOMs even mid-sweep.
    incr_path = outdir / "neural_ode_diffrax_metrics.csv"
    hist_path = outdir / "training_history.csv"
    incr_cols = [
        "marker", "k", "seed", "tag", "split",
        "dt_r2", "integ_r2_median", "ode_integ_r2_median",
        "trajectory_r2_mean", "trajectory_rmse_measured", "n_bins",
        "best_val_loss", "epochs_run", "wall_seconds", "jac_reg",
        "jac_reg_mode", "hess_reg", "hess_reg_mode", "path_reg",
        "test_split_policy",
    ]
    if not incr_path.exists():
        pd.DataFrame(columns=incr_cols).to_csv(incr_path, index=False)
    # Skip any (seed, marker) pairs already in the file (resume support).
    done_pairs: set = set()
    if incr_path.exists() and incr_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(incr_path)
            if not existing.empty and "seed" in existing.columns and "marker" in existing.columns:
                done_pairs = {(int(s), str(m)) for s, m in zip(existing["seed"], existing["marker"])}
        except Exception:
            done_pairs = set()
    if done_pairs:
        print(f"[NeuralODE-diffrax] resuming: {len(done_pairs)} (seed, marker) pairs already on disk",
              flush=True)

    rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        for marker in markers:
            if (int(seed), str(marker)) in done_pairs:
                print(f"[skip-done] seed={seed} marker={marker}", flush=True)
                continue
            bundle = _per_minute_split(raw, marker, seed, args)
            if bundle is None:
                print(f"[skip] seed={seed} marker={marker}: insufficient data", flush=True)
                continue
            print(
                f"[train] seed={seed} marker={marker} "
                f"k={len(bundle['feature_cols'])} "
                f"bins(train/val/test)={bundle['train']['B']}/{bundle['val']['B']}/{bundle['test']['B']}",
                flush=True,
            )
            try:
                if args.eval_only:
                    rhs = _load_trained_rhs(Path(args.eval_only), marker, seed, bundle)
                    if rhs is None:
                        print(f"[skip] seed={seed} marker={marker}: no checkpoint", flush=True)
                        continue
                    info, history = {"best_val_loss": float("nan"), "epochs_run": 0,
                                     "wall_seconds": 0.0}, []
                else:
                    rhs, info, history = _train_marker(marker, seed, args, bundle)
            except Exception as exc:
                import traceback
                print(f"[err] seed={seed} marker={marker} training failed: {exc}", flush=True)
                traceback.print_exc()
                continue

            for h in history:
                history_rows.append({"seed": seed, "marker": marker, **h})

            marker_rows: List[Dict[str, object]] = []
            for split_name, pack in (
                ("train", bundle["train"]),
                ("val", bundle["val"]),
                ("test", bundle["test"]),
            ):
                m = _split_metrics(rhs, bundle, pack, args)
                marker_rows.append({
                    "marker": marker,
                    "k": len(bundle["feature_cols"]),
                    "seed": seed,
                    "tag": args.tag,
                    "split": split_name,
                    **m,
                    "best_val_loss": info["best_val_loss"],
                    "epochs_run": info["epochs_run"],
                    "jac_reg": float(args.jac_reg),
                    "jac_reg_mode": args.jac_reg_mode,
                    "hess_reg": float(args.hess_reg),
                    "hess_reg_mode": args.hess_reg_mode,
                    "path_reg": float(args.path_reg),
                    "test_split_policy": args.test_split_policy,
                    "wall_seconds": info["wall_seconds"],
                })
            rows.extend(marker_rows)
            # Append this marker's rows to disk immediately.
            pd.DataFrame(marker_rows).to_csv(
                incr_path, mode="a", header=False, index=False,
                columns=[c for c in incr_cols if c in marker_rows[0]] if marker_rows else None,
            )
            # Append the loss history for this marker too.
            pd.DataFrame([{"seed": seed, "marker": marker, **h} for h in history]).to_csv(
                hist_path, mode="a",
                header=not hist_path.exists() or hist_path.stat().st_size == 0,
                index=False,
            )
            print(
                f"[done] seed={seed} marker={marker} "
                f"train_traj_r2={marker_rows[0]['trajectory_r2_mean']:.3f} "
                f"val_traj_r2={marker_rows[1]['trajectory_r2_mean']:.3f} "
                f"test_traj_r2={marker_rows[2]['trajectory_r2_mean']:.3f} "
                f"wall={info['wall_seconds']:.1f}s epochs={info['epochs_run']}",
                flush=True,
            )
            # Optional: dump the trained RHS + metadata + scaled train inputs
            # so that downstream causal/interaction analysis can recompute
            # Jacobians and Hessians without retraining.
            if args.save_models:
                models_dir = outdir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                safe = (
                    str(marker).replace("/", "_").replace(" ", "_")
                    .replace("(", "").replace(")", "")
                )
                eqx.tree_serialise_leaves(str(models_dir / f"{safe}.eqx"), rhs)
                meta = {
                    "marker": marker,
                    "seed": int(seed),
                    "in_dim": len(bundle["feature_cols"]),
                    "hidden_dim": int(args.hidden_dim),
                    "hidden_layers": int(args.hidden_layers),
                    "activation": args.activation,
                    "feature_cols": list(bundle["feature_cols"]),
                    "exo_cols": list(bundle["exo_cols"]),
                    "perk_index": int(bundle["perk_index"]),
                    "u_mean": [float(x) for x in bundle["u_mean"]],
                    "u_scale": [float(x) for x in bundle["u_scale"]],
                    "y_mean": float(bundle["y_mean"]),
                    "y_scale": float(bundle["y_scale"]),
                }
                with open(models_dir / f"{safe}.json", "w") as f:
                    json.dump(meta, f, indent=2)
                tr = bundle["train"]
                np.savez_compressed(
                    models_dir / f"{safe}_train.npz",
                    ys=tr["ys"],
                    us_scaled=tr["us"],
                    real_mask=tr["real_mask"],
                )
            # Free per-marker JAX cache + Python refs before moving on. JAX
            # caches XLA-compiled artifacts keyed on input shapes; over a long
            # sweep with varying T_max / feature dims these grow to GBs and
            # send the box into swap. Clearing here is the difference between
            # finishing the sweep and OOMing.
            del rhs, history, marker_rows, bundle
            jax.clear_caches()
            import gc as _gc
            _gc.collect()

    # Reload from disk: the incremental file may contain rows from a previous
    # resumed run plus rows we just appended.
    if incr_path.exists() and incr_path.stat().st_size > 0:
        try:
            metrics_df = pd.read_csv(incr_path)
        except Exception:
            metrics_df = pd.DataFrame(rows)
    else:
        metrics_df = pd.DataFrame(rows)
    print(f"Final metrics on disk: {incr_path} ({len(metrics_df)} rows)", flush=True)

    if not metrics_df.empty:
        agg_cols = [
            "dt_r2", "integ_r2_median", "ode_integ_r2_median",
            "trajectory_r2_mean", "trajectory_rmse_measured",
        ]
        agg = (
            metrics_df.groupby(["marker", "k", "split"], dropna=False)[agg_cols]
            .mean().reset_index()
        )
        agg_path = outdir / "neural_ode_diffrax_metrics_agg.csv"
        agg.to_csv(agg_path, index=False)
        print(f"Wrote {agg_path}", flush=True)

        # Optional side-by-side comparison vs the pipeline neural ODE.
        compare_frames = []
        if args.pipeline_metrics_agg and args.pipeline_metrics_agg.exists():
            pipe = pd.read_csv(args.pipeline_metrics_agg)
            pipe = pipe.rename(columns={
                "dt_r2": "pipeline_dt_r2",
                "integ_r2_median": "pipeline_integ_r2_median",
                "ode_integ_r2_median": "pipeline_ode_integ_r2_median",
            })
            keep = ["marker", "split", "pipeline_dt_r2",
                    "pipeline_integ_r2_median", "pipeline_ode_integ_r2_median"]
            keep = [c for c in keep if c in pipe.columns]
            compare_frames.append(pipe[keep])
        merged = agg.rename(columns={
            "dt_r2": "diffrax_dt_r2",
            "integ_r2_median": "diffrax_integ_r2_median",
            "ode_integ_r2_median": "diffrax_ode_integ_r2_median",
            "trajectory_r2_mean": "diffrax_trajectory_r2_mean",
        })
        for cf in compare_frames:
            merged = merged.merge(cf, on=["marker", "split"], how="left")
        if args.pysr_metrics and args.pysr_metrics.exists():
            try:
                pysr = pd.read_csv(args.pysr_metrics)
                pysr = pysr[(pysr.get("model") == "PySR")
                            & (pysr.get("dataset_mode", "per_minute") == "per_minute")]
                if "integ_r2_median" in pysr.columns and "marker" in pysr.columns:
                    pysr_small = pysr[["marker", "split", "integ_r2_median"]].copy()
                    pysr_small = pysr_small.rename(columns={"integ_r2_median": "pysr_integ_r2_median"})
                    merged = merged.merge(pysr_small, on=["marker", "split"], how="left")
            except Exception as exc:
                print(f"[warn] failed to merge PySR metrics: {exc}", flush=True)
        comparison_path = outdir / "comparison_vs_pipeline.csv"
        merged.to_csv(comparison_path, index=False)
        print(f"Wrote {comparison_path}", flush=True)

        # Brief stdout summary across markers on the TEST split.
        test_df = agg[agg["split"] == "test"]
        if not test_df.empty:
            print("\n=== diffrax Neural ODE — TEST split summary ===")
            print(test_df[["trajectory_r2_mean", "dt_r2",
                           "integ_r2_median", "ode_integ_r2_median"]].describe())

    print("\nDone.")


if __name__ == "__main__":
    main()
