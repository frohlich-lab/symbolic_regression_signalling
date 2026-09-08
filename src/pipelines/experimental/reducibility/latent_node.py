"""Step 6 — test for hidden state directly, with a latent-augmented Neural ODE.

Why the current evidence cannot support the claim
-------------------------------------------------
The manuscript wants to conclude that SR-fail contexts "require more interacting
variables than a rate law can compactly express", and the Discussion reaches for
unobserved state to explain it. But every quantity used to argue this — the
participation ratio, the input Jacobian, the dependency counts of Table S8 — is
computed over the ten *observed* inputs. A statistic defined on observed
variables cannot, even in principle, provide evidence about unobserved ones. The
inference is currently from absence: SR failed, so something must be missing.

This script turns that into a positive measurement.

The test
--------
Augment the state from 1 (p-ERK) to 1 + d, where the d extra dimensions are
unobserved and are never compared to data:

    d/dt [y; z] = f_theta(y, z, u(t)),     y in R, z in R^d
    loss        = MSE on y at measured timepoints only

The latent initial condition is *not* a free per-bin parameter — that would fit
each held-out bin its own escape hatch and inflate held-out R^2 for free.
Instead an encoder maps each bin's own initial observation to its latent start,

    z(0) = enc(y(0), u(0)),

so a held-out GFP bin gets its latent state from data it is entitled to, and
extrapolation stays honest.

Reading the result
------------------
Run d = 0, 1, 2, 3. d = 0 reproduces the published architecture, which is the
control. Then:

  * held-out R^2 improves with d in SR-fail contexts but NOT in SR-success
    contexts  -> positive evidence that the SR-fail contexts carry dynamics not
    expressible in the measured variables. This is the result that would let the
    paper keep its strong claim instead of retreating to a weaker one.
  * held-out R^2 improves everywhere -> the gain is capacity, not hidden state;
    the claim must be dropped.
  * no improvement anywhere -> the six measured timepoints cannot resolve latent
    dynamics, which is itself a limitation worth stating explicitly rather than
    leaving the hidden-state hypothesis unexamined.

The interaction (improvement-with-d contrasted between SR-fail and SR-success
contexts) is the quantity to report, not the raw improvement — a main effect of
d is expected from added capacity alone.

Usage:
    python latent_node.py --output-dir <dir> [--latent-dims 0 1 2 3] [--markers ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    PAPER_CFG,
    _baseline,
    all_markers,
    default_outdir,
    load_bundle,
    load_raw,
    make_baseline_args,
    write_csv,
)

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import equinox as eqx  # noqa: E402
import diffrax  # noqa: E402

COLUMNS = ["marker", "seed", "latent_dim", "train_r2", "val_r2", "test_r2",
           "test_rmse", "best_val_loss", "epochs_run", "n_params", "wall_seconds"]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class LatentRHS(eqx.Module):
    """d/dt [y; z] = MLP([y; z; u]).  Observable is component 0."""
    layers: list
    activation: str = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    n_exo: int = eqx.field(static=True)

    def __init__(self, state_dim, n_exo, hidden_dim, hidden_layers, activation, key):
        keys = jax.random.split(key, hidden_layers + 1)
        dim = state_dim + n_exo
        layers = []
        for i in range(hidden_layers):
            layers.append(eqx.nn.Linear(dim, hidden_dim, key=keys[i]))
            dim = hidden_dim
        layers.append(eqx.nn.Linear(dim, state_dim, key=keys[-1]))
        self.layers = layers
        self.activation = activation
        self.state_dim = state_dim
        self.n_exo = n_exo

    def _act(self, x):
        if self.activation == "tanh":
            return jnp.tanh(x)
        if self.activation == "relu":
            return jax.nn.relu(x)
        return jax.nn.softplus(x)

    def __call__(self, inp):
        x = inp
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x)


class LatentEncoder(eqx.Module):
    """z(0) = enc([y(0); u(0)]) — per-bin latent start from that bin's own data."""
    layers: list
    latent_dim: int = eqx.field(static=True)

    def __init__(self, n_exo, latent_dim, hidden_dim, activation, key):
        self.latent_dim = latent_dim
        if latent_dim == 0:
            self.layers = []
            return
        k1, k2 = jax.random.split(key)
        self.layers = [eqx.nn.Linear(1 + n_exo, hidden_dim, key=k1),
                       eqx.nn.Linear(hidden_dim, latent_dim, key=k2)]

    def __call__(self, y0, u0):
        if self.latent_dim == 0:
            return jnp.zeros((0,))
        x = jnp.concatenate([jnp.atleast_1d(y0), u0])
        return self.layers[1](jnp.tanh(self.layers[0](x)))


class LatentModel(eqx.Module):
    rhs: LatentRHS
    enc: LatentEncoder


def _make_integrator(solver_name, rtol, atol, dt0, max_steps):
    solver = {"dopri5": diffrax.Dopri5, "tsit5": diffrax.Tsit5,
              "heun": diffrax.Heun}[solver_name]()

    def integrate_one(model, y0, ts, us, t_end):
        """Integrate one bin; `us` is linearly interpolated in time."""
        u_interp = diffrax.LinearInterpolation(ts=ts, ys=us)
        z0 = model.enc(y0, us[0])
        s0 = jnp.concatenate([jnp.atleast_1d(y0), z0])

        def field(t, s, _a):
            return model.rhs(jnp.concatenate([s, u_interp.evaluate(t)]))

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(field), solver,
            t0=ts[0], t1=t_end, dt0=dt0, y0=s0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts), max_steps=max_steps,
            throw=False,
        )
        return sol.ys[:, 0]          # observable component only

    return integrate_one


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _pack_jax(pack):
    return {k: jnp.asarray(np.asarray(pack[k])) for k in
            ("ts", "ys", "us", "loss_mask", "t_end", "y0")}


def _loss(model, pack, integrate_one, jac_reg):
    preds = jax.vmap(integrate_one, in_axes=(None, 0, 0, 0, 0))(
        model, pack["y0"], pack["ts"], pack["us"], pack["t_end"])
    preds = jnp.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    mask = pack["loss_mask"]
    n = jnp.maximum(mask.sum(), 1.0)
    mse = jnp.sum(((preds - pack["ys"]) ** 2) * mask) / n

    if jac_reg > 0.0:
        # Same L21 group-sparse pressure as the published model, applied to the
        # observable equation's dependence on the measured inputs, so that d=0
        # is a like-for-like reproduction of the reference architecture.
        def dy0_du(s, u):
            return model.rhs(jnp.concatenate([s, u]))[0]
        B, T = pack["ts"].shape
        s_dim = model.rhs.state_dim
        states = jnp.concatenate(
            [pack["ys"].reshape(-1, 1), jnp.zeros((B * T, s_dim - 1))], axis=1)
        us_flat = pack["us"].reshape(B * T, -1)
        grads = jax.vmap(jax.grad(dy0_du, argnums=1))(states, us_flat)
        # L21: L2 across samples per input, then L1 across inputs.
        mse = mse + jac_reg * jnp.sum(jnp.sqrt(jnp.mean(grads ** 2, axis=0) + 1e-12))
    return mse


def train(bundle, latent_dim, seed, args):
    nodb = _baseline()
    n_exo = len(bundle["exo_cols"])
    key = jax.random.PRNGKey(seed)
    k_rhs, k_enc = jax.random.split(key)
    model = LatentModel(
        rhs=LatentRHS(1 + latent_dim, n_exo, args.hidden_dim, args.hidden_layers,
                      args.activation, k_rhs),
        enc=LatentEncoder(n_exo, latent_dim, args.hidden_dim, args.activation, k_enc),
    )
    integrate_one = _make_integrator(args.solver, args.rtol, args.atol,
                                     args.dt0, args.max_steps)

    train_pack, val_pack = _pack_jax(bundle["train"]), _pack_jax(bundle["val"])
    jac_reg = float(args.jac_reg)

    @eqx.filter_jit
    def loss_fn(m, pack, reg):
        return _loss(m, pack, integrate_one, reg)

    @eqx.filter_jit
    def step_fn(m, opt_state, pack):
        loss, grads = eqx.filter_value_and_grad(
            lambda mm: _loss(mm, pack, integrate_one, jac_reg))(m)
        new_m, new_state = nodb._adam_apply(
            m, grads, opt_state, lr=args.lr, wd=args.weight_decay)
        return new_m, new_state, loss

    opt_state = nodb._adam_init(model)
    best_val, best_model, bad, epochs_run = np.inf, model, 0, 0
    for epoch in range(args.epochs):
        model, opt_state, _ = step_fn(model, opt_state, train_pack)
        # Validation on trajectory MSE only (no penalty), matching the baseline's
        # early-stopping criterion so d=0 stops where the published model stops.
        val = float(loss_fn(model, val_pack, 0.0))
        epochs_run = epoch + 1
        if np.isfinite(val) and val < best_val - args.min_delta:
            best_val, best_model, bad = val, model, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best_model, best_val, epochs_run, integrate_one


def evaluate(model, bundle, split, integrate_one, args):
    """Held-out R^2 on the observable, scored exactly like the baseline's
    `ode_integ_r2_median`: per-bin R^2 at measured timepoints, then the median."""
    from pipelines.experimental.sr_pipeline.metrics import coefficient_of_determination
    pack = bundle[split]
    if pack is None or not pack["trajs"]:
        return {"r2": float("nan"), "rmse": float("nan")}
    jp = _pack_jax(pack)
    preds = np.asarray(jax.vmap(integrate_one, in_axes=(None, 0, 0, 0, 0))(
        model, jp["y0"], jp["ts"], jp["us"], jp["t_end"]))
    measured = np.asarray(args.measured_timepoints, float)

    r2s, rmses = [], []
    for i, tr in enumerate(pack["trajs"]):
        t = np.asarray(tr["t"])
        y = np.asarray(tr.get("y_eval", tr["y"]))
        m = np.isin(t, measured)
        if m.sum() < 2:
            continue
        p = np.nan_to_num(preds[i][:len(t)][m], nan=0.0, posinf=0.0, neginf=0.0)
        r2 = coefficient_of_determination(y[m], p)
        if np.isfinite(r2):
            r2s.append(max(0.0, r2))
        rmses.append(float(np.sqrt(np.mean((y[m] - p) ** 2))))
    return {"r2": float(np.median(r2s)) if r2s else float("nan"),
            "rmse": float(np.mean(rmses)) if rmses else float("nan")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir() / "latent")
    ap.add_argument("--markers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(PAPER_CFG["seeds"]))
    ap.add_argument("--latent-dims", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--resume", action="store_true")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "latent_node.csv"

    args = make_baseline_args(outdir, epochs=args_cli.epochs)
    raw = load_raw(args)
    markers = args_cli.markers or all_markers(raw)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(m), int(s), int(d))
                for m, s, d in zip(prev.marker, prev.seed, prev.latent_dim)}
        print(f"resuming: {len(done)} rows already present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    for marker in markers:
        for seed in args_cli.seeds:
            bundle = load_bundle(raw, marker, seed, args)
            if bundle is None:
                print(f"[skip] {marker} seed {seed}", flush=True)
                continue
            for d in args_cli.latent_dims:
                if (marker, seed, d) in done:
                    continue
                t0 = time.time()
                model, best_val, epochs_run, integ = train(bundle, d, seed, args)
                m = {s: evaluate(model, bundle, s, integ, args)
                     for s in ("train", "val", "test")}
                n_params = sum(x.size for x in
                               jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
                row = {
                    "marker": marker, "seed": seed, "latent_dim": d,
                    "train_r2": m["train"]["r2"], "val_r2": m["val"]["r2"],
                    "test_r2": m["test"]["r2"], "test_rmse": m["test"]["rmse"],
                    "best_val_loss": best_val, "epochs_run": epochs_run,
                    "n_params": int(n_params),
                    "wall_seconds": round(time.time() - t0, 2),
                }
                pd.DataFrame([row])[COLUMNS].to_csv(
                    out_csv, mode="a", header=False, index=False)
                print(f"[{marker} s{seed}] d={d}: test R2={row['test_r2']:.3f} "
                      f"(val {row['val_r2']:.3f}, {epochs_run} ep, "
                      f"{row['wall_seconds']:.0f}s)", flush=True)

    # --- the quantity to report: does d help MORE where SR failed? -----------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    if df.empty:
        return 0
    base = df[df.latent_dim == 0][["marker", "seed", "test_r2"]].rename(
        columns={"test_r2": "test_r2_d0"})
    gain = df.merge(base, on=["marker", "seed"], how="left")
    gain["gain_over_d0"] = gain["test_r2"] - gain["test_r2_d0"]
    write_csv(gain, outdir / "latent_gain.csv")

    print("\n--- held-out R^2 by latent dimension ---")
    print(gain.groupby("latent_dim")
          .agg(mean_test_r2=("test_r2", "mean"),
               mean_gain=("gain_over_d0", "mean"),
               n=("test_r2", "size")).round(4).to_string())
    print("\nJoin latent_gain.csv against pr_context_level.csv on `marker` and "
          "contrast mean_gain between SR-fail and SR-success contexts —\n"
          "that interaction, not the main effect of d, is the hidden-state evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
