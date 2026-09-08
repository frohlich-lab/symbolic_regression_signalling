"""Step 5 — calibrate the participation ratio against known ground truth.

The confound
------------
The published inference reads a participation-ratio gap (SR-fail contexts lean
on ~4.2 inputs, SR-success contexts on ~3.2) as a difference in intrinsic
dimensionality. But the manuscript also concedes the ten inputs are strongly
correlated, and correlation is exactly what breaks this reading: a group-sparse
(L21) penalty has no reason to prefer one member of a correlated block over
another, so it spreads Jacobian mass across the whole block and inflates PR,
while SR's parsimony pressure picks a single representative. Under that story
the gap reports *how each method treats correlated inputs*, not how many
variables the dynamics need.

Arguing about this in prose is unwinnable. Measuring it is straightforward.

The calibration
---------------
Hold the inputs fixed at their real, real-correlated values — the same
`U_scaled` trajectories, the same GFP bins, the same six measured timepoints,
the same OOD split — and replace only the *state*: simulate p-ERK from a rate
law with a known number of true drivers, k_true. Then run the identical L21
Neural ODE and the identical PR computation, and ask what PR it reports.

    PR(measured) vs k_true, on real input correlations

Both possible outcomes are publishable, which is what makes this worth running:

  * PR tracks k_true  -> the metric is calibrated on this data's correlation
    structure, and 4.19 vs 3.23 can be quoted in units of true drivers. The
    confound objection is answered with a measurement rather than a caveat.
  * PR saturates regardless of k_true -> PR is correlation-limited here, the
    published contrast cannot bear the interpretation placed on it, and the
    honest move is to report k* from `input_ablation.py` instead. Finding this
    before a referee does is strictly better than after.

A `--shuffle-inputs` control decorrelates the exogenous block by permuting each
input independently across bins, destroying cross-input correlation while
preserving each input's marginal distribution. Comparing PR with and without it
isolates how much of the measured PR is correlation rather than dependence.

Ground-truth laws (all driven by real inputs u, state y, in scaled space):
    k_true = 0 : dy/dt = -a y                       (autonomous decay only)
    k_true = m : dy/dt = -a y + sum_i b_i g(u_i)    (m distinct drivers)
where g is a saturating nonlinearity, so the target is not linear-in-inputs and
the network has something to fit. The true "effective dependency count" of these
laws is m + 1 (the m drivers plus the state), which is the quantity PR is
supposed to recover.

Usage:
    python pr_calibration.py --output-dir <dir> [--markers ...] [--k-true 0 1 2 3 4 6]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    PAPER_CFG,
    _baseline,
    all_markers,
    default_outdir,
    input_matrix,
    input_names,
    jacobian_summary,
    load_bundle,
    load_raw,
    make_baseline_args,
    write_csv,
)

# The thresholded count is the quantity actually reported, so it must be
# calibrated alongside PR rather than assumed to inherit PR's calibration.
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
COLUMNS = (["marker", "seed", "k_true", "rep", "shuffled", "n_exo",
           "true_drivers", "pr_measured", "pr_true", "test_r2", "train_r2",
           "recovered_top_k", "precision_at_k"]
           + [f"n_above_{t}" for t in THRESHOLDS] + ["wall_seconds"])


# -----------------------------------------------------------------------------
# Synthetic ground-truth dynamics on real inputs
# -----------------------------------------------------------------------------

def simulate_state(bundle, driver_idx, weights, decay, substeps: int = 20):
    """Integrate dy/dt = -a*y + sum_i b_i * tanh(u_i) along each bin's own grid.

    Explicit sub-stepped Euler rather than a stiff solver: the law is smooth and
    mildly damped, and a transparent integrator keeps the ground truth auditable.
    Returns {split: [y_per_bin]} plus the matching derivative, both in the
    scaled space the network trains in.
    """
    out = {}
    for split in ("train", "val", "test"):
        pack = bundle[split]
        if pack is None:
            out[split] = None
            continue
        ys, dys = [], []
        for tr in pack["trajs"]:
            t = np.asarray(tr["t"], float)
            U = np.asarray(tr["U_scaled"], float)
            y = np.zeros_like(t)
            dy = np.zeros_like(t)

            def rhs(state, u_row):
                drive = float(np.sum(weights * np.tanh(u_row[driver_idx]))) \
                    if driver_idx.size else 0.0
                return -decay * state + drive

            state = 0.0
            y[0], dy[0] = state, rhs(state, U[0]) if U.size else -decay * state
            for j in range(len(t) - 1):
                h = (t[j + 1] - t[j]) / substeps
                for s in range(substeps):
                    frac = s / substeps
                    u_row = U[j] + frac * (U[j + 1] - U[j]) if U.size else U[j]
                    state = state + h * rhs(state, u_row)
                y[j + 1] = state
                dy[j + 1] = rhs(state, U[j + 1]) if U.size else -decay * state
            ys.append(y)
            dys.append(dy)
        out[split] = (ys, dys)
    return out


def install_synthetic_state(bundle, sim):
    """Return a bundle whose p-ERK state is the simulated one.

    The synthetic state is generated directly in scaled space, so y_mean/y_scale
    are set to 0/1 — there is no raw-measurement scale to preserve here, and
    collapsing them removes a source of silent unit confusion.
    """
    import copy
    out = copy.copy(bundle)
    out["y_mean"], out["y_scale"] = np.float64(0.0), np.float64(1.0)
    for split in ("train", "val", "test"):
        pack = bundle[split]
        if pack is None or sim[split] is None:
            out[split] = pack
            continue
        ys, dys = sim[split]
        new_pack = dict(pack)
        arr_ys = np.array(new_pack["ys"], copy=True)
        new_trajs = []
        for i, (tr, y, dy) in enumerate(zip(pack["trajs"], ys, dys)):
            new_tr = dict(tr)
            new_tr["y"] = y
            new_tr["y_eval"] = y
            new_tr["dt_obs"] = dy
            new_trajs.append(new_tr)
            arr_ys[i, :len(y)] = y
            if len(y) < arr_ys.shape[1]:
                arr_ys[i, len(y):] = y[-1]
        new_pack["trajs"] = new_trajs
        new_pack["ys"] = arr_ys
        new_pack["y0"] = np.array([float(y[0]) for y in ys])
        out[split] = new_pack
    return out


def shuffle_exogenous(bundle, rng):
    """Decorrelate the exogenous block: permute each input across bins independently.

    Marginals are preserved exactly; cross-input correlation is destroyed. Any
    PR difference between shuffled and unshuffled runs at the same k_true is
    attributable to correlation.
    """
    import copy
    out = copy.copy(bundle)
    for split in ("train", "val", "test"):
        pack = bundle[split]
        if pack is None:
            continue
        new_pack = dict(pack)
        us = np.array(pack["us"], copy=True)          # (B, T, F)
        for f in range(us.shape[-1]):
            us[:, :, f] = us[rng.permutation(us.shape[0]), :, f]
        new_pack["us"] = us
        new_trajs = []
        for i, tr in enumerate(pack["trajs"]):
            new_tr = dict(tr)
            n = len(tr["t"])
            new_tr["U_scaled"] = us[i, :n, :]
            new_trajs.append(new_tr)
        new_pack["trajs"] = new_trajs
        out[split] = new_pack
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir() / "calibration")
    ap.add_argument("--markers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--k-true", nargs="*", type=int, default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--reps", type=int, default=3,
                    help="Random driver subsets per k_true (averages over which "
                         "inputs happen to be chosen).")
    ap.add_argument("--decay", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--patience", type=int, default=int(PAPER_CFG["patience"]))
    ap.add_argument("--hidden-layers", type=int, default=int(PAPER_CFG["hidden_layers"]))
    ap.add_argument("--shuffle-control", action="store_true",
                    help="Also run every cell with the exogenous block decorrelated.")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "pr_calibration.csv"

    nodb = _baseline()
    args = make_baseline_args(outdir, epochs=args_cli.epochs,
                              patience=args_cli.patience,
                              hidden_layers=args_cli.hidden_layers)
    raw = load_raw(args)
    markers = args_cli.markers or all_markers(raw)
    rng = np.random.default_rng(args_cli.rng_seed)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(m), int(s), int(k), int(r), bool(sh)) for m, s, k, r, sh in
                zip(prev.marker, prev.seed, prev.k_true, prev.rep, prev.shuffled)}
        print(f"resuming: {len(done)} rows already present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    shuffle_modes = [False, True] if args_cli.shuffle_control else [False]

    for marker in markers:
        for seed in args_cli.seeds:
            base_bundle = load_bundle(raw, marker, seed, args)
            if base_bundle is None:
                print(f"[skip] {marker} seed {seed}", flush=True)
                continue
            names = input_names(base_bundle)
            n_exo = len(base_bundle["exo_cols"])

            for k_true in args_cli.k_true:
                if k_true > n_exo:
                    continue
                for rep in range(args_cli.reps):
                    # Same driver subset and weights for both shuffle modes, so
                    # the control differs only in input correlation.
                    driver_idx = rng.choice(n_exo, size=k_true, replace=False) \
                        if k_true else np.array([], dtype=int)
                    driver_idx = np.sort(driver_idx)
                    weights = rng.uniform(0.5, 1.5, size=k_true) * \
                        rng.choice([-1.0, 1.0], size=k_true)

                    for shuffled in shuffle_modes:
                        key = (marker, seed, k_true, rep, shuffled)
                        if key in done:
                            continue
                        t0 = time.time()
                        bundle = (shuffle_exogenous(base_bundle, rng) if shuffled
                                  else base_bundle)
                        sim = simulate_state(bundle, driver_idx, weights,
                                             args_cli.decay)
                        synth = install_synthetic_state(bundle, sim)
                        synth["marker"], synth["seed"] = marker, seed

                        rhs, _hist, _ = nodb._train_marker(marker, seed, args, synth)
                        summary = jacobian_summary(rhs, input_matrix(synth, "train"))
                        m_test = nodb._split_metrics(rhs, synth, synth["test"], args)
                        m_train = nodb._split_metrics(rhs, synth, synth["train"], args)

                        # Did the network find the RIGHT inputs, not just the
                        # right number? precision@k over the true driver set.
                        abs_all = np.asarray(summary["abs_jac"])
                        norm = abs_all / max(abs_all.max(), 1e-12)
                        counts = {f"n_above_{t}": float((norm > t).sum())
                                  for t in THRESHOLDS}
                        exo_abs = abs_all[1:]
                        top_k = set(np.argsort(-exo_abs)[:k_true].tolist())
                        truth = set(driver_idx.tolist())
                        precision = (len(top_k & truth) / k_true) if k_true else np.nan

                        row = {
                            "marker": marker, "seed": seed, "k_true": k_true,
                            "rep": rep, "shuffled": shuffled, "n_exo": n_exo,
                            "true_drivers": "|".join(names[1 + i] for i in driver_idx),
                            "pr_measured": summary["pr"],
                            # What PR *should* read if it recovered the truth:
                            # k_true drivers plus the state itself.
                            "pr_true": float(k_true + 1),
                            "test_r2": m_test["ode_integ_r2_median"],
                            "train_r2": m_train["ode_integ_r2_median"],
                            "recovered_top_k": "|".join(
                                names[1 + i] for i in sorted(top_k)),
                            "precision_at_k": precision,
                            **counts,
                            "wall_seconds": round(time.time() - t0, 2),
                        }
                        pd.DataFrame([row])[COLUMNS].to_csv(
                            out_csv, mode="a", header=False, index=False)
                        print(f"[{marker} s{seed}] k_true={k_true} rep={rep} "
                              f"shuf={int(shuffled)}: PR={summary['pr']:.2f} "
                              f"(true {k_true + 1}) fit R2={row['train_r2']:.2f} "
                              f"prec@k={precision if k_true else float('nan'):.2f}",
                              flush=True)

    # --- calibration curve ---------------------------------------------------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    if df.empty:
        return 0
    curve = (df.groupby(["shuffled", "k_true"])
               .agg(pr_mean=("pr_measured", "mean"),
                    pr_sd=("pr_measured", "std"),
                    pr_true=("pr_true", "first"),
                    precision=("precision_at_k", "mean"),
                    fit_r2=("train_r2", "mean"),
                    n=("pr_measured", "size"))
               .reset_index())
    curve["pr_bias"] = curve["pr_mean"] - curve["pr_true"]
    write_csv(curve, outdir / "pr_calibration_curve.csv")

    print("\n--- PR calibration: what PR reads when the truth is known ---")
    print(curve.round(3).to_string(index=False))
    # A slope near 1 means PR is calibrated; near 0 means it is saturated and
    # cannot support a dimensionality reading on this input correlation.
    for shuffled, g in curve.groupby("shuffled"):
        if len(g) >= 2 and g["k_true"].nunique() >= 2:
            slope = np.polyfit(g["k_true"], g["pr_mean"], 1)[0]
            print(f"  shuffled={bool(shuffled)}: dPR/dk_true = {slope:.3f} "
                  f"(1.0 = perfectly calibrated, 0.0 = PR is blind to k_true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
