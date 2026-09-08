"""Step 4 — measure dimensionality directly, by nested input ablation.

The participation ratio is a *proxy*: it summarises how the Jacobian's mass is
spread over inputs, which is not the same thing as how many inputs the dynamics
need. Worse, it is exactly the sort of proxy that correlated inputs corrupt — an
L21 penalty distributes weight across a correlated block, inflating PR, while
SR's parsimony pressure picks one representative of the block. That confound
alone can produce the published SR-fail-vs-SR-success PR gap without any
difference in true dimensionality.

This script replaces the proxy with a measurement:

    k* = the smallest number of measured inputs with which a Neural ODE
         reaches within `--epsilon` of the best achievable VALIDATION fit.

Why validation, and not the held-out test split
-----------------------------------------------
Dimensionality is a property of the fitted model, not of its test performance.
On the symbolic-regression side the paper counts the variables appearing in the
recovered expression -- a quantity derived wholly from the fit, with test
performance playing no part. k* must be derived the same way or the two sides
of the comparison are not the same kind of number.

So: PERFORMANCE is compared on test (both methods); DIMENSIONALITY is selected
on validation (both methods). Mixing those per-quantity is correct; mixing them
per-method is the error. Reading k* off test would additionally make it the one
quantity in the paper selected on the evaluation split.

It is also the least noisy choice: seed-to-seed SD of the integrated R^2 is
0.095 on validation against 0.157 on test, so the same number of seeds buys
nearly twice the resolution.

`k_star_train` and `k_star_test` are recorded as diagnostics only. They differ
systematically rather than randomly -- training R^2 is monotone in k by nesting
so its elbow sits late, test R^2 decays under extrapolation so its elbow sits
early -- and the gap between them measures how much of the input dependence
generalises.

Procedure, per (marker, seed):
  1. Train the full model; rank exogenous inputs by mean |dJ/du| on train rows.
  2. For k = 0 .. n_exo, retrain from scratch on p-ERK plus the top-k inputs,
     retrying any fit that lands below the previous k's training R^2 (nesting
     guarantees that cannot happen except by failed optimisation).
  3. Pool seeds at each k; k* is the first k within epsilon of the best
     validation median.

Retraining rather than masking matters: masking an input in an already-trained
network measures that network's sensitivity, whereas retraining asks the
question we actually care about — is a k-input description *achievable*.

k* is interpretable in the way the manuscript wants PR to be ("this context
needs five measured variables, that one needs two"), it is not inflated by
correlation (a correlated block contributes one useful input, and the greedy
ranking will not pay for the second), and it supports the drop-and-restore
demonstration: remove the input k* says is load-bearing, show SR fails; restore
it, show the compact law returns.

Cost: (n_exo + 1) trainings per (marker, seed). With 9 exogenous inputs, 40
markers and 3 seeds that is 1200 fits — use --markers / --seeds to scope it, and
--resume to build the table incrementally.

Usage:
    python input_ablation.py --output-dir <dir> [--markers PTPN7 PIKFYVE] [--seeds 42]
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    PAPER_CFG,
    _jax,
    _baseline,
    all_markers,
    default_outdir,
    input_matrix,
    input_names,
    jacobian_summary,
    load_bundle,
    load_raw,
    make_baseline_args,
    subset_bundle,
    write_csv,
)

COLUMNS = ["marker", "seed", "k", "n_exo", "kept_inputs", "dropped_inputs",
           "train_r2", "val_r2", "test_r2", "test_dt_r2", "val_loss", "pr", "jac_reg",
           "wall_seconds"]


def rank_inputs(rhs, bundle) -> list[int]:
    """Exogenous input indices ordered by mean |Jacobian| on train rows, desc.

    Index 0 of the Jacobian is p-ERK (the state), which is never a candidate for
    removal, so it is stripped before ranking.
    """
    summary = jacobian_summary(rhs, input_matrix(bundle, "train"))
    exo_abs = np.asarray(summary["abs_jac"])[1:]
    return list(np.argsort(-exo_abs))


def evaluate(bundle, args, floor: float = -np.inf, retries: int = 3,
             tol: float = 0.02) -> tuple[object, dict]:
    """Train on `bundle`, retrying inits that fail to reach a known-achievable fit.

    Adding an input strictly enlarges the function class: the k-input network can
    represent every (k-1)-input network by zeroing a weight column. So TRAIN R^2
    must be non-decreasing in k, and a run that lands below the previous k's
    training fit has failed to optimise — it is not evidence about the inputs.
    Theory therefore hands us a free correctness check, and `floor` is the
    previous k's training R^2.

    Retries change only the initialisation. This rejects diverged fits (e.g. a
    model with train R^2 = 0.69 scoring 0.00 on validation, which is a blown-up
    integration) without touching the genuine out-of-distribution degradation at
    high k, which reproduces across inits.
    """
    nodb = _baseline()
    best, best_metrics, best_train = None, None, -np.inf
    base_seed = int(bundle["seed"])
    for attempt in range(max(1, retries)):
        seed = base_seed if attempt == 0 else base_seed + 1000 * attempt
        rhs, info, _ = nodb._train_marker(bundle["marker"], seed, args, bundle)
        metrics = {split: nodb._split_metrics(rhs, bundle, bundle[split], args)
                   for split in ("train", "val", "test")}
        # The validation trajectory MSE that early stopping actually minimised.
        # Continuous and uncensored, unlike the per-bin R^2 median which is
        # clipped at zero -- so it resolves differences between k values that
        # the R^2 view flattens.
        metrics["best_val_loss"] = float(info.get("best_val_loss", np.nan))
        tr = metrics["train"]["ode_integ_r2_median"]
        if np.isfinite(tr) and tr > best_train:
            best, best_metrics, best_train = rhs, metrics, tr
        if np.isfinite(tr) and tr >= floor - tol:
            break
        print(f"      [retry {attempt + 1}] train R2={tr:.3f} < floor {floor:.3f}",
              flush=True)
    return best, best_metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir() / "ablation")
    ap.add_argument("--markers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(PAPER_CFG["seeds"]))
    ap.add_argument("--epsilon", type=float, default=0.05,
                    help="How close to the full-model held-out R^2 counts as 'as good'.")
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--patience", type=int, default=int(PAPER_CFG["patience"]),
                    help="Early-stopping patience in gradient steps. Training is "
                         "FULL BATCH -- one step per epoch -- so the published "
                         "200/20 gives a median of 36 steps, far short of "
                         "convergence (PTPN7 held-out R2 0.08 at 36 steps vs "
                         "0.93 at 653).")
    ap.add_argument("--hidden-layers", type=int, default=int(PAPER_CFG["hidden_layers"]),
                    help="Network depth. k* should be reported at every depth "
                         "that is indistinguishable on validation; if it agrees, "
                         "the architecture choice stops mattering.")
    ap.add_argument("--hidden-dim", type=int, default=int(PAPER_CFG["hidden_dim"]))
    ap.add_argument("--jac-reg", type=float, default=None,
                    help="Override the L21 penalty. Used to calibrate the "
                         "network's parsimony pressure against SR's complexity "
                         "penalty on contexts where both succeed.")
    ap.add_argument("--scale-jac-reg", action="store_true",
                    help="Scale the L21 penalty with the retained input count: "
                         "lambda_k = lambda_full * k / n_exo. The penalty in the "
                         "baseline is a MEAN over features, i.e. a sum divided by "
                         "F, so its per-feature weight is lambda/F. That is a "
                         "harmless rescaling at the fixed F=9 the model was "
                         "calibrated at, but across an ablation it means a "
                         "1-input model pays 9x the per-feature penalty of a "
                         "9-input one for the same behaviour -- handicapping low "
                         "k and pushing the elbow right. Standard group lasso "
                         "(Yuan & Lin 2006) sums over groups with sqrt(size) "
                         "weights for exactly this comparability reason.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (marker, seed, k) rows already in the output CSV.")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "input_ablation.csv"

    args = make_baseline_args(outdir, epochs=args_cli.epochs,
                              patience=args_cli.patience,
                              hidden_layers=args_cli.hidden_layers,
                              hidden_dim=args_cli.hidden_dim)
    if args_cli.jac_reg is not None:
        args.jac_reg = float(args_cli.jac_reg)
    raw = load_raw(args)
    markers = args_cli.markers or all_markers(raw)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(m), int(s), int(k))
                for m, s, k in zip(prev.marker, prev.seed, prev.k)}
        print(f"resuming: {len(done)} rows already present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    for marker in markers:
        for seed in args_cli.seeds:
            bundle = load_bundle(raw, marker, seed, args)
            if bundle is None:
                print(f"[skip] {marker} seed {seed}: no usable split", flush=True)
                continue
            bundle["marker"], bundle["seed"] = marker, seed
            names = input_names(bundle)
            n_exo = len(bundle["exo_cols"])

            # Full model first: it both anchors the epsilon comparison and
            # provides the greedy ranking.
            t0 = time.time()
            full_rhs, full_metrics = evaluate(bundle, args)
            order = rank_inputs(full_rhs, bundle)
            full_test = full_metrics["test"]["ode_integ_r2_median"]
            print(f"[{marker} s{seed}] full: test R2={full_test:.3f}  "
                  f"order={[names[1 + i] for i in order]}", flush=True)

            rows = []
            prev_train = -np.inf   # monotonicity floor from the previous k
            base_jac = float(args.jac_reg)
            for k in range(0, n_exo + 1):
                if (marker, seed, k) in done:
                    continue
                keep = order[:k]
                sub = subset_bundle(bundle, keep) if k < n_exo else bundle
                sub["marker"], sub["seed"] = marker, seed
                t1 = time.time()
                # Per-k penalty scaling keeps the per-feature weight at the
                # value calibrated for the full input set; k == n_exo reproduces
                # the published model exactly.
                args_k = args
                if args_cli.scale_jac_reg and k < n_exo:
                    args_k = copy.copy(args)
                    args_k.jac_reg = base_jac * (k / float(n_exo))
                rhs_k, m_k = ((full_rhs, full_metrics) if k == n_exo
                              else evaluate(sub, args_k, floor=prev_train))
                prev_train = max(prev_train, m_k["train"]["ode_integ_r2_median"]
                                 if np.isfinite(m_k["train"]["ode_integ_r2_median"])
                                 else -np.inf)
                pr_k = jacobian_summary(rhs_k, input_matrix(sub, "train"))["pr"]
                row = {
                    "marker": marker, "seed": seed, "k": k, "n_exo": n_exo,
                    "kept_inputs": "|".join(names[1 + i] for i in keep),
                    "dropped_inputs": "|".join(names[1 + i] for i in order[k:]),
                    "train_r2": m_k["train"]["ode_integ_r2_median"],
                    "val_r2": m_k["val"]["ode_integ_r2_median"],
                    "val_loss": m_k.get("best_val_loss", np.nan),
                    "test_r2": m_k["test"]["ode_integ_r2_median"],
                    "test_dt_r2": m_k["test"]["dt_r2"],
                    "pr": pr_k,
                    "jac_reg": float(args_k.jac_reg) if k < n_exo else base_jac,
                    "wall_seconds": round(time.time() - t1, 2),
                }
                # Append per row, not per marker: XLA compiles of the diffrax
                # solve can take many minutes, so a run that dies mid-marker
                # must not lose the k values it already paid for.
                pd.DataFrame([row])[COLUMNS].to_csv(
                    out_csv, mode="a", header=False, index=False)
                rows.append(row)
                print(f"    k={k:>2}  test R2={row['test_r2']:.3f}  PR={pr_k:.2f}  "
                      f"({row['wall_seconds']:.0f}s)", flush=True)

                # Each k has a different input width, so XLA compiles and caches
                # a fresh solver for it. Across 11 values those caches accumulate
                # in one process and the task is OOM-killed partway through the
                # curve. Dropping them between k values bounds peak memory; the
                # cost is recompiling, which we pay anyway at every new shape.
                jax_mod, _ = _jax()
                if hasattr(jax_mod, "clear_caches"):
                    jax_mod.clear_caches()

            print(f"[{marker} s{seed}] done in {time.time() - t0:.0f}s", flush=True)

    # --- summarise: k* per (marker, seed) ------------------------------------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    summary = summarise(df, args_cli.epsilon)
    write_csv(summary, outdir / "k_star_summary.csv")
    if not summary.empty:
        print("\n--- k* (measured inputs needed, excluding p-ERK itself) ---")
        print(summary.sort_values("marker").to_string(index=False))
        n_over = int((summary["full_worse_than_best"]).sum())
        if n_over:
            print(f"\n{n_over}/{len(summary)} contexts extrapolate WORSE with all "
                  f"inputs than with k* — the ten-input models are overfitting\n"
                  f"the held-out GFP bins, which is itself a result.")
    return 0


def summarise(df: pd.DataFrame, epsilon: float,
              min_seeds: int = 2, min_fit_r2: float = 0.6) -> pd.DataFrame:
    """k* per MARKER, pooling seeds at each k.

    Why pooling is not optional: the Neural ODE's held-out R^2 varies across
    random seeds by SD 0.157 within a fixed context, architecture and input set
    (range >0.5 in 10 of 40 contexts). k* asks which k first comes within
    `epsilon` = 0.05 of the best — so a single-seed curve is read at ~3x below
    the noise floor, and the resulting "k*" is a draw, not a measurement.
    Pooling seeds at each k and taking the median is the minimum defensible
    estimator; the reported n_seeds column says how many draws each k actually
    had, because with two or three the standard error is still ~0.09.

    Selection is on validation and reporting on test, so k* is never chosen on
    the split it is scored on.
    """
    NOT_FITTED = (
        "k* is only defined where the network fits. If no input set produces a "
        "usable model, the loss curve is flat and the elbow lands at k=0 -- which "
        "reads as 'needs no variables' when it means 'no model works'. Contexts "
        "whose best validation R^2 stays below `min_fit_r2` are therefore reported "
        "with k_star = NaN and fitted = False rather than silently contributing a "
        "spurious low k*. This is the same conditioning the manuscript applies to "
        "the participation ratio."
    )
    rows = []
    for marker, g in df.groupby("marker"):
        per_k = (g.groupby("k")
                   .agg(val_med=("val_r2", "median"), test_med=("test_r2", "median"),
                        train_med=("train_r2", "median"),
                        loss_med=("val_loss", "median"),
                        loss_sd=("val_loss", "std"),
                        val_sd=("val_r2", "std"), n_seeds=("val_r2", "size"),
                        n_exo=("n_exo", "first"),
                        kept=("kept_inputs", "first"))
                   .reset_index().sort_values("k"))
        per_k = per_k[per_k["n_seeds"] >= min_seeds]
        if per_k.empty or not np.isfinite(per_k["val_med"]).any():
            continue
        # k* on each split. Training R^2 is monotone in k by nesting, so its
        # elbow is well defined and free of the input-space extrapolation
        # confound; validation is held out; test is the hardest extrapolation.
        # Reporting all three makes the choice of split an empirical question
        # rather than an assumption -- if they agree, it does not matter.
        ks = {}
        for split, col in (("train", "train_med"), ("val", "val_med"),
                           ("test", "test_med")):
            if col not in per_k or not np.isfinite(per_k[col]).any():
                ks[split] = np.nan
                continue
            b = float(np.nanmax(per_k[col]))
            w = per_k[per_k[col] >= b - epsilon]
            ks[split] = int(w.sort_values("k")["k"].iloc[0]) if not w.empty else np.nan

        # PRIMARY: smallest k whose median validation LOSS is within a relative
        # tolerance of the best achievable. MSE is scale-dependent, so the
        # tolerance is fractional rather than absolute.
        use_loss = ("loss_med" in per_k and np.isfinite(per_k["loss_med"]).any())
        k_star_1se = np.nan
        if use_loss:
            lo = per_k[np.isfinite(per_k["loss_med"])]
            best_loss = float(np.nanmin(lo["loss_med"]))
            within = lo[lo["loss_med"] <= best_loss * (1.0 + epsilon)]

            # One-standard-error rule (Breiman et al.; Hastie/Tibshirani/Friedman
            # ESL §7.10): take the SIMPLEST model whose error is within one
            # standard error of the best. A fixed fractional band cannot work
            # here -- the seed-to-seed SD of validation loss is ~36% of its
            # median, so a 5% band sits far inside the noise and k* degenerates
            # to argmin. Scaling the band by the measured spread is the standard
            # remedy and makes the tolerance self-calibrating.
            row_min = lo.loc[lo["loss_med"].idxmin()]
            n_at_min = max(float(row_min.get("n_seeds", 1)), 1.0)
            sd_at_min = float(row_min.get("loss_sd", np.nan))
            if np.isfinite(sd_at_min) and n_at_min > 1:
                se = sd_at_min / np.sqrt(n_at_min)
                w1 = lo[lo["loss_med"] <= best_loss + se]
                if not w1.empty:
                    k_star_1se = int(w1.sort_values("k")["k"].iloc[0])
        else:
            # Fall back to clipped val R^2 for tables produced before val_loss
            # was recorded, so old runs still summarise.
            best_val = float(np.nanmax(per_k["val_med"]))
            within = per_k[per_k["val_med"] >= best_val - epsilon]
        if within.empty:
            continue
        pick = within.sort_values("k").iloc[0]

        # Validity gate: did ANY input set yield a usable model for this context?
        best_fit_r2 = float(np.nanmax(per_k["val_med"])) if np.isfinite(
            per_k["val_med"]).any() else np.nan
        fitted = bool(np.isfinite(best_fit_r2) and best_fit_r2 >= min_fit_r2)
        full = per_k[per_k.k == per_k.n_exo.iloc[0]]
        full_test = float(full["test_med"].iloc[0]) if not full.empty else np.nan
        rows.append({
            "marker": marker,
            "fitted": fitted,
            "best_val_r2": best_fit_r2,
            "k_star": int(pick["k"]) if fitted else np.nan,   # undefined if unfitted
            "k_star_raw": int(pick["k"]),      # pre-gate, for auditing
            "k_star_train": ks["train"],
            "k_star_val": ks["val"],
            "k_star_test": ks["test"],
            "k_star_splits_agree": bool(
                np.isfinite(ks["train"]) and np.isfinite(ks["val"])
                and ks["train"] == ks["val"]),
            "k_star_val_r2": float(pick["val_med"]),
            "k_star_val_loss": float(pick["loss_med"]) if "loss_med" in pick else np.nan,
            "k_star_1se": k_star_1se,
            "selected_on": "val_loss" if use_loss else "val_r2",
            "k_star_test_r2": float(pick["test_med"]),
            "k_star_inputs": pick["kept"],
            "n_seeds_at_kstar": int(pick["n_seeds"]),
            "val_sd_at_kstar": float(pick["val_sd"]) if np.isfinite(pick["val_sd"]) else np.nan,
            "full_test_r2": full_test,
            "best_test_r2_over_k": float(np.nanmax(per_k["test_med"])),
            "full_worse_than_best": bool(
                np.isfinite(full_test)
                and full_test < np.nanmax(per_k["test_med"]) - epsilon),
            "epsilon": epsilon,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
