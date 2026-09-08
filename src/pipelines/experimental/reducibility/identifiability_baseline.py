"""Step 8 — what does SR-failure buy over the identifiability / observability toolkit?

The novelty objection
---------------------
The manuscript cites the profile-likelihood (Raue et al.), network-observability
(Liu / Slotine / Barabasi) and manifold-boundary reduction (Transtrum / Qiu)
literatures, and then never argues what SR-failure adds over them. Without that
delta, "SR failure diagnoses non-reducibility" reads as a restatement of
structural non-identifiability in a new vocabulary — which is exactly how a
general-interest editor will read it, and it is the objection that no amount of
extra statistical rigour on the participation ratio will answer.

This script runs the standard diagnostics on the *same* contexts, the same OOD
split, so the comparison is head-to-head rather than rhetorical.

Baselines computed per (marker, seed)
-------------------------------------
1. **Observability rank / condition number.** Build the empirical observability
   Gramian of the fitted Neural ODE around the training trajectories, from the
   sensitivity of the observed p-ERK output to state and input perturbations.
   Its numerical rank is the classical answer to "how many directions are
   distinguishable from this output"; the condition number says how badly.

2. **Profile-likelihood-style flatness.** For the *linear* rate-law surrogate
   (which is what admits a cheap exact profile), profile each coefficient: refit
   with that coefficient fixed across a grid and record the increase in
   training loss. A coefficient whose profile stays flat is practically
   non-identifiable. The count of flat directions is the identifiability
   literature's dimensionality answer.

3. **Correlation-based collinearity.** Condition number of the input correlation
   matrix and the variance inflation factor per input — the cheapest possible
   explanation for everything, and therefore the one that has to be excluded.

The comparison that matters
---------------------------
For each context, we then have: SR outcome, Neural-ODE outcome, participation
ratio, k*, observability rank, count of non-identifiable directions,
collinearity. The claim the paper needs is that **SR-failure carries information
the others do not** — concretely, that SR outcome is not predictable from
observability rank and non-identifiability count alone. This script fits exactly
that: a logistic model of SR success on the classical diagnostics, and then asks
whether adding the SR-derived quantities improves it (and vice versa).

Three possible readings, all publishable:
  * classical diagnostics predict SR outcome well -> SR-failure IS a restatement,
    and the paper should say so and pivot to the practical advantages
    (no model specification needed, works on partial data).
  * classical diagnostics predict poorly, SR outcome tracks something else ->
    the delta is real and this is the head-to-head the novelty case needs.
  * neither predicts -> the SR/no-SR split is dominated by nuisance, which is
    the most important thing to know before submitting.

Usage:
    python identifiability_baseline.py --output-dir <dir> [--markers ...]
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
    SUCCESS_THRESHOLD,
    _baseline,
    _jax,
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

COLUMNS = ["marker", "seed", "n_inputs", "obs_rank", "obs_cond",
           "n_flat_directions", "flat_fraction", "profile_curvature_mean",
           "profile_curvature_median", "profile_curvature_max",
           "input_cond_number", "max_vif", "mean_abs_corr", "pr",
           "node_test_r2", "wall_seconds"]


# -----------------------------------------------------------------------------
# 1. Empirical observability
# -----------------------------------------------------------------------------

def observability(rhs, X: np.ndarray, tol_ratio: float = 1e-3) -> dict:
    """Rank and conditioning of the empirical observability Gramian.

    The output map here is the state itself (p-ERK is measured), so what limits
    distinguishability is how the RHS responds to perturbations along each
    input direction. G = J^T J accumulated over the operating points is the
    standard empirical Gramian for that; its numerical rank is the number of
    directions the output can resolve.
    """
    jax, jnp = _jax()
    if X.size == 0:
        return {"rank": np.nan, "cond": np.nan}
    jac = np.asarray(jax.jit(jax.vmap(jax.grad(rhs)))(jnp.asarray(X)))  # (N, F+1)
    G = jac.T @ jac / max(len(jac), 1)
    evals = np.linalg.eigvalsh(G)
    evals = np.clip(evals, 0.0, None)[::-1]
    if evals[0] <= 0:
        return {"rank": 0.0, "cond": np.inf}
    rank = int(np.sum(evals > evals[0] * tol_ratio))
    smallest_pos = evals[evals > 0].min()
    return {"rank": float(rank), "cond": float(evals[0] / smallest_pos)}


# -----------------------------------------------------------------------------
# 2. Profile-likelihood flatness on the linear surrogate
# -----------------------------------------------------------------------------

def profile_flatness(X: np.ndarray, y: np.ndarray, n_grid: int = 21,
                     span: float = 3.0, flat_threshold: float = 0.05) -> dict:
    """Count practically non-identifiable coefficients of the linear surrogate.

    For each coefficient j: fix it across a grid spanning +/- `span` standard
    errors, refit the others by least squares, and record how much the residual
    sum of squares rises. If the worst rise across the grid is below
    `flat_threshold` (relative to the optimum), that direction is flat — the
    data do not constrain it. This is the discrete analogue of a profile
    likelihood, exact for the linear model and cheap enough to run everywhere.

    The linear surrogate is used deliberately: it is the same all-ten-input
    linear bound the manuscript already reports, so the identifiability answer
    is computed on a model the paper already stands behind.
    """
    if X.size == 0 or len(y) < X.shape[1] + 2:
        return {"n_flat": np.nan, "flat_fraction": np.nan}
    Xd = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    rss0 = float(resid @ resid)
    if rss0 <= 0:
        return {"n_flat": 0.0, "flat_fraction": 0.0}

    n, p = Xd.shape
    sigma2 = rss0 / max(n - p, 1)
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * sigma2, 0, None))

    n_flat = 0
    curvatures = []
    n_tested = X.shape[1]          # intercept is not a mechanistic direction
    for j in range(n_tested):
        if se[j] <= 0:
            continue
        grid = beta[j] + np.linspace(-span, span, n_grid) * se[j]
        others = [k for k in range(p) if k != j]
        worst = 0.0
        for v in grid:
            y_adj = y - Xd[:, j] * v
            b_o, *_ = np.linalg.lstsq(Xd[:, others], y_adj, rcond=None)
            r = y_adj - Xd[:, others] @ b_o
            worst = max(worst, (float(r @ r) - rss0) / rss0)
        curvatures.append(worst)
        if worst < flat_threshold:
            n_flat += 1
    # The binary count saturates on this data — every direction is flat in every
    # context, so it has no variance and cannot discriminate between contexts.
    # The continuous profile curvature is retained as the usable comparator, and
    # the saturation of the binary version is itself reportable: the classical
    # identifiability verdict is "everything is non-identifiable everywhere",
    # which is exactly the situation a graded readout could improve on.
    curv = np.asarray(curvatures, float) if curvatures else np.array([np.nan])
    return {"n_flat": float(n_flat),
            "flat_fraction": float(n_flat / max(n_tested, 1)),
            "profile_curvature_mean": float(np.nanmean(curv)),
            "profile_curvature_max": float(np.nanmax(curv)),
            "profile_curvature_median": float(np.nanmedian(curv))}


# -----------------------------------------------------------------------------
# 3. Plain collinearity
# -----------------------------------------------------------------------------

def collinearity(X: np.ndarray) -> dict:
    if X.size == 0 or X.shape[1] < 2:
        return {"cond": np.nan, "max_vif": np.nan, "mean_abs_corr": np.nan}
    Xc = X - X.mean(axis=0)
    sd = Xc.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = Xc / sd
    corr = np.corrcoef(Z, rowvar=False)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    svals = np.linalg.svd(Z, compute_uv=False)
    cond = float(svals[0] / max(svals[-1], 1e-12))

    vifs = []
    for j in range(Z.shape[1]):
        others = [k for k in range(Z.shape[1]) if k != j]
        b, *_ = np.linalg.lstsq(Z[:, others], Z[:, j], rcond=None)
        r = Z[:, j] - Z[:, others] @ b
        ss_tot = float(Z[:, j] @ Z[:, j])
        r2 = 1.0 - float(r @ r) / max(ss_tot, 1e-12)
        vifs.append(1.0 / max(1.0 - r2, 1e-6))
    return {"cond": cond, "max_vif": float(np.max(vifs)),
            "mean_abs_corr": float(np.mean(np.abs(off)))}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path,
                    default=default_outdir() / "identifiability")
    ap.add_argument("--markers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(PAPER_CFG["seeds"]))
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--master", type=Path,
                    default=default_outdir() / "reducibility_master.csv")
    ap.add_argument("--resume", action="store_true")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "identifiability.csv"

    nodb = _baseline()
    args = make_baseline_args(outdir, epochs=args_cli.epochs)
    raw = load_raw(args)
    markers = args_cli.markers or all_markers(raw)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(m), int(s)) for m, s in zip(prev.marker, prev.seed)}
        print(f"resuming: {len(done)} rows present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    for marker in markers:
        for seed in args_cli.seeds:
            if (marker, seed) in done:
                continue
            bundle = load_bundle(raw, marker, seed, args)
            if bundle is None:
                continue
            bundle["marker"], bundle["seed"] = marker, seed
            t0 = time.time()

            rhs, _h, _ = nodb._train_marker(marker, seed, args, bundle)
            X = input_matrix(bundle, "train")
            m_test = nodb._split_metrics(rhs, bundle, bundle["test"], args)

            obs = observability(rhs, X)
            col = collinearity(X)
            # Linear surrogate target: the observed derivative, as in the
            # manuscript's all-ten-input linear bound.
            dt_obs = np.concatenate([np.asarray(tr["dt_obs"])
                                     for tr in bundle["train"]["trajs"]])
            prof = profile_flatness(X, dt_obs)
            pr = jacobian_summary(rhs, X)["pr"]

            row = {
                "marker": marker, "seed": seed, "n_inputs": X.shape[1],
                "obs_rank": obs["rank"], "obs_cond": obs["cond"],
                "n_flat_directions": prof["n_flat"],
                "flat_fraction": prof["flat_fraction"],
                "profile_curvature_mean": prof["profile_curvature_mean"],
                "profile_curvature_median": prof["profile_curvature_median"],
                "profile_curvature_max": prof["profile_curvature_max"],
                "input_cond_number": col["cond"], "max_vif": col["max_vif"],
                "mean_abs_corr": col["mean_abs_corr"], "pr": pr,
                "node_test_r2": m_test["ode_integ_r2_median"],
                "wall_seconds": round(time.time() - t0, 2),
            }
            pd.DataFrame([row])[COLUMNS].to_csv(
                out_csv, mode="a", header=False, index=False)
            # n_flat is counted over all X columns (state + exogenous), which is
            # the same set the observability rank is computed on.
            print(f"[{marker} s{seed}] obs_rank={obs['rank']:.0f}/{X.shape[1]} "
                  f"flat={prof['n_flat']:.0f}/{X.shape[1]} "
                  f"VIF_max={col['max_vif']:.1f} PR={pr:.2f}", flush=True)

    # --- head-to-head: does SR outcome add anything? -------------------------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    if df.empty or not args_cli.master.exists():
        return 0
    master = pd.read_csv(args_cli.master)
    sr = (master.groupby("marker")["pysr_ode_r2"].max()
          .rename("sr_r2_best").reset_index())
    joined = df.merge(sr, on="marker", how="left")
    joined["sr_success"] = joined["sr_r2_best"] >= SUCCESS_THRESHOLD
    write_csv(joined, outdir / "identifiability_vs_sr.csv")

    print("\n--- classical diagnostics, by SR outcome ---")
    print(joined.groupby("sr_success")[
        ["obs_rank", "obs_cond", "n_flat_directions", "profile_curvature_mean",
         "max_vif", "mean_abs_corr", "pr"]].mean().round(3).to_string())

    try:
        import statsmodels.api as sm
        d = joined.dropna(subset=["obs_rank", "profile_curvature_mean", "pr",
                                  "sr_success"])
        if d["sr_success"].nunique() == 2 and len(d) > 12:
            classical = ["obs_rank", "profile_curvature_mean", "max_vif"]
            y = d["sr_success"].astype(float)
            m1 = sm.Logit(y, sm.add_constant(d[classical])).fit(disp=0)
            m2 = sm.Logit(y, sm.add_constant(d[classical + ["pr"]])).fit(disp=0)
            print(f"\nlogit(SR success) pseudo-R^2:")
            print(f"  classical diagnostics only     : {m1.prsquared:.3f}")
            print(f"  classical + participation ratio: {m2.prsquared:.3f}")
            print(f"  likelihood-ratio P for adding PR: "
                  f"{m2.compare_lr_test(m1)[1]:.4f}")
            print("\nIf adding PR does not improve fit, SR-failure is "
                  "predictable from the classical toolkit\nand the novelty case "
                  "must rest on practical advantages, not on new information.")
    except Exception as exc:
        print(f"\n[logit comparison skipped: {exc}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
