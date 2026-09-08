"""Step 3 — re-test the participation-ratio contrast without best-of-seed selection.

The manuscript's inference that SR failure marks genuine non-reducibility rests
on a single number: mean participation ratio 4.19 in the SR-fail contexts (n=12)
versus 3.23 in the SR-success contexts (n=6), P=0.032. Three things make that
weaker than it looks, and all three are fixable from data already on disk
(3 regulariser variants x 3 seeds x 40 markers = 360 participation ratios):

1. *Selection.* Both sides of the contrast are best-of-three-seeds, so the
   groups are defined by a maximum and compared at their maxima. This script
   reports the contrast under `best`, `per_seed` and `majority` conventions.

2. *Pseudo-replication.* Per-seed rows are not independent — three seeds and
   three regularisers per marker. A t-test over pooled rows would inflate the
   effective n roughly ninefold. Handled two ways: a marker-level permutation
   test that permutes the SR label across markers (so the null respects the
   clustering exactly and needs no distributional assumption), and, when
   statsmodels is available, a mixed model with a marker random intercept.

3. *Dichotomisation.* Splitting contexts into fail/success at R^2 = 0.6 throws
   away the fact that SR succeeds in 0, 1, 2 or 3 of the seeds. The continuous
   version — SR recovery frequency, or held-out R^2 itself, against
   participation ratio — uses all 40 contexts instead of 18 and does not depend
   on where the threshold sits. This is both more powerful and harder to argue
   with, so it is reported as the primary test.

Outputs
-------
`pr_contrast_stats.csv`     every test x variant x convention, with effect size
                            and bootstrap CI, not just a P-value
`pr_context_level.csv`      per-(variant, marker) PR and SR recovery frequency
`pr_threshold_sweep.csv`    how the contrast's P and effect size move as the
                            0.6 success threshold is varied — a threshold this
                            load-bearing needs its sensitivity shown

Usage:
    python pr_stats.py [--n-perm 20000] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    REFERENCE_VARIANT,
    SUCCESS_THRESHOLD,
    default_outdir,
    write_csv,
)


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

def context_level(master: pd.DataFrame, variant: str,
                  condition_on_node: bool = False) -> pd.DataFrame:
    """One row per marker: PR summaries and SR recovery frequency.

    `condition_on_node` restricts to contexts where the Neural ODE itself
    cleared threshold. This is the manuscript's implicit conditioning — its
    n=12 SR-fail and n=6 SR-success groups sum to 18, the Neural-ODE-success
    count, not to 40. The conditioning is defensible (a participation ratio
    read off a network that does not fit is not interpretable) but it must be
    stated, because it is what makes the reported n so small.
    """
    d = master[master["variant"] == variant].dropna(subset=["node_pr"]).copy()
    if condition_on_node:
        ok = d.groupby("marker")["node_ode_r2"].transform("max") >= SUCCESS_THRESHOLD
        d = d[ok]
    d["is_control"] = d["is_control"].fillna(False).astype(bool)
    d["sr_ok"] = d["pysr_ode_r2"] >= SUCCESS_THRESHOLD
    g = d.groupby("marker")
    out = pd.DataFrame({
        "pr_mean": g["node_pr"].mean(),
        "pr_median": g["node_pr"].median(),
        "pr_max": g["node_pr"].max(),
        "pr_sd": g["node_pr"].std(),
        "n_seeds": g["node_pr"].size(),
        # Recovery frequency: fraction of seeds in which SR cleared threshold.
        # 0, 1/3, 2/3 or 1 — the continuous replacement for the binary label.
        "sr_recovery_freq": g["sr_ok"].mean(),
        "sr_r2_best": g["pysr_ode_r2"].max(),
        "sr_r2_median": g["pysr_ode_r2"].median(),
        "node_r2_best": g["node_ode_r2"].max(),
        "node_r2_median": g["node_ode_r2"].median(),
        "is_control": g["is_control"].first(),
    }).reset_index()
    out["variant"] = variant
    return out


def _labelled(ctx: pd.DataFrame, convention: str,
              threshold: float = SUCCESS_THRESHOLD) -> pd.DataFrame:
    """Attach the binary SR label under a given seed convention."""
    d = ctx.copy()
    if convention == "best":
        d["sr_ok"] = d["sr_r2_best"] >= threshold
        d["pr"] = d["pr_max"]          # matches best-vs-best as published
    elif convention == "majority":
        d["sr_ok"] = d["sr_recovery_freq"] >= 0.5
        d["pr"] = d["pr_median"]
    elif convention == "mean":
        d["sr_ok"] = d["sr_r2_median"] >= threshold
        d["pr"] = d["pr_mean"]
    else:
        raise ValueError(convention)
    return d


def permutation_diff(pr: np.ndarray, label: np.ndarray, n_perm: int,
                     rng: np.random.Generator) -> dict:
    """Two-sided permutation test on the difference of group means.

    Permuting `label` across markers is the right null here: it holds the PR
    values fixed (so their clustering and any correlation structure survives)
    and only destroys the association with SR outcome.
    """
    pr = np.asarray(pr, float)
    label = np.asarray(label, bool)
    n_fail, n_ok = int((~label).sum()), int(label.sum())
    if n_fail < 2 or n_ok < 2:
        return {"diff": np.nan, "p_perm": np.nan, "n_fail": n_fail, "n_ok": n_ok,
                "mean_fail": np.nan, "mean_ok": np.nan, "cohen_d": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan}

    mean_fail, mean_ok = pr[~label].mean(), pr[label].mean()
    observed = mean_fail - mean_ok

    perm = np.empty(n_perm)
    idx = np.arange(len(pr))
    for i in range(n_perm):
        p = rng.permutation(idx)
        lab = label[p]
        perm[i] = pr[~lab].mean() - pr[lab].mean()
    # +1 correction: an observed statistic can never have p = 0 exactly.
    p_perm = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1)

    # Pooled-SD Cohen's d, and a bootstrap CI on the difference.
    s1, s2 = pr[~label].std(ddof=1), pr[label].std(ddof=1)
    pooled = np.sqrt(((n_fail - 1) * s1 ** 2 + (n_ok - 1) * s2 ** 2)
                     / max(n_fail + n_ok - 2, 1))
    d = observed / pooled if pooled > 0 else np.nan

    boot = np.empty(4000)
    a, b = pr[~label], pr[label]
    for i in range(boot.size):
        boot[i] = (rng.choice(a, a.size, replace=True).mean()
                   - rng.choice(b, b.size, replace=True).mean())
    return {"diff": float(observed), "p_perm": float(p_perm),
            "n_fail": n_fail, "n_ok": n_ok,
            "mean_fail": float(mean_fail), "mean_ok": float(mean_ok),
            "cohen_d": float(d),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5))}


def spearman_test(x: np.ndarray, y: np.ndarray, n_perm: int,
                  rng: np.random.Generator) -> dict:
    """Spearman rho with a permutation P — the continuous, threshold-free test."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 5:
        return {"rho": np.nan, "p_perm": np.nan, "n": int(x.size)}

    def _rho(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / denom) if denom > 0 else np.nan

    observed = _rho(x, y)
    perm = np.array([_rho(x, rng.permutation(y)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1)
    return {"rho": observed, "p_perm": float(p), "n": int(x.size)}


def mixed_model(master: pd.DataFrame, variant: str | None) -> dict:
    """PR ~ sr_ok with a marker random intercept, over all per-seed rows.

    Uses every (variant, seed, marker) observation rather than a per-marker
    summary, and lets the random intercept absorb the fact that the same marker
    contributes up to nine rows. Skipped (with a recorded reason) if statsmodels
    is unavailable, since it is the one test here with a hard dependency.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "note": f"statsmodels unavailable: {exc}"}

    d = master.dropna(subset=["node_pr", "pysr_ode_r2"]).copy()
    if variant is not None:
        d = d[d["variant"] == variant]
    d["sr_ok"] = (d["pysr_ode_r2"] >= SUCCESS_THRESHOLD).astype(float)
    if d["sr_ok"].nunique() < 2:
        return {"available": False, "note": "SR label is constant"}
    try:
        fit = smf.mixedlm("node_pr ~ sr_ok", d, groups=d["marker"]).fit(reml=True)
        return {"available": True,
                "coef_sr_ok": float(fit.params["sr_ok"]),
                "se": float(fit.bse["sr_ok"]),
                "p_value": float(fit.pvalues["sr_ok"]),
                "n_obs": int(len(d)), "n_groups": int(d["marker"].nunique())}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "note": f"fit failed: {exc}"}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path,
                    default=default_outdir() / "reducibility_master.csv")
    ap.add_argument("--output-dir", type=Path, default=default_outdir())
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturbations-only", action="store_true",
                    help="Exclude the 8 control contexts.")
    ap.add_argument("--condition-on-node-success", action="store_true",
                    help="Restrict to Neural-ODE-successful contexts — the "
                         "manuscript's implicit n=18 subset.")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    master = pd.read_csv(args.master)
    variants = sorted(master["variant"].dropna().unique())

    ctx_frames, rows = [], []
    for variant in variants:
        ctx = context_level(master, variant,
                            condition_on_node=args.condition_on_node_success)
        if args.perturbations_only:
            ctx = ctx[~ctx["is_control"]]
        ctx_frames.append(ctx)

        # --- primary, threshold-free: PR vs SR recovery frequency / R^2 -------
        for xname, x in (("sr_recovery_freq", ctx["sr_recovery_freq"]),
                         ("sr_r2_median", ctx["sr_r2_median"])):
            res = spearman_test(ctx["pr_mean"], x, args.n_perm, rng)
            rows.append({"variant": variant, "test": "spearman",
                         "detail": f"pr_mean vs {xname}", **res})

        # --- the dichotomised contrast, under each seed convention -----------
        for convention in ("best", "majority", "mean"):
            lab = _labelled(ctx, convention)
            res = permutation_diff(lab["pr"], lab["sr_ok"], args.n_perm, rng)
            rows.append({"variant": variant, "test": "permutation_diff",
                         "detail": f"convention={convention}", **res})

        # --- mixed model over all per-seed rows ------------------------------
        mm = mixed_model(master if not args.perturbations_only else
                         master[~master["is_control"].fillna(False).astype(bool)],
                         variant)
        rows.append({"variant": variant, "test": "mixedlm",
                     "detail": "node_pr ~ sr_ok + (1|marker)", **mm})

    stats = pd.DataFrame(rows)
    contexts = pd.concat(ctx_frames, ignore_index=True)
    write_csv(stats, args.output_dir / "pr_contrast_stats.csv")
    write_csv(contexts, args.output_dir / "pr_context_level.csv")

    # --- threshold sensitivity ----------------------------------------------
    sweep = []
    ref = context_level(master, REFERENCE_VARIANT,
                        condition_on_node=args.condition_on_node_success)
    if args.perturbations_only:
        ref = ref[~ref["is_control"]]
    for thr in np.round(np.arange(0.3, 0.86, 0.05), 2):
        lab = _labelled(ref, "best", threshold=float(thr))
        res = permutation_diff(lab["pr"], lab["sr_ok"], 4000, rng)
        sweep.append({"variant": REFERENCE_VARIANT, "threshold": float(thr), **res})
    write_csv(pd.DataFrame(sweep), args.output_dir / "pr_threshold_sweep.csv")

    # --- report --------------------------------------------------------------
    print("\n=== primary (threshold-free) : PR vs SR recovery ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(stats[stats.test == "spearman"][
            ["variant", "detail", "rho", "p_perm", "n"]].to_string(index=False))
        print("\n=== dichotomised contrast, by seed convention ===")
        print(stats[stats.test == "permutation_diff"][
            ["variant", "detail", "n_fail", "n_ok", "mean_fail", "mean_ok",
             "diff", "ci_lo", "ci_hi", "cohen_d", "p_perm"]]
            .round(3).to_string(index=False))
        print("\n=== mixed model (all per-seed rows) ===")
        cols = [c for c in ("variant", "coef_sr_ok", "se", "p_value", "n_obs",
                            "n_groups", "note") if c in stats.columns]
        print(stats[stats.test == "mixedlm"][cols].to_string(index=False))
        print(f"\n=== threshold sensitivity ({REFERENCE_VARIANT}, best-of-seed) ===")
        print(pd.DataFrame(sweep)[
            ["threshold", "n_fail", "n_ok", "diff", "cohen_d", "p_perm"]]
            .round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
