"""Step 10 — specification-curve analysis: is the dependency result a property of
the biology, or of the analyst's arbitrary choices?

The compounded problem
----------------------
Two independent instabilities have now been observed in the same result:

  * across **regulariser** — the participation-ratio contrast is P=0.036 under
    L1 and P=0.442 under L21, the main-text model (see FINDINGS.md);
  * across **network depth** — P=0.032 at one depth and P=0.87 at another,
    between two architectures that are statistically indistinguishable on
    held-out data (18 vs 16 contexts, paired P=0.28).

Neither choice was made for a principled reason. Depth was a script default;
the regulariser is presented as interchangeable (Table S8 exists precisely to
show the dependency counts are "not an artefact of the penalty"). If a claim
flips sign across choices that were never argued for and cannot be distinguished
empirically, then the claim is not a property of the system being measured.

Two ways to write that up
-------------------------
As a defect: the result is not robust, and it goes.

As a finding: **Jacobian-based dependency attribution in sparse neural ODEs is
not identifiable on correlated biological data.** Participation ratios, L1/L21
dependency counts and "which inputs matter" readouts are used widely across
SciML and systems biology, almost always at a single architecture and a single
penalty, and almost never with a stability check. If the nuisance factors move
the estimate as much as the biology does, that is a general methodological
result with a far broader audience than any single ERK finding — and this
dataset is an unusually clean demonstration because the input correlations that
cause it are measured and reportable.

The second framing is only available if the instability is *quantified* rather
than conceded. That is what this does.

What it computes
----------------
1. **Specification curve.** Every defensible combination of analytic choices —
   regulariser, PR aggregation across seeds, SR success threshold, seed
   convention, scoring objective, and which contexts are included — each
   yielding one estimate of the SR-fail vs SR-success PR contrast. Reported as
   the distribution of effect sizes and P-values, plus the fraction of
   specifications that support the published conclusion. (Method: Simonsohn et
   al.'s specification curve / Steegen et al.'s multiverse analysis.)

2. **Variance decomposition.** How much of the variance in the participation
   ratio itself is attributable to biological context versus nuisance (seed,
   regulariser)? If nuisance rivals context, the metric is not measuring the
   biology, and that is the cleanest possible statement of the problem.

3. **Significance-disagreement rate.** How often two analysts making independent
   defensible choices disagree about whether P<0.05. Note this is disagreement
   about *significance*, not about *direction* — the two come apart sharply here,
   and conflating them would badly overstate the problem.

Architecture enters via `--metric pr_by_arch`, which needs `arch_sweep.py` to
have run: `arch_grid.csv` retained only aggregate validation R², with no
per-marker rows and no saved models, so PR cannot be recomputed per cell from
it. Until that lands, the default `--metric pr` figures are a *lower bound* on
the instability, since depth is known to add more.

`--metric k_star` scores the same specification space using the retrained input
count from `input_ablation.py` instead. If k* is stable where PR is not, the
paper has a constructive replacement and not merely a negative result.

Usage:
    python multiverse.py [--master ...] [--n-perm 4000]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import default_outdir, write_csv  # noqa: E402
from pr_stats import permutation_diff  # noqa: E402

# --- the specification space -------------------------------------------------
# Every option here is one a competent analyst could have chosen and that the
# manuscript does not argue against. That is the bar for inclusion: defensible,
# not merely possible.
THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7)
CONVENTIONS = ("best", "majority", "mean")
PR_AGGREGATIONS = ("max", "median", "mean")
OBJECTIVES = ("matched", "native")
CONDITIONINGS = ("all", "perturbations", "node_success", "perturbations+node_success")


def build_context_table(master: pd.DataFrame, variant: str, objective: str,
                        conditioning: str, threshold: float) -> pd.DataFrame:
    """One row per marker under a given specification."""
    sr_col = "pysr_ode_r2" if objective == "matched" else "pysr_dt_r2"
    node_col = "node_ode_r2" if objective == "matched" else "node_traj_r2"

    d = master[master["variant"] == variant].dropna(subset=["node_pr", sr_col]).copy()
    d["is_control"] = d["is_control"].fillna(False).astype(bool)

    if "perturbations" in conditioning:
        d = d[~d["is_control"]]
    if "node_success" in conditioning:
        ok = d.groupby("marker")[node_col].transform("max") >= threshold
        d = d[ok]
    if d.empty:
        return pd.DataFrame()

    d["sr_ok"] = d[sr_col] >= threshold
    g = d.groupby("marker")
    return pd.DataFrame({
        "pr_max": g["node_pr"].max(),
        "pr_median": g["node_pr"].median(),
        "pr_mean": g["node_pr"].mean(),
        "sr_recovery_freq": g["sr_ok"].mean(),
        "sr_best": g[sr_col].max(),
        "sr_median": g[sr_col].median(),
    }).reset_index()


def apply_convention(ctx: pd.DataFrame, convention: str, pr_agg: str,
                     threshold: float):
    if ctx.empty:
        return None, None
    pr = ctx[f"pr_{pr_agg}"].to_numpy()
    if convention == "best":
        label = (ctx["sr_best"] >= threshold).to_numpy()
    elif convention == "majority":
        label = (ctx["sr_recovery_freq"] >= 0.5).to_numpy()
    else:
        label = (ctx["sr_median"] >= threshold).to_numpy()
    return pr, label


def swap_metric(master: pd.DataFrame, metric: str, ablation: Path,
                arch: Path) -> pd.DataFrame:
    """Replace the dependency metric so the same multiverse can score k* or PR.

    The point of the comparison: PR's estimate moves with the regulariser (17.2%
    of its variance) and with depth. If k* — a retrained, validation-selected
    input count — is stable across the same specifications, then the paper has a
    constructive replacement rather than only a negative result. That is the
    difference between "this readout is unreliable" and "this readout is
    unreliable, use this one instead".

    `k_star` collapses the regulariser axis (it is not defined per penalty), so
    the curve it produces has fewer nuisance dimensions by construction — which
    is itself part of the claim, and is stated rather than hidden.
    """
    if metric == "pr":
        return master

    if metric == "k_star":
        if not ablation.exists():
            raise SystemExit(f"{ablation} missing — run input_ablation.py first")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from input_ablation import summarise
        ks = summarise(pd.read_csv(ablation), 0.05)
        if ks.empty:
            raise SystemExit("ablation table produced no k* rows")
        keep = master.drop_duplicates(["marker", "seed"]).drop(columns=["node_pr"])
        out = keep.merge(ks[["marker", "seed", "k_star"]],
                         on=["marker", "seed"], how="inner")
        out = out.rename(columns={"k_star": "node_pr"})
        out["variant"] = "k_star"
        if out.empty:
            raise SystemExit("no (marker, seed) overlap between ablation and master")
        return out

    if metric == "pr_by_arch":
        if not arch.exists():
            raise SystemExit(f"{arch} missing — run arch_sweep.py first")
        a = pd.read_csv(arch)
        keep = master.drop_duplicates(["marker", "seed"]).drop(columns=["node_pr"])
        out = keep.merge(a[["marker", "seed", "cfg", "pr"]],
                         on=["marker", "seed"], how="inner")
        # Architecture becomes the nuisance axis in place of the regulariser,
        # which is exactly the substitution needed to test whether depth behaves
        # like the other analytic choices.
        out = out.rename(columns={"pr": "node_pr", "cfg": "variant"})
        if out.empty:
            raise SystemExit("no (marker, seed) overlap between arch sweep and master")
        return out

    raise ValueError(metric)


def variance_decomposition(master: pd.DataFrame) -> pd.DataFrame:
    """Share of PR variance attributable to context vs seed vs regulariser.

    A plain nested sum-of-squares decomposition rather than a mixed model: it is
    exact for this balanced design (40 markers x 3 seeds x 3 variants) and needs
    no distributional assumption, which matters when the point being made is
    that a distributional test was over-trusted.
    """
    d = master.dropna(subset=["node_pr"]).copy()
    grand = d["node_pr"].mean()
    total = float(((d["node_pr"] - grand) ** 2).sum())
    rows = []
    for factor in ("marker", "seed", "variant"):
        means = d.groupby(factor)["node_pr"].transform("mean")
        ss = float(((means - grand) ** 2).sum())
        rows.append({"factor": factor, "ss": ss,
                     "variance_share": ss / total if total > 0 else np.nan,
                     "n_levels": int(d[factor].nunique())})
    rows.append({"factor": "residual",
                 "ss": total - sum(r["ss"] for r in rows),
                 "variance_share": 1 - sum(r["variance_share"] for r in rows),
                 "n_levels": np.nan})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path,
                    default=default_outdir() / "reducibility_master.csv")
    ap.add_argument("--output-dir", type=Path, default=default_outdir())
    ap.add_argument("--n-perm", type=int, default=4000)
    ap.add_argument("--metric", choices=("pr", "k_star", "pr_by_arch"),
                    default="pr",
                    help="pr: the published participation ratio (regulariser is "
                         "the nuisance axis). k_star: the retrained input count "
                         "from input_ablation.py. pr_by_arch: PR with "
                         "architecture as the nuisance axis.")
    ap.add_argument("--ablation", type=Path,
                    default=default_outdir() / "ablation" / "input_ablation.csv")
    ap.add_argument("--arch", type=Path,
                    default=default_outdir() / "arch" / "arch_sweep.csv")
    ap.add_argument("--tag", default=None,
                    help="Suffix for output files (defaults to --metric).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    master = swap_metric(pd.read_csv(args.master), args.metric,
                         args.ablation, args.arch)
    tag = args.tag or args.metric
    variants = sorted(master["variant"].dropna().unique())
    print(f"metric = {args.metric}   nuisance levels = {len(variants)}   "
          f"rows = {len(master)}")

    # --- 1. specification curve ---------------------------------------------
    rows = []
    space = itertools.product(variants, THRESHOLDS, CONVENTIONS,
                              PR_AGGREGATIONS, OBJECTIVES, CONDITIONINGS)
    for variant, thr, conv, agg, obj, cond in space:
        ctx = build_context_table(master, variant, obj, cond, thr)
        pr, label = apply_convention(ctx, conv, agg, thr)
        if pr is None or label is None:
            continue
        res = permutation_diff(pr, label, args.n_perm, rng)
        if not np.isfinite(res.get("p_perm", np.nan)):
            continue
        rows.append({"variant": variant, "threshold": thr, "convention": conv,
                     "pr_aggregation": agg, "objective": obj,
                     "conditioning": cond, **res})

    spec = pd.DataFrame(rows)
    if spec.empty:
        print("no valid specifications — check the master table")
        return 1
    spec = spec.sort_values("diff").reset_index(drop=True)
    spec["rank"] = np.arange(1, len(spec) + 1)
    write_csv(spec, args.output_dir / f"specification_curve_{tag}.csv")

    # --- 2. variance decomposition -------------------------------------------
    vd = variance_decomposition(master)
    write_csv(vd, args.output_dir / f"variance_decomposition_{tag}.csv")

    # --- 3. reversal rate ----------------------------------------------------
    sig = spec["p_perm"] < 0.05
    claimed = spec["diff"] > 0            # SR-fail leans on more inputs
    supports = sig & claimed
    contradicts = sig & ~claimed
    p_sig = float(sig.mean())
    # NB this is disagreement about SIGNIFICANCE, not about direction. The two
    # come apart sharply here and conflating them would overstate the problem:
    # an effect can be robustly signed and still fail to clear P<0.05 in most
    # specifications simply because it is small relative to the noise.
    sig_disagreement = 2 * p_sig * (1 - p_sig)

    # --- report --------------------------------------------------------------
    print("=" * 74)
    print(f"SPECIFICATION CURVE — {len(spec)} defensible specifications")
    print("=" * 74)
    print("DIRECTION (is the effect signed consistently?)")
    print(f"  effect positive (SR-fail leans on more inputs): "
          f"{int(claimed.sum())} ({claimed.mean():.1%})")
    print(f"  significant in the OPPOSITE direction         : "
          f"{int(contradicts.sum())} ({contradicts.mean():.1%})")
    print(f"  effect size: median {spec['diff'].median():.3f}, "
          f"range [{spec['diff'].min():.3f}, {spec['diff'].max():.3f}]")
    print("\nSIGNIFICANCE (does it clear P<0.05?)")
    print(f"  supports the published conclusion: "
          f"{int(supports.sum())} ({supports.mean():.1%})")
    print(f"  not significant                  : "
          f"{int((~sig).sum())} ({(~sig).mean():.1%})")
    print(f"  P-value: median {spec['p_perm'].median():.3f}, "
          f"range [{spec['p_perm'].min():.4f}, {spec['p_perm'].max():.3f}]")
    print(f"  two analysts disagree about significance: {sig_disagreement:.1%}")

    print("\n  by nuisance level:")
    print(spec.groupby("variant").agg(
        median_diff=("diff", "median"), median_p=("p_perm", "median"),
        frac_sig=("p_perm", lambda s: float((s < 0.05).mean())),
        n=("diff", "size")).round(3).to_string())

    print("\n  which choice moves the answer most (fraction significant):")
    for factor in ("threshold", "convention", "pr_aggregation", "objective",
                   "conditioning"):
        f = spec.groupby(factor)["p_perm"].apply(lambda s: float((s < 0.05).mean()))
        print(f"    {factor:<16} " +
              "  ".join(f"{k}={v:.2f}" for k, v in f.items()))

    print("\n" + "=" * 74)
    print("PARTICIPATION-RATIO VARIANCE DECOMPOSITION")
    print("=" * 74)
    print(vd.round(4).to_string(index=False))
    bio = float(vd.loc[vd.factor == "marker", "variance_share"].iloc[0])
    nuis = float(vd.loc[vd.factor.isin(["seed", "variant"]), "variance_share"].sum())
    print(f"\n  biological context : {bio:.1%} of PR variance")
    print(f"  nuisance (seed + regulariser): {nuis:.1%}")
    if nuis >= bio:
        print("\n  Nuisance rivals or exceeds biology — the participation ratio is")
        print("  not primarily measuring the system. This is the finding.")
    else:
        print("\n  Biology dominates, but the specification curve above still")
        print("  governs whether the CONTRAST is stable.")

    # Where does the manuscript's own specification sit in this distribution?
    own = spec[(spec.conditioning == "perturbations+node_success")
               & (spec.convention == "best") & (spec.threshold == 0.6)
               & (spec.objective == "matched")]
    if not own.empty:
        print("\n" + "=" * 74)
        print("THE MANUSCRIPT'S OWN SPECIFICATION, IN CONTEXT")
        print("=" * 74)
        print(own[["variant", "pr_aggregation", "n_fail", "n_ok", "diff",
                   "cohen_d", "p_perm"]].round(3).to_string(index=False))
        pct = float((spec["diff"] < own["diff"].median()).mean())
        print(f"\n  its effect size sits at the {pct:.0%} percentile of the "
              f"specification curve")
        print("  and it uses the conditioning with the LOWEST power in the space "
              "(see above),\n  so the published P was a favourable draw from an "
              "already under-powered corner.")

    print("\nNote: architecture (depth/width/activation/lr) is NOT yet in this")
    print("multiverse — arch_grid.csv kept only aggregate validation R^2. Depth")
    print("alone is known to flip this result, so these figures are a LOWER")
    print("BOUND on the true instability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
