"""Headline numbers and Fig. 5 exemplars for the frozen-config PySR OOD run.

Reads the per-seed integration metrics of one `run_markers.py` output tree and produces:

  * `--output-metrics`   one row per (context, seed): held-out and training integrated R2.
  * `--output-summary`   the success-rate table quoted in the Results.
  * `--output-exemplars` the Fig. 5 exemplar ranking.

Three conventions matter here and they are easy to conflate; see the flags for how each
is pinned down.

1. **Integrated, not derivative.** `ode_integ_r2_median` is the R2 of the numerically
   integrated p-ERK trajectory. `dt_r2` is the fit to the derivative target. They
   correlate only ~0.5 and disagree on individual contexts -- PTPN7 has a derivative R2
   of exactly 0.000 and an integrated R2 of 0.935. Everything here is integrated.

2. **Best of three seeds, not seed-averaged.** Reported per context as the maximum over
   seeds 42/43/44.

3. **Controls are excluded from the success rate.** The eight control contexts
   (untransfected, FLAG-GFP) have no construct and therefore no GFP dose-response, so
   "extrapolating to an unseen dose" means predicting the same trajectory again. They
   score a median 0.85 and would lift the headline rate from 8/32 to 15/40 with no
   dose-response biology behind it. They are reported separately, never merged in.

The Fig. 5 exemplar rule is the dual-threshold one: a context counts as solved only if
the *same fit* clears the R2 threshold on both the training doses and the held-out
highest doses; solved contexts are then ranked by held-out R2. The "both" clause is what
excludes MAP2K2, whose held-out R2 is 0.747 on a training R2 of 0.029 -- a model that
never fitted the data it was trained on has not found the law, however well it happens
to score elsewhere. Note this ranks on held-out R2 and so is test-based selection; the
quantitative claim is the distribution in `--output-summary`, not the exemplars.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

HELDOUT = "ode_integ_r2_median"
TRAIN = "ode_integ_r2_median_train"

# Contexts with no overexpression construct.
CONTROL_RE = re.compile(r"untransfected|FLAG.?GFP", re.IGNORECASE)


def load_run(run_dir: Path) -> pd.DataFrame:
    """Collect per-seed integration metrics from a run_markers.py output tree."""
    paths = sorted(glob.glob(
        str(run_dir / "seeds" / "seed_*" / "metrics"
            / "marker_integration_metrics_per_minute.csv")
    ))
    if not paths:
        raise SystemExit(
            f"no seeds/seed_*/metrics/marker_integration_metrics_per_minute.csv under "
            f"{run_dir}. This stage needs integration metrics, which run_markers.py does "
            f"not produce -- they come from compute_marker_integration.py."
        )
    frames = []
    for path in paths:
        seed = int(re.search(r"seed_(\d+)", path).group(1))
        df = pd.read_csv(path)
        missing = {HELDOUT, TRAIN} - set(df.columns)
        if missing:
            raise SystemExit(f"{path} is missing {sorted(missing)}")
        df["seed"] = seed
        frames.append(df[["marker", "seed", HELDOUT, TRAIN, "formula"]])
    out = pd.concat(frames, ignore_index=True)
    out["is_control"] = out.marker.astype(str).str.contains(CONTROL_RE)
    return out.sort_values(["marker", "seed"]).reset_index(drop=True)


def summarise(fits: pd.DataFrame, threshold: float) -> pd.DataFrame:
    best = fits.sort_values(HELDOUT, ascending=False).drop_duplicates("marker")
    rows = []
    for label, group in (("perturbation", best[~best.is_control]),
                         ("control", best[best.is_control])):
        rows.append({
            "group": label,
            "n_contexts": len(group),
            "median_heldout_r2": group[HELDOUT].median(),
            "mean_heldout_r2": group[HELDOUT].mean(),
            f"n_ge_{threshold:g}": int((group[HELDOUT] >= threshold).sum()),
            "n_ge_0.8": int((group[HELDOUT] >= 0.8).sum()),
        })
    return pd.DataFrame(rows)


def exemplars(fits: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Dual-threshold rule: clear `threshold` on train AND held-out, rank by held-out."""
    perturbations = fits[~fits.is_control].copy()
    solved = perturbations[
        (perturbations[TRAIN] >= threshold) & (perturbations[HELDOUT] >= threshold)
    ].copy()
    # How often does each solved context clear both thresholds? A context solved on one
    # seed of three is a successful individual fit, not a reproducible result, and the
    # figure caption has to say so.
    seeds_solved = solved.groupby("marker").size().rename("seeds_solved")
    total_seeds = perturbations.groupby("marker").size().rename("seeds_run")
    solved = (solved.merge(seeds_solved, on="marker")
                    .merge(total_seeds, on="marker"))
    # One row per context -- its best seed. The panels illustrate distinct contexts, so a
    # context must not occupy two of them just because two of its seeds both succeeded.
    solved = (solved.sort_values(HELDOUT, ascending=False)
                    .drop_duplicates("marker")
                    .reset_index(drop=True))
    solved.insert(0, "rank", range(1, len(solved) + 1))
    return solved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="A run_markers.py output tree with seeds/seed_*/metrics/.")
    ap.add_argument("--output-metrics", required=True)
    ap.add_argument("--output-summary", required=True)
    ap.add_argument("--output-exemplars", required=True)
    ap.add_argument("--r2-threshold", type=float, default=0.6,
                    help="Satisfactory-model criterion (v5 uses 0.6).")
    ap.add_argument("--n-exemplars", type=int, default=3,
                    help="How many exemplars to print (all solved fits are written).")
    args = ap.parse_args(argv)

    fits = load_run(Path(args.run_dir))
    fits.round(6).to_csv(args.output_metrics, index=False)

    summary = summarise(fits, args.r2_threshold)
    summary.round(4).to_csv(args.output_summary, index=False)

    picked = exemplars(fits, args.r2_threshold)
    picked.round(6).to_csv(args.output_exemplars, index=False)

    n_contexts = fits.marker.nunique()
    print(f"=== frozen-config PySR, top-GFP OOD split ===")
    print(f"  {n_contexts} contexts x {fits.seed.nunique()} seeds = {len(fits)} fits")
    print(f"  metric: {HELDOUT} (integrated trajectory), best of seeds\n")
    print(summary.round(3).to_string(index=False))

    print(f"\n=== contexts clearing R2 >= {args.r2_threshold:g} on BOTH train and held-out ===")
    if picked.empty:
        print("  none")
    else:
        cols = ["rank", "marker", "seed", TRAIN, HELDOUT, "seeds_solved", "seeds_run"]
        print(picked[cols].head(max(args.n_exemplars, 10)).round(3).to_string(index=False))
        reproducible = picked[picked.seeds_solved == picked.seeds_run].marker.nunique()
        print(f"\n  {picked.marker.nunique()} distinct contexts solved on at least one seed; "
              f"{reproducible} on every seed.")
        for _, row in picked.head(args.n_exemplars).iterrows():
            print(f"\n  Fig 5 panel {int(row['rank'])}: {row.marker} seed {int(row.seed)} "
                  f"(train {row[TRAIN]:.3f}, held-out {row[HELDOUT]:.3f}, "
                  f"{int(row.seeds_solved)}/{int(row.seeds_run)} seeds)")
            print(f"    {row.formula}")

    # The trap the "both" clause exists to catch, reported so it stays visible.
    perturbations = fits[~fits.is_control]
    by_heldout = perturbations.sort_values(HELDOUT, ascending=False)
    failed_train = by_heldout[by_heldout[TRAIN] < args.r2_threshold].head(1)
    if not failed_train.empty:
        row = failed_train.iloc[0]
        print(f"\n  note: highest held-out R2 among contexts the 'both' clause rejects is "
              f"{row.marker} seed {int(row.seed)} "
              f"(held-out {row[HELDOUT]:.3f} on train {row[TRAIN]:.3f}).")

    print(f"\n  metrics   -> {args.output_metrics}")
    print(f"  summary   -> {args.output_summary}")
    print(f"  exemplars -> {args.output_exemplars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
