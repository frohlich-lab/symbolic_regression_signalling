"""Step 2 — the SR x Neural-ODE contingency table, under three seed conventions.

Why this exists
---------------
The manuscript reports 8 SR successes, 18 Neural-ODE successes and 6 overlapping
contexts. Those three numbers imply 2 contexts where SR succeeds but the far
more expressive network fails, which is never addressed and which undercuts any
reading in which the Neural ODE upper-bounds what is reducible.

The fix is not to explain the 2 away, it is to stop making a claim that needs
them explained. The defensible role for the Neural ODE is a *positive control
for learnability*, not a witness for irreducibility:

    SR fails AND the Neural ODE succeeds
        => deterministic structure IS present in the measured variables, so
           SR's failure is not "these data are noise" and not merely an
           optimiser artefact.  This is the informative quadrant.
    SR fails AND the Neural ODE fails
        => uninformative; either could be at fault.
    SR succeeds AND the Neural ODE fails
        => a Neural-ODE-side failure (capacity/optimisation/objective), NOT
           evidence about reducibility. Expected at non-zero rate, and this
           script quantifies it instead of leaving it implicit.
    both succeed
        => a compact law exists and was found.

Under that framing the whole 2x2 is reportable and the 2 discordant contexts
become a measured error rate of the diagnostic rather than a contradiction.

What it emits
-------------
`quadrant_table.csv`      the 2x2 under each seed convention x variant
`quadrant_contexts.csv`   which marker sits in which cell (so the discordant
                          contexts are named, not just counted)
`discordant_contexts.csv` the SR-succeeds / Neural-ODE-fails cases, with both
                          matched and native metrics, for the paragraph that
                          has to discuss them

Seed conventions
----------------
best      per-context maximum over seeds — the manuscript's convention. Kept so
          the published numbers are reproducible from this table.
per_seed  every (marker, seed) treated as its own observation. No selection.
majority  a context succeeds if it clears threshold in >= 2 of 3 seeds.

`--objective matched` scores both sides by integrated held-out R^2 (the same
quantity). `--objective native` uses each method's own training objective, and
exists to show how much of the discordance is an artefact of comparing
derivative fits to trajectory fits.

Usage:
    python quadrants.py [--variant l21_lam3] [--objective matched]
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

CONVENTIONS = ("best", "per_seed", "majority")


def _labels(df: pd.DataFrame, convention: str, objective: str) -> pd.DataFrame:
    """Reduce per-(marker, seed) rows to labelled observations."""
    if objective == "matched":
        sr_metric, node_metric = "pysr_ode_r2", "node_ode_r2"
    else:
        sr_metric, node_metric = "pysr_dt_r2", "node_traj_r2"

    d = df.dropna(subset=[sr_metric, node_metric]).copy()
    d["sr_ok"] = d[sr_metric] >= SUCCESS_THRESHOLD
    d["node_ok"] = d[node_metric] >= SUCCESS_THRESHOLD
    d["is_control"] = d["is_control"].fillna(False).astype(bool)

    if convention == "per_seed":
        out = d[["marker", "seed", "is_control", "sr_ok", "node_ok",
                 sr_metric, node_metric]].copy()
    elif convention == "best":
        out = (d.groupby("marker")
                .agg(is_control=("is_control", "first"),
                     sr_ok=("sr_ok", "any"), node_ok=("node_ok", "any"),
                     **{sr_metric: (sr_metric, "max"), node_metric: (node_metric, "max")})
                .reset_index())
    elif convention == "majority":
        out = (d.groupby("marker")
                .agg(is_control=("is_control", "first"),
                     sr_frac=("sr_ok", "mean"), node_frac=("node_ok", "mean"),
                     **{sr_metric: (sr_metric, "median"), node_metric: (node_metric, "median")})
                .reset_index())
        out["sr_ok"] = out["sr_frac"] >= 0.5
        out["node_ok"] = out["node_frac"] >= 0.5
    else:
        raise ValueError(convention)

    out = out.rename(columns={sr_metric: "sr_metric", node_metric: "node_metric"})
    out["quadrant"] = np.select(
        [out.sr_ok & out.node_ok, ~out.sr_ok & out.node_ok,
         out.sr_ok & ~out.node_ok],
        ["both_succeed", "sr_fails_node_succeeds",
         "sr_succeeds_node_fails"],
        default="both_fail")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path,
                    default=default_outdir() / "reducibility_master.csv")
    ap.add_argument("--output-dir", type=Path, default=default_outdir())
    ap.add_argument("--variant", default=REFERENCE_VARIANT,
                    help="'all' to tabulate every regulariser variant.")
    ap.add_argument("--objective", choices=("matched", "native", "both"),
                    default="both")
    args = ap.parse_args(argv)

    master = pd.read_csv(args.master)
    variants = (sorted(master["variant"].dropna().unique())
                if args.variant == "all" else [args.variant])
    objectives = (["matched", "native"] if args.objective == "both"
                  else [args.objective])

    table_rows, context_rows = [], []
    for variant in variants:
        sub = master[master["variant"] == variant]
        for objective in objectives:
            for convention in CONVENTIONS:
                lab = _labels(sub, convention, objective)
                if lab.empty:
                    continue
                lab2 = lab.assign(variant=variant, objective=objective,
                                  convention=convention)
                context_rows.append(lab2)
                # Stratify: the manuscript's headline counts are perturbations
                # only, and the control/perturbation gap is a result in itself
                # (SR clears threshold far more often where biology is least
                # interesting), so all three groups are tabulated.
                groups = {"all": lab,
                          "perturbation": lab[~lab["is_control"]],
                          "control": lab[lab["is_control"]]}
                for group, g in groups.items():
                    if g.empty:
                        continue
                    counts = g["quadrant"].value_counts().to_dict()
                    n = len(g)
                    table_rows.append({
                        "variant": variant, "objective": objective,
                        "convention": convention, "group": group,
                        "n_observations": n,
                        "both_succeed": counts.get("both_succeed", 0),
                        "sr_fails_node_succeeds": counts.get("sr_fails_node_succeeds", 0),
                        "sr_succeeds_node_fails": counts.get("sr_succeeds_node_fails", 0),
                        "both_fail": counts.get("both_fail", 0),
                        "n_sr_success": int(g["sr_ok"].sum()),
                        "n_node_success": int(g["node_ok"].sum()),
                        "sr_success_rate": round(float(g["sr_ok"].mean()), 4),
                        # The rate at which the "more expressive model bounds
                        # reducibility" reading is violated outright.
                        "node_false_negative_rate": round(
                            counts.get("sr_succeeds_node_fails", 0) / max(n, 1), 4),
                    })

    tables = pd.DataFrame(table_rows)
    contexts = pd.concat(context_rows, ignore_index=True)
    write_csv(tables, args.output_dir / "quadrant_table.csv")
    write_csv(contexts, args.output_dir / "quadrant_contexts.csv")

    discordant = contexts[contexts["quadrant"] == "sr_succeeds_node_fails"]
    write_csv(discordant.sort_values(["objective", "convention", "marker"]),
              args.output_dir / "discordant_contexts.csv")

    print("\n--- 2x2, reference variant ---")
    show = tables[(tables.variant == variants[0])]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(show.to_string(index=False))

    if not discordant.empty:
        print("\n--- SR succeeds / Neural ODE fails (must be discussed) ---")
        print(discordant.groupby(["objective", "convention"])["marker"]
              .apply(lambda s: ", ".join(sorted(set(s)))).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
