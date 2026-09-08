"""Table S8: the symbolic-regression result for each of the 40 overexpression contexts.

One row per context, reporting the seed the analysis retains and what that fit recovered.
Two things make this a script rather than a spreadsheet:

  * The retained seed follows the same rule as the Results text and Fig. 4E -- highest
    ODE-integrated R2 on the TRAINING bins. Written by hand, the table drifted from the
    figure the moment that rule changed, which is exactly what happened before 2026-08-12
    (the figure was then selecting on a parsimony score and disagreed on AKT3).
  * The neural-ODE columns must be the same numbers the figure plots, so they are read from
    the panel dump the figure itself writes rather than recomputed here.

"Variables used" counts distinct symbols in the recovered expression. It is deliberately
NOT the figure's dependency count, which thresholds a Jacobian and so can be smaller when
a variable enters with negligible sensitivity; the two differ and the caption says which
is which.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# One definition of complexity, not two. Fig. 4E's seed selection and this column have to
# agree, and they only stay agreed if they call the same function -- a re-implementation
# here drifted by up to 3 nodes because sympify reassociates before counting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_parsimony_tradeoff import _formula_complexity        # noqa: E402

# The ten model inputs, in the order the figures use them.
FEATURES = ["p_ERK1_2", "GFP", "p_ERK1_2_min", "p_MEK1_2", "p_MEK1_2_min", "p_RAF",
            "p_p90RSK", "p_MAPKAPK2", "p_PDK1", "p_MKK3_6"]

CONTROLS = {"untransfected1", "untransfected2", "untransfected3", "untransfected4",
            "FLAG-GFP1", "FLAG-GFP2", "FLAG-GFP3", "FLAG-GFP4"}


def n_variables(formula: str) -> int:
    """Distinct model inputs appearing in the expression."""
    if not isinstance(formula, str):
        return 0
    return sum(bool(re.search(rf"\b{re.escape(f)}\b", formula)) for f in FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pysr-integ-csv", type=Path, required=True,
                    help="integrated_r2_per_fit.csv: one row per (marker, seed).")
    ap.add_argument("--panel-inputs", type=Path, required=True,
                    help="Fig. 4 panel dump, for the neural-ODE columns.")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    integ = pd.read_csv(args.pysr_integ_csv)
    need = {"marker", "seed", "ode_integ_r2_median", "ode_integ_r2_median_train", "formula"}
    if not need.issubset(integ.columns):
        raise SystemExit(f"{args.pysr_integ_csv} needs {sorted(need)}")

    # The retained seed, and separately how many of the three generalised -- the recovery
    # fraction is the outcome the graded test uses, and it involves no selection at all.
    kept = integ.loc[integ.groupby("marker").ode_integ_r2_median_train.idxmax()]
    recov = (integ.assign(ok=integ.ode_integ_r2_median >= args.threshold)
             .groupby("marker").ok.agg(["sum", "count"]))

    panel = pd.read_csv(args.panel_inputs).set_index("marker")

    rows = []
    for r in kept.itertuples():
        p = panel.loc[r.marker] if r.marker in panel.index else None
        rows.append({
            "Context": r.marker,
            "Control": "yes" if r.marker in CONTROLS else "no",
            "Seed": int(r.seed),
            "SR train R2": round(float(r.ode_integ_r2_median_train), 3),
            "SR held-out R2": round(float(r.ode_integ_r2_median), 3),
            "SR success": "yes" if r.ode_integ_r2_median >= args.threshold else "no",
            "Seed recovery": f"{int(recov.loc[r.marker, 'sum'])}/"
                             f"{int(recov.loc[r.marker, 'count'])}",
            "Variables used": n_variables(r.formula),
            "Complexity": _formula_complexity(r.formula),
            "NODE held-out R2": None if p is None else round(float(p.nn_r2), 3),
            "NODE dependencies": None if p is None else round(float(p.nn_pr), 2),
            "NODE interacting pairs": None if p is None else round(float(p.nn_inter), 2),
            "Recovered rate law": r.formula,
        })

    df = pd.DataFrame(rows)
    # Successes first, best held-out at the top: the reading order the text implies.
    df = df.sort_values(["SR success", "SR held-out R2"], ascending=[True, False])
    df = pd.concat([df[df["SR success"] == "yes"], df[df["SR success"] == "no"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    n_ok = int((df["SR success"] == "yes").sum())
    print(f"wrote {args.output}: {len(df)} contexts, {n_ok} met R2 >= {args.threshold}")
    print(f"  median variables used, successes: "
          f"{df.loc[df['SR success'] == 'yes', 'Variables used'].median():g}")
    missing = df["NODE held-out R2"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} context(s) absent from --panel-inputs")


if __name__ == "__main__":
    main()
