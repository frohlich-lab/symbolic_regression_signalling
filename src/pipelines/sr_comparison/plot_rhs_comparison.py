"""Direct (non-integrated) RHS comparison for the SR benchmark.

Each discovered formula predicts kcat_cg from the enzyme parameters. This module
evaluates that prediction directly on the held-out test set (no ODE integration)
and reports the log-MAE per method, writing a CSV and a bar-chart PNG.

Per-method scaling (kept consistent with how each method is trained):
  * output (discovery_scales): 'log' methods (aifeynman/kan/dso) predict
    log(kcat_cg) so kfw = exp(formula); 'linear' methods (pysr) predict kcat_cg
    directly so kfw = formula.
  * input: KAN is trained on the CSV's native LOG features, every other method on
    linear (exp'd) features -> feed KAN the raw columns and others exp(column).

Usage:
    python plot_rhs_comparison.py --dataset data_test.csv \
        --formulas f_kan.txt f_dso.txt ... --methods kan dso ... \
        --features P_u,k_off,...,kcat_cg --discovery-scales '{...}' \
        --output rhs_loss_comparison.csv --plot rhs_comparison.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import sympy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_COLUMN = "kcat_cg"
# Methods whose formula is expressed in terms of LOG features (see kan_model).
LOG_INPUT_METHODS = {"kan"}
# Pretty labels for the plot / CSV.
DISPLAY = {
    "aifeynman": "AI-Feynman",
    "kan": "KAN",
    "dso": "DSO",
    "pysr_sQSSA": "PySR (sQSSA)",
    "pysr_tQSSA": "PySR (tQSSA)",
    "pysindy": "PySINDy",
}


def _base_method(method: str) -> str:
    return method.split("_")[0]


def evaluate_method(formula_path, feature_cols, log_inputs, lin_inputs,
                    log_target, discovery_scale, use_log_inputs):
    """Return (mean_logmae, median_logmae, valid_fraction) or None."""
    if not (os.path.exists(formula_path) and os.path.getsize(formula_path) > 0):
        return None
    expr = sympy.sympify(open(formula_path).read().strip())
    kfw_expr = sympy.exp(expr) if discovery_scale == "log" else expr
    func = sympy.lambdify(feature_cols, kfw_expr, modules="numpy")
    cols = log_inputs if use_log_inputs else lin_inputs
    with np.errstate(all="ignore"):
        kfw = np.asarray(func(*[cols[c] for c in feature_cols]), dtype=float)
        err = np.abs(np.log(kfw) - log_target)
    finite = np.isfinite(err)
    if not finite.any():
        return (np.nan, np.nan, 0.0)
    return (float(np.mean(err[finite])), float(np.median(err[finite])),
            float(finite.mean()))


def main():
    parser = argparse.ArgumentParser(description="Direct RHS fit comparison for the SR benchmark.")
    parser.add_argument("--dataset", required=True, help="Test-set CSV (log-scale columns).")
    parser.add_argument("--formulas", nargs="+", required=True, help="Per-method formula files.")
    parser.add_argument("--methods", nargs="+", required=True, help="Method names (aligned with --formulas).")
    parser.add_argument("--features", required=True, help="Comma-separated features incl. target (last).")
    parser.add_argument("--discovery-scales", required=True, help="JSON of base-method -> 'log'|'linear'.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--plot", required=True, help="Output PNG path.")
    args = parser.parse_args()

    if len(args.methods) != len(args.formulas):
        raise ValueError("Mismatch between number of methods and formula files.")

    feature_list = [c.strip() for c in args.features.split(",")]
    # keep every non-target feature (target is the last entry by convention)
    feature_cols = [c for c in feature_list if c != TARGET_COLUMN]
    discovery = json.loads(args.discovery_scales)

    df = pd.read_csv(args.dataset)
    log_inputs = {c: df[c].to_numpy(dtype=float) for c in feature_cols}
    lin_inputs = {c: np.exp(df[c].to_numpy(dtype=float)) for c in feature_cols}
    log_target = df[TARGET_COLUMN].to_numpy(dtype=float)

    rows = []
    for method, formula_path in zip(args.methods, args.formulas):
        base = _base_method(method)
        scale = discovery.get(base, "linear")
        use_log = base in LOG_INPUT_METHODS
        res = evaluate_method(formula_path, feature_cols, log_inputs, lin_inputs,
                              log_target, scale, use_log)
        display = DISPLAY.get(method, method)
        if res is None:
            print(f"{display}: no formula found ({formula_path}); skipping.")
            continue
        mean_mae, median_mae, valid = res
        rows.append({"method": display, "log_mae_mean": mean_mae,
                     "log_mae_median": median_mae, "valid_fraction": valid})
        print(f"{display:16s} mean={mean_mae:.4f} median={median_mae:.4f} valid={100*valid:.1f}%")

    results = pd.DataFrame(rows).sort_values("log_mae_median").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")

    # Bar chart of median log-MAE (log y so the huge spread is visible).
    os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotdf = results[np.isfinite(results["log_mae_median"])]
    bars = ax.bar(plotdf["method"], plotdf["log_mae_median"], color="#4C72B0")
    ax.set_yscale("log")
    ax.set_ylabel("log-MAE of predicted kcat_cg (median, lower = better)")
    ax.set_title("Direct RHS fit — symbolic regression method comparison")
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, plotdf["log_mae_median"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
