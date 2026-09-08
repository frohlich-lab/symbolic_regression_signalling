"""Step 7 — close the loop: predict the missing variable, restore it, recover the law.

The abstract promises that the diagnostic "guides experiment design ... motivating
new measurements where they do not [reduce]". Nothing in the manuscript ever
rescues a failing context, so the generative claim is asserted and never shown.
This is the single largest gap between the paper as written and a general-interest
result, and it is the one thing here that does not need new wet-lab data to
demonstrate at least in principle.

The design: hold-out-then-reintroduce
-------------------------------------
Take a context where SR *succeeds* with the full input set. Then:

  1. **Ablate.** Remove the input the Jacobian ranking says is load-bearing.
     Re-run SR. Predict: the compact law degrades or disappears — the context
     is now artificially "irreducible".
  2. **Diagnose.** Run the diagnostic on the crippled context exactly as it is
     run on genuinely-failing contexts: does the Neural ODE still fit (structure
     present), and does the participation ratio / k* rise?
  3. **Restore.** Put the input back. Re-run SR. Predict: the compact law
     returns.

Step 3 is a positive control for the whole diagnostic. If removing a variable
reliably produces the *same signature* the paper reads as evidence of genuine
irreducibility, and restoring it reliably reverses that signature, then the
signature means what the paper says it means — and the case for "measure this
variable next" becomes a demonstrated inference rather than an aspiration.

If the signature does *not* reverse, the diagnostic is not specific, and better
to learn that here than from a referee.

The honest limitation, which must be stated in any write-up: this demonstrates
the logic on a variable that was measured all along. It is an in-silico
hold-out, not a new measurement. A genuinely new measurement rescuing a
genuinely failing context is the stronger result and remains future work — but
this establishes that the inference is sound before anyone spends bench time on
it.

Ranking counterfactual
----------------------
`--control-input random` repeats the ablation with a randomly chosen *non*-top
input. The rescue signature should be much weaker there; if dropping any input
at random produces the same effect, the Jacobian ranking is not identifying
anything specific and the "measure THIS variable" claim has no support.

Usage:
    python rescue_demo.py --markers PTPN7 PIKFYVE --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

COLUMNS = ["marker", "seed", "condition", "dropped_input", "rank_of_dropped",
           "n_inputs", "node_test_r2", "node_train_r2", "pr", "k_star",
           "sr_test_r2", "sr_complexity", "sr_formula", "wall_seconds"]


def evaluate_node(bundle, args):
    nodb = _baseline()
    rhs, _h, _ = nodb._train_marker(bundle["marker"], bundle["seed"], args, bundle)
    m = {s: nodb._split_metrics(rhs, bundle, bundle[s], args)
         for s in ("train", "test")}
    summary = jacobian_summary(rhs, input_matrix(bundle, "train"))
    return rhs, m, summary


def k_star_for(bundle, args, order, epsilon):
    """Smallest input count within `epsilon` of the full held-out R^2.

    Shares the ablation logic of `input_ablation.py` but stays local so this
    script can be run standalone on one context.
    """
    nodb = _baseline()
    n_exo = len(bundle["exo_cols"])
    _rhs, m_full, _s = evaluate_node(bundle, args)
    full = m_full["test"]["ode_integ_r2_median"]
    for k in range(0, n_exo + 1):
        sub = subset_bundle(bundle, order[:k]) if k < n_exo else bundle
        sub["marker"], sub["seed"] = bundle["marker"], bundle["seed"]
        _r, m_k, _s2 = evaluate_node(sub, args)
        if m_k["test"]["ode_integ_r2_median"] >= full - epsilon:
            return k, full
    return n_exo, full


def run_sr(marker, seed, keep_exo, bundle, outdir, sr_cmd):
    """Optionally re-run PySR on a reduced input set.

    Kept behind an explicit `--sr-command` because PySR needs the Julia-backed
    environment, which is not the environment the Neural-ODE analyses run in.
    The command receives a JSON spec on stdin and must print a JSON object with
    `test_r2`, `complexity` and `formula`. Without it, the Neural-ODE half of
    the demonstration still runs and the SR columns stay empty.
    """
    if not sr_cmd:
        return {"test_r2": np.nan, "complexity": np.nan, "formula": ""}
    spec = {
        "marker": marker, "seed": seed,
        "keep_exogenous": [bundle["exo_cols"][i] for i in keep_exo],
        "state": "p_ERK1_2",
        "output_dir": str(outdir),
    }
    try:
        proc = subprocess.run(sr_cmd, shell=True, input=json.dumps(spec),
                              capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            print(f"    [sr] failed rc={proc.returncode}: {proc.stderr[-400:]}",
                  flush=True)
            return {"test_r2": np.nan, "complexity": np.nan, "formula": ""}
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"test_r2": float(out.get("test_r2", np.nan)),
                "complexity": float(out.get("complexity", np.nan)),
                "formula": str(out.get("formula", ""))}
    except Exception as exc:
        print(f"    [sr] error: {exc}", flush=True)
        return {"test_r2": np.nan, "complexity": np.nan, "formula": ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir() / "rescue")
    ap.add_argument("--markers", nargs="*", default=None,
                    help="Default: contexts where SR succeeded (read from "
                         "reducibility_master.csv).")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(PAPER_CFG["seeds"]))
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--control-input", choices=("none", "random"), default="random",
                    help="Also ablate a non-top-ranked input, as a specificity control.")
    ap.add_argument("--sr-command", default=None,
                    help="Shell command running PySR on a reduced input set "
                         "(JSON spec on stdin, JSON result on stdout).")
    ap.add_argument("--master", type=Path,
                    default=default_outdir() / "reducibility_master.csv")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "rescue_demo.csv"
    rng = np.random.default_rng(args_cli.rng_seed)

    args = make_baseline_args(outdir, epochs=args_cli.epochs)
    raw = load_raw(args)

    # Default target set: contexts where SR *succeeds*, since the demonstration
    # needs a law present to destroy and restore.
    markers = args_cli.markers
    if markers is None:
        if args_cli.master.exists():
            m = pd.read_csv(args_cli.master)
            ok = (m.groupby("marker")["pysr_ode_r2"].max() >= SUCCESS_THRESHOLD)
            markers = sorted(ok[ok].index.tolist())
            print(f"targets (SR-success contexts): {markers}")
        else:
            markers = all_markers(raw)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(a), int(b), str(c)) for a, b, c in
                zip(prev.marker, prev.seed, prev.condition)}
        print(f"resuming: {len(done)} rows present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    for marker in markers:
        for seed in args_cli.seeds:
            bundle = load_bundle(raw, marker, seed, args)
            if bundle is None:
                print(f"[skip] {marker} s{seed}", flush=True)
                continue
            bundle["marker"], bundle["seed"] = marker, seed
            names = input_names(bundle)
            n_exo = len(bundle["exo_cols"])

            # Rank once on the intact model — this is the "which variable does
            # the diagnostic point at" step.
            _rhs, _m, summary = evaluate_node(bundle, args)
            order = list(np.argsort(-np.asarray(summary["abs_jac"])[1:]))
            top = order[0]
            control = (int(rng.choice(order[max(1, len(order) // 2):]))
                       if args_cli.control_input == "random" and len(order) > 2
                       else None)

            conditions = [("intact", None), ("drop_top", top), ("restored", None)]
            if control is not None:
                conditions.insert(2, ("drop_control", control))

            for condition, drop in conditions:
                if (marker, seed, condition) in done:
                    continue
                t0 = time.time()
                keep = [i for i in range(n_exo) if i != drop] if drop is not None \
                    else list(range(n_exo))
                sub = subset_bundle(bundle, keep) if drop is not None else bundle
                sub["marker"], sub["seed"] = marker, seed

                _r, m_c, s_c = evaluate_node(sub, args)
                sub_order = list(np.argsort(-np.asarray(s_c["abs_jac"])[1:]))
                kstar, _full = k_star_for(sub, args, sub_order, args_cli.epsilon)
                sr = run_sr(marker, seed, keep, bundle, outdir, args_cli.sr_command)

                row = {
                    "marker": marker, "seed": seed, "condition": condition,
                    "dropped_input": names[1 + drop] if drop is not None else "",
                    "rank_of_dropped": (order.index(drop) if drop is not None else -1),
                    "n_inputs": len(keep),
                    "node_test_r2": m_c["test"]["ode_integ_r2_median"],
                    "node_train_r2": m_c["train"]["ode_integ_r2_median"],
                    "pr": s_c["pr"], "k_star": kstar,
                    "sr_test_r2": sr["test_r2"], "sr_complexity": sr["complexity"],
                    "sr_formula": sr["formula"],
                    "wall_seconds": round(time.time() - t0, 2),
                }
                pd.DataFrame([row])[COLUMNS].to_csv(
                    out_csv, mode="a", header=False, index=False)
                print(f"[{marker} s{seed}] {condition:<13} "
                      f"drop={row['dropped_input'] or '-':<14} "
                      f"NODE R2={row['node_test_r2']:.3f} PR={row['pr']:.2f} "
                      f"k*={kstar} SR R2={sr['test_r2']}", flush=True)

    # --- did the signature move, and reverse? --------------------------------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    if df.empty:
        return 0
    summary = (df.groupby("condition")
                 .agg(node_test_r2=("node_test_r2", "mean"),
                      pr=("pr", "mean"), k_star=("k_star", "mean"),
                      sr_test_r2=("sr_test_r2", "mean"), n=("marker", "size"))
                 .reindex(["intact", "drop_top", "drop_control", "restored"])
                 .dropna(how="all"))
    write_csv(summary.reset_index(), outdir / "rescue_summary.csv")
    print("\n--- rescue signature ---")
    print(summary.round(3).to_string())
    print("\nWanted: drop_top raises PR/k* and lowers SR R2; restored returns to "
          "intact; drop_control moves much less.\nIf drop_control matches "
          "drop_top, the Jacobian ranking is not specific and the "
          "'measure this variable next' claim has no support.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
