"""Step 11 — put architecture into the multiverse.

`arch_grid.csv` retained only aggregate validation R^2 per cell: no per-marker
rows, no per-seed rows, no saved models. So the one factor known to flip the
dependency result outright — network depth, P=0.032 at one depth and P=0.87 at
another, between architectures that are statistically indistinguishable on
held-out data — cannot be added to `multiverse.py` from what is on disk.

This re-runs the grid cells and records, per (cell, marker, seed): the
participation ratio, held-out and validation R^2, and the input Jacobian. That
turns depth/width/activation/learning-rate into four more axes of the
specification curve, alongside the regulariser, threshold, convention,
aggregation, objective and conditioning axes already covered.

Why this is the decisive run
----------------------------
The analytic multiverse (1,008 specifications, no architecture) already shows
the effect is positive in 97.3% of specifications but significant in only 20.6%.
Adding architecture tests whether depth behaves like the other nuisance factors
— shifting significance while leaving direction intact — or whether it can
genuinely reverse the sign. Those are very different papers:

  * depth shifts significance only -> the claim is "consistently signed, and
    single-specification P-values in this literature are not trustworthy",
    which is defensible and generalises;
  * depth reverses the sign -> the dependency attribution is not identifiable at
    all on these data, which is a stronger negative result and a different
    paper again.

Either way it is answerable, and it is the question a referee re-running the
analysis at a different depth would answer for you.

The default cell list is the subset of the published grid that is defensible on
validation R^2 — there is no point multiversing over architectures nobody would
have chosen. `--all-cells` widens it to the full grid.

Usage:
    python arch_sweep.py --output-dir <dir> --markers <marker> --seeds 42
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
    NDE_ROOT,
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

COLUMNS = ["cfg", "hidden_dim", "hidden_layers", "activation", "lr",
           "marker", "seed", "pr", "best_val_loss", "epochs_run",
           "train_r2", "val_r2", "test_r2",
           "test_dt_r2", "top_input", "wall_seconds"]

#: Depth is the axis of interest; the published default is 3 and the grid's
#: validation winner is 4. Both are always included.
CORE_DEPTHS = (3, 4)


def core_depth_cells(grid: pd.DataFrame) -> pd.DataFrame:
    """The published cell and its depth-4 twin, at matched width/activation/lr.

    This is the minimal run that answers the question that motivated the sweep:
    two architectures that differ only in depth, are indistinguishable on
    held-out data, and give opposite answers about the dependency contrast.
    """
    return grid[(grid.hidden_layers.isin(CORE_DEPTHS))
                & (grid.hidden_dim == PAPER_CFG["hidden_dim"])
                & (grid.activation == PAPER_CFG["activation"])
                & (np.isclose(grid.lr, PAPER_CFG["lr"]))].copy()


def candidate_cells(all_cells: bool, tolerance: float) -> pd.DataFrame:
    """Grid cells worth multiversing over.

    Defensible = within `tolerance` of the best mean validation R^2 in the
    published grid. Architectures nobody would have selected tell us nothing
    about the robustness of a choice a competent analyst might have made.
    """
    path = NDE_ROOT / "arch_grid.csv"
    if not path.exists():
        raise SystemExit(f"{path} missing — run the architecture grid rule first")
    grid = pd.read_csv(path)
    if all_cells:
        return grid
    best = grid["mean_val_r2"].max()
    keep = grid[grid["mean_val_r2"] >= best - tolerance].copy()
    # Always carry both depths of interest at the published width/activation,
    # so the 3-vs-4 comparison is present even if one falls outside tolerance.
    core = grid[(grid.hidden_layers.isin(CORE_DEPTHS))
                & (grid.hidden_dim == PAPER_CFG["hidden_dim"])
                & (grid.activation == PAPER_CFG["activation"])
                & (np.isclose(grid.lr, PAPER_CFG["lr"]))]
    return pd.concat([keep, core]).drop_duplicates("cfg").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=default_outdir() / "arch")
    ap.add_argument("--markers", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(PAPER_CFG["seeds"]))
    ap.add_argument("--epochs", type=int, default=int(PAPER_CFG["epochs"]))
    ap.add_argument("--all-cells", action="store_true")
    ap.add_argument("--core-depths-only", action="store_true",
                    help="Just the published cell and its depth-4 twin — the "
                         "minimal run that answers the 3-vs-4 question.")
    ap.add_argument("--cfgs", nargs="*", default=None,
                    help="Explicit cfg names from arch_grid.csv.")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="Validation-R^2 slack defining a 'defensible' cell.")
    ap.add_argument("--resume", action="store_true")
    args_cli = ap.parse_args(argv)

    outdir = args_cli.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "arch_sweep.csv"

    cells = candidate_cells(args_cli.all_cells, args_cli.tolerance)
    if args_cli.core_depths_only:
        cells = core_depth_cells(pd.read_csv(NDE_ROOT / "arch_grid.csv"))
    if args_cli.cfgs:
        cells = cells[cells["cfg"].isin(args_cli.cfgs)]
    if cells.empty:
        raise SystemExit("no architecture cells selected")
    print(f"{len(cells)} architecture cells:")
    print(cells[["cfg", "hidden_dim", "hidden_layers", "activation", "lr",
                 "mean_val_r2"]].round(4).to_string(index=False))

    nodb = _baseline()
    base_args = make_baseline_args(outdir, epochs=args_cli.epochs)
    raw = load_raw(base_args)
    markers = args_cli.markers or all_markers(raw)

    done: set = set()
    if args_cli.resume and out_csv.exists() and out_csv.stat().st_size > 0:
        prev = pd.read_csv(out_csv)
        done = {(str(c), str(m), int(s))
                for c, m, s in zip(prev.cfg, prev.marker, prev.seed)}
        print(f"resuming: {len(done)} rows present")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    # Marker outermost, architecture innermost. The comparison of interest is
    # *paired within a context* (same marker, same seed, different depth), so
    # this ordering makes complete pairs accumulate from the first marker
    # onward. Cell-outermost would finish every marker at one depth before
    # starting the other, leaving the depth question unanswerable until the
    # whole sweep completed.
    cell_args = {
        str(cell.cfg): make_baseline_args(
            outdir, epochs=args_cli.epochs,
            hidden_dim=int(cell.hidden_dim),
            hidden_layers=int(cell.hidden_layers),
            activation=str(cell.activation), lr=float(cell.lr))
        for _, cell in cells.iterrows()
    }
    for marker in markers:
        for _, cell in cells.iterrows():
            args = cell_args[str(cell.cfg)]
            for seed in args_cli.seeds:
                if (str(cell.cfg), marker, seed) in done:
                    continue
                bundle = load_bundle(raw, marker, seed, args)
                if bundle is None:
                    continue
                bundle["marker"], bundle["seed"] = marker, seed
                t0 = time.time()
                rhs, info, _ = nodb._train_marker(marker, seed, args, bundle)
                m = {s: nodb._split_metrics(rhs, bundle, bundle[s], args)
                     for s in ("train", "val", "test")}
                X = input_matrix(bundle, "train")
                summary = jacobian_summary(rhs, X)
                names = input_names(bundle)
                top = names[int(np.argmax(summary["abs_jac"]))]

                row = {
                    "cfg": str(cell.cfg), "hidden_dim": int(cell.hidden_dim),
                    "hidden_layers": int(cell.hidden_layers),
                    "activation": str(cell.activation), "lr": float(cell.lr),
                    "marker": marker, "seed": seed, "pr": summary["pr"],
                    # The validation trajectory MSE is what early stopping used,
                    # so it — not a clipped median R^2 — is the criterion a
                    # config should be selected on.
                    "best_val_loss": float(info.get("best_val_loss", float("nan"))),
                    "epochs_run": int(info.get("epochs_run", 0)),
                    "train_r2": m["train"]["ode_integ_r2_median"],
                    "val_r2": m["val"]["ode_integ_r2_median"],
                    "test_r2": m["test"]["ode_integ_r2_median"],
                    "test_dt_r2": m["test"]["dt_r2"], "top_input": top,
                    "wall_seconds": round(time.time() - t0, 2),
                }
                pd.DataFrame([row])[COLUMNS].to_csv(
                    out_csv, mode="a", header=False, index=False)
                print(f"[{cell.cfg}] {marker} s{seed}: PR={summary['pr']:.2f} "
                      f"val={row['val_r2']:.3f} test={row['test_r2']:.3f} "
                      f"top={top}", flush=True)

    # --- how much does architecture move the attribution? --------------------
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return 0
    df = pd.read_csv(out_csv)
    if df.empty:
        return 0
    print("\n--- participation ratio by depth ---")
    print(df.groupby("hidden_layers").agg(
        pr_mean=("pr", "mean"), pr_sd=("pr", "std"),
        val_r2=("val_r2", "mean"), test_r2=("test_r2", "mean"),
        n=("pr", "size")).round(3).to_string())

    # Does the SAME context get the SAME top driver at different depths? If the
    # identity of the leading input is unstable, no mechanistic reading of the
    # Jacobian survives, regardless of what the participation ratio does.
    if df["hidden_layers"].nunique() > 1:
        agree = (df.groupby(["marker", "seed"])["top_input"]
                   .apply(lambda s: float(s.value_counts().iloc[0] / len(s))))
        print(f"\ntop-driver agreement across architectures: "
              f"mean {agree.mean():.1%}, median {agree.median():.1%}")
        print("(1.0 = every architecture names the same leading input for that "
              "context)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
