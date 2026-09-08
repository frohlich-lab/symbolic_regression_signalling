"""Build the Tables S8-S11 CSVs from the frozen run artefacts.

S8  Sparse Neural ODE regulariser comparison - per-context held-out R^2 and
    effective first-order dependency count (participation ratio of the input
    Jacobian over the ten model inputs) for the L1, L21 and PathReg variants.
S9  Sparse Neural ODE architecture sweep - marginal effect of hidden width,
    depth, learning rate and activation on in-distribution validation R^2.
S10 PySR configuration sweep - marginal effect of each searched hyperparameter
    on training and held-out R^2.
S11 Phospho-readout key - the HGNC symbol, phospho-site and antibody clone
    behind each protein name used in the figures. Overexpression contexts are
    named by HGNC symbol throughout; readouts are antibodies, four of them
    pan-isoform, so they keep protein common names and this table carries the
    mapping. See display_names.py for the convention.

S9 and S10 are one row per hyperparameter *level*, not per configuration: the
reporting standards ask for the search space, the selection rule and what the
sweep showed, not a per-configuration dump (Heil et al., Nat Methods 2021;
REFORMS item 5e), and Cell Press asks that bulk grids be deposited rather than
typeset. The full 54- and 72-row grids are still written out, as
neural_ode_arch_grid_full.csv and pysr_config_sweep_full.csv, for the repo.

Participation ratios come from participation_ratio_variants.csv, written by
compute_participation_ratios.py.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_names import RSK_CROSSREACTIVITY, gene_labels, readout_table  # noqa: E402

VARIANTS = [("l1", "L1"), ("l21_lam3", "L21"), ("pathreg", "PathReg")]
RETAINED_ARCH = dict(hidden_dim=64, hidden_layers=4, lr=0.003, activation="tanh")


def heldout(root: Path, variant: str) -> pd.Series:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(
        str(root / variant / "seed_*" / "neural_ode_diffrax_metrics_agg.csv")))]
    d = pd.concat(frames)
    return d[d.split == "test"].groupby("marker")["ode_integ_r2_median"].max()


def regulariser_comparison(root: Path, out: Path) -> pd.DataFrame:
    pr = pd.read_csv(root / "participation_ratio_variants.csv")
    cols: dict[str, pd.Series] = {}
    for key, label in VARIANTS:
        cols[f"{label} held-out R2"] = heldout(root, key).round(3)
        cols[f"{label} dependencies (PR)"] = (
            pr[pr.variant == key].groupby("marker").pr.mean().round(2))
    df = pd.DataFrame(cols)
    # Contexts are transfected genes, so they are typeset on their HGNC symbol
    # (PIP5K3 -> PIKFYVE); the run artefacts keep the source-data key.
    df.index = gene_labels(df.index).values
    df.index.name = "Overexpression context"
    df = df.sort_values(df.columns[2], ascending=False)

    summary = pd.DataFrame(
        [df.median().round(3), df.mean().round(3)],
        index=pd.Index(["Median across contexts", "Mean across contexts"],
                       name=df.index.name))
    out_df = pd.concat([df, summary])
    out_df.to_csv(out)
    print(f"wrote {out}  ({len(df)} contexts)")
    return out_df


def _marginals(d: pd.DataFrame, factors: list[tuple[str, str]],
               metrics: list[tuple[str, str]], count_label: str) -> pd.DataFrame:
    """One row per (hyperparameter, level): what varying it alone buys.

    The full per-configuration grids stay in the repository CSVs; published
    supplementary tables report the search space and what it showed, which is
    what the reporting standards ask for (Heil et al. 2021; REFORMS item 5e).
    """
    def fmt(v):
        # Levels span ints (700, 30) and floats (0.1, 0.003) in one column, so
        # pandas would upcast the lot to float and print "700.0". Format here.
        if isinstance(v, str):
            return v
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:g}"

    rows = []
    for col, label in factors:
        for level, g in d.groupby(col, sort=True):
            row = {"Hyperparameter": label, "Value": fmt(level),
                   count_label: len(g)}
            for src_col, out_col in metrics:
                med = float(g[src_col].median())
                # R2 columns are typeset to a fixed 3 dp so the column reads as
                # a column; counts stay bare integers.
                row[out_col] = (f"{med:.3f}" if "R2" in out_col
                                else fmt(med))
            rows.append(row)
    return pd.DataFrame(rows)


def neural_ode_arch_marginals(root: Path, out: Path, full_out: Path) -> pd.DataFrame:
    d = pd.read_csv(root / "arch_grid.csv")
    # Full grid, deposited with the code rather than typeset.
    d.sort_values("mean_val_r2", ascending=False).to_csv(full_out, index=False)

    compact = _marginals(
        d,
        factors=[("hidden_dim", "Hidden width"),
                 ("hidden_layers", "Hidden layers"),
                 ("lr", "Learning rate"),
                 ("activation", "Activation")],
        metrics=[("median_val_r2", "Median validation R2"),
                 ("mean_val_r2", "Mean validation R2")],
        count_label="Configurations")
    compact["Retained"] = ""
    for col, key in (("Hidden width", "hidden_dim"),
                     ("Hidden layers", "hidden_layers"),
                     ("Learning rate", "lr"), ("Activation", "activation")):
        want = RETAINED_ARCH[key]
        want = want if isinstance(want, str) else (
            str(int(want)) if float(want) == int(want) else f"{float(want):g}")
        m = (compact.Hyperparameter == col) & (compact.Value == want)
        compact.loc[m, "Retained"] = "yes"
    compact.to_csv(out, index=False)
    print(f"wrote {out}  ({len(compact)} levels; full grid of {len(d)} "
          f"configurations -> {full_out.name})")
    return compact


def pysr_config_marginals(src: Path, out: Path, full_out: Path) -> pd.DataFrame:
    d = pd.read_csv(src)
    d.sort_values("train_rank").to_csv(full_out, index=False)

    compact = _marginals(
        d,
        factors=[("max_iterations", "Max iterations"),
                 ("populations", "Populations"),
                 ("population_size", "Population size"),
                 ("max_size", "Max complexity"),
                 ("parsimony", "Parsimony")],
        metrics=[("median_train_r2", "Median training R2"),
                 ("median_heldout_r2", "Median held-out R2"),
                 ("n_heldout_ge_06", "Contexts >= 0.6")],
        count_label="Arms")
    sel = d[d.train_rank == 1].iloc[0]
    compact["Selected"] = ""
    for col, key in (("Max iterations", "max_iterations"),
                     ("Populations", "populations"),
                     ("Population size", "population_size"),
                     ("Max complexity", "max_size"),
                     ("Parsimony", "parsimony")):
        want = float(sel[key])
        want = str(int(want)) if want == int(want) else f"{want:g}"
        m = (compact.Hyperparameter == col) & (compact.Value == want)
        compact.loc[m, "Selected"] = "yes"
    compact.to_csv(out, index=False)
    print(f"wrote {out}  ({len(compact)} levels; full grid of {len(d)} "
          f"configurations -> {full_out.name})")
    return compact


def readout_key(out: Path) -> pd.DataFrame:
    """The readout-to-gene key. Static, but written by the rule like the rest so
    the manuscript never carries a hand-maintained table."""
    df = readout_table()
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} readouts)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nde-root", type=Path,
                    default=Path("data/experimental/runs/sparse_neural_ode"))
    ap.add_argument("--table-s10-src", type=Path,
                    default=Path("data/experimental/runs/pysr_config_sweep/"
                                 "metrics/table_s10_config_sweep.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/experimental/runs/paper_figures/"
                                 "supplementary"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    # S9 is the PySR sweep and S10 the neural-ODE search: the order the main text cites
    # them in. These two were numbered the other way round until 2026-08-12, so a file
    # named table_s9_neural_ode_* is from a stale run, not an alternative convention.
    s9 = pysr_config_marginals(args.table_s10_src,
                               args.out_dir / "table_s9_pysr_config_marginals.csv",
                               args.out_dir / "pysr_config_sweep_full.csv")
    s10 = neural_ode_arch_marginals(args.nde_root,
                                    args.out_dir / "table_s10_neural_ode_arch_marginals.csv",
                                    args.out_dir / "neural_ode_arch_grid_full.csv")

    # Dropped from the manuscript but still computed, into archive/ so a full run cannot
    # put an uncited table back beside the cited ones. The regulariser comparison went
    # when the draft kept only the L21 calibration sweep; the readout key was never cited.
    reg = regulariser_comparison(args.nde_root,
                                 archive / "table_regulariser_comparison.csv")
    key = readout_key(archive / "table_readout_key.csv")

    print("\nTable S9 (PySR configuration marginals):")
    print(s9.to_string(index=False))
    print("\nTable S10 (neural-ODE architecture marginals):")
    print(s10.to_string(index=False))
    print(f"\narchived, not cited: {reg.shape[0]} regulariser rows, "
          f"{key.shape[0]} readout-key rows")
    print(f"\nFootnote for Table S11: {RSK_CROSSREACTIVITY}")


if __name__ == "__main__":
    main()
