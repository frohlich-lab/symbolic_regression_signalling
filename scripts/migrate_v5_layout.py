#!/usr/bin/env python3
"""One-shot migration to the canonical v5 experimental data layout.

Draft SR-MM v5 reports a single frozen PySR configuration ("Config A"), chosen by a
72-arm grid sweep on a six-context development set, applied unchanged to all 40
overexpression contexts under the top-GFP out-of-distribution split. Before this
migration the artefacts behind those numbers were scattered across date-stamped run
directories, a `figures/FINAL/` subdirectory and -- for the raw per-fit outputs -- a
session scratchpad under /private/tmp that was not part of the repository at all.

This script moves every artefact v5 depends on into `data/experimental/runs/<stage>/`
under names that say what the stage is, and moves everything v5 does not use into
`archive/superseded/`. It is idempotent: a source that has already been moved is
skipped, so it can be re-run after a partial failure.

    python scripts/migrate_v5_layout.py --dry-run     # print the plan, touch nothing
    python scripts/migrate_v5_layout.py               # execute

The mapping from v5 figure/table to destination is documented in
`docs/experimental_provenance.md`.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "experimental" / "runs"
ARCHIVE = REPO / "archive" / "superseded"

# Raw Config A per-fit outputs never made it into the repo; they live in the
# scratchpad of the session that produced them (2026-08-04/05). /private/tmp is
# reaped, so rescuing these is the whole reason this script exists.
SCRATCH = Path(
    "/private/tmp/claude-502/-Users-pomeret-Desktop-symbolic-regression-signalling"
    "/96e01705-2051-4b4b-a64d-e2d64367bbbf/scratchpad"
)

SEEDS = (42, 43, 44)

# (source, destination) -- sources may be absent if already migrated.
MOVES: list[tuple[Path, Path]] = [
    # Table S10: the 72-arm config sweep (stored as the full 288-arm grid it was a
    # subset of; the 72 arms are those with unary=none, binary=div).
    (RUNS / "pysr_grid_20260804", RUNS / "pysr_config_sweep"),
    # Fig 4C/4D, Fig 5, Fig S2: Config A on all 40 contexts.
    (RUNS / "final_par08_20260804", RUNS / "pysr_ood_final"),
    # Fig 4D, Fig S4: the OOD linear-regression baseline. NOT runs/select_k, which is
    # an in-distribution (random_bins) run and reverses the conclusion.
    (RUNS / "select_k_top_gfp_ood", RUNS / "linreg_ood"),
    # Fig 4E + sparsity panel: L21 group-sparse Jacobian, lambda_jac = 3.
    (RUNS / "neural_ode_diffrax" / "_seeds_ood_l21j_only",
     RUNS / "sparse_neural_ode" / "l21_lam3"),
    # Fig S3A / Table S8: regulariser robustness check.
    (RUNS / "neural_ode_diffrax" / "_seeds_ood_l1_archive",
     RUNS / "sparse_neural_ode" / "l1"),
    (RUNS / "neural_ode_diffrax" / "_seeds_ood_pathreg",
     RUNS / "sparse_neural_ode" / "pathreg"),
    # Fig S3B: the lambda_jac elbow sweep over {1,2,3,5,8,15,30}.
    (RUNS / "neural_ode_diffrax" / "_seeds_ood_l21j_tuned",
     RUNS / "sparse_neural_ode" / "lambda_sweep"),
    # Table S9 / Fig S3C: the 54-cell architecture grid.
    (RUNS / "paper_figures" / "hpo_summary.csv",
     RUNS / "sparse_neural_ode" / "arch_grid.csv"),
    # Parsimony variants: not reported, but they are the evidence that the train-rank
    # rule was applied rather than chosen for its outcome. Small; kept as provenance.
    (RUNS / "final_par01_20260804",
     RUNS / "pysr_ood_final" / "parsimony_variants" / "par0.1"),
    (RUNS / "final_config_20260804",
     RUNS / "pysr_ood_final" / "parsimony_variants" / "par3.0"),
]

ARCHIVE_MOVES: list[tuple[Path, str]] = [
    # v4-era PySR OOD run, superseded by pysr_ood_final (Config A).
    (RUNS / "top_gfp_ood_pysr", "v4 PySR OOD run; superseded by pysr_ood_final"),
    (RUNS / "top_gfp_ood_pysr_augmented", "v4 PySR OOD + augmented inputs; not in v5"),
    # In-distribution 80/20 PySR run. v5 reports OOD only (Methods: top 20% GFP held out).
    (RUNS / "seeds", "in-distribution PySR run; v5 is OOD-only"),
    (RUNS / "aggregated", "in-distribution PySR aggregation; v5 is OOD-only"),
    # v5 Methods: "the earlier full-input versus matched-input distinction is no longer
    # required and is not used."
    (RUNS / "neural_ode", "full/matched-input Neural ODE; explicitly superseded in v5"),
    # Random forest / decision tree never appear in v5.
    (RUNS / "random_forest", "RF baseline; not in v5"),
    (RUNS / "random_forest_dt", "RF/DT baseline; not in v5"),
    (RUNS / "random_forest_dt_top_gfp_ood", "RF/DT OOD baseline; not in v5"),
    # In-distribution linreg baseline. Using it as the OOD baseline reverses the result.
    (RUNS / "select_k", "in-distribution linreg; NOT an OOD baseline"),
]

# Everything left under neural_ode_diffrax/ after the four load-bearing variants move
# out is calibration scaffolding, partial runs and explicit archives.
EXP = REPO / "data" / "experimental"

ARCHIVE_GLOBS: list[tuple[str, str]] = [
    (str(RUNS / "neural_ode_diffrax"), "stale Neural ODE calibration/partial variants"),
    # Whole parallel run trees from before the OOD split existed. Each holds its own
    # aggregated/, seeds/ and select_k/, so none of them is a partial view of the
    # current run -- they are earlier configurations kept side by side.
    (str(EXP / "runs_800nit_25maxsize*"), "pre-OOD PySR config (800 iterations, maxsize 25)"),
    (str(EXP / "runs_no_gfp"), "pre-OOD PySR run with GFP dropped from the inputs"),
    (str(EXP / "runs_old_*"), "pre-OOD PySR run"),
    (str(EXP / "runs_random"), "pre-OOD PySR run, randomised control"),
    (str(EXP / "seeds"), "orphaned per-seed PySR outputs predating runs/"),
]

# Ad hoc figures inside pysr_ood_final that the FINAL README marks as wrong or paused.
PRUNE_IN_FINAL = [
    ("mindrivers_partial", "8/120 fits of a paused experiment; not in v5"),
    ("figures/nn_compressibility_vs_sr.png", "aggregates NN metrics without split=='test'"),
    ("figures/nn_compressibility_vs_sr.pdf", "aggregates NN metrics without split=='test'"),
    ("figures/fig_par08_accuracy_compressibility.png", "same split bug"),
    ("figures/fig_par08_accuracy_compressibility.pdf", "same split bug"),
    ("figures/fig_accuracy_and_compressibility.png", "same split bug"),
    ("figures/fig_accuracy_and_compressibility.pdf", "same split bug"),
    ("figures/nn_pr_vs_sr_r2.png", "compressibility line of work; not in v5"),
    ("figures/nn_pr_vs_sr_r2.pdf", "compressibility line of work; not in v5"),
]


class Runner:
    def __init__(self, dry: bool):
        self.dry = dry
        self.done: list[str] = []
        self.skipped: list[str] = []

    def _say(self, verb: str, msg: str) -> None:
        print(f"  {verb:8s} {msg}")

    def move(self, src: Path, dst: Path, why: str = "") -> None:
        rel_s = os.path.relpath(src, REPO)
        rel_d = os.path.relpath(dst, REPO)
        if not src.exists():
            self.skipped.append(f"{rel_s} (absent)")
            return
        if dst.exists():
            self.skipped.append(f"{rel_d} (destination exists)")
            return
        note = f" -- {why}" if why else ""
        self._say("MOVE", f"{rel_s}\n           -> {rel_d}{note}")
        if not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        self.done.append(rel_d)

    def copytree(self, src: Path, dst: Path, why: str = "") -> None:
        rel_s = str(src)
        rel_d = os.path.relpath(dst, REPO)
        if not src.exists():
            self.skipped.append(f"{rel_s} (absent -- scratchpad may already be reaped)")
            return
        if dst.exists():
            self.skipped.append(f"{rel_d} (destination exists)")
            return
        note = f" -- {why}" if why else ""
        self._say("RESCUE", f"{rel_s}\n           -> {rel_d}{note}")
        if not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(src), str(dst))
        self.done.append(rel_d)

    def write(self, path: Path, text: str) -> None:
        rel = os.path.relpath(path, REPO)
        self._say("WRITE", rel)
        if not self.dry:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.done.append(rel)


def consolidate_seed_tree(r: Runner, fits_root: Path, out_root: Path) -> None:
    """Build the per-seed layout the downstream plotting scripts expect.

    The Config A run was sharded one SLURM task per (marker, seed), so it wrote one
    formula report and one metrics CSV per marker. Every consumer
    (`plot_parsimony_tradeoff.py --pysr-dir`, `plot_sr_vs_linreg_ood.py`,
    `compute_marker_integration.py`) expects instead one combined report per seed, as
    the unsharded pipeline produces. Concatenate the shards into that shape so the
    frozen run is a drop-in for a normal `run_markers.py` output directory.
    """
    import pandas as pd

    for seed in SEEDS:
        blocks: list[str] = []
        for path in sorted(fits_root.glob(f"*/seed_{seed}/formulas/all_per_minute.txt")):
            blocks.append(path.read_text().rstrip("\n"))
        if not blocks:
            r.skipped.append(f"seed {seed}: no formula shards under {fits_root}")
            continue
        r.write(out_root / f"seed_{seed}" / "formulas" / "all_per_minute.txt",
                "\n\n".join(blocks) + "\n")

        for name, subdir in (
            ("marker_integration_metrics_per_minute.csv", "metrics"),
            ("predicted_trajectories_per_minute.csv", "trajectories"),
        ):
            frames = []
            for path in sorted(fits_root.glob(f"*/seed_{seed}/{subdir}/{name}")):
                try:
                    df = pd.read_csv(path)
                except Exception as exc:  # a truncated shard should not abort the merge
                    print(f"  WARN     unreadable {path}: {exc}")
                    continue
                if not df.empty:
                    frames.append(df)
            if not frames:
                r.skipped.append(f"seed {seed}: no {subdir}/{name} shards")
                continue
            merged = pd.concat(frames, ignore_index=True)
            dst = out_root / f"seed_{seed}" / subdir / name
            r._say("MERGE", f"{len(frames)} shards -> {os.path.relpath(dst, REPO)} "
                            f"({len(merged)} rows)")
            if not r.dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                merged.to_csv(dst, index=False)
            r.done.append(os.path.relpath(dst, REPO))


def rescue_config_a_fits(r: Runner) -> None:
    """Copy the sharded Config A outputs out of the session scratchpad."""
    src = SCRATCH / "par08" / "out_seed" / "trainbest_ms26_par08"
    fits = RUNS / "pysr_ood_final" / "fits"
    if not src.exists():
        r.skipped.append(f"{src} (absent -- Config A raw fits are GONE if not already rescued)")
        return
    if fits.exists():
        r.skipped.append(f"{os.path.relpath(fits, REPO)} (destination exists)")
        return
    # Flatten <MARKER>/s<seed>/seeds/seed_<seed>/ to <MARKER>/seed_<seed>/, dropping the
    # duplicated aggregated/ view (a single-seed "aggregate" is the seed itself).
    n = 0
    for marker_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for seed in SEEDS:
            inner = marker_dir / f"s{seed}" / "seeds" / f"seed_{seed}"
            if not inner.exists():
                continue
            dst = fits / marker_dir.name / f"seed_{seed}"
            if not r.dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(inner), str(dst))
            n += 1
    r._say("RESCUE", f"{n} (marker, seed) fits from scratchpad\n"
                     f"           -> {os.path.relpath(fits, REPO)}")
    r.done.append(os.path.relpath(fits, REPO))


def rescue_final_scripts_and_tables(r: Runner) -> None:
    """Pull the small derived tables produced during the v5 figure work."""
    pairs = [
        (SCRATCH / "full40_configA.csv",
         RUNS / "pysr_ood_final" / "metrics" / "integrated_r2_all40.csv"),
        (SCRATCH / "configA_dump_withctrl.csv",
         RUNS / "pysr_ood_final" / "metrics" / "panel_inputs_with_controls.csv"),
    ]
    for src, dst in pairs:
        if not src.exists():
            r.skipped.append(f"{src} (absent)")
            continue
        if dst.exists():
            r.skipped.append(f"{os.path.relpath(dst, REPO)} (exists)")
            continue
        r._say("RESCUE", f"{src.name} -> {os.path.relpath(dst, REPO)}")
        if not r.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
        r.done.append(os.path.relpath(dst, REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args()
    r = Runner(args.dry_run)

    if args.dry_run:
        print("=== DRY RUN -- nothing will be modified ===\n")

    # The renames must come first: the rescue writes *into* pysr_ood_final, and a
    # directory move refuses to overwrite an existing destination.
    print("--- 1. rename run directories to canonical stage names ---")
    for src, dst in MOVES:
        r.move(src, dst)

    print("\n--- 2. rescue Config A raw fits from the session scratchpad ---")
    rescue_config_a_fits(r)
    rescue_final_scripts_and_tables(r)

    print("\n--- 3. consolidate the sharded Config A run into a per-seed tree ---")
    fits = RUNS / "pysr_ood_final" / "fits"
    if fits.exists():
        consolidate_seed_tree(r, fits, RUNS / "pysr_ood_final" / "seeds")
    else:
        r.skipped.append("seed consolidation (no fits/ directory)")

    print("\n--- 4. archive superseded outputs ---")
    reasons: dict[str, str] = {}

    def archive(p: Path, why: str) -> None:
        # data/experimental/seeds and data/experimental/runs/seeds both reduce to the
        # basename "seeds"; qualify with the parent so neither clobbers the other.
        name = p.name if not (ARCHIVE / p.name).exists() else f"{p.parent.name}_{p.name}"
        r.move(p, ARCHIVE / name, why)
        reasons[name] = why

    for src, why in ARCHIVE_MOVES:
        archive(src, why)
    for pattern, why in ARCHIVE_GLOBS:
        for hit in sorted(glob.glob(pattern)):
            archive(Path(hit), why)

    print("\n--- 5. prune known-wrong figures out of pysr_ood_final ---")
    final_root = RUNS / "pysr_ood_final"
    for rel, why in PRUNE_IN_FINAL:
        src = final_root / rel
        r.move(src, ARCHIVE / "pysr_ood_final_rejected" / Path(rel).name, why)
        reasons[Path(rel).name] = why

    print("\n--- 6. write the archive manifest ---")
    if reasons:
        lines = [
            "# Superseded artefacts",
            "",
            "Moved out of `data/experimental/runs/` by `scripts/migrate_v5_layout.py`.",
            "Nothing here is used by Draft SR-MM v5. Kept because `data/` is gitignored,",
            "so a deletion would be unrecoverable -- delete this directory once you are",
            "confident none of it is needed.",
            "",
            "| artefact | why it is not in v5 |",
            "|---|---|",
        ]
        lines += [f"| `{k}` | {v} |" for k, v in sorted(reasons.items())]
        lines.append("")
        r.write(ARCHIVE / "README.md", "\n".join(lines))

    print(f"\n=== {len(r.done)} actions, {len(r.skipped)} skipped ===")
    if r.skipped:
        print("\nskipped:")
        for s in r.skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
