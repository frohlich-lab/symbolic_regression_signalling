"""Appendix figure: compare the three Neural ODE regularisers (L1, L21, C-NODE).

For each method's _seeds_ood directory (3 seeds), reports per-method:
  - best-of-seed mean and median OOD test R²
  - number of markers with R² > threshold
  - mean and median PR (effective non-GFP drivers) on those markers

Renders one figure with three boxplots: best-of-seed R² (left), PR on R²>θ
markers (right). Saves PNG + PDF.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_repo_src = Path(__file__).resolve().parents[4]
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pipelines.experimental.sr_pipeline.neural_ode_diffrax_causal import (  # noqa: E402
    _load_marker_checkpoint,
    _per_marker_causal,
    _safe,
)


def pr_of(v: np.ndarray) -> float:
    v = np.abs(np.asarray(v)).ravel()
    s, sq = v.sum(), (v ** 2).sum()
    return float(s * s / sq) if sq > 0 else 0.0


def collect_method(directory: Path, seeds: list[int],
                   excluded: set[str]) -> pd.DataFrame:
    r2: dict[str, dict[int, float]] = {}
    for csv_path in glob.glob(str(directory / "seed_*" / "neural_ode_diffrax_metrics.csv")):
        seed = int(re.search(r"seed_(\d+)", csv_path).group(1))
        df = pd.read_csv(csv_path)
        for _, row in df[df.split == "test"].iterrows():
            r2.setdefault(str(row.marker), {})[seed] = float(row.trajectory_r2_mean)
    rows = []
    for marker in sorted(set(r2) - excluded):
        seeds_present = r2[marker]
        if not seeds_present:
            continue
        best_seed, best_r2 = max(seeds_present.items(), key=lambda kv: kv[1])
        loaded = _load_marker_checkpoint(directory / f"seed_{best_seed}" / "models",
                                         _safe(marker))
        if loaded is None:
            continue
        meta, model, train = loaded
        causal = _per_marker_causal(meta, model, train)
        non_gfp_idx = [i for i, f in enumerate(meta["feature_cols"]) if f != "GFP"]
        rows.append(dict(marker=marker, best_r2=best_r2,
                         pr=pr_of(causal["mean_signed_jac"][non_gfp_idx])))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-dir", required=True)
    ap.add_argument("--l21-dir", required=True)
    ap.add_argument("--cnode-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--r2-threshold", type=float, default=0.6)
    ap.add_argument("--exclude-marker", action="append", default=["untransfected1"])
    args = ap.parse_args()

    excluded = set(args.exclude_marker)
    method_dirs = [
        ("L1 @ λ=1", Path(args.l1_dir)),
        ("L21 @ λ=3", Path(args.l21_dir)),
        ("C-NODE @ λ=0.01", Path(args.cnode_dir)),
    ]

    method_data = []
    summary_rows = []
    for label, directory in method_dirs:
        df = collect_method(directory, args.seeds, excluded)
        if df.empty:
            raise SystemExit(f"No data found for {label} at {directory}")
        method_data.append((label, df))
        working = df[df.best_r2 > args.r2_threshold]
        summary_rows.append(dict(
            method=label,
            n_markers=len(df),
            mean_r2=df.best_r2.mean(),
            median_r2=df.best_r2.median(),
            n_works=len(working),
            mean_pr=working.pr.mean() if len(working) else float("nan"),
            median_pr=working.pr.median() if len(working) else float("nan"),
        ))
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    mpl.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax_r2, ax_pr) = plt.subplots(1, 2, figsize=(11, 5),
                                       gridspec_kw={"width_ratios": [1, 1]})
    palette = ["#3b6ea0", "#2c5f9e", "#1d4068"]

    r2_data = [d.best_r2.values for _, d in method_data]
    bp = ax_r2.boxplot(
        r2_data, positions=range(len(method_data)), widths=0.55, patch_artist=True,
        showmeans=True,
        medianprops=dict(color="black", lw=1.6),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        boxprops=dict(lw=0.8, edgecolor="black"),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markerfacecolor="#999", markeredgecolor="none",
                        markersize=4, alpha=0.6),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for i, (_, d) in enumerate(method_data):
        jitter = i + rng.uniform(-0.12, 0.12, len(d))
        ax_r2.scatter(jitter, d.best_r2, s=20, c=palette[i], alpha=0.5, edgecolors="none")
    ax_r2.axhline(args.r2_threshold, color="#888", lw=0.7, ls="--")
    ax_r2.set_xticks(range(len(method_data)))
    ax_r2.set_xticklabels([m for m, _ in method_data], fontsize=10)
    ax_r2.set_ylabel("OOD test R² (best of 3 seeds)", fontsize=12)
    ax_r2.set_title("A.  Accuracy across regularisers", loc="left", fontsize=11.5)
    ax_r2.set_ylim(-0.2, 1.05)

    pr_data = [d[d.best_r2 > args.r2_threshold].pr.values for _, d in method_data]
    bp2 = ax_pr.boxplot(
        pr_data, positions=range(len(method_data)), widths=0.55, patch_artist=True,
        showmeans=True,
        medianprops=dict(color="black", lw=1.6),
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        boxprops=dict(lw=0.8, edgecolor="black"),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker="o", markerfacecolor="#999", markeredgecolor="none",
                        markersize=4, alpha=0.6),
    )
    for patch, color in zip(bp2["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, (_, d) in enumerate(method_data):
        working = d[d.best_r2 > args.r2_threshold]
        jitter = i + rng.uniform(-0.12, 0.12, len(working))
        ax_pr.scatter(jitter, working.pr, s=20, c=palette[i], alpha=0.5, edgecolors="none")
    ax_pr.set_xticks(range(len(method_data)))
    ax_pr.set_xticklabels([m for m, _ in method_data], fontsize=10)
    ax_pr.set_ylabel("Effective non-GFP drivers (PR)", fontsize=12)
    ax_pr.set_title(f"B.  Parsimony on R² > {args.r2_threshold} markers",
                    loc="left", fontsize=11.5)
    ax_pr.set_ylim(0, max(np.max(p) for p in pr_data if len(p)) + 0.5)

    plt.tight_layout()
    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    summary_csv = out_png.with_suffix(".csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Wrote {out_png}, {out_png.with_suffix('.pdf')}, {summary_csv}")


if __name__ == "__main__":
    main()
