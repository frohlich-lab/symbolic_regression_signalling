"""Plot per-marker sanity grids comparing per-minute fits and raw trajectories."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-minute fit sanity plots.")
    parser.add_argument(
        "--time-trajectories",
        type=Path,
        required=True,
        help="Path to functional_groups_time_series.csv (raw averages).",
    )
    parser.add_argument(
        "--fit-trajectories",
        type=Path,
        required=True,
        help="Path to functional_groups_per_minute_fit.csv (dense fits).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experimental/runs/aggregated/plots/fit_sanity"),
        help="Directory for emitted PNG grids.",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default="p-ERK1-2",
        help="Protein column to plot (default: p-ERK1-2).",
    )
    parser.add_argument(
        "--markers",
        nargs="*",
        default=None,
        help="Optional subset of markers. Defaults to all markers present in the fit CSV.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=4,
        help="Maximum subplot columns per figure (default: 4).",
    )
    return parser.parse_args()


def _load_dataframe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    if "marker" in df.columns:
        df["marker"] = df["marker"].astype(str)
    if "GFP_bin" in df.columns:
        df["GFP_bin"] = pd.to_numeric(df["GFP_bin"], errors="coerce")
    if "timepoint" in df.columns:
        df["timepoint"] = pd.to_numeric(df["timepoint"], errors="coerce")
    return df


def _model_label(raw_slice: pd.DataFrame, protein: str) -> str:
    model_col = f"{protein}_fit_model"
    if model_col in raw_slice.columns:
        values = raw_slice[model_col].dropna().astype(str).unique()
        if len(values):
            return values[0]
    return "unknown"


def _r2_label(raw_slice: pd.DataFrame, protein: str) -> str:
    r2_col = f"{protein}_fit_r2"
    if r2_col in raw_slice.columns:
        vals = raw_slice[r2_col].dropna().to_numpy(dtype=float)
        if vals.size:
            r2 = vals[0]
            if np.isfinite(r2):
                return f"R²={r2:.3f}"
    return "R²=NA"


def _plot_bin(
    ax: plt.Axes,
    marker: str,
    gfp_bin: int,
    protein: str,
    raw_slice: Optional[pd.DataFrame],
    fit_slice: Optional[pd.DataFrame],
) -> None:
    label_parts = [f"GFP bin {int(gfp_bin)}"]
    if raw_slice is not None and not raw_slice.empty:
        label_parts.append(_model_label(raw_slice, protein))
        label_parts.append(_r2_label(raw_slice, protein))
    ax.set_title(" • ".join(label_parts), fontsize=9.5)
    ax.set_xlabel("time (min)")
    ax.set_ylabel(protein)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    if raw_slice is not None and not raw_slice.empty and protein in raw_slice.columns:
        raw_series = raw_slice[["timepoint", protein]].dropna().sort_values("timepoint")
        if not raw_series.empty:
            ax.scatter(
                raw_series["timepoint"],
                raw_series[protein],
                label="raw",
                color="#1b9e77",
                s=28,
                zorder=3,
            )

    fit_col = f"{protein}_fit"
    if fit_slice is not None and not fit_slice.empty and fit_col in fit_slice.columns:
        fit_series = fit_slice[["timepoint", fit_col]].dropna().sort_values("timepoint")
        if not fit_series.empty:
            ax.plot(
                fit_series["timepoint"],
                fit_series[fit_col],
                label="per-minute fit",
                color="#d95f02",
                linewidth=1.6,
            )


def main() -> None:
    args = parse_args()
    raw_df = _load_dataframe(args.time_trajectories, "time_trajectories CSV")
    fit_df = _load_dataframe(args.fit_trajectories, "per-minute fit CSV")

    required_fit_cols = {"marker", "GFP_bin", "timepoint", f"{args.protein}_fit"}
    if missing := [c for c in required_fit_cols if c not in fit_df.columns]:
        raise ValueError(f"Fit CSV missing columns: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markers = args.markers or sorted(fit_df["marker"].dropna().unique())
    if not markers:
        raise ValueError("No markers found in per-minute fit CSV.")

    for marker in markers:
        marker_fit = fit_df[fit_df["marker"] == marker].copy()
        if marker_fit.empty:
            continue
        bins = sorted(marker_fit["GFP_bin"].dropna().unique())
        if not bins:
            continue

        ncols = min(max(1, args.max_columns), len(bins))
        nrows = math.ceil(len(bins) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for ax in axes_flat[len(bins):]:
            ax.set_visible(False)

        for ax, gfp_bin in zip(axes_flat, bins):
            raw_slice = raw_df[
                (raw_df["marker"] == marker)
                & (raw_df["GFP_bin"] == gfp_bin)
            ].copy() if {"marker", "GFP_bin"}.issubset(raw_df.columns) else None
            fit_slice = marker_fit[marker_fit["GFP_bin"] == gfp_bin].copy()
            _plot_bin(ax, marker, gfp_bin, args.protein, raw_slice, fit_slice)

        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
        fig.suptitle(f"{marker} — {args.protein} per-minute fit", fontsize=14)
        fig.tight_layout(rect=(0.03, 0.05, 0.97, 0.95))

        slug = "".join(ch if ch.isalnum() else "_" for ch in marker).strip("_").lower() or "marker"
        out_path = args.output_dir / f"{slug}_{args.protein.replace('-', '').replace(' ', '')}_fit.png"
        fig.savefig(out_path, dpi=320, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
