"""Generate functional group SR plots from stored summary results."""
from __future__ import annotations

import argparse
import logging
import ast
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

LOGGER = logging.getLogger("experimental.plot_functional_groups")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

GROUPS_FRESH = {
    "RTK_tri_alt_EMF": ["EGFR", "MET", "FGFR1"],
    "RTK_tri_alt_BMF": ["ERBB2", "MET", "FGFR1"],
    "RTK_quad_noFGFR1": ["EGFR", "ERBB2", "MET", "MST1R"],
    "RTK_quad_noERBB2": ["EGFR", "FGFR1", "MET", "MST1R"],
    "RTK_MEK_crosstalk_1": ["EGFR", "TYRO3", "MAP2K2"],
    "RTK_MEK_crosstalk_2": ["ERBB2", "ABL1", "MAP2K2"],
    "RTK_RAF_bridge": ["MST1R", "ARAF", "EGFR"],
    "RAF_with_cytTK": ["ARAF", "ABL1", "TEC"],
    "MEK_with_cytTK": ["MAP2K2", "TYRO3", "TEC"],
    "RTK_cyt_duo_BA": ["ERBB2", "ABL1"],
    "RTK_cyt_duo_MT": ["MET", "TEC"],
    "RTK_cyt_duo_FT": ["FGFR1", "TYRO3"],
    "ERK_pair_plus_EGFR": ["MAPK1", "MAPK3", "EGFR"],
    "ERK_pair_plus_MEK": ["MAPK1", "MAPK3", "MAP2K2"],
    "DUSP_PTPN_mix_1": ["DUSP4", "PTPN2 (P1)"],
    "DUSP_PTPN_mix_2": ["DUSP7", "PTPN7"],
    "DUSP_trio_noDUSP7": ["DUSP4", "DUSP10 (P2)", "DUSP16"],
    "PTPN_DUSP_combo": ["PTPN5", "DUSP7"],
    "RSK_PKA_combo_alt": ["RPS6KA3", "RPS6KA6", "PRKACA"],
    "AKT_PKA_gate": ["AKT3", "PRKACA"],
    "PIK3R1_with_RSK1": ["PIK3R1", "RPS6KA1"],
    "AKT_with_RSK3": ["AKT3", "RPS6KA3"],
    "PTPN_trio_tilted": ["PTPN2 (P1)", "PTPN5", "DUSP10 (P2)"],
    "DUSP_pair_alt": ["DUSP10 (P2)", "DUSP7"],
    "Stress_mod_trio_A": ["TBK1", "DYRK2", "PIP5K3"],
    "Stress_mod_trio_B": ["MAP4K2", "DYRK3", "MAST2"],
    "Scaffold_traffic_pair": ["MAST2", "PIP5K3"],
    "DYRK_TBK_axis": ["DYRK2", "TBK1"],
    "MAP4K2_scaffold_axis": ["MAP4K2", "ALPK2"],
    "Scaffold_mix_with_TEC": ["MAST2", "ALPK2", "TEC"],
    "DYRK_scaffold_mix": ["DYRK3", "MAST2", "PIP5K3"],
    "Trafficking_with_FGFR1": ["PIP5K3", "FGFR1"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create diagnostic plots for functional group symbolic regression results.")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to the functional_group_sr_summary.csv file produced by sr_functional_groups.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="functional_group_log_r2",
        help="Base filename (without extension) for the emitted plots.",
    )
    parser.add_argument(
        "--variant",
        choices=("legacy", "combined", "fresh"),
        default="legacy",
        help="Which group set to visualise: legacy (default), combined (legacy + curated), or fresh only.",
    )
    return parser.parse_args()


def _prettify_label(raw: str) -> str:
    label = raw.replace("_", " ")
    tokens = label.split()
    pretty: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower == "and":
            pretty.append("&")
        elif token.islower():
            pretty.append(token.capitalize())
        else:
            pretty.append(token)
    return " ".join(pretty)


def _filter_summary(summary: pd.DataFrame, variant: str) -> pd.DataFrame:
    fresh_names = set(GROUPS_FRESH.keys())
    if variant == "legacy":
        mask = ~summary["group_name"].isin(fresh_names)
    elif variant == "combined":
        mask = summary["group_name"].notna()
    elif variant == "fresh":
        mask = summary["group_name"].isin(fresh_names)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    filtered = summary.loc[mask].copy()
    if variant in {"combined", "fresh"}:
        missing = sorted(fresh_names - set(summary["group_name"]))
        if missing:
            LOGGER.warning(
                "Fresh group results missing for: %s",
                ", ".join(missing),
            )
    return filtered


def plot_log_r2(summary: pd.DataFrame, output_dir: Path, basename: str, variant: str) -> None:
    df = summary.dropna(subset=["test_log_r2"]).copy()
    if df.empty:
        LOGGER.warning("Summary contains no valid log R² values; skipping plot generation")
        return

    df["test_log_r2"] = df["test_log_r2"].clip(lower=0.0)

    df["pretty_group"] = df["group_name"].apply(_prettify_label)

    marker_lookup: Dict[str, List[str]] = {}
    for _, row in df[["pretty_group", "markers"]].drop_duplicates("pretty_group").iterrows():
        markers_raw = row["markers"]
        parsed: List[str]
        if isinstance(markers_raw, (list, tuple)):
            parsed = [str(m) for m in markers_raw]
        else:
            try:
                parsed = [str(m).strip() for m in ast.literal_eval(str(markers_raw))]
            except Exception:
                cleaned = str(markers_raw).strip("[]")
                parsed = [m.strip().strip("'\"") for m in cleaned.split(",") if m.strip()]
        marker_lookup[row["pretty_group"]] = parsed

    pivot = df.pivot_table(index="pretty_group", columns="feature_mode", values="test_log_r2")
    modes = sorted(pivot.columns.tolist())

    if "gfp" in pivot.columns:
        order = pivot["gfp"].fillna(-np.inf).sort_values(ascending=False).index
    else:
        order = pivot.max(axis=1).fillna(-np.inf).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    groups = list(pivot.index)

    a4_width_in = 8.27
    a4_height_in = 11.69
    fig_width = max(1.3 * a4_width_in / 2, 8.0)
    base_height = (3 / 5) * a4_height_in
    fig_height = base_height

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })

    if variant == "combined":
        fig_height = max(base_height * 1.1, len(groups) * 0.5)
    elif variant == "fresh":
        fig_height = max(base_height * 0.9, len(groups) * 0.45)
    else:
        fig_height = max(base_height, len(groups) * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    y = np.arange(len(groups))
    height = 0.35 if len(modes) == 2 else 0.6
    mode_labels = {
        "gfp": "GFP",
        "all": "GFP + neighbouring nodes",
    }
    mode_colors = {
        "all": "#1f77b4",
        "gfp": "#ff7f0e",
    }

    max_val = float(np.nanmax(pivot.values)) if pivot.size else 1.0

    for idx, mode in enumerate(modes):
        mode_values = pivot[mode].values
        offset = (idx - (len(modes) - 1) / 2) * height
        if len(modes) == 2:
            if mode == "gfp":
                offset = -abs(offset)
            else:
                offset = abs(offset)
        label = mode_labels.get(mode, mode)
        color = mode_colors.get(mode)
        ax.barh(y + offset, mode_values, height=height, label=label, color=color)
        for y_pos, value in zip(y, mode_values):
            ax.text(
                value + max_val * 0.02,
                y_pos + offset,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7,
                color="#333333",
            )

    samples = df.groupby("pretty_group")[["train_samples", "test_samples"]].first()
    y_labels: List[str] = []
    for group in groups:
        train = int(samples.loc[group, "train_samples"]) if group in samples.index else 0
        test = int(samples.loc[group, "test_samples"]) if group in samples.index else 0
        proteins = marker_lookup.get(group, [])
        protein_label = ", ".join(proteins) if proteins else group
        lines = [protein_label, f"train={train:,} test={test:,}"]
        y_labels.append("\n".join(lines))

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test R² (log$_{10}$ space)")
    variant_titles = {
        "legacy": "Symbolic Regression Performance by Functional Group",
        "combined": "Symbolic Regression Performance: Canonical + curated stacks",
        "fresh": "Symbolic Regression Performance: Curated stack candidates",
    }
    ax.set_title(variant_titles.get(variant, "Symbolic Regression Performance"))
    ax.legend(title="Perturbational features", loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False, prune=None))

    ax.set_xlim(0, max(max_val * 1.2, 0.5))

    fig.tight_layout()
    png_path = output_dir / f"{basename}.png"
    svg_path = output_dir / f"{basename}.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)
    LOGGER.info("Wrote plots: %s, %s", png_path.name, svg_path.name)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading summary from %s", args.summary)
    summary = pd.read_csv(args.summary)
    filtered = _filter_summary(summary, args.variant)
    if filtered.empty:
        LOGGER.warning("No rows available for variant '%s'; skipping plot", args.variant)
        return
    LOGGER.info("Generating %s plot in %s", args.variant, output_dir)
    plot_log_r2(filtered, output_dir, args.basename, args.variant)


if __name__ == "__main__":
    main()
