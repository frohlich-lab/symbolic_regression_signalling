"""Pairwise metric comparison plots for functional group SR results with KDE and zoomed variants."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import sys
from typing import Dict, Optional, List, Tuple, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
from tqdm.auto import tqdm

from experimental.sr_pipeline.run_functional_groups import EXCLUDE_COLUMNS, sanitize_feature_names
from experimental.sr_pipeline import seed_annotations as seed_annot

# Class colouring shared across plots
CLASS_PALETTE = {"driver": "#1b9e77", "brake": "#d95f02", "neutral": "#636363"}
MEASURED_DEFAULT = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
PLACEHOLDER_TEXT = "No data"
DEFAULT_SEED_NOTE: Optional[str] = seed_annot.DEFAULT_SEED_NOTE

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise scatter plots for R2 and relMAE across models/modes.")
    parser.add_argument(
        "--summary", type=Path, required=True, help="functional_group_summary.csv produced by run_functional_groups.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory to save plots (scatter outputs under scatter/, line plots under lines/).",
    )
    parser.add_argument(
        "--group-definitions-csv",
        type=Path,
        default=Path("data/experimental/processed/functional_groups.csv"),
        help="Optional group definitions with directions for colouring (columns: group, members, direction).",
    )
    parser.add_argument(
        "--integration-metrics-dir",
        type=Path,
        default=None,
        help="Directory containing marker_integration_metrics_{snapshot,per_minute}.csv for integrated R2 vs dt R2 plots.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=None,
        help="Directory containing predicted_trajectories_{snapshot,per_minute}.csv and marker_integration_trajectories_{snapshot,per_minute}.csv.",
    )
    parser.add_argument(
        "--snapshot-dataset",
        type=Path,
        default=None,
        help="Snapshot dataset CSV used for feature importance plots (optional; skips importance plots when missing).",
    )
    parser.add_argument(
        "--per-minute-dataset",
        type=Path,
        default=None,
        help="Per-minute dataset CSV used for feature importance plots (optional; skips importance plots when missing).",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_DEFAULT,
        help="Measured timepoints used for per-minute filtering in integration plots.",
    )
    parser.add_argument(
        "--zoom-threshold",
        type=float,
        default=0.7,
        help="R2 zoom threshold; zoomed plots show x/y >= this value (default 0.7).",
    )
    parser.add_argument(
        "--model-scatter-dataset-mode",
        type=str,
        default="per_minute",
        help="Dataset mode to use for PySR vs Linear Regression scatter plots (default per_minute; "
        "falls back to any available mode if missing).",
    )
    parser.add_argument(
        "--model-scatter-feature-mode",
        type=str,
        default=None,
        help="Optional feature mode filter for the PySR vs Linear Regression scatter plots "
        "(case-insensitive; defaults to 'all' when multiple modes are present and no value is provided).",
    )
    parser.add_argument(
        "--diagnostic-max-markers",
        type=int,
        default=4,
        help="Max PySR markers to plot in the ODE diagnostics (0 disables).",
    )
    parser.add_argument(
        "--diagnostic-dt-threshold",
        type=float,
        default=0.6,
        help="Minimum dt R² required for diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-ode-threshold",
        type=float,
        default=0.2,
        help="Maximum ODE R² allowed for diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-gap-threshold",
        type=float,
        default=0.3,
        help="Minimum dt R² − ODE R² gap required for diagnostics.",
    )
    parser.add_argument(
        "--binwise-perk-markers",
        nargs="*",
        type=str,
        default=("ERBB2", "PIP5K3", "DUSP10 (P2)", "ARAF"),
        help="Markers to plot predicted vs ground-truth pERK per GFP bin (per seed, compact panels).",
    )
    return parser.parse_args()


def _flatten_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column pivot to single-level names."""
    if not isinstance(pivot.columns, pd.MultiIndex):
        return pivot
    pivot = pivot.copy()
    pivot.columns = [
        "_".join([str(level) for level in col if str(level) != "nan"]).strip("_")
        for col in pivot.columns
    ]
    return pivot


def _load_marker_classes(path: Path) -> Dict[str, str]:
    classes: Dict[str, str] = {}
    if not path.exists():
        return classes
    try:
        meta = pd.read_csv(path)
        if {"group", "members", "direction"}.issubset(meta.columns):
            for _, row in meta.iterrows():
                direction = str(row["direction"])
                if direction == "1":
                    cls = "driver"
                elif direction == "-1":
                    cls = "brake"
                elif direction == "0":
                    cls = "neutral"
                else:
                    cls = "neutral"
                members_raw = row.get("members", "")
                vals = []
                if isinstance(members_raw, str):
                    try:
                        vals = [str(v).strip() for v in ast.literal_eval(members_raw)]
                    except Exception:
                        vals = [m.strip() for m in members_raw.strip("[]").split(",") if m.strip()]
                for m in vals:
                    classes[m] = cls
    except Exception:
        classes = {}
    return classes


def _compute_samples_by_mode(df: pd.DataFrame) -> pd.Series:
    """Return per (group, dataset_mode) sample counts without double-counting models."""
    required = {"group_name", "dataset_mode", "total_samples"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float)
    return (
        df[["group_name", "dataset_mode", "total_samples"]]
        .dropna(subset=["group_name", "dataset_mode"])
        .drop_duplicates()
        .groupby(["group_name", "dataset_mode"])["total_samples"]
        .max()
    )


def _infer_seeds_from_path(path: Path) -> List[str]:
    """Extract seed ids from a summary path (e.g., .../seed_42/summary/...)."""
    return seed_annot.seeds_from_paths([path])


def _collect_seeds_from_csv(path: Path) -> List[str]:
    """Collect seed-like columns from a CSV if present; prefer explicit columns."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ("seed", "random_state"))
    except Exception:
        return []
    return seed_annot.seeds_from_df(df)


def _match_preferred_label(options: Sequence[object], target: Optional[str]) -> Optional[object]:
    """Return the first value from options whose normalized string matches target."""
    if not options or target is None:
        return None
    normalized = str(target).strip().lower()
    for val in options:
        if str(val).strip().lower() == normalized:
            return val
    return None


def _add_seed_note(fig: plt.Figure, note: Optional[str]) -> None:
    """Write seed/aggregation note onto a figure."""
    seed_annot.add_seed_note(fig, note=note, default=DEFAULT_SEED_NOTE)


def _format_seed_label(seeds: Sequence[object], *, averaged: bool) -> Optional[str]:
    """Format a seed label indicating whether values are averaged."""
    return seed_annot.format_seed_label(seeds, averaged=averaged, default=None)


def _progress_printer(planned: Sequence[str]) -> Tuple[callable, int]:
    total = len(planned)
    counter = {"idx": 0}

    def _log(label: str) -> None:
        counter["idx"] += 1
        print(f"[plot_metrics_summary] ({counter['idx']}/{total}) {label}", flush=True)

    return _log, total


def _plot_scatter_variants(
    outdir: Path,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    base_title: str,
    stem_prefix: str,
    zoom_thr: float,
    *,
    zoom: bool = False,
    log_axes: bool = False,
    draw_threshold: bool = True,
    metric_group: str = "dt",
    seed_note: Optional[str] = None,
) -> None:
    if data.empty or x_col not in data or y_col not in data:
        return
    variants = [
        ("plain", None, None, "No colour"),
        ("samples", "total_samples", "viridis", "Total samples"),
        ("class", "marker_class", CLASS_PALETTE, "Marker type"),
    ]
    for variant, hue_col, palette, legend_title in variants:
        df_local = data.copy()
        if hue_col and hue_col not in df_local.columns:
            continue
        title = f"{base_title} ({legend_title})"
        stem = stem_prefix + ("_zoom" if zoom else "")
        if zoom:
            df_local = df_local[(df_local[x_col] >= zoom_thr) & (df_local[y_col] >= zoom_thr)]
        if df_local.empty:
            continue

        parts = [metric_group, variant, "zoomed" if zoom else "full"]
        variant_dir = outdir.joinpath(*parts)
        variant_dir.mkdir(parents=True, exist_ok=True)

        g = sns.JointGrid(data=df_local, x=x_col, y=y_col, height=6)
        sns.scatterplot(
            data=df_local,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette=palette,
            s=35,
            ax=g.ax_joint,
            legend="brief" if hue_col else False,
        )
        # Diagonal and thresholds
        diag_min = min(df_local[x_col].min(), df_local[y_col].min(), zoom_thr if zoom else 0.0)
        diag_max = max(df_local[x_col].max(), df_local[y_col].max(), zoom_thr)
        g.ax_joint.plot([diag_min, diag_max], [diag_min, diag_max], color="gray", linestyle="--", alpha=0.6)
        if draw_threshold:
            g.ax_joint.axhline(zoom_thr, color="red", linestyle="--", alpha=0.5)
            g.ax_joint.axvline(zoom_thr, color="red", linestyle="--", alpha=0.5)
        g.ax_joint.grid(True, linestyle="--", alpha=0.3)

        if log_axes:
            eps = 1e-6
            g.ax_joint.set_xscale("log")
            g.ax_joint.set_yscale("log")
            g.ax_joint.set_xlim(left=max(eps, df_local[x_col].min()), right=max(df_local[x_col].max(), zoom_thr))
            g.ax_joint.set_ylim(bottom=max(eps, df_local[y_col].min()), top=max(df_local[y_col].max(), zoom_thr))
        else:
            if zoom:
                g.ax_joint.set_xlim(left=zoom_thr, right=max(df_local[x_col].max(), zoom_thr))
                g.ax_joint.set_ylim(bottom=zoom_thr, top=max(df_local[y_col].max(), zoom_thr))
            else:
                g.ax_joint.set_xlim(left=0, right=max(1.0, df_local[x_col].max()))
                g.ax_joint.set_ylim(bottom=0, top=max(1.0, df_local[y_col].max()))

        # KDE/Hist on margins
        sns.kdeplot(data=df_local, x=x_col, fill=True, ax=g.ax_marg_x)
        sns.kdeplot(data=df_local, y=y_col, fill=True, ax=g.ax_marg_y)
        if log_axes:
            g.ax_marg_x.set_xscale("log")
            g.ax_marg_y.set_yscale("log")
        g.ax_marg_x.grid(True, linestyle="--", alpha=0.2)
        g.ax_marg_y.grid(True, linestyle="--", alpha=0.2)

        if hue_col and g.ax_joint.get_legend():
            g.ax_joint.legend(title=legend_title)

        g.fig.suptitle(title)
        _add_seed_note(g.fig, seed_note)
        g.fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "svg"):
            g.fig.savefig(variant_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(g.fig)


def _copy_preferred_variant_to_base(outdir: Path, stem: str, metric_group: str = "dt") -> None:
    """Expose a single preferred variant at the parent level via symlink (no duplicate data)."""
    preferences = ("class", "samples", "plain")
    for variant in preferences:
        created = False
        for ext in ("png", "svg"):
            src = outdir / metric_group / variant / "full" / f"{stem}.{ext}"
            if not src.exists():
                continue
            dst = outdir / metric_group / f"{stem}.{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                try:
                    dst.unlink()
                except Exception:
                    pass
            try:
                # Symlink avoids duplicating the actual image contents.
                dst.symlink_to(src.relative_to(dst.parent))
                created = True
            except Exception:
                try:
                    shutil.copyfile(src, dst)
                    created = True
                except Exception:
                    continue
        if created:
            break


def _collect_feature_counts(df: pd.DataFrame, marker_type: Dict[str, str]) -> Dict[tuple, Dict[str, Dict[str, int]]]:
    """Count feature occurrences in formulas per (model, dataset_mode) broken down by marker class."""
    counts: Dict[tuple, Dict[str, Dict[str, int]]] = {}
    for (model, dataset_mode), group in df.groupby(["model", "dataset_mode"]):
        combo_counts: Dict[str, Dict[str, int]] = {}
        for _, row in group.iterrows():
            formula = row.get("formula")
            if not isinstance(formula, str) or not formula.strip():
                continue
            try:
                symbols = [str(s) for s in sp.sympify(formula).free_symbols]
            except Exception:
                continue
            marker_cls = marker_type.get(row.get("group_name"), "neutral")
            for sym in symbols:
                bucket = combo_counts.setdefault(sym, {"driver": 0, "brake": 0, "neutral": 0})
                bucket[marker_cls] = bucket.get(marker_cls, 0) + 1
        if combo_counts:
            counts[(model, dataset_mode)] = combo_counts
    return counts


def _plot_feature_bars(
    counts_by_combo: Dict[tuple, Dict[str, Dict[str, int]]],
    outdir: Path,
    seed_note: Optional[str] = None,
) -> None:
    """Plot stacked bar charts of feature usage counts per (model, dataset_mode)."""
    if not counts_by_combo:
        return
    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for (model, mode), feat_counts in counts_by_combo.items():
        df_counts = pd.DataFrame.from_dict(feat_counts, orient="index").fillna(0)
        if df_counts.empty:
            continue
        df_counts = df_counts.reindex(columns=["driver", "brake", "neutral"], fill_value=0)
        df_counts["total"] = df_counts.sum(axis=1)
        df_counts = df_counts[df_counts["total"] > 0].sort_values("total", ascending=False)
        if df_counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(df_counts) * 0.35), 5))
        df_counts[["driver", "brake", "neutral"]].plot(
            kind="bar",
            stacked=True,
            color=CLASS_PALETTE,
            ax=ax,
            width=0.8,
        )
        ax.set_xlabel("Feature")
        ax.set_ylabel("Count across formulas")
        ax.set_title(f"Feature usage: {model} | {mode}")
        ax.legend(title="Marker class")
        ax.set_xticklabels(df_counts.index, rotation=45, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        _add_seed_note(fig, seed_note)
        fig.tight_layout()
        model_slug = model.lower().replace(" ", "_")
        mode_slug = mode.replace(" ", "_")
        for ext in ("png", "svg"):
            fig.savefig(features_dir / f"feature_usage_{model_slug}_{mode_slug}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _extract_markers_list(markers_value: object, fallback: str) -> List[str]:
    """Parse the markers column into a list, falling back to the group name."""
    if isinstance(markers_value, (list, tuple, set)):
        return [str(m) for m in markers_value if str(m)]
    if markers_value is None or (isinstance(markers_value, float) and np.isnan(markers_value)):
        return [fallback]
    try:
        parsed = ast.literal_eval(str(markers_value))
        if isinstance(parsed, (list, tuple, set)):
            items = [str(m) for m in parsed if str(m)]
            if items:
                return items
    except Exception:
        pass
    if isinstance(markers_value, str):
        cleaned = [p.strip().strip("'\"") for p in markers_value.strip("[]").split(",") if p.strip()]
        if cleaned:
            return cleaned
    return [fallback]


def _parse_linear_coefficients(formula: str) -> Dict[str, float]:
    """Extract linear coefficients from a sympy-parsable formula string."""
    try:
        expr = sp.sympify(formula)
    except Exception:
        return {}
    coeffs: Dict[str, float] = {}
    for sym in expr.free_symbols:
        try:
            coeffs[str(sym)] = float(expr.diff(sym))
        except Exception:
            continue
    return coeffs


def _prepare_importance_dataset(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load and sanitize a dataset for feature importance computation."""
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df = df.copy()
    df.columns = df.columns.str.replace("_fit$", "", regex=True)
    if "marker" not in df.columns or "GFP_bin" not in df.columns:
        return None
    df["marker"] = df["marker"].astype(str)
    df["GFP_bin"] = pd.to_numeric(df["GFP_bin"], errors="coerce")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    sanitized = sanitize_feature_names(feature_cols)
    rename_map = dict(zip(feature_cols, sanitized))
    df = df.rename(columns=rename_map)
    for col in sanitized:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _compute_feature_importance(
    summary_lin: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Compute normalized feature importance per marker from linear regression formulas."""
    records: List[Dict[str, object]] = []
    if summary_lin.empty or dataset is None or dataset.empty:
        return pd.DataFrame()

    feature_cols = [c for c in dataset.columns if c not in EXCLUDE_COLUMNS]
    for _, row in summary_lin.iterrows():
        formula = row.get("formula")
        if not isinstance(formula, str) or not formula.strip():
            continue
        coeffs = _parse_linear_coefficients(formula)
        if not coeffs:
            continue
        group_name = str(row.get("group_name"))
        markers = _extract_markers_list(row.get("markers"), group_name)
        subset = dataset[dataset["marker"].isin(markers)].copy()
        subset = subset.dropna(subset=["GFP_bin"])
        if subset.empty:
            continue
        grouped = subset.groupby("GFP_bin")[feature_cols].mean()
        if grouped.empty:
            continue
        raw_vals: Dict[str, float] = {}
        coef_vals: Dict[str, float] = {}
        for feat in grouped.columns:
            coef = float(coeffs.get(feat, 0.0))
            vals = grouped[feat].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            contrib = vals * coef
            contrib = contrib[np.isfinite(contrib)]
            if contrib.size == 0:
                continue
            raw = float(np.var(contrib))
            if np.isfinite(raw):
                raw_vals[feat] = max(0.0, raw)
            coef_vals[feat] = abs(coef)
        if not raw_vals:
            continue
        total_raw = float(sum(raw_vals.values()))
        total_coef = float(sum(v for v in coef_vals.values() if np.isfinite(v)))
        if total_raw <= 0:
            continue
        for feat, raw_val in raw_vals.items():
            coef_val = coef_vals.get(feat, 0.0)
            records.append(
                {
                    "group_name": group_name,
                    "feature": feat,
                    "dataset_mode": row.get("dataset_mode"),
                    "feature_mode": row.get("feature_mode"),
                    "raw_importance": raw_val,
                    "normalized_importance": raw_val / total_raw if total_raw > 0 else np.nan,
                    "total_raw_importance": total_raw,
                    "coef_importance": coef_val,
                    "normalized_coef_importance": (coef_val / total_coef) if total_coef > 0 else np.nan,
                    "total_coef_importance": total_coef,
                }
            )
    return pd.DataFrame(records)


def _count_pysr_variables(formula: str, allowed_features: Optional[Sequence[str]] = None) -> Optional[int]:
    """Return the number of unique feature variables in a PySR formula; None on parse failure."""
    if not isinstance(formula, str) or not formula.strip():
        return None
    allowed_set = set(allowed_features) if allowed_features is not None else None
    try:
        expr = sp.sympify(formula)
        symbols = {str(sym) for sym in expr.free_symbols}
        if allowed_set is not None:
            symbols = {s for s in symbols if s in allowed_set}
        return len(symbols)
    except Exception:
        return None


def _plot_pysr_symbol_counts(
    summary_df: pd.DataFrame,
    outdir: Path,
    expected_modes: Sequence[str],
    seed_note: Optional[str],
    feature_sets: Optional[Dict[str, Sequence[str]]] = None,
) -> None:
    """Boxplots of unique variable counts per marker for PySR formulas."""
    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    if summary_df.empty or "formula" not in summary_df.columns:
        stems = [f"pysr_symbol_counts_{mode}" for mode in expected_modes]
        _write_feature_importance_placeholders(features_dir, stems, seed_note)
        return

    records: List[Dict[str, object]] = []
    for _, row in summary_df.iterrows():
        mode = row.get("dataset_mode", "snapshot")
        allowed = None
        if feature_sets is not None and mode in feature_sets:
            allowed = feature_sets[mode]
        count = _count_pysr_variables(row.get("formula"), allowed_features=allowed)
        if count is None:
            continue
        records.append(
            {
                "group_name": row.get("group_name"),
                "dataset_mode": mode,
                "unique_symbols": count,
            }
        )
    counts_df = pd.DataFrame(records)
    if counts_df.empty:
        stems = [f"pysr_symbol_counts_{mode}" for mode in expected_modes]
        _write_feature_importance_placeholders(features_dir, stems, seed_note)
        return

    for mode in expected_modes:
        stem = f"pysr_symbol_counts_{mode}"
        subset = counts_df[counts_df["dataset_mode"] == mode]
        if subset.empty:
            _write_feature_importance_placeholders(features_dir, [stem], seed_note)
            continue
        plt.figure(figsize=(4, 3.2))
        sns.boxplot(data=subset, y="unique_symbols", color="#1f77b4")
        plt.ylabel("Unique variables in PySR formula")
        plt.xlabel("")
        plt.title(f"PySR variable counts — {mode.replace('_', ' ')}")
        plt.tight_layout()
        _add_seed_note(plt.gcf(), seed_note)
        for ext in ("png", "svg"):
            plt.savefig(features_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close()


def _write_feature_importance_placeholders(
    features_dir: Path, stems: Sequence[str], seed_note: Optional[str] = None
) -> None:
    """Write placeholder plots for missing feature-importance data."""
    features_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        plt.figure(figsize=(3.2, 2.4))
        plt.text(0.5, 0.5, PLACEHOLDER_TEXT, ha="center", va="center")
        plt.axis("off")
        _add_seed_note(plt.gcf(), seed_note)
        for ext in ("png", "svg"):
            plt.savefig(features_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close()


def _plot_feature_importance(
    importance_df: pd.DataFrame,
    outdir: Path,
    expected_modes: Sequence[str],
    seed_note: Optional[str] = None,
) -> None:
    """Create total, stacked-share, coef-share, and boxplot views of feature importance per dataset mode."""
    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    if importance_df is None or importance_df.empty:
        for mode in expected_modes:
            stems = [
                f"feature_importance_total_{mode}",
                f"feature_importance_stacked_{mode}",
                f"feature_importance_coef_stacked_{mode}",
                f"feature_importance_box_{mode}",
            ]
            _write_feature_importance_placeholders(features_dir, stems, seed_note)
        return

    all_features = sorted(importance_df["feature"].dropna().unique().tolist())
    if not all_features:
        for mode in expected_modes:
            stems = [
                f"feature_importance_total_{mode}",
                f"feature_importance_stacked_{mode}",
                f"feature_importance_coef_stacked_{mode}",
                f"feature_importance_box_{mode}",
            ]
            _write_feature_importance_placeholders(features_dir, stems, seed_note)
        return

    palette = sns.color_palette("tab20", n_colors=max(3, len(all_features)))
    color_map = {feat: palette[idx % len(palette)] for idx, feat in enumerate(all_features)}
    feature_order_global = (
        importance_df.groupby("feature")["normalized_importance"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    coef_feature_order_global = (
        importance_df.groupby("feature")["normalized_coef_importance"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    for mode in expected_modes:
        stems = [
            f"feature_importance_total_{mode}",
            f"feature_importance_stacked_{mode}",
            f"feature_importance_coef_stacked_{mode}",
            f"feature_importance_box_{mode}",
        ]
        df_mode = importance_df[importance_df["dataset_mode"] == mode].copy()
        if df_mode.empty:
            _write_feature_importance_placeholders(features_dir, stems, seed_note)
            continue
        totals = (
            df_mode.groupby("group_name")["total_raw_importance"]
            .max()
            .sort_values(ascending=False)
        )
        if totals.empty:
            _write_feature_importance_placeholders(features_dir, stems, seed_note)
            continue
        marker_order = sorted(df_mode["group_name"].unique().tolist())
        feature_order = [f for f in feature_order_global if f in df_mode["feature"].unique()]
        if not feature_order:
            feature_order = df_mode["feature"].dropna().unique().tolist()
        coef_feature_order = [f for f in coef_feature_order_global if f in df_mode["feature"].unique()]
        if not coef_feature_order:
            coef_feature_order = feature_order

        # Plot 1: total raw importance per marker
        fig, ax = plt.subplots(figsize=(8, max(3, len(marker_order) * 0.35)))
        ax.barh(marker_order, totals.reindex(marker_order).values, color="#2c7fb8")
        ax.invert_yaxis()
        ax.set_xlabel("Total raw importance (variance across bins)")
        ax.set_title(f"Linear Regression feature importance — {mode}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        _add_seed_note(fig, seed_note)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(features_dir / f"{stems[0]}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Plot 2: stacked normalized importance shares
        pivot = df_mode.pivot_table(
            index="group_name",
            columns="feature",
            values="normalized_importance",
            fill_value=0.0,
        )
        pivot = pivot.reindex(marker_order)
        pivot = pivot[[f for f in feature_order if f in pivot.columns]]
        fig, ax = plt.subplots(figsize=(10, max(4, len(marker_order) * 0.4)))
        left = np.zeros(len(pivot))
        for feat in pivot.columns:
            vals = pivot[feat].to_numpy(dtype=float)
            ax.barh(
                pivot.index,
                vals,
                left=left,
                label=feat,
                color=color_map.get(feat, "#7f7f7f"),
                edgecolor="white",
            )
            left = left + vals
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalized importance share")
        ax.set_title(f"Feature importance share by marker — {mode}")
        ax.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        _add_seed_note(fig, seed_note)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(features_dir / f"{stems[1]}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Plot 2b: stacked normalized importance using |coef| shares
        pivot_coef = df_mode.pivot_table(
            index="group_name",
            columns="feature",
            values="normalized_coef_importance",
            fill_value=0.0,
        )
        pivot_coef = pivot_coef.reindex(marker_order)
        pivot_coef = pivot_coef[[f for f in coef_feature_order if f in pivot_coef.columns]]
        fig, ax = plt.subplots(figsize=(10, max(4, len(marker_order) * 0.4)))
        left = np.zeros(len(pivot_coef))
        for feat in pivot_coef.columns:
            vals = pivot_coef[feat].to_numpy(dtype=float)
            ax.barh(
                pivot_coef.index,
                vals,
                left=left,
                label=feat,
                color=color_map.get(feat, "#7f7f7f"),
                edgecolor="white",
            )
            left = left + vals
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Normalized |coef| share")
        ax.set_title(f"Feature importance (|coef|) share by marker — {mode}")
        ax.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        _add_seed_note(fig, seed_note)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(features_dir / f"{stems[2]}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # Plot 3: distribution across markers per feature
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(feature_order) * 0.35)))
        sns.boxplot(
            data=df_mode,
            x="normalized_importance",
            y="feature",
            order=feature_order,
            palette=color_map,
            ax=ax,
            linewidth=1.1,
        )
        ax.set_xlabel("Normalized importance")
        ax.set_ylabel("Feature")
        ax.set_title(f"Feature importance across markers — {mode}")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        _add_seed_note(fig, seed_note)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(features_dir / f"{stems[3]}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

def _plot_metric_boxplot(
    source_df: pd.DataFrame,
    outdir: Path,
    series_defs: List[Tuple[str, pd.Series]],
    stem: str,
    ylabel: str,
    seed_note: Optional[str] = None,
) -> None:
    """Boxplot across markers comparing a metric by model/dataset mode with side-by-side splits."""
    frames: List[pd.DataFrame] = []
    for label, series in series_defs:
        if series is None:
            continue
        s = pd.Series(series).dropna()
        if s.empty:
            continue
        df_sub = source_df.loc[s.index].copy()
        df_sub["metric_clamped"] = s.clip(lower=0)
        df_sub["split"] = label
        frames.append(df_sub)

    if not frames:
        return
    df_box = pd.concat(frames, axis=0)
    if df_box.empty or "model" not in df_box or "dataset_mode" not in df_box:
        return
    df_box["combo"] = df_box["model"] + " | " + df_box["dataset_mode"]
    order = [
        "PySR | snapshot",
        "PySR | per_minute",
        "Linear Regression | snapshot",
        "Linear Regression | per_minute",
    ]
    order = [c for c in order if c in df_box["combo"].unique()]
    if not order:
        return
    palette = {
        "PySR | snapshot": "#9ecae1",
        "PySR | per_minute": "#1f77b4",
        "Linear Regression | snapshot": "#fdd0a2",
        "Linear Regression | per_minute": "#ff7f0e",
    }
    hue_order = [lbl for lbl in ("Train", "Test", "Overall") if lbl in df_box["split"].unique()]
    split_palette = {
        "Train": "#74add1",
        "Test": "#f46d43",
        "Overall": "#7f7f7f",
    }
    box_dir = outdir / "boxes"
    box_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df_box,
        x="combo",
        y="metric_clamped",
        order=order,
        hue="split",
        hue_order=hue_order,
        palette=split_palette,
        linewidth=1.2,
    )
    counts_combo_split = df_box.groupby(["combo", "split"])["metric_clamped"].count()
    def _fmt_counts(combo: str) -> str:
        t = counts_combo_split.get((combo, "Train"), 0)
        te = counts_combo_split.get((combo, "Test"), 0)
        ov = counts_combo_split.get((combo, "Overall"), 0)
        return f"{combo}\nTrain n={t} | Test n={te} | Overall n={ov}"
    tick_labels = [_fmt_counts(c) for c in order]
    plt.xticks(ticks=range(len(order)), labels=tick_labels, rotation=15, ha="right")
    plt.ylim(0, max(1.0, df_box["metric_clamped"].max()))
    plt.xlabel("Model | Dataset mode")
    plt.ylabel(ylabel)
    split_counts = df_box.groupby("split")["metric_clamped"].count()
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        new_labels = [f"{lbl} (n={split_counts.get(lbl, 0)})" for lbl in labels]
        plt.legend(handles, new_labels, title="Split", framealpha=0.9)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    _add_seed_note(plt.gcf(), seed_note)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(box_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close()


def _plot_metric_lineplot(
    df: pd.DataFrame,
    outdir: Path,
    marker_type: Dict[str, str],
    metric_col: str,
    stem: str,
    title: str,
    seed_note: Optional[str] = None,
) -> None:
    """Line plot across markers for both models and dataset modes with two colours and opacity by mode."""
    if metric_col not in df.columns:
        return
    df_line = df[df[metric_col].notna()].copy()
    if df_line.empty:
        return
    df_line["metric_clamped"] = df_line[metric_col].clip(lower=0)

    # Order markers by PySR per_minute R2 (descending); fall back to appearance order.
    order_source = df_line[
        (df_line["model"] == "PySR") & (df_line["dataset_mode"] == "per_minute")
    ].sort_values("metric_clamped", ascending=False)
    ordered_markers = order_source["group_name"].tolist()
    for m in df_line["group_name"].unique():
        if m not in ordered_markers:
            ordered_markers.append(m)

    model_colors = {"PySR": "#1f77b4", "Linear Regression": "#ff7f0e"}
    model_markers = {"PySR": "o", "Linear Regression": "x"}
    mode_alpha = {"per_minute": 0.95, "snapshot": 0.45}

    line_dir = outdir / "lines"
    line_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(10, len(ordered_markers) * 0.35), 5))

    mode_order = ["snapshot", "per_minute"]
    for mode in mode_order:
        for model, color in model_colors.items():
            sub = df_line[(df_line["model"] == model) & (df_line["dataset_mode"] == mode)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["xpos"] = [ordered_markers.index(g) for g in sub["group_name"]]
            sub = sub.sort_values("xpos")
            ax.plot(
                sub["xpos"],
                sub["metric_clamped"],
                label=f"{model} | {mode}",
                color=color,
                alpha=mode_alpha.get(mode, 0.7),
                marker=model_markers.get(model, "o"),
                linestyle="-",
                linewidth=1.8,
                markersize=7,
            )

    ax.set_xlim(-0.5, len(ordered_markers) - 0.5)
    ax.set_ylim(0, max(1.0, df_line["metric_clamped"].max()))
    ax.set_xticks(range(len(ordered_markers)))
    tick_labels = []
    tick_bg = {
        "driver": "#cdecd9",  # light green
        "brake": "#f9dcc8",  # light orange/red
        "neutral": "#dcdcdc",  # light gray
    }
    for name in ordered_markers:
        cls = marker_type.get(name, "neutral")
        tick_labels.append((name, tick_bg.get(cls, "#ededed")))
    ax.set_xticklabels([t[0] for t in tick_labels], rotation=45, ha="right")
    for lbl, (_, bg) in zip(ax.get_xticklabels(), tick_labels):
        lbl.set_bbox(
            dict(facecolor=bg, alpha=0.6, edgecolor="none", boxstyle="round,pad=0.15")
        )

    ax.set_xlabel("Marker")
    ax.set_ylabel(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(framealpha=0.9)
    _add_seed_note(fig, seed_note)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(line_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_placeholder(outdir: Path, stem: str, subdir: str, seed_note: Optional[str] = None) -> None:
    """Save a simple placeholder plot."""
    target_dir = outdir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.text(0.5, 0.5, PLACEHOLDER_TEXT, ha="center", va="center")
    plt.axis("off")
    _add_seed_note(plt.gcf(), seed_note)
    for ext in ("png", "svg"):
        plt.savefig(target_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close()


def _blend_overall(series_train: pd.Series, series_test: pd.Series, n_train: pd.Series, n_test: pd.Series) -> pd.Series:
    """Compute per-row overall metric weighted by sample counts; fall back to mean of available."""
    t = pd.to_numeric(series_train, errors="coerce")
    te = pd.to_numeric(series_test, errors="coerce")
    n_tr = pd.to_numeric(n_train, errors="coerce").fillna(0)
    n_te = pd.to_numeric(n_test, errors="coerce").fillna(0)
    total = n_tr + n_te
    weighted = (t * n_tr + te * n_te) / total.where(total > 0, np.nan)
    fallback = pd.concat([t, te], axis=1).mean(axis=1)
    overall = weighted.fillna(fallback)
    return overall


def _compute_split_integrated_metrics(
    mode: str,
    integration_dir: Optional[Path],
    measured_timepoints: Sequence[float],
) -> Optional[pd.DataFrame]:
    """Compute train/test integrated R2 from integration trajectories using bin-phase mapping from predicted trajectories."""
    if integration_dir is None:
        return None
    traj_path = integration_dir / f"marker_integration_trajectories_{mode}.csv"
    split_path = integration_dir / f"predicted_trajectories_{mode}.csv"
    if not traj_path.exists() or not split_path.exists():
        return None
    try:
        traj = pd.read_csv(traj_path)
        split = pd.read_csv(split_path)
    except Exception:
        return None
    if traj.empty or split.empty:
        return None
    # Map bins to phase per marker/model
    bin_phase = (
        split.dropna(subset=["GFP_bin"])
        .groupby(["marker", "model", "GFP_bin"])["phase"]
        .first()
        .to_dict()
    )
    traj["phase"] = traj.apply(
        lambda r: bin_phase.get((r.get("marker"), r.get("model"), r.get("GFP_bin"))), axis=1
    )
    traj = traj[traj["phase"].notna()]
    if traj.empty or "pred_integrated" not in traj or "obs_pERK1_2" not in traj:
        return None

    records: List[Dict[str, object]] = []
    for (marker, model, phase), group in traj.groupby(["marker", "model", "phase"]):
        g = group.copy()
        g.sort_values("timepoint", inplace=True)
        seed_val = g["seed"].iloc[0] if "seed" in g.columns else None
        obs = pd.to_numeric(g["obs_pERK1_2"], errors="coerce").to_numpy()
        pred = pd.to_numeric(g["pred_integrated"], errors="coerce").to_numpy()
        t = pd.to_numeric(g["timepoint"], errors="coerce").to_numpy()
        mask = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(t)
        if mode == "per_minute":
            mask &= np.isin(t, measured_timepoints)
        if mask.sum() < 2:
            continue
        try:
            r2 = np.corrcoef(obs[mask], pred[mask])[0, 1] ** 2
        except Exception:
            continue
        ode_r2 = np.nan
        if "pred_integrated_ode" in g:
            pred_ode = pd.to_numeric(g["pred_integrated_ode"], errors="coerce").to_numpy()
            mask_ode = mask & np.isfinite(pred_ode)
            if mode == "per_minute":
                mask_ode &= np.isin(t, measured_timepoints)
            if mask_ode.sum() >= 2:
                try:
                    ode_r2 = np.corrcoef(obs[mask_ode], pred_ode[mask_ode])[0, 1] ** 2
                except Exception:
                    ode_r2 = np.nan
        records.append(
            {
                "marker": marker,
                "group_name": marker,
                "model": model,
                "dataset_mode": mode,
                "seed": seed_val,
                "split": str(phase).title(),
                "integrated_r2": r2,
                "integrated_ode_r2": ode_r2,
            }
        )
    return pd.DataFrame(records) if records else None


def _plot_marker_trajectories(
    outdir: Path,
    mode: str,
    model: str,
    *,
    predicted_path: Optional[Path],
    integration_path: Optional[Path],
) -> None:
    """Plot per-marker, per-bin trajectories for dt, integrated, and integrated ODE."""
    if predicted_path is None or integration_path is None:
        return
    if not predicted_path.exists() or not integration_path.exists():
        return
    try:
        pred_df = pd.read_csv(predicted_path)
        integ_df = pd.read_csv(integration_path)
    except Exception:
        return
    if pred_df.empty or integ_df.empty:
        return
    pred_df = pred_df[(pred_df["model"] == model) & (pred_df.get("dataset_mode") == mode)]
    integ_df = integ_df[(integ_df["model"] == model) & (integ_df.get("dataset_mode") == mode)]
    if pred_df.empty or integ_df.empty:
        return

    def _safe_save(fig: plt.Figure, path: Path) -> None:
        try:
            fig.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as exc:
            print(f"[plot_metrics_summary] Failed to save {path}: {exc}", flush=True)

    seeds_available: List[str] = []
    for cand in ("seed", "random_state"):
        if cand in pred_df.columns:
            seeds_available = [str(s) for s in pred_df[cand].dropna().unique()]
            break
    seed_note = _format_seed_label(seeds_available, averaged=False) or "Seed: single"

    markers = sorted(pred_df["marker"].unique())
    root = outdir / "trajectories" / model.lower().replace(" ", "_") / mode
    for marker in tqdm(markers, desc=f"{model} {mode} trajectories"):
        marker_dir = root / marker
        marker_dir.mkdir(parents=True, exist_ok=True)
        pred_marker = pred_df[pred_df["marker"] == marker].copy()
        integ_marker = integ_df[integ_df["marker"] == marker].copy()
        bins = sorted(pred_marker["GFP_bin"].dropna().unique().tolist())
        if not bins:
            continue
        ncols = 3
        nrows = int(np.ceil(len(bins) / ncols))

        # dt plot
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
        for ax, b in zip(axes.flat, bins):
            g = pred_marker[pred_marker["GFP_bin"] == b].copy()
            g.sort_values("timepoint", inplace=True)
            ax.plot(
                g["timepoint"],
                g["p-ERK1-2_dt_true"],
                label="dt true",
                marker="o",
                markersize=5,
                color="#2c3e50",
            )
            ax.plot(
                g["timepoint"],
                g["p-ERK1-2_dt_pred"],
                label="dt pred",
                marker="x",
                markersize=5,
                linestyle="--",
                color="#e74c3c",
            )
            ax.set_title(f"GFP bin {b}")
            ax.set_xlabel("Time")
            ax.set_ylabel("p-ERK1/2 dt")
            ax.grid(True, linestyle="--", alpha=0.3)
        for ax in axes.flat[len(bins) :]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{marker} | {model} {mode} dt")
        _add_seed_note(fig, seed_note)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        _safe_save(fig, marker_dir / "dt.png")
        _safe_save(fig, marker_dir / "dt.svg")
        plt.close(fig)

        # Integrated plots
        for y_col, title_suffix, fname in (
            ("pred_integrated", "Integrated", "integrated"),
            ("pred_integrated_ode", "Integrated ODE", "integrated_ode"),
        ):
            if y_col not in integ_marker:
                continue
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
            for ax, b in zip(axes.flat, bins):
                g = integ_marker[integ_marker["GFP_bin"] == b].copy()
                g.sort_values("timepoint", inplace=True)
                ax.plot(
                    g["timepoint"],
                    g["obs_pERK1_2"],
                    label="obs pERK",
                    marker="o",
                    markersize=5,
                    color="#2c3e50",
                )
                ax.plot(
                    g["timepoint"],
                    g[y_col],
                    label=title_suffix,
                    marker="x",
                    markersize=5,
                    linestyle="--",
                    color="#e74c3c",
                )
                ax.set_title(f"GFP bin {b}")
                ax.set_xlabel("Time")
                ax.set_ylabel("p-ERK1/2")
                ax.grid(True, linestyle="--", alpha=0.3)
            for ax in axes.flat[len(bins) :]:
                ax.axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right")
            fig.suptitle(f"{marker} | {model} {mode} {title_suffix}")
            _add_seed_note(fig, seed_note)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            _safe_save(fig, marker_dir / f"{fname}.png")
            _safe_save(fig, marker_dir / f"{fname}.svg")
            plt.close(fig)


def _select_worst_markers(
    metrics_df: pd.DataFrame,
    model: str,
    dt_threshold: float,
    ode_threshold: float,
    gap_threshold: float,
    max_rows: int,
) -> pd.DataFrame:
    df = metrics_df.copy()
    if df.empty or "dt_r2" not in df:
        return pd.DataFrame()
    df = df[df["model"] == model]
    if df.empty:
        return pd.DataFrame()
    if "ode_integ_r2_median" in df.columns:
        df = df.rename(columns={"ode_integ_r2_median": "integrated_ode_r2"})
    if "integrated_ode_r2" not in df.columns:
        return pd.DataFrame()
    df = df[df["dt_r2"].notna() & df["integrated_ode_r2"].notna()]
    if df.empty:
        return pd.DataFrame()
    df["ode_gap"] = df["dt_r2"] - df["integrated_ode_r2"]
    mask = df["dt_r2"] >= dt_threshold
    if np.isfinite(ode_threshold):
        mask &= df["integrated_ode_r2"] <= ode_threshold
    if np.isfinite(gap_threshold):
        mask &= df["ode_gap"] >= gap_threshold
    df = df[mask]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["integrated_ode_r2", "ode_gap"], ascending=[True, False])
    return df.head(max_rows)


def _plot_marker_diagnostic_panel(
    outdir: Path,
    marker_row: pd.Series,
    traj_df: pd.DataFrame,
    mode: str,
    model: str,
) -> None:
    marker = marker_row.get("marker") or marker_row.get("group_name")
    if not isinstance(marker, str) or not marker:
        return
    sub = traj_df[traj_df["marker"] == marker].copy()
    sub = sub[sub["model"] == model]
    if "dataset_mode" in sub.columns:
        sub = sub[sub["dataset_mode"] == mode]
    if sub.empty:
        return
    if "pred_integrated_ode" not in sub.columns or "obs_pERK1_2" not in sub.columns:
        return
    sub["GFP_bin"] = sub.get("GFP_bin")
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    if not bins:
        sub["__bin"] = "all"
        bins = ["all"]
    else:
        sub["__bin"] = sub["GFP_bin"].astype(str)
    colors = sns.color_palette("tab10", max(len(bins), 1))
    colour_map = dict(zip([str(b) for b in bins], colors))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    state_min = marker_row.get("ode_state_min")
    state_max = marker_row.get("ode_state_max")
    deriv_min = marker_row.get("ode_deriv_min")
    deriv_max = marker_row.get("ode_deriv_max")
    state_span = (state_min, state_max) if pd.notna(state_min) and pd.notna(state_max) else None

    for bin_label, color in colour_map.items():
        g = sub[sub["__bin"] == bin_label].copy()
        g.sort_values("timepoint", inplace=True)
        axes[0].plot(
            g["timepoint"],
            g["obs_pERK1_2"],
            marker="o",
            markersize=4.5,
            linewidth=1.2,
            color=color,
            alpha=0.9,
            label=f"bin {bin_label}",
        )
        axes[1].plot(
            g["timepoint"],
            g["pred_integrated_ode"],
            marker="x",
            markersize=5,
            linewidth=1.6,
            linestyle="--",
            color=color,
            alpha=0.9,
        )
        if "pred_integrated" in g.columns:
            axes[1].plot(
                g["timepoint"],
                g["pred_integrated"],
                marker=".",
                markersize=4,
                linewidth=1.0,
                linestyle=":",
                color=color,
                alpha=0.6,
            )
    for ax in axes[:2]:
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("p-ERK1/2")
        ax.grid(True, linestyle="--", alpha=0.3)
        if state_span:
            ax.axhspan(state_span[0], state_span[1], color="#fef0d9", alpha=0.35)
    axes[0].set_title("Observed p(t)")
    axes[1].set_title("Integrated trajectories")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, framealpha=0.9, fontsize=8)

    scatter_ax = axes[2]
    scatter_ax.set_title("d(state)/dt vs predicted p")
    if "pred_dt_ode" in sub.columns:
        scatter_data = sub[["pred_integrated_ode", "pred_dt_ode", "timepoint"]].dropna()
        if not scatter_data.empty:
            sc = scatter_ax.scatter(
                scatter_data["pred_integrated_ode"],
                scatter_data["pred_dt_ode"],
                c=scatter_data["timepoint"],
                cmap="viridis",
                s=35,
                alpha=0.85,
                edgecolor="none",
            )
            cbar = fig.colorbar(sc, ax=scatter_ax)
            cbar.set_label("Time (min)")
    scatter_ax.set_xlabel("Predicted p (ODE)")
    scatter_ax.set_ylabel("Predicted d(state)/dt")
    scatter_ax.grid(True, linestyle="--", alpha=0.3)
    if state_span:
        scatter_ax.axvspan(state_span[0], state_span[1], color="#fef0d9", alpha=0.35)
    if pd.notna(deriv_min):
        scatter_ax.axhline(deriv_min, color="#b30000", linestyle="--", alpha=0.4, linewidth=1.2)
    if pd.notna(deriv_max):
        scatter_ax.axhline(deriv_max, color="#b30000", linestyle="--", alpha=0.4, linewidth=1.2)

    dt_r2 = marker_row.get("dt_r2")
    ode_r2 = marker_row.get("integrated_ode_r2", marker_row.get("ode_integ_r2_median"))
    gap = marker_row.get("ode_gap")
    subtitle_parts = []
    if pd.notna(dt_r2):
        subtitle_parts.append(f"dt R²={dt_r2:.2f}")
    if pd.notna(ode_r2):
        subtitle_parts.append(f"ODE R²={ode_r2:.2f}")
    if pd.notna(gap):
        subtitle_parts.append(f"gap={gap:.2f}")
    subtitle = " | ".join(subtitle_parts)
    fig.suptitle(f"{marker} | {model} {mode} diagnostics{(' — ' + subtitle) if subtitle else ''}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    diag_dir = outdir / "diagnostics" / mode
    diag_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{marker}_{model}_{mode}".replace(" ", "_")
    for ext in ("png", "svg"):
        fig.savefig(diag_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_worst_ode_diagnostics(
    outdir: Path,
    integration_dir: Optional[Path],
    trajectories_dir: Optional[Path],
    *,
    max_markers: int,
    dt_threshold: float,
    ode_threshold: float,
    gap_threshold: float,
    model: str,
) -> None:
    if integration_dir is None or trajectories_dir is None or max_markers <= 0:
        return
    integration_dir = Path(integration_dir)
    trajectories_dir = Path(trajectories_dir)
    for mode in ("snapshot", "per_minute"):
        metrics_path = integration_dir / f"marker_integration_metrics_{mode}.csv"
        traj_path = trajectories_dir / f"marker_integration_trajectories_{mode}.csv"
        if not metrics_path.exists() or not traj_path.exists():
            continue
        try:
            metrics_df = pd.read_csv(metrics_path)
            traj_df = pd.read_csv(traj_path)
        except Exception:
            continue
        if metrics_df.empty or traj_df.empty:
            continue
        candidates = _select_worst_markers(
            metrics_df,
            model,
            dt_threshold,
            ode_threshold,
            gap_threshold,
            max_markers,
        )
        if candidates.empty:
            continue
        for _, marker_row in candidates.iterrows():
            _plot_marker_diagnostic_panel(outdir, marker_row, traj_df, mode, model)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    denom = np.var(y_true_f)
    if denom <= 0:
        return float("nan")
    ss_res = float(np.sum((y_true_f - y_pred_f) ** 2))
    ss_tot = float(np.sum((y_true_f - np.mean(y_true_f)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _plot_binwise_perk_traj(
    outdir: Path,
    trajectories_dir: Optional[Path],
    markers: Sequence[str],
    models: Sequence[str],
    mode: str,
    measured_timepoints: Optional[Sequence[float]] = None,
) -> None:
    if trajectories_dir is None:
        return
    traj_path = Path(trajectories_dir) / f"marker_integration_trajectories_{mode}.csv"
    if not traj_path.exists():
        return
    try:
        traj = pd.read_csv(traj_path)
    except Exception:
        return
    if traj.empty:
        return

    markers_norm = {str(m).strip().upper() for m in markers if str(m).strip()}
    pred_cols = ["pred_integrated_ode", "pred_integrated"]
    obs_candidates = ["obs_pERK1_2", "p-ERK1-2", "p_ERK1_2"]

    def _pick_obs_col(df: pd.DataFrame) -> Optional[str]:
        for c in obs_candidates:
            if c in df.columns:
                return c
        return None

    for model in models:
        df_model = traj[traj.get("model") == model]
        if "dataset_mode" in df_model.columns:
            df_model = df_model[df_model["dataset_mode"] == mode]
        if df_model.empty:
            continue
        seeds = sorted(df_model.get("seed", pd.Series([])).dropna().unique().tolist())
        for seed in seeds:
            df_seed = df_model[df_model.get("seed") == seed]
            if df_seed.empty:
                continue
            for marker in markers_norm:
                sub = df_seed[df_seed["marker"].str.upper() == marker]
                if sub.empty:
                    continue
                obs_col = _pick_obs_col(sub)
                pred_col = next((c for c in pred_cols if c in sub.columns), None)
                if not obs_col or not pred_col:
                    continue
                bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
                if not bins:
                    continue
                min_bin = min(bins)
                max_bin = max(bins)
                strength_targets = [
                    (0.0, "none"),
                    (1.0 / 3.0, "low"),
                    (2.0 / 3.0, "high"),
                    (1.0, "maximal"),
                ]

                def _bin_fraction(bin_value: float) -> float:
                    if max_bin == min_bin:
                        return 0.0
                    return (bin_value - min_bin) / (max_bin - min_bin)

                def _strength_label_for_bin(bin_value: float) -> str:
                    frac = _bin_fraction(bin_value)
                    distances = [abs(frac - t) for t, _ in strength_targets]
                    min_dist = min(distances)
                    candidates = [pair for pair, d in zip(strength_targets, distances) if d == min_dist]
                    # Prefer higher target on ties
                    target, label = max(candidates, key=lambda p: p[0])
                    return label

                cmap = plt.colormaps.get_cmap("viridis").resampled(max(1, len(bins)))
                bin_colors = {b: cmap(idx) for idx, b in enumerate(bins)}
                n_bins = len(bins)
                ncols = 2
                nrows = int(np.ceil(n_bins / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(6.0, 1.7 * nrows), sharey=True)
                axes = np.atleast_1d(axes).flatten()
                for ax in axes[n_bins:]:
                    ax.axis("off")

                # Collect global y-limits across bins for consistent scaling
                all_vals: List[float] = []
                per_bin_data = []
                for b in bins:
                    g = sub[sub["GFP_bin"] == b].copy().sort_values("timepoint")
                    t = pd.to_numeric(g.get("timepoint"), errors="coerce").to_numpy(dtype=float)
                    obs = pd.to_numeric(g[obs_col], errors="coerce").to_numpy(dtype=float)
                    pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(dtype=float)
                    if mode == "per_minute" and measured_timepoints:
                        mask = np.isin(t, measured_timepoints)
                        t, obs, pred = t[mask], obs[mask], pred[mask]
                    per_bin_data.append((b, t, obs, pred))
                    finite = np.isfinite(obs) | np.isfinite(pred)
                    if finite.any():
                        all_vals.extend(list(obs[np.isfinite(obs)]))
                        all_vals.extend(list(pred[np.isfinite(pred)]))
                y_min, y_max = None, None
                if all_vals:
                    y_min = min(all_vals)
                    y_max = max(all_vals)
                    span = y_max - y_min if y_max > y_min else 1.0
                    pad = 0.08 * span
                    y_min -= pad
                    y_max += pad
                x_min, x_max = None, None
                all_times = [t for _, t, _, _ in per_bin_data if len(t) > 0]
                if all_times:
                    flat_times = np.concatenate(all_times)
                    finite_times = flat_times[np.isfinite(flat_times)]
                    if finite_times.size:
                        x_min = float(np.min(finite_times))
                        x_max = float(np.max(finite_times))
                        span = x_max - x_min if x_max > x_min else 1.0
                        x_pad_right = 0.05 * span

                for ax, (b, t, obs, pred) in zip(axes, per_bin_data):
                    r2 = _safe_r2(obs, pred)
                    if len(t) == 0 or (not np.isfinite(obs).any() and not np.isfinite(pred).any()):
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, transform=ax.transAxes)
                        ax.axis("off")
                        continue
                    # Ensure finite ordering for connected line
                    finite_mask = np.isfinite(t) & np.isfinite(pred)
                    t_plot = t[finite_mask]
                    pred_plot = pred[finite_mask]
                    order = np.argsort(t_plot)
                    t_plot = t_plot[order]
                    pred_plot = pred_plot[order]
                    pred_color = bin_colors.get(b, "#111111")
                    ax.plot(t_plot, pred_plot, "-o", ms=3.6, lw=2.4, label="pred", alpha=0.9, color=pred_color)
                    ax.scatter(t, obs, marker="x", s=36, linewidths=1.4, color="#111111", label="obs", alpha=0.95)
                    if y_min is not None and y_max is not None:
                        ax.set_ylim(y_min, y_max)
                    if x_min is not None and x_max is not None:
                        ax.set_xlim(x_min, x_max + x_pad_right)
                    try:
                        ax.set_box_aspect(0.8)
                    except Exception:
                        pass
                    ax.grid(True, linestyle="--", alpha=0.3)
                    r2_txt = f"{r2:.2f}" if np.isfinite(r2) else "N/A"
                    ax.text(
                        0.98,
                        0.98,
                        f"R² {r2_txt}",
                        transform=ax.transAxes,
                        va="top",
                        ha="right",
                        fontsize=12,
                    )
                    ax.tick_params(labelsize=11)
                    ax.set_xlabel("Time", fontsize=11)
                    ax.set_ylabel("pERK", fontsize=11)
                if axes.size:
                    axes[0].get_legend().remove() if axes[0].get_legend() else None
                fig.suptitle(
                    f"{marker} | seed {seed} | {model} | {mode}",
                    fontsize=15,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                out_dir = outdir / "seeds" / f"seed_{seed}" / "binwise_perk" / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                secondary_dirs = []
                if trajectories_dir is not None:
                    traj_root = Path(trajectories_dir)
                    # Mirror outputs into data/experimental/seeds/… (two levels up from runs/aggregated/trajectories)
                    try:
                        seeds_root_exp = traj_root.parents[2] / "seeds"
                        print(seeds_root_exp)
                        secondary_dirs.append(seeds_root_exp / f"seed_{seed}" / "binwise_perk" / mode)
                    except Exception:
                        pass
                for sec in secondary_dirs:
                    sec.mkdir(parents=True, exist_ok=True)
                stem = f"{marker}_seed{seed}_{model}_{mode}_bin_traj".replace(" ", "_")
                for ext in ("png", "svg"):
                    fig.savefig(out_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
                    for sec in secondary_dirs:
                        try:
                            fig.savefig(sec / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
                        except Exception:
                            pass
                plt.close(fig)

                # 2x2 subset for bins closest to 0/33/66/100% (prefer higher on ties)
                bin_lookup = {b: (t, obs, pred) for b, t, obs, pred in per_bin_data}
                selected_bins = []
                selected_labels = []
                for target, label in strength_targets:
                    distances = [abs(_bin_fraction(b) - target) for b in bins]
                    if not distances:
                        continue
                    min_dist = min(distances)
                    candidates = [b for b, d in zip(bins, distances) if d == min_dist]
                    chosen = max(candidates)
                    if chosen in selected_bins:
                        remaining = [b for b in candidates if b not in selected_bins]
                        if remaining:
                            chosen = max(remaining)
                    selected_bins.append(chosen)
                    selected_labels.append(label)
                if selected_bins:
                    fig_sel, axes_sel = plt.subplots(2, 2, figsize=(5.2, 4.4), sharex=True, sharey=True)
                    axes_sel = np.atleast_1d(axes_sel).flatten()
                    for ax in axes_sel[len(selected_bins):]:
                        ax.axis("off")

                    sel_vals: List[float] = []
                    sel_times: List[float] = []
                    for b in selected_bins:
                        t, obs, pred = bin_lookup.get(b, (np.array([]), np.array([]), np.array([])))
                        if len(t) == 0:
                            continue
                        if np.isfinite(obs).any():
                            sel_vals.extend(list(obs[np.isfinite(obs)]))
                        if np.isfinite(pred).any():
                            sel_vals.extend(list(pred[np.isfinite(pred)]))
                        if np.isfinite(t).any():
                            sel_times.extend(list(t[np.isfinite(t)]))

                    y_min_sel, y_max_sel = None, None
                    if sel_vals:
                        y_min_sel = min(sel_vals)
                        y_max_sel = max(sel_vals)
                        span = y_max_sel - y_min_sel if y_max_sel > y_min_sel else 1.0
                        pad = 0.08 * span
                        y_min_sel -= pad
                        y_max_sel += pad

                    x_min_sel, x_max_sel = None, None
                    if sel_times:
                        x_min_sel = min(sel_times)
                        x_max_sel = max(sel_times)
                        span = x_max_sel - x_min_sel if x_max_sel > x_min_sel else 1.0
                        x_pad_right = 0.05 * span

                    for idx, (ax, b, label) in enumerate(zip(axes_sel, selected_bins, selected_labels)):
                        t, obs, pred = bin_lookup.get(b, (np.array([]), np.array([]), np.array([])))
                        r2 = _safe_r2(obs, pred)
                        if len(t) == 0 or (not np.isfinite(obs).any() and not np.isfinite(pred).any()):
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11, transform=ax.transAxes)
                            ax.axis("off")
                            continue
                        finite_mask = np.isfinite(t) & np.isfinite(pred)
                        t_plot = t[finite_mask]
                        pred_plot = pred[finite_mask]
                        order = np.argsort(t_plot)
                        t_plot = t_plot[order]
                        pred_plot = pred_plot[order]
                        pred_color = bin_colors.get(b, "#111111")
                        ax.plot(t_plot, pred_plot, "-o", ms=3.6, lw=2.4, label="pred", alpha=0.9, color=pred_color)
                        ax.scatter(t, obs, marker="x", s=36, linewidths=1.4, color="#111111", label="obs", alpha=0.95)
                        if y_min_sel is not None and y_max_sel is not None:
                            ax.set_ylim(y_min_sel, y_max_sel)
                        if x_min_sel is not None and x_max_sel is not None:
                            ax.set_xlim(x_min_sel, x_max_sel + x_pad_right)
                        ax.grid(True, linestyle="--", alpha=0.3)
                        r2_txt = f"{r2:.2f}" if np.isfinite(r2) else "N/A"
                        ax.text(
                            0.98,
                            0.98,
                            f"R² {r2_txt}",
                            transform=ax.transAxes,
                            va="top",
                            ha="right",
                            fontsize=12,
                        )
                        ax.tick_params(labelsize=11)
                        row = idx // 2
                        col = idx % 2
                        if row == 1:
                            ax.set_xlabel("Time", fontsize=11)
                        else:
                            ax.set_xlabel("")
                        if col == 0:
                            ax.set_ylabel("pERK", fontsize=11)
                        else:
                            ax.set_ylabel("")

                    if axes_sel.size:
                        axes_sel[0].get_legend().remove() if axes_sel[0].get_legend() else None
                    fig_sel.suptitle(
                        f"{marker} | seed {seed} | {model} | {mode} (bins 10/20/30/40)",
                        fontsize=15,
                    )
                    fig_sel.tight_layout(rect=(0, 0, 1, 0.90))
                    stem_sel = f"{marker}_seed{seed}_{model}_{mode}_bin_traj_selected".replace(" ", "_")
                    for ext in ("png", "svg"):
                        fig_sel.savefig(out_dir / f"{stem_sel}.{ext}", dpi=220, bbox_inches="tight")
                        for sec in secondary_dirs:
                            try:
                                fig_sel.savefig(sec / f"{stem_sel}.{ext}", dpi=220, bbox_inches="tight")
                            except Exception:
                                pass
                    plt.close(fig_sel)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    seed_col = None
    for cand in ("seed", "random_state"):
        if cand in df.columns:
            seed_col = cand
            break
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    scatter_dir = outdir / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    df_seed = df.copy() if seed_col else None
    seeds_seen = df[seed_col].dropna().unique() if seed_col else None
    aggregated_over_seeds = False
    inferred_seeds = _infer_seeds_from_path(args.summary)
    seed_hints: List[str] = []
    # Collect seed hints from integration metrics or trajectories if provided
    for mode in ("snapshot", "per_minute"):
        if args.integration_metrics_dir:
            metrics_path = Path(args.integration_metrics_dir) / f"marker_integration_metrics_{mode}.csv"
            seed_hints.extend(_collect_seeds_from_csv(metrics_path))
        if args.trajectories_dir:
            traj_path = Path(args.trajectories_dir) / f"predicted_trajectories_{mode}.csv"
            seed_hints.extend(_collect_seeds_from_csv(traj_path))
    summary_path_lower = str(args.summary).lower()
    if "aggregated" in summary_path_lower or "all_seeds" in args.summary.name.lower():
        aggregated_over_seeds = True
    if args.integration_metrics_dir and "aggregated" in str(args.integration_metrics_dir).lower():
        aggregated_over_seeds = True
    if not seed_col:
        summary_name = args.summary.name.lower()
        if "seed" in summary_name or "agg" in summary_name or "mean" in summary_name:
            aggregated_over_seeds = True
        if inferred_seeds:
            seeds_seen = inferred_seeds
            seed_col = "inferred_seed"
            aggregated_over_seeds = aggregated_over_seeds or len(inferred_seeds) > 1
        elif seed_hints:
            seeds_seen = seed_hints
            seed_col = "inferred_seed"
            aggregated_over_seeds = aggregated_over_seeds or len(seed_hints) > 1
    if seed_col and df_seed is None:
        df_seed = df.copy()
    if seed_col and seeds_seen is None:
        if seed_col in df_seed.columns:
            seeds_seen = df_seed[seed_col].dropna().unique()
        elif inferred_seeds:
            seeds_seen = inferred_seeds
        elif seed_hints:
            seeds_seen = seed_hints
    if seed_col:
        if seed_col in df_seed.columns:
            uniq_seeds = sorted(df_seed[seed_col].dropna().unique().tolist())
            print(f"[plot_metrics_summary] Detected seed column '{seed_col}' with {len(uniq_seeds)} seeds: {uniq_seeds}")
            keys = [k for k in ["group_name", "dataset_mode", "feature_mode", "model"] if k in df.columns]
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != seed_col]
            df_mean_num = df.groupby(keys, dropna=False)[numeric_cols].mean().reset_index()
            df_non_num = pd.DataFrame()
            if "formula" in df.columns:
                df_non_num = df.groupby(keys, dropna=False)["formula"].first().reset_index()
            if not df_non_num.empty:
                df = df_mean_num.merge(df_non_num, on=keys, how="left")
            else:
                df = df_mean_num
            aggregated_over_seeds = True
            print(f"[plot_metrics_summary] Aggregated metrics across seeds using mean per {keys}.")
        else:
            if seeds_seen is None:
                uniq_seeds = []
            elif hasattr(seeds_seen, "tolist"):
                uniq_seeds = sorted(seeds_seen.tolist())
            else:
                uniq_seeds = sorted([str(s) for s in seeds_seen])
            if uniq_seeds:
                print(f"[plot_metrics_summary] Using inferred seeds {uniq_seeds} (no seed column present).")

    # If seeds slipped through, average numeric columns across seeds before plotting.
    if "seed" in df.columns:
        keys_dt = [k for k in ("group_name", "dataset_mode", "feature_mode", "model") if k in df.columns]
        num_cols_dt = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_dt = [c for c in num_cols_dt if c != "seed"]
        if keys_dt and num_cols_dt:
            df_dt_numeric = df.groupby(keys_dt, dropna=False)[num_cols_dt].mean().reset_index()
            df_dt_non = df[keys_dt].drop_duplicates()
            df = df_dt_numeric.merge(df_dt_non, on=keys_dt, how="left")

    global DEFAULT_SEED_NOTE
    if seed_col and seeds_seen is not None and len(seeds_seen) > 0:
        if len(seeds_seen) == 1:
            DEFAULT_SEED_NOTE = _format_seed_label(seeds_seen, averaged=False) or "Seed: single"
        else:
            DEFAULT_SEED_NOTE = _format_seed_label(seeds_seen, averaged=True) or "Seed: averaged"
    elif aggregated_over_seeds:
        DEFAULT_SEED_NOTE = "Seed: averaged"
    else:
        DEFAULT_SEED_NOTE = "Seed: single"

    df["test_r2_clamped"] = df["test_r2"].clip(lower=0)
    relmae_eps = 1e-6  # avoid zeros when plotting on log axes
    zoom_thr = float(args.zoom_threshold)

    marker_type = _load_marker_classes(args.group_definitions_csv)
    df["total_samples"] = df["train_samples"].fillna(0) + df["test_samples_raw"].fillna(0)
    df["marker_class"] = df["group_name"].map(marker_type).fillna("neutral")
    samples_by_mode_all = _compute_samples_by_mode(df)

    saved_stems = set()
    integration_frames: List[pd.DataFrame] = []
    integration_split_frames: List[pd.DataFrame] = []
    integration_seed_notes: Dict[str, Optional[str]] = {}

    planned_steps: List[str] = [
        "Model scatter/KDE",
        "Integration scatter (dt vs integrated)",
        "Integration lines/boxes",
        "R2 line/box plots",
        "Feature usage/importance",
        "PySR symbol counts",
        "Seed variance/lines",
        "Binwise pERK scatters",
        "Worst ODE diagnostics",
        "Marker trajectory overlays",
    ]
    progress_log, _ = _progress_printer(planned_steps)
    print("[plot_metrics_summary] Planned plot tasks:", flush=True)
    for idx, label in enumerate(planned_steps, start=1):
        print(f"[plot_metrics_summary]   {idx}/{len(planned_steps)} {label}", flush=True)

    # Model comparison: PySR vs Linear Regression
    progress_log("Model scatter/KDE")
    models = ["PySR", "Linear Regression"]
    df_models = df[df["model"].isin(models)].copy()
    if not df_models.empty:
        samples_by_mode_models = _compute_samples_by_mode(df_models)
        samples_models_df = samples_by_mode_models.rename("total_samples").reset_index()
        meta_models = (
            df_models.groupby(["group_name", "dataset_mode", "feature_mode"])
            .agg(marker_class=("marker_class", "first"))
            .reset_index()
            .merge(samples_models_df, on=["group_name", "dataset_mode"], how="left")
        )
        pivot = (
            df_models.pivot_table(
                index=["group_name", "dataset_mode", "feature_mode"],
                columns="model",
                values=["test_r2_clamped", "test_relative_mae"],
                dropna=False,
            )
        )
        if not pivot.empty:
            flat = _flatten_columns(pivot).reset_index()
            flat = flat.rename(
                columns={
                    "test_r2_clamped_Linear Regression": "LinReg_r2",
                    "test_r2_clamped_PySR": "PySR_r2",
                    "test_relative_mae_Linear Regression": "LinReg_relmae",
                    "test_relative_mae_PySR": "PySR_relmae",
                }
            )
            flat = flat.merge(meta_models, on=["group_name", "dataset_mode", "feature_mode"], how="left")
            flat["marker_class"] = flat["marker_class"].fillna("neutral")
            keep_cols = ["LinReg_r2", "PySR_r2", "LinReg_relmae", "PySR_relmae"]
            available = [c for c in keep_cols if c in flat.columns]
            if available:
                wide_models = flat.dropna(subset=available, how="all")
                if not wide_models.empty:
                    scatter_mode_label: Optional[str] = None
                    scatter_feature_label: Optional[str] = None
                    if "dataset_mode" in wide_models.columns:
                        mode_options = [m for m in wide_models["dataset_mode"].dropna().unique().tolist() if str(m).strip()]
                        preferred_mode = args.model_scatter_dataset_mode
                        match_mode: Optional[object] = None
                        if mode_options:
                            match_mode = _match_preferred_label(mode_options, preferred_mode)
                            if match_mode is None and preferred_mode:
                                match_mode = mode_options[0]
                                print(
                                    f"[plot_metrics_summary] Requested model scatter dataset_mode '{preferred_mode}' "
                                    f"not found; using '{match_mode}'."
                                )
                        if match_mode is not None:
                            scatter_mode_label = str(match_mode)
                            wide_models = wide_models[wide_models["dataset_mode"] == match_mode]
                    if "feature_mode" in wide_models.columns:
                        feature_options = [f for f in wide_models["feature_mode"].dropna().unique().tolist() if str(f).strip()]
                        preferred_feature = args.model_scatter_feature_mode
                        if preferred_feature is None and len(feature_options) > 1:
                            preferred_feature = "all"
                        match_feature: Optional[object] = None
                        if feature_options:
                            match_feature = _match_preferred_label(feature_options, preferred_feature)
                            if match_feature is None and preferred_feature:
                                match_feature = feature_options[0]
                                print(
                                    f"[plot_metrics_summary] Requested model scatter feature_mode '{preferred_feature}' "
                                    f"not found; using '{match_feature}'."
                                )
                            elif match_feature is None and len(feature_options) > 1:
                                match_feature = feature_options[0]
                                print(
                                    "[plot_metrics_summary] Multiple feature modes detected "
                                    f"({feature_options}); defaulting to '{match_feature}' for model comparison scatter."
                                )
                            elif match_feature is None and len(feature_options) == 1:
                                match_feature = feature_options[0]
                        if match_feature is not None:
                            scatter_feature_label = str(match_feature)
                            wide_models = wide_models[
                                wide_models["feature_mode"].astype(str).str.lower()
                                == scatter_feature_label.strip().lower()
                            ]
                    if not wide_models.empty:
                        title_suffix_parts = []
                        if scatter_mode_label:
                            title_suffix_parts.append(f"mode={scatter_mode_label}")
                        if scatter_feature_label:
                            title_suffix_parts.append(f"features={scatter_feature_label}")
                        title_suffix = " | ".join(title_suffix_parts)
                        r2_title = "Test R2: LinReg vs PySR"
                        relmae_title = "relMAE: LinReg vs PySR"
                        if title_suffix:
                            r2_title = f"{r2_title} ({title_suffix})"
                            relmae_title = f"{relmae_title} ({title_suffix})"
                        saved_stems.add("metrics_models_scatter")
                        _plot_scatter_variants(
                            scatter_dir,
                            wide_models,
                            "LinReg_r2",
                            "PySR_r2",
                            r2_title,
                            "metrics_models_scatter_r2",
                            zoom_thr,
                            zoom=False,
                            metric_group="dt",
                        )
                        _plot_scatter_variants(
                            scatter_dir,
                            wide_models,
                            "LinReg_r2",
                            "PySR_r2",
                            r2_title,
                            "metrics_models_scatter_r2",
                            zoom_thr,
                            zoom=True,
                            metric_group="dt",
                        )
                        _copy_preferred_variant_to_base(scatter_dir, "metrics_models_scatter_r2")
                        _plot_scatter_variants(
                            scatter_dir,
                            wide_models.assign(
                                LinReg_relmae=wide_models["LinReg_relmae"].clip(lower=relmae_eps),
                                PySR_relmae=wide_models["PySR_relmae"].clip(lower=relmae_eps),
                            ),
                            "LinReg_relmae",
                            "PySR_relmae",
                            relmae_title,
                            "metrics_models_scatter_relmae",
                            zoom_thr,
                            zoom=False,
                            log_axes=True,
                            draw_threshold=False,
                            metric_group="dt",
                        )
                        _plot_scatter_variants(
                            scatter_dir,
                            wide_models.assign(
                                LinReg_relmae=wide_models["LinReg_relmae"].clip(lower=relmae_eps),
                                PySR_relmae=wide_models["PySR_relmae"].clip(lower=relmae_eps),
                            ),
                            "LinReg_relmae",
                            "PySR_relmae",
                            relmae_title,
                            "metrics_models_scatter_relmae",
                            zoom_thr,
                            zoom=True,
                            log_axes=True,
                            draw_threshold=False,
                            metric_group="dt",
                        )
                        _copy_preferred_variant_to_base(scatter_dir, "metrics_models_scatter_relmae")

    # PySR snapshot vs per_minute
    df_pysr = df[df["model"] == "PySR"].copy()
    if not df_pysr.empty:
        wide_pysr = (
            df_pysr.pivot_table(
                index=["group_name", "feature_mode"],
                columns="dataset_mode",
                values=["test_r2_clamped", "test_relative_mae"],
                dropna=False,
            )
        )
        if not wide_pysr.empty:
            flat = _flatten_columns(wide_pysr).reset_index()
            flat = flat.rename(
                columns={
                    "test_r2_clamped_snapshot": "snapshot_r2",
                    "test_r2_clamped_per_minute": "per_minute_r2",
                    "test_relative_mae_snapshot": "snapshot_relmae",
                    "test_relative_mae_per_minute": "per_minute_relmae",
                }
            )
            flat["marker_class"] = flat["group_name"].map(marker_type).fillna("neutral")
            samples_pysr = _compute_samples_by_mode(df_pysr)
            samples_pysr_wide = samples_pysr.unstack() if not samples_pysr.empty else pd.DataFrame()
            snapshot_samples = samples_pysr_wide.get("snapshot")
            per_minute_samples = samples_pysr_wide.get("per_minute")
            flat["total_samples"] = np.nan
            if per_minute_samples is not None:
                flat["total_samples"] = flat["group_name"].map(per_minute_samples)
            if snapshot_samples is not None:
                flat["total_samples"] = flat["total_samples"].fillna(
                    flat["group_name"].map(snapshot_samples)
                )
            needed = ["snapshot_r2", "per_minute_r2", "snapshot_relmae", "per_minute_relmae"]
            available_subset = [c for c in needed if c in flat.columns]
            if available_subset:
                flat = flat.dropna(subset=available_subset, how="all")
                if not flat.empty:
                    saved_stems.add("metrics_pysr_modes")
                    _plot_scatter_variants(
                        scatter_dir,
                        flat,
                        "snapshot_r2",
                        "per_minute_r2",
                        "PySR: snapshot vs per_minute",
                        "metrics_pysr_modes_r2",
                        zoom_thr,
                        zoom=False,
                        metric_group="dt",
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat,
                        "snapshot_r2",
                        "per_minute_r2",
                        "PySR: snapshot vs per_minute",
                        "metrics_pysr_modes_r2",
                        zoom_thr,
                        zoom=True,
                        metric_group="dt",
                    )
                    _copy_preferred_variant_to_base(scatter_dir, "metrics_pysr_modes_r2")
                    flat_rel = flat.assign(
                        snapshot_relmae=flat["snapshot_relmae"].clip(lower=relmae_eps),
                        per_minute_relmae=flat["per_minute_relmae"].clip(lower=relmae_eps),
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat_rel,
                        "snapshot_relmae",
                        "per_minute_relmae",
                        "PySR relMAE: snapshot vs per_minute",
                        "metrics_pysr_modes_relmae",
                        zoom_thr,
                        zoom=False,
                        log_axes=True,
                        draw_threshold=False,
                        metric_group="dt",
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat_rel,
                        "snapshot_relmae",
                        "per_minute_relmae",
                        "PySR relMAE: snapshot vs per_minute",
                        "metrics_pysr_modes_relmae",
                        zoom_thr,
                        zoom=True,
                        log_axes=True,
                        draw_threshold=False,
                        metric_group="dt",
                    )
                    _copy_preferred_variant_to_base(scatter_dir, "metrics_pysr_modes_relmae")

    # LinReg snapshot vs per_minute
    df_linreg = df[df["model"] == "Linear Regression"].copy()
    if not df_linreg.empty:
        wide_lin = (
            df_linreg.pivot_table(
                index=["group_name", "feature_mode"],
                columns="dataset_mode",
                values=["test_r2_clamped", "test_relative_mae"],
                dropna=False,
            )
        )
        if not wide_lin.empty:
            flat = _flatten_columns(wide_lin).reset_index()
            flat = flat.rename(
                columns={
                    "test_r2_clamped_snapshot": "snapshot_r2",
                    "test_r2_clamped_per_minute": "per_minute_r2",
                    "test_relative_mae_snapshot": "snapshot_relmae",
                    "test_relative_mae_per_minute": "per_minute_relmae",
                }
            )
            flat["marker_class"] = flat["group_name"].map(marker_type).fillna("neutral")
            samples_linreg = _compute_samples_by_mode(df_linreg)
            samples_linreg_wide = samples_linreg.unstack() if not samples_linreg.empty else pd.DataFrame()
            snapshot_samples_lin = samples_linreg_wide.get("snapshot")
            per_minute_samples_lin = samples_linreg_wide.get("per_minute")
            flat["total_samples"] = np.nan
            if per_minute_samples_lin is not None:
                flat["total_samples"] = flat["group_name"].map(per_minute_samples_lin)
            if snapshot_samples_lin is not None:
                flat["total_samples"] = flat["total_samples"].fillna(
                    flat["group_name"].map(snapshot_samples_lin)
                )
            needed = ["snapshot_r2", "per_minute_r2", "snapshot_relmae", "per_minute_relmae"]
            available_subset = [c for c in needed if c in flat.columns]
            if available_subset:
                flat = flat.dropna(subset=available_subset, how="all")
                if not flat.empty:
                    saved_stems.add("metrics_linreg_modes")
                    _plot_scatter_variants(
                        scatter_dir,
                        flat,
                        "snapshot_r2",
                        "per_minute_r2",
                        "Linear Regression: snapshot vs per_minute",
                        "metrics_linreg_modes_r2",
                        zoom_thr,
                        zoom=False,
                        metric_group="dt",
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat,
                        "snapshot_r2",
                        "per_minute_r2",
                        "Linear Regression: snapshot vs per_minute",
                        "metrics_linreg_modes_r2",
                        zoom_thr,
                        zoom=True,
                        metric_group="dt",
                    )
                    _copy_preferred_variant_to_base(scatter_dir, "metrics_linreg_modes_r2")
                    flat_rel = flat.assign(
                        snapshot_relmae=flat["snapshot_relmae"].clip(lower=relmae_eps),
                        per_minute_relmae=flat["per_minute_relmae"].clip(lower=relmae_eps),
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat_rel,
                        "snapshot_relmae",
                        "per_minute_relmae",
                        "Linear Regression relMAE: snapshot vs per_minute",
                        "metrics_linreg_modes_relmae",
                        zoom_thr,
                        zoom=False,
                        log_axes=True,
                        draw_threshold=False,
                        metric_group="dt",
                    )
                    _plot_scatter_variants(
                        scatter_dir,
                        flat_rel,
                        "snapshot_relmae",
                        "per_minute_relmae",
                        "Linear Regression relMAE: snapshot vs per_minute",
                        "metrics_linreg_modes_relmae",
                        zoom_thr,
                        zoom=True,
                        log_axes=True,
                        draw_threshold=False,
                        metric_group="dt",
                    )
                    _copy_preferred_variant_to_base(scatter_dir, "metrics_linreg_modes_relmae")

    # Integrated R2 vs dt R2 (PySR only)
    progress_log("Integration scatter (dt vs integrated)")
    def _load_integration_metrics(mode: str) -> Optional[pd.DataFrame]:
        if args.integration_metrics_dir is None:
            return None
        path = Path(args.integration_metrics_dir) / f"marker_integration_metrics_{mode}.csv"
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    for mode in tqdm(("snapshot", "per_minute"), desc="Integration modes"):
        if mode == "snapshot":
            print("[plot_metrics_summary] Skipping snapshot integration scatter/lines (disabled).", flush=True)
            continue
        df_int = _load_integration_metrics(mode)
        if df_int is None or df_int.empty:
            continue
        if "dt_r2" not in df_int or "integ_r2_median" not in df_int:
            continue
        df_int = df_int.rename(columns={"integ_r2_median": "integrated_r2"})
        seeds_int: List[object] = []
        if "dataset_mode" not in df_int.columns:
            df_int["dataset_mode"] = mode
        seed_col_int = None
        for cand in ("seed", "random_state"):
            if cand in df_int.columns:
                seed_col_int = cand
                break
        if seed_col_int is not None:
            seeds_int = df_int[seed_col_int].dropna().unique().tolist()
            group_keys = [k for k in ("marker", "model", "dataset_mode") if k in df_int.columns]
            num_cols = df_int.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != seed_col_int]
            if group_keys and num_cols:
                df_numeric = df_int.groupby(group_keys, dropna=False)[num_cols].mean().reset_index()
                df_non_num = df_int[group_keys + [seed_col_int]].groupby(group_keys, dropna=False).first().reset_index()
                df_int = df_numeric.merge(df_non_num[group_keys], on=group_keys, how="left")
        integration_seed_notes[mode] = _format_seed_label(seeds_int, averaged=bool(seeds_int)) or DEFAULT_SEED_NOTE
        df_int["marker_class"] = df_int["marker"].map(marker_type).fillna("neutral")
        mode_samples = pd.Series(dtype=float)
        if not samples_by_mode_all.empty:
            try:
                mode_samples = samples_by_mode_all.xs(mode, level="dataset_mode")
            except KeyError:
                mode_samples = pd.Series(dtype=float)
        df_int["total_samples"] = df_int["marker"].map(mode_samples).fillna(0)
        df_int["group_name"] = df_int["marker"]
        integration_frames.append(df_int.copy())
        split_df = _compute_split_integrated_metrics(mode, args.integration_metrics_dir, args.measured_timepoints)
        if split_df is not None:
            integration_split_frames.append(split_df)

        for model_name, sub_title in (("PySR", "PySR"), ("Linear Regression", "Linear Regression")):
            sub = df_int[df_int["model"] == model_name]
            if sub.empty:
                continue
            saved_stems.add(f"metrics_integrated_{mode}_{model_name}")
            _plot_scatter_variants(
                scatter_dir,
                sub,
                "dt_r2",
                "integrated_r2",
                f"{sub_title}: dt R2 vs integrated R2 ({mode})",
                f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                zoom_thr,
                zoom=False,
                metric_group="integrated",
                seed_note=integration_seed_notes.get(mode),
            )
            _plot_scatter_variants(
                scatter_dir,
                sub,
                "dt_r2",
                "integrated_r2",
                f"{sub_title}: dt R2 vs integrated R2 ({mode})",
                f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                zoom_thr,
                zoom=True,
                metric_group="integrated",
                seed_note=integration_seed_notes.get(mode),
            )
            _copy_preferred_variant_to_base(
                scatter_dir,
                f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                metric_group="integrated",
            )

            # relMAE variants (dt vs integrated)
            if {
                "dt_rel_mae_mean_bins",
                "integ_rel_mae_median_bins",
            }.issubset(sub.columns):
                _plot_scatter_variants(
                    scatter_dir,
                    sub,
                    "dt_rel_mae_mean_bins",
                    "integ_rel_mae_median_bins",
                    f"{sub_title}: dt relMAE (mean bins) vs integrated relMAE (median bins) ({mode})",
                    f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_relmae",
                    zoom_thr,
                    zoom=False,
                    log_axes=True,
                    draw_threshold=False,
                    metric_group="integrated_relmae",
                    seed_note=integration_seed_notes.get(mode),
                )
                _plot_scatter_variants(
                    scatter_dir,
                    sub,
                    "dt_rel_mae_mean_bins",
                    "integ_rel_mae_median_bins",
                    f"{sub_title}: dt relMAE (mean bins) vs integrated relMAE (median bins) ({mode})",
                    f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_relmae",
                    zoom_thr,
                    zoom=True,
                    log_axes=True,
                    draw_threshold=False,
                    metric_group="integrated_relmae",
                    seed_note=integration_seed_notes.get(mode),
                )
                _copy_preferred_variant_to_base(
                    scatter_dir,
                    f"metrics_integrated_{mode}_{model_name.lower().replace(' ', '_')}_relmae",
                    metric_group="integrated_relmae",
                )

            if "ode_integ_r2_median" in sub.columns:
                sub_ode = sub.rename(columns={"ode_integ_r2_median": "integrated_ode_r2"}).copy()
                _plot_scatter_variants(
                    scatter_dir,
                    sub_ode,
                    "dt_r2",
                    "integrated_ode_r2",
                    f"{sub_title}: dt R2 vs integrated ODE R2 ({mode})",
                    f"metrics_integrated_ode_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    zoom_thr,
                    zoom=False,
                    metric_group="integrated",
                    seed_note=integration_seed_notes.get(mode),
                )
                _plot_scatter_variants(
                    scatter_dir,
                    sub_ode,
                    "dt_r2",
                    "integrated_ode_r2",
                    f"{sub_title}: dt R2 vs integrated ODE R2 ({mode})",
                    f"metrics_integrated_ode_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    zoom_thr,
                    zoom=True,
                    metric_group="integrated",
                    seed_note=integration_seed_notes.get(mode),
                )
                _copy_preferred_variant_to_base(
                    scatter_dir,
                    f"metrics_integrated_ode_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    metric_group="integrated",
                )

                # Compare feature vs ODE integration
                _plot_scatter_variants(
                    scatter_dir,
                    sub_ode,
                    "integrated_r2",
                    "integrated_ode_r2",
                    f"{sub_title}: integrated vs integrated ODE R2 ({mode})",
                    f"metrics_integrated_compare_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    zoom_thr,
                    zoom=False,
                    metric_group="integrated",
                    seed_note=integration_seed_notes.get(mode),
                )
                _plot_scatter_variants(
                    scatter_dir,
                    sub_ode,
                    "integrated_r2",
                    "integrated_ode_r2",
                    f"{sub_title}: integrated vs integrated ODE R2 ({mode})",
                    f"metrics_integrated_compare_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    zoom_thr,
                    zoom=True,
                    metric_group="integrated",
                    seed_note=integration_seed_notes.get(mode),
                )
                _copy_preferred_variant_to_base(
                    scatter_dir,
                    f"metrics_integrated_compare_{mode}_{model_name.lower().replace(' ', '_')}_r2",
                    metric_group="integrated",
                )

    integrated_line_done = False
    integrated_box_done = False
    integrated_ode_line_done = False
    integrated_ode_box_done = False

    # Integrated line/box plots (feature integration)
    progress_log("Integration lines/boxes")
    if integration_frames:
        df_int_all = pd.concat(integration_frames, ignore_index=True)
        if "seed" in df_int_all.columns:
            keys_int = [k for k in ("marker", "group_name", "model", "dataset_mode") if k in df_int_all.columns]
            num_cols_int = df_int_all.select_dtypes(include=[np.number]).columns.tolist()
            num_cols_int = [c for c in num_cols_int if c != "seed"]
            if keys_int and num_cols_int:
                df_int_all = df_int_all.groupby(keys_int, dropna=False)[num_cols_int].mean().reset_index()

        if "seed" in df_int_all.columns:
            seed_note_int_all = _format_seed_label(df_int_all["seed"].dropna().unique(), averaged=True) or DEFAULT_SEED_NOTE
        else:
            seed_note_int_all = DEFAULT_SEED_NOTE
        if "integrated_r2" in df_int_all.columns:
            _plot_metric_lineplot(
                df_int_all,
                outdir,
                marker_type,
                "integrated_r2",
                "metrics_integrated_r2_by_marker",
                "Integrated R2",
                seed_note=seed_note_int_all,
            )
            integrated_line_done = True
            series_defs_int = [("Overall", df_int_all["integrated_r2"])]
            _plot_metric_boxplot(
                df_int_all,
                outdir,
                series_defs_int,
                "metrics_integrated_r2_boxplot",
                "Integrated R2 (clipped at 0)",
                seed_note=seed_note_int_all,
            )
            integrated_box_done = True
        if "integ_rel_mae_median_bins" in df_int_all.columns:
            _plot_metric_lineplot(
                df_int_all,
                outdir,
                marker_type,
                "integ_rel_mae_median_bins",
                "metrics_integrated_relmae_by_marker",
                "Integrated relMAE (median bins)",
                seed_note=seed_note_int_all,
            )
            series_defs_rel = [("Overall", df_int_all["integ_rel_mae_median_bins"])]
            _plot_metric_boxplot(
                df_int_all,
                outdir,
                series_defs_rel,
                "metrics_integrated_relmae_boxplot",
                "Integrated relMAE (median bins)",
                seed_note=seed_note_int_all,
            )
        if "integrated_ode_r2" not in df_int_all.columns and "ode_integ_r2_median" in df_int_all.columns:
            df_int_all["integrated_ode_r2"] = df_int_all["ode_integ_r2_median"]
        if "integrated_ode_r2" in df_int_all.columns:
            df_int_all_ode = df_int_all[df_int_all["integrated_ode_r2"].notna()]
            dedup_cols = [c for c in ("group_name", "dataset_mode", "feature_mode", "model") if c in df_int_all_ode.columns]
            if dedup_cols:
                df_int_all_ode = df_int_all_ode.drop_duplicates(subset=dedup_cols)
            if not df_int_all_ode.empty:
                _plot_metric_lineplot(
                    df_int_all_ode,
                    outdir,
                    marker_type,
                    "integrated_ode_r2",
                    "metrics_integrated_ode_r2_by_marker",
                    "Integrated ODE R2",
                    seed_note=seed_note_int_all,
                )
                integrated_ode_line_done = True
                _plot_metric_boxplot(
                    df_int_all_ode,
                    outdir,
                    [("Overall", df_int_all_ode["integrated_ode_r2"])],
                    "metrics_integrated_ode_r2_boxplot",
                    "Integrated ODE R2 (clipped at 0)",
                    seed_note=seed_note_int_all,
                )
                integrated_ode_box_done = True

                # PySR vs Linear Regression scatter (Integrated ODE R2)
                group_cols = ["group_name"]
                if "dataset_mode" in df_int_all_ode.columns:
                    group_cols.append("dataset_mode")
                if {"model", "integrated_ode_r2"}.issubset(df_int_all_ode.columns):
                    df_int_collapsed = (
                        df_int_all_ode.groupby(group_cols + ["model"], dropna=False)["integrated_ode_r2"]
                        .mean()
                        .reset_index()
                    )
                    mode_values = (
                        sorted(df_int_collapsed["dataset_mode"].dropna().unique())
                        if "dataset_mode" in df_int_collapsed.columns
                        else [None]
                    )
                    for mode_val in mode_values:
                        sub_mode = (
                            df_int_collapsed[df_int_collapsed["dataset_mode"] == mode_val]
                            if mode_val is not None and "dataset_mode" in df_int_collapsed.columns
                            else df_int_collapsed
                        )
                        if sub_mode.empty:
                            continue
                        pivot_ode = sub_mode.pivot_table(
                            index=group_cols,
                            columns="model",
                            values="integrated_ode_r2",
                            dropna=False,
                        )
                        if {"PySR", "Linear Regression"}.issubset(pivot_ode.columns):
                            wide_ode = (
                                pivot_ode.reset_index()
                                .rename(columns={"PySR": "PySR_ode_r2", "Linear Regression": "LinReg_ode_r2"})
                            )
                            wide_ode = wide_ode.dropna(subset=["PySR_ode_r2", "LinReg_ode_r2"], how="all")
                            if not wide_ode.empty:
                                mode_tag = str(mode_val).lower() if mode_val is not None else "all"
                                stem_base = f"metrics_integrated_ode_models_{mode_tag}"
                                saved_stems.add(stem_base)
                                _plot_scatter_variants(
                                    scatter_dir,
                                    wide_ode,
                                    "LinReg_ode_r2",
                                    "PySR_ode_r2",
                                    f"Integrated ODE R2: LinReg vs PySR ({mode_val})",
                                    stem_base,
                                    zoom_thr,
                                    zoom=False,
                                    metric_group="integrated",
                                )
                                _plot_scatter_variants(
                                    scatter_dir,
                                    wide_ode,
                                    "LinReg_ode_r2",
                                    "PySR_ode_r2",
                                    f"Integrated ODE R2 (zoomed): LinReg vs PySR ({mode_val})",
                                    stem_base,
                                    zoom_thr,
                                    zoom=True,
                                    metric_group="integrated",
                                )
                                _copy_preferred_variant_to_base(
                                    scatter_dir,
                                    stem_base,
                                    metric_group="integrated",
                                )

        # Split boxplots directly from metrics columns when available (train/test/overall).
        def _plot_split_from_metrics(
            df_src: pd.DataFrame, train_col: str, test_col: str, stem: str, label: str, seed_note: Optional[str]
        ) -> None:
            def _col_as_series(df_local: pd.DataFrame, col: str) -> Optional[pd.Series]:
                if col not in df_local.columns:
                    return None
                out = df_local[col]
                if isinstance(out, pd.DataFrame):
                    # Duplicate column name; take the first
                    out = out.iloc[:, 0]
                return pd.Series(out)
            series_defs_split: List[Tuple[str, pd.Series]] = []
            s_train = _col_as_series(df_src, train_col)
            s_test = _col_as_series(df_src, test_col)
            if s_train is not None:
                series_defs_split.append(("Train", s_train))
            if s_test is not None:
                series_defs_split.append(("Test", s_test))
            if series_defs_split:
                overall_series = pd.concat([s for _, s in series_defs_split], axis=1).mean(axis=1)
                series_defs_split.append(("Overall", overall_series))
                _plot_metric_boxplot(df_src, outdir, series_defs_split, stem, label, seed_note=seed_note)

        if {"ode_integ_r2_median_train", "ode_integ_r2_median"}.issubset(df_int_all.columns):
            _plot_split_from_metrics(
                df_int_all.rename(columns={"ode_integ_r2_median": "integrated_ode_r2"}),
                "ode_integ_r2_median_train",
                "integrated_ode_r2",
                "metrics_integrated_ode_r2_boxplot",
                "Integrated ODE R2 (train/test/overall)",
                seed_note_int_all,
            )
            integrated_ode_box_done = True
        if {"dt_r2_train", "dt_r2"}.issubset(df_int_all.columns):
            _plot_split_from_metrics(
                df_int_all,
                "dt_r2_train",
                "dt_r2",
                "metrics_dt_r2_boxplot_integration",
                "Integration dt R2 (train/test/overall)",
                seed_note_int_all,
            )

    # Integrated split boxplots if available
    if integration_split_frames:
        df_int_split = pd.concat(integration_split_frames, ignore_index=True)
        if "seed" in df_int_split.columns:
            keys_split = [k for k in ("marker", "group_name", "model", "dataset_mode", "split") if k in df_int_split.columns]
            num_cols_split = df_int_split.select_dtypes(include=[np.number]).columns.tolist()
            num_cols_split = [c for c in num_cols_split if c != "seed"]
            if keys_split and num_cols_split:
                df_num = df_int_split.groupby(keys_split, dropna=False)[num_cols_split].mean().reset_index()
                df_non = df_int_split[keys_split + ["seed"]].groupby(keys_split, dropna=False).first().reset_index()
                df_int_split = df_num.merge(df_non[keys_split], on=keys_split, how="left")
        seed_note_split = None
        if "seed" in df_int_split.columns:
            seed_note_split = _format_seed_label(df_int_split["seed"].dropna().unique(), averaged=True)
        if not seed_note_split:
            seed_note_split = DEFAULT_SEED_NOTE
        if not df_int_split.empty:
            for metric, stem, label in (
                ("integrated_r2", "metrics_integrated_r2_boxplot", "Integrated R2 (clipped at 0)"),
                ("integrated_ode_r2", "metrics_integrated_ode_r2_boxplot", "Integrated ODE R2 (clipped at 0)"),
            ):
                if metric not in df_int_split:
                    continue
                series_defs_split = []
                for split_name in ("Train", "Test"):
                    series_defs_split.append(
                        (
                            split_name,
                            df_int_split[df_int_split["split"] == split_name][metric],
                        )
                    )
                overall_series = df_int_split[metric]
                series_defs_split.append(("Overall", overall_series))
                _plot_metric_boxplot(df_int_split, outdir, series_defs_split, stem, label, seed_note=seed_note_split)
                if metric == "integrated_r2":
                    integrated_box_done = True
                if metric == "integrated_ode_r2":
                    integrated_ode_box_done = True

    # Placeholders if integration plots missing
    if not integrated_line_done:
        _save_placeholder(outdir, "metrics_integrated_r2_by_marker", "lines", seed_note=DEFAULT_SEED_NOTE)
    if not integrated_box_done:
        _save_placeholder(outdir, "metrics_integrated_r2_boxplot", "boxes", seed_note=DEFAULT_SEED_NOTE)
    if not integrated_ode_line_done:
        _save_placeholder(outdir, "metrics_integrated_ode_r2_by_marker", "lines", seed_note=DEFAULT_SEED_NOTE)
    if not integrated_ode_box_done:
        _save_placeholder(outdir, "metrics_integrated_ode_r2_boxplot", "boxes", seed_note=DEFAULT_SEED_NOTE)

    # R2 line plot across markers/models/modes
    progress_log("R2 line/box plots")
    line_dir = outdir / "lines"
    _plot_metric_lineplot(df, outdir, marker_type, "test_r2", "metrics_r2_by_marker", "Test R2", seed_note=DEFAULT_SEED_NOTE)
    series_defs_dt: List[Tuple[str, pd.Series]] = []
    if "train_r2" in df:
        series_defs_dt.append(("Train", df["train_r2"]))
    if "test_r2" in df:
        series_defs_dt.append(("Test", df["test_r2"]))
    if series_defs_dt:
        overall_series = _blend_overall(
            df.get("train_r2"),
            df.get("test_r2"),
            df.get("train_samples"),
            df.get("test_samples_raw"),
        )
        series_defs_dt.append(("Overall", overall_series))
        _plot_metric_boxplot(df, outdir, series_defs_dt, "metrics_r2_boxplot", "R2 (clipped at 0)", seed_note=DEFAULT_SEED_NOTE)

    # Ensure expected base outputs exist if nothing was plotted
    expected_stems = {"metrics_models_scatter", "metrics_pysr_modes", "metrics_linreg_modes"}
    for stem in expected_stems - saved_stems:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        for ext in ("png", "svg"):
            default_dir = scatter_dir / "dt" / "plain" / "full"
            default_dir.mkdir(parents=True, exist_ok=True)
            _add_seed_note(plt.gcf(), DEFAULT_SEED_NOTE)
            plt.savefig(default_dir / f"{stem}_r2.{ext}", dpi=200, bbox_inches="tight")
            plt.savefig(default_dir / f"{stem}_relmae.{ext}", dpi=200, bbox_inches="tight")
        plt.close()

    progress_log("Feature usage/importance")
    # Linear regression feature importance plots (per dataset mode)
    datasets_for_importance = {
        "snapshot": _prepare_importance_dataset(args.snapshot_dataset),
        "per_minute": _prepare_importance_dataset(args.per_minute_dataset),
    }
    feature_sets: Dict[str, Sequence[str]] = {}
    for mode, data in datasets_for_importance.items():
        if data is not None:
            feature_sets[mode] = [c for c in data.columns if c not in EXCLUDE_COLUMNS]
    importance_frames: List[pd.DataFrame] = []
    linreg_df = df[df["model"] == "Linear Regression"].copy()
    has_dataset_mode = "dataset_mode" in linreg_df.columns
    for mode, data in datasets_for_importance.items():
        if data is None:
            continue
        mode_rows = linreg_df.copy()
        if has_dataset_mode:
            mode_rows = mode_rows[mode_rows["dataset_mode"] == mode]
        elif mode != "snapshot":
            continue
        if mode_rows.empty:
            continue
        imp_df = _compute_feature_importance(mode_rows, data)
        if not imp_df.empty:
            importance_frames.append(imp_df)
    importance_df = pd.concat(importance_frames, axis=0) if importance_frames else pd.DataFrame()
    _plot_feature_importance(importance_df, outdir, expected_modes=("snapshot", "per_minute"), seed_note=DEFAULT_SEED_NOTE)

    progress_log("PySR symbol counts")
    # PySR formula variable count boxplots (snapshot and per-minute)
    df_pysr = df[df["model"] == "PySR"].copy()
    _plot_pysr_symbol_counts(
        df_pysr,
        outdir,
        expected_modes=("snapshot", "per_minute"),
        seed_note=DEFAULT_SEED_NOTE,
        feature_sets=feature_sets,
    )

    # Feature usage stacked bars per model/mode
    feature_counts = _collect_feature_counts(df, marker_type)
    _plot_feature_bars(feature_counts, outdir, seed_note=DEFAULT_SEED_NOTE)

    # Seed-aware diagnostics: variance barplot and seed lines
    progress_log("Seed variance/lines")
    if df_seed is not None and seed_col and seed_col not in df_seed.columns:
        print(f"[plot_metrics_summary] Seed column '{seed_col}' missing from summary; skipping seed variance plots.")
    if df_seed is not None and seed_col and seed_col in df_seed.columns and "test_r2" in df_seed.columns:
        var_dir = outdir / "seed_variance"
        var_dir.mkdir(parents=True, exist_ok=True)
        keys_seed = [k for k in ["group_name", "dataset_mode", "feature_mode", "model"] if k in df_seed.columns]
        var_df = (
            df_seed.groupby(keys_seed + [seed_col], dropna=False)["test_r2"].mean()
            .reset_index()
            .groupby(keys_seed, dropna=False)["test_r2"]
            .var()
            .reset_index(name="test_r2_var")
        )
        agg_var = var_df.groupby(["model", "dataset_mode"], dropna=False)["test_r2_var"].mean().reset_index()
        if not agg_var.empty:
            plt.figure(figsize=(6, 4))
            sns.barplot(data=agg_var, x="model", y="test_r2_var", hue="dataset_mode")
            plt.ylabel("Average test R2 variance across seeds")
            plt.title("Seed variance by model/dataset mode")
            plt.tight_layout()
            for ext in ("png", "svg"):
                plt.savefig(var_dir / f"seed_variance_bar.{ext}", dpi=220, bbox_inches="tight")
            plt.close()

        line_dir = outdir / "seed_lines"
        line_dir.mkdir(parents=True, exist_ok=True)
        for model in df_seed["model"].unique():
            for mode in df_seed.get("dataset_mode", pd.Series([])).unique():
                sub = df_seed[(df_seed["model"] == model) & (df_seed.get("dataset_mode") == mode)]
                if sub.empty:
                    continue
                pivot = sub.pivot_table(index="group_name", columns=seed_col, values="test_r2", dropna=False)
                if pivot.empty:
                    continue
                pivot["max_r2"] = pivot.max(axis=1)
                pivot = pivot.sort_values("max_r2", ascending=False).drop(columns="max_r2")
                plt.figure(figsize=(8, max(6, pivot.shape[0] * 0.2)))
                for marker, row in pivot.iterrows():
                    vals = row.dropna()
                    seeds = vals.index.astype(str)
                    plt.plot(vals.values, [marker] * len(vals), marker="o", linestyle="-", label=None, alpha=0.8)
                    for x, s in zip(vals.values, seeds):
                        plt.text(x, marker, s, fontsize=7, va="center", ha="left", alpha=0.7)
                plt.xlabel("Test R2")
                plt.ylabel("Marker")
                plt.title(f"Seed-wise test R2: {model} | {mode}")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                stem = f"seed_lines_{model.lower().replace(' ', '_')}_{mode}"
                for ext in ("png", "svg"):
                    plt.savefig(line_dir / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
                plt.close()

    progress_log("Binwise pERK scatters")
    bin_markers = [m for m in args.binwise_perk_markers if str(m).strip()]
    traj_dir_bin = args.trajectories_dir or args.integration_metrics_dir
    if bin_markers and traj_dir_bin:
        bin_models = ("PySR", "Linear Regression")
        traj_dir_bin = Path(traj_dir_bin)
        for mode in ("snapshot", "per_minute"):
            _plot_binwise_perk_traj(
                outdir,
                traj_dir_bin,
                bin_markers,
                bin_models,
                mode,
                measured_timepoints=args.measured_timepoints,
            )

    progress_log("Worst ODE diagnostics")
    diag_traj_dir = args.trajectories_dir or args.integration_metrics_dir
    _plot_worst_ode_diagnostics(
        outdir,
        Path(args.integration_metrics_dir) if args.integration_metrics_dir else None,
        Path(diag_traj_dir) if diag_traj_dir else None,
        max_markers=args.diagnostic_max_markers,
        dt_threshold=args.diagnostic_dt_threshold,
        ode_threshold=args.diagnostic_ode_threshold,
        gap_threshold=args.diagnostic_gap_threshold,
        model="PySR",
    )

    # Per-marker trajectories per strategy (dt, integrated, integrated ODE) — last to keep other plots first
    progress_log("Marker trajectory overlays")
    traj_dir = args.trajectories_dir or args.integration_metrics_dir
    if traj_dir:
        traj_dir = Path(traj_dir)
        for mode in ("per_minute",):
            pred_path = traj_dir / f"predicted_trajectories_{mode}.csv"
            integ_path = traj_dir / f"marker_integration_trajectories_{mode}.csv"
            if not pred_path.exists() and not integ_path.exists():
                continue
            for model in ("PySR", "Linear Regression"):
                try:
                    _plot_marker_trajectories(
                        outdir,
                        mode,
                        model,
                        predicted_path=pred_path,
                        integration_path=integ_path,
                    )
                except Exception as exc:
                    print(f"[plot_metrics_summary] Skipped trajectory overlays for {model} {mode}: {exc}", flush=True)


if __name__ == "__main__":
    main()
