"""Symbolic regression over functional marker groups.

This script mirrors the SR exploration notebook logic for part 3 by
training PySR models on balanced trajectories grouped by functional
markers. It supports two feature configurations "gfp" (GFP-only
features) and "all" (all available dynamical features) and reports test
metrics only, per user request.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pysr import PySRRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Columns that should not be passed to the regression model as inputs.
EXCLUDE_COLUMNS = {"p-ERK1-2_dt", "p-MEK1-2_dt", "marker", "timepoint", "GFP_bin"}


@dataclass
class GroupResult:
    group_name: str
    markers: Tuple[str, ...]
    feature_mode: str
    formula: Optional[str]
    test_log_mae: Optional[float]
    test_log_r2: Optional[float]
    test_mae: Optional[float]
    test_r2: Optional[float]
    test_samples: int
    train_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PySR over functional marker groups.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the CSV file produced from the perturbation trajectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where reports and plots will be written.",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="*",
        choices=("gfp", "all"),
        default=("gfp", "all"),
        help="Feature modes to run. 'gfp' keeps GFP-only columns; 'all' keeps every feature except exclusions.",
    )
    parser.add_argument(
        "--gfp-columns",
        nargs="*",
        default=("GFP",),
        help="Columns to keep when --feature-modes includes 'gfp'.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test evaluation (default: 0.2 for an 80/20 split).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and resampling.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=500,
        help="Minimum samples per order-of-magnitude bin when balancing the dataset.",
    )
    parser.add_argument(
        "--max-bin-samples",
        type=int,
        default=1000,
        help="Maximum samples per order-of-magnitude bin when balancing the dataset.",
    )
    parser.add_argument(
        "--log10-cutoff",
        type=float,
        default=-3.0,
        help="Cutoff used for the signed log transform (matches the notebook default).",
    )
    parser.add_argument(
        "--marker-groups-json",
        type=Path,
        default=None,
        help="Optional JSON file overriding the default marker groups. Should map group names to marker lists.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=200,
        help="Skip groups with fewer combined samples than this threshold after preprocessing.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="PySR niterations parameter (default mirrors notebook).",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=30,
        help="PySR population size per generation.",
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=30,
        help="PySR populations (parallel demes).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=35,
        help="Maximum expression size for PySR.",
    )
    parser.add_argument(
        "--parsimony",
        type=float,
        default=0.8,
        help="Parsimony coefficient for PySR (complexity penalty).",
    )
    parser.add_argument(
        "--binary-operators",
        nargs="*",
        default=("+", "-", "*", "/"),
        help="Binary operators exposed to PySR.",
    )
    parser.add_argument(
        "--unary-operators",
        nargs="*",
        default=(),
        help="Optional unary operators for PySR (e.g. exp, log).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="PySR verbosity level (default silences per-iteration logging).",
    )
    parser.add_argument(
        "--annealing",
        action="store_true",
        default=True,
        help="Enable PySR simulated annealing (on by default).",
    )
    parser.add_argument(
        "--no-annealing",
        action="store_false",
        dest="annealing",
        help="Disable PySR simulated annealing.",
    )
    parser.add_argument(
        "--batching",
        action="store_true",
        default=True,
        help="Enable PySR mini-batching (on by default).",
    )
    parser.add_argument(
        "--no-batching",
        action="store_false",
        dest="batching",
        help="Disable PySR mini-batching.",
    )
    return parser.parse_args()


def load_marker_groups(dataset: pd.DataFrame, json_path: Optional[Path]) -> Dict[str, List[str]]:
    if json_path is not None:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {str(k): list(v) for k, v in payload.items()}

    # Default groups mirror the expanded sets used in the notebook.
    groups: Dict[str, List[str]] = {
        "MAPK_core": ["MAP2K2", "MAPK1", "MAPK3"],
        "MAPK_RSK": ["MAPK1", "MAPK3", "RPS6KA1"],
        "MAPK_upstream": ["MAP2K2", "MAP4K2", "ARAF"],
        "RTKs_1": ["EGFR", "ERBB2", "FGFR1"],
        "RTKs_2": ["MET", "MST1R", "FGFR1"],
        "RTKs_combo": ["EGFR", "ERBB2", "MET", "MST1R", "FGFR1"],
        "PI3K_AKT": ["AKT3", "PIK3R1", "PIP5K3", "PRKACA"],
        "PI3K_AKT_short": ["AKT3", "PRKACA", "PIP5K3"],
        "DUSPs": ["DUSP4", "DUSP7", "DUSP10 (P2)", "DUSP16"],
        "PTPNs": ["PTPN2 (P1)", "PTPN5", "PTPN7"],
        "Mixed_phosphatases": ["DUSP4", "PTPN2 (P1)", "DUSP10 (P2)"],
        "Cytoplasmic_TKs": ["ABL1", "TEC", "TYRO3"],
        "DYRK_Module": ["DYRK2", "DYRK3", "MAP4K2"],
        "RSK_feedback": ["RPS6KA1", "RPS6KA3", "RPS6KA6"],
        "Scaffold_1": ["MAST2", "ALPK2", "ARAF"],
        "Scaffold_2": ["TEC", "MAST2", "ALPK2"],
        "Stress_linked": ["TBK1", "MAP4K2", "DYRK2"],
        "Feedback_plus_core": ["MAPK1", "DUSP4", "PTPN2 (P1)"],
        "Mixed_positive_negative": ["EGFR", "AKT3", "DUSP4"],
        "Mixed_scaffolds": ["MAST2", "ARAF", "RPS6KA1"],
        "Mixed_scaffolds_alt": ["TYRO3", "MAPK3", "ALPK2"],
        "Control_FLAG_GFP": ["FLAG-GFP1", "FLAG-GFP2", "FLAG-GFP3", "FLAG-GFP4"],
        "Control_untransfected": [
            "untransfected1",
            "untransfected2",
            "untransfected3",
            "untransfected4",
        ],
        "RTK_PI3K_combo": ["EGFR", "ERBB2", "AKT3", "PRKACA"],
        "RTK_PI3K_combo_alt": ["MET", "MST1R", "PIK3R1", "PIP5K3"],
        "RTK_phosphatase_combo": ["EGFR", "ERBB2", "PTPN2 (P1)", "DUSP4"],
        "RTK_phosphatase_combo_alt": ["MET", "FGFR1", "DUSP10 (P2)", "PTPN5"],
        "MAPK_PI3K_combo": ["MAP2K2", "MAPK3", "AKT3", "PIP5K3"],
        "MAPK_PI3K_combo_alt": ["MAPK1", "MAPK3", "PRKACA", "PIK3R1"],
        "Feedback_stress_combo": ["DUSP4", "DUSP16", "DYRK2", "MAP4K2"],
        "Coverage_fill_1": ["ABL1", "PRKACA", "RPS6KA3"],
        "Coverage_fill_2": ["TBK1", "PIP5K3", "DUSP7"],
        "Coverage_fill_3": ["TYRO3", "MAST2", "PIP5K3"],
    }

    # Filter out markers that are absent from the dataset to avoid silent failures.
    available_markers = set(dataset["marker"].unique())
    filtered_groups: Dict[str, List[str]] = {}
    for group_name, markers in groups.items():
        present = [m for m in markers if m in available_markers]
        if present:
            filtered_groups[group_name] = present
    return filtered_groups


def signed_log(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    epsilon = 10.0 ** log10_cutoff
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * (np.log10(np.abs(arr) + epsilon) - log10_cutoff)


def balance_by_order_of_magnitude(
    df: pd.DataFrame,
    target_column: str,
    log10_cutoff: float,
    min_samples: int,
    max_samples: int,
    random_state: int,
) -> pd.DataFrame:
    epsilon = 10.0 ** log10_cutoff
    work = df.copy()
    work[target_column] = work[target_column].astype(float)
    work[f"oom_bins_{target_column}"] = np.floor(
        np.log10(np.abs(work[target_column]) + epsilon)
    )

    balanced_frames: List[pd.DataFrame] = []
    rng = np.random.RandomState(random_state)
    for _, group in work.groupby(f"oom_bins_{target_column}"):
        n = len(group)
        if n < min_samples:
            balanced = resample(
                group,
                replace=True,
                n_samples=min_samples,
                random_state=rng.randint(0, 1_000_000),
            )
        elif n > max_samples:
            balanced = resample(
                group,
                replace=False,
                n_samples=max_samples,
                random_state=rng.randint(0, 1_000_000),
            )
        else:
            balanced = group
        balanced_frames.append(balanced)

    combined = pd.concat(balanced_frames, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    combined = combined.drop(columns=[f"oom_bins_{target_column}"])
    return combined


def sanitize_feature_names(columns: Iterable[str]) -> List[str]:
    renamed: List[str] = []
    for col in columns:
        clean = col
        clean = clean.replace("-", "_")
        clean = clean.replace(" ", "_")
        clean = clean.replace("(", "")
        clean = clean.replace(")", "")
        clean = clean.replace("/", "_")
        if clean and clean[0].isdigit():
            clean = f"f_{clean}"
        renamed.append(clean)
    return renamed


def select_features(
    df: pd.DataFrame, feature_mode: str, gfp_columns: Sequence[str]
) -> List[str]:
    candidate_columns = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    if feature_mode == "gfp":
        return [c for c in candidate_columns if c in gfp_columns]
    if feature_mode == "all":
        return candidate_columns
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def train_group_model(
    data: pd.DataFrame,
    group_label: str,
    markers: Sequence[str],
    feature_mode: str,
    feature_columns: List[str],
    test_size: float,
    random_state: int,
    log10_cutoff: float,
    sr_kwargs: Dict[str, object],
) -> Optional[GroupResult]:
    subset = data[data["marker"].isin(markers)].copy()
    if subset.empty:
        return None

    subset = subset.dropna(subset=feature_columns + ["p-ERK1-2_dt"])
    if subset.empty:
        return None

    X = subset[feature_columns].copy()
    y = subset["p-ERK1-2_dt"].astype(float)

    if len(y) < 5 or X.shape[1] == 0:
        return None

    sanitized_names = sanitize_feature_names(X.columns)
    X.columns = sanitized_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if len(y_test) == 0 or len(y_train) == 0:
        return None

    log_loss_expr = (
        f"my_loss(x,y)=(sign(x)*(log10(abs(x)+10^{log10_cutoff})+{abs(log10_cutoff)})"
        f"-sign(y)*(log10(abs(y)+10^{log10_cutoff})+{abs(log10_cutoff)}))^2"
    )

    model = PySRRegressor(
        elementwise_loss=log_loss_expr,
        **sr_kwargs,
    )

    try:
        model.fit(X_train, y_train)
    except Exception:
        return GroupResult(
            group_name=group_label,
            markers=tuple(markers),
            feature_mode=feature_mode,
            formula=None,
            test_log_mae=None,
            test_log_r2=None,
            test_mae=None,
            test_r2=None,
            test_samples=len(y_test),
            train_samples=len(y_train),
        )

    try:
        y_pred_test = model.predict(X_test)
    except Exception:
        return GroupResult(
            group_name=group_label,
            markers=tuple(markers),
            feature_mode=feature_mode,
            formula=None,
            test_log_mae=None,
            test_log_r2=None,
            test_mae=None,
            test_r2=None,
            test_samples=len(y_test),
            train_samples=len(y_train),
        )

    log_y_test = signed_log(y_test.values, log10_cutoff)
    log_y_pred = signed_log(y_pred_test, log10_cutoff)

    result = GroupResult(
        group_name=group_label,
        markers=tuple(markers),
        feature_mode=feature_mode,
        formula=str(model.sympy()) if model.equations_ is not None else None,
        test_log_mae=float(mean_absolute_error(log_y_test, log_y_pred)),
        test_log_r2=float(r2_score(log_y_test, log_y_pred)) if len(np.unique(log_y_pred)) > 1 else None,
        test_mae=float(mean_absolute_error(y_test.values, y_pred_test)),
        test_r2=float(r2_score(y_test.values, y_pred_test)) if len(np.unique(y_pred_test)) > 1 else None,
        test_samples=len(y_test),
        train_samples=len(y_train),
    )
    return result


def format_results_text(results: Sequence[GroupResult]) -> str:
    lines: List[str] = []
    for res in results:
        lines.append(f"Group: {res.group_name}")
        lines.append(f"Markers: {', '.join(res.markers)}")
        lines.append(f"Feature mode: {res.feature_mode}")
        lines.append(f"Train samples: {res.train_samples}")
        lines.append(f"Test samples: {res.test_samples}")
        lines.append(f"Formula: {res.formula if res.formula is not None else 'N/A'}")
        lines.append(
            "Test log MAE: "
            + (f"{res.test_log_mae:.4f}" if res.test_log_mae is not None else "N/A")
        )
        lines.append(
            "Test log R2: "
            + (f"{res.test_log_r2:.4f}" if res.test_log_r2 is not None else "N/A")
        )
        lines.append(
            "Test MAE: "
            + (f"{res.test_mae:.4f}" if res.test_mae is not None else "N/A")
        )
        lines.append(
            "Test R2: "
            + (f"{res.test_r2:.4f}" if res.test_r2 is not None else "N/A")
        )
        lines.append("")
    return "\n".join(lines)


def plot_log_r2(results: Sequence[GroupResult], output_dir: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "group": res.group_name,
                "mode": res.feature_mode,
                "log_r2": res.test_log_r2,
            }
            for res in results
            if res.test_log_r2 is not None
        ]
    )

    if df.empty:
        return

    pivot = df.pivot_table(index="group", columns="mode", values="log_r2")
    modes = sorted(df["mode"].unique())
    groups = list(pivot.index)

    x = np.arange(len(groups))
    width = 0.35 if len(modes) == 2 else 0.6

    fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * 0.35)))
    for idx, mode in enumerate(modes):
        mode_values = pivot[mode].values
        offset = (idx - (len(modes) - 1) / 2) * width
        ax.bar(x + offset, mode_values, width=width, label=mode.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=60, ha="right")
    ax.set_ylabel("Test log R²")
    ax.set_title("Test log R² per marker group")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=False, prune=None))

    fig.tight_layout()
    png_path = output_dir / "functional_group_log_r2.png"
    svg_path = output_dir / "functional_group_log_r2.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.dataset)
    data.columns = data.columns.str.replace("_fit$", "", regex=True)

    required_columns = {"marker", "GFP_bin", "p-ERK1-2_dt"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.groupby(["marker", "GFP_bin"]).filter(lambda g: not g.isnull().any().any())
    data.loc[
        data["p-ERK1-2_dt"].abs() < 10.0 ** args.log10_cutoff,
        "p-ERK1-2_dt",
    ] = 0.0

    balanced = balance_by_order_of_magnitude(
        data,
        target_column="p-ERK1-2_dt",
        log10_cutoff=args.log10_cutoff,
        min_samples=args.min_bin_samples,
        max_samples=args.max_bin_samples,
        random_state=args.random_state,
    )

    if len(balanced) < args.min_group_size:
        raise ValueError(
            "Balanced dataset is too small after preprocessing; adjust thresholds or provide more data."
        )

    marker_groups = load_marker_groups(balanced, args.marker_groups_json)
    if not marker_groups:
        raise ValueError("No marker groups available after intersecting with dataset markers.")

    sr_kwargs = {
        "niterations": args.max_iterations,
        "population_size": args.population_size,
        "populations": args.populations,
        "maxsize": args.max_size,
        "parsimony": args.parsimony,
        "binary_operators": list(args.binary_operators),
        "unary_operators": list(args.unary_operators),
        "batching": args.batching,
        "annealing": args.annealing,
        "verbosity": args.verbosity,
    }

    all_results: List[GroupResult] = []
    for feature_mode in args.feature_modes:
        feature_columns = select_features(balanced, feature_mode, args.gfp_columns)
        if not feature_columns:
            print(f"[WARN] No usable features for mode '{feature_mode}'. Skipping.")
            continue

        mode_results: List[GroupResult] = []
        for group_name, markers in marker_groups.items():
            result = train_group_model(
                balanced,
                group_name,
                markers,
                feature_mode,
                feature_columns,
                args.test_size,
                args.random_state,
                args.log10_cutoff,
                sr_kwargs,
            )
            if result is None:
                print(
                    f"[INFO] Skipping group '{group_name}' for mode '{feature_mode}' (insufficient data)."
                )
                continue

            print(
                f"[INFO] Mode={feature_mode} | Group={group_name} | Test log MAE={result.test_log_mae} | Test log R2={result.test_log_r2}"
            )
            mode_results.append(result)

        if not mode_results:
            continue

        text_output = format_results_text(mode_results)
        text_path = output_dir / f"functional_group_formulas_{feature_mode}.txt"
        text_path.write_text(text_output, encoding="utf-8")
        all_results.extend(mode_results)

    if not all_results:
        raise RuntimeError("No successful symbolic regressions were produced. Check preprocessing settings.")

    plot_log_r2(all_results, output_dir)

    summary_df = pd.DataFrame(
        [
            {
                "group_name": res.group_name,
                "markers": list(res.markers),
                "feature_mode": res.feature_mode,
                "formula": res.formula,
                "test_log_mae": res.test_log_mae,
                "test_log_r2": res.test_log_r2,
                "test_mae": res.test_mae,
                "test_r2": res.test_r2,
                "train_samples": res.train_samples,
                "test_samples": res.test_samples,
            }
            for res in all_results
        ]
    )
    summary_df.to_csv(output_dir / "functional_group_sr_summary.csv", index=False)


if __name__ == "__main__":
    main()
