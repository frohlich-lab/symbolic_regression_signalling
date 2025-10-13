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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from utils.seeding import seed_everything

# Columns that should not be passed to the regression model as inputs.
EXCLUDE_COLUMNS = {"p-ERK1-2_dt", "p-MEK1-2_dt", "marker", "timepoint", "GFP_bin"}


LOGGER = logging.getLogger("sr.functional_groups")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def log_progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        LOGGER.info("%s [%d]", stage, current)
        return
    percent = (current / total) * 100.0
    LOGGER.info("%s [%d/%d | %.1f%%]", stage, current, total, percent)


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


GROUPS_FRESH: Dict[str, List[str]] = {
    # ===== Predicted pERK ↑ (drivers / upstream) =====
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
    # ===== Predicted pERK ↓ (phosphatases / feedback brakes) =====
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
    # ===== Ambiguous / modulators =====
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
    parser.add_argument(
        "--include-fresh-groups",
        action="store_true",
        help="Augment the default marker groups with the curated fresh set defined in the script.",
    )
    return parser.parse_args()


def load_marker_groups(
    dataset: pd.DataFrame,
    json_path: Optional[Path],
    include_fresh: bool,
) -> Dict[str, List[str]]:
    if json_path is not None:
        with json_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return {str(k): list(v) for k, v in payload.items()}

    # Default groups mirror the expanded sets used in the notebook.
    groups: Dict[str, List[str]] = {
        # ---- BIG, SIGN-CONSISTENT SETS ----
        # Predicted pERK ↑
        "RTKs_pos": ["EGFR", "ERBB2", "FGFR1", "MET", "MST1R"],
        "RAF_MEK_core": ["ARAF", "MAP2K2"],
        "ERK_substrate_risk": ["MAPK1", "MAPK3"],  # analyze separately (readout confound)
        "Cytoplasmic_TKs_pos": ["ABL1", "TEC", "TYRO3"],

        # Predicted pERK ↓
        "DUSPs_all": ["DUSP4", "DUSP7", "DUSP10 (P2)", "DUSP16"],
        "PTPNs_all": ["PTPN2 (P1)", "PTPN5", "PTPN7"],
        "Feedback_brakes": ["RPS6KA1", "RPS6KA3", "RPS6KA6", "PRKACA"],
        "PI3K_AKT_brake": ["AKT3", "PIK3R1"],

        # Ambiguous / modulators (analyze separately)
        "Stress_or_altMAPK": ["TBK1", "MAP4K2", "DYRK2", "DYRK3"],
        "Trafficking_scaffold": ["PIP5K3", "MAST2", "ALPK2"],

        # Controls
        "Control_FLAG_GFP": ["FLAG-GFP1", "FLAG-GFP2", "FLAG-GFP3", "FLAG-GFP4"],
        "Control_untransfected": ["untransfected1", "untransfected2", "untransfected3", "untransfected4"],

        # ---- SMALLER, “INTELLIGENT” SUBSETS (SAME-SIGN) ----
        # RTK subsets (all ↑)
        "RTKs_EGFR_ERBB2": ["EGFR", "ERBB2"],
        "RTKs_MET_MST1R": ["MET", "MST1R"],
        "RTKs_FGFR1_only": ["FGFR1"],

        # Core pathway splits (↑)
        "RAF_only": ["ARAF"],
        "MEK_only": ["MAP2K2"],

        # ERK readout splits (treat separately)
        "ERK1_only": ["MAPK3"],
        "ERK2_only": ["MAPK1"],

        # Cytoplasmic TKs (↑)
        "ABL1_TEC": ["ABL1", "TEC"],
        "TYRO3_only": ["TYRO3"],

        # RSK/PKA feedback (↓)
        "RSK_feedback": ["RPS6KA1", "RPS6KA3", "RPS6KA6"],
        "PKA_only": ["PRKACA"],

        # DUSPs: ERK-biased vs stress-biased (↓)
        "DUSPs_ERK_biased": ["DUSP4", "DUSP7"],
        "DUSPs_stress_biased": ["DUSP10 (P2)", "DUSP16"],

        # PTPNs splits (↓)
        "PTPN2_only": ["PTPN2 (P1)"],
        "PTPN5_7": ["PTPN5", "PTPN7"],

        # PI3K/AKT axis (↓)
        "AKT_only": ["AKT3"],
        "PI3K_reg_only": ["PIK3R1"],

        # Ambiguous/modulators (neutral bucket for separate analysis)
        "DYRK_module": ["DYRK2", "DYRK3"],
        "MAP4K2_only": ["MAP4K2"],
        "TBK1_only": ["TBK1"],
        "PIP5K3_only": ["PIP5K3"],
        "Scaffold_core": ["MAST2", "ALPK2"],

        # ---- SAME-SIGN MULTI-PATHWAY “STACKS” (optional) ----
        "RTK_plus_RAF_MEK": ["EGFR", "ERBB2", "ARAF", "MAP2K2"],        # ↑
        "Brake_stack_phosphatases": ["DUSP4", "DUSP7", "PTPN2 (P1)"],    # ↓
        "Brake_stack_kinase": ["RPS6KA1", "PRKACA", "AKT3"],             # ↓
    }

    # Filter out markers that are absent from the dataset to avoid silent failures.
    available_markers = set(dataset["marker"].unique())
    filtered_groups: Dict[str, List[str]] = {}
    for group_name, markers in groups.items():
        present = [m for m in markers if m in available_markers]
        if present:
            filtered_groups[group_name] = present
    if include_fresh:
        for group_name, markers in GROUPS_FRESH.items():
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
    if work[target_column].dropna().empty:
        LOGGER.warning(
            "Target column '%s' has no finite values; returning empty frame", target_column
        )
        return work.iloc[0:0]
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

    if not balanced_frames:
        LOGGER.warning("No magnitude bins produced samples; returning empty frame")
        return work.iloc[0:0]

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


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.random_state)
    LOGGER.info("Starting functional group symbolic regression")
    if args.include_fresh_groups:
        LOGGER.info("Including %d curated fresh marker groups", len(GROUPS_FRESH))
    LOGGER.info("Loading dataset from %s", args.dataset)
    data = pd.read_csv(args.dataset)
    data.columns = data.columns.str.replace("_fit$", "", regex=True)

    required_columns = {"marker", "GFP_bin", "p-ERK1-2_dt"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    LOGGER.info("Dropping rows with missing target derivatives")
    before_rows = len(data)
    data = data.dropna(subset=["p-ERK1-2_dt"])
    LOGGER.info("Retained %d/%d rows after target drop", len(data), before_rows)
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

    LOGGER.info("Loaded %d samples after balancing", len(balanced))
    marker_groups = load_marker_groups(
        balanced,
        args.marker_groups_json,
        include_fresh=args.include_fresh_groups,
    )
    if not marker_groups:
        raise ValueError("No marker groups available after intersecting with dataset markers.")
    LOGGER.info("Running on %d marker groups", len(marker_groups))

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
    total_modes = len(args.feature_modes)
    for mode_idx, feature_mode in enumerate(args.feature_modes, start=1):
        log_progress("Feature modes", mode_idx, total_modes)
        LOGGER.info("Selecting features for mode '%s'", feature_mode)
        feature_columns = select_features(balanced, feature_mode, args.gfp_columns)
        if not feature_columns:
            LOGGER.warning("No usable features for mode '%s'. Skipping.", feature_mode)
            continue

        mode_results: List[GroupResult] = []
        total_groups = len(marker_groups)
        LOGGER.info(
            "Evaluating %d marker groups for mode '%s'", total_groups, feature_mode
        )
        for group_idx, (group_name, markers) in enumerate(
            marker_groups.items(), start=1
        ):
            log_progress(f"{feature_mode} groups", group_idx, total_groups)
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
                LOGGER.info(
                    "Skipping group '%s' for mode '%s' (insufficient data)",
                    group_name,
                    feature_mode,
                )
                continue

            LOGGER.info(
                "Mode=%s | Group=%s | Features=%s | Test log MAE=%s | Test log R2=%s",
                feature_mode,
                group_name,
                feature_columns,
                result.test_log_mae,
                result.test_log_r2,
            )
            mode_results.append(result)

        if not mode_results:
            LOGGER.warning(
                "No successful regressions for feature mode '%s'", feature_mode
            )
            continue

        text_output = format_results_text(mode_results)
        text_path = output_dir / f"functional_group_formulas_{feature_mode}.txt"
        text_path.write_text(text_output, encoding="utf-8")
        LOGGER.info("Saved formula report to %s", text_path)
        all_results.extend(mode_results)

    if not all_results:
        LOGGER.error("No successful symbolic regressions were produced")
        raise RuntimeError("No successful symbolic regressions were produced. Check preprocessing settings.")

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
    summary_path = output_dir / "functional_group_sr_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Wrote summary CSV to %s", summary_path)
    LOGGER.info("Symbolic regression pipeline completed")


if __name__ == "__main__":
    main()
