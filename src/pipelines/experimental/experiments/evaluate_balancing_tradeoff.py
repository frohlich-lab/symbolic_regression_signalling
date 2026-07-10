"""Evaluate the effect of different max-bin caps on functional-group datasets.

This script emulates the order-of-magnitude bin truncation performed in
``run_markers.py`` while allowing the maximum samples per bin to vary.
It reports how the resulting dataset size and order-of-magnitude balance change
for a shortlist of high-performing groups (or for groups supplied on the CLI).

Example usage:
    python src/pipelines/experimental/experiments/evaluate_balancing_tradeoff.py \\
        --max-samples 1500 1200 1000 750 500 \\
        --include-uncapped
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp


DEFAULT_DATASET = Path("data/experimental/processed/markers/markers_fit_snapshot.csv")
DEFAULT_GROUP_DEFS = Path("data/experimental/processed/markers.csv")
DEFAULT_SUMMARY = Path("data/experimental/runs/aggregated/reports/summary/marker_summary.csv")

TARGET_COLUMN_SANITIZED = "p_ERK1_2_dt"
RAW_MARKER_KEY = "marker"

EXCLUDE_COLUMNS_RAW = {"p-ERK1-2_dt", "p-MEK1-2_dt", "marker", "timepoint", "GFP_bin"}

_FULL_DATASET_CACHE: Optional[List[Dict[str, object]]] = None
_FULL_DATASET_MARKER_INDEX: Optional[Dict[str, List[Dict[str, object]]]] = None
_FULL_DATASET_RENAME_MAP: Optional[Dict[str, str]] = None
_FULL_DATASET_COLUMNS: Optional[List[str]] = None


def _sanitize_column_name(label: str) -> str:
    clean = label.replace("-", "_")
    clean = clean.replace(" ", "_")
    clean = clean.replace("(", "")
    clean = clean.replace(")", "")
    clean = clean.replace("/", "_")
    if clean and clean[0].isdigit():
        clean = f"f_{clean}"
    return clean


SANITIZED_EXCLUDE_COLUMNS = {_sanitize_column_name(col) for col in EXCLUDE_COLUMNS_RAW}


def _sanitize_column_map(columns: Iterable[str]) -> Dict[str, str]:
    rename: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    for col in columns:
        base = _sanitize_column_name(str(col))
        counter = counts.get(base, 0)
        new_name = base if counter == 0 else f"{base}_{counter}"
        counts[base] = counter + 1
        rename[str(col)] = new_name
    return rename


@dataclass(frozen=True)
class DataRow:
    """Minimal representation of a trajectory sample."""

    index: int
    marker: str
    target: float
    gfp_bin: Optional[int]
    timepoint: Optional[float]
    oom_bin: int


def parse_float(value: object) -> Optional[float]:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_oom_bin(value: float, log10_cutoff: float) -> int:
    epsilon = 10.0 ** log10_cutoff
    return int(math.floor(math.log10(abs(value) + epsilon)))


def load_dataset(csv_path: Path, log10_cutoff: float) -> List[DataRow]:
    rows: List[DataRow] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    epsilon = 10.0 ** log10_cutoff
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, record in enumerate(reader):
            marker = (record.get("marker") or "").strip()
            if not marker:
                continue
            target_raw = parse_float(record.get("p-ERK1-2_dt"))
            if target_raw is None:
                continue
            target = 0.0 if abs(target_raw) < epsilon else target_raw

            gfp_val = parse_float(record.get("GFP_bin"))
            gfp_bin = None if gfp_val is None else int(gfp_val)

            tp_val = parse_float(record.get("timepoint"))
            oom_bin = compute_oom_bin(target, log10_cutoff)

            rows.append(
                DataRow(
                    index=idx,
                    marker=marker,
                    target=target,
                    gfp_bin=gfp_bin,
                    timepoint=tp_val,
                    oom_bin=oom_bin,
                )
            )
    if not rows:
        raise ValueError(f"No usable rows parsed from dataset {csv_path}")
    return rows


def _ensure_full_dataset_loaded(csv_path: Path, log10_cutoff: float) -> None:
    global _FULL_DATASET_CACHE, _FULL_DATASET_MARKER_INDEX, _FULL_DATASET_RENAME_MAP, _FULL_DATASET_COLUMNS
    if _FULL_DATASET_CACHE is not None and _FULL_DATASET_MARKER_INDEX is not None:
        return

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        initial_map = _sanitize_column_map(fieldnames)
        rename_map: Dict[str, str] = {}
        existing_names: set[str] = set()
        for original in fieldnames:
            sanitized = initial_map.get(original, original)
            if sanitized.endswith("_fit"):
                base = sanitized[:-4]
                if base and base not in existing_names:
                    sanitized = base
            if sanitized in existing_names:
                # fall back to original sanitized name to avoid collision
                sanitized = initial_map.get(original, original)
            rename_map[original] = sanitized
            existing_names.add(sanitized)
        epsilon = 10.0 ** log10_cutoff
        cache: List[Dict[str, object]] = []
        marker_index: Dict[str, List[Dict[str, object]]] = defaultdict(list)

        for record in reader:
            sanitized_row: Dict[str, object] = {}
            for original_key, value in record.items():
                key = rename_map.get(original_key, original_key)
                if key == RAW_MARKER_KEY:
                    sanitized_row[key] = (value or "").strip()
                    continue

                numeric_val = parse_float(value)
                if key == TARGET_COLUMN_SANITIZED and numeric_val is not None and abs(numeric_val) < epsilon:
                    numeric_val = 0.0
                sanitized_row[key] = numeric_val

            marker_value = str(sanitized_row.get(RAW_MARKER_KEY, "") or "")
            cache.append(sanitized_row)
            marker_index[marker_value].append(sanitized_row)

    _FULL_DATASET_CACHE = cache
    _FULL_DATASET_MARKER_INDEX = marker_index
    _FULL_DATASET_RENAME_MAP = rename_map
    _FULL_DATASET_COLUMNS = list(cache[0].keys()) if cache else []


def _get_group_full_rows(markers: Sequence[str], csv_path: Path, log10_cutoff: float) -> List[Dict[str, object]]:
    _ensure_full_dataset_loaded(csv_path, log10_cutoff)
    assert _FULL_DATASET_MARKER_INDEX is not None
    subset: List[Dict[str, object]] = []
    for marker in markers:
        subset.extend(_FULL_DATASET_MARKER_INDEX.get(str(marker), []))
    return subset


def _get_available_feature_columns() -> List[str]:
    if _FULL_DATASET_COLUMNS is None:
        return []
    return [
        col
        for col in _FULL_DATASET_COLUMNS
        if col not in SANITIZED_EXCLUDE_COLUMNS and col != TARGET_COLUMN_SANITIZED
    ]


def _get_sanitized_gfp_columns(args: argparse.Namespace) -> List[str]:
    if args.pysr_gfp_columns:
        return [_sanitize_column_name(col) for col in args.pysr_gfp_columns]
    # fall back to default used in run_markers.py when nothing provided
    return [_sanitize_column_name(col) for col in ("GFP",)]


def _feature_columns_for_mode(feature_mode: str, args: argparse.Namespace) -> List[str]:
    available = _get_available_feature_columns()
    if not available:
        return []
    feature_mode = feature_mode.lower()
    if feature_mode == "gfp":
        sanitized = set(_get_sanitized_gfp_columns(args))
        return [col for col in available if col in sanitized]
    if feature_mode == "all":
        return available
    # default fallback mirrors select_features behavior
    return available


def _get_group_clean_rows(
    markers: Sequence[str],
    feature_columns: Sequence[str],
    csv_path: Path,
    log10_cutoff: float,
) -> List[Dict[str, object]]:
    rows = _get_group_full_rows(markers, csv_path, log10_cutoff)
    clean: List[Dict[str, object]] = []
    for row in rows:
        target_val = row.get(TARGET_COLUMN_SANITIZED)
        if target_val is None or not isinstance(target_val, (int, float)) or not math.isfinite(target_val):
            continue
        skip = False
        for col in feature_columns:
            val = row.get(col)
            if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                skip = True
                break
        if skip:
            continue
        clean.append(row)
    return clean


def load_group_definitions(csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    groups: Dict[str, List[str]] = {}
    raw_members: Dict[str, str] = {}
    if not csv_path.exists():
        raise FileNotFoundError(f"Group definition CSV not found: {csv_path}")
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for record in reader:
            name = (record.get("group") or "").strip()
            members_raw = record.get("members")
            if not name or members_raw is None:
                continue
            try:
                parsed = ast.literal_eval(members_raw)
                members_iter: Iterable[str] = parsed if isinstance(parsed, (list, tuple, set)) else []
            except (SyntaxError, ValueError):
                cleaned = members_raw.strip().strip("[]")
                members_iter = [part.strip() for part in cleaned.split(",")]
            members = [str(member).strip() for member in members_iter if str(member).strip()]
            if members:
                groups[name] = members
                raw_members[name] = members_raw if isinstance(members_raw, str) else repr(members)
    if not groups:
        raise ValueError(f"No marker groups could be parsed from {csv_path}")
    return groups, raw_members


def select_top_groups(summary_path: Path, top_k: int) -> List[str]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    scored: List[Tuple[str, float]] = []
    with summary_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for record in reader:
            model = (record.get("model") or "").strip().lower()
            feature_mode = (record.get("feature_mode") or "").strip().lower()
            if model != "pysr" or feature_mode != "gfp":
                continue
            name = (record.get("group_name") or "").strip()
            score_val = parse_float(record.get("test_log_r2"))
            if not name or score_val is None or math.isnan(score_val):
                continue
            scored.append((name, score_val))

    scored.sort(key=lambda x: x[1], reverse=True)
    unique: List[str] = []
    for name, _score in scored:
        if name not in unique:
            unique.append(name)
        if len(unique) >= top_k:
            break
    return unique


def balance_rows(rows: Sequence[DataRow], max_samples: Optional[int], seed: int) -> List[DataRow]:
    buckets: Dict[int, List[DataRow]] = defaultdict(list)
    for row in rows:
        buckets[row.oom_bin].append(row)

    rng = random.Random(seed)
    balanced: List[DataRow] = []
    for _, group in sorted(buckets.items()):
        if max_samples is not None and max_samples > 0 and len(group) > max_samples:
            balanced.extend(rng.sample(group, max_samples))
        else:
            balanced.extend(group)

    rng.shuffle(balanced)
    return balanced


def filter_group_rows(rows: Sequence[DataRow], group_markers: Sequence[str]) -> List[DataRow]:
    candidate_markers = set(group_markers)
    return [row for row in rows if row.marker in candidate_markers]


def compute_group_stats(rows: Sequence[DataRow]) -> Dict[str, object]:
    if not rows:
        return {
            "rows": 0,
            "unique_markers": 0,
            "unique_trajectories": 0,
            "oom_bins": 0,
            "oom_min": 0,
            "oom_max": 0,
            "oom_ratio": float("nan"),
        }

    marker_counts = Counter(row.marker for row in rows)
    trajectory_counts = {(row.marker, row.gfp_bin) for row in rows}
    oom_counts = Counter(row.oom_bin for row in rows)

    oom_min = min(oom_counts.values()) if oom_counts else 0
    oom_max = max(oom_counts.values()) if oom_counts else 0
    oom_ratio = float("inf") if oom_min == 0 else (oom_max / oom_min if oom_min else float("inf"))

    return {
        "rows": len(rows),
        "unique_markers": len(marker_counts),
        "unique_trajectories": len(trajectory_counts),
        "oom_bins": len(oom_counts),
        "oom_min": oom_min,
        "oom_max": oom_max,
        "oom_ratio": oom_ratio,
    }


def _signed_log_array(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    eps = 10.0 ** log10_cutoff
    return np.sign(values) * (np.log10(np.abs(values) + eps) - log10_cutoff)


def _log_r2_score(y_true: np.ndarray, y_pred: np.ndarray, log10_cutoff: float) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    y_log = _signed_log_array(y_true, log10_cutoff)
    yhat_log = _signed_log_array(y_pred, log10_cutoff)

    ss_res = np.sum((y_log - yhat_log) ** 2)
    ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _evaluate_formula_on_rows(formula: str, rows: Sequence[Dict[str, object]], log10_cutoff: float) -> float:
    if not rows:
        return float("nan")

    safe_formula = formula.replace("^", "**")
    try:
        expr = sp.sympify(safe_formula)
    except Exception:
        return float("nan")

    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    symbol_names = [str(sym) for sym in symbols]

    target_values: List[float] = []
    feature_arrays: Dict[str, List[float]] = {name: [] for name in symbol_names}
    epsilon = 10.0 ** log10_cutoff

    for row in rows:
        target_val = row.get(TARGET_COLUMN_SANITIZED)
        if target_val is None or not isinstance(target_val, (int, float)) or not math.isfinite(target_val):
            continue

        feature_vals: List[float] = []
        skip = False
        for name in symbol_names:
            feature_val = row.get(name)
            if feature_val is None or not isinstance(feature_val, (int, float)) or not math.isfinite(feature_val):
                if name.startswith("log10_"):
                    base = name[6:]
                    base_val = row.get(base)
                    if base_val is None or not isinstance(base_val, (int, float)) or not math.isfinite(base_val):
                        skip = True
                        break
                    base_val = float(base_val)
                    feature_val = float(np.sign(base_val) * (np.log10(abs(base_val) + epsilon) - log10_cutoff))
                else:
                    skip = True
                    break
            feature_vals.append(float(feature_val))
        if skip:
            continue

        target_values.append(float(target_val))
        for name, value in zip(symbol_names, feature_vals):
            feature_arrays[name].append(value)

    if len(target_values) < 2:
        return float("nan")

    try:
        func = sp.lambdify(symbols, expr, modules={"numpy": np})
        feature_inputs = [np.asarray(feature_arrays[name], dtype=float) for name in symbol_names]
        predictions = np.asarray(func(*feature_inputs), dtype=float).reshape(-1)
    except Exception:
        return float("nan")

    targets = np.asarray(target_values, dtype=float)
    return _log_r2_score(targets, predictions, log10_cutoff)


def format_group_stats(group: str, stats: Dict[str, object]) -> str:
    if stats["rows"] == 0:
        return f"  - {group}: no samples present"

    ratio = stats["oom_ratio"]
    ratio_str = "∞" if math.isinf(ratio) else f"{ratio:.1f}×"
    return (
        f"  - {group}: rows={stats['rows']} | markers={stats['unique_markers']} "
        f"| trajectories={stats['unique_trajectories']} | oom bins={stats['oom_bins']} "
        f"(min={stats['oom_min']}, max={stats['oom_max']}, max/min={ratio_str})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dataset size and imbalance for varying max-bin caps."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Trajectory CSV used for SR.")
    parser.add_argument(
        "--group-definitions",
        type=Path,
        default=DEFAULT_GROUP_DEFS,
        help="CSV listing functional groups and member markers.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary CSV with SR metrics (for auto-selecting top groups).",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Specific group names to analyse. Defaults to top performers in the summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of high-performing groups to auto-select when --groups is omitted.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        nargs="+",
        default=[1500, 1200, 1000, 750, 500, 350],
        help="List of max samples per OOM bin to evaluate.",
    )
    parser.add_argument(
        "--include-uncapped",
        action="store_true",
        help="Also evaluate the uncapped dataset (no max samples).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    parser.add_argument(
        "--log10-cutoff",
        type=float,
        default=-3.0,
        help="Signed-log cutoff applied before computing OOM bins.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a text report. Creates parent directory if needed.",
    )
    parser.add_argument(
        "--run-pysr",
        action="store_true",
        help="For each configuration, run run_markers.py and collect PySR metrics.",
    )
    parser.add_argument(
        "--pysr-script",
        type=Path,
        default=Path("src/pipelines/experimental/sr_pipeline/run_markers.py"),
        help="Path to run_markers.py when --run-pysr is enabled.",
    )
    parser.add_argument(
        "--pysr-output-root",
        type=Path,
        default=Path("data/experimental/runs/tradeoff_runs"),
        help="Root directory for per-cap PySR outputs.",
    )
    parser.add_argument(
        "--pysr-feature-modes",
        nargs="*",
        default=["gfp"],
        help="Feature modes to pass to run_markers.py.",
    )
    parser.add_argument(
        "--pysr-gfp-columns",
        nargs="*",
        default=None,
        help="Optional GFP column list forwarded when feature modes include 'gfp'.",
    )
    parser.add_argument(
        "--pysr-models",
        nargs="*",
        default=["pysr"],
        help="Models to request when running run_markers.py.",
    )
    parser.add_argument(
        "--pysr-random-state",
        type=int,
        default=42,
        help="Random seed passed to run_markers.py.",
    )
    parser.add_argument(
        "--pysr-test-size",
        type=float,
        default=0.2,
        help="Test-size fraction passed to run_markers.py.",
    )
    parser.add_argument(
        "--min-bin-samples",
        type=int,
        default=500,
        help="Minimum samples per OOM bin (also forwarded to run_markers.py).",
    )
    parser.add_argument(
        "--uncapped-max-value",
        type=int,
        default=1_000_000,
        help="Effective max-bin value used to approximate an uncapped configuration.",
    )
    parser.add_argument(
        "--pysr-feature-mode-eval",
        type=str,
        default="gfp",
        help="Feature mode used when reading PySR metrics from the summary.",
    )
    parser.add_argument(
        "--pysr-model-name",
        type=str,
        default="pysr",
        help="Model name (case-insensitive) to extract from the PySR summary.",
    )
    parser.add_argument(
        "--pysr-extra-args",
        nargs="*",
        default=None,
        help="Additional arguments appended when invoking run_markers.py.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PySR runs if the summary CSV already exists for a configuration.",
    )
    parser.add_argument(
        "--include-linreg",
        action="store_true",
        help="If set, also compute Linear Regression metrics alongside PySR for each data slice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_rows = load_dataset(args.dataset, args.log10_cutoff)
    total_rows = len(dataset_rows)
    lines: List[str] = []
    lines.append(f"Loaded {total_rows} rows from {args.dataset}")

    group_definitions, raw_group_rows = load_group_definitions(args.group_definitions)

    if args.groups:
        target_groups: List[str] = []
        for group_name in args.groups:
            if group_name not in group_definitions:
                lines.append(f"Warning: group '{group_name}' not found in definitions; skipping.")
                continue
            target_groups.append(group_name)
    else:
        try:
            auto_groups = select_top_groups(args.summary, args.top_k)
        except FileNotFoundError:
            raise RuntimeError(
                f"Summary CSV not found at {args.summary}. Provide explicit --groups or generate the summary first."
            ) from None
        target_groups = [name for name in auto_groups if name in group_definitions]
        missing = set(auto_groups) - set(target_groups)
        for missing_name in missing:
            lines.append(f"Warning: top group '{missing_name}' not present in group definitions; skipped.")

    if not target_groups:
        raise RuntimeError("No target groups available for analysis.")

    lines.append(f"Analysing groups: {', '.join(target_groups)}")

    caps: List[Optional[int]] = []
    if args.include_uncapped:
        caps.append(None)
    for cap in args.max_samples:
        if cap <= 0:
            raise ValueError("Max-samples entries must be positive integers.")
        caps.append(cap)

    # Preserve order while removing duplicates
    unique_caps: List[Optional[int]] = []
    for cap in caps:
        if cap not in unique_caps:
            unique_caps.append(cap)

    if args.run_pysr:
        args.pysr_output_root.mkdir(parents=True, exist_ok=True)
        if target_groups:
            subset_csv = args.pysr_output_root / "selected_groups.csv"
            with subset_csv.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["group", "members"])
                for name in target_groups:
                    raw = raw_group_rows.get(name)
                    members_raw = raw if raw is not None else json.dumps(group_definitions[name])
                    writer.writerow([name, members_raw])
            group_defs_path = subset_csv
        else:
            group_defs_path = args.group_definitions
    else:
        group_defs_path = args.group_definitions

    for cap in unique_caps:
        cap_label = "uncapped" if cap is None else str(cap)
        balanced_rows = balance_rows(dataset_rows, cap, args.seed)
        ratio = len(balanced_rows) / total_rows

        lines.append("")
        lines.append(f"=== Max samples per OOM bin: {cap_label} ===")
        lines.append(f"Balanced dataset rows: {len(balanced_rows)} ({ratio:.1%} of original)")

        for group_name in target_groups:
            markers = group_definitions[group_name]
            subset = filter_group_rows(balanced_rows, markers)
            stats = compute_group_stats(subset)
            lines.append(format_group_stats(group_name, stats))

        if args.run_pysr:
            effective_cap = args.uncapped_max_value if cap is None else cap
            lines.extend(
                run_pysr_for_cap(
                    cap_label=cap_label,
                    cap_value=effective_cap,
                    args=args,
                    group_defs_path=group_defs_path,
                    target_groups=target_groups,
                    group_markers=group_definitions,
                )
            )

    report_text = "\n".join(lines)
    print(report_text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")


def run_pysr_for_cap(
    cap_label: str,
    cap_value: int,
    args: argparse.Namespace,
    group_defs_path: Path,
    target_groups: Sequence[str],
    group_markers: Dict[str, List[str]],
) -> List[str]:
    """Execute run_markers.py with a specific max-bin cap and parse metrics."""
    output_dir = args.pysr_output_root / f"cap_{cap_label}"
    summary_path = output_dir / "reports" / "summary" / "marker_summary.csv"

    _ensure_full_dataset_loaded(args.dataset, args.log10_cutoff)
    feature_mode_eval = args.pysr_feature_mode_eval.lower()
    feature_columns = _feature_columns_for_mode(feature_mode_eval, args)
    include_linreg = args.include_linreg and bool(feature_columns)

    if args.skip_existing and summary_path.exists():
        result_stdout = ""
        result_stderr = ""
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd: List[str] = [sys.executable, str(args.pysr_script)]
        cmd += ["--dataset", str(args.dataset)]
        cmd += ["--output-dir", str(output_dir)]
        if args.pysr_feature_modes:
            cmd += ["--feature-modes", *args.pysr_feature_modes]
        if args.pysr_gfp_columns:
            cmd += ["--gfp-columns", *args.pysr_gfp_columns]
        cmd += ["--group-definitions-csv", str(group_defs_path)]
        if args.pysr_models:
            cmd += ["--models", *args.pysr_models]
        cmd += ["--random-state", str(args.pysr_random_state)]
        cmd += ["--test-size", str(args.pysr_test_size)]
        cmd += ["--min-bin-samples", str(args.min_bin_samples)]
        cmd += ["--max-bin-samples", str(cap_value)]
        cmd += ["--log10-cutoff", str(args.log10_cutoff)]
        if args.pysr_extra_args:
            cmd += list(args.pysr_extra_args)

        result = subprocess.run(cmd, capture_output=True, text=True)
        result_stdout = result.stdout
        result_stderr = result.stderr
        if result.returncode != 0:
            notes = [
                f"    PySR run command: {' '.join(cmd)}",
                f"    PySR run failed (exit code {result.returncode}). stderr:",
            ]
            notes.extend(f"      {line}" for line in result_stderr.strip().splitlines())
            return notes
        # invalidate skip cache so we refresh metrics
        summary_path.unlink(missing_ok=False)
        # rerun to ensure summary created after removing prior state
        result = subprocess.run(cmd, capture_output=True, text=True)
        result_stdout = result.stdout
        result_stderr = result.stderr
        if result.returncode != 0:
            notes = [
                f"    PySR run command: {' '.join(cmd)}",
                f"    PySR rerun failed (exit code {result.returncode}). stderr:",
            ]
            notes.extend(f"      {line}" for line in result_stderr.strip().splitlines())
            return notes

    lines: List[str] = [f"    PySR summary path: {summary_path}"]

    if not summary_path.exists():
        if result_stderr:
            lines.append("    PySR summary not found; stderr output:")
            lines.extend(f"      {line}" for line in result_stderr.strip().splitlines())
        else:
            lines.append("    PySR summary not found; skipping metric extraction.")
        return lines

    model_key = args.pysr_model_name.lower()
    feature_mode_key = args.pysr_feature_mode_eval.lower()

    with summary_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    for group_name in target_groups:
        matches = [
            row
            for row in rows
            if (row.get("group_name") or "").strip() == group_name
            and (row.get("model") or "").strip().lower() == model_key
            and (row.get("feature_mode") or "").strip().lower() == feature_mode_key
        ]
        if not matches:
            lines.append(
                f"    {group_name}: no PySR row for model='{model_key}' feature_mode='{feature_mode_key}'"
            )
            continue
        row = matches[0]
        log_r2 = row.get("test_log_r2")
        log_mae = row.get("test_log_mae")
        log_rel_mae = row.get("test_log_relative_mae")
        test_samples = row.get("test_samples")
        train_samples = row.get("train_samples")
        formula = row.get("formula", "N/A")
        full_log_r2 = float("nan")
        clean_log_r2 = float("nan")
        linreg_clean_log_r2 = float("nan")
        linreg_full_log_r2 = float("nan")
        linreg_test_log_r2 = None
        linreg_formula = None
        if include_linreg:
            lin_matches = [
                row
                for row in rows
                if (row.get("group_name") or "").strip() == group_name
                and (row.get("model") or "").strip().lower() == "linear regression"
                and (row.get("feature_mode") or "").strip().lower() == feature_mode_key
            ]
            if lin_matches:
                lin_row = lin_matches[0]
                linreg_test_log_r2 = lin_row.get("test_log_r2")
                linreg_formula = lin_row.get("formula", "")
        if formula and group_name in group_markers:
            markers = group_markers[group_name]
            full_rows = _get_group_full_rows(markers, args.dataset, args.log10_cutoff)
            full_log_r2 = _evaluate_formula_on_rows(formula, full_rows, args.log10_cutoff)
            if feature_columns:
                clean_rows = _get_group_clean_rows(
                    markers,
                    feature_columns,
                    args.dataset,
                    args.log10_cutoff,
                )
                clean_log_r2 = _evaluate_formula_on_rows(formula, clean_rows, args.log10_cutoff)
                if include_linreg and linreg_formula:
                    linreg_full_log_r2 = _evaluate_formula_on_rows(linreg_formula, full_rows, args.log10_cutoff)
                    linreg_clean_log_r2 = _evaluate_formula_on_rows(linreg_formula, clean_rows, args.log10_cutoff)
        full_log_r2_str = "NA" if math.isnan(full_log_r2) else f"{full_log_r2:.6f}"
        clean_log_r2_str = "NA" if math.isnan(clean_log_r2) else f"{clean_log_r2:.6f}"
        linreg_clean_str = "NA" if math.isnan(linreg_clean_log_r2) else f"{linreg_clean_log_r2:.6f}"
        linreg_full_str = "NA" if math.isnan(linreg_full_log_r2) else f"{linreg_full_log_r2:.6f}"
        lines.append(
            f"    {group_name}: log_R2={log_r2} | log_MAE={log_mae} | log_rel_MAE={log_rel_mae} "
            f"| train={train_samples}, test={test_samples} | formula={formula} | full_log_R2_all={full_log_r2_str} "
            f"| unbalanced_log_R2_clean={clean_log_r2_str}"
        )
        if include_linreg:
            if linreg_formula is None:
                lines.append(
                    f"    (LinReg) {group_name}: no Linear Regression row found for feature_mode='{feature_mode_key}'"
                )
            else:
                test_str = linreg_test_log_r2 if linreg_test_log_r2 is not None else "NA"
                lines.append(
                    f"    (LinReg) {group_name}: test_log_R2={test_str} | full_log_R2_all={linreg_full_str} | unbalanced_log_R2_clean={linreg_clean_str}"
                )

    if args.skip_existing and summary_path.exists():
        lines.append("    (PySR run skipped; summary already present.)")

    return lines


if __name__ == "__main__":
    main()
