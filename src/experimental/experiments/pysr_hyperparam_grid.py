"""PySR hyperparameter grid search on high-performing functional groups.

This script selects groups whose PySR models achieved log-space R² > 0.7 for
both "gfp" and "all" feature modes, samples three groups at random, and runs a
grid search over PySR hyperparameters (niterations, population size, populations).
For each combination it retrains the model, evaluates log-space R² on a held-out
test split, and writes both raw results and heatmaps visualising the scores.

Changes vs. previous version:
- Strip trailing `_fit` from dataset columns to match functional-groups script.
- Accept `--random-state` (alias of --seed) and use a single RNG across balancing/splits.
- Require `GFP_bin` in the dataset and drop NA targets before balancing to mirror the functional pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import importlib.util

# Load global PySR config
_constants_path = Path(__file__).resolve().parents[1] / "constants.py"
spec = importlib.util.spec_from_file_location("constants", str(_constants_path))
_constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_constants)
PYSR_CONFIG = _constants.PYSR_CONFIG

from rich.console import Console
from rich.table import Table

console = Console()

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from experimental.sr_pipeline.run_functional_groups import (
    balance_by_order_of_magnitude,
    load_marker_groups,
    sanitize_feature_names,
    select_features,
    _compute_regression_metrics,
)
from utils.seeding import seed_everything

LOGGER = logging.getLogger("experimental.pysr_grid")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--group-definitions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    # Seed handling: keep --seed for compatibility, add --random-state as alias.
    # If both omitted → default 42 (matches functional-groups script).
    parser.add_argument("--seed", type=int, default=None, help="Random seed for splits and balancing.")
    parser.add_argument("--random-state", type=int, default=None, help="Alias of --seed.")

    # Optional “baseline+scale” (kept for compatibility, but not used if explicit grids are given)
    parser.add_argument("--base-niterations", type=int, default=300)
    parser.add_argument("--base-population-size", type=int, default=30)
    parser.add_argument("--base-populations", type=int, default=30)
    parser.add_argument("--lower-scale", type=float, default=0.5)
    parser.add_argument("--upper-scale", type=float, default=2.0)

    # Explicit grids. If provided, these override base/scales.
    parser.add_argument(
        "--niterations-grid", type=int, nargs="*",
        default=[150, 300, 450],
        help="Explicit grid for niterations (overrides base/scales). Default: 150 300 450",
    )
    parser.add_argument(
        "--population-size-grid", type=int, nargs="*",
        default=[20, 30, 40],
        help="Explicit grid for population_size. Default: 20 30 40",
    )
    parser.add_argument(
        "--populations-grid", type=int, nargs="*",
        default=[20, 30, 40],
        help="Explicit grid for populations. Default: 20 30 40",
    )

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--log10-cutoff", type=float, default=-3.0)
    parser.add_argument("--gfp-columns", type=str, nargs="+", default=["GFP"])
    parser.add_argument(
        "--feature-modes", nargs="*", choices=("gfp", "all"), default=("gfp", "all")
    )
    parser.add_argument("--min-bin-samples", type=int, default=500)
    parser.add_argument("--max-bin-samples", type=int, default=1000)
    return parser.parse_args()


def _resolve_seed(args: argparse.Namespace) -> int:
    # Priority: --random-state > --seed > 42
    if args.random_state is not None:
        return int(args.random_state)
    if args.seed is not None:
        return int(args.seed)
    return 42


def _build_grid(base: int, lower_scale: float, upper_scale: float) -> List[int]:
    lower = max(int(round(base * lower_scale)), 1)
    middle = max(int(round(base)), 1)
    upper = max(int(round(base * upper_scale)), 1)
    grid = sorted({lower, middle, upper})
    if len(grid) == 2:
        grid.insert(1, middle)
    return grid


def _ensure_grid(vals: List[int], fallback: List[int]) -> List[int]:
    # Guard against weird/empty CLI; if empty, use fallback
    if not vals:
        return fallback
    # Deduplicate + sort
    return sorted({int(v) for v in vals if int(v) > 0})


def _log_loss_expression(log10_cutoff: float) -> str:
    return (
        f"my_loss(x,y)="
        f"(sign(x)*(log10(abs(x)+10^{log10_cutoff})+{abs(log10_cutoff)})"
        f" - "
        f"sign(y)*(log10(abs(y)+10^{log10_cutoff})+{abs(log10_cutoff)}))^2"
    )


def _prepare_group_dataset(
    data: pd.DataFrame,
    markers: Iterable[str],
    feature_columns: List[str],
    test_size: float,
    seed: int,
):
    subset = data[data["marker"].isin(markers)].copy()
    subset = subset.dropna(subset=feature_columns + ["p-ERK1-2_dt"])
    if subset.empty:
        raise ValueError("Selected group has no data after filtering.")

    X = subset[feature_columns].copy()
    y = subset["p-ERK1-2_dt"].astype(float)
    if len(y) < 5 or X.shape[1] == 0:
        raise ValueError("Group dataset is too small for training.")

    sanitized_names = sanitize_feature_names(X.columns)
    X.columns = sanitized_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    if X_train.empty or X_test.empty:
        raise ValueError("Train/test split produced empty partitions.")
    return X_train, X_test, y_train, y_test, sanitized_names


def _run_pysr_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparams: Dict[str, int],
    log10_cutoff: float,
    random_state: int,
) -> Dict[str, float]:
    loss_expr = _log_loss_expression(log10_cutoff)

    # Start from global config; override with sweep params and alignments to the functional_groups script
    config = dict(PYSR_CONFIG)
    config.update(
        {
            # sweep:
            "niterations": hyperparams["niterations"],
            "population_size": hyperparams["population_size"],
            "populations": hyperparams["populations"],
            # alignments:
            "elementwise_loss": loss_expr,   # signed-log loss used by evaluation
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [],           # functional-groups default: none
            "maxsize": 20,
            "parsimony": 0.8,
            "batching": True,
            "annealing": True,
            "verbosity": 0,
            "random_state": int(random_state),
        }
    )

    # Helpful one-time print so you can see the exact loss string in logs
    console.print(f"[dim]Using loss:[/dim] {loss_expr}")

    model = PySRRegressor(**config)
    try:
        console.print(
            f"[bold cyan]→ Running PySR:[/bold cyan] "
            f"niterations={hyperparams['niterations']}, "
            f"population_size={hyperparams['population_size']}, "
            f"populations={hyperparams['populations']}"
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    except Exception as exc:
        console.print(f"[bold red]✗ Training failed[/bold red] for {json.dumps(hyperparams)}: {exc}")
        return {"test_log_r2": float("nan")}
    metrics = _compute_regression_metrics(y_test.values, preds, log10_cutoff)
    score = metrics.get("test_log_r2", float("nan"))
    if np.isnan(score):
        console.print(f"[yellow]⚠️ test_log_r2 = NaN[/yellow]")
    else:
        color = "green" if score >= 0.7 else "red" if score < 0.3 else "yellow"
        console.print(f"[bold {color}]✓ test_log_r2 = {score:.3f}[/bold {color}]")
    return {"test_log_r2": score}


def _pick_candidate_groups(summary: pd.DataFrame, seed: int) -> List[str]:
    pysr_df = summary[summary["model"].str.lower() == "pysr"].copy()
    grouped = pysr_df.groupby(["group_name", "feature_mode"])["test_log_r2"].mean().unstack()
    qualified = grouped[(grouped.get("gfp", 0) >= 0.7) & (grouped.get("all", 0) >= 0.7)].index.tolist()
    if len(qualified) < 3:
        raise ValueError("Fewer than three groups satisfy the R² threshold in both modes.")
    rng = random.Random(seed)
    rng.shuffle(qualified)
    return qualified[:3]


def _plot_heatmaps(
    results_by_mode: Dict[str, Dict[int, np.ndarray]],
    populations: List[int],
    pop_sizes: List[int],
    out_dir: Path,
    group_name: str,
    mode_order: Optional[Iterable[str]] = None,
) -> None:
    if not results_by_mode:
        LOGGER.warning("No heatmap data available for group '%s'; skipping figure.", group_name)
        return

    ordered_modes: List[str]
    if mode_order is None:
        ordered_modes = list(results_by_mode.keys())
    else:
        ordered_modes = [mode for mode in mode_order if mode in results_by_mode]
        for mode in results_by_mode:
            if mode not in ordered_modes:
                ordered_modes.append(mode)

    if not ordered_modes:
        LOGGER.warning("No feature modes produced heatmaps for group '%s'; skipping.", group_name)
        return

    ncols = max((len(results_by_mode[mode]) for mode in ordered_modes), default=0)
    if ncols == 0:
        LOGGER.warning("Heatmap grids empty for group '%s'; nothing to plot.", group_name)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    populations_arr = np.array(populations)
    pop_sizes_arr = np.array(pop_sizes)

    finite_chunks = []
    for mode in ordered_modes:
        for matrix in results_by_mode[mode].values():
            finite_vals = matrix[np.isfinite(matrix)]
            if finite_vals.size:
                finite_chunks.append(finite_vals)
    if finite_chunks:
        combined = np.concatenate(finite_chunks)
        vmin = float(np.min(combined))
        vmax = float(np.max(combined))
        if np.isclose(vmin, vmax):
            delta = 0.05 if vmax == 0 else abs(vmax) * 0.05
            vmin -= delta
            vmax += delta
    else:
        vmin, vmax = 0.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    nrows = len(ordered_modes)
    fig_width = max(4.0 * ncols, 6.0)
    fig_height = max(3.2 * nrows, 4.0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    last_im = None
    for row_idx, mode in enumerate(ordered_modes):
        items = sorted(results_by_mode[mode].items())
        for col_idx in range(ncols):
            ax = axs[row_idx, col_idx]
            if col_idx >= len(items):
                ax.axis("off")
                continue

            niter, matrix = items[col_idx]
            im = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto", norm=norm)
            last_im = im

            ax.set_xticks(np.arange(len(pop_sizes_arr)))
            ax.set_yticks(np.arange(len(populations_arr)))
            ax.set_xticklabels(pop_sizes_arr)
            ax.set_yticklabels(populations_arr)

            if row_idx == nrows - 1:
                ax.set_xlabel("Population size")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Populations")
            else:
                ax.set_ylabel("")

            ax.set_title(f"{mode} • niterations={niter}")

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if np.isfinite(value):
                        norm_val = norm(value)
                        if not np.isfinite(norm_val):
                            norm_val = 0.5
                        norm_val = float(np.clip(norm_val, 0.0, 1.0))
                        text_color = "white" if norm_val > 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=9,
                        )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axs.ravel().tolist(), shrink=0.85)
        cbar.set_label("Log-space R²")
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    fig.suptitle(f"{group_name} • PySR hyperparameter grid", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in group_name)
    png_path = out_dir / f"{slug}_combined_heatmaps.png"
    svg_path = out_dir / f"{slug}_combined_heatmaps.svg"
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved combined heatmap figure %s", png_path)


def main() -> None:
    args = parse_args()
    rs = _resolve_seed(args)  # unified random seed / random state

    seed_everything(rs)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Reading summary from %s", args.summary)
    summary = pd.read_csv(args.summary)
    if "model" not in summary.columns or "test_log_r2" not in summary.columns:
        raise ValueError("Summary file missing required columns 'model' or 'test_log_r2'.")

    selected_groups = _pick_candidate_groups(summary, rs)
    LOGGER.info("Selected groups: %s", ", ".join(selected_groups))

    LOGGER.info("Loading dataset %s", args.dataset)
    data = pd.read_csv(args.dataset)

    # *** CHANGE 1: Strip trailing `_fit` to match functional-groups preprocessing ***
    data.columns = data.columns.str.replace("_fit$", "", regex=True)

    # *** CHANGE 2: Require same columns and drop NA targets before balancing ***
    required_cols = {"marker", "GFP_bin", "p-ERK1-2_dt"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Dataset must contain {sorted(required_cols)}; missing: {sorted(missing)}")

    # Drop rows with missing targets (functional-groups behavior)
    before_rows = len(data)
    data = data.dropna(subset=["p-ERK1-2_dt"])
    LOGGER.info("Retained %d/%d rows after target drop", len(data), before_rows)

    # Zero tiny targets before balancing (same as functional-groups)
    data.loc[data["p-ERK1-2_dt"].abs() < 10.0 ** args.log10_cutoff, "p-ERK1-2_dt"] = 0.0

    LOGGER.info("Balancing dataset by order of magnitude")
    balanced = balance_by_order_of_magnitude(
        data,
        target_column="p-ERK1-2_dt",
        log10_cutoff=args.log10_cutoff,
        min_samples=args.min_bin_samples,
        max_samples=args.max_bin_samples,
        random_state=rs,
    )

    LOGGER.info("Loading group definitions from %s", args.group_definitions_csv)
    marker_groups = load_marker_groups(
        balanced,
        json_path=None,
        csv_path=args.group_definitions_csv,
    )

    # Build grids (explicit CLI grids take precedence)
    niterations_grid = _ensure_grid(args.niterations_grid, [150, 300, 450])
    population_size_grid = _ensure_grid(args.population_size_grid, [20, 30, 40])
    populations_grid = _ensure_grid(args.populations_grid, [20, 30, 40])

    results_records: List[Dict[str, object]] = []
    gfp_columns = list(args.gfp_columns)

    for group_name in selected_groups:
        markers = marker_groups.get(group_name)
        if not markers:
            LOGGER.warning("Group '%s' missing in definitions; skipping.", group_name)
            continue

        group_slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in group_name)
        group_dir = output_dir / group_slug
        group_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Processing group '%s' (%d markers)", group_name, len(markers))
        group_heatmap_results: Dict[str, Dict[int, np.ndarray]] = {}

        for feature_mode in args.feature_modes:
            feature_cols = select_features(balanced, feature_mode, gfp_columns)
            if not feature_cols:
                LOGGER.warning("No features for mode '%s'; skipping group '%s'", feature_mode, group_name)
                continue

            try:
                X_train, X_test, y_train, y_test, _ = _prepare_group_dataset(
                    balanced, markers, feature_cols, args.test_size, rs
                )
            except ValueError as exc:
                LOGGER.warning("Skipping group '%s' mode '%s': %s", group_name, feature_mode, exc)
                continue

            heatmap_data: Dict[int, np.ndarray] = {}

            for niter in niterations_grid:
                matrix = np.full((len(populations_grid), len(population_size_grid)), np.nan, dtype=float)
                for i, pops in enumerate(populations_grid):
                    for j, pop_size in enumerate(population_size_grid):
                        hyperparams = {"niterations": niter, "populations": pops, "population_size": pop_size}
                        metrics = _run_pysr_model(
                            X_train, y_train, X_test, y_test, hyperparams, args.log10_cutoff, rs
                        )
                        matrix[i, j] = metrics["test_log_r2"]
                        results_records.append(
                            {
                                "group": group_name,
                                "feature_mode": feature_mode,
                                "niterations": niter,
                                "populations": pops,
                                "population_size": pop_size,
                                "test_log_r2": metrics["test_log_r2"],
                            }
                        )
                heatmap_data[niter] = matrix

                summary_table = Table(title=f"Results Summary • {group_name} • {feature_mode}")
                summary_table.add_column("niterations", justify="right")
                summary_table.add_column("populations", justify="right")
                summary_table.add_column("pop_size", justify="right")
                summary_table.add_column("test_log_r2", justify="right")

                for rec in [r for r in results_records if r["group"] == group_name and r["feature_mode"] == feature_mode]:
                    val = rec["test_log_r2"]
                    if isinstance(val, float) and not np.isnan(val):
                        color = "green" if val >= 0.7 else "red" if val < 0.3 else "yellow"
                        fmt = f"[{color}]{val:.3f}[/{color}]"
                    else:
                        fmt = "[yellow]NaN[/yellow]"
                    summary_table.add_row(
                        str(rec["niterations"]), str(rec["populations"]), str(rec["population_size"]), fmt
                    )
                console.print(summary_table)

            if heatmap_data:
                group_heatmap_results[feature_mode] = heatmap_data

        if group_heatmap_results:
            heatmap_dir = group_dir / "heatmaps"
            _plot_heatmaps(
                group_heatmap_results,
                populations_grid,
                population_size_grid,
                heatmap_dir,
                group_name,
                mode_order=list(args.feature_modes),
            )
        else:
            LOGGER.warning("No heatmaps generated for group '%s'; skipping combined figure.", group_name)

    if not results_records:
        LOGGER.error("No results recorded; aborting without writing outputs.")
        return

    results_df = pd.DataFrame(results_records)
    results_path = output_dir / "pysr_hyperparam_grid_results.csv"
    results_df.to_csv(results_path, index=False)
    LOGGER.info("Wrote aggregated results to %s", results_path)


# if __name__ == "__main__":
#     main()
