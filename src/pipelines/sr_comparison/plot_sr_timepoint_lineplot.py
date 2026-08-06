"""Generate a per-timepoint relative-MAE line plot for SR comparison models."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sympy
import torch
import matplotlib.pyplot as plt

from shared.plot_style import apply_cell_systems_style
from shared.mm_models import TARGET_COLUMN as MM_TARGET_COLUMN, mm_predictions
from shared.regime_variants import (
    MODEL_COLOR_MAP,
    MODEL_LINE_ORDER,
    VARIANTS,
    augment_for_variant,
    model_display_name,
)
from shared.nn_model import NeuralNet, device as NN_DEVICE, load_dataset

# Small epsilon to avoid division by zero in relative error
EPS = 1e-12

apply_cell_systems_style()


def split_method_variant(name: str) -> Tuple[str, Optional[str]]:
    for variant in VARIANTS.keys():
        suffix = f"_{variant}"
        if name.endswith(suffix):
            return name[: -len(suffix)], variant
    return name, None


def prepare_linear_dataset(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Convert all numeric, non-time columns from log space to linear via exp."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()
    for col in numeric_cols:
        if col == time_col:
            continue
        result[col] = np.exp(result[col].astype(float))
    return result


def evaluate_formula(expression: sympy.Expr, data: pd.DataFrame, discovery_scale: str) -> np.ndarray:
    """Evaluate a SymPy expression on a DataFrame (columns must match symbol names).
    If the expression was discovered in log space, exponentiate predictions to linear.
    """
    feature_names = sorted({str(symbol) for symbol in expression.free_symbols if str(symbol) in data.columns})
    if not feature_names:
        raise ValueError("No matching features found in dataset for formula evaluation.")
    feature_arrays = [data[name].to_numpy(dtype=float) for name in feature_names]
    func = sympy.lambdify(tuple(feature_names), expression, modules=["numpy"])
    predictions = func(*feature_arrays)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    if discovery_scale == "log":
        predictions = np.exp(predictions)
    return predictions


def evaluate_nn(
    variant: str,
    model_path: Path,
    train_dataset: Path,
    test_dataset: pd.DataFrame,
    features: Optional[str],
    dataset_size: Optional[int],
    seed: int,
) -> pd.Series:
    """Run NN checkpoint for a given variant; model outputs log-space ŷ, we exp() to linear."""
    if not model_path.exists():
        raise FileNotFoundError(f"Missing NN checkpoint: {model_path}")

    _, _, _, _, metadata = load_dataset(
        str(train_dataset),
        dataset_size=dataset_size,
        features=features,
        seed=seed,
        variant=variant,
        return_metadata=True,
    )
    scaler = metadata["scaler"]
    feature_cols = metadata["feature_cols"]
    target_col = metadata["target_col"]

    feature_frame = test_dataset[feature_cols + [target_col]].copy()
    augmented = augment_for_variant(feature_frame, variant) if variant != "sQSSA" else feature_frame
    augmented.index = test_dataset.index

    feature_values = augmented[feature_cols].apply(pd.to_numeric, errors="coerce")
    target_values = pd.to_numeric(augmented[target_col], errors="coerce")
    combined = pd.concat([feature_values, target_values], axis=1)
    combined.columns = feature_cols + [target_col]
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    if combined.empty:
        raise ValueError("No valid rows remaining for NN inference after cleaning.")

    scaled = scaler.transform(combined[feature_cols].to_numpy(dtype=float))
    X_tensor = torch.tensor(scaled, dtype=torch.float32, device=NN_DEVICE)

    model = NeuralNet(input_dim=len(feature_cols), output_dim=1).to(NN_DEVICE)
    state_dict = torch.load(model_path, map_location=NN_DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred_log = model(X_tensor).cpu().numpy().flatten()
    predictions = np.exp(pred_log)  # convert from log-space to linear

    return pd.Series(predictions, index=combined.index)


def relative_errors(predictions: pd.Series, truth_linear: pd.Series) -> pd.Series:
    """Compute per-point relative error in linear space: |ŷ − y| / (|y| + ε)."""
    common_index = predictions.index.intersection(truth_linear.index)
    if common_index.empty:
        return pd.Series(dtype=float)
    pred = predictions.loc[common_index].to_numpy(dtype=float)
    truth = truth_linear.loc[common_index].to_numpy(dtype=float)
    denom = np.maximum(np.abs(truth), EPS)
    errors = np.abs(pred - truth) / denom
    return pd.Series(errors, index=common_index)


def compute_stats(
    distributions: Dict[str, Dict[str, List[float]]],
    ordered_labels: List[str],
    models: Iterable[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Compute median and IQR per timepoint, per model."""
    stats = {model: {"median": [], "q1": [], "q3": []} for model in models}
    for label in ordered_labels:
        for model in models:
            samples = np.array(distributions[label].get(model, []), dtype=float)
            samples = samples[np.isfinite(samples)]
            if samples.size:
                stats[model]["median"].append(float(np.median(samples)))
                stats[model]["q1"].append(float(np.quantile(samples, 0.25)))
                stats[model]["q3"].append(float(np.quantile(samples, 0.75)))
            else:
                stats[model]["median"].append(np.nan)
                stats[model]["q1"].append(np.nan)
                stats[model]["q3"].append(np.nan)
    return stats


def lineplot_limits(stats: Dict[str, Dict[str, List[float]]]) -> Tuple[Optional[float], Optional[float]]:
    """Compute y-limits from finite values across medians and IQRs (linear scale)."""
    ymin, ymax = None, None
    for model_stats in stats.values():
        values = []
        for key in ("median", "q1", "q3"):
            values.extend(model_stats[key])
        finite = np.array(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low = finite.min()
            high = finite.max()
            ymin = low if ymin is None else min(ymin, low)
            ymax = high if ymax is None else max(ymax, high)
    return ymin, ymax


def plot_lineplot(
    ordered_labels: List[str],
    stats: Dict[str, Dict[str, List[float]]],
    x_coords: np.ndarray,
    plot_path: Path,
    template_path: Optional[Path] = None,
) -> None:
    available_models = [model for model in MODEL_LINE_ORDER if model in stats]
    if not available_models:
        raise ValueError("No models available to plot.")

    ymin, ymax = lineplot_limits(stats)

    def _render(is_template: bool, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        if ymin is not None and ymax is not None:
            # Add a small margin for aesthetics
            y_range = ymax - ymin
            if np.isfinite(y_range) and y_range > 0:
                ax.set_ylim(max(0.0, ymin - 0.05 * y_range), ymax + 0.05 * y_range)
        if not is_template:
            for model in available_models:
                med = np.array(stats[model]["median"], dtype=float)
                if med.size == 0 or np.all(np.isnan(med)):
                    continue
                q1 = np.array(stats[model]["q1"], dtype=float)
                q3 = np.array(stats[model]["q3"], dtype=float)
                color = MODEL_COLOR_MAP.get(model, "#7f7f7f")
                ax.plot(x_coords, med, label=model, color=color, marker="o", linewidth=2)
                ax.fill_between(x_coords, q1, q3, color=color, alpha=0.25)
            ax.legend(loc="upper left", frameon=False)
        else:
            handles = [
                plt.Line2D([0], [0], color=MODEL_COLOR_MAP.get(model, "#7f7f7f"), marker="o", linewidth=2, label=model)
                for model in available_models
            ]
            ax.legend(handles=handles, loc="upper left", frameon=False)

        ax.set_xticks(x_coords)
        ax.set_xticklabels(ordered_labels, rotation=20, ha="right")
        ax.set_xlabel("Timepoint Number", fontsize=12)
        ax.set_ylabel("Relative MAE\n|ŷ − y| / (|y| + ε), ε=1e−12", fontsize=12)
        # Linear y-scale for relative error
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)

    _render(False, plot_path)
    if template_path is not None:
        _render(True, template_path)


def parse_nn_models(entries: List[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid --nn-model entry '{item}'. Expected format variant=path.")
        variant, path = item.split("=", 1)
        variant = variant.strip()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown NN variant '{variant}'.")
        result[variant] = Path(path.strip()).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-timepoint relative-MAE for SR comparison models.")
    parser.add_argument("--dataset", required=True, help="Path to test dataset (CSV).")
    parser.add_argument("--train-dataset", required=True, help="Path to training dataset used for NN scaling (CSV).")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature list used for SR/NN training.")
    parser.add_argument("--formulas", nargs="*", default=[], help="Paths to formula files (one per method).")
    parser.add_argument("--methods", nargs="*", default=[], help="Method names corresponding to --formulas.")
    parser.add_argument("--discovery-scales", type=str, default="{}", help="JSON mapping of method to discovery scale.")
    parser.add_argument("--nn-model", dest="nn_models", action="append", default=[], help="Variant=path for NN checkpoints.")
    parser.add_argument("--nn-dataset-size", type=int, default=None, help="Dataset size used for NN training (optional).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for NN data splits.")
    parser.add_argument("--output", required=True, help="Path to save the line plot.")
    parser.add_argument("--template-output", default=None, help="Optional path to save a legend template plot.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    train_dataset_path = Path(args.train_dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not train_dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_dataset_path}")

    df_raw = pd.read_csv(dataset_path)
    if "time" not in df_raw.columns:
        raise KeyError("Dataset must contain a 'time' column for grouping.")
    target_col = MM_TARGET_COLUMN if MM_TARGET_COLUMN in df_raw.columns else df_raw.columns[-1]

    # Establish timepoint ordering/labels
    if "condition_id" in df_raw.columns:
        order_frame = df_raw[["condition_id", "time"]].copy()
        order_frame["_original_idx"] = np.arange(len(order_frame))
        order_frame = order_frame.sort_values(
            by=["condition_id", "time", "_original_idx"],
            kind="mergesort",
        )
        order_frame["_tp"] = order_frame.groupby("condition_id").cumcount()
        timepoint_index = order_frame.set_index("_original_idx")["_tp"].sort_index()
        max_tp = int(timepoint_index.max()) + 1 if not timepoint_index.empty else 0
        ordered_labels = [str(i + 1) for i in range(max_tp)]
        time_labels = timepoint_index.apply(lambda x: str(int(x) + 1))
        time_labels = time_labels.reindex(df_raw.index)
        x_coords = np.arange(1, max_tp + 1, dtype=float)
    else:
        rounded_times = np.round(df_raw["time"].astype(float).to_numpy(), decimals=8)
        unique_times = np.unique(rounded_times)
        time_label_map = {val: str(idx + 1) for idx, val in enumerate(unique_times)}
        ordered_labels = [time_label_map[val] for val in unique_times]
        time_labels = pd.Series([time_label_map[val] for val in rounded_times], index=df_raw.index)
        x_coords = np.arange(1, len(ordered_labels) + 1, dtype=float)

    # Convert full dataset to linear space for evaluation of truth and any needed features
    df_linear = prepare_linear_dataset(df_raw)
    df_linear.index = df_raw.index
    truth_linear = pd.Series(np.exp(df_raw[target_col].astype(float)), index=df_raw.index)

    # Parse methods/formulas and discovery scales
    if len(args.methods) != len(args.formulas):
        raise ValueError("Number of methods and formulas must match.")

    discovery_scales = json.loads(args.discovery_scales)
    method_to_formula = dict(zip(args.methods, args.formulas))
    nn_model_paths = parse_nn_models(args.nn_models)

    model_predictions: Dict[str, pd.Series] = {}

    # Formula-based models
    for method_name, formula_path in method_to_formula.items():
        formula_file = Path(formula_path)
        if not formula_file.exists():
            raise FileNotFoundError(f"Formula file missing: {formula_file}")
        formula_str = formula_file.read_text().strip()
        if not formula_str:
            continue
        expression = sympy.sympify(formula_str)
        base_method, variant = split_method_variant(method_name)
        variant = variant or "sQSSA"
        display_name = model_display_name(base_method, variant)
        scale = discovery_scales.get(base_method, discovery_scales.get(method_name, "linear"))
        variant_data = df_linear if variant == "sQSSA" else augment_for_variant(df_linear, variant)
        variant_data.index = df_linear.index
        preds = evaluate_formula(expression, variant_data, scale)
        model_predictions[display_name] = pd.Series(preds, index=df_linear.index)

    # Michaelis-Menten baselines
    mm_pred_map = mm_predictions(df_linear)
    for variant in VARIANTS.keys():
        display_name = model_display_name("mm", variant)
        preds = mm_pred_map.get(variant)
        if preds is None:
            continue
        model_predictions[display_name] = pd.Series(preds, index=df_linear.index)

    # Neural networks
    for variant, model_path in nn_model_paths.items():
        try:
            preds = evaluate_nn(
                variant=variant,
                model_path=model_path,
                train_dataset=train_dataset_path,
                test_dataset=df_raw,  # raw because we rely on NN scaler/metadata loading
                features=args.features,
                dataset_size=args.nn_dataset_size,
                seed=args.seed,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        display_name = model_display_name("nn", variant)
        model_predictions[display_name] = preds

    available_models = [model for model in MODEL_LINE_ORDER if model in model_predictions]
    if not available_models:
        raise ValueError("No model predictions available to plot.")

    # Aggregate per-timepoint distributions of relative errors
    distributions: Dict[str, Dict[str, List[float]]] = {label: defaultdict(list) for label in ordered_labels}

    for model in available_models:
        errors = relative_errors(model_predictions[model], truth_linear)
        if errors.empty:
            continue
        for idx, err in errors.items():
            label = time_labels.loc[idx]
            distributions[label][model].append(err)

    stats = compute_stats(distributions, ordered_labels, available_models)

    output_path = Path(args.output)
    template_path = Path(args.template_output) if args.template_output else None
    plot_lineplot(ordered_labels, stats, x_coords, output_path, template_path)


if __name__ == "__main__":
    main()