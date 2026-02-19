"""
Symbolic regression via ODEFormer.

This wrapper loads the time-series dataset, prepares the time grid and state
trajectory, runs the pretrained ODEFormer beam search, and writes the top
candidate equation to a temporary file for downstream sweep orchestration.
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ODEFORMER_IMPORT_ERROR = None
try:
    from odeformer.model import SymbolicTransformerRegressor
except Exception as exc:  # pragma: no cover - allow offline installs
    SymbolicTransformerRegressor = None
    ODEFORMER_IMPORT_ERROR = exc


TIME_COLUMN = "timepoint"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _select_columns(
    data: pd.DataFrame,
    features: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Return dataframe with only time + requested features; warn on missing."""
    if TIME_COLUMN not in data.columns:
        raise ValueError(f"Expected time column '{TIME_COLUMN}' in dataset.")

    if features and features != "all":
        requested = [col.strip() for col in features.split(",") if col.strip()]
        available = [col for col in requested if col in data.columns]
        missing = [col for col in requested if col not in data.columns]
        if missing:
            print(f"Warning: missing features for ODEFormer: {missing}. Using available columns {available}.")
        selected = [TIME_COLUMN] + available
    else:
        non_time = [col for col in data.columns if col != TIME_COLUMN]
        selected = [TIME_COLUMN] + non_time

    return data[selected].copy(), [col for col in selected if col != TIME_COLUMN]


def load_dataset(
    file_path: str,
    dataset_size: Optional[int] = None,
    features: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load dataset, optionally restrict to a subset of features, and apply exp
    transform to state variables (not to the time column).
    """
    data = pd.read_csv(file_path)
    data, state_cols = _select_columns(data, features)

    data = data.dropna(subset=[TIME_COLUMN])
    data = data.sort_values(TIME_COLUMN)

    if dataset_size is not None:
        data = data.head(min(dataset_size, len(data)))

    if not state_cols:
        raise ValueError("ODEFormer received no state columns after selection.")

    data.loc[:, state_cols] = np.exp(data[state_cols])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        raise ValueError("ODEFormer received an empty dataset after cleaning.")

    return data


def _extract_score(candidate: object) -> Optional[float]:
    """Best-effort extraction of a numeric score from a candidate object."""
    for attr in ("score", "log_likelihood", "loglikelihood", "likelihood"):
        if hasattr(candidate, attr):
            try:
                return float(getattr(candidate, attr))
            except Exception:
                continue
    if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
        try:
            return float(candidate[1])
        except Exception:
            return None
    if isinstance(candidate, dict):
        for key in ("score", "log_likelihood", "loss"):
            if key in candidate:
                try:
                    return float(candidate[key])
                except Exception:
                    continue
    return None


def find_best_formula(
    data: pd.DataFrame,
    temp_file: str,
    beam_size: Optional[int] = None,
    beam_temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> None:
    times = data[TIME_COLUMN].to_numpy(dtype=float)
    trajectory = data.drop(columns=[TIME_COLUMN]).to_numpy(dtype=float)
    temp_path = Path(temp_file)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.unlink(missing_ok=True)

    if SymbolicTransformerRegressor is None:
        raise RuntimeError(
            "ODEFormer is not installed. Please install it (e.g., 'pip install odeformer'). "
            f"Original import error: {ODEFORMER_IMPORT_ERROR}"
        )

    if seed is not None:
        seed_everything(seed)

    model = SymbolicTransformerRegressor(from_pretrained=True)
    model_args = {}
    if beam_size is not None:
        model_args["beam_size"] = beam_size
    if beam_temperature is not None:
        model_args["beam_temperature"] = beam_temperature
    if model_args:
        model.set_model_args(model_args)

    candidates = model.fit(times, trajectory)
    if not candidates:
        # Fallback to print for visibility, but write an empty file to signal failure
        try:
            model.print(n_predictions=1)
        except Exception:
            pass
        temp_path.write_text("")
        print("ODEFormer returned no candidates.")
        return

    best_candidate = candidates[0]
    best_formula = str(best_candidate)
    best_score = _extract_score(best_candidate)

    payload = pd.DataFrame(
        [
            {
                "Equation": best_formula,
                "Score": best_score if best_score is not None else 0.0,
            }
        ]
    )
    payload.to_csv(temp_path, index=False)
    print(f"Best ODEFormer equation saved to {temp_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ODEFormer for symbolic regression of ODEs.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--dataset_size", type=int, help="Maximum number of rows to use (keeps earliest by time).")
    parser.add_argument("--features", type=str, help="Comma-separated list of state columns to include.")
    parser.add_argument("--temp_file", required=True, help="Path to save the best equation.")
    parser.add_argument("--beam_size", type=int, help="Beam size for decoding.")
    parser.add_argument("--beam_temperature", type=float, help="Beam temperature for decoding.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()
    data = load_dataset(args.dataset, args.dataset_size, args.features)
    find_best_formula(
        data,
        args.temp_file,
        beam_size=args.beam_size,
        beam_temperature=args.beam_temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
