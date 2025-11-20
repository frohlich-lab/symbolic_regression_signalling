"""Utilities for sQSSA and tQSSA Michaelis–Menten variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

EPS = 1e-20
TARGET_COLUMN = "kcat_cg"
MM_REQUIRED_COLS = ["P_u", "k_off", "k_D", "k_cat", "tK"]


@dataclass
class MMComponents:
    """Container with numeric arrays required for MM variants."""

    P_u: np.ndarray
    k_off: np.ndarray
    k_D: np.ndarray
    k_cat: np.ndarray
    tK: np.ndarray
    k_inact: Optional[np.ndarray]
    target: Optional[np.ndarray]


def _to_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()


def _components(
    df: pd.DataFrame,
    *,
    require_target: bool = False,
    target_col: str = TARGET_COLUMN,
) -> MMComponents:
    missing = [col for col in MM_REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for MM evaluation: {missing}")

    values = {col: _to_numeric(df[col]) for col in MM_REQUIRED_COLS}

    k_inact = None
    if "k_inact" in df.columns:
        raw_k_inact = _to_numeric(df["k_inact"])
        if np.isfinite(raw_k_inact).any():
            k_inact = np.where(np.isfinite(raw_k_inact), raw_k_inact, 0.0)

    target = None
    if target_col in df.columns:
        raw_target = _to_numeric(df[target_col])
        if require_target and not np.isfinite(raw_target).all():
            raise ValueError(
                f"Non-finite entries in target column '{target_col}' required for tQSSA"
            )
        target = raw_target

    if require_target and target is None:
        raise KeyError(f"Column '{target_col}' required for tQSSA evaluation but missing")

    return MMComponents(
        P_u=values["P_u"],
        k_off=values["k_off"],
        k_D=values["k_D"],
        k_cat=values["k_cat"],
        tK=values["tK"],
        k_inact=k_inact,
        target=target,
    )


def km_value(comp: MMComponents) -> np.ndarray:
    denom = np.maximum(comp.k_off * comp.k_D, EPS)
    numer = comp.k_cat + comp.k_off
    if comp.k_inact is not None:
        numer = numer + comp.k_inact
    return numer / denom


def sqssa_prediction(comp: MMComponents) -> np.ndarray:
    km = km_value(comp)
    return (comp.tK * comp.P_u) / np.maximum(comp.P_u + km, EPS)


def tqssa_prediction(comp: MMComponents) -> np.ndarray:
    if comp.target is None:
        raise ValueError("Target values are required for tQSSA prediction")
    km = km_value(comp)
    s_hat = comp.P_u + comp.target
    e_total = comp.tK
    sum_term = e_total + s_hat + km
    discriminant = np.maximum(sum_term**2 - 4 * e_total * s_hat, 0.0)
    return 0.5 * (sum_term - np.sqrt(discriminant))


def mm_predictions(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, np.ndarray]:
    comp = _components(df, require_target=True, target_col=target_col)
    return {
        "sQSSA": sqssa_prediction(comp),
        "tQSSA": tqssa_prediction(comp),
    }


def sqssa_only(df: pd.DataFrame) -> np.ndarray:
    comp = _components(df, require_target=False)
    return sqssa_prediction(comp)


def tqssa_only(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> np.ndarray:
    comp = _components(df, require_target=True, target_col=target_col)
    return tqssa_prediction(comp)


def mm_required_columns() -> list:
    return list(MM_REQUIRED_COLS)


__all__ = [
    "EPS",
    "TARGET_COLUMN",
    "MM_REQUIRED_COLS",
    "MMComponents",
    "mm_predictions",
    "sqssa_only",
    "tqssa_only",
    "mm_required_columns",
    "km_value",
    "sqssa_prediction",
    "tqssa_prediction",
]
