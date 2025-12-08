"""Metric helpers shared across experimental SR scripts."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def coefficient_of_determination(
    y_true: np.ndarray | Sequence[float], y_pred: np.ndarray | Sequence[float]
) -> float:
    """Plain SSE/SST R² with finite filtering; returns NaN when undefined."""
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return float("nan")
    y = y[mask]
    yhat = yhat[mask]
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def binwise_r2(
    frame: pd.DataFrame,
    target_col: str,
    pred_col: str,
    measured_timepoints: Sequence[float] | np.ndarray = (),
    dataset_mode: str = "snapshot",
    *,
    clamp_negative: bool = False,
) -> Optional[float]:
    """Mean per-bin R² (SSE/SST) with optional negative clamping."""
    if "GFP_bin" not in frame.columns:
        return None
    r2_vals: list[float] = []
    for _, g in frame.groupby("GFP_bin"):
        y_true = pd.to_numeric(g[target_col], errors="coerce").to_numpy()
        y_pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if dataset_mode == "per_minute" and "timepoint" in g.columns and len(measured_timepoints) > 0:
            measured_mask = np.isin(g["timepoint"].to_numpy(), measured_timepoints)
            mask &= measured_mask
        if mask.sum() < 2:
            continue
        r2 = coefficient_of_determination(y_true[mask], y_pred[mask])
        if not np.isfinite(r2):
            continue
        r2_vals.append(max(0.0, r2) if clamp_negative else r2)
    return float(np.mean(r2_vals)) if r2_vals else None
