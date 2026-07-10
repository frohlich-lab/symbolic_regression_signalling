"""Balancing utilities shared across functional-group experiments."""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.utils import resample

LOGGER = logging.getLogger("experimental.data_prep.balancing")


def _oom_bins(values: np.ndarray, log10_cutoff: float) -> np.ndarray:
    eps = 10.0 ** log10_cutoff
    return np.floor(np.log10(np.abs(values) + eps)).astype(int)


def _soft_inv_sqrt_upsample(
    y_train: np.ndarray,
    log10_cutoff: float,
    mean_factor: float = 1.0,
    max_factor: float = 4.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = _oom_bins(y_train, log10_cutoff)
    uniq, counts = np.unique(bins, return_counts=True)
    c = {b: float(n) for b, n in zip(uniq, counts)}
    w = np.array([1.0 / np.sqrt(max(c[b], 1.0)) for b in bins])
    w /= w.mean() / mean_factor
    w = np.clip(w, 1.0, max_factor)
    base, frac = np.floor(w).astype(int), w - np.floor(w)
    copies = base + rng.binomial(1, frac).astype(int)
    idx = np.arange(len(y_train))
    return np.repeat(idx, copies)


def balance_by_order_of_magnitude(
    df: pd.DataFrame,
    target_column: str,
    log10_cutoff: float,
    min_samples: int,
    max_samples: int,
    random_state: int,
    *,
    soft_upsample: bool = True,
    max_dup_factor: float = 4.0,
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
            balanced = group
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
    if soft_upsample:
        idx = _soft_inv_sqrt_upsample(
            combined[target_column].to_numpy(dtype=float),
            log10_cutoff,
            mean_factor=1.0,
            max_factor=max_dup_factor,
            seed=random_state,
        )
        combined = combined.iloc[idx].reset_index(drop=True)
    return combined.drop(columns=[f"oom_bins_{target_column}"])
