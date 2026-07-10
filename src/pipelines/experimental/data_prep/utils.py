"""Shared helpers for experimental data prep scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def signed_log(values: pd.Series | np.ndarray, log10_cutoff: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    eps = 10.0 ** log10_cutoff
    return np.sign(arr) * (np.log10(np.abs(arr) + eps) - log10_cutoff)


def load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"⚠️  {label} not found: {path}")
        return None
    except Exception as exc:
        print(f"⚠️  Failed to load {label} ({path}): {exc}")
        return None
    return df
