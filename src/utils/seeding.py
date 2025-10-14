"""Utility helpers for consistent RNG seeding across libraries."""

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed:
        The integer seed to propagate.
    deterministic_torch:
        When ``True`` and PyTorch is available, enable deterministic CuDNN
        kernels (at the cost of potential slowdown).

    Returns
    -------
    int
        The seed value, for convenience/chaining.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # noqa: WPS433 (optional dependency)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch absent in some envs
        pass

    return seed


def resolve_seed(seed: Optional[int], default: int = 42) -> int:
    """Return ``seed`` if not ``None`` otherwise ``default``."""

    return default if seed is None else seed
