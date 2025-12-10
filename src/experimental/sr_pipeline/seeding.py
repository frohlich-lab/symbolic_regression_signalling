"""Helpers for consistent RNG handling within the experimental SR pipeline."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from utils.seeding import resolve_seed, seed_everything


def canonicalize_seeds(
    base_seed: Optional[int],
    seeds: Optional[Sequence[int]],
    *,
    default: int = 42,
    extra_runs: int = 2,
) -> List[int]:
    """
    Return a stable, de-duplicated list of seeds.

    - If ``seeds`` is provided, preserve the original order but drop duplicates/None.
    - Otherwise, fall back to ``base_seed`` (or ``default`` when absent) and append
      the next ``extra_runs`` integers to mirror previous behaviour.
    """

    ordered: List[int] = []
    source: Iterable[Optional[int]] = seeds if seeds is not None else ()
    for raw in source:
        if raw is None:
            continue
        val = int(raw)
        if val not in ordered:
            ordered.append(val)

    if ordered:
        return ordered

    start = resolve_seed(base_seed, default=default)
    return [start + offset for offset in range(extra_runs + 1)]


def seed_all(seed: int) -> int:
    """Seed Python, NumPy (and torch if present) and return the seed."""

    return seed_everything(int(seed))

