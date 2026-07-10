"""Shared helpers for consistent seed annotation on experimental plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SEED_NOTE = "Seed: single"


def _dedupe(seq: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def seeds_from_df(df: Optional[pd.DataFrame], columns: Sequence[str] = ("seed", "random_state")) -> List[str]:
    """Collect seed-like values from a dataframe."""

    if df is None or df.empty:
        return []
    seeds: List[str] = []
    for col in columns:
        if col in df.columns:
            seeds.extend([str(s) for s in df[col].dropna().unique()])
    return _dedupe(seeds)


def seeds_from_paths(paths: Iterable[Path]) -> List[str]:
    """Extract seed ids from paths containing .../seed_<id>/... segments."""

    seeds: List[str] = []
    for path in paths:
        for part in path.parts:
            lower = part.lower()
            if "seed_" in lower:
                token = lower.split("seed_", 1)[1]
                token = token.split("/", 1)[0]
                if token:
                    seeds.append(token)
    return _dedupe(seeds)


def format_seed_label(
    seeds: Sequence[object],
    *,
    averaged: bool = False,
    default: Optional[str] = DEFAULT_SEED_NOTE,
) -> Optional[str]:
    """Format a human-readable seed label for figure annotations."""

    cleaned = [str(s) for s in seeds if pd.notna(s)]
    if not cleaned:
        return default
    if len(cleaned) == 1 and not averaged:
        return f"Seed: {cleaned[0]}"
    joined = ", ".join(cleaned)
    return f"Averaged over seeds: {joined}" if averaged else f"Seeds: {joined}"


def add_seed_note(
    fig: plt.Figure,
    *,
    seeds: Sequence[object] = (),
    averaged: bool = False,
    note: Optional[str] = None,
    default: Optional[str] = DEFAULT_SEED_NOTE,
) -> None:
    """Write a seed/aggregation note onto a figure."""

    final_note = note if note is not None else format_seed_label(seeds, averaged=averaged, default=default)
    if final_note:
        fig.text(0.01, 0.01, final_note, ha="left", va="bottom", fontsize=9, color="#444444")


def pick_seed(
    df: Optional[pd.DataFrame],
    *,
    preferred: Optional[object] = None,
    columns: Sequence[str] = ("seed", "random_state"),
) -> Optional[str]:
    """Pick a deterministic seed from a dataframe or return the preferred value."""

    if preferred is not None:
        return str(preferred)
    seeds = seeds_from_df(df, columns=columns)
    if seeds:
        return sorted(seeds, key=str)[0]
    return None

