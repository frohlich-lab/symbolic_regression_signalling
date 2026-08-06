"""Shared rcParams for the supplementary figures.

Matches the main-text panels: Helvetica/Arial at 6-8 pt, hairline spines, no
top/right spine, Type-42 embedded fonts. Arial is listed first because macOS
only registers Helvetica's regular face, so a bold Helvetica request falls back
silently to DejaVu and the figure stops matching the manuscript.
"""
from __future__ import annotations

import matplotlib as mpl

# Point sizes used across every supplementary panel.
FS_LABEL = 7.0    # axis labels, tick labels, annotations
FS_TICK = 6.0     # tick labels where space is tight
FS_PANEL = 8.0    # bold panel letters

# Palette lifted from the main-text figures.
INK = "#2f3b3b"          # data markers
GREEN_FILL = "#cdecd9"   # "SR wins" half-plane (Fig. 4D)
BLUE_FILL = "#cfe3f2"    # "both fail" quadrant (Fig. 4E)
YELLOW_FILL = "#f6e6bd"  # "NODE only" quadrant (Fig. 4E)
GRID = "#e8e7e3"
ACCENT = "#b4553f"       # highlighted / retained setting
MUTED = "#8a9494"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    })


def panel_letter(ax, letter: str, x: float = -0.22, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="bottom", ha="left", color="#111111")


def save(fig, path) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {path} and {path.with_suffix('.pdf')}")
