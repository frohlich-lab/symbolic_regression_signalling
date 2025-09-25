import matplotlib as mpl
import seaborn as sns


def apply_cell_systems_style(context: str = "paper") -> None:
    """Apply a consistent publication-style theme inspired by Cell Systems.

    - Sans-serif (Helvetica/Arial) font family
    - Modest font sizes suitable for multi-panel figures
    - Thin, black axes spines, outward ticks
    - Subtle grid (when enabled by caller)
    - High-resolution export and tight bounding box
    """
    mpl.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Lines and ticks
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # Save/export
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    # Base theme; callers can still override style/palette if needed
    sns.set_theme(style="white", context=context, palette="Set2")
    sns.set_style("white", {
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
    })

