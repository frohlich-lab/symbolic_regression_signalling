import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


_SAVEFIG_PATCHED = False


def apply_cell_systems_style(context: str = "paper") -> None:
    """Apply a consistent publication-style theme inspired by Cell Systems.

    - Sans-serif (Helvetica/Arial) font family
    - Modest font sizes suitable for multi-panel figures
    - Thin, black axes spines, outward ticks
    - Subtle grid (when enabled by caller)
    - High-resolution export and tight bounding box
    """
    global _SAVEFIG_PATCHED

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

    if not _SAVEFIG_PATCHED:
        original_savefig = plt.savefig

        def savefig_dual(fname, *args, **kwargs):
            original_savefig(fname, *args, **kwargs)
            root, ext = os.path.splitext(fname)
            if ext.lower() == ".svg":
                return
            svg_path = root + ".svg"
            svg_kwargs = dict(kwargs)
            svg_kwargs.pop("format", None)
            original_savefig(svg_path, format="svg", **svg_kwargs)

        plt.savefig = savefig_dual
        _SAVEFIG_PATCHED = True
