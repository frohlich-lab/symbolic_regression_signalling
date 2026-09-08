"""Fig. 4 alternative: accuracy scatter + a single complexity panel.

Replaces the two right-hand panels of fig_4ef_accuracy_and_parsimony (Parsimony and
Interactions, each a boxplot over quadrant groups) with ONE scatter that shows both
complexity axes at once:

    x = effective dependency count   (variables the network's rate law depends on)
    y = interacting pairs            (off-diagonal Hessian count among those drivers)
    size = SR held-out R^2           (the outcome, encoded on the marks themselves)
    colour = SR above/below the 0.6 threshold

Restricted to contexts where the neural ODE generalises out of distribution
(R^2 > 0.6), i.e. where a compact description demonstrably exists -- so every point
is a context in which SR's success or failure is interpretable.

SR values are the frozen production configuration, seed chosen per marker by train parsimony (the printed
figure's rule; no held-out data is used to select),
read from the published run: the specification with the strongest contrast.

Colour uses slots 1-2 of the reference categorical palette in fixed order. The
palette validator could not be run here (no node runtime); the pair is the
palette's own validated adjacent pair, and size redundantly encodes the same
quantity so identity is never carried by colour alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SOLVED = "#2a78d6"   # palette slot 1 (blue)
FAILED = "#eb6834"   # palette slot 2 (orange)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
THR = 0.6


def bubble_area(r2: np.ndarray) -> np.ndarray:
    """Marker area from SR held-out R^2. Floor keeps zero-R^2 contexts visible."""
    return 44.0 + 400.0 * np.clip(r2, 0.0, 1.0) ** 1.5


def _overlap(a, b) -> float:
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _hline_box(ax, y, pad=4.0):
    """Display-coord band around a horizontal reference line, so labels don't sit on it."""
    from matplotlib.transforms import Bbox
    if y is None:
        return None
    ab = ax.get_window_extent()
    py = ax.transData.transform((ax.get_xlim()[0], y))[1]
    return Bbox.from_extents(ab.x0, py - pad, ab.x1, py + pad)


def place_labels(ax, points, obstacles=(), fontsize=8.5, color=INK2):
    """Annotate `points` [(x, y, text, area)], choosing per-label offsets that avoid the
    marks, the given obstacles (legends, captions) and the labels already placed.
    Small label sets only -- greedy, no solver."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    occupied = [o if isinstance(o, Bbox) else o.get_window_extent(renderer)
                for o in obstacles if o is not None]
    for x, y, _txt, area in points:
        px, py = ax.transData.transform((x, y))
        r = np.sqrt(area / np.pi) * fig.dpi / 72.0
        occupied.append(Bbox.from_extents(px - r, py - r, px + r, py + r))

    cands = [(9, 6, "left", "bottom"), (9, -6, "left", "top"),
             (-9, 6, "right", "bottom"), (-9, -6, "right", "top"),
             (0, 11, "center", "bottom"), (0, -11, "center", "top"),
             (13, 0, "left", "center"), (-13, 0, "right", "center")]
    for x, y, txt, _area in points:
        best, best_cost = None, None
        for dx, dy, ha, va in cands:
            ann = ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points",
                              fontsize=fontsize, color=color, ha=ha, va=va, zorder=6)
            bb = ann.get_window_extent(renderer)
            cost = sum(_overlap(bb, o) for o in occupied)
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost = ann, cost
            else:
                ann.remove()
            if best_cost == 0.0:
                break
        if best is not None:
            occupied.append(best.get_window_extent(renderer))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-inputs", required=True,
                    help="--dump-inputs CSV from plot_parsimony_tradeoff.py: marker, nn_r2, "
                         "nn_pr, nn_inter and py_r2. Its py_r2 is the seed chosen by train "
                         "parsimony, which is the printed figure's rule and uses no held-out "
                         "data -- so it is what this figure uses.")
    ap.add_argument("--pysr-per-fit",
                    help="DEPRECATED and unsafe. Takes max held-out R2 over seeds, which is "
                         "test-set selection on the seed axis; it inflated PySR by 4 of 40 "
                         "contexts and flipped EGFR from fail to pass. Only honoured with "
                         "--allow-testset-seed-selection.")
    ap.add_argument("--allow-testset-seed-selection", action="store_true",
                    help="Opt in to the leaky --pysr-per-fit path. Diagnostics only.")
    ap.add_argument("--output", required=True, help="PNG path; a PDF is written alongside.")
    ap.add_argument("--nn-threshold", type=float, default=THR)
    ap.add_argument("--include-controls", action="store_true",
                    help="Keep FLAG-GFP / untransfected controls. Off by default: the "
                         "claim is about perturbation contexts, and including controls "
                         "adds untransfected4 as a third low-complexity SR failure.")
    args = ap.parse_args()

    CONTROLS = {f"untransfected{i}" for i in (1, 2, 3, 4)} | {f"FLAG-GFP{i}" for i in (1, 2, 3, 4)}

    d = pd.read_csv(args.panel_inputs).set_index("marker")
    if args.allow_testset_seed_selection and args.pysr_per_fit:
        fit = pd.read_csv(args.pysr_per_fit)
        d["sr"] = fit.groupby("marker").ode_integ_r2_median.max()
        print("  WARNING: seed chosen by MAX HELD-OUT R2 (test-set selection). "
              "Diagnostic only -- not a result.")
    elif "py_r2" in d.columns:
        d["sr"] = d.py_r2
    else:
        raise SystemExit("--panel-inputs has no py_r2 column; regenerate it with "
                         "plot_parsimony_tradeoff.py --sr-parsimony --dump-inputs")
    d["is_control"] = [m in CONTROLS for m in d.index]
    d = d.dropna(subset=["sr", "nn_r2", "nn_pr", "nn_inter"])

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "axes.edgecolor": INK2, "axes.linewidth": 0.9,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.0))

    # ---- panel A: accuracy out of distribution (unchanged in spirit from Fig. 4E) ----
    axA.axhline(THR, color=GRID, lw=1.0, ls="--", zorder=1)
    axA.axvline(THR, color=GRID, lw=1.0, ls="--", zorder=1)
    axA.plot([0, 1], [0, 1], color=GRID, lw=1.0, ls=":", zorder=1)
    axA.scatter(d.nn_r2, d.sr.clip(lower=0), s=46, c=INK, alpha=0.85,
                edgecolors="white", linewidths=0.8, zorder=3)
    axA.set_xlabel("Neural ODE held-out R²")
    axA.set_ylabel("SR held-out R²")
    axA.set_title("Accuracy out of distribution", loc="left", pad=10)
    axA.set_xlim(-0.04, 1.04)
    axA.set_ylim(-0.04, 1.04)
    axA.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # ---- panel B: the two complexity axes at once, only where the network generalises ----
    # Panel A keeps every context, as printed. The restriction applies to panel B only:
    # the claim is about perturbation contexts, and the untransfected/FLAG-GFP controls
    # add untransfected4 as a third low-complexity SR failure.
    g = d[d.nn_r2 > args.nn_threshold].copy()
    if not args.include_controls:
        drop = list(g.index[g.is_control.fillna(False)])
        g = g[~g.is_control.fillna(False)]
        print(f"  panel B excludes {len(drop)} controls: {', '.join(drop) or 'none'}")
    g["ok"] = g.sr > THR
    g["area"] = bubble_area(g.sr.values)

    # Largest first, so the small marks -- which are the SR failures, small by
    # construction -- end up on top instead of hidden underneath a success.
    o = g.sort_values("area", ascending=False)
    axB.scatter(o.nn_pr, o.nn_inter, s=o.area,
                c=[SOLVED if v else FAILED for v in o.ok],
                alpha=0.85, edgecolors="white", linewidths=1.3, zorder=3)
    axB.margins(0.15)      # headroom for the labels; must precede place_labels

    # the ceiling set by the contexts SR solves -- the readable form of the contrast
    ceil = g[g.ok].nn_inter.max() if g.ok.any() else None
    cap = None
    if ceil is not None:
        axB.axhline(ceil, color=INK2, lw=0.9, ls="--", zorder=2)
        cap = axB.annotate(f"SR-success ceiling ({ceil:.1f} pairs)",
                           xy=(0.985, ceil), xycoords=("axes fraction", "data"),
                           xytext=(0, 6), textcoords="offset points",
                           ha="right", va="bottom", fontsize=9, color=INK2)

    axB.set_xlabel("Effective dependency count (variables)")
    axB.set_ylabel("Interacting input pairs")
    sub = "perturbation contexts" if not args.include_controls else "all contexts"
    axB.set_title(f"Complexity of the dynamics\n{sub}, neural ODE R² > {args.nn_threshold:g}",
                  loc="left", pad=10, linespacing=1.5)
    axB.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    def dot(colour, size, label):
        return Line2D([], [], marker="o", linestyle="none", markersize=np.sqrt(size),
                      markerfacecolor=colour, markeredgecolor="white",
                      markeredgewidth=1.0, label=label)

    leg1 = axB.legend(handles=[dot(SOLVED, 150, "SR recovers a law"),
                               dot(FAILED, 150, "SR fails")],
                      loc="upper left", frameon=False, fontsize=9.5,
                      handletextpad=0.6, borderpad=0.2)
    axB.add_artist(leg1)
    leg2 = axB.legend(handles=[dot("#b9b8b2", bubble_area(np.array([v]))[0], f"{v:.1f}")
                               for v in (0.0, 0.4, 0.8)],
                      title="SR held-out R²", loc="lower right",
                      frameon=False, fontsize=9, title_fontsize=9, labelspacing=1.1,
                      handletextpad=0.9, borderpad=0.4)

    # Labels last, so they can dodge the marks, the caption and both legends.
    # Selective: the SR failures at or below the success ceiling (the exceptions to
    # the pattern) plus the most complex contexts -- never a label on every point.
    lab = list(g[(~g.ok) & (g.nn_inter <= ceil)].index) if ceil is not None else []
    lab += list(g.nn_inter.nlargest(3).index) + list(g[g.ok].nn_inter.nlargest(2).index)
    # A solved context sitting on a failure's exact coordinates is the panel's most
    # important feature -- no complexity measure can separate the pair. Label it too,
    # or the two marks read as one point.
    coincident = [m for m in g.index if m not in lab and any(
        abs(g.loc[m, "nn_pr"] - g.loc[o, "nn_pr"]) < 1e-9
        and abs(g.loc[m, "nn_inter"] - g.loc[o, "nn_inter"]) < 1e-9 for o in lab)]
    lab += coincident
    place_labels(axB, [(g.loc[m, "nn_pr"], g.loc[m, "nn_inter"], m, g.loc[m, "area"])
                       for m in dict.fromkeys(lab)],
                 obstacles=[cap, leg1, leg2, _hline_box(axB, ceil)])

    fig.tight_layout(w_pad=2.4)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} and {out.with_suffix('.pdf')}")
    print(f"  contexts plotted (panel B): {len(g)}  solved {int(g.ok.sum())}  failed {int((~g.ok).sum())}")
    print(f"  pairs: solved {g[g.ok].nn_inter.mean():.2f}  failed {g[~g.ok].nn_inter.mean():.2f}")


if __name__ == "__main__":
    main()
