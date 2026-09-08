"""Main-text figure: PySR vs Neural ODE (L21) per-marker R² scatter + side PR boxplot.

For each marker, takes the best-of-3-seeds OOD test R² for both methods and plots
the scatter. The side boxplot shows the effective-driver count (participation
ratio of |J|) on the markers where each method works (R² > threshold).

Saves to --output (png) and the corresponding .pdf next to it.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats

# Allow `from pipelines.experimental.sr_pipeline...` regardless of CWD.
_repo_src = Path(__file__).resolve().parents[4]
if str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_names import gene_label  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import sympy as sp  # noqa: E402

from pipelines.experimental.sr_pipeline.neural_ode_diffrax_causal import (  # noqa: E402
    _load_marker_checkpoint,
    _per_marker_causal,
    _safe,
)

FEATURES = [
    "GFP",
    "p_ERK1_2",
    "p_ERK1_2_min",
    "p_MEK1_2",
    "p_MEK1_2_min",
    "p_RAF",
    "p_p90RSK",
    "p_MAPKAPK2",
    "p_PDK1",
    "p_MKK3_6",
]

# Copied from plot_pysr_vs_neural_ode.py rather than imported: that module
# applies a seaborn style at import time, which would clobber the type and
# grid settings this figure sets.
def _annotate_non_overlapping(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    *,
    min_r2: float = 0.6,
    fontsize: int = 12,
    point_keepout_px: float = 10.0,
    pad_px: float = 2.0,
    max_labels: int | None = None,
) -> None:
    """
    Directional offset placement with a penalty heuristic (no arrows).
    """
    d = df.copy()
    d = d[np.isfinite(d[x_col]) & np.isfinite(d[y_col])]
    d = d[(d[[x_col, y_col]].max(axis=1) >= min_r2)]
    if d.empty:
        return
    d = d.sort_values(by=[y_col, x_col], ascending=False)
    if max_labels is not None:
        d = d.head(max_labels)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax_bbox = ax.get_window_extent(renderer=renderer)
    placed: List[mtransforms.Bbox] = []

    dirs = [
        (1, 1, "left", "bottom"),
        (0, 1, "center", "bottom"),
        (-1, 1, "right", "bottom"),
        (1, 0, "left", "center"),
        (-1, 0, "right", "center"),
        (1, -1, "left", "top"),
        (0, -1, "center", "top"),
        (-1, -1, "right", "top"),
    ]
    radii = [10, 15, 20, 27, 36, 48, 62, 80]

    def expanded_bbox(t):
        b = t.get_window_extent(renderer=renderer)
        return mtransforms.Bbox.from_extents(b.x0 - pad_px, b.y0 - pad_px, b.x1 + pad_px, b.y1 + pad_px)

    def overlaps_any(b):
        return any(b.overlaps(bb) for bb in placed)

    def hits_own_point(b, x, y):
        px, py = ax.transData.transform((x, y))
        keep = mtransforms.Bbox.from_extents(px - point_keepout_px, py - point_keepout_px, px + point_keepout_px, py + point_keepout_px)
        return b.overlaps(keep)

    def outside_axes(b):
        return (b.x0 < ax_bbox.x0) or (b.x1 > ax_bbox.x1) or (b.y0 < ax_bbox.y0) or (b.y1 > ax_bbox.y1)

    for _, r in d.iterrows():
        x = float(r[x_col])
        y = float(r[y_col])
        label = gene_label(r[label_col])

        best = None
        best_bbox = None
        best_score = None

        for rad in radii:
            for sx, sy, ha, va in dirs:
                dx, dy = sx * rad, sy * rad
                t = ax.annotate(
                    label,
                    xy=(x, y),
                    xycoords="data",
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=fontsize,
                    ha=ha,
                    va=va,
                    arrowprops=None,
                )
                fig.canvas.draw()
                b = expanded_bbox(t)

                if overlaps_any(b) or hits_own_point(b, x, y):
                    t.remove()
                    continue

                penalty = 0.0
                if outside_axes(b):
                    penalty += 1000.0
                penalty += (abs(dx) + abs(dy)) * 0.1
                penalty += max(0.0, ax_bbox.x0 - b.x0) + max(0.0, b.x1 - ax_bbox.x1)
                penalty += max(0.0, ax_bbox.y0 - b.y0) + max(0.0, b.y1 - ax_bbox.y1)

                if best_score is None or penalty < best_score:
                    if best is not None:
                        best.remove()
                    best = t
                    best_bbox = b
                    best_score = penalty
                else:
                    t.remove()
            if best is not None:
                break

        if best is None:
            best = ax.annotate(
                label,
                xy=(x, y),
                xycoords="data",
                xytext=(-90, -90),
                textcoords="offset points",
                fontsize=fontsize,
                ha="right",
                va="top",
                arrowprops=None,
            )
            fig.canvas.draw()
            best_bbox = expanded_bbox(best)

        placed.append(best_bbox)

MEASURED_DEFAULT = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
SAVE_DPI = 300   # also the dpi the bracket rule is measured at


def pr_of(v: np.ndarray) -> float:
    v = np.abs(np.asarray(v)).ravel()
    s, sq = v.sum(), (v ** 2).sum()
    return float(s * s / sq) if sq > 0 else 0.0


def _parse_pysr(path: Path) -> dict[str, tuple[float, str, float]]:
    """marker -> (test R2, formula, train R2). Train R2 is carried so a seed can be
    chosen by PySR's parsimony score without any held-out information entering."""
    text = path.read_text()
    out: dict[str, tuple[float, str, float]] = {}
    for block in text.split("\nGroup: "):
        if not block.strip():
            continue
        if not block.startswith("Group:"):
            block = "Group: " + block
        m = re.search(r"Group:\s*(.+)", block)
        r2 = re.search(r"Test R2:\s*([-+0-9.eE]+)", block)
        tr = re.search(r"Train R2:\s*([-+0-9.eE]+)", block)
        f = re.search(r"Formula:\s*(.+)", block)
        if m and r2 and f:
            out[m.group(1).strip()] = (float(r2.group(1)), f.group(1).strip(),
                                       float(tr.group(1)) if tr else float("nan"))
    return out


def _formula_complexity(formula: str) -> float:
    """Node count of a recovered expression, PySR's own complexity convention."""
    try:
        e = sp.sympify(formula)
        return float(sp.count_ops(e) + len(e.atoms(sp.Symbol)) + len(e.atoms(sp.Number)))
    except Exception:
        return float("nan")



def _perm_p(values: np.ndarray, is_fail: np.ndarray, covariate: np.ndarray,
            max_exact: int = 200000) -> float:
    """Two-sided permutation p for the group difference, after regressing out `covariate`.

    Exhaustive over all label assignments when C(n, k) is small enough (it is at these
    sample sizes), so the p-value is exact rather than sampled.
    """
    import itertools
    v = np.asarray(values, float); f = np.asarray(is_fail, bool)
    x = np.asarray(covariate, float)
    n, nf = len(v), int(f.sum())
    if nf < 2 or n - nf < 2:
        return float("nan")
    X = np.column_stack([np.ones(n), x])
    res = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    obs = res[f].mean() - res[~f].mean()
    idx = range(n)
    from math import comb
    if comb(n, nf) <= max_exact:
        stats_ = np.array([res[list(c)].mean() - res[[i for i in idx if i not in c]].mean()
                           for c in itertools.combinations(idx, nf)])
    else:
        rng = np.random.default_rng(0)
        stats_ = np.empty(20000)
        for j in range(len(stats_)):
            pm = rng.permutation(n)
            stats_[j] = res[pm[:nf]].mean() - res[pm[nf:]].mean()
    return float((np.abs(stats_) >= abs(obs) - 1e-12).mean())


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.s."
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _feature_columns(data: pd.DataFrame) -> dict[str, str]:
    """Map each model input to its column in the per-minute table.

    The SR symbols are Julia-safe (`p_MEK1_2`); the table keeps the readout names
    (`p-MEK1-2_fit`, `p-MEK1-2_min`). Every input must resolve -- a missing one used to
    fall back to a column of ones, which sends any law with a difference of two readouts
    in a denominator through a pole and returns a zero Jacobian.
    """
    out: dict[str, str] = {}
    for f in FEATURES:
        if f.endswith("_min"):
            candidates = [f.replace("_", "-")[:-4] + "_min", f]
        else:
            hyphen = f.replace("_", "-")
            candidates = [f, hyphen, f"{hyphen}_fit"]
        for c in candidates:
            if c in data.columns:
                out[f] = c
                break
        else:
            raise SystemExit(f"{f}: no column among {candidates} in the per-minute table")
    return out


def _pysr_mean_abs_jac(formula: str, marker: str, data: pd.DataFrame) -> np.ndarray | None:
    """Mean |df/dx| over the marker's training rows, one entry per model input.

    Row-wise rather than at the mean point: several recovered laws put a pole near the
    marker's mean (a difference of two correlated phospho-readouts in a denominator), where
    a single-point evaluation is undefined and would silently return a zero vector.
    """
    syms = {f: sp.Symbol(f) for f in FEATURES}
    try:
        expr = sp.sympify(formula, locals=syms)
    except Exception:
        return None
    sub = data[data.get("marker") == marker] if "marker" in data.columns else data
    if len(sub) == 0:
        sub = data
    cols = {f: sub[c].to_numpy(dtype=float) for f, c in _feature_columns(sub).items()}
    j = np.zeros(len(FEATURES))
    for i, fi in enumerate(FEATURES):
        try:
            d = sp.lambdify([syms[f] for f in FEATURES], sp.diff(expr, syms[fi]), "numpy")
            v = np.asarray(d(*[cols[f] for f in FEATURES]), dtype=float)
            v = np.broadcast_to(v, (len(sub),))
            v = v[np.isfinite(v)]
            j[i] = float(np.abs(v).mean()) if v.size else 0.0
        except Exception:
            pass
    return j


def _pysr_interaction_count(formula: str, marker: str, data: pd.DataFrame,
                            frac: float = 0.25, jac: np.ndarray | None = None,
                            dep_frac: float = 0.15) -> float | None:
    """Interacting input pairs in a recovered symbolic law.

    The same operator applied to the network in the interaction panel: mean |d2f/dxi dxj|
    over the marker's training rows, counting off-diagonal entries above `frac` of the
    largest curvature. Computing it on the recovered expression as well makes the symbolic
    and network boxes strictly comparable -- one measure, two laws.
    """
    syms = {f: sp.Symbol(f) for f in FEATURES}
    try:
        expr = sp.sympify(formula, locals=syms)
    except Exception:
        return None
    sub = data[data.get("marker") == marker] if "marker" in data.columns else data
    if len(sub) == 0:
        sub = data
    cols = {f: sub[c].to_numpy(dtype=float) for f, c in _feature_columns(sub).items()}
    n = len(FEATURES)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            try:
                d2 = sp.diff(expr, syms[FEATURES[i]], syms[FEATURES[j]])
                if d2 == 0:
                    continue
                fn = sp.lambdify([syms[f] for f in FEATURES], d2, "numpy")
                v = np.asarray(fn(*[cols[f] for f in FEATURES]), dtype=float)
                v = np.broadcast_to(v, (len(sub),))
                v = v[np.isfinite(v)]
                m = float(np.abs(v).mean()) if v.size else 0.0
            except Exception:
                m = 0.0
            H[i, j] = H[j, i] = m
    # Score off-diagonals against the OFF-DIAGONAL maximum, not the global one: recovered
    # laws containing divisions carry diagonal curvature (2c/x^3) that dwarfs every
    # cross-term, so a global reference suppresses their interaction counts relative to a
    # tanh network's. Both function classes are then measured on their own interaction scale.
    # Restricted to the inputs the dependency criterion retains, which is what the network
    # side does (compute_interaction_counts.py scores H[ix_(idx, idx)]). Scoring the law
    # over all 45 pairs instead counted cross-terms carried by inputs too weak to be
    # dependencies at all, so the symbolic box was measured on a wider index set than the
    # boxes beside it -- one measure, two laws only if the index set matches too.
    if jac is not None:
        a = np.abs(np.asarray(jac)).ravel()
        mx = a.max() if a.size else 0.0
        sel = np.where(a > dep_frac * mx)[0] if mx > 0 else np.array([], dtype=int)
        sub = H[np.ix_(sel, sel)]
        off = sub[np.triu_indices(len(sel), 1)] if len(sel) > 1 else np.array([])
    else:
        off = H[np.triu_indices(n, 1)]
    omax = off.max() if off.size else 0.0
    if omax <= 0:
        return 0.0
    return float((off > frac * omax).sum())


def _group_bracket(ax, tick_labels, name_text, color, fs, save_dpi, rule_y=None):
    """Place a labelled rule under a group of box positions, measured off rendered ink.

    Both box panels need this and neither can use fixed coordinates. Vertically, text
    bounding boxes include ascent and descent, so "n=9" (no descender) and "Neural ODE"
    (no ascender above cap height) sit unequally inside their boxes and splitting the
    boxes puts the rule visibly low; splitting the ink is what "in the middle" means.
    Horizontally, "n=12" is wider than "n=9", so the label pair's ink centre lies right
    of the midpoint of the tick positions and a rule spanning the ticks overhangs on the
    left. Span the labels themselves with equal margins, and hang the name off that centre.

    Returns the axes-fraction height it used. Pass that back as `rule_y` for the second
    panel: both box panels share a y position and a height, so one measurement is correct
    for both, and measuring twice let the two rules land 8 px apart -- the ink scan picks
    the last blank run, and the two panels' label blocks bound it differently.
    """
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    name = ax.text(0.0, -0.175, name_text, transform=tr, color=color, fontsize=fs,
                   fontweight="bold", ha="center", va="top", clip_on=False, zorder=6)
    fig = ax.figure
    # Measure at the dpi the file is written at: at the default canvas dpi one measured
    # row is three saved pixels, and the rounding shows as a visible offset.
    fig.set_dpi(save_dpi)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    H = buf.shape[0]

    # Horizontal span from the labels' own boxes, not from pixels: the panel may sit past
    # the canvas edge (bbox_inches="tight" recovers it at save time), so a pixel scan can
    # silently miss the right-hand label. Horizontal boxes hug the glyphs closely enough.
    MARGIN_PX = 0.55 * fs * save_dpi / 72.0
    lo = min(t.get_window_extent().x0 for t in tick_labels) - MARGIN_PX
    hi = max(t.get_window_extent().x1 for t in tick_labels) + MARGIN_PX
    inv_x = ax.transData.inverted()
    rx0 = inv_x.transform((lo, 0))[0]
    rx1 = inv_x.transform((hi, 0))[0]
    name.set_x(0.5 * (rx0 + rx1))

    # The name's -0.175 is a fixed axes fraction, so raising the type shrinks the band
    # between the tick labels and the name until the rule has nowhere to sit -- at
    # FS 15.9 it collapsed and the rule ran into both. Re-hang the name a
    # type-proportional distance below the labels, then put the rule at the midpoint of
    # the band. A bbox midpoint rather than an ink scan: the scan centred on the last
    # blank run, which after this change is the whole band below the labels, so the rule
    # rode up against them.
    tick_bb = mtransforms.Bbox.union([t.get_window_extent() for t in tick_labels])
    gap_px = 0.80 * fs * save_dpi / 72.0
    name.set_y(ax.transAxes.inverted().transform((0, tick_bb.y0 - gap_px))[1])
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    name_bb = name.get_window_extent()
    # Centre on INK, not on the bounding boxes. The tick labels' box runs below their
    # glyphs (descent) and the name's runs above its cap height (ascent), so a box
    # midpoint sat 52 px under the numbers and 38 px over the name -- visibly off centre
    # in a band that should read as one unit.
    xa, xb = sorted((int(tick_bb.x0), int(tick_bb.x1)))
    xa, xb = max(xa, 0), min(xb, buf.shape[1])
    r_from = max(int(np.floor(H - tick_bb.y1)), 0)
    r_to = min(int(np.ceil(H - name_bb.y0)), H)
    measured = ax.transAxes.inverted().transform(
        (0, 0.5 * (tick_bb.y0 + name_bb.y1)))[1]        # fallback
    if r_to > r_from and xb > xa:
        inked = (buf[r_from:r_to, xa:xb] < 250).any(axis=(1, 2))
        rows = np.flatnonzero(inked)
        if len(rows) >= 2:
            blank = np.flatnonzero(~inked)
            blank = blank[(blank > rows[0]) & (blank < rows[-1])]
            if len(blank):
                runs = np.split(blank, np.flatnonzero(np.diff(blank) > 1) + 1)
                band = runs[-1]
                row = r_from + (float(band[0]) - 1.0 + float(band[-1]) + 1.0) / 2.0
                measured = ax.transAxes.inverted().transform((0, H - row))[1]

    y = measured if rule_y is None else rule_y
    ax.plot([rx0, rx1], [y, y], transform=tr, color=color, lw=0.9,
            clip_on=False, zorder=6)
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-dir", required=True,
                    help="Directory holding seed_42/, seed_43/, seed_44/ for the Neural ODE method (L21).")
    ap.add_argument("--pysr-dir", required=True,
                    help="Directory holding seed_42/formulas/all_per_minute.txt etc.")
    ap.add_argument("--dataset", required=True,
                    help="markers_per_minute_fit.csv (for evaluating PySR partials at training mean).")
    ap.add_argument("--output", required=True, help="PNG output path. PDF is written next to it.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--r2-threshold", type=float, default=0.6)
    ap.add_argument("--exclude-marker", action="append", default=[])
    ap.add_argument("--box-width-frac", type=float, default=0.70,
                    help="Width of the parsimony box panel as a fraction of the "
                         "drawn scatter width.")
    ap.add_argument("--panel-gap", type=float, default=0.080,
                    help="Gutter between adjacent panels, as a fraction of figure width, "
                         "measured between tight bounding boxes so multi-line axis "
                         "labels count toward the spacing.")
    ap.add_argument("--box-gap", type=float, default=0.46,
                    help="Gap between the scatter and the box panel, as a "
                         "fraction of the scatter's drawn width. Wide enough "
                         "that the box reads as its own panel rather than as a "
                         "marginal of the scatter. Relative to the scatter, not "
                         "to the figure, so the one-scatter and two-scatter "
                         "layouts (which differ in total width) space the box "
                         "the same way.")
    ap.add_argument("--linreg-metrics",
                    help="select_k_metrics.csv from the top_gfp_bins OOD run. When given, the "
                         "SR-vs-linear-regression panel (Fig. 4D) is drawn as a third panel in "
                         "THIS figure rather than as a separate one, so all panels share a "
                         "single type scale. Drawn by plot_sr_vs_linreg_ood.draw_accuracy_panel "
                         "-- one code path, so the panel cannot drift between the two figures.")
    ap.add_argument("--linreg-k", type=int, default=10,
                    help="Which k for the linear-regression baseline (10 = all features).")
    ap.add_argument("--no-panel-titles", action="store_true",
                    help="Drop the per-panel titles. For the composed figure, where the "
                         "panels are lettered in the manuscript instead.")
    ap.add_argument("--no-tick-legend", action="store_true",
                    help="Drop the tick-mark key beside the parsimony panel.")
    ap.add_argument("--label-skip", action="append", default=[],
                    help="Context to leave unlabelled in the linreg scatter. The dot is "
                         "still drawn; only the callout is dropped. Unlike "
                         "--exclude-marker this changes no number in any panel.")
    ap.add_argument("--dump-inputs",
                    help="Optional CSV to write the per-marker panel inputs to "
                         "(marker, nn_r2, py_r2, nn_pr, py_pr). Lets the numbers quoted in "
                         "the text be regenerated instead of recomputed by hand.")
    ap.add_argument("--symbolic-interaction-box", action="store_true",
                    help="Also box the recovered symbolic laws' interaction counts. Off "
                         "by default: laws containing divisions have diagonal curvature "
                         "that dwarfs their cross-terms, so counts thresholded against "
                         "the global Hessian maximum are biased low relative to the "
                         "network and the two are not on a common scale.")
    ap.add_argument("--node-integrated-r2", action="store_true",
                    help="Score the neural ODE by ODE-integrated R2, the same quantity "
                         "the SR axis uses, instead of the trajectory mean.")
    ap.add_argument("--select-on-val", action="store_true",
                    help="Choose the neural-ODE seed on validation R2 and report its test "
                         "score, instead of taking the best test score across seeds.")
    ap.add_argument("--sr-parsimony", action="store_true",
                    help="Choose the PySR seed by its parsimony score on training data "
                         "instead of the best held-out R2 across seeds.")
    ap.add_argument("--sr-box-quadrant", action="store_true",
                    help="Restrict the SR box (and its interaction box) to the scatter's "
                         "top-right quadrant, so it covers the same contexts as the "
                         "neural-ODE box beside it and the bracket is a paired comparison.")
    ap.add_argument("--sr-best-train", action="store_true",
                    help="Choose the PySR seed by training R2 (the rule used by Table S8 "
                         "and the Results text). Takes precedence over --sr-parsimony.")
    ap.add_argument("--dependency-count", action="store_true",
                    help="Score parsimony as the number of inputs carrying at least "
                         "--count-threshold of the largest mean |dF/dx| instead of the "
                         "participation ratio. The count is the calibrated measure "
                         "(slope 0.95 against synthetic dynamics with known drivers).")
    ap.add_argument("--count-threshold", type=float, default=0.15,
                    help="Sensitivity cutoff for --dependency-count, as a fraction of the "
                         "largest input sensitivity.")
    ap.add_argument("--interaction-csv",
                    help="Per-marker interaction-term counts (columns: marker, p0.25). "
                         "Adds a second boxplot panel contrasting interaction structure "
                         "on the markers SR solves vs those it fails.")
    ap.add_argument("--pysr-integ-csv",
                    help="Put PySR on the ODE-INTEGRATED R² axis, read from this CSV "
                         "(needs 'marker' and 'ode_integ_r2_median'; best of seeds is "
                         "taken). Without it the PySR axis is the derivative-fit 'Test "
                         "R2' scraped from the formula dump, which is a different "
                         "quantity from the network's trajectory R² on the x-axis — the "
                         "two correlate only ~0.5. Pass this to compare like with like.")
    args = ap.parse_args()

    nn_dir = Path(args.nn_dir)
    pysr_dir = Path(args.pysr_dir)
    seeds = args.seeds
    threshold = args.r2_threshold

    # Load Neural ODE per-seed OOD test R² and per-seed Jacobians (at winning seed).
    nn_r2: dict[str, dict[int, float]] = {}
    for csv_path in glob.glob(str(nn_dir / "seed_*" / "neural_ode_diffrax_metrics.csv")):
        seed_match = re.search(r"seed_(\d+)", csv_path)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))
        df = pd.read_csv(csv_path)
        for _, row in df[df.split == "test"].iterrows():
            marker = str(row.marker)
            nn_r2.setdefault(marker, {})[seed] = float(
                row.ode_integ_r2_median if args.node_integrated_r2
                and "ode_integ_r2_median" in df.columns else row.trajectory_r2_mean)

    # Validation R2 per marker/seed, for --select-on-val.
    nn_val_r2: dict[str, dict[int, float]] = {}
    for csv_path in glob.glob(str(nn_dir / "seed_*" / "neural_ode_diffrax_metrics.csv")):
        sm = re.search(r"seed_(\d+)", csv_path)
        if not sm:
            continue
        sd = int(sm.group(1))
        try:
            mdf = pd.read_csv(csv_path)
        except Exception:
            continue
        if not {"marker", "split", "ode_integ_r2_median"}.issubset(mdf.columns):
            continue
        vv = mdf[mdf.split == "val"]
        for mk, val in zip(vv.marker, vv.ode_integ_r2_median):
            nn_val_r2.setdefault(str(mk), {})[sd] = float(val)

    # Load PySR formulas + R² per seed.
    pysr_per_seed: dict[str, dict[int, tuple[float, str]]] = {}
    for seed in seeds:
        formula_path = pysr_dir / f"seed_{seed}" / "formulas" / "all_per_minute.txt"
        if not formula_path.exists():
            continue
        for marker, (r2, formula, tr2) in _parse_pysr(formula_path).items():
            pysr_per_seed.setdefault(marker, {})[seed] = (r2, formula, tr2)

    # Sibling module, same directory: importable because a script's own directory heads
    # sys.path. Imported here rather than at module scope so the two-panel path keeps working
    # if that file is ever moved.
    lin_df = lin_ylab = None
    if args.linreg_metrics:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_sr_vs_linreg_ood import (draw_accuracy_panel,       # noqa: E402
                                           load_accuracy_frame)
        print("  panel D (SR vs linear regression):")
        lin_df, lin_ylab = load_accuracy_frame(
            linreg_metrics=args.linreg_metrics, pysr_dir=str(pysr_dir),
            pysr_integ_csv=args.pysr_integ_csv, seeds=seeds,
            linreg_k=args.linreg_k, exclude=args.exclude_marker,
            sr_best_train=args.sr_best_train,
        )

    dataset = pd.read_csv(args.dataset)
    common = sorted(set(nn_r2) & set(pysr_per_seed) - set(args.exclude_marker))

    # Parsimony scorer: the calibrated dependency count, or the participation ratio the
    # earlier drafts used. The count is what the synthetic calibration validates, so it is
    # the measure the figure reports when --dependency-count is given.
    def score_dependencies(v: np.ndarray) -> float:
        a = np.abs(np.asarray(v)).ravel()
        if not args.dependency_count:
            return pr_of(a)
        mx = a.max() if a.size else 0.0
        return float((a > args.count_threshold * mx).sum()) if mx > 0 else 0.0

    inter_by_marker: dict[str, float] = {}
    if args.interaction_csv:
        idf = pd.read_csv(args.interaction_csv)
        icol = "p0.25" if "p0.25" in idf.columns else idf.columns[-1]
        inter_by_marker = idf.groupby("marker")[icol].mean().to_dict()

    rows = []
    for marker in common:
        nn_seeds = nn_r2[marker]
        py_seeds = pysr_per_seed[marker]
        if not nn_seeds or not py_seeds:
            continue
        if args.select_on_val and marker in nn_val_r2:
            # Textbook protocol: choose the seed on validation, report its held-out test
            # score. Selecting on test (the default here) is optimistic and admits
            # contexts that only one seed happens to generalise on.
            cand = {s: v for s, v in nn_val_r2[marker].items() if s in nn_seeds}
            nn_best_seed = max(cand, key=cand.get) if cand else max(nn_seeds, key=nn_seeds.get)
            nn_best_r2 = nn_seeds[nn_best_seed]
        else:
            nn_best_seed, nn_best_r2 = max(nn_seeds.items(), key=lambda kv: kv[1])
        if args.sr_best_train:
            # Training R2 alone, which is the rule Table S8 and the Results text use. Like
            # --sr-parsimony it reads no held-out information, so selection stays clean; it
            # differs in not trading accuracy against expression size.
            def _train(sf) -> float:
                t = sf[2] if len(sf) > 2 else float("nan")
                return t if np.isfinite(t) else -np.inf
            py_best_seed = max(py_seeds, key=lambda s: _train(py_seeds[s]))
            py_best_r2, py_formula = py_seeds[py_best_seed][0], py_seeds[py_best_seed][1]
        elif args.sr_parsimony:
            # PySR's own accuracy-per-complexity score on TRAINING data, so no held-out
            # information enters the choice of which recovered law represents the marker.
            def _score(sf) -> float:
                c = _formula_complexity(sf[1]); t = sf[2] if len(sf) > 2 else float("nan")
                if not np.isfinite(c) or c <= 0 or not np.isfinite(t) or t <= 0:
                    return 0.0
                return -np.log(max(1 - min(t, 0.999999), 1e-12)) / c
            py_best_seed = max(py_seeds, key=lambda s: _score(py_seeds[s]))
            py_best_r2, py_formula = py_seeds[py_best_seed][0], py_seeds[py_best_seed][1]
        else:
            py_best_seed = max(py_seeds, key=lambda s_: py_seeds[s_][0])
            py_best_r2, py_formula = py_seeds[py_best_seed][0], py_seeds[py_best_seed][1]

        # Dependency count is a property estimate, not a competitive score, so it is
        # averaged over every seed rather than read off the one selected for accuracy:
        # a single seed's count is noticeably noisier and biases the contrast toward null.
        per_seed_scores = []
        for sd in seeds:
            ld = _load_marker_checkpoint(nn_dir / f"seed_{sd}" / "models", _safe(marker))
            if ld is None:
                continue
            mt, md, tn = ld
            per_seed_scores.append(score_dependencies(_per_marker_causal(mt, md, tn)["mean_abs_jac"]))
        if not per_seed_scores:
            continue
        nn_pr = float(np.mean(per_seed_scores))

        # The recovered expression is scored the same way as the network: the same
        # dependency measure applied to the mean absolute Jacobian over the marker's
        # training rows, on the same ten inputs. Both boxes are then on one scale.
        py_jac = _pysr_mean_abs_jac(py_formula, marker, dataset)
        py_pr = score_dependencies(py_jac) if py_jac is not None else 0.0

        py_inter = (_pysr_interaction_count(py_formula, marker, dataset, jac=py_jac,
                                            dep_frac=args.count_threshold)
                    if args.interaction_csv else None)
        rows.append(dict(marker=marker, nn_r2=nn_best_r2, py_r2=py_best_r2,
                         nn_pr=nn_pr, py_pr=py_pr,
                         nn_inter=inter_by_marker.get(marker, np.nan),
                         py_inter=np.nan if py_inter is None else py_inter))

    if not rows:
        raise SystemExit("No matched markers found between NN and PySR inputs.")
    df = pd.DataFrame(rows)

    if args.pysr_integ_csv:
        integ = pd.read_csv(args.pysr_integ_csv)
        need = {"marker", "ode_integ_r2_median"}
        if not need.issubset(integ.columns):
            raise SystemExit(f"{args.pysr_integ_csv} needs columns {sorted(need)}")
        pick = None            # the one selected row per marker, when a rule selects one
        if args.sr_best_train and "ode_integ_r2_median_train" in integ.columns:
            # Same rule as the per-seed path above, applied to the integrated table the
            # panel is actually drawn from, so figure and Table S8 partition identically.
            pick = integ.loc[integ.groupby("marker").ode_integ_r2_median_train.idxmax()]
            best = pick.set_index("marker").ode_integ_r2_median
        elif args.sr_parsimony and {"seed", "formula", "ode_integ_r2_median_train"}.issubset(integ.columns):
            # Grouping must use the same seed the formula came from, or the figure
            # classifies markers by a different criterion than the text.
            tmp = integ.dropna(subset=["formula"]).copy()
            tmp["_cx"] = tmp.formula.map(_formula_complexity)
            tr = tmp.ode_integ_r2_median_train.clip(upper=0.999999)
            tmp["_score"] = np.where(
                (tr > 0) & np.isfinite(tmp._cx) & (tmp._cx > 0),
                -np.log(np.maximum(1 - tr, 1e-12)) / tmp._cx.replace(0, np.nan), 0.0)
            pick = tmp.loc[tmp.groupby("marker")._score.idxmax()]
            best = pick.set_index("marker").ode_integ_r2_median
        else:
            best = integ.groupby("marker").ode_integ_r2_median.max()
        matched = df.marker.map(best)
        # Never fill an unmatched marker with 0.0. Doing so previously made the eight
        # control contexts look like uniform PySR failures when in truth they were simply
        # absent from the input table (their real median is 0.85). Drop them instead, so a
        # missing row shows up as a missing point rather than as a fabricated failure.
        missing = df.marker[matched.isna()].tolist()
        if missing:
            print(f"  WARNING: {len(missing)} marker(s) absent from "
                  f"{args.pysr_integ_csv} and therefore DROPPED, not zero-filled: "
                  f"{missing}")
        df = df.assign(py_r2=matched)[matched.notna()].reset_index(drop=True)
        print(f"  PySR axis: ODE-integrated R² from {args.pysr_integ_csv} "
              f"({len(df)} markers)")

        # The SR parsimony and interaction boxes must describe the law that supplied py_r2.
        # They were computed from seeds/formulas/all_per_minute.txt, which selects on a
        # DIFFERENT quantity (derivative-fit R2) than this table (ODE-integrated R2), so the
        # two disagree on the seed for some contexts -- untransfected1 was plotted with seed
        # 42's accuracy against seed 44's six-variable law, giving it a dependency count of
        # 5 on a four-variable expression. Recompute from this table's own formula.
        if pick is not None and "formula" in pick.columns:
            fmap = pick.dropna(subset=["formula"]).set_index("marker").formula.to_dict()
            new_pr, new_inter, moved = [], [], []
            for r in df.itertuples():
                f = fmap.get(r.marker)
                if f is None:
                    new_pr.append(r.py_pr); new_inter.append(r.py_inter); continue
                jac = _pysr_mean_abs_jac(f, r.marker, dataset)
                pr = score_dependencies(jac) if jac is not None else 0.0
                it = (_pysr_interaction_count(f, r.marker, dataset, jac=jac,
                                             dep_frac=args.count_threshold)
                      if args.interaction_csv else r.py_inter)
                if pr != r.py_pr:
                    moved.append(f"{r.marker} {r.py_pr:g}->{pr:g}")
                new_pr.append(pr)
                new_inter.append(it if it is not None else r.py_inter)
            df = df.assign(py_pr=new_pr, py_inter=new_inter)
            if moved:
                print(f"  SR dependency count re-read from the scored law "
                      f"({len(moved)} changed): {', '.join(moved)}")
        if df.empty:
            raise SystemExit(
                "no markers left after matching against --pysr-integ-csv; the marker "
                "names in that CSV probably differ from those in the formula dump."
            )

    df["py_r2_clip"] = df.py_r2.clip(lower=-0.1)

    if args.dump_inputs:
        Path(args.dump_inputs).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.dump_inputs, index=False)
        print(f"  wrote panel inputs -> {args.dump_inputs} ({len(df)} markers)")

    # PR floor is 1.0 by construction, so a 0 is the _pysr_mean_abs_jac failure
    # sentinel leaking in -- drop those markers rather than let them drag the
    # PySR distribution down.
    nn_ok = df.nn_r2 > threshold
    py_ok = df.py_r2 > threshold
    nn_pr_good = df[nn_ok & (df.nn_pr > 0)].nn_pr.values
    # The SR box spans every context SR solves, which is a different set from the two
    # neural-ODE boxes (those are gated on the network generalising). With
    # --sr-box-quadrant it is restricted to the scatter's top-right quadrant instead, so
    # SR and the middle box describe the SAME contexts and the bracket between them
    # compares methods rather than sets.
    py_mask = (py_ok & nn_ok) if args.sr_box_quadrant else py_ok
    py_pr_good = df[py_mask & (df.py_pr > 0)].py_pr.values
    # Split the Neural ODE's working markers by whether PySR also got there.
    nn_pr_py_ok = df[nn_ok & py_ok & (df.nn_pr > 0)].nn_pr.values
    nn_pr_py_bad = df[nn_ok & ~py_ok & (df.nn_pr > 0)].nn_pr.values

    # Sized for a panel occupying 3/5 of a full manuscript figure width
    # (~7.2 in), so 8 pt here is 8 pt on the page -- no downstream rescaling.
    # Bumped 20% over the original 10.0/8.0 for legibility at print size.
    # +15% over the previous 12.0/9.6 on request. The row is drawn 11 in wide and
    # placed at ~7 in, so these land at ~7.9 pt and ~6.3 pt on the page.
    FS = 15.9       # everything
    FS_DOT = 12.7   # per-marker dot labels only
    mpl.rcParams.update({
        # Arial, not Helvetica: macOS ships Helvetica as a .ttc with only the
        # regular face registered, so every bold string silently fell back to
        # DejaVu Sans Bold. Arial has real Bold/Italic files and journals accept
        # it interchangeably with Helvetica. DejaVu trails it purely as a
        # per-glyph fallback -- neither Arial nor Helvetica carries ✓ (U+2713)
        # or ✗ (U+2717), and matplotlib >= 3.6 falls back glyph-by-glyph.
        "font.family": ["Arial", "DejaVu Sans"],
        # Type 42 (TrueType) rather than matplotlib's default Type 3, which
        # many journals reject outright at submission.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333",
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })
    # Same panel height either way, so a panel is the same size on the page whether or not
    # the linreg column is present; only the total width changes.
    # The interaction panel is a second, narrower box column; it is only added when
    # interaction counts were supplied, so the figure keeps its original proportions
    # when they are not.
    want_inter = bool(inter_by_marker) and df.nn_inter.notna().any()
    if lin_df is not None:
        fig = plt.figure(figsize=(11.0 if want_inter else 9.5, 3.8))
        ratios = [3, 3, 1, 1] if want_inter else [3, 3, 1]
        gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.34)
        ax_lin = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])
        ax_box = fig.add_subplot(gs[0, 2])
        ax_int = fig.add_subplot(gs[0, 3]) if want_inter else None
    else:
        ax_lin = None
        fig = plt.figure(figsize=(7.4 if want_inter else 5.9, 3.8))
        ratios = [3, 1, 1] if want_inter else [3, 1]
        gs = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.34)
        ax = fig.add_subplot(gs[0, 0])
        ax_box = fig.add_subplot(gs[0, 1])
        ax_int = fig.add_subplot(gs[0, 2]) if want_inter else None
    # Must precede the manual set_position() below -- subplots_adjust re-applies
    # the gridspec layout to every subplot axes and would wipe it. tight_layout
    # can't reconcile the equal-aspect scatter with the gridspec either, so
    # bbox_inches="tight" at save time does the trimming.
    fig.subplots_adjust(left=0.125 if lin_df is None else 0.078,
                        right=0.985, top=0.945, bottom=0.200)
    # Flat fills only -- no dashes, no hatching. The scatter is two tones of one
    # blue, marking the Neural ODE's two subsets, and each tone is reused as the
    # face of that subset's box. Orange belongs to PySR and appears only in the
    # box panel; on the scatter PySR is read off the y axis and its threshold.
    INK = "#111111"
    NN_COLOR = "#2a78d6"     # single blue hue, used at two alphas
    # Alphas chosen so the two tones clear CIEDE2000 = 10 both in normal vision
    # and under full dichromacy, and stay ~23% apart in greyscale luminance for
    # B&W print (0.13/0.30 measured only dE 5.8 and 14%).
    NN_A_SOLO = 0.20         # light tone: Neural ODE works, PySR does not
    NN_A_BOTH = 0.48         # dark tone: both work
    NN_DARK = "#1a5490"      # box median, whiskers, dots (7.7:1 on white)
    PY_BOX = "#f0902a"       # PySR box face
    PY_DARK = "#96520a"      # PySR median, whiskers, dots (6.0:1)
    PY_A = 0.42              # PySR box-face alpha

    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.set_axisbelow(True)

    # ---- Neural ODE success bands, two tones ----------------------------
    # The Neural ODE works to the right of the vertical threshold. That half
    # splits at the horizontal threshold into the two subsets the box panel
    # compares: darker where PySR also works, lighter where it does not. So each
    # tone maps 1:1 onto one blue box. The left half carries no fill -- PySR's
    # own set is read off the y axis, and its colour lives in the box panel.
    t = threshold
    ax.add_patch(Rectangle((t, t), 1 - t, 1 - t, facecolor=NN_COLOR,
                           edgecolor="none", alpha=NN_A_BOTH,
                           zorder=0))                  # NN works, PySR too
    ax.add_patch(Rectangle((t, 0), 1 - t, t, facecolor=NN_COLOR,
                           edgecolor="none", alpha=NN_A_SOLO,
                           zorder=0))                  # NN works, PySR does not

    ax.grid(True, which="major", color="#e8e7e3", lw=0.6, zorder=1)
    ax.plot([0, 1.0], [0, 1.0], color="#888", lw=1.0, ls="--", zorder=2)
    ax.axvline(t, color="#333", lw=1.0, ls=(0, (5, 3)), alpha=0.75, zorder=2)
    ax.axhline(t, color="#333", lw=1.0, ls=(0, (5, 3)), alpha=0.75, zorder=2)
    # Two short titles, one over each half. They name what each half MEASURES,
    # which is what stops the box panel being read as a per-quadrant marker
    # count: "parsimony" makes the vertical axis unmistakably driver count, and
    # the n's under the boxes read as group sizes rather than the plotted value.
    # Matched style and baseline, so they read as two labelled halves of one
    # panel rather than two panels.
    # "Accuracy out of distribution" is ambiguous once a second accuracy panel shares the
    # figure -- each title now names the baseline it is against.
    if not args.no_panel_titles:
        ax.set_title("Accuracy vs neural ODE" if lin_df is not None
                     else "Accuracy out of distribution", loc="left",
                     fontsize=FS, fontweight="bold", color=INK, pad=5)

    df["py_r2_plot"] = df.py_r2.clip(lower=0)
    # Region counts go to stdout for the caption rather than onto the plot --
    # the bands carry the partition and the box panel carries the n's.
    print(f"  regions: PySR works n={int(py_ok.sum())}  "
          f"Neural ODE works n={int(nn_ok.sum())}  "
          f"both n={int((nn_ok & py_ok).sum())}  "
          f"neither n={int((~nn_ok & ~py_ok).sum())}")

    # The POOLED dependency-count claim, which is the one quoted in the Results:
    # wherever PySR recovers a closed form the network needs few effective drivers,
    # wherever PySR fails it needs more -- *whether or not the network itself
    # generalises*. This pools both failure branches, which the box panel below does
    # not: the panel's narrower Neural-ODE-successes-only contrast is the weaker test
    # (the two failure branches are statistically indistinguishable from each other, so
    # splitting them only costs power). Printed here because it is the reported number.
    pooled_solved = df[py_ok & (df.nn_pr > 0)].nn_pr.values
    pooled_failed = df[~py_ok & (df.nn_pr > 0)].nn_pr.values
    if len(pooled_solved) and len(pooled_failed):
        pooled_p = stats.mannwhitneyu(pooled_solved, pooled_failed).pvalue
        # Median is the reported statistic -- Mann-Whitney is a rank test, so quoting a
        # mean beside it invites the reader to check the wrong number. Both are printed
        # and both are labelled, because they differ enough to matter here (median 3.34
        # against mean 3.58 for the solved group).
        measure = ("dependency count" if args.dependency_count
                   else "participation ratio")
        print(f"  pooled NN {measure} over the ten inputs:")
        for label, values in (("PySR solved", pooled_solved), ("PySR failed", pooled_failed)):
            print(f"    {label}   n={len(values):2d}  "
                  f"median PR = {np.median(values):.2f}  (mean {values.mean():.2f})")
        print(f"    Mann-Whitney U p = {pooled_p:.4g}")

    ax.scatter(df.nn_r2, df.py_r2_plot, s=26, c="#111", alpha=0.92,
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    # Same rule as the y axis below: no seed convention in the axis label. "best of 3 seeds"
    # was wrong under --select-on-val, where the seed is chosen on validation and its test
    # score reported; the convention belongs in the caption, which states it for both axes.
    ax.set_xlabel("Integrated neural ODE test R²", fontsize=FS)
    # The y axis is ODE-integrated R² whenever --pysr-integ-csv is passed, which is how every
    # figure in the paper is built; calling it "test R²" named the derivative-fit quantity
    # instead, a different number (the two correlate ~0.5).
    # The seed convention depends on how the PySR axis was selected, so the label must not
    # hard-code "best of 3 seeds": with --sr-parsimony the seed comes from the training
    # parsimony score, and when a pre-selected one-row-per-marker table is supplied the
    # selection happened upstream (production runs select on training R2).
    # No seed convention in the axis label: it belongs in the caption, and the longer
    # strings overrun the axis height and get clipped at save time.
    y_quantity = ("Integrated SR test R²" if args.pysr_integ_csv
                  else "SR OOD test R²")
    # Both scatters carry the same y quantity on the same 0-1 scale and sit side by side, so
    # the label goes on the leftmost panel only; tick numerals stay on both.
    ax.set_ylabel(y_quantity, fontsize=FS)

    # The equal-aspect scatter does not fill its gridspec cell, so width_ratios
    # alone would leave the box panel too wide. Anchor it to the scatter's
    # resolved position instead, at --box-width-frac of the scatter's drawn
    # width.
    if lin_df is not None:
        # Same problem as the box panel below, one column earlier: an equal-aspect axes shrinks
        # inside its gridspec cell, so width_ratios leave ~0.6 in of slack per scatter and the
        # two end up far apart. Anchor the neural-ODE panel to the linreg panel's drawn box.
        # The limits/aspect have to be set here rather than left to draw_accuracy_panel, since
        # they are what fixes that box and it is measured now.
        ax_lin.set_aspect("equal", adjustable="box")
        ax_lin.set_xlim(0, 1)
        ax_lin.set_ylim(0, 1)
        fig.canvas.draw()
        q = ax_lin.get_window_extent().transformed(fig.transFigure.inverted())
        # 0.05 leaves room for this panel's own y tick numerals, which sit outside its box.
        ax.set_position([q.x1 + 0.05, q.y0, q.width, q.height])

    fig.canvas.draw()
    # get_position() returns the gridspec CELL, not the box the equal-aspect
    # scatter actually shrank to -- using it leaves a gap and oversizes the
    # marginal. The drawn extent is what we need.
    p = ax.get_window_extent().transformed(fig.transFigure.inverted())
    ax_box.set_position([p.x1 + args.box_gap * p.width, p.y0,
                         p.width * args.box_width_frac, p.height])
    # Its own left-hand y axis, and a gap wide enough to clear the label: the
    # box is a separate panel about a different quantity, not a marginal
    # distribution of the scatter it sits beside.
    ax_box.spines["left"].set_visible(True)
    ax_box.spines["right"].set_visible(False)
    ax_box.yaxis.set_label_position("left")
    ax_box.yaxis.tick_left()

    # Last, because draw_accuracy_panel places its dot labels in pixel coordinates against
    # the resolved axes box -- anything that moves an axes afterwards invalidates them.
    if lin_df is not None:
        draw_accuracy_panel(fig, ax_lin, lin_df, linreg_k=args.linreg_k,
                            ylab=y_quantity, label_threshold=threshold,
                            fs_axis=FS, fs_dot=FS_DOT,
                            title=("" if args.no_panel_titles
                                   else "Accuracy vs linear regression"),
                            label_skip=args.label_skip)

    ax_box.grid(True, which="major", axis="y", color="#e5e5e5", lw=0.7, zorder=0)
    ax_box.set_axisbelow(True)

    def _modern_box(x_pos, values, color, ax, face, alpha):
        """Narrow-panel box. `fills` is a list of (facecolor, hatch, alpha)
        layers, so a box can reproduce the exact treatment of the scatter
        region its markers were drawn from."""
        values = np.asarray(values)
        rng = np.random.default_rng(int(x_pos * 100) + 7)
        jitter = x_pos + rng.uniform(-0.11, 0.11, len(values))
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        w = 0.52
        ax.add_patch(FancyBboxPatch(
            (x_pos - w / 2, q1), w, q3 - q1,
            boxstyle="round,pad=0.004,rounding_size=0.05",
            linewidth=0, facecolor=face, alpha=alpha, zorder=2,
        ))
        ax.scatter(jitter, values, s=12, c=color, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(med, x_pos - w / 2, x_pos + w / 2,
                  colors=color, linewidth=2.4, zorder=4)
        ax.plot([x_pos, x_pos], [values.min(), q1], color=color, lw=0.9, alpha=0.7, zorder=2)
        ax.plot([x_pos, x_pos], [q3, values.max()], color=color, lw=0.9, alpha=0.7, zorder=2)

    # Only the FIRST box is PySR. The other two are both the Neural ODE's
    # driver count, split by whether PySR also solved that marker -- so they
    # share the Neural ODE's black hatch, and the middle one additionally
    # carries the PySR wash because its markers sit in the scatter's overlap.
    groups = [
        (0, py_pr_good, PY_DARK, PY_BOX, PY_A),
        (1, nn_pr_py_ok, NN_DARK, NN_COLOR, NN_A_BOTH),
        (2, nn_pr_py_bad, NN_DARK, NN_COLOR, NN_A_SOLO),
    ]
    for x_pos, vals, col, face, alpha in groups:
        if len(vals):
            _modern_box(x_pos, vals, col, ax_box, face, alpha)

    # Stats: paired where the markers coincide, independent otherwise. Every
    # Every p-value goes to stdout for the caption. The SR-vs-network bracket is always
    # drawn -- it is the comparison the panel exists to make -- and the network-internal
    # one only when significant, so the narrow panel stays readable.
    # Test the boxes that are actually drawn. The paired Wilcoxon used here previously
    # ran on the overlap only, so the bracket spanned two boxes whose comparison was never
    # performed -- and at n<10 with ties its normal approximation is unreliable besides.
    paired = df[nn_ok & py_ok & (df.nn_pr > 0)]
    # Both boxes are participation ratios on the same ten inputs, so SR-vs-network is a
    # like-for-like test and is drawn on the same footing as the network-internal one.
    tests = []
    if len(py_pr_good) >= 3 and len(nn_pr_py_ok) >= 3:
        p_sr_nn = stats.mannwhitneyu(py_pr_good, nn_pr_py_ok).pvalue
        tests.append((0, 1, "Mann-Whitney U (boxes as drawn)", p_sr_nn, True))
    if len(nn_pr_py_ok) and len(nn_pr_py_bad):
        # Same estimator as the Results text: exact permutation of the failure label on
        # the accuracy-residualised count, so panel and prose report one number.
        _g = df[nn_ok & (df.nn_pr > 0)]
        tests.append((1, 2, "permutation (accuracy-residualised)",
                      _perm_p(_g.nn_pr.values,
                              (~_g.marker.isin(df[py_ok].marker)).values,
                              _g.nn_r2.values), True))

    # Each bracket sits just above the boxes it actually spans, not on a shared tier: a
    # bracket floated over unrelated taller boxes reads as covering them too. Overlapping
    # spans then stack.
    gmax = {x: float(v.max()) for x, v, _, _, _ in groups if len(v)}
    top = max(gmax.values())
    placed: list[tuple[int, int, float]] = []
    for x0, x1, name, p, always in tests:
        print(f"  {name}: group {x0} vs {x1}  p = {p:.4g}")
        if p >= 0.05 and not always:
            continue
        # _stars, not a local copy: the interaction panel uses it, and the two spellings
        # ("ns" here against its "n.s.") appeared side by side in one figure row.
        label = _stars(p)
        y = max(gmax.get(x, 1.0) for x in range(x0, x1 + 1)) + 0.35
        for px0, px1, py in placed:
            if not (x1 < px0 or x0 > px1):
                y = max(y, py + 0.60)
        ax_box.plot([x0, x0, x1, x1], [y, y + 0.12, y + 0.12, y],
                    color="#444", lw=0.9, clip_on=False, zorder=6)
        ax_box.text((x0 + x1) / 2, y + 0.16, label, ha="center", va="bottom",
                    fontsize=FS if p < 0.05 else FS - 2.0,
                    fontweight="bold" if p < 0.05 else "normal",
                    color="#333", zorder=6)
        placed.append((x0, x1, y))

    ax_box.set_ylim(1.0, max([top] + [y for *_, y in placed]) + 0.8)

    # Two-tier axis. The tick labels carry the CONDITION and the brackets below
    # carry the METHOD, because two of the three boxes are the same method: box
    # 1 and box 2 are both the Neural ODE's driver count, split by whether PySR
    # also solved that marker. Labelling the middle box "both" would wrongly
    # imply it is a joint quantity.
    ax_box.set_xticks([0, 1, 2])
    # Three "n=NN" labels side by side need roughly half the scatter's width at
    # the full size; below that they collide, so step down rather than let them
    # overlap. Tied to the width knob so changing one cannot silently break the
    # other.
    FS_BOX_TICK = FS if args.box_width_frac >= 0.5 else FS - 2.0
    ax_box.set_xticklabels([f"SR\nn={len(py_pr_good)}",
                            f"=SR\nn={len(nn_pr_py_ok)}",
                            f">SR\nn={len(nn_pr_py_bad)}"], fontsize=FS_BOX_TICK)
    for label, col in zip(ax_box.get_xticklabels(), [PY_DARK, NN_DARK, NN_DARK]):
        label.set_color(col)
        label.set_fontweight("bold")
    if not args.no_tick_legend:
        ax_box.text(1.09, -0.096, f"=SR  SR R²>{threshold}\n>SR  SR R²≤{threshold}",
                    transform=ax_box.transAxes, color="#555555", fontsize=FS_BOX_TICK,
                    ha="left", va="top", linespacing=1.5, clip_on=False, zorder=6)
    # A single bracket over the two Neural ODE boxes; the tick marks under them
    # are the condition. Short glyphs because a quarter-width panel has no room
    # for "PySR works" / "PySR fails" side by side.
    #
    # One implementation for both box panels. This used to be an inline copy that drew
    # the rule in data coordinates and only then called set_xlim, so the pixel margin was
    # converted against the pre-limit axes: it came out 13 px against the 27.5 px the
    # same formula gives in the interactions panel, and the two rules did not match.
    # Set the limits first, then measure.
    ax_box.set_xlim(-0.68, 2.68)
    _rule_y = _group_bracket(ax_box, ax_box.get_xticklabels()[1:], "Neural ODE",
                             NN_DARK, FS, SAVE_DPI)
    ax_box.tick_params(axis="y", labelsize=FS)
    if not args.no_panel_titles:
        ax_box.set_title("Parsimony", loc="left", fontsize=FS,
                         fontweight="bold", color=INK, pad=5)
    ax_box.set_ylabel(
        ("Effective dependencies count"
         if args.dependency_count else
         "Effective dependencies PR"),
        fontsize=FS, linespacing=1.35)

    # Interaction panel: the same two network groups as the parsimony panel, scored by
    # the number of interacting input pairs in the learned law. Reported as an effect
    # size with its exact-permutation p-value -- at this sample it does not reach
    # significance, and the panel says so rather than leaving the reader to assume.
    if ax_int is not None:
        # Same three groups as the parsimony panel, so the two read in parallel: the
        # recovered symbolic law, the network where SR also succeeds, the network where
        # SR fails. The measure is identical for all three -- mean |d2f/dxi dxj| over the
        # marker's rows -- so symbolic and network laws are strictly comparable.
        py_int_good = df.loc[py_mask, "py_inter"].dropna().values
        int_ok = df.loc[nn_ok & py_ok, "nn_inter"].dropna().values
        int_bad = df.loc[nn_ok & ~py_ok, "nn_inter"].dropna().values
        ax_int.grid(True, which="major", axis="y", color="#e5e5e5", lw=0.7, zorder=0)
        ax_int.set_axisbelow(True)
        groups_int = [(1, int_ok, NN_DARK, NN_COLOR, NN_A_BOTH),
                      (2, int_bad, NN_DARK, NN_COLOR, NN_A_SOLO)]
        if args.symbolic_interaction_box:
            groups_int.insert(0, (0, py_int_good, PY_DARK, PY_BOX, PY_A))
        for x_pos, vals, col, face, alpha in groups_int:
            if len(vals):
                _modern_box(x_pos, vals, col, ax_int, face, alpha)
        if len(int_ok) and len(int_bad):
            _sub = df[nn_ok & df.nn_inter.notna()]
            p_int = _perm_p(_sub.nn_inter.values, (~_sub.marker.isin(
                df[py_ok].marker)).values, _sub.nn_r2.values)
            ratio = float(np.mean(int_bad) / np.mean(int_ok)) if np.mean(int_ok) else float("nan")
            # p_int is the accuracy-residualised permutation test (same as the parsimony
            # panel's bracket), not a Mann-Whitney; the old label said "MWU" and misled.
            print(f"  interaction terms  network: SR-fails {np.mean(int_bad):.2f} vs "
                  f"SR-solves {np.mean(int_ok):.2f}  ratio {ratio:.2f}x  "
                  f"accuracy-residualised permutation p = {p_int:.4g}")
        if len(py_int_good):
            print(f"  interaction terms  recovered symbolic law: {np.mean(py_int_good):.2f}")
        if len(int_ok) and len(int_bad):
            top = max(float(np.max(int_bad)), float(np.max(int_ok)),
                      float(np.max(py_int_good)) if len(py_int_good) else 0.0)
            # Headroom for two stacked brackets, then place them at fixed axes
            # fractions so their spacing does not track the data range.
            ax_int.set_ylim(bottom=min(0.0, ax_int.get_ylim()[0]), top=top * 1.34)
            _trb = mtransforms.blended_transform_factory(ax_int.transData, ax_int.transAxes)
            def _bracket(x0, x1, yfrac, text):
                pad = 0.09
                a, b = x0 + pad, x1 - pad
                ax_int.plot([a, a, b, b],
                            [yfrac, yfrac + 0.020, yfrac + 0.020, yfrac],
                            transform=_trb, lw=0.9, color="#4a4a4a", clip_on=False)
                ax_int.text((a + b) / 2, yfrac + 0.032, text, transform=_trb,
                            ha="center", va="bottom", fontsize=FS - 1, color=INK)
            # Significance only. The fold-change was printed beside the star, which put a
            # second number on a bracket whose job is the test; it is still reported to
            # stdout and in the text.
            _bracket(1, 2, 0.86, _stars(p_int))
            # SR vs network-where-SR-succeeds: paired across the markers both solve, the
            # same footing the parsimony panel uses for its symbolic-vs-network bracket.
            if args.symbolic_interaction_box and len(py_int_good) >= 3:
                pair = df[py_ok & nn_ok & df.nn_inter.notna() & df.py_inter.notna()]
                if len(pair) >= 3:
                    p_sr = stats.wilcoxon(pair.py_inter.values, pair.nn_inter.values).pvalue
                    print(f"  interaction terms  symbolic vs network (paired, n={len(pair)}):"
                          f" {pair.py_inter.mean():.2f} vs {pair.nn_inter.mean():.2f}"
                          f"  Wilcoxon p = {p_sr:.4g}")
                    _bracket(0, 1, 0.86, _stars(p_sr))
        ax_int.set_xlim(-0.68, 2.68)
        _ticks = [0, 1, 2] if args.symbolic_interaction_box else [1, 2]
        _tlabs = ([f"SR\nn={len(py_int_good)}", f"=SR\nn={len(int_ok)}", f">SR\nn={len(int_bad)}"]
                  if args.symbolic_interaction_box
                  else [f"=SR\nn={len(int_ok)}", f">SR\nn={len(int_bad)}"])
        _tcols = (PY_DARK, NN_DARK, NN_DARK) if args.symbolic_interaction_box else (NN_DARK, NN_DARK)
        ax_int.set_xticks(_ticks)
        ax_int.set_xticklabels(_tlabs, fontsize=FS)
        for lab, c in zip(ax_int.get_xticklabels(), _tcols):
            lab.set_color(c)
            lab.set_fontweight("bold")
        # Tick marks on, matching the parsimony panel: whatever the two do here they
        # must do together, or their label blocks -- and the rules under them -- sit at
        # different heights. The shared rule_y keeps the rules level regardless.
        ax_int.tick_params(axis="x", length=3.5)
        ax_int.tick_params(axis="y", labelsize=FS)
        ax_int.spines["top"].set_visible(False)
        ax_int.spines["right"].set_visible(False)
        ax_int.spines["left"].set_visible(True)
        ax_int.yaxis.set_label_position("left")
        ax_int.yaxis.tick_left()
        if not args.no_panel_titles:
            ax_int.set_title("Interactions", loc="left", fontsize=FS,
                             fontweight="bold", color=INK, pad=5)
        ax_int.set_ylabel("Interacting input pairs count", fontsize=FS, linespacing=1.35)
        # Mirror the parsimony panel's manual placement: the gridspec cell is not where
        # that panel ends up, so an un-nudged neighbour lands on top of it.
        pb = ax_box.get_position()
        ax_int.set_position([pb.x1 + args.box_gap * pb.width * 1.9, pb.y0,
                             pb.width, pb.height])
        # Same ink-measured placement as the parsimony panel: fixed coordinates put the
        # rule left of the labels' visual centre, because "n=12" is wider than "n=9".
        # AFTER set_position, not before: _group_bracket converts a pixel margin into
        # data units and a pixel row into an axes fraction, so moving *and resizing* the
        # panel afterwards leaves both stale -- the rule overhung to the right (narrower
        # panel, same data span) and sat low (shorter panel, same axes fraction).
        _int_ticks = ax_int.get_xticklabels()[-2:]     # the two Neural ODE groups
        _group_bracket(ax_int, _int_ticks, "Neural ODE", NN_DARK, FS, SAVE_DPI,
                       rule_y=_rule_y)

    # Equal gutters. Each panel is placed relative to its left neighbour by a different
    # rule -- gridspec wspace between the scatters, box_gap*scatter_width for the box,
    # box_gap*box_width*1.9 for the interactions panel -- so the four sat at three
    # different spacings. Equalise on the TIGHT bboxes, not the axes boxes: the box
    # panels carry multi-line y-labels outside their axes, so equal axes-box gaps still
    # read as unequal. Translate only, never resize, so every ink-measured placement
    # above (which is in axes fractions or data units) stays valid.
    _panels = [a for a in (ax_lin, ax, ax_box, ax_int) if a is not None]
    if len(_panels) > 2:
        fig.canvas.draw()
        _r = fig.canvas.get_renderer()
        _inv = fig.transFigure.inverted()
        _tb = [a.get_tightbbox(_r).transformed(_inv) for a in _panels]
        _gaps = [_tb[i + 1].x0 - _tb[i].x1 for i in range(len(_tb) - 1)]
        _target = max(min(_gaps), args.panel_gap)
        _shift = 0.0
        for i in range(1, len(_panels)):
            _shift += _target - _gaps[i - 1]
            _p = _panels[i].get_position()
            _panels[i].set_position([_p.x0 + _shift, _p.y0, _p.width, _p.height])
        print(f"  panel gutters {[round(g, 4) for g in _gaps]} -> equalised at {_target:.4f}")

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_png} and {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
