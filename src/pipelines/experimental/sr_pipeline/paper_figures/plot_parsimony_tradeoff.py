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
NON_GFP = [f for f in FEATURES if f != "GFP"]


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
    radii = [8, 12, 16, 22, 30, 40, 55, 75]

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
        label = str(r[label_col])

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


def pr_of(v: np.ndarray) -> float:
    v = np.abs(np.asarray(v)).ravel()
    s, sq = v.sum(), (v ** 2).sum()
    return float(s * s / sq) if sq > 0 else 0.0


def _parse_pysr(path: Path) -> dict[str, tuple[float, str]]:
    text = path.read_text()
    out: dict[str, tuple[float, str]] = {}
    for block in text.split("\nGroup: "):
        if not block.strip():
            continue
        if not block.startswith("Group:"):
            block = "Group: " + block
        m = re.search(r"Group:\s*(\S+)", block)
        r2 = re.search(r"Test R2:\s*([-+0-9.eE]+)", block)
        f = re.search(r"Formula:\s*(.+)", block)
        if m and r2 and f:
            out[m.group(1).strip()] = (float(r2.group(1)), f.group(1).strip())
    return out


def _pysr_jac_at_mean(formula: str, marker: str, data: pd.DataFrame) -> np.ndarray | None:
    syms = {f: sp.Symbol(f) for f in FEATURES}
    try:
        expr = sp.sympify(formula, locals=syms)
    except Exception:
        return None
    sub = data[data.get("marker") == marker] if "marker" in data.columns else data
    if len(sub) == 0:
        sub = data
    subs = {syms[f]: (float(sub[f].mean()) if f in sub.columns else 1.0) for f in FEATURES}
    j = np.zeros(len(FEATURES))
    for i, fi in enumerate(FEATURES):
        try:
            j[i] = float(sp.diff(expr, syms[fi]).subs(subs))
        except Exception:
            pass
    return j


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
    ap.add_argument("--exclude-marker", action="append", default=["untransfected1"])
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
            nn_r2.setdefault(marker, {})[seed] = float(row.trajectory_r2_mean)

    # Load PySR formulas + R² per seed.
    pysr_per_seed: dict[str, dict[int, tuple[float, str]]] = {}
    for seed in seeds:
        formula_path = pysr_dir / f"seed_{seed}" / "formulas" / "all_per_minute.txt"
        if not formula_path.exists():
            continue
        for marker, (r2, formula) in _parse_pysr(formula_path).items():
            pysr_per_seed.setdefault(marker, {})[seed] = (r2, formula)

    dataset = pd.read_csv(args.dataset)
    common = sorted(set(nn_r2) & set(pysr_per_seed) - set(args.exclude_marker))

    rows = []
    for marker in common:
        nn_seeds = nn_r2[marker]
        py_seeds = pysr_per_seed[marker]
        if not nn_seeds or not py_seeds:
            continue
        nn_best_seed, nn_best_r2 = max(nn_seeds.items(), key=lambda kv: kv[1])
        py_best_seed, (py_best_r2, py_formula) = max(py_seeds.items(), key=lambda kv: kv[1][0])

        loaded = _load_marker_checkpoint(nn_dir / f"seed_{nn_best_seed}" / "models", _safe(marker))
        if loaded is None:
            continue
        meta, model, train = loaded
        causal = _per_marker_causal(meta, model, train)
        non_gfp_idx = [i for i, f in enumerate(meta["feature_cols"]) if f != "GFP"]
        nn_pr = pr_of(causal["mean_signed_jac"][non_gfp_idx])

        py_jac = _pysr_jac_at_mean(py_formula, marker, dataset)
        py_pr = pr_of(py_jac[[FEATURES.index(f) for f in NON_GFP]]) if py_jac is not None else 0.0

        rows.append(dict(marker=marker, nn_r2=nn_best_r2, py_r2=py_best_r2,
                         nn_pr=nn_pr, py_pr=py_pr))

    if not rows:
        raise SystemExit("No matched markers found between NN and PySR inputs.")
    df = pd.DataFrame(rows)

    if args.pysr_integ_csv:
        integ = pd.read_csv(args.pysr_integ_csv)
        need = {"marker", "ode_integ_r2_median"}
        if not need.issubset(integ.columns):
            raise SystemExit(f"{args.pysr_integ_csv} needs columns {sorted(need)}")
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
        if df.empty:
            raise SystemExit(
                "no markers left after matching against --pysr-integ-csv; the marker "
                "names in that CSV probably differ from those in the formula dump."
            )

    df["py_r2_clip"] = df.py_r2.clip(lower=-0.1)

    # PR floor is 1.0 by construction, so a 0 is the _pysr_jac_at_mean failure
    # sentinel leaking in -- drop those markers rather than let them drag the
    # PySR distribution down.
    nn_ok = df.nn_r2 > threshold
    py_ok = df.py_r2 > threshold
    nn_pr_good = df[nn_ok & (df.nn_pr > 0)].nn_pr.values
    py_pr_good = df[py_ok & (df.py_pr > 0)].py_pr.values
    # Split the Neural ODE's working markers by whether PySR also got there.
    nn_pr_py_ok = df[nn_ok & py_ok & (df.nn_pr > 0)].nn_pr.values
    nn_pr_py_bad = df[nn_ok & ~py_ok & (df.nn_pr > 0)].nn_pr.values

    # Sized for a panel occupying 3/5 of a full manuscript figure width
    # (~7.2 in), so 8 pt here is 8 pt on the page -- no downstream rescaling.
    FS = 8.0        # everything
    FS_DOT = 6.0    # per-marker dot labels only
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
    fig = plt.figure(figsize=(5.9, 3.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.20)
    ax = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    # Must precede the manual set_position() below -- subplots_adjust re-applies
    # the gridspec layout to every subplot axes and would wipe it. tight_layout
    # can't reconcile the equal-aspect scatter with the gridspec either, so
    # bbox_inches="tight" at save time does the trimming.
    fig.subplots_adjust(left=0.125, right=0.985, top=0.945, bottom=0.200)
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
    ax.set_title("Accuracy out of distribution", loc="left",
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
        print("  pooled NN dependency count (participation ratio, GFP excluded):")
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
    ax.set_xlabel("Neural ODE OOD test R² (L21, best of 3 seeds)", fontsize=FS)
    ax.set_ylabel("SR OOD test R² (best of 3 seeds)", fontsize=FS)

    # The equal-aspect scatter does not fill its gridspec cell, so width_ratios
    # alone would leave the box panel too wide. Anchor it to the scatter's
    # resolved position instead: exactly one third of the scatter's width, i.e.
    # one quarter of the two panels combined.
    fig.canvas.draw()
    # get_position() returns the gridspec CELL, not the box the equal-aspect
    # scatter actually shrank to -- using it leaves a gap and oversizes the
    # marginal. The drawn extent is what we need.
    p = ax.get_window_extent().transformed(fig.transFigure.inverted())
    ax_box.set_position([p.x1 + 0.022, p.y0, p.width / 1.8, p.height])
    ax_box.spines["left"].set_visible(False)
    ax_box.spines["right"].set_visible(True)
    ax_box.yaxis.set_label_position("right")
    ax_box.yaxis.tick_right()

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
    # p-value goes to stdout for the caption; only significant brackets are
    # drawn, so the narrow panel stays readable.
    paired = df[nn_ok & py_ok & (df.nn_pr > 0) & (df.py_pr > 0)]
    tests = []
    if len(paired) >= 3:
        tests.append((0, 1, "paired Wilcoxon",
                      stats.wilcoxon(paired.py_pr.values, paired.nn_pr.values).pvalue))
    if len(nn_pr_py_ok) and len(nn_pr_py_bad):
        tests.append((1, 2, "Mann-Whitney U",
                      stats.mannwhitneyu(nn_pr_py_ok, nn_pr_py_bad).pvalue))
    if len(py_pr_good) and len(nn_pr_py_bad):
        tests.append((0, 2, "Mann-Whitney U",
                      stats.mannwhitneyu(py_pr_good, nn_pr_py_bad).pvalue))

    all_vals = np.concatenate([v for _, v, _, _, _ in groups if len(v)])
    top = float(all_vals.max())
    ax_box.set_ylim(1.0, top + 1.5)
    drawn = 0
    for x0, x1, name, p in tests:
        print(f"  {name}: group {x0} vs {x1}  p = {p:.4g}")
        if p >= 0.05:
            continue
        stars = "***" if p < 1e-3 else ("**" if p < 1e-2 else "*")
        y = top + 0.35 + 0.55 * drawn
        ax_box.plot([x0, x0, x1, x1], [y, y + 0.12, y + 0.12, y],
                    color="#444", lw=0.9, clip_on=False, zorder=6)
        ax_box.text((x0 + x1) / 2, y + 0.16, stars, ha="center", va="bottom",
                    fontsize=FS, fontweight="bold", color="#333", zorder=6)
        drawn += 1

    # Two-tier axis. The tick labels carry the CONDITION and the brackets below
    # carry the METHOD, because two of the three boxes are the same method: box
    # 1 and box 2 are both the Neural ODE's driver count, split by whether PySR
    # also solved that marker. Labelling the middle box "both" would wrongly
    # imply it is a joint quantity.
    ax_box.set_xticks([0, 1, 2])
    ax_box.set_xticklabels([f"SR\nn={len(py_pr_good)}",
                            f"✓\nn={len(nn_pr_py_ok)}",
                            f"✗\nn={len(nn_pr_py_bad)}"], fontsize=FS)
    for label, col in zip(ax_box.get_xticklabels(), [PY_DARK, NN_DARK, NN_DARK]):
        label.set_color(col)
        label.set_fontweight("bold")
    ax_box.text(1.05, -0.096, f"✓ SR R²>{threshold}\n✗ SR R²≤{threshold}",
                transform=ax_box.transAxes, color="#555555", fontsize=FS,
                ha="left", va="top", linespacing=1.5, clip_on=False, zorder=6)
    # A single bracket over the two Neural ODE boxes; the tick marks under them
    # are the condition. Short glyphs because a quarter-width panel has no room
    # for "PySR works" / "PySR fails" side by side.
    tr = mtransforms.blended_transform_factory(ax_box.transData, ax_box.transAxes)
    ax_box.plot([0.70, 2.30], [-0.140, -0.140], transform=tr,
                color=NN_DARK, lw=0.8, clip_on=False, zorder=6)
    ax_box.text(1.5, -0.175, "Neural ODE", transform=tr, color=NN_DARK,
                fontsize=FS, fontweight="bold", ha="center", va="top",
                clip_on=False, zorder=6)
    ax_box.set_xlim(-0.68, 2.68)
    ax_box.tick_params(axis="y", labelsize=FS)
    ax_box.set_title("Parsimony", loc="left", fontsize=FS,
                     fontweight="bold", color=INK, pad=5)
    ax_box.set_ylabel("Effective number of drivers (PR)\non markers that method solves",
                      fontsize=FS, linespacing=1.35)

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_png} and {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
