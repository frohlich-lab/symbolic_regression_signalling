"""Main-text figure: SR vs linear-regression OOD accuracy, per marker.

Parity scatter of best-of-3-seeds OOD test R² (ODE-integrated) for symbolic
regression against the SelectKBest linear-regression baseline on the same
top-GFP-bin extrapolation split. The half-plane where SR wins is filled, so
"SR above the line" is readable without comparing to the diagonal by eye.

The linear-regression side comes from select_k_linreg_per_minute.py run with
--test-split-policy top_gfp_bins; the stored runs/select_k results are
random_bins (in-distribution) and are NOT interchangeable.

Saves to --output (png) and the corresponding .pdf next to it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_names import gene_labels  # noqa: E402


def _parse_pysr(path: Path) -> dict[str, float]:
    """Best test R² per marker from a PySR formulas dump."""
    out: dict[str, float] = {}
    for block in path.read_text().split("\nGroup: "):
        if not block.strip():
            continue
        # The split consumes the "Group: " prefix on every block after the
        # first, so put it back before matching.
        if not block.startswith("Group:"):
            block = "Group: " + block
        # Marker names contain spaces ("DUSP10 (P2)"), so take the rest of the
        # line rather than one non-space token.
        m = re.search(r"Group:\s*(.+)", block)
        r2 = re.search(r"Test R2:\s*([-+0-9.eE]+)", block)
        if m and r2:
            out[m.group(1).strip()] = float(r2.group(1))
    return out


# Callout-stack geometry. COL_X_TOL_PX is how far apart two dots may sit horizontally and
# still count as the same column; the left spine's dots share x exactly, so it only has to
# absorb rounding. The other two are multiples of the text's line height rather than fixed
# pixel counts, so the stack keeps its proportions when the font size changes -- at 9 pt a
# line is 23 px here, and a 10 px offset put the names on top of the dots they name.
COL_X_TOL_PX = 6.0
COL_DX_LINES = 1.2    # common left edge, measured from the dots' x
COL_GAP_LINES = 0.15  # breathing room between stacked entries


def _place_labels(ax, xs, ys, labels, *, fontsize, pad_px=1.6, keepout_px=5.0,
                  reserved=(), all_points=None, prefer_right=(), nudge=None):
    """Offset placement that never drops a label.

    Candidates are scored by how much they overlap already-placed labels and *every* dot in the
    scatter, not just the labelled one; a clean position wins immediately, otherwise the
    least-overlapping one is kept. A label that ends up no nearer its own dot than to some other
    dot gets a hairline leader, so it cannot be read as belonging to the neighbour.

    all_points: (x, y) for the whole scatter, labelled or not. Unlabelled dots still have to be
    avoided -- a label parked on one of them reads as its name.
    """
    nudge = nudge or {}
    _to_nudge: list = []
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    # The spacings below were calibrated against 9 pt labels. They are pixel
    # distances, so raising the font size alone would leave labels sitting
    # proportionally closer to their neighbours -- which is what crowded PTPN7
    # against PIKFYVE. Scale them with the type instead.
    k = fontsize / 9.0
    pad_px, keepout_px = pad_px * k, keepout_px * k
    line_px = 1.35 * fontsize * fig.dpi / 72.0    # one line of text, in pixels
    # Crossing penalty in the same units as overlap_area, which is px squared. A flat 40
    # was nothing against a label box of ~8,000 px squared, so a candidate whose leader
    # ran through a neighbour's name always beat one that merely sat near a dot -- which
    # is how PTPN7's leader ended up through ALPK2. One em squared is ~2,800 px here, so
    # 3 em squared makes a crossing cost about as much as covering a short label.
    # Same units as overlap_area (px squared at the placement dpi). A flat 40 was nothing
    # against a label box of ~1,200 px squared here, so a leader through a neighbour's
    # name was effectively free; 2 em squared makes it cost about half a label.
    CROSS_PEN = 2.0 * (fontsize * fig.dpi / 72.0) ** 2
    placed: list[mtransforms.Bbox] = list(reserved)
    leaders: list[tuple[float, float, float, float]] = []   # drawn connector segments
    pts_px = [ax.transData.transform((x, y))
              for x, y in (all_points if all_points is not None else zip(xs, ys))]
    # Every dot is an obstacle, at a wider margin than a label's own keep-out: a label touching a
    # foreign dot is a misattribution, which is worse than two labels touching each other.
    dot_boxes = [mtransforms.Bbox.from_extents(px - keepout_px, py - keepout_px,
                                               px + keepout_px, py + keepout_px)
                 for px, py in pts_px]
    # Rightward first: the crowded cluster sits on the left spine, where a leftward label would
    # run off the axis.
    # Left and right both offered at each radius: a label to the left of its dot is often the
    # only uncrowded option, and the inside-axes check rules it out automatically where there is
    # no room (the x = 0 column).
    dirs = [(-1, 0, "right", "center"), (1, 0, "left", "center"),
            (-1, 1, "right", "bottom"), (1, 1, "left", "bottom"),
            (-1, -1, "right", "top"), (1, -1, "left", "top"),
            (0, 1, "center", "bottom"), (0, -1, "center", "top")]
    radii = tuple(r * k for r in (8, 13, 19, 26, 34, 44, 58, 74))
    ax_bb = ax.get_window_extent(renderer=rend)
    # Drawn whenever a label is not unambiguously nearest its own dot -- see needs_leader.
    # Radius alone is the wrong test: in the x = 0 column the dots are ~0.03 apart, closer
    # than one line of text is tall, so even a minimum-offset label lands between two of
    # them. Where the label does own its dot outright, no leader: it would just be a smudge.
    LEADER = dict(arrowstyle="-", lw=0.7, color="#666666", shrinkA=0.0, shrinkB=2.5)

    def dist_to_box(b, px, py):
        return float(np.hypot(max(b.x0 - px, 0.0, px - b.x1),
                              max(b.y0 - py, 0.0, py - b.y1)))

    def needs_leader(b, px, py):
        """True when some other dot is a plausible owner of this label.

        Both distances are measured from the point where the leader would attach -- the
        point of the text box nearest its own dot -- not from the box as a whole. Measured
        to the box, a dot sitting off to one side of a wide label scores as "close" because
        it is near the far end of the text, and the label gets a connector pointing away
        from it. That is what put a stray line on ALPK2, whose competing dot is 29 px from
        the text box but 53 px from where the line would actually start.
        """
        anchor = (min(max(px, b.x0), b.x1), min(max(py, b.y0), b.y1))
        own = dist_to_box(b, px, py)
        other = min((float(np.hypot(anchor[0] - qx, anchor[1] - qy))
                     for qx, qy in pts_px if (qx, qy) != (px, py)), default=float("inf"))
        # 2.2x: EGFR, the most displaced label that still owns its dot outright, sits at
        # 2.0 and keeps its leader; ALPK2 at 2.6 does not need one.
        if other < 2.2 * own:
            return True
        # The ratio test alone is not enough in a narrow panel. A dot pinned against
        # the axis edge leaves nowhere adjacent to put its name, so the label lands
        # off on the diagonal; if the nearest competing dot happens to be further
        # still, the ratio passes and the label is left floating unattached.
        #
        # 0.7 of a line splits the two groups this panel actually produces: the
        # labels that sit alongside their dot land at 0.56 of a line and stay clean,
        # while every diagonal placement lands at 0.79 or beyond and gets a
        # connector. Without it PTPN7 and MAPK1 -- displaced by exactly the same
        # distance -- were treated differently, which reads as arbitrary.
        return own > 0.7 * line_px

    def overlap_area(b, boxes):
        total = 0.0
        for o in boxes:
            ix = min(b.x1, o.x1) - max(b.x0, o.x0)
            iy = min(b.y1, o.y1) - max(b.y0, o.y0)
            if ix > 0 and iy > 0:
                total += ix * iy
        return total

    def leader_crossings(px, py, b):
        """How many already-placed labels the connector would be drawn through."""
        cx, cy = 0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1)
        n = 0
        for o in placed:
            # 24 samples along the segment: the boxes are tens of pixels wide, so a hit
            # cannot slip between samples on any leader this figure produces.
            if any(o.contains(px + f * (cx - px), py + f * (cy - py))
                   for f in np.linspace(0.0, 1.0, 24)):
                n += 1
        return n

    def on_existing_leader(b):
        """How many already-drawn leaders pass through this candidate box.

        leader_crossings covers the other asymmetry -- a new leader crossing an old
        label. Without this one, a label placed later can land squarely on an earlier
        label's connector, which is what put ALPK2 on PTPN7's leader.
        """
        n = 0
        for lx, ly, mx, my in leaders:
            if any(b.contains(lx + f * (mx - lx), ly + f * (my - ly))
                   for f in np.linspace(0.0, 1.0, 24)):
                n += 1
        return n

    def crowding(px, py):
        """How many other dots sit within a label's typical reach."""
        return sum(1 for qx, qy in pts_px
                   if abs(qx - px) < 55 and abs(qy - py) < 22 and (qx, qy) != (px, py))

    # ---- Packed columns are labelled as one aligned block ---------------------------
    # Where neighbouring dots sit closer together than a line of text is tall, no per-label
    # offset can be both near its own dot and clear of the next one: that is how the left
    # spine ended up with four names floating at ragged heights among five dots, none of
    # them obviously belonging to anything. Such a cluster gets a callout stack instead --
    # one left edge, even spacing, the dots' own top-to-bottom order, a leader on every
    # entry. Aligned-and-connected reads better than individually-nearest-but-ambiguous.
    probe = ax.annotate("Ag", (xs[0], ys[0]), xytext=(0, 0),
                        textcoords="offset points", fontsize=fontsize)
    fig.canvas.draw()
    line_h = probe.get_window_extent(renderer=rend).height
    probe.remove()
    lab_px = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]

    stacked: set[int] = set()
    columns: list[list[int]] = []
    for i in range(len(lab_px)):
        if i in stacked:
            continue
        grp = [j for j in range(len(lab_px))
               if j not in stacked and abs(lab_px[j][0] - lab_px[i][0]) <= COL_X_TOL_PX]
        if len(grp) < 3:
            continue
        grp.sort(key=lambda j: -lab_px[j][1])        # top of the panel first
        # One colliding pair is enough to stack the whole column. Summary statistics do not
        # work here: the median gap hides two impossible gaps behind one roomy one, and the
        # total span calls the 40-marker column comfortable while two of its dots sit 0.007
        # apart. If any adjacent pair is closer than a line of text, the labels cannot all
        # sit at their own heights, and a partly-stacked column would be worse than either
        # alternative -- some names aligned, others not, with no rule a reader can infer.
        gaps = [lab_px[grp[k]][1] - lab_px[grp[k + 1]][1] for k in range(len(grp) - 1)]
        if min(gaps) >= line_h:                      # they fit where they are
            continue
        stacked.update(grp)
        columns.append(grp)

    px_to_pt = 72.0 / fig.dpi
    for grp in columns:
        step = line_h * (1.0 + COL_GAP_LINES)
        # Centred on the cluster, so the stack drifts as little as possible from the dots
        # it names and the leaders stay short and roughly parallel.
        top = float(np.mean([lab_px[j][1] for j in grp])) + step * (len(grp) - 1) / 2.0
        for k, j in enumerate(grp):
            dy_px = (top - k * step) - lab_px[j][1]
            t = ax.annotate(labels[j], (xs[j], ys[j]),
                            xytext=(line_h * COL_DX_LINES * px_to_pt, dy_px * px_to_pt),
                            textcoords="offset points", fontsize=fontsize,
                            ha="left", va="center", zorder=6, arrowprops=LEADER)
            fig.canvas.draw()
            b = t.get_window_extent(renderer=rend)
            placed.append(mtransforms.Bbox.from_extents(b.x0 - pad_px, b.y0 - pad_px,
                                                        b.x1 + pad_px, b.y1 + pad_px))

    # Most crowded first, so the dots with the fewest legal positions claim them before the
    # roomy ones do. Sorting by coordinate instead (the top-right corner first) served the
    # dense x = 0 column last and flung its labels halfway across the panel.
    # Most crowded first, ties broken top-down. The tie-break matters: with crowding
    # alone, ALPK2 chose before PTPN7 and took the only corridor out of the top-right
    # corner, leaving PTPN7 no route to its dot that did not cross the ALPK2 label. No
    # crossing penalty fixes that -- every alternative for PTPN7 was worse -- but letting
    # the higher dot pick first does, and it leaves every other label where it was.
    def _order_key(t):
        px_, py_ = ax.transData.transform((t[0], t[1]))
        return (-crowding(px_, py_), -py_)
    order = sorted((t for i, t in enumerate(zip(xs, ys, labels)) if i not in stacked),
                   key=_order_key)
    for x, y, lab in order:
        px, py = ax.transData.transform((x, y))
        own = mtransforms.Bbox.from_extents(px - keepout_px, py - keepout_px,
                                           px + keepout_px, py + keepout_px)
        # A named label can ask for a side. The scoring below is blind to which half of the
        # panel a label lands in, so where both sides score zero it just takes the first
        # direction offered; this is the hook for overriding that.
        lab_dirs = dirs
        if lab in prefer_right:
            lab_dirs = [d for d in dirs if d[0] > 0] + [d for d in dirs if d[0] <= 0]
        best = None          # (score, artist, bbox, placement)
        for rad in radii:
            for sx, sy, ha, va in lab_dirs:
                # Searched without the leader: the arrow does not change the text bbox, and
                # whether one is needed is only decidable once the position is settled.
                t = ax.annotate(lab, (x, y), xytext=(sx * rad, sy * rad),
                                textcoords="offset points", fontsize=fontsize,
                                ha=ha, va=va, zorder=6)
                fig.canvas.draw()
                b = t.get_window_extent(renderer=rend)
                b = mtransforms.Bbox.from_extents(b.x0 - pad_px, b.y0 - pad_px,
                                                  b.x1 + pad_px, b.y1 + pad_px)
                inside = (b.x0 >= ax_bb.x0 and b.x1 <= ax_bb.x1
                          and b.y0 >= ax_bb.y0 and b.y1 <= ax_bb.y1)
                # Sitting on a foreign dot is weighted well above merely clipping another
                # label: the first misnames a point, the second is only untidy.
                score = (overlap_area(b, placed)
                         + 8.0 * overlap_area(b, dot_boxes)
                         + CROSS_PEN * leader_crossings(px, py, b)
                         + CROSS_PEN * on_existing_leader(b)
                         + (0.0 if inside else 1e6)
                         + (1e5 if b.overlaps(own) else 0.0)
                         + 0.6 * rad)   # break ties towards the dot, so leaders stay short
                if score == 0.6 * rad:
                    if best is not None:
                        best[1].remove()   # else the earlier candidate stays drawn -> duplicate
                    best = (score, t, b, (rad, sx, sy, ha, va))
                    break
                if best is None or score < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (score, t, b, (rad, sx, sy, ha, va))
                else:
                    t.remove()
            if best is not None and best[0] == 0.6 * best[3][0]:
                break
        if best is not None:
            _, t, b, (rad, sx, sy, ha, va) = best
            # Connectors only where the name has drifted off its dot. Blanket connectors
            # are indeed more distracting than they are worth -- most labels here sit
            # right beside their point and need nothing. But in a narrow panel the dots
            # near the top-right corner have no adjacent free space, so their names land
            # a long way off on the diagonal and read as belonging to whatever they have
            # drifted next to. Those get a leader; the adjacent majority still do not.
            if needs_leader(b, px, py):
                t.remove()
                t = ax.annotate(lab, (x, y), xytext=(sx * rad, sy * rad),
                                textcoords="offset points", fontsize=fontsize,
                                ha=ha, va=va, zorder=6, arrowprops=LEADER)
                fig.canvas.draw()
                # Keep the box from the search, which is the TEXT. Recomputing it here
                # measured the annotation, whose extent includes the arrow, so a label
                # with a leader occupied a phantom box reaching all the way back to its
                # dot -- later labels then dodged empty space, and the footprint the
                # search had optimised was thrown away.
                leaders.append((px, py, 0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1)))
            placed.append(b)
            if lab in nudge:
                _to_nudge.append((lab, t, sx * rad, sy * rad))

    # Nudges last, once every label is placed. Applying one inside the loop changed the
    # boxes later labels were avoiding, which moved PTPN7 to a slot it no longer needed a
    # leader for -- a knock-on from what is meant to be a purely cosmetic shift.
    for lab, t, ox, oy in _to_nudge:
        nx, ny = nudge[lab]
        t.set_position((ox + nx, oy + ny))
    if _to_nudge:
        fig.canvas.draw()



def load_accuracy_frame(*, linreg_metrics, pysr_dir, pysr_integ_csv, seeds, linreg_k,
                        exclude=(), sr_best_train=False):
    """Per-marker linear-regression vs SR OOD R², and the y-axis label that describes it.

    Lifted out of main() so the combined Fig. 4 script draws this panel from the same code
    rather than a copy -- a copy is how the two versions of a panel silently diverge.
    """
    lin = pd.read_csv(linreg_metrics)
    lin = lin[(lin.split == "test") & (lin.k == linreg_k)]
    lin_r2 = lin.groupby("marker")["ode_integ_r2_median"].max()

    sr_per_seed: dict[str, dict[int, float]] = {}
    for seed in seeds:
        fp = Path(pysr_dir) / f"seed_{seed}" / "formulas" / "all_per_minute.txt"
        if not fp.exists():
            continue
        for marker, r2 in _parse_pysr(fp).items():
            sr_per_seed.setdefault(marker, {})[seed] = r2
    sr_r2 = pd.Series({m: max(v.values()) for m, v in sr_per_seed.items() if v})

    if pysr_integ_csv:
        # Note this deliberately does NOT fall back to the derivative value for markers the CSV
        # is missing: filling gaps would invent data points (a fillna(0.0) elsewhere in this
        # project made the controls look like uniform PySR failures when they are its easiest
        # markers). Missing markers drop out of the intersection instead.
        integ = pd.read_csv(pysr_integ_csv)
        # `ode_integ_r2_median` is the column the integration stage writes and so the one
        # the pipeline's own table carries; `ode_r2` is the name the earlier hand-built
        # table used. Accept either rather than silently requiring the older one.
        for column in ("ode_integ_r2_median", "ode_r2"):
            if column in integ.columns:
                break
        else:
            raise SystemExit(
                f"{pysr_integ_csv} has neither 'ode_integ_r2_median' nor 'ode_r2'; "
                f"columns are {sorted(integ.columns)}"
            )
        if sr_best_train and "ode_integ_r2_median_train" in integ.columns:
            # Same rule as Fig. 4E and Table S8: retain the seed with the best TRAINING
            # score and report its held-out value. Taking the max over seeds instead is
            # selection on the test set -- it admits contexts only one seed generalises on,
            # and it put 15 markers above the cutoff where the text reports 13.
            pick = integ.loc[integ.groupby("marker").ode_integ_r2_median_train.idxmax()]
            sr_r2 = pick.set_index("marker")[column]
            rule = "seed selected by training R²"
        else:
            sr_r2 = integ.groupby("marker")[column].max()
            rule = "best of 3 seeds"
        print(f"  PySR axis: ODE-integrated OOD R² from {pysr_integ_csv} "
              f"column '{column}' ({len(sr_r2)} markers, {rule})")

    common = sorted((set(lin_r2.index) & set(sr_r2.index)) - set(exclude))
    df = pd.DataFrame({
        "marker": common,
        "lin": [max(0.0, float(lin_r2[m])) for m in common],
        "sr": [max(0.0, float(sr_r2[m])) for m in common],
    })
    df["gap"] = df.sr - df.lin
    n_sr_better = int((df.gap > 0).sum())
    r = float(np.corrcoef(df.lin, df.sr)[0, 1]) if len(df) > 2 else float("nan")
    print(f"  n={len(df)}  SR better={n_sr_better}  r={r:.2f}  "
          f"mean SR={df.sr.mean():.3f}  mean LinReg={df.lin.mean():.3f}")

    # The selection rule belongs in the label, because the two rules give different panels.
    conv = "seed by training R²" if (pysr_integ_csv and sr_best_train) else "best of 3 seeds"
    ylab = (f"SR OOD ODE-integrated R² ({conv})" if pysr_integ_csv
            else f"SR OOD test R² ({conv})")
    return df, ylab


# Labels that must sit to the right of their dot. Both sides are clear for ERBB2, so the
# placer had no reason to prefer either and took the left one on offer; it belongs on the
# right, away from the filled half-plane it does not belong to. Display names, not marker
# ids -- these are matched after gene_labels().
LABEL_PREFER_RIGHT = frozenset({"ERBB2"})

# Per-label final nudge, in points, (dx, dy) with y up. The placer optimises collisions,
# not balance, so a name it puts legally can still sit awkwardly close to a neighbour.
# Applied after the search, so it cannot change which slot was chosen; the recorded box
# is translated to match, keeping `placed` and the leader endpoints honest.
LABEL_NUDGE = {"PIKFYVE": (6.0, -6.0)}


def draw_accuracy_panel(fig, ax, df, *, linreg_k, ylab, label_threshold=0.6,
                        fs_axis=8.0, fs_dot=9.0, title="Accuracy out of distribution",
                        label_skip=()):
    """Draw the SR-vs-linreg scatter onto `ax`.

    Font sizes are arguments rather than module constants so the host figure sets one type
    scale for every panel it contains.

    `label_skip` drops names from the callouts without dropping their dots. In the
    four-panel row this axes is height-limited by the figure row, not by its gridspec
    cell, so it cannot be widened and the labels cannot be shrunk either -- the figure is
    drawn 11 in wide and placed at ~7 in, so fs_dot 9.6 is already ~6 pt on the page.
    Fewer names is the only lever left, and the controls are the ones to lose: they carry
    the longest strings ("untransfected2") for the least biology.
    """
    GREEN_FILL = "#cdecd9"      # SR-wins half-plane
    INK = "#111111"             # markers

    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)

    # Green covers every quadrant except the bottom-left: at least one of the two
    # methods clears the threshold. The white corner is the set neither extrapolates,
    # and the SR-beats-baseline comparison is read off the dashed diagonal.
    ax.fill_between([0, 1], label_threshold, 1,
                    facecolor=GREEN_FILL, edgecolor="none", zorder=0)   # top half
    ax.fill_between([label_threshold, 1], 0, label_threshold,
                    facecolor=GREEN_FILL, edgecolor="none", zorder=0)   # bottom right
    ax.grid(True, which="major", color="#e8e7e3", lw=0.6, zorder=1)
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.9, zorder=2)
    # Satisfactory-performance threshold on both axes, in the same ink as the neural-ODE
    # panel it sits beside, so "clears 0.6" is read off one gridline in both scatters
    # rather than off a line in one and a number in the other.
    ax.axvline(label_threshold, color="#333", lw=1.0, ls=(0, (5, 3)), alpha=0.75, zorder=2)
    ax.axhline(label_threshold, color="#333", lw=1.0, ls=(0, (5, 3)), alpha=0.75, zorder=2)
    ax.scatter(df.lin, df.sr, s=26, c=INK, alpha=0.92,
               edgecolors="white", linewidths=0.5, zorder=3)

    # Everything that moves the axes has to happen before the labels are placed. The limits
    # are what fix the axes box (aspect="equal" with adjustable="box" reshapes it when they
    # change), and placement is measured in pixels against that box.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Integrated linear regression test R²", fontsize=fs_axis)
    ax.set_ylabel(ylab, fontsize=fs_axis)
    handle = ax.set_title(title, loc="left", fontsize=fs_axis,
                          fontweight="bold", color="#111111", pad=5)

    lab = df[df[["lin", "sr"]].max(axis=1) >= label_threshold]
    if len(label_skip):
        lab = lab[~lab.marker.isin(set(label_skip))]
    # The title sits just above the axes and the top row of dots reaches R² = 0.94, so a
    # label there runs into it unless the title is an obstacle like any other.
    fig.canvas.draw()
    _place_labels(ax, lab.lin.values, lab.sr.values, gene_labels(lab.marker).values,
                  fontsize=fs_dot,
                  all_points=list(zip(df.lin.values, df.sr.values)),
                  reserved=[handle.get_window_extent(fig.canvas.get_renderer())],
                  prefer_right=LABEL_PREFER_RIGHT, nudge=LABEL_NUDGE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linreg-metrics", required=True,
                    help="select_k_metrics.csv from the top_gfp_bins OOD run.")
    ap.add_argument("--pysr-dir", required=True,
                    help="Directory holding seed_*/formulas/all_per_minute.txt.")
    ap.add_argument("--pysr-integ-csv",
                    help="Optional CSV with columns marker,ode_r2 (per fit). When given, the "
                         "PySR axis uses ODE-integrated OOD R² from here instead of the "
                         "derivative 'Test R2' in the formulas dump. The two conventions "
                         "correlate only ~0.5, and the linreg axis is already ODE-integrated, "
                         "so this is what makes the comparison like-for-like.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--linreg-k", type=int, default=10,
                    help="Which k to use for the linear-regression baseline "
                         "(10 = all features, the strongest LinReg).")
    ap.add_argument("--label-threshold", type=float, default=0.6,
                    help="Label a marker when either method clears this R².")
    ap.add_argument("--exclude-marker", action="append", default=[])
    ap.add_argument("--sr-best-train", action="store_true",
                    help="Retain the PySR seed with the best TRAINING integrated R2 and "
                         "report its held-out score, as Fig. 4E and Table S8 do, instead "
                         "of taking the best held-out score across seeds.")
    args = ap.parse_args()

    df, ylab = load_accuracy_frame(
        linreg_metrics=args.linreg_metrics, pysr_dir=args.pysr_dir,
        pysr_integ_csv=args.pysr_integ_csv, seeds=args.seeds,
        linreg_k=args.linreg_k, exclude=args.exclude_marker,
        sr_best_train=args.sr_best_train,
    )

    # FS covers the axis labels and title; ticks and the per-dot marker labels are set
    # independently because they are read differently -- ticks are scanned, and a marker
    # label has to be legible next to the dot it names.
    # FS_TICK matches plot_parsimony_tradeoff.py's 10.0 so the tick numerals here and in
    # panels E/F are the same size when the two are printed at their native widths. Where
    # the panels share one figure (plot_fig4_combined.py) the host sets the scale instead.
    # Bumped 20% over the original 8.0/10.0/9.0 for legibility at print size.
    FS, FS_TICK, FS_DOT = 9.6, 12.0, 10.8
    mpl.rcParams.update({
        # Arial (not Helvetica: macOS registers only Helvetica's regular face,
        # so bold silently falls back). Type 42 because many journals reject
        # matplotlib's default Type 3.
        "font.family": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333", "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    draw_accuracy_panel(fig, ax, df, linreg_k=args.linreg_k, ylab=ylab,
                        label_threshold=args.label_threshold, fs_axis=FS, fs_dot=FS_DOT)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
