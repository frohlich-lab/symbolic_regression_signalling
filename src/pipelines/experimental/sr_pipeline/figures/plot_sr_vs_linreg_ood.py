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
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


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


def _place_labels(ax, xs, ys, labels, *, fontsize, pad_px=1.0, keepout_px=3.2,
                  reserved=()):
    """Greedy 8-direction offset placement; skips a label rather than collide."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    # Seeded with any artist we must not collide with (e.g. the stats block).
    placed: list[mtransforms.Bbox] = list(reserved)
    dirs = [(1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
            (1, -1, "left", "top"), (-1, -1, "right", "top"),
            (1, 0, "left", "center"), (-1, 0, "right", "center"),
            (0, 1, "left", "bottom"), (0, -1, "left", "top")]
    ax_bb = ax.get_window_extent(renderer=rend)
    for x, y, lab in sorted(zip(xs, ys, labels), key=lambda t: -max(t[0], t[1])):
        best = None
        for rad in (3, 5, 8, 12, 18):
            for sx, sy, ha, va in dirs:
                t = ax.annotate(lab, (x, y), xytext=(sx * rad, sy * rad),
                                textcoords="offset points", fontsize=fontsize,
                                ha=ha, va=va, zorder=6)
                fig.canvas.draw()
                b = t.get_window_extent(renderer=rend)
                b = mtransforms.Bbox.from_extents(b.x0 - pad_px, b.y0 - pad_px,
                                                  b.x1 + pad_px, b.y1 + pad_px)
                px, py = ax.transData.transform((x, y))
                own = mtransforms.Bbox.from_extents(px - keepout_px, py - keepout_px,
                                                    px + keepout_px, py + keepout_px)
                inside = (b.x0 >= ax_bb.x0 and b.x1 <= ax_bb.x1
                          and b.y0 >= ax_bb.y0 and b.y1 <= ax_bb.y1)
                if inside and not b.overlaps(own) and not any(b.overlaps(o) for o in placed):
                    best = b
                    break
                t.remove()
            if best is not None:
                break
        if best is not None:
            placed.append(best)


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
    args = ap.parse_args()

    lin = pd.read_csv(args.linreg_metrics)
    lin = lin[(lin.split == "test") & (lin.k == args.linreg_k)]
    lin_r2 = lin.groupby("marker")["ode_integ_r2_median"].max()

    sr_per_seed: dict[str, dict[int, float]] = {}
    for seed in args.seeds:
        fp = Path(args.pysr_dir) / f"seed_{seed}" / "formulas" / "all_per_minute.txt"
        if not fp.exists():
            continue
        for marker, r2 in _parse_pysr(fp).items():
            sr_per_seed.setdefault(marker, {})[seed] = r2
    sr_r2 = pd.Series({m: max(v.values()) for m, v in sr_per_seed.items() if v})

    if args.pysr_integ_csv:
        # Note this deliberately does NOT fall back to the derivative value for markers the CSV
        # is missing: filling gaps would invent data points (a fillna(0.0) elsewhere in this
        # project made the controls look like uniform PySR failures when they are its easiest
        # markers). Missing markers drop out of the intersection instead.
        integ = pd.read_csv(args.pysr_integ_csv)
        # `ode_integ_r2_median` is the column the integration stage writes and so the one
        # the pipeline's own table carries; `ode_r2` is the name the earlier hand-built
        # table used. Accept either rather than silently requiring the older one.
        for column in ("ode_integ_r2_median", "ode_r2"):
            if column in integ.columns:
                break
        else:
            raise SystemExit(
                f"{args.pysr_integ_csv} has neither 'ode_integ_r2_median' nor 'ode_r2'; "
                f"columns are {sorted(integ.columns)}"
            )
        sr_r2 = integ.groupby("marker")[column].max()
        print(f"  PySR axis: ODE-integrated OOD R² from {args.pysr_integ_csv} "
              f"column '{column}' ({len(sr_r2)} markers)")

    common = sorted((set(lin_r2.index) & set(sr_r2.index)) - set(args.exclude_marker))
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

    FS, FS_DOT = 8.0, 6.0
    GREEN_FILL = "#cdecd9"      # SR-wins half-plane
    INK = "#111111"             # markers
    mpl.rcParams.update({
        # Arial (not Helvetica: macOS registers only Helvetica's regular face,
        # so bold silently falls back). Type 42 because many journals reject
        # matplotlib's default Type 3.
        "font.family": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333", "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    })

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)

    # The half-plane above y=x is where SR beats the linear baseline. Filling it
    # means the comparison is read by region, not by eyeballing the diagonal.
    ax.fill_between([0, 1], [0, 1], 1, facecolor=GREEN_FILL, edgecolor="none",
                    zorder=0)
    ax.grid(True, which="major", color="#e8e7e3", lw=0.6, zorder=1)
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.9, zorder=2)
    ax.scatter(df.lin, df.sr, s=26, c=INK, alpha=0.92,
               edgecolors="white", linewidths=0.5, zorder=3)

    lab = df[df[["lin", "sr"]].max(axis=1) >= args.label_threshold]
    _place_labels(ax, lab.lin.values, lab.sr.values, lab.marker.values,
                  fontsize=FS_DOT)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(f"Linear regression OOD test R² (k={args.linreg_k})", fontsize=FS)
    ylab = ("SR OOD ODE-integrated R² (best of 3 seeds)" if args.pysr_integ_csv
            else "SR OOD test R² (best of 3 seeds)")
    ax.set_ylabel(ylab, fontsize=FS)
    ax.set_title("Accuracy out of distribution", loc="left", fontsize=FS,
                 fontweight="bold", color="#111111", pad=5)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
