"""One box column for closure cost, instead of separate dependency and interaction columns.

Why combine them
----------------
The paper currently reports two parsimony measures side by side: the number of measured
variables the dynamics depend on (k), and the number of those variables' pairs that interact
(m). They are not independent -- m is bounded by C(k, 2), and the networks sit at ~0.98 of
that bound -- so two boxes spend two panels and two significance tests on close to one fact,
and a reader cannot tell whether the second panel is corroboration or double counting.

A closed form has to carry a term per dependency and a term per interaction, so the natural
single quantity is their weighted sum,

    closure cost  C(rho) = k + rho * m

with rho the price of an interaction term relative to a linear one. rho = 0 is the pure
dependency count; large rho is the pure interaction count; anything between is a compromise
whose two limits are the panels being replaced. Because the answer depends on rho, the
figure shows the whole rho curve beside the chosen panel rather than one flattering value:
if the separation only exists in a narrow window of rho, that is visible here.

Usage:
    python plot_fig4_composite_complexity.py \
        --panel-inputs <dump from plot_parsimony_tradeoff.py --dump-inputs> \
        --rho 1.0 --output <png>
"""
from __future__ import annotations

import argparse
import itertools
from math import comb
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from scipy import stats                                               # noqa: E402

SAVE_DPI = 300
NN_COLOR = "#2a78d6"
SR_COLOR = "#c8781e"
GRID = "#e5e5e5"


def perm_p(values, is_fail, covariate, max_exact: int = 200_000) -> float:
    """Two-sided permutation p for the group difference after regressing out `covariate`.

    Identical to plot_parsimony_tradeoff._perm_p, so the number this figure prints is
    comparable with the one the two-column version prints. Residualising on accuracy matters
    because the groups are defined by a model's success, and a more accurate fit has more
    room to be complex; without it the test can read an accuracy gap as a complexity gap.
    """
    v = np.asarray(values, float)
    f = np.asarray(is_fail, bool)
    x = np.asarray(covariate, float)
    n, nf = len(v), int(f.sum())
    if nf < 2 or n - nf < 2:
        return float("nan")
    X = np.column_stack([np.ones(n), x])
    res = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    obs = res[f].mean() - res[~f].mean()
    if comb(n, nf) <= max_exact:
        idx = range(n)
        null = np.array([res[list(c)].mean() - res[[i for i in idx if i not in c]].mean()
                         for c in itertools.combinations(idx, nf)])
    else:
        rng = np.random.default_rng(0)
        null = np.empty(20_000)
        for j in range(null.size):
            pm = rng.permutation(n)
            null[j] = res[pm[:nf]].mean() - res[pm[nf:]].mean()
    return float((np.abs(null) >= abs(obs) - 1e-12).mean())


def auc_of(fail, solve) -> float:
    """P(a failed context scores higher than a solved one), ties at a half."""
    fail, solve = np.asarray(fail, float), np.asarray(solve, float)
    if not fail.size or not solve.size:
        return float("nan")
    gt = (fail[:, None] > solve[None, :]).sum()
    eq = (fail[:, None] == solve[None, :]).sum()
    return float((gt + 0.5 * eq) / (fail.size * solve.size))


def cost(df: pd.DataFrame, rho: float, who: str) -> np.ndarray:
    k = df[f"{who}_pr"].to_numpy(float)
    m = df[f"{who}_inter"].to_numpy(float)
    return k + rho * m


def stats_at(df: pd.DataFrame, rho: float, thr: float) -> dict:
    """Every number the panel quotes, at one rho."""
    solved = df.py_r2 >= thr
    c = cost(df, rho, "nn")
    out = {"rho": rho}

    # All 40: the headline contrast, network closure cost by whether SR recovered a law.
    f, s = c[~solved.to_numpy()], c[solved.to_numpy()]
    out["all_fail"], out["all_solve"] = f.mean(), s.mean()
    out["all_p"] = stats.mannwhitneyu(f, s, alternative="two-sided").pvalue
    out["all_auc"] = auc_of(f, s)

    # Gated: only where the network generalises is its closure cost a statement about the
    # dynamics rather than about a model that failed to fit them.
    g = df[df.nn_r2 >= thr]
    gs = (g.py_r2 >= thr).to_numpy()
    cg = cost(g, rho, "nn")
    out["n_gated"], out["n_gated_solve"] = len(g), int(gs.sum())
    out["gate_fail"], out["gate_solve"] = cg[~gs].mean(), cg[gs].mean()
    out["gate_p"] = stats.mannwhitneyu(cg[~gs], cg[gs], alternative="two-sided").pvalue
    out["gate_perm_p"] = perm_p(cg, ~gs, g.nn_r2.to_numpy(float))
    out["gate_auc"] = auc_of(cg[~gs], cg[gs])

    if "recovery" in df.columns:
        r = stats.spearmanr(c, df.recovery.to_numpy(float))
        out["rho_recovery"], out["p_recovery"] = r.statistic, r.pvalue
    return out


def draw_box(ax, x, vals, face, alpha, rng, width=0.52):
    """House box: IQR body, median rule, whiskers to the extremes, jittered points."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.add_patch(plt.Rectangle((x - width / 2, q1), width, q3 - q1,
                               facecolor=face, alpha=alpha, linewidth=0, zorder=2))
    ax.plot([x - width / 2, x + width / 2], [med, med], color=face, lw=2.4, zorder=4,
            solid_capstyle="butt")
    ax.plot([x, x], [v.min(), v.max()], color=face, lw=1.0, zorder=1)
    ax.scatter(x + rng.uniform(-0.13, 0.13, v.size), v, s=13, color=face,
               alpha=0.85, linewidths=0, zorder=5)
    return v


def bracket(ax, x0, x1, y, label, fs):
    ax.plot([x0, x0, x1, x1], [y, y + 0.012, y + 0.012, y], transform=ax.get_xaxis_transform(),
            clip_on=False, lw=0.9, color="#4a4a4a")
    ax.text((x0 + x1) / 2, y + 0.028, label, transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=fs, color="#333")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-inputs", type=Path, required=True,
                    help="CSV from plot_parsimony_tradeoff.py --dump-inputs "
                         "(marker, nn_r2, py_r2, nn_pr, py_pr, nn_inter, py_inter).")
    ap.add_argument("--table-s8", type=Path,
                    help="Optional table_s8_sr_per_context.csv, to add the graded test "
                         "against the fraction of seeds that recovered a generalising law.")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="Price of an interaction term relative to a linear one.")
    ap.add_argument("--rho-max", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.panel_inputs)
    need = {"marker", "nn_r2", "py_r2", "nn_pr", "py_pr", "nn_inter", "py_inter"}
    if not need.issubset(df.columns):
        raise SystemExit(f"--panel-inputs needs {sorted(need)}; got {sorted(df.columns)}")
    df = df.dropna(subset=sorted(need - {"marker"})).reset_index(drop=True)

    if args.table_s8 and args.table_s8.exists():
        s8 = pd.read_csv(args.table_s8)
        col = "Seed recovery"
        if col in s8.columns:
            rec = (s8[col].astype(str).str.split("/")
                   .apply(lambda p: float(p[0]) / float(p[1]) if len(p) == 2 else np.nan))
            df = df.merge(s8[["Context"]].assign(recovery=rec),
                          left_on="marker", right_on="Context", how="left").drop(
                              columns="Context")

    at = stats_at(df, args.rho, args.threshold)
    print(f"closure cost C = k + {args.rho:g} x pairs")
    print(f"  all 40   failed {at['all_fail']:.2f} vs solved {at['all_solve']:.2f}   "
          f"MWU P = {at['all_p']:.4g}  AUC {at['all_auc']:.3f}")
    print(f"  gated n={at['n_gated']} ({at['n_gated_solve']} solved)  "
          f"failed {at['gate_fail']:.2f} vs solved {at['gate_solve']:.2f}   "
          f"MWU P = {at['gate_p']:.4g}  perm P = {at['gate_perm_p']:.4g}  "
          f"AUC {at['gate_auc']:.3f}")
    if "rho_recovery" in at:
        print(f"  graded: Spearman rho = {at['rho_recovery']:.3f}  "
              f"P = {at['p_recovery']:.4g}  (cost vs fraction of seeds recovering)")

    # rho curve, so the choice above is auditable rather than asserted
    grid = np.round(np.arange(0.0, args.rho_max + 1e-9, 0.1), 2)
    curve = pd.DataFrame([stats_at(df, r, args.threshold) for r in grid])

    FS, FS_T = 12.0, 10.5
    mpl.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333", "axes.linewidth": 0.7,
    })
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(5.6, 2.45),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    rng = np.random.default_rng(0)

    # ---- panel A: the single box column ----
    solved = (df.py_r2 >= args.threshold).to_numpy()
    gated = (df.nn_r2 >= args.threshold).to_numpy()
    groups = [("SR", cost(df, args.rho, "py")[solved], SR_COLOR, 0.30),
              ("✓", cost(df, args.rho, "nn")[gated & solved], NN_COLOR, 0.30),
              ("✗", cost(df, args.rho, "nn")[gated & ~solved], NN_COLOR, 0.16)]
    axb.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    for i, (_, v, face, alpha) in enumerate(groups):
        draw_box(axb, i, v, face, alpha, rng)
    axb.set_xticks(range(3))
    axb.set_xticklabels([f"{n}\nn={np.isfinite(v).sum()}" for n, v, _, _ in groups])
    for t, (_, _, face, _) in zip(axb.get_xticklabels(), groups):
        t.set_color(face)
    axb.set_ylabel(f"Closure cost  k + {args.rho:g}·pairs", fontsize=FS)
    axb.set_xlim(-0.6, 2.6)
    lo = min(v.min() for _, v, _, _ in groups)
    hi = max(v.max() for _, v, _, _ in groups)
    axb.set_ylim(max(0.0, lo - 0.06 * (hi - lo)), hi + 0.30 * (hi - lo))
    p_ns = stats.mannwhitneyu(groups[0][1], groups[1][1], alternative="two-sided").pvalue
    bracket(axb, 0, 1, 0.88, "ns" if p_ns >= 0.05 else f"P = {p_ns:.3f}", FS_T)
    star = "*" if at["gate_perm_p"] < 0.05 else "n.s."
    bracket(axb, 1, 2, 0.97, star, FS_T)

    # ---- panel B: does the answer depend on rho? ----
    axr.grid(True, color=GRID, lw=0.7, zorder=0)
    axr.plot(curve.rho, curve.gate_perm_p, color=NN_COLOR, lw=1.8, zorder=3,
             label="gated, permutation")
    axr.plot(curve.rho, curve.all_p, color="#777", lw=1.5, ls=(0, (4, 2)), zorder=3,
             label="all 40, Mann–Whitney")
    axr.axhline(0.05, color="#c0392b", lw=1.0, ls=(0, (2, 2)), zorder=2)
    axr.axvline(args.rho, color="#333", lw=0.9, alpha=0.5, zorder=2)
    axr.set_yscale("log")
    axr.set_ylim(0.005, 0.30)
    axr.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    axr.set_yticklabels(["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"])
    axr.minorticks_off()
    axr.text(args.rho_max, 0.053, "P = 0.05", color="#c0392b", fontsize=FS_T,
             va="bottom", ha="right")
    axr.set_xlabel("Interaction price ρ", fontsize=FS)
    axr.set_ylabel("P", fontsize=FS)
    axr.set_xlim(0, args.rho_max)
    axr.legend(frameon=False, fontsize=FS_T, loc="upper right", handlelength=1.6,
               borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(w_pad=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    csv = args.output.with_name(args.output.stem + "_rho_curve.csv")
    curve.to_csv(csv, index=False)
    print(f"\nwrote {args.output}, {args.output.with_suffix('.pdf')} and {csv}")

    sig = curve[(curve.gate_perm_p < 0.05)]
    if len(sig):
        print(f"  gated permutation P < 0.05 for rho in "
              f"[{sig.rho.min():g}, {sig.rho.max():g}] of [0, {args.rho_max:g}] "
              f"({100 * len(sig) / len(curve):.0f}% of the grid)")
    else:
        print("  gated permutation P never drops below 0.05 on this rho grid")


if __name__ == "__main__":
    main()
