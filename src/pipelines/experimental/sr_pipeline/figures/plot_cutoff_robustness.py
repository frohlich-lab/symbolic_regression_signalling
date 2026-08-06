"""Main-text figure: driver-set robustness vs top-X% mass cutoff.

Sweeps the cumulative-|J| mass cutoff from 50% to 100% and plots:
  Panel A — Jaccard (% of union shared) for three lines:
      L21 between-seed   (on its R² > θ markers)
      PySR between-seed  (on its R² > θ markers)
      L21 ↔ PySR         (seed-stable sets, on markers where both work)
  Panel B — mean number of features required to cover the top-X% mass.

GFP and p_ERK1_2 (state) are excluded throughout — PySR is forced to use both.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
EXCLUDE = {"GFP", "p_ERK1_2"}
KEEP_IDX = [i for i, f in enumerate(FEATURES) if f not in EXCLUDE]
KEEP = [FEATURES[i] for i in KEEP_IDX]


def top_set(values: np.ndarray, names: list[str], cutoff: float) -> set:
    v = np.abs(np.asarray(values))
    total = v.sum()
    if total == 0:
        return set()
    order = np.argsort(-v)
    cum = np.cumsum(v[order])
    k = int(np.searchsorted(cum, cutoff * total)) + 1
    return {names[i] for i in order[:k]}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / max(len(a | b), 1)


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


def _pysr_jac(formula: str, marker: str, data: pd.DataFrame) -> np.ndarray | None:
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
    ap.add_argument("--pysr-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--r2-threshold", type=float, default=0.6)
    ap.add_argument("--cutoffs", nargs="+", type=float,
                    default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
    ap.add_argument("--exclude-marker", action="append", default=["untransfected1"])
    args = ap.parse_args()

    nn_dir = Path(args.nn_dir)
    pysr_dir = Path(args.pysr_dir)
    seeds = args.seeds
    threshold = args.r2_threshold
    cutoffs = np.array(args.cutoffs)

    # NN per-seed R² and Jacobians
    nn_r2: dict[str, dict[int, float]] = {}
    for csv_path in glob.glob(str(nn_dir / "seed_*" / "neural_ode_diffrax_metrics.csv")):
        seed = int(re.search(r"seed_(\d+)", csv_path).group(1))
        df = pd.read_csv(csv_path)
        for _, row in df[df.split == "test"].iterrows():
            nn_r2.setdefault(str(row.marker), {})[seed] = float(row.trajectory_r2_mean)

    common = set(nn_r2) - set(args.exclude_marker)
    nn_jac: dict[tuple[str, int], np.ndarray] = {}
    for marker in common:
        for seed in seeds:
            loaded = _load_marker_checkpoint(nn_dir / f"seed_{seed}" / "models", _safe(marker))
            if loaded is None:
                continue
            meta, model, train = loaded
            nn_jac[(marker, seed)] = _per_marker_causal(meta, model, train)["mean_signed_jac"]

    # PySR per-seed
    dataset = pd.read_csv(args.dataset)
    pysr_r2: dict[str, dict[int, float]] = {}
    pysr_jac: dict[tuple[str, int], np.ndarray] = {}
    for seed in seeds:
        formula_path = pysr_dir / f"seed_{seed}" / "formulas" / "all_per_minute.txt"
        if not formula_path.exists():
            continue
        for marker, (r2, formula) in _parse_pysr(formula_path).items():
            if marker not in common:
                continue
            pysr_r2.setdefault(marker, {})[seed] = r2
            jac = _pysr_jac(formula, marker, dataset)
            if jac is not None:
                pysr_jac[(marker, seed)] = jac

    nn_works = {m for m, vs in nn_r2.items() if max(vs.values()) > threshold}
    pysr_works = {m for m, vs in pysr_r2.items() if max(vs.values()) > threshold}
    both_works = nn_works & pysr_works

    def self_jaccards(jac_dict, marker_set, cutoff):
        per_marker = []
        for marker in marker_set:
            sets = [
                top_set(jac_dict[(marker, s)][KEEP_IDX], KEEP, cutoff)
                for s in seeds if (marker, s) in jac_dict
            ]
            if len(sets) < 2:
                continue
            pairs = [jaccard(a, b) for a, b in combinations(sets, 2)]
            pairs = [x for x in pairs if not np.isnan(x)]
            if pairs:
                per_marker.append(np.mean(pairs))
        return np.array(per_marker)

    def seed_stable_set(jac_dict, marker, cutoff):
        sets = [
            top_set(jac_dict[(marker, s)][KEEP_IDX], KEEP, cutoff)
            for s in seeds if (marker, s) in jac_dict
        ]
        if len(sets) == len(seeds):
            return set.intersection(*sets)
        return None

    def between_jaccards(cutoff):
        out = []
        for marker in both_works:
            a = seed_stable_set(nn_jac, marker, cutoff)
            b = seed_stable_set(pysr_jac, marker, cutoff)
            if a is None or b is None:
                continue
            if not a and not b:
                continue
            out.append(jaccard(a, b))
        return np.array([x for x in out if not np.isnan(x)])

    def mean_set_size(jac_dict, marker_set, cutoff):
        sizes = []
        for marker in marker_set:
            for seed in seeds:
                if (marker, seed) in jac_dict:
                    sizes.append(len(top_set(jac_dict[(marker, seed)][KEEP_IDX], KEEP, cutoff)))
        return float(np.mean(sizes)) if sizes else float("nan")

    nn_self, py_self, between = [], [], []
    nn_self_sem, py_self_sem, between_sem = [], [], []
    nn_size, py_size = [], []
    for cutoff in cutoffs:
        a = self_jaccards(nn_jac, nn_works, cutoff)
        nn_self.append(a.mean() * 100)
        nn_self_sem.append(a.std() / np.sqrt(max(len(a), 1)) * 100)
        b = self_jaccards(pysr_jac, pysr_works, cutoff)
        py_self.append(b.mean() * 100)
        py_self_sem.append(b.std() / np.sqrt(max(len(b), 1)) * 100)
        c = between_jaccards(cutoff)
        between.append(c.mean() * 100)
        between_sem.append(c.std() / np.sqrt(max(len(c), 1)) * 100)
        nn_size.append(mean_set_size(nn_jac, nn_works, cutoff))
        py_size.append(mean_set_size(pysr_jac, pysr_works, cutoff))
    nn_self = np.array(nn_self)
    py_self = np.array(py_self)
    between = np.array(between)
    nn_self_sem = np.array(nn_self_sem)
    py_self_sem = np.array(py_self_sem)
    between_sem = np.array(between_sem)

    mpl.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                     gridspec_kw={"width_ratios": [1.4, 1]})
    nn_color = "#2c5f9e"
    py_color = "#c8482e"
    bt_color = "#5e3a87"
    x = 100 * cutoffs

    ax_a.fill_between(x, nn_self - nn_self_sem, nn_self + nn_self_sem,
                      color=nn_color, alpha=0.18)
    ax_a.plot(x, nn_self, "-o", color=nn_color, lw=2, ms=7,
              label=f"L21 seed-robustness (n={len(nn_works)})")
    ax_a.fill_between(x, py_self - py_self_sem, py_self + py_self_sem,
                      color=py_color, alpha=0.18)
    ax_a.plot(x, py_self, "-s", color=py_color, lw=2, ms=7,
              label=f"PySR seed-robustness (n={len(pysr_works)})")
    ax_a.fill_between(x, between - between_sem, between + between_sem,
                      color=bt_color, alpha=0.18)
    ax_a.plot(x, between, "-^", color=bt_color, lw=2, ms=7,
              label=f"L21 ↔ PySR agreement (n={len(both_works)})")
    ax_a.axvline(80, color="#888", lw=0.8, ls=":")
    ax_a.text(80.5, 4, "80%", fontsize=9, color="#666")
    ax_a.set_xlabel("Top-X% cumulative |J| mass cutoff", fontsize=12)
    ax_a.set_ylabel("Driver-set overlap  (% of union shared)", fontsize=12)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlim(48, 102)
    ax_a.set_yticks(range(0, 101, 20))
    ax_a.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax_a.set_title(
        f"A.  Seed robustness and between-method agreement (R²>{threshold})",
        loc="left", fontsize=11.5,
    )
    ax_a.legend(loc="upper left", frameon=False, fontsize=10)

    ax_b.plot(x, nn_size, "-o", color=nn_color, lw=2, ms=7, label="L21")
    ax_b.plot(x, py_size, "-s", color=py_color, lw=2, ms=7, label="PySR")
    ax_b.axvline(80, color="#888", lw=0.8, ls=":")
    ax_b.text(80.5, 8.5, "80%", fontsize=9, color="#666", va="top")
    ax_b.set_xlabel("Top-X% cumulative |J| mass cutoff", fontsize=12)
    ax_b.set_ylabel("Mean drivers in set", fontsize=12)
    ax_b.set_xlim(48, 102)
    ax_b.set_ylim(0, 9)
    ax_b.set_title("B.  Number of drivers needed to cover top-X% mass",
                   loc="left", fontsize=11.5)
    ax_b.legend(loc="upper left", frameon=False, fontsize=10)

    plt.tight_layout()
    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_png} and {out_png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
