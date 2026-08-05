"""
Causal driver + interaction analysis for the diffrax Neural ODE baseline.

Reads the per-(seed, marker) model checkpoints emitted by
`neural_ode_diffrax_baseline.py --save-models`. For each (seed, marker):
  - Evaluate the RHS network's Jacobian d(dy/dt)/d(x) at every observed
    training input (B × T × {real timepoints}); axes are
    [p-ERK1-2, *exo_cols] in scaled space.
  - Mean signed Jacobian per feature   → driver (+) vs brake (-).
  - Mean |Jacobian| per feature        → absolute driving strength.
  - Participation ratio of |Jacobian|  → effective number of drivers,
        PR = (Σ|J_i|)² / Σ|J_i|², in [1, F+1].
  - Mean Hessian d²/dx_i dx_j (averaged over training inputs)
        → off-diagonals expose gating / cooperativity structure.

Per-marker plots aggregate across seeds by simple mean (matching the
sweep's other aggregations). Per-seed PR values are preserved in a CSV
so the spread across seeds is visible in the box-strip plot.

Outputs (under --output-dir):
  - jacobian_bar_<marker>.png         signed |J| bar plot per marker
  - hessian_heatmap_<marker>.png      F+1 x F+1 mean Hessian per marker
  - participation_ratio.png           PR boxplot across markers
  - participation_ratio_per_seed.csv  raw (marker, seed, PR)
  - causal_summary.csv                long form: (marker, feature, J̄, |J̄|, n_seeds, PR mean/std)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.experimental.sr_pipeline.neural_ode_diffrax_baseline import RHS


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _load_marker_checkpoint(model_dir: Path, safe: str):
    """Return (meta, model, train_npz_dict). Returns None if files missing."""
    meta_path = model_dir / f"{safe}.json"
    model_path = model_dir / f"{safe}.eqx"
    train_path = model_dir / f"{safe}_train.npz"
    if not (meta_path.exists() and model_path.exists() and train_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    template = RHS(
        in_dim=int(meta["in_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        hidden_layers=int(meta["hidden_layers"]),
        activation=str(meta["activation"]),
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(str(model_path), template)
    train = dict(np.load(train_path))
    return meta, model, train


# -----------------------------------------------------------------------------
# Per-marker causal summary
# -----------------------------------------------------------------------------

def _per_marker_causal(meta: dict, model, train: dict) -> dict:
    """Mean signed/abs Jacobian, mean Hessian, and PR over training inputs."""
    y_mean = float(meta["y_mean"])
    y_scale = float(meta["y_scale"])
    ys = train["ys"]                    # (B, T)  unscaled state
    us_scaled = train["us_scaled"]      # (B, T, F)
    real_mask = train["real_mask"].astype(bool)  # (B, T)

    y_scaled = (ys - y_mean) / y_scale
    F = us_scaled.shape[-1]

    y_flat = y_scaled.reshape(-1)[real_mask.reshape(-1)]
    u_flat = us_scaled.reshape(-1, F)[real_mask.reshape(-1)]
    inputs = np.concatenate([y_flat[:, None], u_flat], axis=1)  # (N, F+1)
    if inputs.size == 0:
        # Degenerate; shouldn't happen on real markers
        return None
    inputs_j = jnp.asarray(inputs)

    # Jacobian: d f / d x evaluated at each input
    jac_fn = jax.jit(jax.vmap(jax.grad(model)))
    jacs = np.asarray(jac_fn(inputs_j))    # (N, F+1)

    # Hessian: d² f / dx_i dx_j  (symmetric)
    hess_fn = jax.jit(jax.vmap(jax.hessian(model)))
    hess = np.asarray(hess_fn(inputs_j))   # (N, F+1, F+1)

    mean_signed = jacs.mean(axis=0)
    mean_abs = np.abs(jacs).mean(axis=0)
    mean_hess = hess.mean(axis=0)

    sum_abs = float(mean_abs.sum())
    sum_sq = float((mean_abs ** 2).sum())
    pr = (sum_abs ** 2) / max(sum_sq, 1e-12)

    feature_names = ["p-ERK1-2"] + list(meta["exo_cols"])
    return {
        "feature_names": feature_names,
        "mean_signed_jac": mean_signed,
        "mean_abs_jac": mean_abs,
        "mean_hess": mean_hess,
        "pr": float(pr),
        "n_inputs": int(inputs.shape[0]),
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")


def _plot_jacobian_bar(feature_names: Sequence[str], signed: np.ndarray,
                       absval: np.ndarray, marker: str, output: Path) -> None:
    colors = ["#1f77b4" if s >= 0 else "#d62728" for s in signed]  # blue=driver, red=brake
    plt.figure(figsize=(max(6.0, 0.6 * len(feature_names)), 4.0))
    plt.bar(range(len(feature_names)), absval, color=colors,
            edgecolor="black", linewidth=0.5)
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.ylabel(r"$|\partial(\dot{p\text{-}ERK})/\partial x|$")
    plt.title(f"{marker} — driver (blue) / brake (red) strength")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_hessian_heatmap(feature_names: Sequence[str], H: np.ndarray,
                          marker: str, output: Path) -> None:
    vmax = float(np.max(np.abs(H)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-12
    plt.figure(figsize=(7, 6))
    sns.heatmap(H, xticklabels=feature_names, yticklabels=feature_names,
                cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
                square=True, cbar_kws={"label": r"$\overline{\partial^2 \dot{y}/\partial x_i \partial x_j}$"})
    plt.title(f"{marker} — mean Hessian (off-diagonal = gating)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    plt.close()


def _plot_pr_distribution(pr_df: pd.DataFrame, output: Path) -> None:
    if pr_df.empty:
        return
    order = pr_df.groupby("marker")["pr"].mean().sort_values().index.tolist()
    plt.figure(figsize=(max(8.0, 0.35 * len(order) + 4.0), 5.0))
    sns.boxplot(data=pr_df, x="marker", y="pr", order=order,
                color="lightsteelblue", showfliers=False)
    sns.stripplot(data=pr_df, x="marker", y="pr", order=order,
                  color="black", size=3.5, alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Participation ratio (effective # drivers)")
    plt.xlabel("")
    plt.title("Effective number of drivers per marker (across seeds)")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Parent dir holding seed_{42,43,44}/models/ checkpoints.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=(42, 43, 44))
    ap.add_argument("--marker-filter", nargs="*", default=None,
                    help="Optional subset of markers (canonical names).")
    ap.add_argument("--consensus-only", action="store_true",
                    help="Mask Hessian entries (and signed Jacobian entries) "
                         "where sign disagrees across seeds. Only robustly-"
                         "agreed interactions survive — strictly stronger than "
                         "plain seed averaging when n_seeds >= 2.")
    args = ap.parse_args()

    seed_dirs = {s: args.source_dir / f"seed_{s}" / "models" for s in args.seeds}
    missing = [s for s, d in seed_dirs.items() if not d.exists()]
    if missing:
        print(
            f"[warn] no models dir for seeds {missing}. "
            f"Re-run the diffrax baseline with --save-models to populate them.",
            file=sys.stderr,
        )

    # Discover (marker, seed) checkpoints.
    per_marker: Dict[str, List[Tuple[int, dict]]] = {}
    for seed, mdir in seed_dirs.items():
        if not mdir.exists():
            continue
        for json_path in sorted(mdir.glob("*.json")):
            safe = json_path.stem
            loaded = _load_marker_checkpoint(mdir, safe)
            if loaded is None:
                continue
            meta, model, train = loaded
            marker = str(meta["marker"])
            if args.marker_filter and marker not in args.marker_filter:
                continue
            r = _per_marker_causal(meta, model, train)
            if r is None:
                continue
            per_marker.setdefault(marker, []).append((seed, r))
            print(
                f"[done] seed={seed} marker={marker} "
                f"n_inputs={r['n_inputs']} pr={r['pr']:.2f}",
                flush=True,
            )

    if not per_marker:
        raise SystemExit(
            "No checkpoints found. Train with `neural_ode_diffrax_baseline.py "
            "--save-models` first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across seeds + plot per marker.
    summary_rows: List[dict] = []
    pr_rows: List[dict] = []
    for marker, entries in sorted(per_marker.items()):
        # Ensure feature ordering is consistent across seeds (it should be, since
        # the splits are deterministic per seed but feature_cols come from data).
        feat_names = entries[0][1]["feature_names"]
        F1 = len(feat_names)
        signed_stack = np.stack([r["mean_signed_jac"] for _, r in entries], axis=0)
        abs_stack = np.stack([r["mean_abs_jac"] for _, r in entries], axis=0)
        hess_stack = np.stack([r["mean_hess"] for _, r in entries], axis=0)
        prs = [r["pr"] for _, r in entries]

        signed_mean = signed_stack.mean(axis=0)
        abs_mean = abs_stack.mean(axis=0)
        hess_mean = hess_stack.mean(axis=0)
        pr_mean = float(np.mean(prs))
        pr_std = float(np.std(prs))

        if args.consensus_only and signed_stack.shape[0] >= 2:
            # Mask any element whose sign disagrees across seeds. The mean is
            # zeroed out at those positions; the magnitude survives only at
            # positions where every seed independently agreed on the sign.
            sgn_jac = np.sign(signed_stack)
            jac_agree = np.all(sgn_jac == sgn_jac[0:1], axis=0)
            signed_mean = np.where(jac_agree, signed_mean, 0.0)
            abs_mean = np.where(jac_agree, abs_mean, 0.0)

            sgn_h = np.sign(hess_stack)
            hess_agree = np.all(sgn_h == sgn_h[0:1], axis=0)
            hess_mean = np.where(hess_agree, hess_mean, 0.0)

        # Per-marker plots
        safe = _safe(marker)
        suffix = "_consensus" if args.consensus_only else ""
        _plot_jacobian_bar(feat_names, signed_mean, abs_mean, marker,
                           args.output_dir / f"jacobian_bar_{safe}{suffix}.png")
        _plot_hessian_heatmap(feat_names, hess_mean, marker,
                              args.output_dir / f"hessian_heatmap_{safe}{suffix}.png")

        for i, name in enumerate(feat_names):
            summary_rows.append({
                "marker": marker,
                "feature": name,
                "mean_signed_jac": float(signed_mean[i]),
                "mean_abs_jac": float(abs_mean[i]),
                "n_seeds": len(entries),
                "pr_mean": pr_mean,
                "pr_std": pr_std,
            })
        for seed, _ in entries:
            pr_rows.append({"marker": marker, "seed": int(seed),
                            "pr": float(prs[[s for s, _ in entries].index(seed)])})

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "causal_summary.csv", index=False)
    pr_df = pd.DataFrame(pr_rows)
    pr_df.to_csv(args.output_dir / "participation_ratio_per_seed.csv", index=False)
    _plot_pr_distribution(pr_df, args.output_dir / "participation_ratio.png")

    print(f"\nWrote {len(per_marker)} markers x {len(args.seeds)} seeds to {args.output_dir}")


if __name__ == "__main__":
    main()
