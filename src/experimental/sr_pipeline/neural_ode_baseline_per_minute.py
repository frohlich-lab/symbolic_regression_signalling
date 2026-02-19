"""Neural ODE-style baseline on per-minute data (full feature set, no K sweep).

Trains a small MLP per marker on the per-minute dataset (using the same
GFP-bin train/test split and per-minute sampling as the SR pipeline), then
evaluates:
- dt R² on train/test (per-bin)
- feature-driven integrated R² (Euler on predicted dt at observed features)
- ODE-style integrated R² where pERK is treated as the state and other
  features are exogenous (Euler with the network re-evaluated at the current
  state).

Outputs:
- metrics CSV (per marker/seed/split)
- aggregated metrics CSV (mean across seeds)
- boxplots for dt / integrated / ODE R²
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from matplotlib.backends.backend_pdf import PdfPages
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimental.sr_pipeline.metrics import binwise_r2, coefficient_of_determination
from experimental.sr_pipeline.compute_marker_integration import (
    prepare_sr_dataset,
    integrate_single_marker,
)
from experimental.sr_pipeline.run_functional_groups import (
    EXCLUDE_COLUMNS,
    REL_MAE_EPS,
    apply_per_minute_sampling,
    choose_bin_split,
    sanitize_feature_names,
)
from experimental.sr_pipeline.seeding import canonicalize_seeds, seed_all

MEASURED_TIMEPOINTS = (0.0, 5.0, 10.0, 15.0, 30.0, 60.0)
K_THRESHOLD = 2  # minimum k when matching PySR complexity


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural ODE baseline on per-minute data (full feature set).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experimental/processed/functional_groups/functional_groups_per_minute_fit.csv"),
        help="Per-minute CSV used for training/evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write plots and CSVs.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction for GFP-bin split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits/sampling.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=(42, 43, 44),
        help="Seeds to run; results are aggregated across seeds for plots.",
    )
    parser.add_argument(
        "--per-minute-max-time",
        type=float,
        default=60.0,
        help="Keep rows with timepoint ≤ this value before sampling.",
    )
    parser.add_argument(
        "--per-minute-sampling-strategy",
        choices=("max_time", "early_plus_sparse_late"),
        default="early_plus_sparse_late",
        help="Downsampling strategy mirroring the SR pipeline.",
    )
    parser.add_argument(
        "--late-sample-window",
        nargs=2,
        type=float,
        default=(30.0, 60.0),
        metavar=("START", "END"),
        help="Window used by early_plus_sparse_late.",
    )
    parser.add_argument(
        "--late-sample-points",
        type=int,
        default=15,
        help="Late-window samples per marker/bin for early_plus_sparse_late.",
    )
    parser.add_argument(
        "--measured-timepoints",
        nargs="*",
        type=float,
        default=MEASURED_TIMEPOINTS,
        help="Measured timepoints to retain for per-minute evaluation.",
    )
    # Training hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width.")
    parser.add_argument("--hidden-layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (applied after each hidden layer).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2).")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum improvement to reset patience.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction (in GFP-bin space).")
    # Sweep options
    parser.add_argument(
        "--sweep-output",
        type=Path,
        default=None,
        help="Optional CSV to write a small hyperparameter sweep (skips when not provided).",
    )
    parser.add_argument(
        "--sweep-marker-limit",
        type=int,
        default=10,
        help="Max markers to include in the sweep subset (sorted).",
    )
    parser.add_argument(
        "--sweep-seed-count",
        type=int,
        default=1,
        help="How many seeds from the provided list to use in the sweep (start of list).",
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run only the sweep (no full run with the default hyperparameters).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional label to annotate outputs/metrics (e.g., run identifier).",
    )
    parser.add_argument(
        "--pysr-metrics",
        type=Path,
        default=None,
        help="Optional marker_integration_metrics_per_minute CSV to match PySR effective k per marker.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Model + training
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    train_ds = torch.utils.data.TensorDataset(_to_tensor(X_train, device), _to_tensor(y_train, device))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    no_improve = 0
    train_history: List[float] = []
    val_history: List[float] = []

    X_val_t = _to_tensor(X_val, device)
    y_val_t = _to_tensor(y_val, device)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())

        train_history.append(train_loss)
        val_history.append(val_loss)

        if not np.isfinite(val_loss):
            break

        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    return model, {"train": train_history, "val": val_history}


# -----------------------------------------------------------------------------
# Integration helpers (feature-driven + simple ODE-style Euler)
# -----------------------------------------------------------------------------

def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns are present: {candidates}")


def _predict_dt(
    model: nn.Module,
    scaler: StandardScaler,
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=float)
    X = frame[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_scaled = scaler.transform(X.to_numpy(dtype=float))
    with torch.no_grad():
        preds = model(_to_tensor(X_scaled, device)).cpu().numpy().reshape(-1)
    preds[~np.isfinite(preds)] = np.nan
    return preds


def _coerce_features(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Numeric cast with NaN fill for a column subset."""
    if frame.empty:
        return pd.DataFrame(columns=list(cols))
    return frame[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _split_bins_three(
    bins: Sequence[float],
    test_size: float,
    val_size: float,
    rng: np.random.Generator,
) -> Tuple[Optional[set], Optional[set], Optional[set]]:
    unique_bins = list(sorted(set([b for b in bins if pd.notna(b)])))
    if len(unique_bins) < 3:
        return None, None, None
    shuffled = unique_bins.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    if n_test + n_val >= n:
        n_test = max(1, min(n - 2, n_test))
        n_val = max(1, min(n - 1 - n_test, n_val))
    if n_test + n_val >= n:
        return None, None, None
    test_bins = set(shuffled[:n_test])
    val_bins = set(shuffled[n_test : n_test + n_val])
    train_bins = set(shuffled[n_test + n_val :])
    if not (train_bins and val_bins and test_bins):
        return None, None, None
    return train_bins, val_bins, test_bins


def _pysr_effective_k(formula: str, max_k: int) -> int:
    try:
        expr = sp.sympify(formula)
        syms = {s for s in expr.free_symbols if not str(s).startswith("__")}
        k = max(1, len(syms))
    except Exception:
        k = 1
    k = int(round(k))
    return min(max_k, max(1, k))


def _load_pysr_k_map(path: Optional[Path], max_k: int) -> Dict[str, int]:
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
        df = df[(df.get("model") == "PySR") & (df.get("dataset_mode", "per_minute") == "per_minute")].copy()
        if df.empty:
            return {}
        df["k_pysr"] = df["formula"].apply(lambda f: _pysr_effective_k(f, max_k))
        return {str(r["marker"]): int(r["k_pysr"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def _default_sweep_grid(random_state: int = 42, n_samples: int = 60) -> List[Dict[str, float]]:
    """
    Latin-hypercube style sample over a small hyperparameter box for a quick sweep.
    Ranges (inclusive-ish):
      - hidden_dim: 32–192
      - hidden_layers: 1–4 (integer)
      - dropout: 0.0–0.4
      - lr: 2e-4–2e-3 (log space)
      - weight_decay: 1e-5–1e-3 (log space)
    """
    rng = np.random.default_rng(random_state)

    def _lhs_strata(n: int) -> np.ndarray:
        # Stratified samples in [0,1)
        base = (np.arange(n, dtype=float) + rng.random(n)) / n
        rng.shuffle(base)
        return base

    # Generate stratified samples per dimension
    u_layers = _lhs_strata(n_samples)
    u_dim = _lhs_strata(n_samples)
    u_drop = _lhs_strata(n_samples)
    u_lr = _lhs_strata(n_samples)
    u_wd = _lhs_strata(n_samples)

    def _scale_linear(u: float, lo: float, hi: float) -> float:
        return lo + u * (hi - lo)

    def _scale_log(u: float, lo: float, hi: float) -> float:
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))

    configs: List[Dict[str, float]] = []
    for i in range(n_samples):
        hidden_layers = int(np.clip(np.floor(1 + u_layers[i] * 4), 1, 4))
        hidden_dim = int(round(_scale_linear(u_dim[i], 32, 192)))
        dropout = float(np.clip(_scale_linear(u_drop[i], 0.0, 0.4), 0.0, 0.4))
        lr = _scale_log(u_lr[i], 2e-4, 2e-3)
        weight_decay = _scale_log(u_wd[i], 1e-5, 1e-3)
        configs.append(
            {
                "hidden_dim": hidden_dim,
                "hidden_layers": hidden_layers,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return configs


def integrate_marker_ode_nn(
    sub: pd.DataFrame,
    target_col: str,
    measured_timepoints: Sequence[float],
    dataset_mode: str,
    model: nn.Module,
    scaler: StandardScaler,
    feature_cols: Sequence[str],
    perk_name: str,
    device: torch.device,
    *,
    restrict_to_measured: bool = False,
) -> Dict[str, float]:
    bins = sorted(sub["GFP_bin"].dropna().unique().tolist())
    if not bins:
        return {
            "ode_integ_r2_median": 0.0,
            "ode_integ_rel_mae_mean_bins": np.nan,
            "ode_integ_rel_mae_median_bins": np.nan,
            "ode_integ_rel_mae_valid_bins": 0,
            "ode_bins": 0,
            "ode_integrated_bins": 0,
        }

    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.where(np.asarray(scaler.scale_, dtype=float) == 0.0, 1.0, scaler.scale_)
    feature_index = {name: i for i, name in enumerate(feature_cols)}

    obs_pe_candidates = ("p_ERK1_2", "p-ERK1-2")

    r2_vals: List[float] = []
    rel_mae_vals: List[float] = []
    integrated_bins = 0

    model = model.to(device)
    model.eval()

    for b in bins:
        g = sub[sub["GFP_bin"] == b].copy()
        g.sort_values("timepoint", inplace=True)

        t = pd.to_numeric(g["timepoint"], errors="coerce").to_numpy(dtype=float)
        try:
            obs_pe_col = _pick_column(g, obs_pe_candidates)
            obs_pe = pd.to_numeric(g[obs_pe_col], errors="coerce").to_numpy(dtype=float)
        except KeyError:
            obs_pe = np.full_like(t, np.nan, dtype=float)

        mask = np.isfinite(t) & np.isfinite(obs_pe)
        if mask.sum() < 2:
            continue

        t = t[mask]
        obs_pe = obs_pe[mask]
        order = np.argsort(t)
        t = t[order]
        obs_pe = obs_pe[order]

        state = obs_pe[0] if np.isfinite(obs_pe[0]) else float(np.nanmean(obs_pe))
        if not np.isfinite(state):
            state = 0.0

        integ = np.full_like(t, np.nan, dtype=float)
        integ[0] = state

        for j in range(len(t) - 1):
            dt = t[j + 1] - t[j]
            row = g.iloc[mask.nonzero()[0][order[j]]]
            feats = []
            for name in feature_cols:
                if name == perk_name:
                    val = state
                else:
                    val = pd.to_numeric(row.get(name, 0.0), errors="coerce")
                feats.append(0.0 if not np.isfinite(val) else float(val))
            feats = np.asarray(feats, dtype=float)
            feats_scaled = (feats - mean) / scale
            with torch.no_grad():
                deriv = model(_to_tensor(feats_scaled[None, :], device)).item()
            if not np.isfinite(deriv):
                continue
            state = state + dt * float(deriv)
            integ[j + 1] = state

        use_mask = dataset_mode == "per_minute" and restrict_to_measured
        meas_mask = np.isin(t, measured_timepoints) if use_mask else np.ones_like(t, dtype=bool)

        if (
            meas_mask.sum() > 1
            and np.isfinite(integ[meas_mask]).sum() > 1
            and np.isfinite(obs_pe[meas_mask]).sum() > 1
        ):
            r2 = coefficient_of_determination(obs_pe[meas_mask], integ[meas_mask])
            if np.isfinite(r2):
                r2_vals.append(max(0.0, float(r2)))
                integrated_bins += 1

        rel_mask = meas_mask & np.isfinite(integ) & np.isfinite(obs_pe)
        if np.any(rel_mask):
            denom = np.maximum(np.abs(obs_pe[rel_mask]), REL_MAE_EPS)
            rel_mae = float(np.mean(np.abs(integ[rel_mask] - obs_pe[rel_mask]) / denom))
            if np.isfinite(rel_mae):
                rel_mae_vals.append(rel_mae)

    ode_r2_median = float(np.median(r2_vals)) if r2_vals else 0.0
    ode_rel_mae_mean_bins = float(np.mean(rel_mae_vals)) if rel_mae_vals else np.nan
    ode_rel_mae_median_bins = float(np.median(rel_mae_vals)) if rel_mae_vals else np.nan
    return {
        "ode_integ_r2_median": ode_r2_median,
        "ode_integ_rel_mae_mean_bins": ode_rel_mae_mean_bins,
        "ode_integ_rel_mae_median_bins": ode_rel_mae_median_bins,
        "ode_integ_rel_mae_valid_bins": len(rel_mae_vals),
        "ode_bins": len(bins),
        "ode_integrated_bins": integrated_bins,
    }


def _plot_boxplot(df: pd.DataFrame, metric_col: str, ylabel: str, output: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.xlabel("Split")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    sns = __import__("seaborn")
    sns.boxplot(data=df, x="split", y=metric_col, palette="Set2")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def _format_hparam_label(hparams: Dict[str, float]) -> str:
    fields = ["hidden_dim", "hidden_layers", "dropout", "lr", "weight_decay"]
    bits = []
    for key in fields:
        val = hparams.get(key)
        if val is not None:
            bits.append(f"{key}={val}")
    return ", ".join(bits)


def _plot_true_vs_pred(ax, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    if len(y_true) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.any():
        y_true_f = y_true[finite]
        y_pred_f = y_pred[finite]
        lo = min(np.min(y_true_f), np.min(y_pred_f))
        hi = max(np.max(y_true_f), np.max(y_pred_f))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("True dt")
    ax.set_ylabel("Predicted dt")
    ax.set_title(title)


def _plot_loss_curve(ax, train_loss: Sequence[float], val_loss: Sequence[float]) -> None:
    if not train_loss and not val_loss:
        ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
        ax.set_title("Training loss")
        return
    epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
    if train_loss:
        ax.plot(epochs[: len(train_loss)], train_loss, label="train")
    if val_loss:
        ax.plot(epochs[: len(val_loss)], val_loss, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training loss")
    ax.legend()


def _save_seed_diagnostics(
    seed: int,
    entries: Sequence[Dict[str, object]],
    outdir: Path,
) -> None:
    if not entries:
        return
    diag_dir = outdir / "training_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = diag_dir / f"seed_{seed}_training_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        for entry in entries:
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            _plot_loss_curve(axes[0], entry.get("train_loss", []), entry.get("val_loss", []))
            _plot_true_vs_pred(axes[1], entry["train_true"], entry["train_pred"], "Train dt")
            _plot_true_vs_pred(axes[2], entry.get("val_true", np.array([])), entry.get("val_pred", np.array([])), "Val dt")
            _plot_true_vs_pred(axes[3], entry["test_true"], entry["test_pred"], "Test dt")
            fig.suptitle(
                f"{entry['marker']} | seed={seed} | k={entry['k']} | {entry['hparams']}"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[NeuralODE] saved training diagnostics to {pdf_path}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 0.9:
        raise ValueError("val_size and test_size must be positive and leave room for train (val+test < 0.9 recommended).")
    seeds = canonicalize_seeds(args.random_state, args.seeds)

    raw = pd.read_csv(args.dataset)
    full_data, target_col = prepare_sr_dataset(raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    marker_pool = sorted(full_data["marker"].dropna().unique().tolist())
    pysr_k_map = _load_pysr_k_map(args.pysr_metrics, max_k=100)

    def _run_setting(
        hparams: Dict[str, float],
        marker_filter: Optional[Sequence[str]],
        seeds_override: Optional[Sequence[int]],
        *,
        save_diagnostics: bool = False,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        seeds_local = list(seeds_override) if seeds_override is not None else seeds
        for seed in seeds_local:
            seed_all(seed)
            seed_diag_entries: List[Dict[str, object]] = []
            sampled = apply_per_minute_sampling(
                raw,
                strategy=args.per_minute_sampling_strategy,
                max_time=args.per_minute_max_time,
                late_window=tuple(args.late_sample_window),
                late_points=args.late_sample_points,
                random_state=seed,
            )
            data_local, target_col_local = prepare_sr_dataset(sampled)
            data_local = data_local.dropna(subset=[target_col_local])
            feature_cols = [c for c in data_local.columns if c not in EXCLUDE_COLUMNS and c != target_col_local]
            if len(feature_cols) < 2:
                print(f"[NeuralODE] seed={seed} skipped: <2 features after exclude ({len(feature_cols)})")
                continue
            markers_local = sorted(data_local["marker"].dropna().unique().tolist())
            if marker_filter:
                markers_local = [m for m in markers_local if m in marker_filter]

            for marker in markers_local:
                subset = data_local[data_local["marker"] == marker].copy()
                miss_feat_rows = subset[feature_cols].isna().any(axis=1).sum()
                miss_target_rows = subset[target_col_local].isna().sum()
                if miss_feat_rows or miss_target_rows:
                    print(
                        f"[NeuralODE] seed={seed} marker={marker} missing rows "
                        f"(features {miss_feat_rows}, target {miss_target_rows})",
                        flush=True,
                    )
                total_rows = len(subset)
                target_valid = subset[target_col_local].notna().sum()
                if target_valid == 0:
                    print(f"[NeuralODE] seed={seed} marker={marker} skipped: 0 valid targets out of {total_rows} rows", flush=True)
                    continue
                subset = subset.dropna(subset=[target_col_local])
                if subset.empty:
                    print(f"[NeuralODE] seed={seed} marker={marker} skipped: all rows dropped due to NaN target")
                    continue

                rng_split = np.random.default_rng(seed)
                train_bins, val_bins, test_bins = _split_bins_three(
                    subset["GFP_bin"].dropna().unique().tolist(),
                    test_size=args.test_size,
                    val_size=args.val_size,
                    rng=rng_split,
                )
                if not train_bins or not val_bins or not test_bins:
                    print(
                        f"[NeuralODE] seed={seed} marker={marker} skipped: could not split bins (train={train_bins} val={val_bins} test={test_bins})",
                        flush=True,
                    )
                    continue

                subset = subset[subset["GFP_bin"].isin(train_bins | val_bins | test_bins)].copy()
                train_df = subset[subset["GFP_bin"].isin(train_bins)].copy()
                val_df = subset[subset["GFP_bin"].isin(val_bins)].copy()
                test_df = subset[subset["GFP_bin"].isin(test_bins)].copy()
                if train_df.empty or val_df.empty or test_df.empty:
                    print(
                        f"[NeuralODE] seed={seed} marker={marker} skipped: empty train/val/test after split (train={len(train_df)} val={len(val_df)} test={len(test_df)})",
                        flush=True,
                    )
                    continue

                print(
                    f"[NeuralODE] seed={seed} marker={marker} train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)} h={hparams}",
                    flush=True,
                )

                X_train_num = _coerce_features(train_df, feature_cols)
                X_val_num = _coerce_features(val_df, feature_cols)
                X_test_num = _coerce_features(test_df, feature_cols)
                y_train = train_df[target_col_local].astype(float).to_numpy()
                y_val = val_df[target_col_local].astype(float).to_numpy()
                y_test = test_df[target_col_local].astype(float).to_numpy()

                # Optional K matching to PySR complexity
                selected_features = list(feature_cols)
                perk_name = sanitize_feature_names(["p_ERK1_2"])[0]
                if marker in pysr_k_map:
                    target_k = max(K_THRESHOLD, min(len(feature_cols), pysr_k_map[marker]))
                    if target_k < len(feature_cols):
                        # Always include pERK; select remaining k-1 by F-score
                        k_for_selector = max(1, target_k - 1) if perk_name in feature_cols else target_k
                        selector = SelectKBest(score_func=f_regression, k=k_for_selector)
                        selector.fit(X_train_num, y_train)
                        mask = selector.get_support()
                        selected = [f for f, keep in zip(feature_cols, mask) if keep]
                        selected_features = []
                        if perk_name in feature_cols:
                            selected_features.append(perk_name)
                        selected_features.extend([f for f in selected if f != perk_name])
                        if len(selected_features) > target_k:
                            selected_features = selected_features[:target_k]
                        elif len(selected_features) < target_k:
                            remaining = [f for f in feature_cols if f not in selected_features]
                            selected_features.extend(remaining[: max(0, target_k - len(selected_features))])
                        X_train_num = X_train_num[selected_features]
                        X_val_num = X_val_num[selected_features]
                        X_test_num = X_test_num[selected_features]
                        print(
                            f"[NeuralODE] marker={marker} matched k={target_k} (features {len(selected_features)}/{len(feature_cols)})",
                            flush=True,
                        )

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train_num.to_numpy(dtype=float))
                Xval = (
                    scaler.transform(X_val_num.to_numpy(dtype=float))
                    if len(X_val_num) > 0
                    else np.zeros((0, Xtr.shape[1]))
                )
                Xte = (
                    scaler.transform(X_test_num.to_numpy(dtype=float))
                    if len(X_test_num) > 0
                    else np.zeros((0, Xtr.shape[1]))
                )
                ytr = np.asarray(y_train, dtype=float)
                yva = np.asarray(y_val, dtype=float)
                yte = np.asarray(y_test, dtype=float)

                model = MLP(
                    in_dim=Xtr.shape[1],
                    hidden_dim=int(hparams.get("hidden_dim", args.hidden_dim)),
                    hidden_layers=int(hparams.get("hidden_layers", args.hidden_layers)),
                    dropout=float(hparams.get("dropout", args.dropout)),
                )
                model, loss_history = train_model(
                    model,
                    Xtr,
                    ytr,
                    Xval if len(Xval) > 0 else Xtr,
                    yva if len(yva) > 0 else ytr,
                    lr=float(hparams.get("lr", args.lr)),
                    weight_decay=float(hparams.get("weight_decay", args.weight_decay)),
                    epochs=int(hparams.get("epochs", args.epochs)),
                    batch_size=int(hparams.get("batch_size", args.batch_size)),
                    patience=int(hparams.get("patience", args.patience)),
                    min_delta=float(hparams.get("min_delta", args.min_delta)),
                    device=device,
                )

                infer_device = torch.device("cpu")
                model = model.to(infer_device)

                preds_train = _predict_dt(model, scaler, train_df, selected_features, infer_device)
                preds_val = _predict_dt(model, scaler, val_df, selected_features, infer_device)
                preds_test = _predict_dt(model, scaler, test_df, selected_features, infer_device)

                def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    y_true = np.asarray(y_true, dtype=float)
                    y_pred = np.asarray(y_pred, dtype=float)
                    mask = np.isfinite(y_true) & np.isfinite(y_pred)
                    if mask.sum() == 0:
                        return float("nan")
                    diff = y_true[mask] - y_pred[mask]
                    return float(np.mean(diff * diff))

                dt_mse_train = _mse(ytr, preds_train)
                dt_mse_val = _mse(yva, preds_val)
                dt_mse_test = _mse(yte, preds_test)

                train_with_pred = train_df.copy()
                train_with_pred["__y_pred__"] = preds_train
                val_with_pred = val_df.copy()
                val_with_pred["__y_pred__"] = preds_val
                test_with_pred = test_df.copy()
                test_with_pred["__y_pred__"] = preds_test

                dt_r2_train = binwise_r2(
                    train_with_pred,
                    target_col=target_col_local,
                    pred_col="__y_pred__",
                    measured_timepoints=args.measured_timepoints,
                    dataset_mode="per_minute",
                    clamp_negative=True,
                )
                dt_r2_val = binwise_r2(
                    val_with_pred,
                    target_col=target_col_local,
                    pred_col="__y_pred__",
                    measured_timepoints=args.measured_timepoints,
                    dataset_mode="per_minute",
                    clamp_negative=True,
                )
                dt_r2_test = binwise_r2(
                    test_with_pred,
                    target_col=target_col_local,
                    pred_col="__y_pred__",
                    measured_timepoints=args.measured_timepoints,
                    dataset_mode="per_minute",
                    clamp_negative=True,
                )

                # Integration on full-grid frames (use full dataset per marker)
                full_marker = full_data[full_data["marker"] == marker].copy()
                full_marker = full_marker.dropna(subset=[target_col_local, "GFP_bin", "timepoint"])
                full_train = full_marker[full_marker["GFP_bin"].isin(train_bins)].copy()
                full_val = full_marker[full_marker["GFP_bin"].isin(val_bins)].copy()
                full_test = full_marker[full_marker["GFP_bin"].isin(test_bins)].copy()

                def _add_preds(df_full: pd.DataFrame) -> pd.DataFrame:
                    if df_full.empty:
                        return df_full
                    df_full = df_full.copy()
                    df_full["__y_pred__"] = _predict_dt(model, scaler, df_full, selected_features, infer_device)
                    return df_full

                full_train_with_pred = _add_preds(full_train) if not full_train.empty else train_with_pred
                full_val_with_pred = _add_preds(full_val) if not full_val.empty else val_with_pred
                full_test_with_pred = _add_preds(full_test) if not full_test.empty else test_with_pred

                integ_train, _ = integrate_single_marker(
                    full_train_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                )
                integ_val, _ = integrate_single_marker(
                    full_val_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                )
                integ_test, _ = integrate_single_marker(
                    full_test_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                )

                ode_train = integrate_marker_ode_nn(
                    full_train_with_pred if not full_train_with_pred.empty else train_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                    model,
                    scaler,
                    selected_features,
                    perk_name,
                    infer_device,
                    restrict_to_measured=False,
                )
                ode_val = integrate_marker_ode_nn(
                    full_val_with_pred if not full_val_with_pred.empty else val_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                    model,
                    scaler,
                    selected_features,
                    perk_name,
                    infer_device,
                    restrict_to_measured=False,
                )
                ode_test = integrate_marker_ode_nn(
                    full_test_with_pred if not full_test_with_pred.empty else test_with_pred,
                    target_col_local,
                    args.measured_timepoints,
                    "per_minute",
                    model,
                    scaler,
                    selected_features,
                    perk_name,
                    infer_device,
                    restrict_to_measured=False,
                )

                k_full = len(selected_features)
                if save_diagnostics:
                    seed_diag_entries.append(
                        {
                            "marker": marker,
                            "train_true": ytr,
                            "train_pred": preds_train,
                            "val_true": yva,
                            "val_pred": preds_val,
                            "test_true": yte,
                            "test_pred": preds_test,
                            "k": k_full,
                            "hparams": _format_hparam_label(hparams),
                            "train_loss": loss_history.get("train", []),
                            "val_loss": loss_history.get("val", []),
                        }
                    )
                rows.extend(
                    [
                        {
                            "marker": marker,
                            "k": k_full,
                            "seed": seed,
                            "tag": args.tag,
                            "split": "train",
                            "dt_r2": dt_r2_train,
                            "dt_mse": dt_mse_train,
                            "dt_rel_mae_mean_bins": integ_train.get("dt_rel_mae_mean_bins"),
                            "dt_rel_mae_median_bins": integ_train.get("dt_rel_mae_median_bins"),
                            "integ_r2_median": integ_train.get("integ_r2_median"),
                            "integ_rel_mae_mean_bins": integ_train.get("integ_rel_mae_mean_bins"),
                            "integ_rel_mae_median_bins": integ_train.get("integ_rel_mae_median_bins"),
                            "ode_integ_r2_median": ode_train.get("ode_integ_r2_median"),
                            "ode_integ_rel_mae_mean_bins": ode_train.get("ode_integ_rel_mae_mean_bins"),
                            "ode_integ_rel_mae_median_bins": ode_train.get("ode_integ_rel_mae_median_bins"),
                        },
                        {
                            "marker": marker,
                            "k": k_full,
                            "seed": seed,
                            "tag": args.tag,
                            "split": "val",
                            "dt_r2": dt_r2_val,
                            "dt_mse": dt_mse_val,
                            "dt_rel_mae_mean_bins": integ_val.get("dt_rel_mae_mean_bins"),
                            "dt_rel_mae_median_bins": integ_val.get("dt_rel_mae_median_bins"),
                            "integ_r2_median": integ_val.get("integ_r2_median"),
                            "integ_rel_mae_mean_bins": integ_val.get("integ_rel_mae_mean_bins"),
                            "integ_rel_mae_median_bins": integ_val.get("integ_rel_mae_median_bins"),
                            "ode_integ_r2_median": ode_val.get("ode_integ_r2_median"),
                            "ode_integ_rel_mae_mean_bins": ode_val.get("ode_integ_rel_mae_mean_bins"),
                            "ode_integ_rel_mae_median_bins": ode_val.get("ode_integ_rel_mae_median_bins"),
                        },
                        {
                            "marker": marker,
                            "k": k_full,
                            "seed": seed,
                            "tag": args.tag,
                            "split": "test",
                            "dt_r2": dt_r2_test,
                            "dt_mse": dt_mse_test,
                            "dt_rel_mae_mean_bins": integ_test.get("dt_rel_mae_mean_bins"),
                            "dt_rel_mae_median_bins": integ_test.get("dt_rel_mae_median_bins"),
                            "integ_r2_median": integ_test.get("integ_r2_median"),
                            "integ_rel_mae_mean_bins": integ_test.get("integ_rel_mae_mean_bins"),
                            "integ_rel_mae_median_bins": integ_test.get("integ_rel_mae_median_bins"),
                            "ode_integ_r2_median": ode_test.get("ode_integ_r2_median"),
                            "ode_integ_rel_mae_mean_bins": ode_test.get("ode_integ_rel_mae_mean_bins"),
                            "ode_integ_rel_mae_median_bins": ode_test.get("ode_integ_rel_mae_median_bins"),
                        },
                    ]
                )
            if save_diagnostics:
                _save_seed_diagnostics(seed, seed_diag_entries, args.output_dir)
        return rows

    # Optional sweep on a small marker subset
    sweep_best: Optional[Dict[str, float]] = None
    if args.sweep_output is not None:
        sweep_out = Path(args.sweep_output)
    else:
        sweep_out = None

    if sweep_out is not None:
        sweep_grid = _default_sweep_grid(random_state=args.random_state, n_samples=60)
        if marker_pool:
            rng = np.random.default_rng(args.random_state)
            if len(marker_pool) > max(1, args.sweep_marker_limit):
                chosen = rng.choice(marker_pool, size=max(1, args.sweep_marker_limit), replace=False)
                sweep_markers = sorted(map(str, chosen))
            else:
                sweep_markers = sorted(marker_pool)
        else:
            sweep_markers = []
        sweep_seeds = seeds[: max(1, args.sweep_seed_count)]
        sweep_rows: List[Dict[str, object]] = []
        print(f"[Sweep] markers={sweep_markers} seeds={sweep_seeds}", flush=True)
        for idx, cfg in enumerate(sweep_grid, start=1):
            print(f"[Sweep] {idx}/{len(sweep_grid)} markers={len(sweep_markers)} seeds={len(sweep_seeds)} cfg={cfg}", flush=True)
            cfg_rows = _run_setting(cfg, sweep_markers, sweep_seeds, save_diagnostics=False)
            cfg_df = pd.DataFrame(cfg_rows)
            val_df = cfg_df[cfg_df["split"] == "val"] if not cfg_df.empty else pd.DataFrame()
            sweep_rows.append(
                {
                    **cfg,
                    "markers": len(sweep_markers),
                    "seeds": len(sweep_seeds),
                    "mean_val_dt_mse": float(val_df["dt_mse"].mean()) if not val_df.empty and "dt_mse" in val_df.columns else np.nan,
                    "mean_val_dt_r2": float(val_df["dt_r2"].mean()) if not val_df.empty else np.nan,
                    "mean_val_integ_r2": float(val_df["integ_r2_median"].mean()) if not val_df.empty else np.nan,
                    "mean_val_ode_r2": float(val_df["ode_integ_r2_median"].mean()) if not val_df.empty else np.nan,
                }
            )
        sweep_df = pd.DataFrame(sweep_rows)
        if sweep_out is None:
            sweep_out = Path(args.output_dir) / "neural_ode_sweep.csv"
        sweep_out.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(sweep_out, index=False)
        if not sweep_df.empty and "mean_val_dt_mse" in sweep_df.columns:
            metric = sweep_df["mean_val_dt_mse"].replace([np.inf, -np.inf], np.nan)
            if metric.notna().any():
                best_idx = metric.idxmin()
                best_row = sweep_df.loc[best_idx]
                print(f"[Sweep] Best (mean val dt MSE) -> {best_row.to_dict()}", flush=True)
                # Promote best sweep config to the default run unless sweep-only.
                sweep_best = {
                    "hidden_dim": float(best_row.get("hidden_dim", args.hidden_dim)),
                    "hidden_layers": float(best_row.get("hidden_layers", args.hidden_layers)),
                    "dropout": float(best_row.get("dropout", args.dropout)),
                    "lr": float(best_row.get("lr", args.lr)),
                    "weight_decay": float(best_row.get("weight_decay", args.weight_decay)),
                }
                # Persist best hyperparameters for posterity.
                best_params_path = sweep_out.with_name("neural_ode_best_params.json")
                best_payload = {
                    "best_params": {
                        "hidden_dim": int(round(float(best_row.get("hidden_dim", args.hidden_dim)))),
                        "hidden_layers": int(round(float(best_row.get("hidden_layers", args.hidden_layers)))),
                        "dropout": float(best_row.get("dropout", args.dropout)),
                        "lr": float(best_row.get("lr", args.lr)),
                        "weight_decay": float(best_row.get("weight_decay", args.weight_decay)),
                    },
                    "metric": "mean_val_dt_mse",
                    "best_metric": float(best_row.get("mean_val_dt_mse", float("nan"))),
                    "sweep_csv": str(sweep_out),
                    "markers": int(best_row.get("markers", 0)) if "markers" in best_row else None,
                    "seeds": int(best_row.get("seeds", 0)) if "seeds" in best_row else None,
                }
                try:
                    best_params_path.write_text(json.dumps(best_payload, indent=2))
                    print(f"[Sweep] Wrote best params to {best_params_path}", flush=True)
                except Exception as exc:
                    print(f"[Sweep] Failed to write best params JSON: {exc}", flush=True)
        if args.sweep_only:
            print(f"[Sweep] Completed sweep only. Results at {sweep_out}")
            return

    metrics_rows: List[Dict[str, object]] = []
    default_hparams: Dict[str, float] = {
        "hidden_dim": args.hidden_dim,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
    }
    if sweep_best:
        # Cast back to expected types; keep training schedule from CLI args.
        default_hparams.update(
            {
                "hidden_dim": int(round(sweep_best["hidden_dim"])),
                "hidden_layers": int(round(sweep_best["hidden_layers"])),
                "dropout": float(sweep_best["dropout"]),
                "lr": float(sweep_best["lr"]),
                "weight_decay": float(sweep_best["weight_decay"]),
            }
        )
    metrics_rows.extend(_run_setting(default_hparams, None, None, save_diagnostics=True))

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = outdir / "neural_ode_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    agg_cols = [
        "dt_r2",
        "dt_rel_mae_mean_bins",
        "dt_rel_mae_median_bins",
        "integ_r2_median",
        "integ_rel_mae_mean_bins",
        "integ_rel_mae_median_bins",
        "ode_integ_r2_median",
        "ode_integ_rel_mae_mean_bins",
        "ode_integ_rel_mae_median_bins",
    ]
    agg_cols = [c for c in agg_cols if c in metrics_df.columns]
    agg_metrics = (
        metrics_df.groupby(["marker", "k", "split"], dropna=False)[agg_cols]
        .mean()
        .reset_index()
    )
    agg_metrics_path = outdir / "neural_ode_metrics_agg.csv"
    agg_metrics.to_csv(agg_metrics_path, index=False)

    # Add per-marker overall rows (mean of train/test)
    overall_rows: List[Dict[str, object]] = []
    for (marker, k), grp in agg_metrics.groupby(["marker", "k"]):
        overall_rows.append(
            {
                "marker": marker,
                "k": k,
                "split": "overall",
                "dt_r2": float(np.nanmean(grp["dt_r2"])) if not grp["dt_r2"].isna().all() else np.nan,
                "dt_rel_mae_mean_bins": float(np.nanmean(grp["dt_rel_mae_mean_bins"]))
                if "dt_rel_mae_mean_bins" in grp and not grp["dt_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "dt_rel_mae_median_bins": float(np.nanmean(grp["dt_rel_mae_median_bins"]))
                if "dt_rel_mae_median_bins" in grp and not grp["dt_rel_mae_median_bins"].isna().all()
                else np.nan,
                "integ_r2_median": float(np.nanmean(grp["integ_r2_median"]))
                if not grp["integ_r2_median"].isna().all()
                else np.nan,
                "integ_rel_mae_mean_bins": float(np.nanmean(grp["integ_rel_mae_mean_bins"]))
                if "integ_rel_mae_mean_bins" in grp and not grp["integ_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "integ_rel_mae_median_bins": float(np.nanmean(grp["integ_rel_mae_median_bins"]))
                if "integ_rel_mae_median_bins" in grp and not grp["integ_rel_mae_median_bins"].isna().all()
                else np.nan,
                "ode_integ_r2_median": float(np.nanmean(grp["ode_integ_r2_median"]))
                if not grp["ode_integ_r2_median"].isna().all()
                else np.nan,
                "ode_integ_rel_mae_mean_bins": float(np.nanmean(grp["ode_integ_rel_mae_mean_bins"]))
                if "ode_integ_rel_mae_mean_bins" in grp and not grp["ode_integ_rel_mae_mean_bins"].isna().all()
                else np.nan,
                "ode_integ_rel_mae_median_bins": float(np.nanmean(grp["ode_integ_rel_mae_median_bins"]))
                if "ode_integ_rel_mae_median_bins" in grp and not grp["ode_integ_rel_mae_median_bins"].isna().all()
                else np.nan,
            }
        )
    metrics_plot_df = agg_metrics.copy()
    if overall_rows:
        metrics_plot_df = pd.concat([metrics_plot_df, pd.DataFrame(overall_rows)], axis=0, ignore_index=True)

    _plot_boxplot(
        metrics_plot_df[metrics_plot_df["split"] != "overall"],
        "dt_r2",
        "dt R²",
        outdir / "boxplot_dt_r2.png",
        "dt R² (train vs test)",
    )
    _plot_boxplot(
        metrics_plot_df[metrics_plot_df["split"] != "overall"],
        "integ_r2_median",
        "Integrated R²",
        outdir / "boxplot_integ_r2.png",
        "Integrated R² (train vs test)",
    )
    _plot_boxplot(
        metrics_plot_df[metrics_plot_df["split"] != "overall"],
        "ode_integ_r2_median",
        "ODE integrated R²",
        outdir / "boxplot_ode_r2.png",
        "ODE integrated R² (train vs test)",
    )

    print(f"Wrote per-seed metrics to {metrics_path}")
    print(f"Wrote seed-aggregated metrics to {agg_metrics_path}")


if __name__ == "__main__":
    main()
