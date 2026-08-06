#!/usr/bin/env python3
"""Throwaway sanity plots: exp(kcat_cg) vs exp(P_u/P_p) for a handful of trajectories."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path("data")
PLOT_DIR = "sanity_plots"
CSV_CANDIDATES = [
    "data_merged.csv",
    "filtered_data.csv",
    "train_data.csv",
    "train.csv",
    "data_train.csv",
]
TIME_COLS = ["time", "Time", "t"]
KCAT_COLS = ["kcat_cg", "kcat", "log_kcat"]

STATIC_RENAME = {
    'K(p=None)': 'K',
    "K(p=None, d=None)": 'K',
    "P(phospho='u', k=None)": 'P_u',
    "P(phospho='p', k=None)": 'P_p',
    "S(k=None)": 'P_u',
    "P(k=None)": 'P_p',
    "K(p=1) % P(phospho='u', k=1)": 'KPu',
    "K(p=1) % P(phospho='p', k=1)": 'KPp',
    "K(p=1) % S(k=1)": 'KPu',
    "K(p=1) % P(k=1)": 'KPp',
    'koff_substrate': 'k_off',
    'kD_substrate': 'k_D',
    'kcat': 'k_cat',
    'kinact': 'k_inact',
}


def find_dataset_path(root: Path):
    if not root.is_dir():
        return None
    for name in CSV_CANDIDATES:
        path = root / name
        if path.is_file():
            return path
    csvs = sorted(root.glob("*.csv"))
    return csvs[0] if csvs else None


def pick(ids, k):
    ids = [str(x) for x in ids]
    ids = sorted(dict.fromkeys(ids))  # preserve order, enforce uniqueness
    if len(ids) <= k:
        return ids
    return list(np.random.default_rng().choice(ids, size=k, replace=False))


def plot_variant(model, df, ids, column, time_col, suffix, prefix):
    out_dir = DATA_ROOT / model / PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list(ids)
    cols = min(3, max(1, len(ids)))
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)

    legend_shown = False

    for ax, cid in zip(axes.flat, ids):
        traj = df[df["condition_id"] == cid]
        if time_col in traj:
            traj = traj.sort_values(time_col)
        x = np.exp(pd.to_numeric(traj[column], errors="coerce").to_numpy())
        y = np.exp(pd.to_numeric(traj["kcat_cg"], errors="coerce").to_numpy())
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.plot(x[mask], y[mask], marker="o", linewidth=1.2, label="exp(kcat_cg)")

            if column == "P_u" and {"k_off", "k_D", "k_cat", "tK"}.issubset(traj.columns):
                k_off = np.exp(pd.to_numeric(traj["k_off"], errors="coerce").to_numpy())
                k_D = np.exp(pd.to_numeric(traj["k_D"], errors="coerce").to_numpy())
                k_cat = np.exp(pd.to_numeric(traj["k_cat"], errors="coerce").to_numpy())
                tK = np.exp(pd.to_numeric(traj["tK"], errors="coerce").to_numpy())
                if "k_inact" in traj.columns:
                    k_inact = np.exp(pd.to_numeric(traj["k_inact"], errors="coerce").to_numpy())
                else:
                    k_inact = np.zeros_like(k_off)

                denom = x + ((k_cat + k_off + k_inact) / (k_off * k_D))
                mm_vals = np.divide(
                    tK * x,
                    denom,
                    out=np.full_like(x, np.nan),
                    where=np.isfinite(denom) & (denom != 0),
                )
                mm_mask = np.isfinite(mm_vals) & mask
                if mm_mask.any():
                    ax.plot(x[mm_mask], mm_vals[mm_mask], color="orange", linewidth=1.2, label="MM (no k_cat)")
                    if not legend_shown:
                        ax.legend(loc="best", frameon=False)
                        legend_shown = True
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(str(cid))
        ax.set_xlabel(f"exp({column})")
        ax.set_ylabel("exp(kcat_cg)")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes.flat[len(ids):]:
        ax.set_axis_off()

    fig.suptitle(f"{model} – {column}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{prefix}_{suffix}.png", dpi=300)
    plt.close(fig)


def plot_vs_time(model, df, ids, column, time_col, suffix, prefix):
    if time_col is None:
        print(f"[skip] {model}: no time column for {column} vs time plot")
        return

    out_dir = DATA_ROOT / model / PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list(ids)
    cols = min(3, max(1, len(ids)))
    rows = math.ceil(len(ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)

    for ax, cid in zip(axes.flat, ids):
        traj = df[df["condition_id"] == cid]
        subset = traj[[time_col, column]].apply(pd.to_numeric, errors="coerce")
        subset = subset.dropna()
        if subset.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
        else:
            ax.plot(subset[time_col].to_numpy(), np.exp(subset[column].to_numpy()), marker="o", linewidth=1.2)
        ax.set_title(str(cid))
        ax.set_xlabel(time_col)
        ax.set_ylabel(f"exp({column})")
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes.flat[len(ids):]:
        ax.set_axis_off()

    fig.suptitle(f"{model} – {column} vs time", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / f"{prefix}_{suffix}_vs_time.png", dpi=300)
    plt.close(fig)


def process_model(model: str, mode: str, dataset_path: Path, max_traj: int):
    if dataset_path is None:
        print(f"[skip] {model} ({mode}): no dataset found")
        return

    df = pd.read_csv(dataset_path)

    if mode.startswith("static_traj"):
        df = df.rename(columns=STATIC_RENAME)
        df = df.loc[:, ~df.columns.duplicated()]
    missing_cols = {"condition_id", "P_u", "P_p"} - set(df.columns)
    if missing_cols:
        print(f"[skip] {model} ({mode}): missing {', '.join(sorted(missing_cols))}")
        return

    kcat_col = next((c for c in KCAT_COLS if c in df.columns), None)
    if kcat_col is None:
        print(f"[skip] {model} ({mode}): no kcat column")
        return
    if kcat_col != "kcat_cg":
        df = df.rename(columns={kcat_col: "kcat_cg"})

    time_col = next((c for c in TIME_COLS if c in df.columns), None)
    if not mode.startswith("dynamic"):
        time_col = None

    if mode.startswith("static_traj") and "P_p" in df.columns:
        product_variation = df.groupby("condition_id")["P_p"].nunique(dropna=True)
        varying_product_ids = product_variation[product_variation > 1].index.tolist()
        if varying_product_ids:
            preview = ", ".join(str(cid) for cid in varying_product_ids[:20])
            ellipsis = " ..." if len(varying_product_ids) > 20 else ""
            print(
                f"[info] {model} ({mode}): P_p varies for {len(varying_product_ids)} trajectories: "
                f"{preview}{ellipsis}"
            )
        else:
            print(f"[warn] {model} ({mode}): P_p is constant across all trajectories")

    ids = df["condition_id"].dropna().unique()
    if len(ids) == 0:
        print(f"[skip] {model} ({mode}): empty condition_id")
        return
    ids = pick(ids, max_traj)

    base_cols = [c for c in ["condition_id", time_col, "P_u", "P_p", "kcat_cg"] if c in df.columns]
    mm_cols = [c for c in ["k_off", "k_D", "k_cat", "k_inact", "tK"] if c in df.columns]
    for cid in ids:
        print(f"\n=== {model} ({mode}) | {cid} ===")
        traj = df[df["condition_id"] == cid]
        varying_cols = []
        constant_cols = []
        for col in mm_cols:
            series = pd.to_numeric(traj[col], errors="coerce")
            finite = series[np.isfinite(series)]
            if finite.empty:
                continue
            if finite.nunique(dropna=True) == 1:
                constant_cols.append((col, finite.iloc[0]))
            else:
                varying_cols.append(col)

        display_cols = base_cols + [c for c in varying_cols if c not in base_cols]
        print(traj[display_cols].to_string(index=False))
        for col, value in constant_cols:
            print(f"  {col}: {value}")

    prefix = mode
    plot_variant(model, df, ids, "P_u", time_col, "substrate", prefix)
    plot_variant(model, df, ids, "P_p", time_col, "product", prefix)
    if mode.startswith("dynamic"):
        plot_vs_time(model, df, ids, "P_p", time_col, "product", prefix)
        plot_vs_time(model, df, ids, "P_u", time_col, "substrate", prefix)


def discover_models():
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Plot exp(kcat_cg) vs exp(P_u/P_p) for random trajectories")
    parser.add_argument("--models", nargs="*", help="Which model directories to hit under data/")
    parser.add_argument("--max-trajectories", type=int, default=10, help="How many condition IDs to plot per model")
    args = parser.parse_args()

    models = args.models or discover_models()
    for model in models:
        model_dir = DATA_ROOT / model
        if not model_dir.is_dir():
            continue

        # dynamic
        dyn_path = find_dataset_path(model_dir / "dynamic" / "processed")
        dyn_mode = "dynamic"
        if dyn_path is None:
            dyn_path = find_dataset_path(model_dir / "dynamic" / "raw")
            dyn_mode = "dynamic_raw" if dyn_path is not None else dyn_mode
        if dyn_path is not None:
            process_model(model, dyn_mode, dyn_path, args.max_trajectories)

        # any static trajectory variants
        for static_dir in sorted(model_dir.glob("static_traj*/")):
            mode_name = static_dir.name
            data_path = find_dataset_path(static_dir / "raw")
            if data_path is not None:
                process_model(model, mode_name, data_path, args.max_trajectories)


if __name__ == "__main__":
    main()
