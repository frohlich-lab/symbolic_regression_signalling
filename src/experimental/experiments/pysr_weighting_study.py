#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySR weighting study over functional marker groups (enhanced logging + NaN fixes).

Changes vs. your original:
- Added a --debug flag for detailed print logs at key steps (data loading, grouping,
  feature selection, weighting stats, split sizes, model training, and evaluation).
- Fixed NaNs in full/unbalanced log_R2 by:
  * Passing stable, sanitized feature names to PySR via feature_names.
  * Adding corresponding sanitized columns to dataframes so evaluate_formula can
    actually find the variables referenced in the symbolic expression.
  * Vectorized evaluate_formula and added robust masking.
- Preserved your defaults and CLI, so the Snakefile call still works.

New in this revision:
- FIX for SymbolicRegression.jl calling loss with 2 args:
  We define BOTH methods in Julia:
    myloss(x, y, w) = ...
    myloss(x, y)    = myloss(x, y, 1.0)
  so the "MethodError: no method matching myloss(::Float32, ::Float32)" is gone.

"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from sklearn.model_selection import train_test_split

# Ensure julia sees a clean python (PySR embeds Julia)
os.environ.pop("PYTHON", None)
os.environ["JULIA_ARGS"] = "--startup-file=no"
os.environ.setdefault("JULIA_NUM_THREADS", "auto")

from pysr import PySRRegressor  # noqa: E402

TARGET_COLUMN = "p-ERK1-2_dt"


# ---------------------------
# Args
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weighting strategies study for PySR (with debug + NaN fixes).")
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--group-definitions", required=True, type=str,
                   help="CSV with columns: group,members")
    p.add_argument("--groups", nargs="*", default=None,
                   help="Subset of group names to run; default = all from CSV.")
    p.add_argument("--gfp-columns", nargs="*", default=("GFP", "p-ERK1-2", "p-MEK1-2", "p-ERK1-2_min", "p-MEK1-2_min"))
    p.add_argument("--feature-mode", choices=("gfp",), default="gfp")

    p.add_argument("--weighting", nargs="*", default=("none", "inverse_oom", "sqrt_inverse_oom", "marker_equal"),
                   choices=("none", "inverse_oom", "sqrt_inverse_oom", "marker_equal", "combined"))
    p.add_argument("--fit-modes", nargs="*", default=("weighted",), choices=("weighted", "duplicate"))

    p.add_argument("--log10-cutoff", type=float, default=-3.0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--min-group-size", type=int, default=200)

    # PySR params (match your project defaults)
    p.add_argument("--max-iterations", type=int, default=300)
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--populations", type=int, default=30)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--parsimony", type=float, default=0.8)
    p.add_argument("--binary-operators", nargs="*", default=("+", "-", "*", "/"))
    p.add_argument("--unary-operators", nargs="*", default=())
    p.add_argument("--verbosity", type=int, default=0)

    p.add_argument("--output", required=True, type=str,
                   help="Path to write the summary text report.")

    # New: debug printing without changing PySR's own verbosity
    p.add_argument("--debug", action="store_true", help="Print detailed progress logs.")

    return p.parse_args()


# ---------------------------
# Utilities
# ---------------------------
def dprint(debug: bool, *args, **kwargs) -> None:
    if debug:
        print(*args, **kwargs, flush=True)


def _parse_members_forgiving(cell) -> List[str]:
    """Accept ['A','B'], ["A","B"], or A,B. Strip quotes/spaces."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(m).strip().strip("'\"") for m in parsed]
    except Exception:
        pass
    return [m.strip().strip("'\"") for m in s.strip("[]").split(",") if m.strip()]


def load_groups_csv(path: str, available_markers: Sequence[str], debug: bool = False) -> Dict[str, List[str]]:
    df = pd.read_csv(path)
    if "group" not in df.columns or "members" not in df.columns:
        raise ValueError("Group definitions CSV must include 'group' and 'members' columns.")
    avail = set(str(m) for m in available_markers if pd.notna(m))
    groups: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        name = str(row["group"]).strip()
        members_raw = row["members"]
        if not name or pd.isna(members_raw):
            continue
        members = _parse_members_forgiving(members_raw)
        keep = [m for m in members if m in avail]
        if keep:
            groups[name] = keep
        else:
            dprint(debug, f"[groups] '{name}' had no usable members after intersection.")

    if not groups:
        ds_sample = ", ".join(list(avail)[:12])
        raw = []
        for _, r in df.head(5).iterrows():
            raw += _parse_members_forgiving(r["members"])
        csv_sample = ", ".join(list(dict.fromkeys(raw))[:12])
        raise ValueError(
            "no usable groups after intersecting with dataset markers\n"
            f"- sample dataset markers: [{ds_sample}]\n"
            f"- sample CSV members:     [{csv_sample}]\n"
            "Check spelling/case/whitespace and that CSV 'members' are actual values from the dataset 'marker' column."
        )
    return groups


def sanitize_feature_names(columns: Sequence[str]) -> List[str]:
    out = []
    for c in columns:
        s = c.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        if s and s[0].isdigit():
            s = f"f_{s}"
        out.append(s)
    return out


def ensure_sanitized_columns(df: pd.DataFrame, feature_cols: Sequence[str], debug: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Add sanitized copies of feature columns so SymPy/PySR names are present as columns.
    Returns (df_with_copies, sanitized_names).
    """
    sanitized = sanitize_feature_names(feature_cols)
    for src, dst in zip(feature_cols, sanitized):
        if dst not in df.columns:
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")
                dprint(debug, f"[sanitize] Added column '{dst}' from '{src}'.")
            else:
                # Create empty column so later masks will drop non-finite
                df[dst] = np.nan
                dprint(debug, f"[sanitize] Missing source '{src}', created NaN column '{dst}'.")
    return df, sanitized


def load_dataset(path: str, log10_cutoff: float, debug: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize column names if *_fit exists in upstream
    df.columns = df.columns.str.replace("_fit$", "", regex=True)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not present.")
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    eps = 10.0 ** log10_cutoff
    df.loc[df[TARGET_COLUMN].abs() < eps, TARGET_COLUMN] = 0.0
    after = len(df)
    dprint(debug, f"[dataset] Loaded {before} rows, kept {after} after target cleaning.")
    return df


def select_feature_columns(df: pd.DataFrame, mode: str, gfp_columns: Sequence[str], debug: bool = False) -> List[str]:
    if mode != "gfp":
        raise ValueError("Only feature-mode 'gfp' supported in this simple script.")
    sanitized = sanitize_feature_names(gfp_columns)
    # sanitize df columns the same way (but only for lookup; don’t rename df)
    df_sanitized_map = {sanitize_feature_names([c])[0]: c for c in df.columns}
    cols = [df_sanitized_map[s] for s in sanitized if s in df_sanitized_map]
    if not cols:
        raise ValueError("No requested GFP columns were found in dataset after sanitisation.")
    dprint(debug, f"[features] Selected columns: {cols}")
    return cols


# ---------------------------
# Weighting bits
# ---------------------------
def weighting_function(strategy: str, df: pd.DataFrame, log10_cutoff: float) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    if strategy == "none":
        return np.ones(n, dtype=float)

    eps = 10.0 ** log10_cutoff
    target = df[TARGET_COLUMN].to_numpy(dtype=float)

    if strategy in {"inverse_oom", "sqrt_inverse_oom"}:
        bins = np.floor(np.log10(np.abs(target) + eps))
        unique, counts = np.unique(bins, return_counts=True)
        count_map = dict(zip(unique, counts))
        base = np.array([count_map.get(b, 1.0) for b in bins], dtype=float)
        inv = np.where(base > 0, 1.0 / base, np.nan)
        if strategy == "sqrt_inverse_oom":
            inv = np.sqrt(inv)
        return inv

    if strategy == "marker_equal":
        if "marker" not in df.columns:
            raise ValueError("Column 'marker' required for marker_equal weighting.")
        counts = df["marker"].value_counts()
        w = df["marker"].map(lambda m: 1.0 / counts.get(m, 1))
        return w.to_numpy(dtype=float)

    if strategy == "combined":
        # oom
        bins = np.floor(np.log10(np.abs(target) + eps))
        unique, counts = np.unique(bins, return_counts=True)
        count_map = dict(zip(unique, counts))
        inv = np.array([count_map.get(b, 1.0) for b in bins], dtype=float)
        inv = np.where(inv > 0, 1.0 / inv, np.nan)
        # marker
        if "marker" not in df.columns:
            raise ValueError("Column 'marker' required for 'combined' weighting.")
        mc = df["marker"].value_counts()
        mi = df["marker"].map(lambda m: 1.0 / mc.get(m, 1))
        return inv * mi.to_numpy(dtype=float)

    raise ValueError(f"Unknown weighting strategy: {strategy}")


def normalise_weights(w: np.ndarray) -> np.ndarray:
    w = w.astype(float)
    m = np.isfinite(w) & (w > 0)
    if not m.any():
        return np.ones_like(w)
    w = np.where(m, w, np.nan)
    mu = np.nanmean(w)
    if not np.isfinite(mu) or mu <= 0:
        return np.ones_like(w)
    w = np.where(np.isfinite(w), w, mu)
    return w / mu


def expand_training_samples(X: np.ndarray, y: np.ndarray, weights: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    total = np.sum(w)
    if not np.isfinite(total) or total <= 0:
        w = np.ones_like(w)
        total = len(w)
    scale = len(w) / total
    counts = np.round(w * scale).astype(int)
    counts[counts < 1] = 1
    idx = np.repeat(np.arange(len(w)), counts)
    dprint(debug, f"[duplicate] Expanded from {len(X)} to {len(idx)} samples (scale~{scale:.3f}).")
    return X[idx], y[idx]


# ---------------------------
# Metrics helpers
# ---------------------------
def _signed_log(v: np.ndarray, log10_cutoff: float) -> np.ndarray:
    eps = 10.0 ** log10_cutoff
    v = np.asarray(v, dtype=float)
    return np.sign(v) * (np.log10(np.abs(v) + eps) - log10_cutoff)


def _log_r2_score(y_true: np.ndarray, y_pred: np.ndarray, log10_cutoff: float) -> float:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return float("nan")
    y_true = y_true[m]
    y_pred = y_pred[m]
    yt = _signed_log(y_true, log10_cutoff)
    yp = _signed_log(y_pred, log10_cutoff)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def evaluate_formula(
    expr: sp.Expr | str,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    log10_cutoff: float,
    debug: bool = False,
    sanitized_feature_names: Optional[Sequence[str]] = None,
) -> float:
    """Vectorized evaluation that ensures sanitized feature-name columns exist.
    Also supports legacy PySR expressions that use x0, x1, ... by aliasing those
    to the sanitized features when provided.
    Returns log_R2 or NaN if insufficient data.
    """
    if df.empty:
        return float("nan")
    try:
        safe = sp.sympify(str(expr).replace("^", "**"))
    except Exception:
        return float("nan")

    # Ensure sanitized copies so symbol names (e.g., p-ERK1-2 -> p_ERK1_2) exist
    df_local = df.copy()
    df_local, sanitized = ensure_sanitized_columns(df_local, feature_cols, debug=debug)

    # If model used x0, x1, ... create aliases to sanitized columns
    if sanitized_feature_names is None:
        alias_names = sanitized
    else:
        alias_names = list(sanitized_feature_names)
    try:
        for i, name in enumerate(alias_names):
            xcol = f"x{i}"
            if xcol not in df_local.columns and name in df_local.columns:
                df_local[xcol] = pd.to_numeric(df_local[name], errors="coerce")
    except Exception:
        pass

    symbols = sorted(safe.free_symbols, key=lambda s: s.name)
    if not symbols:
        c = float(safe)
        y = pd.to_numeric(df_local[TARGET_COLUMN], errors="coerce").to_numpy(dtype=float)
        yhat = np.full_like(y, c)
        return _log_r2_score(y, yhat, log10_cutoff)

    # Build design matrix in symbol order; missing symbol columns -> NaN
    sym_names = [s.name for s in symbols]
    for s in sym_names:
        if s not in df_local.columns:
            df_local[s] = np.nan
    X = df_local[sym_names].apply(pd.to_numeric, errors="coerce")

    y = pd.to_numeric(df_local[TARGET_COLUMN], errors="coerce")
    mask = np.isfinite(X.to_numpy()).all(axis=1) & np.isfinite(y.to_numpy())
    n_mask = int(mask.sum())
    if n_mask < 2:
        dprint(debug, f"[eval] Too few valid rows for evaluation: {n_mask} (symbols={sym_names}).")
        return float("nan")

    f = sp.lambdify(symbols, safe, modules={"numpy": np})
    try:
        y_pred = f(*[X.loc[mask, s].to_numpy(dtype=float) for s in sym_names])
        y_pred = np.asarray(y_pred, dtype=float)
    except Exception as exc:
        dprint(debug, f"[eval] Sympy lambdify/predict failed: {exc}")
        return float("nan")

    return _log_r2_score(y.loc[mask].to_numpy(dtype=float), y_pred, log10_cutoff)


# ---------------------------
# Core run
# ---------------------------
def run_weighting_strategy(
    group_name: str,
    markers: Sequence[str],
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    sanitized_feature_names: Sequence[str],
    strategy: str,
    fit_mode: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    debug = args.debug
    base = {"group": group_name, "strategy": strategy, "fit_mode": fit_mode}
    gdf = df[df["marker"].isin(markers)].copy()
    if gdf.empty:
        dprint(debug, f"[group:{group_name}] No rows after marker filter.")
        return {**base, "status": "error", "reason": "no_rows"}

    clean = gdf.dropna(subset=list(feature_cols) + [TARGET_COLUMN]).copy()
    for c in feature_cols + [TARGET_COLUMN]:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=list(feature_cols) + [TARGET_COLUMN])

    dprint(debug, f"[group:{group_name}] Clean rows: {len(clean)} (min required {args.min_group_size}).")
    if len(clean) < args.min_group_size:
        return {**base, "status": "skipped", "reason": f"clean rows < {args.min_group_size}", "clean_rows": len(clean)}

    # Ensure sanitized columns exist in dataframes used for evaluation
    gdf_eval, _ = ensure_sanitized_columns(gdf.copy(), feature_cols, debug=debug)
    clean_eval, _ = ensure_sanitized_columns(clean.copy(), feature_cols, debug=debug)

    w = normalise_weights(weighting_function(strategy, clean, args.log10_cutoff))
    X = clean[list(feature_cols)].to_numpy(dtype=float)
    y = clean[TARGET_COLUMN].to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    dprint(debug, f"[split] train={len(X_tr)} test={len(X_te)} | w(mean/min/max)={np.mean(w_tr):.3f}/{np.min(w_tr):.3f}/{np.max(w_tr):.3f}")

    duplicate_mode = fit_mode == "duplicate"
    if duplicate_mode:
        X_fit, y_fit = expand_training_samples(X_tr, y_tr, w_tr, debug=debug)
        w_arg = None
    else:
        X_fit, y_fit = X_tr, y_tr
        w_arg = w_tr

    # ---- FIX: define BOTH loss arities for SymbolicRegression ----
    offset = abs(args.log10_cutoff)
    log_loss_expr = f"""
myloss(x,y,w) = (sign(x)*(log10(abs(x)+10^{args.log10_cutoff})+{offset}) - sign(y)*(log10(abs(y)+10^{args.log10_cutoff})+{offset}))^2 * w
myloss(x,y) = myloss(x,y,1.0)
""".strip()

    # Build model; some PySR versions don't accept variable_names. Retry without if needed.
    try:
        model = PySRRegressor(
            elementwise_loss=log_loss_expr,
            niterations=args.max_iterations,
            population_size=args.population_size,
            populations=args.populations,
            maxsize=args.max_size,
            parsimony=args.parsimony,
            binary_operators=list(args.binary_operators),
            unary_operators=list(args.unary_operators),
            verbosity=args.verbosity,
            procs=0,
            variable_names=list(sanitized_feature_names),  # preferred API
        )
    except TypeError as exc:
        dprint(debug, f"[fit] PySRRegressor didn't accept variable_names: {exc} — retrying without names.")
        model = PySRRegressor(
            elementwise_loss=log_loss_expr,
            niterations=args.max_iterations,
            population_size=args.population_size,
            populations=args.populations,
            maxsize=args.max_size,
            parsimony=args.parsimony,
            binary_operators=list(args.binary_operators),
            unary_operators=list(args.unary_operators),
            verbosity=args.verbosity,
            procs=0,
        )

    try:
        dprint(debug, f"[fit] Starting PySR | mode={'duplicate' if duplicate_mode else 'weighted'} | X={X_fit.shape} ...")
        if duplicate_mode:
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_fit, y_fit, weights=w_arg)
        y_pred = model.predict(X_te)
        dprint(debug, f"[fit] Finished PySR. Predict on test: {y_pred.shape}.")
    except Exception as exc:
        dprint(debug, f"[fit] PySR failure: {exc}")
        return {**base, "status": "error", "reason": f"PySR failure: {exc}"}

    test_log_r2 = _log_r2_score(y_te, y_pred, args.log10_cutoff)
    try:
        formula_expr = model.sympy()
        formula = str(formula_expr)
        dprint(debug, f"[model] Best formula: {formula}")
    except Exception as exc:
        dprint(debug, f"[model] Could not extract sympy formula: {exc}")
        formula_expr = None
        formula = "<no formula>"

    full_log_r2 = (
        evaluate_formula(formula_expr, gdf_eval, feature_cols, args.log10_cutoff, debug=debug, sanitized_feature_names=sanitized_feature_names)
        if formula_expr is not None else float("nan")
    )
    clean_log_r2 = (
        evaluate_formula(formula_expr, clean_eval, feature_cols, args.log10_cutoff, debug=debug, sanitized_feature_names=sanitized_feature_names)
        if formula_expr is not None else float("nan")
    )

    dprint(debug, f"[scores] test={test_log_r2:.4f} full={full_log_r2} clean={clean_log_r2}")

    return {
        **base,
        "status": "ok",
        "train_samples": int(len(X_tr)),
        "effective_train_samples": int(len(X_fit)),
        "test_samples": int(len(X_te)),
        "weights_mean": float(np.mean(w_tr)),
        "weights_min": float(np.min(w_tr)),
        "weights_max": float(np.max(w_tr)),
        "test_log_r2": float(test_log_r2) if (isinstance(test_log_r2, float) and math.isfinite(test_log_r2)) else float("nan"),
        "full_log_r2": float(full_log_r2) if (isinstance(full_log_r2, float) and math.isfinite(full_log_r2)) else float("nan"),
        "clean_log_r2": float(clean_log_r2) if (isinstance(clean_log_r2, float) and math.isfinite(clean_log_r2)) else float("nan"),
        "formula": formula,
        "duplicate_mode": duplicate_mode,
    }


def format_result(res: Mapping[str, object]) -> List[str]:
    status = res.get("status")
    if status == "error":
        return [f"  ! Strategy '{res['strategy']}' failed for group '{res['group']}': {res.get('reason','?')}"]
    if status == "skipped":
        reason = res.get("reason", "skipped")
        clean_rows = res.get("clean_rows")
        extra = f" | clean_rows={clean_rows}" if clean_rows is not None else ""
        return [f"  ~ Strategy '{res['strategy']}' skipped for group '{res['group']}': {reason}{extra}"]

    def fval(x):
        return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.6f}"

    mode_desc = "duplicated" if res.get("duplicate_mode", False) else "weighted"
    lines = [
        f"  - Strategy: {res.get('strategy')}",
        f"    Fit mode: {res.get('fit_mode')}",
        f"    Train samples: {res.get('train_samples')} | Effective train: {res.get('effective_train_samples')} ({mode_desc}) | Test samples: {res.get('test_samples')}",
        f"    Weight stats: mean={res.get('weights_mean'):.3f} min={res.get('weights_min'):.3f} max={res.get('weights_max'):.3f}",
        f"    Test log_R2={fval(res.get('test_log_r2'))}",
        f"    Full dataset log_R2={fval(res.get('full_log_r2'))}",
        f"    Unbalanced clean log_R2={fval(res.get('clean_log_r2'))}",
        f"    Formula: {res.get('formula','<no formula>')}",
    ]
    return lines


# # ---------------------------
# # Main
# # ---------------------------
# def main() -> None:
#     args = parse_args()
#     debug = args.debug
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)

#     dprint(debug, f"[setup] sys.executable={sys.executable}")
#     dprint(debug, f"[setup] JULIA_NUM_THREADS={os.environ.get('JULIA_NUM_THREADS')}")

#     df = load_dataset(args.dataset, args.log10_cutoff, debug=debug)

    # Load groups with old-style robustness
    all_groups = load_groups_csv(args.group_definitions, available_markers=df["marker"].unique(), debug=debug)
    if args.groups:
        missing = [g for g in args.groups if g not in all_groups]
        for g in missing:
            print(f"[warn] Requested group '{g}' not found in definitions after intersection.", flush=True)
        groups = {g: all_groups[g] for g in args.groups if g in all_groups}
        if not groups:
            raise ValueError("None of the requested groups were found/usable.")
    else:
        groups = all_groups

    # Select features
    feature_cols = select_feature_columns(df, args.feature_mode, args.gfp_columns, debug=debug)
    # Precompute sanitized names once and ensure training matrices align with them
    _, sanitized_feature_names = ensure_sanitized_columns(pd.DataFrame(columns=feature_cols), feature_cols, debug=False)

    # Header
    lines: List[str] = []
    lines.append(f"Dataset: {args.dataset}")
    lines.append(f"Feature mode: {args.feature_mode} -> {feature_cols}")
    lines.append(f"Sanitized feature names: {sanitized_feature_names}")
    lines.append(f"Weighting strategies: {', '.join(args.weighting)}")
    lines.append(f"Fit modes: {', '.join(args.fit_modes)}")
    lines.append(f"sys.executable: {sys.executable}")
    lines.append(f"PYTHON env: {os.environ.get('PYTHON')}")
    lines.append("")

    group_items = list(groups.items())
    total_groups = len(group_items)
    total_work = total_groups * len(args.weighting) * len(args.fit_modes)
    work_done = 0

    for gi, (group_name, markers) in enumerate(group_items, start=1):
        lines.append(f"=== Group [{gi}/{total_groups}]: {group_name} | markers={markers} ===")
        print(f"[{gi}/{total_groups}] Group: {group_name}  (markers={len(markers)})", flush=True)
        for si, strategy in enumerate(args.weighting, start=1):
            for mi, mode in enumerate(args.fit_modes, start=1):
                work_done += 1
                print(f"  - {work_done}/{total_work}  strategy={strategy}  mode={mode}", flush=True)
                res = run_weighting_strategy(
                    group_name=group_name,
                    markers=markers,
                    df=df,
                    feature_cols=feature_cols,
                    sanitized_feature_names=sanitized_feature_names,
                    strategy=strategy,
                    fit_mode=mode,
                    args=args,
                )
                lines.extend(format_result(res))
                lines.append("")
        lines.append("")

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote weighting study report to {args.output}", flush=True)


if __name__ == "__main__":
    main()