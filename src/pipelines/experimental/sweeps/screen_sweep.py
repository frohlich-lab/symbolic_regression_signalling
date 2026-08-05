"""Screen the overnight OOD sweep.

For every (arm, marker, seed) this computes the held-out trajectory metrics plus the
shape diagnostics that actually track overlay quality -- peak-timing error, decay
fraction error, and GFP-bin ordering fidelity -- because ode_integ_r2_median alone
ranks flat-fan fits above dynamically correct ones.

Usage: python screen_sweep.py <sweep_dir> [out_csv]
"""
import sys
import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SWEEP = sys.argv[1].rstrip("/")
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else f"{SWEEP}/screen_results.csv"

CONTROL_PAT = "untransfected|FLAG-GFP"


def shape_stats(g, col):
    """Median peak time and median decay fraction across GFP bins."""
    peak_t, decay = [], []
    for _, h in g.groupby("GFP_bin"):
        h = h.sort_values("timepoint")
        y = h[col].to_numpy(float)
        if len(y) < 5 or not np.isfinite(y).all():
            continue
        i = int(np.argmax(y))
        pk, y0, yf = y[i], y[0], y[-1]
        if pk - y0 <= 1e-9:
            continue
        peak_t.append(h["timepoint"].to_numpy()[i])
        decay.append((pk - yf) / (pk - y0))
    return np.array(peak_t), np.array(decay)


METRIC_GLOBS = [
    # arm x marker x seed layout: out_seed/<arm>/<slug>/s<seed>/seeds/seed_NN/metrics/
    (f"{SWEEP}/out_seed/*/*/s*/seeds/seed_*/metrics/marker_integration_metrics_per_minute.csv", -7),
    # arm x marker sharded layout: out_shard/<arm>/<marker_slug>/seeds/seed_NN/metrics/
    (f"{SWEEP}/out_shard/*/*/seeds/seed_*/metrics/marker_integration_metrics_per_minute.csv", -6),
    # per-arm layout: out/<arm>/seeds/seed_NN/metrics/
    (f"{SWEEP}/out/*/seeds/seed_*/metrics/marker_integration_metrics_per_minute.csv", -5),
]

rows = []
metric_paths = []
for pattern, arm_pos in METRIC_GLOBS:
    for p in sorted(glob.glob(pattern)):
        metric_paths.append((p, p.split(os.sep)[arm_pos]))

if metric_paths:
    print(f"found {len(metric_paths)} metrics files")

for mpath, arm in metric_paths:
    if arm.startswith("_"):
        continue
    if True:
        seed = int(mpath.split("seed_")[1].split("/")[0])
        try:
            met = pd.read_csv(mpath)
        except Exception as exc:
            print(f"  skip {mpath}: {exc}")
            continue
        tpath = os.path.join(os.path.dirname(mpath), "marker_integration_trajectories_per_minute.csv")
        traj = None
        if os.path.exists(tpath):
            try:
                t = pd.read_csv(tpath)
                # Fits whose formula could not be ODE-integrated have no such column.
                traj = t[t["pred_integrated_ode"].notna()] if "pred_integrated_ode" in t.columns else None
            except Exception as exc:
                print(f"  traj unreadable {tpath}: {exc}")
                traj = None

        for _, r in met.iterrows():
            marker = r.get("marker")
            rec = {
                "arm": arm,
                "marker": marker,
                "seed": seed,
                "ode_r2": r.get("ode_integ_r2_median", np.nan),
                "ode_r2_train": r.get("ode_integ_r2_median_train", np.nan),
                "dt_r2": r.get("dt_r2", np.nan),
                "rel_mae": r.get("ode_integ_rel_mae_median_bins", np.nan),
                "valid_bins": r.get("ode_integ_rel_mae_valid_bins", np.nan),
                "formula": r.get("formula"),
                "peak_dt": np.nan, "decay_p": np.nan, "decay_o": np.nan,
                "decay_err": np.nan, "order": np.nan, "pred_min": np.nan,
            }
            if traj is not None and marker is not None and "marker" in traj.columns:
                g = traj[traj["marker"] == marker]
                if len(g) > 0:
                    pp, dp = shape_stats(g, "pred_integrated_ode")
                    po, do = shape_stats(g, "obs_pERK1_2")
                    if len(pp) >= 3 and len(po) >= 3:
                        rec["peak_dt"] = float(np.median(pp) - np.median(po))
                        rec["decay_p"] = float(np.median(dp))
                        rec["decay_o"] = float(np.median(do))
                        rec["decay_err"] = rec["decay_p"] - rec["decay_o"]
                    bp = g.groupby("GFP_bin").agg(
                        pred=("pred_integrated_ode", "mean"),
                        obs=("obs_pERK1_2", "mean")).dropna()
                    if len(bp) >= 5:
                        rec["order"] = float(spearmanr(bp["obs"], bp["pred"]).statistic)
                    rec["pred_min"] = float(g["pred_integrated_ode"].min())
            rows.append(rec)

if not rows:
    print("No results found under", f"{SWEEP}/out/*/seeds/seed_*/metrics/")
    sys.exit(1)

d = pd.DataFrame(rows)
d = d[~d["marker"].astype(str).str.contains(CONTROL_PAT, case=False, na=False)]

# Pre-declared screen: dynamically admissible AND correct shape AND ordered fan.
d["pass_screen"] = (
    (d["ode_r2"] >= 0.6)
    & (d["dt_r2"] > 0)
    & (d["peak_dt"].abs() <= 6)
    & (d["decay_err"].abs() <= 0.25)
    & (d["order"] >= 0.85)
    & (d["pred_min"] > 0)
    & (d["valid_bins"] >= 9)
)

d.to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 95)

print(f"\n=== {len(d)} (arm,marker,seed) results across {d['arm'].nunique()} arms ===")
print(f"written: {OUT_CSV}\n")

print("=== per-arm summary (non-control) ===")
summ = d.groupby("arm").agg(
    n=("ode_r2", "size"),
    n_pass=("pass_screen", "sum"),
    ode_r2_median=("ode_r2", "median"),
    ode_r2_max=("ode_r2", "max"),
    n_ode_gt_06=("ode_r2", lambda s: int((s >= 0.6).sum())),
    mean_abs_decay_err=("decay_err", lambda s: s.abs().mean()),
).sort_values(["n_pass", "ode_r2_median"], ascending=False)
print(summ.round(3).to_string())

print("\n=== fits PASSING the screen, best first ===")
p = d[d["pass_screen"]].sort_values("ode_r2", ascending=False)
cols = ["arm", "marker", "seed", "ode_r2", "dt_r2", "rel_mae", "peak_dt", "decay_p", "decay_o", "order", "formula"]
print(p[cols].round(3).to_string(index=False) if len(p) else "  (none)")

print("\n=== reference: current OOD run best was ERBB2/44 ode_r2=0.955 (only clean overlay) ===")
print("=== distinct markers passing, best fit each ===")
if len(p):
    best = p.loc[p.groupby("marker")["ode_r2"].idxmax()].sort_values("ode_r2", ascending=False)
    print(best[cols].round(3).to_string(index=False))
