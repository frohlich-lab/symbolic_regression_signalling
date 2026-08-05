"""Score PySR grid arms and extract marginal hyperparameter effects.

Consumes the CSV written by screen_sweep.py.

Selection rule (declared BEFORE results existed):
  the unit of selection is the ARM, scored over all its (marker, seed) fits by
    1. n_pass       - fits clearing the shape screen   (higher better)
    2. median ode_r2 - median held-out trajectory R2   (higher better)
    3. mean |decay_err| - shape fidelity               (lower better)
  The best single fit inside an arm is explicitly NOT the criterion.

Usage: python score_arms.py <screen_csv> [out_prefix]
"""
import re
import sys
import pandas as pd

CSV = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else CSV.rsplit(".", 1)[0]

d = pd.read_csv(CSV)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)

ARM_RE = re.compile(
    r"^it(?P<iters>\d+)_p(?P<populations>\d+)_ps(?P<pop_size>\d+)_"
    r"ms(?P<max_size>\d+)_par(?P<parsimony>\d+)_u(?P<unary>\w+)_b(?P<binary>\w+)$"
)

parsed = d["arm"].astype(str).str.extract(ARM_RE)
unparsed = parsed["iters"].isna()
if unparsed.all():
    print("No grid-style arm names found; is this the right screen CSV?")
    sys.exit(1)
if unparsed.any():
    print(f"note: dropping {int(unparsed.sum())} rows with non-grid arm names "
          f"({sorted(d.loc[unparsed, 'arm'].unique())[:4]})")
d = pd.concat([d, parsed], axis=1)[~unparsed].copy()
# par01 -> 0.1, par08 -> 0.8, par30 -> 3.0
d["parsimony"] = d["parsimony"].map(lambda s: f"{float(s[0])}.{s[1:]}" if len(s) > 1 else s)
for c in ["iters", "populations", "pop_size", "max_size"]:
    d[c] = d[c].astype(int)

HP = ["iters", "populations", "pop_size", "max_size", "parsimony", "unary", "binary"]

arm = (d.groupby(["arm"] + HP, as_index=False)
         .agg(n=("ode_r2", "size"),
              n_pass=("pass_screen", "sum"),
              median_r2=("ode_r2", "median"),
              max_r2=("ode_r2", "max"),
              n_gt06=("ode_r2", lambda s: int((s >= 0.6).sum())),
              mean_abs_decay=("decay_err", lambda s: s.abs().mean())))

arm = arm.sort_values(["n_pass", "median_r2", "mean_abs_decay"],
                      ascending=[False, False, True]).reset_index(drop=True)
arm.to_csv(f"{PREFIX}_arm_scores.csv", index=False)

print(f"=== {len(arm)} arms scored ({len(d)} fits) -> {PREFIX}_arm_scores.csv ===\n")
print("=== TOP 15 ARMS by declared rule (n_pass, median R2, decay err) ===")
cols = ["arm", "n", "n_pass", "median_r2", "max_r2", "n_gt06", "mean_abs_decay"]
print(arm.head(15)[cols].round(3).to_string(index=False))

print("\n=== MARGINAL EFFECT of each hyperparameter (mean over all arms) ===")
for h in HP:
    g = (arm.groupby(h)
            .agg(arms=("arm", "size"),
                 mean_n_pass=("n_pass", "mean"),
                 mean_median_r2=("median_r2", "mean"),
                 mean_decay=("mean_abs_decay", "mean"))
            .round(3))
    print(f"\n-- {h} --")
    print(g.to_string())

winner = arm.iloc[0]
print("\n=== SELECTED ARM ===")
print(winner[cols].to_string())
print("\nits fits:")
w = d[d["arm"] == winner["arm"]].sort_values("ode_r2", ascending=False)
fcols = [c for c in ["marker", "seed", "ode_r2", "dt_r2", "rel_mae", "decay_err", "order", "formula"] if c in w.columns]
print(w[fcols].round(3).to_string(index=False))
