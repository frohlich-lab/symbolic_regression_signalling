"""Step 9 — merge the fanned-out runs and test the Path A claims.

`submit_nemo.sh` writes one output tree per marker so array tasks never contend
for a file. This merges them and then runs the four tests that Path A actually
turns on. Each is stated here as a falsifiable prediction, with the outcome that
would sink it, because a claim that cannot fail is not evidence.

  A. **PR is calibrated.**  dPR/dk_true ~ 1 on real (correlated) inputs.
     Fails if PR saturates -> the published contrast cannot mean what it says,
     and k* replaces PR throughout.

  B. **k* separates SR-fail from SR-success.**  A direct, correlation-robust
     input count should do at least as well as PR did, on all 40 contexts and
     without best-of-seed selection.
     Fails if k* does not separate -> the dimensionality story is wrong, and
     what distinguishes the contexts is something else.

  C. **Hidden state is present where SR fails.**  Latent augmentation should
     help MORE in SR-fail contexts. The interaction is the evidence; a main
     effect of latent dimension is just added capacity.
     Fails if the gain is flat or uniform -> drop the hidden-state language.

  D. **SR-failure is not redundant with the classical toolkit.**  Adding the
     SR-derived quantity to a model of observability rank + non-identifiable
     directions should improve prediction of SR outcome.
     Fails if it does not -> the novelty case rests on practical advantages
     (no model specification, works on partial data), which is a weaker but
     honest claim.

Usage:
    python collect.py --root data/experimental/runs/reducibility
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import SUCCESS_THRESHOLD, default_outdir, write_csv  # noqa: E402


def merge(root: Path, step: str, filename: str) -> pd.DataFrame:
    """Concatenate <root>/<step>/<marker>/<filename> across markers."""
    paths = sorted((root / step).glob(f"*/{filename}"))
    if not paths:
        single = root / step / filename
        paths = [single] if single.exists() else []
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"  [warn] {p}: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    print(f"  {step}/{filename}: {len(out)} rows from {len(paths)} files")
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(x[ok]).rank().to_numpy()
    rb = pd.Series(y[ok]).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def perm_p(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearman(x, y)
    if not np.isfinite(obs):
        return np.nan
    y = np.asarray(y, float)
    null = [abs(spearman(x, rng.permutation(y))) for _ in range(n)]
    return float((np.sum(np.array(null) >= abs(obs)) + 1) / (n + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=default_outdir())
    ap.add_argument("--master", type=Path, default=None)
    args = ap.parse_args(argv)
    root = args.root
    master_path = args.master or (root / "reducibility_master.csv")

    print("merging fanned-out runs...")
    calib = merge(root, "calibration", "pr_calibration.csv")
    abl = merge(root, "ablation", "input_ablation.csv")
    lat = merge(root, "latent", "latent_node.csv")
    ident = merge(root, "identifiability", "identifiability.csv")
    resc = merge(root, "rescue", "rescue_demo.csv")
    arch = merge(root, "arch", "arch_sweep.csv")

    for name, df in (("calibration", calib), ("ablation", abl), ("latent", lat),
                     ("identifiability", ident), ("rescue", resc),
                     ("arch", arch)):
        if not df.empty:
            write_csv(df, root / f"merged_{name}.csv")

    # SR outcome per context, without best-of-seed where possible.
    sr = pd.DataFrame()
    if master_path.exists():
        m = pd.read_csv(master_path)
        m["sr_ok"] = m["pysr_ode_r2"] >= SUCCESS_THRESHOLD
        sr = (m.groupby("marker")
                .agg(sr_recovery_freq=("sr_ok", "mean"),
                     sr_r2_best=("pysr_ode_r2", "max"),
                     is_control=("is_control", "first"))
                .reset_index())
        sr["sr_success"] = sr["sr_r2_best"] >= SUCCESS_THRESHOLD

    verdicts = []

    # --- A. is PR calibrated? ------------------------------------------------
    print("\n" + "=" * 72)
    print("A. Is the participation ratio calibrated against known dimensionality?")
    if calib.empty:
        print("   [no calibration data]")
    else:
        for shuffled, g in calib.groupby("shuffled"):
            curve = g.groupby("k_true")["pr_measured"].mean()
            if len(curve) >= 2:
                slope = float(np.polyfit(curve.index, curve.values, 1)[0])
                prec = float(g["precision_at_k"].mean(skipna=True))
                print(f"   shuffled={bool(shuffled)}: dPR/dk_true = {slope:.3f}   "
                      f"mean precision@k = {prec:.3f}")
                print("     " + "  ".join(f"k={k}:PR={v:.2f}"
                                          for k, v in curve.items()))
                verdicts.append({
                    "claim": f"A: PR calibrated (shuffled={bool(shuffled)})",
                    "statistic": f"dPR/dk_true={slope:.3f}",
                    "supported": slope > 0.5,
                    "note": "PR tracks true dimensionality" if slope > 0.5
                            else "PR is saturated — use k*, not PR"})

    # --- B. does k* separate SR outcomes? ------------------------------------
    print("\n" + "=" * 72)
    print("B. Does k* (measured input count) separate SR-fail from SR-success?")
    if abl.empty or sr.empty:
        print("   [no ablation data]")
    else:
        # Reuse the ablation script's own definition: k* selected on validation,
        # anchored to the best achievable over k (held-out R^2 is not monotone
        # in k, so anchoring to the full model is not meaningful).
        from input_ablation import summarise
        kdf = summarise(abl, 0.05)
        if not kdf.empty:
            agg = kdf.groupby("marker")["k_star"].mean().reset_index()
            j = agg.merge(sr, on="marker", how="inner")
            write_csv(j, root / "k_star_vs_sr.csv")
            rho = spearman(j["k_star"], j["sr_recovery_freq"])
            p = perm_p(j["k_star"], j["sr_recovery_freq"])
            fail = j[~j.sr_success]["k_star"].mean()
            ok = j[j.sr_success]["k_star"].mean()
            print(f"   mean k*  SR-fail={fail:.2f}  SR-success={ok:.2f}  "
                  f"(n={len(j)})")
            print(f"   k* vs SR recovery frequency: rho={rho:.3f}, P={p:.4f}")
            verdicts.append({"claim": "B: k* separates SR outcomes",
                             "statistic": f"rho={rho:.3f}, P={p:.4f}",
                             "supported": bool(np.isfinite(p) and p < 0.05),
                             "note": f"k* fail={fail:.2f} vs success={ok:.2f}"})

    # --- C. hidden state where SR fails? -------------------------------------
    print("\n" + "=" * 72)
    print("C. Does latent state help MORE where SR fails? (the interaction)")
    if lat.empty or sr.empty:
        print("   [no latent data]")
    else:
        base = (lat[lat.latent_dim == 0][["marker", "seed", "test_r2"]]
                .rename(columns={"test_r2": "r2_d0"}))
        g = lat.merge(base, on=["marker", "seed"], how="left")
        g["gain"] = g["test_r2"] - g["r2_d0"]
        best = (g[g.latent_dim > 0].groupby(["marker", "seed"])["gain"]
                .max().reset_index())
        per_marker = best.groupby("marker")["gain"].mean().reset_index()
        j = per_marker.merge(sr, on="marker", how="inner")
        write_csv(j, root / "latent_gain_vs_sr.csv")
        if not j.empty:
            fail = j[~j.sr_success]["gain"].mean()
            ok = j[j.sr_success]["gain"].mean()
            rho = spearman(j["gain"], j["sr_recovery_freq"])
            p = perm_p(j["gain"], j["sr_recovery_freq"])
            print(f"   mean best latent gain  SR-fail={fail:.4f}  "
                  f"SR-success={ok:.4f}  (n={len(j)})")
            print(f"   gain vs SR recovery frequency: rho={rho:.3f}, P={p:.4f}")
            print("   (want: gain concentrated in SR-fail contexts, i.e. rho<0)")
            verdicts.append({
                "claim": "C: hidden state concentrated where SR fails",
                "statistic": f"gain fail={fail:.4f} vs success={ok:.4f}, "
                             f"rho={rho:.3f}, P={p:.4f}",
                "supported": bool(fail > ok and np.isfinite(p) and p < 0.05),
                "note": "the interaction, not the main effect of d"})

    # --- D. non-redundancy with the classical toolkit ------------------------
    print("\n" + "=" * 72)
    print("D. Does SR-failure add information over observability / identifiability?")
    if ident.empty or sr.empty:
        print("   [no identifiability data]")
    else:
        agg = (ident.groupby("marker")
               .agg(obs_rank=("obs_rank", "mean"),
                    n_flat=("n_flat_directions", "mean"),
                    curv=("profile_curvature_mean", "mean"),
                    max_vif=("max_vif", "mean"),
                    pr=("pr", "mean")).reset_index())
        j = agg.merge(sr, on="marker", how="inner").dropna()
        write_csv(j, root / "identifiability_vs_sr_merged.csv")
        try:
            import statsmodels.api as sm
            if len(j) > 12 and j["sr_success"].nunique() == 2:
                y = j["sr_success"].astype(float)
                # n_flat saturates at 10/10 on this data (no variance),
                # so the continuous profile curvature is the usable comparator.
                cl = ["obs_rank", "curv", "max_vif"]
                m1 = sm.Logit(y, sm.add_constant(j[cl])).fit(disp=0)
                m2 = sm.Logit(y, sm.add_constant(j[cl + ["pr"]])).fit(disp=0)
                lr_p = float(m2.compare_lr_test(m1)[1])
                print(f"   pseudo-R2  classical={m1.prsquared:.3f}  "
                      f"+PR={m2.prsquared:.3f}   LR P={lr_p:.4f}")
                verdicts.append({
                    "claim": "D: SR/PR adds over classical diagnostics",
                    "statistic": f"LR P={lr_p:.4f}, pseudo-R2 "
                                 f"{m1.prsquared:.3f}->{m2.prsquared:.3f}",
                    "supported": lr_p < 0.05,
                    "note": "if unsupported, novelty rests on practical "
                            "advantages, not new information"})
        except Exception as exc:
            print(f"   [logit skipped: {exc}]")

    # --- rescue demonstration ------------------------------------------------
    print("\n" + "=" * 72)
    print("E. Rescue demonstration (drop the predicted variable, then restore it)")
    if resc.empty:
        print("   [no rescue data]")
    else:
        s = (resc.groupby("condition")
             .agg(node_r2=("node_test_r2", "mean"), pr=("pr", "mean"),
                  k_star=("k_star", "mean"), sr_r2=("sr_test_r2", "mean"),
                  n=("marker", "size"))
             .reindex(["intact", "drop_top", "drop_control", "restored"])
             .dropna(how="all"))
        print(s.round(3).to_string())
        if {"intact", "drop_top", "restored"} <= set(s.index):
            moved = s.loc["drop_top", "pr"] - s.loc["intact", "pr"]
            back = abs(s.loc["restored", "pr"] - s.loc["intact", "pr"])
            ctrl = (s.loc["drop_control", "pr"] - s.loc["intact", "pr"]
                    if "drop_control" in s.index else np.nan)
            print(f"\n   PR shift on dropping top input : {moved:+.3f}")
            print(f"   PR shift on dropping a control : {ctrl:+.3f}")
            print(f"   PR residual after restoring    : {back:.3f}")
            verdicts.append({
                "claim": "E: rescue signature is specific and reversible",
                "statistic": f"top={moved:+.3f}, control={ctrl:+.3f}, "
                             f"residual={back:.3f}",
                "supported": bool(np.isfinite(moved) and moved > 0
                                  and (not np.isfinite(ctrl) or moved > 2 * abs(ctrl))
                                  and back < abs(moved) / 2),
                "note": "specificity requires top >> control"})

    # --- verdict table -------------------------------------------------------
    if verdicts:
        v = pd.DataFrame(verdicts)
        write_csv(v, root / "path_a_verdicts.csv")
        print("\n" + "=" * 72)
        print("PATH A VERDICTS")
        print("=" * 72)
        for _, r in v.iterrows():
            mark = "SUPPORTED    " if r["supported"] else "NOT SUPPORTED"
            print(f"  [{mark}] {r['claim']}\n      {r['statistic']}\n      {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
