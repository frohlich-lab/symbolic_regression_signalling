# Draft SR-MM v5 — experimental results provenance

Every experimental number, figure and table in the v5 draft, mapped to the artefact it
comes from and the pipeline stage that builds it. Reproduce the cheap stages with:

```bash
snakemake --use-conda -j1 --config enzyme_model=experimental experimental_metrics
snakemake --use-conda -j1 --config enzyme_model=experimental experimental_results
```

## The three conventions that must not be mixed

These caused real errors during the v5 work and every stage below is pinned to one
choice of each.

1. **Integrated, not derivative, R².** Reported performance is the R² of the numerically
   integrated p-ERK trajectory (`ode_integ_r2_median`), never the derivative-target fit
   (`dt_r2`). The two correlate ~0.5 and disagree per context: PTPN7 has a derivative R²
   of exactly **0.000** and an integrated R² of **0.935**. Four of the eight best
   contexts in the final run have derivative R² of exactly 0.000.
2. **Out-of-distribution split.** The top 20% of GFP bins are held out
   (`--test-split-policy top_gfp_bins`), so scores measure dose *extrapolation*. The
   in-distribution alternative reverses the headline comparison — linreg solves 10/32 at
   k=10 and PySR holds no advantage (p = 0.455) — so any artefact on random bins is not
   a valid baseline. This is why the baseline directory is named `linreg_ood`.
3. **Best of three seeds, not seed-averaged.** Per context, the maximum over seeds
   42/43/44.

Two further standing traps:

- **`neural_ode_diffrax_metrics.csv` holds train, val and test rows.** A bare `.max()`
  reads training accuracy. Filter `split == "test"`. This once fabricated an entire
  "compressibility" result before it was caught.
- **Never zero-fill a missing context.** A `fillna(0.0)` made the eight controls look
  like uniform PySR failures when their real median is 0.85 — they were simply absent
  from the input table. Both figure scripts now drop unmatched contexts with a warning.

## Stage map

| stage | directory | Snakemake rule | cost |
|---|---|---|---|
| data prep | `data/experimental/processed/functional_groups/` | `experimental_marker_inputs` | minutes |
| PySR config sweep | `runs/pysr_config_sweep/` | `experimental_pysr_config_sweep` | ~3,000 CPU-h |
| config selection | same | `experimental_pysr_select_config` | seconds |
| frozen-config run | `runs/pysr_ood_final/` | `experimental_pysr_ood_final` | ~70 CPU-h |
| summary + exemplars | same | `experimental_pysr_ood_summary` | seconds |
| linreg baseline | `runs/linreg_ood/` | `experimental_select_k_linreg` | minutes |
| sparse Neural ODE | `runs/sparse_neural_ode/l21_lam3/` | `experimental_neural_ode_diffrax_ood_l21` | hours (GPU) |
| regulariser variants | `runs/sparse_neural_ode/{l1,pathreg}/` | `..._ood_l1`, `..._ood_cnode` | hours (GPU) |
| λ sweep | `runs/sparse_neural_ode/lambda_sweep/` | `experimental_sparse_node_lambda_sweep` | hours (GPU) |
| custom-loss ablation | `runs/custom_loss_ablation/` | `experimental_custom_loss_ablation` | hours |
| driver read-out | `runs/sparse_neural_ode/driver_readout/` | `experimental_node_driver_readout` | seconds |
| regulariser comparison | `runs/sparse_neural_ode/regulariser_comparison/` | `experimental_node_regulariser_comparison` | seconds |
| SR vs linreg | `runs/comparisons/sr_vs_linreg/` | `experimental_compare_sr_vs_linreg` | seconds |
| SR vs Neural ODE | `runs/comparisons/sr_vs_neural_ode/` | `experimental_compare_sr_vs_neural_ode` | seconds |

The two expensive PySR stages ran on NEMO via the sharding helpers in
`src/pipelines/experimental/sweeps/`. Their outputs are committed, and their rules
declare inputs as `ancient()` so a touched timestamp cannot offer to spend thousands of
CPU-hours rebuilding a recorded artefact. Delete an output to re-run deliberately.

## The frozen configuration

```
--max-iterations 1400 --populations 30 --population-size 30 --max-size 26 --parsimony 0.8
--binary-operators + - * /        # division kept, NO unary operators
--test-split-policy top_gfp_bins --test-size 0.2
```

Selection rule, fixed before results were seen: among the 72 grid arms that keep division
and use no unary operators, take the highest **median integrated R² on the training GFP
bins**. This arm is rank **1/72** (median train R² 0.888). Held-out data plays no part —
`select_pysr_config.py` also prints the held-out-best arm, which is a *different* one
(`it1400_p60_ps60_ms22_par01_unone_bdiv`, median 0.687), as the rule requires.

Division is retained because saturating, Michaelis-Menten-like rate laws need it. The
stability penalties stay at their defaults in every arm: they encode the
dynamical-admissibility prior (d[p-ERK]/dt decreasing in p-ERK), so relaxing them to gain
R² would be tuning away the prior.

The configuration was applied **unchanged** to all 40 contexts, including the 34 the
sweep never saw. That separation is load-bearing: an earlier iteration reported
sweep-derived numbers as general and they did not survive contact with the unseen
contexts (median OOD R² 0.044 on development contexts, 0.000 on the 26 never seen).

## Figure and table map

| v5 artefact | numbers | source |
|---|---|---|
| **Fig. 4D** headline | 8/32 contexts ≥ 0.6, median 4 variables (range 2–6) | `pysr_ood_final/metrics/success_rate_summary.csv` |
| **Fig. 4D** baseline | SR better 19/40; 8 vs 3 solved; p = 0.035; mean 0.351 vs 0.178 | `comparisons/sr_vs_linreg/heldout_r2_k10.png` |
| **Fig. 4E** + sparsity panel | 30 contexts; NN 18, PySR 7, both 6, neither 11; PR 3.34 vs 5.20, p = 0.0071; SR vs NN-✓ p = 0.063 | `comparisons/sr_vs_neural_ode/accuracy_and_drivers.png` |
| **Fig. 5** exemplars | PIKfyve/PIP5K3 s43 0.942 (3/3 seeds), PTPN7 s42 0.935 (2/3), ALPK2 s43 0.734 (1/3) | `pysr_ood_final/metrics/exemplar_ranking.csv` |
| **Fig. S1** | custom-loss penalty ablation | `runs/custom_loss_ablation/` |
| **Fig. S2** | trajectories near the R² threshold | `experimental_pysr_r2_threshold_sanity` |
| **Fig. S3A / Table S8** | L21 0.617, L1 0.629, pathreg 0.610; p = 0.19 and 0.29 | `sparse_neural_ode/regulariser_comparison/l21_vs_l1_vs_cnode.png` |
| **Fig. S3B** | λ ladder, elbow retains λ = 3.0 | `sparse_neural_ode/lambda_sweep_summary.csv` |
| **Fig. S3C / Table S9** | 54-cell architecture grid; retained hd=64, 4 layers, lr=3e-3, tanh | `sparse_neural_ode/arch_grid.csv` |
| **Fig. S4** | SR better 21/40; 8 vs 1 solved; p = 0.0032 | `comparisons/sr_vs_linreg/heldout_r2_k4.png` |
| **Table S10** | all 72 candidate arms, train-ranked | `pysr_config_sweep/metrics/table_s10_config_sweep.csv` |

Fig. 5's exemplar rule is the **dual-threshold** one: a context counts as solved only if
the *same fit* clears R² ≥ 0.6 on both the training doses and the held-out ones, and
solved contexts are ranked by held-out R². The "both" clause does real work — ranked on
held-out R² alone, third place is MAP2K2 at 0.747 on a **training** R² of 0.029, a model
that never fitted the data it was trained on and whose fan has no peak at all. Note this
ranking does read held-out values, so it is test-based selection; the quantitative claim
is the distribution, not the exemplars. Two rejected alternative rules are kept in
`archive/superseded/adhoc_scripts/` with the reasoning in
`runs/pysr_ood_final/exemplar_selection/README.md`.

## Controls

The eight control contexts (`untransfected1-4`, `FLAG-GFP1-4`) have no construct, so no
GFP dose-response, so "extrapolating to an unseen dose" means predicting the same
trajectory again. They are **trivially easy**, not failures: median held-out R² 0.846,
7/8 above threshold, against 0.031 and 8/32 for the perturbations. They are excluded from
the headline rate — counting them would lift it from 8/32 to 15/40 with no dose-response
biology behind it — and from the sparsity panel, which does not reproduce with them in.

The pooled statistical claim is robust either way and is in fact *stronger* with controls
included at their real values (p = 0.0044 vs 0.0071). The narrower ✓-vs-✗ panel test is
the fragile one (p = 0.032), which is why the Results quote the pooled version.

## Known limitations, recorded

- **Seed robustness is weak.** Only PIP5K3 clears both thresholds on all three seeds;
  PTPN7 on two; ALPK2, PTPN2, TBK1 and AKT3 on exactly one each. ALPK2 is a successful
  individual fit, not a reproducible result, and the Fig. 5 caption says so.
- **PySR and the Neural ODE do not share a training set.** Same test bins (top 20% GFP),
  but PySR trains on the remaining 80% and the network on 60% (it holds out 20% for
  validation). This favours PySR on held-out accuracy.
- **`--pysr-sample-weighting` is a silent no-op.** The custom Julia loss never reads
  `dataset.weights`, so the flag is inert. A weighting result from this pipeline is
  untested, not disproven.
- **PySR is not reproducible across machines.** Identical seeds reproduce bit-exactly on
  one machine but not across environments, so exemplar choice must be made on the run
  being shipped. Aggregate statistics are stable; individual exemplars are a lottery.
- **Config B (maxsize 22) was tested and rejected.** On all 40 contexts it ties the
  frozen config (paired Wilcoxon p = 0.62) and destroys the sparsity panel (p = 1.000),
  because its extra successes are exactly the high-PR contexts.
- **Every input is a fitted curve.** Inputs are rise-and-fall fits and the target is that
  curve's analytic derivative: ~2,989 per-minute rows derive from ~294 measurements.
- **Table S9's grid was scored on validation R² only**, which tracks held-out performance
  weakly (Fig. S3C). Read the retained architecture as a reasonable default, not an
  optimum for extrapolation. This caveat does *not* apply to the λ ladder, where
  validation and held-out R² fall together (ρ = 1.00).

## What was removed, and why

`archive/superseded/` holds ~8 GB of run outputs that v5 does not use, with a manifest.
The notable entries: `top_gfp_ood_pysr` (the v4-era PySR run superseded by the frozen
config), `seeds`/`aggregated` (the in-distribution run), `neural_ode` (the
full-input/matched-input baseline v5 explicitly retires), `random_forest*` (absent from
v5 entirely), and `select_k` (the in-distribution linreg that must not be used as an OOD
baseline). `archive/superseded/adhoc_scripts/` holds the scratchpad-era analysis scripts,
each annotated with what replaced it.
