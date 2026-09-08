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
   (`dt_r2`). The two correlate ~0.5 and disagree per context: PTPN7 (s43) has a
   derivative R² of exactly **0.000** and an integrated R² of **0.910**. Three of the
   eight best contexts in the final run have derivative R² of exactly 0.000 (PTPN7,
   MAP2K2, PTPN2).
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

## Naming convention

One rule, applied to every figure, table and caption:

- An **overexpression context** is a transfected gene and carries its HGNC-approved
  symbol. Of the 40 contexts in the source data exactly one is not on its approved
  symbol: `PIP5K3` is a previous symbol for **PIKFYVE** (HGNC:23785). The data keys stay
  `PIP5K3` — the raw columns, the per-shard run directories and every deposited artefact
  use it, and renaming them would break the joins — so the mapping lives in
  `figures/display_names.py` and is applied on the way into a panel or table.
- A **readout** is a phospho-antibody, not a gene, and keeps its protein common name.
  Four of the seven are pan-isoform (one antibody, two genes), so a single HGNC symbol
  would misdescribe the reagent. Table S11 carries the mapping instead. This is also the
  rule that settles `p90RSK` vs `RPS6KA1` for the same node: it is `p90RSK` everywhere.

Antibody targets are from the Key Resources Table of Lun et al., Mol Cell 2019, the
source of the measurements. Two readouts were previously drawn ambiguously: `RAF` is
**RAF1** (the antibody is p-c-RAF Ser259), and `PDK1` is **PDPK1** — HGNC `PDK1` is
pyruvate dehydrogenase kinase 1, a different gene. Both are recorded in Table S11 rather
than relabelled, so the panels keep the pathway names a signalling reader expects.

`display_names.py` is the only place the mapping is written down; figure scripts import
`gene_label`/`gene_labels` from it rather than carrying their own copy.

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
| architecture grid | `runs/sparse_neural_ode/arch_grid/` | `experimental_sparse_node_arch_grid` | ~54 GPU-h |
| custom-loss ablation | `runs/custom_loss_ablation/` | `experimental_custom_loss_ablation` | hours |
| participation ratios | `runs/sparse_neural_ode/participation_ratio_variants.csv` | `experimental_node_participation_ratios` | minutes |
| driver read-out | `runs/sparse_neural_ode/driver_readout/` | `experimental_node_driver_readout` | seconds |
| regulariser comparison | `runs/sparse_neural_ode/regulariser_comparison/` | `experimental_node_regulariser_comparison` | seconds |
| SR vs linreg | `runs/comparisons/sr_vs_linreg/` | `experimental_compare_sr_vs_linreg` | seconds |
| SR vs Neural ODE | `runs/comparisons/sr_vs_neural_ode/` | `experimental_compare_sr_vs_neural_ode` | seconds |
| Fig. 5 exemplar panels | `runs/paper_figures/` | `experimental_fig5_perbin_trajectories` | seconds |
| Figs. S2 / S3 / S4 | `runs/paper_figures/supplementary/` | `experimental_fig_s2_threshold_examples`, `..._fig_s3_node_selection`, `..._fig_s4_sr_vs_linreg_matched` | seconds |
| threshold defence | `runs/paper_figures/supplementary/` | `experimental_threshold_intuition`, `..._threshold_calibration`, `..._threshold_error_tradeoff` | seconds (calibration ~3 s: 1.92M null draws) |
| Tables S8–S11 | `runs/paper_figures/supplementary/` | `experimental_supplementary_tables` | seconds |

`snakemake --config enzyme_model=experimental -- experimental_results` builds every panel
and table the manuscript prints. Nothing under `paper_figures/` is hand-run: a figure that
needs different arguments gets another output on its rule and the arguments in `params`,
never a shell invocation. Each figure script writes a PDF beside its PNG, and both are
declared — an undeclared sibling is exactly how the printed Fig. 4E/F and the pipeline's
own copy of it drifted onto different metrics for the v5 draft.

The two expensive PySR stages ran on NEMO via the sharding helpers in
`src/pipelines/experimental/sweeps/`. Their outputs are committed, and their rules
declare inputs as `ancient()` so a touched timestamp cannot offer to spend thousands of
CPU-hours rebuilding a recorded artefact. Delete an output to re-run deliberately. The
architecture grid is `ancient()` for the same reason.

Two inputs are recorded runs that no rule reproduces, and both are marked `ancient()`
where they are consumed:

- **`pysr_ood_final/fits/`** — the per-(context, seed) tree the frozen-config run was
  recovered from by `scripts/migrate_v5_layout.py`, out of a cluster scratchpad. Figs. 5
  and S2 draw individual held-out bins from it. `experimental_pysr_ood_final` now declares
  the equivalent per-seed trajectory CSVs, so a genuine re-run regenerates the data in the
  `seeds/` layout, but not this tree.
- **`sparse_neural_ode/arch_grid.csv`** — the checked-in copy records **1 to 39 contexts
  per cell**, so its cells are not strictly comparable to each other. The rule above runs
  every cell over every context, which is the comparison Table S9 claims to report; expect
  the marginals to move the first time it is run.

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
| **Fig. 4D** headline | 8/32 overexpression contexts ≥ 0.6 and 7/8 controls (15/40 overall, 37%); median 4 of the ten model inputs. Perturbation median held-out R2 0.034, controls 0.798 (raw-scored) | `pysr_ood_final/metrics/success_rate_summary.csv` |
| **Fig. 4D** baseline | 32 perturbations, k=10: SR better 14, linreg 3, neither 15; 82% of the 17 where either works; 8 vs 3 solved; p = 0.035; mean 0.253 vs 0.141 | `paper_figures/fig_4d_sr_vs_linreg.png` |
| **Fig. 4E/F** sparsity panel | 32 perturbations; NN 18, PySR 8, both 6, neither 12. Both boxes are the participation ratio of the mean absolute Jacobian over the ten model inputs: ✓ 3.23 vs ✗ 4.19, p = 0.032; pooled 3.23 vs 4.38, p = 0.026; SR 2.68 on its 8 (2.46 on the 6 both solve), SR-vs-network paired Wilcoxon p = 0.22, drawn | `paper_figures/fig_4ef_accuracy_and_parsimony.png`, inputs in `pysr_ood_final/metrics/panel_inputs_32perturbations.csv` |
| **Fig. 5** exemplars | two panels: PIKfyve/PIP5K3 s43 0.937 (3/3 seeds) and PTPN7 s43 0.910 (2/3) — the two highest held-out R², 0.18 clear of third. ALPK2 s43 0.729 (1/3) dropped to SI | ranking in `pysr_ood_final/metrics/exemplar_ranking.csv`; the pair is pinned as `fig5_exemplars` in `common.smk` and drawn by `experimental_fig5_perbin_trajectories` |
| **Fig. S1** | custom-loss penalty ablation | `runs/custom_loss_ablation/` |
| **Threshold figures** | three panels defending R² = 0.6: `fig_threshold_intuition` (what a curve at 0.5/0.6/0.7 looks like), `fig_threshold_calibration` (≈9,000× above a wrong-context null), `fig_threshold_error_tradeoff` (error vs contexts kept) | `paper_figures/supplementary/` via `experimental_threshold_intuition`, `..._threshold_calibration`, `..._threshold_error_tradeoff` |
| **Fig. S2** | trajectories bracketing the R² cutoff: the 6 perturbation fits below (0.490–0.576) and 6 above (0.616–0.747), every fit in that band, controls excluded, normalised axes | `paper_figures/supplementary/fig_s2_threshold_examples.png` (`experimental_fig_s2_threshold_examples`) — see the note below on the second, older rule |
| **Fig. S3A / Table S8** | 40 contexts, scored against the raw measurements: median held-out R2 L1 0.760, L21 0.784, PathReg 0.726; 25, 25 and 24 above 0.6. Third variant is PathReg (Aliee et al. 2022, arXiv:2210.14672), NOT C-NODE — C-NODE is the baseline that paper compares against | `sparse_neural_ode/regulariser_comparison/l21_vs_l1_vs_cnode.png` |
| **Fig. S3B** | λ ladder, elbow retains λ = 3.0 | `sparse_neural_ode/lambda_sweep_summary.csv` |
| **Fig. S3C / Table S9** | 54-cell architecture grid; retained hd=64, 4 layers, lr=3e-3, tanh | `sparse_neural_ode/arch_grid.csv` |
| **Fig. S4** | 32 perturbations, k=4 (complexity-matched): SR better 14; 8 vs 1 solved; p = 0.0032; mean 0.253 vs 0.045 | `paper_figures/supplementary/fig_s4_sr_vs_linreg_matched.png` |
| **Fig. S5** | sparsity panel on all 40 contexts as a robustness check: ✓ 3.59 vs ✗ 4.04, p = 0.18; pooled 3.60 vs 4.35, p = 0.069; SR 2.78, SR-vs-network paired Wilcoxon p = 0.0049 | `paper_figures/supplementary/fig_s5_parsimony_all40.png`, inputs in `pysr_ood_final/metrics/panel_inputs_all40.csv` |
| **Table S10** | all 72 candidate arms, train-ranked | `pysr_config_sweep/metrics/table_s10_config_sweep.csv` |
| **Table S11** | readout key: protein name -> HGNC symbol, phospho-site, antibody clone | `figures/display_names.py` via `experimental_supplementary_tables` |

**Fig. S2 does not show a discontinuity at 0.6, and must not be captioned as if it did.**
It shows a gradual transition with the cutoff sitting inside it. The adjacent pair across
the line — PTPN2 s44 at 0.576 (rejected) and EGFR s43 at 0.616 (accepted) — is a visual
wash, and EGFR clears the cutoff on a 1.2× fold readout, the narrowest in the panel. The
earlier version of this figure appeared to show a clean transition, but that was a control
confound: 6 of the 9 panels at and above 0.6 were controls and 0 of 6 below, and controls
are trivially easy. The rebuilt figure is perturbations-only and complete within its band,
which is honest but weaker. **The load-bearing justification for the value 0.6 is not this
figure** — it is `experimental_threshold_calibration` (see below).

### Calibration of the 0.6 cutoff

`fig_threshold_calibration.png` (`experimental_threshold_calibration`) is a **single
panel**: fraction of contexts clearing each candidate cutoff, real vs a wrong-context
null, log y. One number on it — the separation at 0.6.

**The null.** Each held-out bin is scored against an integrated trajectory belonging to a
*different* context, then passed through the same median-across-bins and
best-of-three-seeds selection the reported statistic uses. That last step matters: without
it the null is a single draw compared against a maximum of three, which inflates the
separation. 1,920,000 draws (60,000 replicates x 32 contexts), seed 20260807, ~3 s.

| cutoff | real contexts | null |
|---|---|---|
| 0.4 | 11/32 | 0.105% |
| 0.5 | 9/32 | 0.0168% |
| **0.6** | **8/32 (25%)** | **0.0027% (52/1.92M)** |
| 0.7 | 5/32 | 0.0004% |
| 0.8 | 2/32 | 0 |

**Do not caption where the null curve ends.** That point is the Monte Carlo resolution
limit, not a property of the data. An earlier 128,000-draw version terminated at R² = 0.63
and looked like a cliff exactly at the cutoff; at 1.92M draws the tail runs to 0.796. The
claim is the *vertical gap at 0.6* (~9,000x), which is stable. The quoted ratio rests on
~50 null counts, so its Poisson error is ~15% — quote it rounded, never as four digits.

**Insensitivity** is a Methods sentence, not a panel: median participation ratio 3.2–3.5
(solved) vs 4.3–4.4 (unsolved), a gap of 0.9–1.4 units across the whole range 0.35–0.73.
The script prints this as a diagnostic.

### What the cutoff buys (`fig_threshold_error_tradeoff.png`)

`experimental_threshold_error_tradeoff` plots the held-out prediction error of everything
a cutoff accepts — median across bins of the mean absolute residual, as a % of that bin's
measured dynamic range — against contexts kept.

| cutoff | contexts | fits | error | 95% CI |
|---|---|---|---|---|
| 0.4 | 11 | 20 | 15.5% | [13, 17] |
| 0.5 | 9 | 17 | 14.9% | [12, 16] |
| **0.6** | **8** | **11** | **12.5%** | **[9, 14]** |
| 0.7 | 5 | 8 | 9.1% | [7, 13] |
| 0.8 | 2 | 5 | 7.0% | [6, 8] |

**This is a smooth monotone tradeoff with no knee**, and the intervals overlap heavily
across 0.5–0.7. Do not describe it as a plateau or a sweet spot. An earlier version binned
fits *marginally* (within ±0.05 of each cutoff) and appeared to show a 0.55–0.75 plateau —
an artefact of bands holding 3–5 fits, where mean and median disagreed by 11 points
(at 0.6: median 17%, mean 28%). Summarise cumulatively and bootstrap over **fits**, not
bins: bins within a fit share a formula and are not independent.

The usable claim is the accuracy statement, which is cutoff-independent in form: at 0.6 the
accepted fits track held-out trajectories to within ~12% of their measured range.

**Three caveats that belong in any caption.**

1. This justifies 0.6 as **safe and inconsequential, not optimal.** Nothing in the data
   distinguishes it from 0.55 or 0.65, and the null floor alone would license anything
   above ~0.35.
2. **A scale-matched null is much harsher.** Allowing the donor trajectory to be rescaled
   to the recipient's level and range, so only *shape* is tested, is passed by 27% of
   draws at 0.6 (5% only at R² ≈ 0.91). p-ERK trajectories are generically rise-and-fall,
   so most of the discrimination comes from predicting the absolute level at an unseen
   dose. That is the intended claim, but the number should be reported rather than left
   for a reviewer to find.
3. **Never draw a bootstrap CI on the median participation-ratio difference.** It spans
   zero at every cutoff from 0.35 to 0.73, which reads as "no effect anywhere" and
   contradicts the rank test the paper reports (p = 0.015 at 0.6). Report the rank test.
   This fragility is tracked under Known limitations.

### What a curve at each cutoff looks like (`fig_threshold_intuition.png`)

`experimental_threshold_intuition` is the figure for the reviewer who arrives assuming
0.7 is the standard R² bar. Three blocks — R² ≈ 0.5, 0.6, 0.7 — each drawing the **8
held-out curves closest in R² to that value**, measurement and prediction together, one
panel per curve, each scaled to its own measured range.

| block | typical miss |
|---|---|
| R² ≈ 0.5 | 20% of the measured range |
| **R² ≈ 0.6** | **18%** |
| R² ≈ 0.7 | 15% |

The 0.6 and 0.7 blocks are hard to tell apart; the 0.5 block is visibly looser. That is
the whole argument: **0.7 buys about three points of accuracy.** It does *not* show 0.6
beating 0.7, and it must not be captioned as if it did.

**Selection is the whole game here, and four framings were tried before this one.** Record
of what failed, so it is not repeated:

1. **Nearest the cutoff** — shows the cutoff's *worst* admitted case. Honest, but the 0.6
   panels look poor next to 0.7, and against a 0.8 column they look much worse. Cropping
   to 0.5/0.6/0.7 fixes that, but the crop was chosen *after* seeing the unfavourable
   version, which is a frame doing argumentative work.
2. **Quantiles of the accepted population** (25th/50th/75th of everything a bar admits) —
   more favourable and arguably more principled, since a 0.6 bar admits 100 curves with
   median R² 0.82. **Rejected: it is misleading.** The panels showed curves at 0.70/0.82/
   0.91 under a header reading "bar at R² = 0.6" — no panel was near 0.6 — and the
   impression survives any caption. It also cannot distinguish bars, since a 0.5 bar
   admits a median of 0.74 and looks equally reassuring.
3. **Every curve in a ±0.05 band** (78 panels) — the most airtight, no selection at all,
   but unreadable at page width.
4. **Every curve in a ±0.02 band** — blocks come out lopsided (18/7/12) because the R²
   distribution is not uniform.

Nearest-8 is symmetric across the three levels, has no free parameter beyond N, and
answers the question actually asked. An earlier version also carried one `used` marker set
through the level loop, which made the picks depend on the order the levels were listed
in; selection is now independent per block.

Bins with less than 1.5× measured fold-change are excluded throughout. R² is scale-free,
so a near-flat readout scores well while showing nothing — EGFR bin 45 spans 1.15× and
flattered the 0.60 column before the filter went in.

**Caveat that belongs in the caption.** This is a *per-bin* figure and the cutoff is
applied to a fit's *median across bins*. It therefore says nothing about contexts whose
median hides a catastrophic bin — PTPN2 reports 0.70 with one bin at R² = −8.0, TBK1 0.64
with two negative bins. See Known limitations.

**Two rules draw a threshold figure, and they are not the same figure.** The printed Fig.
S2 is `experimental_fig_s2_threshold_examples`: it reads `pysr_ood_final/fits/`, so its
examples come from the reported OOD run that the 0.6 cutoff is actually applied to. The older
`experimental_pysr_r2_threshold_sanity` writes
`plots/metrics/lines/pysr_r2_threshold_sanity.png` from the **in-distribution** reference
run and carries two hardcoded `--override-example` picks. Keep it as the in-distribution
diagnostic it is; do not print it as Fig. S2.

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
trajectory again. They are **trivially easy**, not failures: median held-out R² 0.798,
7/8 above threshold, against 0.034 and 8/32 for the perturbations. They are excluded from
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
- **The parsimony effect is fragile to how it is summarised.** The rank test gives
  p = 0.015 at the 0.6 cutoff, but a bootstrap interval on the *median difference* in
  participation ratio spans zero at every cutoff from 0.35 to 0.73 (n = 32, discrete-ish
  PR values). The point estimate is stable (gap 0.9-1.4 units throughout); the precision
  is not. Report the rank test, and do not draw a median-difference CI.
- **Persistence baseline: mostly tautological, but check the headline eight.** Predicting
  every held-out bin with the highest *training* dose's trajectory reaches median R² 0.80
  and clears 0.6 on 18/31 perturbations, against 0.07 and 8/31 for PySR (paired Wilcoxon
  p = 0.0025). The aggregate comparison is **not** meaningful: on contexts with no dose
  response, persistence wins by construction, and all six contexts where SR beats it are
  ones where persistence scores exactly 0 — i.e. SR wins exactly where there is dose
  response to recover. That part is the expected result, not a problem.
  The part that is *not* tautological: of the **8 headline contexts, 6 are also cleared by
  persistence** (PIP5K3 0.97, ALPK2 0.95, MAP2K2 0.94, TBK1 0.86, AKT3 0.86, PTPN7 0.90).
  Only **PTPN2 (P1), EGFR and PTPN7** strictly beat it. So "extrapolates to an unseen dose"
  is demonstrated on 3 of 8, not 8 of 8 — PIP5K3, the Fig. 5 flagship, is beaten by
  copying the nearest dose (0.971 vs 0.937). Adjacent GFP bins agree at median R² = 0.96,
  which is why persistence is strong. Diagnostic only, deliberately **not wired** into
  `experimental_results`: `figures/plot_persistence_baseline.py`. Keep it to hand for
  review; decision taken 7 Aug 2026 to leave the headline framing unchanged.
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


## Corrections applied 7 Aug 2026 (run regenerated mid-session)

`pysr_ood_final` was rebuilt during the 7 Aug working session — 491 files, the whole
`fits/` tree plus every metrics file. **The previous copy is unrecoverable**: `data/` is
gitignored and nothing under it was tracked, so the regenerated run is now the only run
and is authoritative. PySR is not reproducible across environments, so every non-zero
value shifted slightly. The headline survived (8/32 perturbations ≥ 0.6, 7/8 controls);
these did not, and were corrected:

| claim | was | now |
|---|---|---|
| Fig. 5 exemplar 1 | PIP5K3 s43 0.942 | PIP5K3 s43 **0.937** (3/3 seeds) |
| Fig. 5 exemplar 2 | PTPN7 s42 0.935 | PTPN7 s42 **0.907** (2/3); s43 0.910 |
| gap to third | 0.19 | 0.18 (third is ALPK2 0.729) |
| PTPN7 integrated R² (§conventions) | 0.935 | 0.910 |
| top-8 with derivative R² = 0.000 | four | **three** (PTPN7, MAP2K2, PTPN2) |
| control median held-out R² | 0.846 | 0.798 |
| perturbation median held-out R² | 0.031 | 0.034 |
| controls n ≥ 0.8 | 5 | 4 |

**PTPN7's two best seeds are a tie and the pin stays on s42.** s42 scores 0.9071 and s43
0.9097 — 0.003 apart, both rounding to 0.91, a margin well inside PySR's run-to-run
variation. `exemplar_ranking.csv` lists s43 first purely on that margin, and the pin was
briefly changed to s43 on 7 Aug before being reverted: **s42 is the seed the printed Fig. 5
panel and the Results text use**, and the draft already discloses the tie ("seed 42, tied
with seed 43"). Do not repin on row order alone. The rule is only discriminating when the
gap exceeds run-to-run noise; here it does not.

The values themselves did move with the rebuild, and the manuscript already reflects that
(0.94 and 0.91). This is the "individual exemplars are a lottery" hazard under Known
limitations — **re-check the pinned pair against `exemplar_ranking.csv` after any rebuild,
but change it only if the ranking moves by more than ~0.01.**

Not re-derived after the rebuild, and still carrying pre-rebuild numbers: the Fig. 4E/F
and Fig. S5 sparsity panels, Fig. S3A/Table S8 regulariser comparison, and the
SR-vs-linreg comparisons. Their inputs (`panel_inputs_*.csv`) were regenerated, so
**their reported values must be re-checked before submission.**

## Corrections applied 6 Aug 2026

Three defects in `figures/plot_parsimony_tradeoff.py` changed every number in the sparsity panel:

1. **`_parse_pysr` truncated marker names** — `r"Group:\s*(\S+)"` stops at the first space, so
   `DUSP10 (P2)` was keyed as `DUSP10` and, with `PTPN2 (P1)`, silently fell out of the
   NN/PySR intersection. The panel ran on 30 markers, not 32. (The sibling
   `plot_sr_vs_linreg_ood.py` always used `(.+)` and was unaffected.)
2. **`--exclude-marker` defaulted to `["untransfected1"]`** — hardcoded, and because the action is
   `append` it could not be overridden from the command line. This is the origin of v4's "39
   overexpression contexts"; it was an argparse default, not an analysis decision.
3. **SR's parsimony number was a participation ratio**, computed from the formula's Jacobian at
   the training mean, and is now an exact count of variables in the expression. The two boxes
   therefore measure different kinds of quantity (exact vs effective), so SR-vs-network p-values
   are reported to stdout but no longer drawn as brackets. A `py_pr > 0` filter that silently
   dropped AKT3 (unparseable Jacobian) was also removed.

Denominator convention now used throughout: **32 perturbation contexts** for every method
comparison, because a construct-free control has no dose-response and cannot inform which
perturbation responses admit a compact description; **all 40** for the success rate, with the
composition stated, and as the Fig. S5 robustness check. Controls are not visually distinguished
in Fig. 4D — they span held-out R² 0.000–0.953 and one fails, so they are genuine fits rather
than free points.
