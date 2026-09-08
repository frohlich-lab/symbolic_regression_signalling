# v5 -> new version: exhaustive technical changelog

Everything needed to reformat v5 into the new version. Numbers reproduce from
`data/experimental/runs/pysr_ood_final/`, `data/experimental/runs/reducibility/l21_hl4_local/`
and `data/experimental/runs/analysis_2026-08-11/`.

---

## 0. NAMING FIX

v7's Results calls the sparsity robustness check "the path-regularised C-NODE". This is wrong:
v5 and both Methods sections call it **PathReg** (a Jacobian sparsity penalty, with its own
citation), whereas C-NODE is an unrelated architecture. Use **"a path-regularised (PathReg)
variant"** everywhere.

---

## 1. THE SPECIFICATION (state verbatim in Methods)

### 1.1 Symbolic regression
- PySR, frozen production configuration: `niterations` 1400, `populations` 30,
  `population_size` 30, `maxsize` 26, `parsimony` 0.8, binary operators `+ - * /`, no unary
  operators.
- Fixed in code, not per-run: `deterministic=True`, `parallelism="serial"`, `procs=0`,
  `batching=True`, `annealing=True`, `verbosity=0`, `loss_scale="log"`.
  Note on the last: `loss_scale` governs only how Pareto-front scores are computed for
  `model_selection`, not the objective being minimised, and `"log"` is PySR's default — so it is
  not a deviation and does not log-transform anything.
- `model_selection` left at PySR's default `"best"`: the reported expression is chosen off the
  Pareto front by log-loss improvement per unit complexity. This interacts with the penalties
  below — infeasible expressions sit on the front at loss ~1000, which can pin the selected
  expression near complexity 5. State this.
- Library defaults elsewhere: `ncycles_per_iteration`, `maxdepth`, `constraints`,
  `nested_constraints`, `complexity_of_*`, `warm_start`, `turbo`, `precision`,
  `timeout_in_seconds`, `early_stop_condition`.
- Custom Julia loss. Data term is a plain **mean squared error on the raw derivative target**,
  `sum((f0 - y)^2)/length(y)`. No log transform is applied to the target or to the loss. Hard
  constraints, each returning `base + 1000.0`: **GFP must appear in every
  expression** (`REQUIRE_GFP = true`), **p-ERK must appear in every expression**;
  `MISSING_PENALTY = REDUNDANT_PENALTY = 1000.0`, `DEP_TOL = 1e-4`. Stability penalties
  `WRONGSIGN_P_LINEAR = WRONGSIGN_INV_LINEAR = 1000.0`. Conditioning penalty **off**
  (`COND_THRESH = 0.0`). Tolerances `EPS_REL = LIN_TOL_REL = 1e-3`. The 1000.0 penalties are
  load-bearing, not nominal: replacing them with `Inf` breaks the search entirely, because finite
  penalties preserve ordering among infeasible candidates.
- Data: `markers_per_minute_fit.csv`, `--dataset-mode per_minute`. Target `p-ERK1-2_dt`, the
  **raw** derivative (ratio to d/dt of raw p-ERK = 1.0000; a log target would be 2.31). Features
  are all columns except `{p-ERK1-2_dt, p-MEK1-2_dt, marker, timepoint, GFP_bin}`, giving ten:
  `p-ERK1-2_fit`, `GFP`, `p-ERK1-2_min`, `p-MEK1-2_fit`, `p-MEK1-2_min`, `p-RAF_fit`,
  `p-p90RSK_fit`, `p-MAPKAPK2_fit`, `p-PDK1_fit`, `p-MKK3-6_fit`.
- Split `top_gfp_bins` (NOT the argparse default `random_bins`), test size 0.2.
- Seeds 42/43/44. **Seed chosen by training R2; its held-out score reported.**
- Flags left off in production: `--standardise-target`, `--no-require-gfp`,
  `--conditioning-threshold`, `--pysr-sample-weighting` (the last is a silent no-op — the custom
  loss ignores sample weights).

### 1.2 Success criterion
Held-out ODE-integrated R2 > 0.6, law integrated forward per bin against raw measurements at the
six acquired timepoints. **13 of 40 contexts.**

Two aggregation details that v5's Methods states incorrectly or omits, both to be fixed:

- **It is the MEDIAN across held-out bins, not the mean.** v5 says "bin-averaged", which describes
  a mean; the code computes `np.median(integ_r2_vals)`. The median is the defensible choice and the
  reason should be given: per-bin R2 is unbounded below, and a single poorly-integrated bin would
  dominate a mean. Verifiable from the stored `dt_r2_mean_bins` and `dt_r2_median_bins` columns
  in `pysr_ood_final/seeds/seed_*/metrics/`: across 40 markers the mean-over-bins derivative R2
  averages -23.3 (worst context -576.1, DUSP10 (P2)) against -19.9 for the median, and the two
  aggregations differ by a median of 0.153. Individual seeds reach -1711. Per-bin values
  themselves are not retained, so the mean-versus-median switch cannot be quantified for the
  integrated R2, where only the median is written.
- **Values are floored at zero** (`ode_r2_median = max(0.0, ode_r2_median)`). A reported 0.000
  therefore means R2 <= 0, not R2 = 0. This is undocumented in v5 and matters for interpretation,
  since many contexts are reported at exactly 0.000.
- A mean-over-bins variant cannot be computed from the stored outputs: only the median is written,
  per-bin values are not retained, and this run has no trajectories file. Quantifying the switch
  would require re-running `compute_marker_integration.py` with trajectory output enabled.

### 1.3 Learnability gate
The **21** contexts where the sparse neural ODE cleared held-out R2 > 0.6 on the same split,
taken as the **best of three network seeds**. All primary statistics are computed on these 21.

### 1.4 Complexity
Effective **dependency count**: inputs whose mean absolute input Jacobian exceeds **0.15** of the
largest, **averaged over the three network seeds**. Interacting pairs: off-diagonal Hessian
entries above 0.25 of the largest off-diagonal, restricted to the selected inputs, also averaged
over seeds.

### 1.5 Aggregation principle (new; state in Methods)
Existence questions take the best of three seeds — does a compact law exist, does a reduced
description exist. Quantities being estimated are averaged over seeds — complexity. Justifies the
asymmetry a referee would otherwise query.

### 1.6 Statistics
Two-sided throughout. Unit of analysis is the context; permutations shuffle contexts, never
seeds.

---

## 2. CHANGES FROM v5, BY CATEGORY

### 2.1 Complexity metric: participation ratio -> dependency count
v5 used the participation ratio of the mean absolute input Jacobian. Replace with the thresholded
count at 0.15. **Justification:** calibration against synthetic dynamics of known driver count
gives slope **0.947** for the count versus **0.688** for the PR, which under-reads dimensionality
by ~30%; the 0.15 cutoff is itself selected on that synthetic ground truth. Every PR value in the
text, Fig. 4E and Table S8 is superseded.

### 2.2 SR seed selection: held-out -> train
v5 selected the best of three seeds on held-out R2 (selection on test data). Now the seed is
chosen on **training** R2 and its held-out score reported. Changes SR successes from 8 (of 32) to
**13 of 40**.

### 2.3 Sample
v5 analysed 32 overexpression contexts. All 40 are now fitted and the primary analysis runs on the
**21** that clear the learnability gate. Controls are included in the sample without separate
discussion.

### 2.4 Quadrant counts
| | v5 | new |
|---|---|---|
| SR clears 0.6 | 8 | **13 of 40** |
| network clears 0.6 | 18 | **21 of 40** |
| both | 6 | **9** |
| network only | 12 | **12** |

### 2.5 Core statistics
| | v5 | new |
|---|---|---|
| complexity, failed vs solved | 4.19 vs 3.23 (PR) | **4.06 vs 3.11** (dependency count) |
| P | 0.032 | **0.019** |
| AUC | not given | **0.806** |
| graded test | not present | **rho = -0.543, P = 0.011** (seed recovery frequency vs dependency count, n = 21 contexts) |
| interacting pairs | not present | **6.58 vs 3.78, P = 0.039** |
| SR-vs-network paired comparison on both-solved | 3.23 vs 2.46, Wilcoxon P = 0.22 | recompute on the new metric or drop |

### 2.6 Three v5 sentences to rewrite
| v5 | problem | replacement |
|---|---|---|
| Results: "dynamics requiring more interacting variables **than a rate law can compactly express**" | SR recovers laws at 4.33 dependencies (TBK1) and writes five-variable expressions | state the association; add the quadratic mechanism |
| Abstract: "suggests that **no low-dimensional description exists** within the observed variables" | the network closes them; the wider configuration search recovers compact laws in several failing contexts | "the dynamics close over more of the measured variables" |
| Discussion: "The **limitation is therefore the number** of interacting variables the dynamics require, not SR's ability to identify a functional form" | association only; distributions overlap | "failures concentrate where the dynamics close over more variables" |

### 2.7 New main-text content
- **Interaction density is saturated.** Within each fit the interacting-pair count is at or near
  the maximum its dependency count allows: median exactly the maximum, mean **0.970** of it,
  never above (61 gated fits with k >= 2). Consequence: interaction terms grow as the square of
  the closure dimension, so roughly one extra variable corresponds to roughly twice the terms
  (6.6 vs 3.8). No context admits an additive or separable rate law.
- **The calibration sentence** (slope 0.947) moves into the main text as the justification for the
  metric.
- **The gate is earned, not mentioned:** paragraph 1 must state that the dependency count is read
  from the network's learned law and is therefore a meaningful readout of the dynamics only where
  that law demonstrably extrapolates — then restrict the analysis to the 21.

### 2.8 Wording that must not appear
- "more variables than a compact expression can hold" (refuted by TBK1 at 4.33).
- dependency count and interacting pairs as independent lines of evidence.
- any claim that a failure proves no compact law exists.
- a ceiling at 4.3 dependencies: it holds only under the production configuration and seeds,
  while the train-selected configuration search recovered a compact law at 4.67 (PRKACA).
- "C-NODE" for the PathReg variant.

---

## 3. FIGURE 4E

- Legend counts: both-solved n = 6 -> **9**; SR box n = 8 -> **13**; network-only n = 12 -> 12.
- Boxes become **dependency counts**, not participation ratios.
- Bracket statistic becomes **P = 0.019**.
- If the panel plots SR held-out R2 against network held-out R2, both axes must use the same seed
  aggregation, or SR is disadvantaged by construction. v5 used best-of-three for both; the SR axis
  now uses the train-selected seed, so state the asymmetry in the legend or use matched
  aggregation for the scatter only.

---

## 4. SI — INCLUDE

1. **Ungated analysis, all 40 contexts.** 4.09 vs 3.36, P = 0.009; graded rho = -0.467,
   P = 0.002. Frame as: the gate does not manufacture the effect. Note that outside the gate the
   dependency count is read from a network that failed to extrapolate, which is why the gated
   analysis is primary.
2. **Gate-definition sensitivity.** With the network seed chosen on validation R2 instead of
   best-of-three, the gate holds 17 contexts and the contrast is 4.12 vs 3.11, P = 0.051 (graded
   rho = -0.487, P = 0.050) — same direction and effect size, reduced power.
3. **Jacobian sensitivity-threshold sweep.** The contrast holds at thresholds 0.05, 0.10, 0.15 and
   0.20 and vanishes at 0.25-0.30, where the mean count degenerates toward two and stops
   discriminating. Gated dependency-count P values: 0.017 (0.05), 0.099 (0.10), 0.019 (0.15),
   0.073 (0.20), 0.056 (0.25), 0.589 (0.30).
4. **Contexts where symbolic regression outperforms the network.** **Four** contexts under the
   primary train-selected rule: PIP5K3 (SR 0.928, network 0.568), untransfected1 (0.794, network
   **0.000**), PTPN2 (P1) (0.698, network **0.000**), TBK1 (0.636, network 0.594). In two of
   them the network scores exactly zero while SR reaches 0.70 and 0.79. Report as a point in
   favour of the method.
   NOTE: AKT3 is NOT in this set. Its best-train seed is 43 (train 0.9241), whose held-out R2 is
   0.000, so AKT3 is an SR failure under the primary rule; it appears as a success only under
   seed selection that peeks at held-out data.
5. **PathReg robustness check** (renamed from "C-NODE"): the contrast holds under a structurally
   different sparsity penalty. Recompute the two group means on the dependency count rather than
   the PR before quoting.
6. **Existing v5 SI that must be updated:** Table S8 (PR -> dependency count), Fig. S3C's
   "72-arm PySR configuration grid" (the sweep is now 144 configurations), and any denominator
   stated over 32 contexts.

## 5. SI — EXCLUDED (your decision), with the consequences

Each of these was cut. The consequence is listed so the choice is deliberate.

1. **Covariate control** (network accuracy as a covariate; beta = -1.02, P = 0.033).
   *Consequence:* delete the clause "it persists when the network's own held-out accuracy is
   controlled for" from the paragraph — it would reference SI content that does not exist. The
   alternative explanation "poorly measured contexts defeat both methods" is then unaddressed.
   This is the most likely referee question.
2. **Measurement-coverage filter.** *Consequence:* delete "and when contexts with sparse
   measurement coverage are excluded". MST1R is classified on a single held-out bin and
   untransfected1 on one held-out bin with two training bins; both are visible in the metrics
   table.
3. **Pairs redundancy note** (per-seed density, partial rho = +0.16 P = 0.33, convexity note).
   *Consequence:* the paragraph reports two P values, 0.019 for dependencies and 0.039 for pairs,
   on quantities that are a deterministic transform of one another. Without the note this reads as
   double-counting. Either restore the note or drop the pairs P and keep "equivalently, 6.6
   against 3.8 interaction terms".
4. **Specification curve** (84 analyses, 83 in the predicted direction). *Consequence:* the
   "why this gate and this criterion?" objection has no pre-emptive answer.
5. **Parsimony-budget ablation** (already in v5: relaxing 3.0 -> 0.1 rescues 6 of 24 and loses 5;
   18 of 24 admit no generalising expression at any budget). *Consequence:* the search-adequacy
   argument disappears — nothing in the paper then rules out "the search stopped too early".
6. **Interaction density as a standalone SI result.** *Consequence:* the main text asserts the
   density figures (median exactly the maximum, mean 0.970) with no supporting distribution.

---

## 6. UNCHANGED FROM v5 — no edits needed

`top_gfp_bins` split with the highest 20% of bins held out; the network's additional 20%
early-stopping block (60/20/20 against SR's 80/20); scoring against raw binned measurements at the
six acquired timepoints, equal-weight bin averaging; R2 threshold 0.6; L21 lambda = 3.0 chosen by
elbow rule; network architecture (hidden dim 64, four layers, tanh, lr 3e-3, Adam, weight decay
1e-5, batch 64, 200 epochs, patience 20, Dopri5); complexity averaged over three network seeds
(v5 already did this, Table S8); the 50 GFP quantile bins and within-bin population averaging;
the population-average and non-autonomous scope statements.
