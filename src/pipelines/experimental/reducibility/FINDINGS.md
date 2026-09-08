# Findings — reducibility diagnostic, steps 1–3

Everything below is re-analysis of data already on disk (360 participation
ratios = 3 regulariser variants × 3 seeds × 40 contexts, plus per-seed PySR
held-out R²). No retraining. Reproduce with:

```bash
cd src/pipelines/experimental/reducibility && ./run_all.sh
```

Outputs in `data/experimental/runs/reducibility/`.

---

## 1. The manuscript's group sizes are reproducible — and identify a conditioning that must be stated

The published contrast reports n=12 SR-fail and n=6 SR-success contexts. Those
sum to 18, not 40, and the combination that reproduces them exactly is:

> **perturbations only** (8 controls excluded) **× conditional on Neural-ODE
> success** × **best-of-three-seeds**

The conditioning on Neural-ODE success is defensible — a participation ratio
read off a network that does not fit is not interpretable — but it is currently
implicit, and it is what makes the reported n so small. It should be stated in
the Methods, because a referee who reconstructs n=12/6 from a 40-context dataset
will otherwise ask why.

The same reconstruction confirms the headline success counts:
**8/32 perturbations and 7/8 controls** clear threshold under best-of-seed
(`quadrant_table.csv`, `convention=best`).

## 2. The participation-ratio contrast does not survive in the main-text variant

Under the manuscript's own subset (n=12/6), tested with a marker-level
permutation test:

| Variant | PR (SR-fail) | PR (SR-success) | Difference | 95% CI | Cohen's *d* | *P* |
|---|---|---|---|---|---|---|
| L1 | 5.23 | 4.40 | 0.83 | [0.19, 1.42] | 1.15 | **0.036** |
| **L21 (λ=3) — main text** | **4.85** | **4.44** | **0.41** | **[−0.56, 1.38]** | **0.39** | **0.442** |
| PathReg | 5.73 | 5.35 | 0.38 | [−0.24, 0.97] | 0.65 | 0.242 |

The reference model on which the paper's Neural-ODE argument rests gives
*P* = 0.44 with a confidence interval spanning zero. The published *P* = 0.032 is
matched in significance only by the L1 variant. (Absolute PR values also differ
from the published 4.19 / 3.23, which points to a different across-seed
aggregation — worth pinning down, and consistent with the other SI numbers
already noted as not reproducing.)

**This is not the same as "the effect isn't there."** The direction is
remarkably consistent: SR-fail contexts have higher PR in *every* variant,
*every* seed convention and *every* test run here. What fails is the claim to
statistical significance in the specific model the main text relies on.

## 3. The threshold-free test is stronger, and splits 2–1

Dropping the 0.6 dichotomisation in favour of PR against SR **recovery
frequency** (fraction of seeds clearing threshold) uses all 40 contexts:

| Variant | Spearman ρ | *P* (permutation) | Mixed model coef. (120 rows) | *P* |
|---|---|---|---|---|
| L1 | −0.48 | **0.002** | −0.55 | **0.007** |
| **L21 (λ=3)** | **−0.27** | **0.093** | **−0.38** | **0.104** |
| PathReg | −0.40 | **0.010** | −0.27 | 0.152 |

The mixed model is `node_pr ~ sr_ok + (1|marker)` over every per-seed row, so the
marker random intercept absorbs the 3-seed × 3-variant clustering rather than
pretending 120 independent observations.

Two of three regularisers give a significant, moderate negative association on
the full 40 contexts. The main-text variant is marginal in both tests.

## 4. The 0.6 threshold is doing more work than it should

Sweeping it from 0.30 to 0.85 (L21, best-of-seed, all 40 contexts): **no value
yields *P* < 0.12**, and the effect size wanders non-monotonically (*d* between
0.21 and 0.56). A threshold this load-bearing needs its sensitivity shown
(`pr_threshold_sweep.csv`); at present the result is not robust to it in the
reference variant.

## 5. Best-of-three-seeds inflates the SR success rate by 2–4×

Perturbation contexts only, matched objective:

| Seed convention | SR successes / 32 | Rate |
|---|---|---|
| best of 3 (published) | 8 | 25.0% |
| per-seed (no selection) | 11 / 96 obs | 11.5% |
| majority of 3 | 2 | **6.3%** |

Best-of-seed is legitimate for an *existence* claim ("a compact law can be
found"). It is not legitimate for the *failure* interpretation, which needs a
rate. Reporting single-seed recovery frequency alongside is the cheap fix, and
it is a substantial downward revision: 2 of 32 perturbation contexts yield a
compact law in a majority of seeds.

## 6. The 2 discordant contexts, resolved by reframing rather than explained away

The 2×2 under the matched objective (L21, best-of-seed, all 40):

|  | Neural ODE ✓ | Neural ODE ✗ |
|---|---|---|
| **SR ✓** | 12 | **3** |
| **SR ✗** | 13 | 12 |

The SR-succeeds/Neural-ODE-fails cell is 3 contexts (AKT3, PTPN2 (P1),
untransfected1) — a **7.5% Neural-ODE false-negative rate**, and 6.3% among
perturbations. These are only a contradiction if the Neural ODE is claimed to
upper-bound what is reducible. Recast as a *positive control for learnability*
(SR fails + network succeeds ⇒ real structure is present, so SR's failure is
neither noise nor merely an optimiser artefact), the whole table is reportable
and this cell becomes a measured error rate of the diagnostic.

Matching objectives matters here: under each method's native objective the cell
shrinks to 1 context (AKT3 only), so part of the discordance is an artefact of
comparing a derivative fit to a trajectory fit. Worth stating either way.

## 7. Controls beat perturbations, and it is worse per-seed

SR clears threshold for 87.5% of controls versus 25.0% of perturbations
(best-of-seed) — 75% vs 6.3% under majority. The method succeeds most where the
biology is least interesting. This needs a paragraph; it is visible in
`quadrant_table.csv` and a referee will find it.

## 8. The specification curve — 1,008 analyses of the same data

`multiverse.py` runs the contrast across every defensible combination of
regulariser × PR aggregation × threshold × seed convention × objective ×
conditioning. No new compute; all of it from the 360 PR values on disk.

The result separates into two parts that must not be conflated:

| | |
|---|---|
| **Direction** | positive in **97.3%** of specifications; significantly negative in **0 of 1,008**; median ΔPR **+0.45** (range −0.37 to +1.28) |
| **Significance** | clears P<0.05 in **20.6%**; median P **0.203** (range 0.0002–0.995); two analysts disagree about significance **32.8%** of the time |

Which choices move the answer (fraction of specifications significant):

| Factor | Spread |
|---|---|
| objective | matched 0.34 · native 0.04 |
| conditioning | all 0.35 · node_success 0.33 · perturbations 0.09 · **perturbations+node_success 0.05** |
| threshold | 0.5→0.12 rising monotonically to 0.7→0.33 |
| regulariser | L1 0.28 · PathReg 0.27 · **L21 0.07** |
| convention / PR aggregation | ~0.20 each — these barely matter |

Two things follow. First, the manuscript's conditioning is the *least* powered
cell in the entire space. Second, run under all nine equally-defensible
regulariser × PR-aggregation combinations, that same specification gives **P from
0.037 to 0.443** — the published result is one of nine, sitting at the 43rd
percentile of effect size. Typical in magnitude, simply under-powered.

**Variance decomposition of the participation ratio itself:** biological context
29.8%, regulariser 17.2%, seed 1.0%, residual 52.0%. The choice of penalty
accounts for more than half as much variation in PR as the biology does.

### The claim this supports

> SR-fail contexts consistently lean on more measured inputs — the effect is
> positive in 97.3% of 1,008 analytic specifications and never significantly
> negative — but it is small relative to specification noise, and any single
> (architecture, penalty, threshold) choice yields P anywhere from 0.0002 to
> 0.995 on identical data.

This is stronger evidence for the original claim than one *P*-value, and it
carries a second, more general finding: **Jacobian-based dependency attribution
in sparse neural ODEs is not identifiable on correlated biological data.**

Not yet in the multiverse: architecture. `arch_grid.csv` kept only aggregate
validation R², so PR cannot be recomputed per cell from it; `arch_sweep.py` is
re-running the cells. Depth is already known to flip this result, so the figures
above are a **lower bound** on the instability. 12 architectures sit within 0.05
validation R² of one another — that is the size of the choice being made
arbitrarily.

---

## What this means for the two paths

**The load-bearing statistic does not currently support the flagship claim in
the main-text model.** That is fixable, but not by rewording:

1. **Immediate (writing only).** Correct the abstract to the body's weaker
   claim. State the n=12/6 conditioning. Report per-seed recovery frequency
   alongside best-of-3. Report all three variants with CIs and effect sizes
   instead of one *P*. Add the threshold sweep. Discuss the controls gap and the
   Neural-ODE false-negative rate. This alone converts the section from
   "over-claimed" to "honestly reported" — and it is what Path B needs.

2. **To keep a strong claim, run steps 4–6.** The re-analysis has taken the
   existing evidence about as far as it goes; the remaining questions are
   empirical:
   - `pr_calibration.py` — is PR even calibrated on this data's input
     correlations? If it saturates, the metric can't carry the interpretation
     and `k*` should replace it. This is the single highest-value run.
   - `input_ablation.py` — `k*` as a direct, correlation-robust dimensionality
     measurement, and the drop-and-restore demonstration.
   - `latent_node.py` — positive evidence for hidden state, which is the claim
     the paper actually wants and which no observed-input statistic can supply.

The consistent direction across all nine variant × convention combinations is
genuine and worth reporting. Whether it can carry a general-interest headline
depends on steps 4–6, not on further re-analysis of what is already here.
