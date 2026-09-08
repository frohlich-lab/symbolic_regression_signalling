# Pre-registration: does dynamical complexity predict SR failure rate?

Written 2026-08-11, **before the maxsize=26 stratum existed**. State of the data at
writing, verifiable from the balanced grid (`pysr_fullsweep/manifest_bal.tsv`, job 51687791):

| stratum | cells available |
|---|---|
| maxsize 18 | 355 |
| maxsize 22 | 315 |
| **maxsize 26** | **9 of 240** |

## Why this test exists

Fifteen exploratory analyses of the "SR failure marks complex dynamics" claim were run on
2026-08-11. Pooled over configurations the effect is absent on perturbation contexts
(complexity beta=+0.006, p=0.97, with neural-ODE accuracy at beta=-0.48, p=0.0063). One
stratified cell came out significant -- maxsize=22, contexts with neural-ODE R2>0.6,
rho=+0.53 (no-division) and +0.58 (with-division). It is not reportable as-is: the
mechanistically predicted ordering (effect strongest where the size budget binds, i.e. at
maxsize 18) FAILED, no floor effect explains it (pass rates and outcome variance are
near-identical at 18 and 22), and the perturbation-only versions are null or negative.

That result is therefore treated as a hypothesis to be tested on held-out data, not a
finding. The maxsize=26 stratum is the held-out data.

## Specification, fixed in advance

- **Outcome** `frac_fail`: fraction of (config, seed) cells for that marker with
  `ode_integ_r2_median <= 0.6`, computed within the maxsize=26 stratum only.
- **Predictor**: `nn_inter`, interacting input pairs, from `interactions_offnorm.csv`.
  Unchanged from the manuscript. No re-derivation, no alternative complexity measure.
- **Population**: contexts with neural-ODE held-out R2 > 0.6 (the manuscript's learnability
  gate). SR R2 comes from train-parsimony seed selection -- never held-out selection.
- **Test**: Spearman rho, one-sided positive, pooled over operator sets. alpha = 0.05.

## Predictions, and the decision rule

1. **Primary.** maxsize=26, NN>0.6, pooled operators: **rho >= +0.40 with p < 0.05**.
2. **Secondary.** The perturbation-only subset must be at least directionally positive
   (rho > 0). If the whole effect is carried by the 8 control contexts, it does not support
   a claim about signalling dynamics.
3. **Consistency.** Both operator strata must be positive. A result present with division
   and absent without it is a cancellation artefact, not an expressivity limit.

**If 1 fails, the maxsize=22 result is declared noise and is not reported as a finding
anywhere in the manuscript.** No further complexity measures will be tried, no further
strata inspected, and no re-specification of the outcome or predictor is permitted. That
undertaking is the entire point of writing this down first.

If all three hold, this is a confirmatory result on data unseen at the time of writing, and
reportable as such -- with the exploratory history above disclosed.
