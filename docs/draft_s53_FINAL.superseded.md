# The main claim, shown

Specification: the 21 contexts where the sparse neural ODE cleared held-out R2 > 0.6
(ODE-integrated, test split, best of three network seeds) — the manuscript's existing gate.
Symbolic regression: frozen production configuration, seeds 42/43/44, success at held-out
ODE-integrated R2 >= 0.6. Complexity: effective dependency count and interacting-pair count,
both read from the network. All P values two-sided.

## THE CLAIM

Where symbolic regression fails, the dynamics require more measured variables — quantified
independently of symbolic regression by the neural ODE.

| | failed | solved | P (two-sided MWU) | AUC |
|---|---|---|---|---|
| **dependency count** | **4.15** | **3.10** | **0.0082** | **0.841** |
| **interacting pairs** | **6.97** | **3.73** | **0.0182** | **0.809** |

n = 21 (11 failed, 10 solved).

## It is not an artefact of the seed-selection rule

Replacing best-of-three-seeds (which selects on held-out data) with the leak-free
train-parsimony rule, 13 failed / 8 solved:

| | failed | solved | P | AUC |
|---|---|---|---|---|
| dependency count | 3.97 | 3.12 | 0.0397 | 0.774 |
| interacting pairs | 6.49 | 3.71 | 0.0424 | 0.774 |

## It is graded, not just a group difference

Recovery frequency — the fraction of the three seeds that clear threshold, which uses no seed
selection at all — falls with closure dimension: rho = -0.543, P = 0.0116 (dependency count);
rho = -0.516, P = 0.0179 (interacting pairs). Every leave-one-out estimate stays negative
(-0.67 to -0.51).

## There is a ceiling

Mean recovery frequency is 0.44 up to three variables, 0.44 between three and four, and zero
above four. None of the six contexts above four variables recovered a law in any seed (DYRK2,
DYRK3, MAPK1, MAPK3, MET, PRKACA); ten of fifteen at or below four did. Fisher P = 0.012.

## It is not explained by data quality

Including the network's own held-out accuracy as a covariate leaves the complexity term
significant (logistic beta = -1.02, P = 0.033). The pattern is not the trivial one in which
poorly-measured contexts defeat both methods together.

## It is robust across the specification space

84 pre-enumerated specifications — three learnability gates including none, seven definitions
of symbolic recovery including three that use no threshold, two complexity measures, with and
without control contexts: **83 of 84 run in the predicted direction (99%), median rho = -0.36.
All 19 specifications reaching P < 0.05 do so in the predicted direction. None contradicts it.**

## Results sentence (drop-in)

Restricting to the 21 contexts where the network generalised, contexts where symbolic
regression failed required more measured variables than those it solved (4.15 vs 3.10;
two-sided Mann-Whitney P = 0.008; AUC = 0.84) and carried more interacting input pairs (6.97
vs 3.73; P = 0.018; AUC = 0.81). The relationship is graded: the fraction of seeds recovering
a generalising law falls with the closure dimension (Spearman rho = -0.54, P = 0.012), and no
context closing over more than four measured variables recovered a law in any seed, against
ten of fifteen at or below four (Fisher P = 0.012). It is not a restatement of how well the
network itself performed, since including the network's held-out accuracy as a covariate
leaves the complexity term significant (P = 0.033), and it holds in 83 of 84 pre-enumerated
specifications of the gate, success criterion and complexity measure.

## Two corrections to v7

- AUC: the draft says ~0.75; recomputed it is **0.84**.
- Solved-group mean: the draft says 2.90; recomputed it is **3.10**.

## Stated scope

One configuration applied uniformly to all contexts. Claims sufficiency, not necessity: five
of the six above-ceiling contexts yield generalising laws under a different operator basis, so
individual failures are not proof that no compact law exists. Criterion is explained variance,
justified in Methods on the grounds that law recovery is a statement about capturing the shape
of the dynamics rather than average magnitude error.
