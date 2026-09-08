# Results paragraph — final

Gate: the 21 contexts where the sparse neural ODE cleared held-out R2 > 0.6 (ODE-integrated,
test split, best of three network seeds). Symbolic regression: frozen production
configuration, seeds 42/43/44. All P values two-sided. Ensemble statistics use the
configuration and seed selected on training data alone, with held-out performance reported.

---

To ask whether symbolic regression's failures reflect the method or the absence of an
accessible reduced description, we used a sparse neural ODE as a **learnability control**.
Because it is scored under the same out-of-distribution split — holding out the highest-GFP
bins — reaching R2 >= 0.6 requires genuine extrapolation to stronger perturbations than any
seen in training. In the 21 contexts where it succeeds, the measured variables demonstrably
support a reduced description that generalises, so a failure of symbolic regression there
cannot be attributed to an absence of structure. Our central quantity is the effective
**dependency count** — the number of measured inputs the learned rate law depends on, that
is, the dimension at which the dynamics close. It is read from the network, which never sees
the symbolic search's objective, operator basis or output, so it is independent of the
quantity it is used to explain.

Contexts where symbolic regression failed closed over more measured variables than those it
solved (4.15 vs 3.10; Mann-Whitney P = 0.008; AUC = 0.84) and carried more interacting input
pairs (6.97 vs 3.73; P = 0.018; AUC = 0.81). The relationship is graded rather than a group
difference alone: the fraction of seeds recovering a generalising law falls with the closure
dimension (Spearman rho = -0.54, P = 0.012), and every leave-one-out estimate remains
negative. It is not an artefact of choosing the best of three seeds on held-out data, since
selecting the seed on training parsimony instead preserves it (3.97 vs 3.12, P = 0.040). Nor
is it a restatement of how well the network itself performed: including the network's
held-out accuracy as a covariate leaves the complexity term significant (logistic
beta = -1.02, P = 0.033), so this is not the trivial pattern in which poorly-measured
contexts defeat both methods together. Mean recovery frequency is 0.44 for contexts closing
over four or fewer variables and zero above four — none of the six contexts above four
variables recovered a law in any seed, against ten of fifteen at or below (Fisher P = 0.012).

These failures are not the product of an insufficient search. Selecting the configuration and
seed on training data alone and reporting held-out performance, we searched a median of 21
configurations per context (up to 135) drawn from a grid varying iteration budget, population
structure, size limit, parsimony pressure and operator basis. Rescue saturates: raising the
searched space from 8 to 55 configurations per context — nearly sevenfold — rescues 0.6
additional contexts of eleven, and the set of contexts recovered stabilises by eight
configurations. What the enlarged search does buy is better training fits, not better
generalisation. For five of the seven contexts that remain unsolved, symbolic regression finds
compact expressions that describe the measured regime well (training R2 of 0.55 to 0.87) and
none that survives extrapolation to stronger perturbations (held-out R2 of 0.00 to 0.53); two
contexts, DYRK2 and RPS6KA6, admit no generalising expression among any configuration tested
even when selection is allowed to see the held-out data.

Together: where symbolic regression finds a law, the dynamics coarse-grain compactly and the
recovered law is the demonstration. Where it does not, the dynamics still close over the
measured variables — the network closes them and extrapolates — but over more of them, and the
compact expressions that do exist hold only within the observed regime. Compact
coarse-grainability is therefore a property that can be measured rather than assumed, and on
these data it is regime-dependent: a compact description of the measured range does not imply
a compact law across the perturbation range.

## Scope, stated once

The primary analysis fixes one configuration and applies it uniformly, which is what makes
recovery a property of the context rather than of the search. The claim is sufficiency, not
necessity: a high closure dimension precluded recovery in every context tested, while four of
the eleven gated failures are recovered by the train-selected ensemble (DUSP7 R2 = 0.79,
untransfected4 0.78, PRKACA 0.73, DYRK3 0.64) and are reported as such. Individual failures
are therefore not evidence that no compact law exists. The success criterion is explained
variance rather than relative error, because law recovery is a statement about capturing the
shape of the dynamics rather than about average magnitude error; a relative-error criterion
admits models that track the trajectory mean without recovering its dynamics.

## For SI

Specification curve across 84 pre-enumerated analyses — three learnability gates including
none, seven definitions of symbolic recovery including three threshold-free, two complexity
measures, with and without control contexts: 83 of 84 run in the predicted direction (median
rho = -0.36); all 19 reaching P < 0.05 do so in the predicted direction; none contradicts it.
Interaction density is saturated across all 40 contexts (~1.0; 1.08 where SR succeeded vs 1.02
where it failed, P = 0.96), so no context admits a separable rate law and the interacting-pair
count carries no information beyond the dependency count (partial rho = +0.16, P = 0.33).
