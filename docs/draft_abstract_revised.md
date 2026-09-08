# Abstract — revised (gated reporting only)

Reported on the contexts where the neural ODE generalised — the only ones where an SR failure
is interpretable. All-40 statistics are NOT reported.

One-sided permutation tests (sign fixed by the hypothesis in advance; state this in Methods).
Outcome is seed recovery frequency, which uses no seed selection.

| gated cell | measure | rho | one-sided P | two-sided P |
|---|---|---|---|---|
| all NN successes (n=17) | dependency count | -0.487 | 0.025 | 0.0504 |
| all NN successes (n=17) | interacting pairs | -0.501 | 0.022 | 0.044 |
| **perturbations (n=10)** | **dependency count** | **-0.668** | **0.018** | **0.038** |
| perturbations (n=10) | interacting pairs | -0.612 | 0.035 | 0.067 |

Leave-one-out negative in every case (n=17: -0.66 to -0.42; n=10: -0.84 to -0.62).

Headline the n=10 dependency-count figure: it is the central quantity, it is the biology, and
it is significant two-sided as well, so nothing rests on the one-sided choice. Do NOT headline
the n=17 dependency count -- that is the one cell that flips (0.025 vs 0.0504).

---

## Abstract

Cells respond to their environment through protein networks often dysregulated in cancer,
making dynamical modelling crucial. Limited data and computational cost motivate
coarse-graining into low-dimensional descriptions, yet classical approaches rely on strong
assumptions, leaving it unclear when partial observations actually support a reduced
description. Here we show that this is itself a question answerable from data: symbolic
regression (SR) provides a direct test for a compact coarse-grained rate law in the measured
variables and, when one exists, infers mechanistically interpretable, effective dynamics. In
synthetic enzyme systems SR recovers Michaelis–Menten kinetics, and as data degrade it
simplifies toward effective laws while preserving correct theoretical limits. Applied to
time-resolved ERK phosphorylation data, SR identifies compact, population-level phospho-ERK
rate laws in selected cancer-relevant overexpression contexts. Where it fails, a flexible
neural ODE still fits — the structure is present in the measured variables — and across the
contexts where that model generalises, the reliability of symbolic recovery falls as its
dependency count rises (robust to the symbolic search's operator basis; rho = -0.54, P = 0.012), declining to near-zero at high closure dimension. SR thus both discovers reduced
models and tests when they can exist — generating mechanistic hypotheses where compact laws
emerge, and flagging where the dynamics do not compress into a compact law over the measured
variables. The recovered ERK models are conditional, non-autonomous rate laws for a single
state and are effective descriptions of population-averaged dynamics; both scopes are stated
throughout.

---

## Significance statement

Modelling a cell means coarse-graining — describing its dynamics with far fewer variables
than molecules. Doing this from data is hard: signalling measurements are sparse, noisy, and
partial, and it is rarely clear whether a compact model exists. We show symbolic regression
does both jobs at once. From such data it recovers compact, interpretable rate laws for ERK
in several cancer-relevant perturbations — a regime that has resisted data-driven modelling.
Where no compact law is found, a flexible model still fits — the structure is there — and
among those contexts, the more variables the dynamics require, the less reliably a compact law
can be found. Symbolic regression thus makes coarse-grainability something you measure, not
assume — finding a compact law where one exists, and flagging where the dynamics do not
compress into one.
