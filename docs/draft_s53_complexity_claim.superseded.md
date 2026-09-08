# §53 — Where symbolic regression fails, the dynamics are more complex

Draft, 2026-08-11. Statistics recomputed from the frozen production configuration
(`it1400_p30_ps30_ms26_par08_unone_bdiv`) with the seed chosen per marker by train
parsimony — the printed figure's own rule, which uses no held-out data to select. All
numbers below reproduce from `plot_parsimony_tradeoff.py` at the manuscript spec
(`--dependency-count --select-on-val --sr-parsimony --node-integrated-r2
--symbolic-interaction-box`).

---

## Text

Symbolic regression recovered a generalising rate law in 14 of 40 contexts. To ask what
distinguishes the contexts it solved from those it did not, we used the sparse neural ODE
as an independent readout of how complex each context's dynamics are — independent because
it is fitted without reference to symbolic regression, and because its two complexity
readouts are properties of the learned vector field rather than of any symbolic search. We
quantify complexity two ways: the effective dependency count, the number of inputs the
learned rate law depends on, and the number of interacting input pairs, the off-diagonal
curvature of that law.

Both separate the two groups. Where symbolic regression failed, the network's rate law
depended on more variables (4.15 vs 3.36 inputs, Mann–Whitney P = 0.0030) and contained
more interacting pairs (7.04 vs 3.78, a 1.86-fold difference, P = 0.023). Complexity ranks
failures above successes with an area under the curve of 0.757 (permutation P = 0.0029),
and the relationship is graded rather than threshold-like: across terciles of interaction
count the fraction of contexts yielding a generalising law falls from 0.54 to 0.43 to 0.08
(trend P = 0.013). At the top of the range the effect is absolute — none of the twelve most
complex contexts, all of them perturbations, yielded a law that generalised (0/12; a
failure run this long arises by chance with P = 0.0016).

This is not a restatement of how well the network itself performed. Including the network's
held-out accuracy as a covariate leaves the complexity term significant (logistic
β = −1.02, P = 0.033), so the pattern is not the trivial one in which noisy contexts defeat
both methods together.

The relationship is distributional, not deterministic. Three contexts with low measured
complexity nonetheless resisted symbolic regression (EGFR, 3.3 interacting pairs; DUSP16,
5.0; DUSP7, 6.0), and one of them, EGFR, shares its complexity coordinates with MAP2K2,
which was solved. Low-complexity failures are expected under a graded relationship — the
lowest tercile yields a law in only about half of contexts — and their existence bounds the
strength of the claim rather than contradicting it: high dynamical complexity is sufficient
to defeat the search, but it is not the only thing that can.

## Scope, stated explicitly

These statements characterise one symbolic-regression configuration applied uniformly to
all 40 contexts. That uniformity is what makes "symbolic regression fails on this context"
a property of the context; allowing each context its own configuration would make failure a
property of the search instead. We therefore fix the configuration in advance and report it
unchanged throughout, and we note that two of the three low-complexity failures above
(DUSP7, DUSP16) do yield generalising laws under a different operator basis — evidence that
those particular failures reflect the search rather than the dynamics.

## Numbers, with their sources

| quantity | value | source |
|---|---|---|
| contexts solved | 14 / 40 | `pysr_ood_final`, train-parsimony seed selection |
| dependency count, fails vs solves | 4.15 vs 3.36, P = 0.0030 | Fig. 4E |
| interacting pairs, fails vs solves | 7.04 vs 3.78, 1.86x, P = 0.023 | Fig. 4F |
| AUC, complexity ranks failures | 0.757, perm P = 0.0029 | this draft |
| tercile trend | 0.54 / 0.43 / 0.08, P = 0.013 | `fig_4_complexity_doseresponse` |
| contexts above 6.7 pairs | 0 / 12, perm P = 0.0016 | this draft |
| complexity controlling for NN accuracy | β = −1.02, P = 0.033 | this draft |

## Limitation to disclose in the SI

Pooled across a twelve-configuration ensemble rather than the single frozen configuration,
the complexity term does not survive on perturbation contexts alone once neural-ODE
accuracy is included (β = +0.006, P = 0.97; accuracy β = −0.48, P = 0.0063). The main-text
claim is therefore specific to a fixed search configuration, as scoped above. A
pre-registered confirmatory test of the ensemble version is specified in
`docs/preregistration/2026-08-11_gated_complexity_test.md` and awaits the maxsize=26
stratum of the balanced grid.
