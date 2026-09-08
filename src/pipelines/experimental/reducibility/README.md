# `reducibility/` — strengthening the SR-failure / Neural-ODE diagnostic

This subpackage exists to answer one question with measurements instead of
argument: **how strong is the evidence that SR failure marks a genuine absence
of a compact low-dimensional rate law, rather than a limitation of SR?**

That claim is what the manuscript's bid for a general-interest venue rests on,
and it is currently the least-evidenced part of the paper. Nothing here changes
the published pipeline; everything either re-analyses outputs already on disk or
runs the *same* Neural-ODE machinery
([`neural_ode_diffrax_baseline.py`](../sr_pipeline/neural_ode_diffrax_baseline.py))
under a controlled change, so every number is directly comparable to Table S8 /
Fig. S3.

The published OOD configuration is pinned once, in `PAPER_CFG` in
[`_shared.py`](_shared.py), transcribed from `workflow/rules/experimental.smk`
and `workflow/rules/common.smk`. Keep it in sync with those rules.

## The six steps

| Script | What it does | Needs training? |
|---|---|---|
| [`_shared.py`](_shared.py) | Pinned config, bundle loading, input subsetting, Jacobian/PR helpers | — |
| [`assemble.py`](assemble.py) | One per-(marker, seed, variant) table joining PySR, Neural ODE and PR, with **matched objectives** | no |
| [`quadrants.py`](quadrants.py) | The SR × Neural-ODE 2×2 under three seed conventions, stratified by control/perturbation | no |
| [`pr_stats.py`](pr_stats.py) | Re-tests the participation-ratio contrast without best-of-seed selection | no |
| [`pr_calibration.py`](pr_calibration.py) | Calibrates PR against **known** ground-truth dimensionality on real input correlations | yes |
| [`input_ablation.py`](input_ablation.py) | Replaces the PR proxy with `k*`, a directly measured input count | yes |
| [`latent_node.py`](latent_node.py) | Tests for **hidden state** with a latent-augmented Neural ODE | yes |
| [`rescue_demo.py`](rescue_demo.py) | Drop the predicted variable → law breaks → restore it → law returns | yes |
| [`identifiability_baseline.py`](identifiability_baseline.py) | Head-to-head vs observability rank / profile-likelihood flatness | yes |
| [`collect.py`](collect.py) | Merges fanned-out runs and adjudicates the Path A claims | no |
| [`submit_nemo.sh`](submit_nemo.sh) | SLURM job arrays for the training steps (one task per marker) | — |

## Which reviewer objection each step answers

1. **`assemble.py` — objective parity.** SR is fit on derivative MSE, the Neural
   ODE on trajectory MSE, and the manuscript then compares them "on the same
   scale". Both sources also record an integrated held-out R², which *is* the
   same quantity; this table carries it for both sides and keeps the native
   metrics alongside so the effect of matching is auditable.

2. **`quadrants.py` — the 2 discordant contexts.** 8 SR successes, 18 Neural-ODE
   successes and 6 overlapping implies 2 contexts where SR succeeds and the more
   expressive network fails. Those only look like a contradiction if the Neural
   ODE is claimed to upper-bound reducibility. The defensible role is a
   **positive control for learnability**: SR fails + network succeeds ⇒ real
   structure is present, so SR's failure is neither noise nor merely an
   optimiser artefact. Under that framing the full 2×2 is reportable and the
   discordant cells become a measured error rate.

3. **`pr_stats.py` — the single P=0.032.** Replaces it with: a threshold-free
   Spearman test of PR against SR *recovery frequency* (all 40 contexts, no
   dichotomisation); a marker-level permutation test whose null respects the
   3-seed × 3-variant clustering; a mixed model with a marker random intercept;
   bootstrap CIs and effect sizes; and a sweep showing how the result moves with
   the 0.6 threshold.

4. **`pr_calibration.py` — the correlated-input confound.** An L21 penalty
   spreads Jacobian mass over a correlated block (inflating PR) while SR's
   parsimony picks one representative, so the PR gap may report *how each method
   treats correlation* rather than dimensionality. This holds the real, real-
   correlated inputs fixed and swaps in a state simulated from a law with a
   **known** driver count, then asks what PR reads back. `--shuffle-control`
   decorrelates the inputs while preserving marginals, isolating the correlation
   contribution.

5. **`input_ablation.py` — proxy → measurement.** `k*` = the smallest number of
   measured inputs reaching within ε of the full-model held-out R², by
   retraining (not masking) on greedily ranked inputs. Interpretable in the way
   the paper wants PR to be, not inflated by correlation, and it sets up the
   drop-and-restore demonstration.

6. **`latent_node.py` — the claim that needs positive evidence.** PR is defined
   over *observed* inputs and so cannot, in principle, be evidence about
   unobserved ones; the present inference is from absence. This augments the
   state to 1+d with an encoder-supplied latent initial condition (not a free
   per-bin parameter, which would hand held-out bins a free escape hatch) and
   asks whether held-out R² improves **more** in SR-fail than SR-success
   contexts. That interaction is the evidence; a main effect of `d` is just
   capacity.

7. **`rescue_demo.py` — the generative promise.** The abstract says the method
   "guides experiment design ... motivating new measurements", but no failing
   context is ever rescued. This drops the input the diagnostic points at,
   confirms the irreducibility signature appears, then restores it and confirms
   the signature reverses — with a random-input ablation as a specificity
   control. It is an in-silico hold-out, not a new measurement, and any write-up
   must say so; but it establishes the inference is sound before anyone spends
   bench time acting on it.

8. **`identifiability_baseline.py` — the novelty gap.** The paper cites the
   profile-likelihood, observability and manifold-boundary literatures and never
   says what SR-failure adds. This runs observability rank, profile-likelihood
   flatness and plain collinearity on the same contexts and asks whether SR
   outcome is predictable from them. If it is, the novelty case has to rest on
   practical advantages instead — better to know now.

9. **`collect.py` — adjudication.** Merges the per-marker outputs and states
   each Path A claim as a falsifiable prediction with the outcome that would
   sink it, writing `path_a_verdicts.csv`.

## Running it

The re-analysis steps need only pandas/numpy and run in any environment:

```bash
cd src/pipelines/experimental/reducibility
python assemble.py          # -> data/experimental/runs/reducibility/reducibility_master.csv
python quadrants.py
python pr_stats.py --n-perm 20000
# the manuscript's own subset — perturbations only, conditioned on Neural-ODE success:
python pr_stats.py --condition-on-node-success --perturbations-only
```

`assemble.py` requires `participation_ratio_variants.csv`; regenerate it with
[`figures/compute_participation_ratios.py`](../sr_pipeline/figures/compute_participation_ratios.py)
if missing (the variant rules must have run with `--save-models`).

The three training steps need jax + equinox + diffrax + sklearn. `run_all.sh`
resolves a suitable environment automatically, or set `PYTHON` yourself:

```bash
./run_all.sh                       # re-analysis only (fast)
./run_all.sh --with-training       # adds a scoped calibration / ablation / latent run
```

All three training scripts take `--markers`, `--seeds` and `--resume`, and
append to their CSV per row, so they can be scoped, interrupted and continued.
Scope them: `input_ablation.py` costs `(n_exo + 1)` fits per (marker, seed) —
1200 fits for all 40 markers × 3 seeds.

## Compute

Path A needs ~6,000 Neural-ODE fits. Two things make that a cluster job:

* each fit at the published 200 epochs takes minutes, and
* the diffrax solve pays a **multi-minute XLA compile** the first time it sees a
  given input count in a process — one observed compile took 16 minutes.

Fanning out one array task per marker fixes both (each task compiles once per
input count, 40 run concurrently). [`submit_nemo.sh`](submit_nemo.sh) does this
on NEMO's `ncpu` partition using the `sr_metrics2` env — the one built with
`jax[cpu]` + `diffrax` + `equinox` + `scikit-learn`; `pysr_env` predates those
and cannot run these scripts. Every script appends per row and supports
`--resume`, so a timed-out or requeued task loses nothing.

```bash
# on the cluster checkout
src/pipelines/experimental/reducibility/submit_nemo.sh all
# when the arrays finish
python src/pipelines/experimental/reducibility/collect.py --root <runs>/reducibility
```

## Status

Steps 1–3 have been run; outputs are in `data/experimental/runs/reducibility/`
and the results are in [`FINDINGS.md`](FINDINGS.md) — including the fact that the
participation-ratio contrast does **not** reach significance in the main-text
L21 variant once best-of-seed selection is removed.

Steps 4–9 are implemented and smoke-tested. First real ablation result (PTPN7,
seed 42): `k* = 4`, test R² 0.78, versus **0.08 for the full ten-input model** —
i.e. the published architecture extrapolates far worse than a four-input one.
If that holds across contexts it is a result in its own right, and it means PR
was computed on overfitted models.

See [`PATH_A.md`](PATH_A.md) for what still has to be true for a
general-interest submission, and which of it is compute versus new science.
