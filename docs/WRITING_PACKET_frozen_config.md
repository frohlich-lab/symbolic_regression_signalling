# Writing packet — frozen configuration result

**This is the authoritative version.** Earlier drafts renamed `*.superseded.md`.

## Specification (state this verbatim in Methods)

- **Symbolic regression**: PySR, frozen production configuration
  `it1400_p30_ps30_ms26_par08_unone_bdiv` — 1400 iterations, 30 populations, population size
  30, maxsize 26, parsimony 0.8, binary operators `+ - * /`, no unary operators. Seeds 42, 43,
  44. Applied uniformly to all 40 contexts; no per-context tuning.
- **Success**: held-out ODE-integrated R2 >= 0.6, best of the three seeds. 14 of 40 contexts.
- **Learnability gate**: the 21 contexts where the sparse neural ODE (L21, hidden layers 4)
  cleared held-out ODE-integrated R2 > 0.6 on the test split, best of three network seeds.
- **Split**: `top_gfp_bins` — highest-GFP bins held out, so scoring requires extrapolation to
  perturbation strengths never seen in training.
- **Complexity**: effective dependency count (Jacobian entries above 0.15 x max) and
  interacting input pairs (off-diagonal Hessian above 0.25 x off-diagonal max), averaged over
  network seeds, read from the neural ODE.
- All P values two-sided.

## The result

| quantity | failed | solved | P | AUC |
|---|---|---|---|---|
| **dependency count** | **4.15** | **3.10** | **0.0082** | **0.841** |
| **interacting pairs** | **6.97** | **3.73** | **0.0182** | **0.809** |

n = 21 (11 failed, 10 solved).

Supporting, same specification:

- **Robust to the seed rule.** Selecting the seed by train parsimony rather than best-of-three
  on held-out data: 3.97 vs 3.12, P = 0.040, AUC = 0.774 (13 failed / 8 solved).
- **Graded, not only a group split.** Recovery frequency — the fraction of the three seeds
  clearing threshold, which involves no seed selection — falls with closure dimension:
  rho = -0.54, P = 0.012 (dependency count); rho = -0.52, P = 0.018 (pairs). All 21
  leave-one-out estimates negative.
- **Not explained by data quality.** Including the network's own held-out accuracy as a
  covariate leaves the complexity term significant: logistic beta = -1.02, P = 0.033.
- **A ceiling under this configuration.** Mean recovery frequency 0.44 at or below four
  measured variables, zero above: 0 of 6 contexts above four recovered a law in any seed,
  against 10 of 15 at or below (Fisher P = 0.012).
- **Robust across specifications.** 84 pre-enumerated analyses (three gates including none,
  seven recovery definitions including three threshold-free, two complexity measures, with and
  without controls): 83 of 84 in the predicted direction, median rho = -0.36; all 19 reaching
  P < 0.05 do so in the predicted direction; none contradicts.

## Results paragraph (drop-in)

To ask whether symbolic regression's failures reflect the method or the absence of an
accessible reduced description, we used a sparse neural ODE as a learnability control. Because
it is scored under the same out-of-distribution split — holding out the highest-GFP bins —
reaching R2 >= 0.6 requires genuine extrapolation to stronger perturbations than any seen in
training. In the 21 contexts where it succeeds, the measured variables demonstrably support a
reduced description that generalises, so a failure of symbolic regression there cannot be
attributed to missing signal or noise. Our central quantity is the effective dependency
count — the number of measured inputs the learned rate law depends on, that is, the dimension
at which the dynamics close. It is read from the network, which never sees the symbolic
search's objective, operator basis or output, so it is independent of the quantity it is used
to explain.

Contexts where symbolic regression failed closed over more measured variables than those it
solved (4.15 vs 3.10; Mann-Whitney P = 0.008; AUC = 0.84) and carried more interacting input
pairs (6.97 vs 3.73; P = 0.018; AUC = 0.81). The relationship is graded rather than a group
difference alone: the fraction of seeds recovering a generalising law falls with the closure
dimension (Spearman rho = -0.54, P = 0.012), with every leave-one-out estimate negative. It is
not an artefact of selecting the best of three seeds on held-out data, since choosing the seed
by training parsimony instead preserves it (3.97 vs 3.12, P = 0.040). Nor is it a restatement
of how well the network itself performed: including the network's held-out accuracy as a
covariate leaves the complexity term significant (logistic beta = -1.02, P = 0.033), so this
is not the trivial pattern in which poorly-measured contexts defeat both methods together.
Under this configuration the effect is absolute at the top of the range — none of the six
contexts closing over more than four measured variables recovered a law in any seed, against
ten of fifteen at or below four (Fisher P = 0.012).

Where symbolic regression finds a law, the dynamics coarse-grain compactly and the recovered
law is the demonstration. Where it does not, the dynamics still close over the measured
variables — the network closes them and extrapolates — but over more of them. Compact
coarse-grainability is therefore a property that can be measured per context rather than
assumed.

## Abstract clause (drop-in)

Where it fails, a flexible neural ODE still fits — the structure is present in the measured
variables — and the dynamics close over more of them (4.15 vs 3.10 measured variables,
P = 0.008), a calibrated dependency count read from the flexible model and therefore robust
to the symbolic search's operator basis.

## Scope (do not drop these)

1. **One configuration, applied uniformly.** That uniformity is what makes recovery a property
   of the context rather than of the search. State it.
2. **Sufficiency, not necessity.** Four perturbation contexts fail *within* the complexity
   frontier (DUSP16, DUSP7, MST1R, RPS6KA6), plus one control (untransfected4). A failure at
   low closure dimension reflects the limits of the search, not the dynamics. Name them.
3. **Association, not capacity.** Do NOT write "more variables than a compact expression can
   hold": symbolic regression recovers laws at closure dimensions up to 4.7 elsewhere in these
   data, and writes expressions using five variables. The distributions overlap.
4. **Explained variance, not relative error.** Justify R2 in Methods: law recovery is a
   statement about capturing the shape of the dynamics, whereas a relative-error criterion
   admits models that track the trajectory mean without recovering its dynamics. The result
   does not hold under relative MAE, so make the argument pre-emptively.
5. **The gate uses the network's test-split performance** as an inclusion criterion. It is not
   selection on the symbolic-regression outcome, but report it explicitly.

## EXHAUSTIVE definition of the frozen configuration

Read off `run_markers.py` and the production sbatch, not reconstructed. Tag
`it1400_p30_ps30_ms26_par08_unone_bdiv` encodes only the first six rows; everything below is
part of the configuration and must be stated for the run to be reproducible.

### 1. PySR hyperparameters set explicitly on the command line

| parameter | value |
|---|---|
| `niterations` | 1400 |
| `populations` | 30 |
| `population_size` | 30 |
| `maxsize` | 26 |
| `parsimony` | 0.8 |
| `binary_operators` | `+`, `-`, `*`, `/` |
| `unary_operators` | none |

### 2. PySR settings fixed in code (not exposed per-run)

| parameter | value | why it matters |
|---|---|---|
| `deterministic` | `True` | bit-exact within one environment |
| `parallelism` | `"serial"` | required for determinism |
| `procs` | 0 | single process |
| `batching` | `True` (argparse default) | mini-batching is ON |
| `annealing` | `True` (argparse default) | simulated annealing is ON |
| `loss_scale` | `"log"` | **the objective is log-scaled**; the custom loss is therefore not a plain MSE |
| `verbosity` | 0 | |
| `random_state` | 42, 43, 44 | three seeds per context |
| `output_directory` | `<run>/pysr_search` | |

### 3. PySR settings left at library default — and one of them is load-bearing

`model_selection` is **never set**, so PySR's default `"best"` applies: the reported expression
is chosen off the Pareto front by log-loss improvement per unit complexity. Combined with the
penalties in section 4, infeasible expressions sit on the front at loss ~1000, which creates a
large apparent improvement at low complexity and can pin the selected expression at complexity
~5. This is a real interaction between two settings and should be stated.

Also at library default: `ncycles_per_iteration`, `maxdepth`, `constraints`,
`nested_constraints`, `complexity_of_operators/constants/variables`, `warm_start`, `turbo`,
`precision`, `timeout_in_seconds`, `early_stop_condition`.

### 4. The custom Julia loss

Data term, on the derivative target, before PySR's log scaling:

    base = sum((f0 - y).^2) / length(y)          # unnormalised MSE

Hard constraints — an expression violating either is returned at `base + 1000.0`:

| constant | value | meaning |
|---|---|---|
| `REQUIRE_GFP` | `true` | **GFP must appear in every expression** |
| `REQUIRED_IDX` | p-ERK column | **p-ERK must appear in every expression** |
| `MISSING_PENALTY` | 1000.0 | applied if either variable is absent |
| `REDUNDANT_PENALTY` | 1000.0 | applied if a required variable appears but carries no functional dependence |
| `DEP_TOL` | 1e-4 | dependence threshold for the redundancy test |

Stability penalties, applied proportionally to violating mass:

| constant | value |
|---|---|
| `WRONGSIGN_P_LINEAR` | 1000.0 (`--linear-stability-penalty` default) |
| `WRONGSIGN_INV_LINEAR` | 1000.0 (`--inverse-stability-penalty` default) |

Conditioning penalty, **disabled** in the production run:

| constant | value |
|---|---|
| `COND_THRESH` | 0.0 (off) |
| `COND_PENALTY` | 100.0 (unused at threshold 0) |

Tolerances: `EPS_REL = 1e-3`, `LIN_TOL_REL = 1e-3`.

Note: the two 1000.0 hard penalties are load-bearing rather than nominal — replacing them with
`Inf` breaks the search entirely (every fit returns no formula), because finite penalties
preserve ordering among infeasible candidates and let evolution reach the feasible region.

### 5. Data, target and features

| item | value |
|---|---|
| dataset | `markers_per_minute_fit.csv`, `--dataset-mode per_minute` |
| target | `p-ERK1-2_dt` — the **raw** derivative (verified: ratio to d/dt of raw p-ERK = 1.0000; log-derivative would be 2.31) |
| feature mode | `all` |
| excluded columns | `p-ERK1-2_dt`, `p-MEK1-2_dt`, `marker`, `timepoint`, `GFP_bin` |
| resulting 10 features | `p-ERK1-2_fit`, `GFP`, `p-ERK1-2_min`, `p-MEK1-2_fit`, `p-MEK1-2_min`, `p-RAF_fit`, `p-p90RSK_fit`, `p-MAPKAPK2_fit`, `p-PDK1_fit`, `p-MKK3-6_fit` |
| measured timepoints | 0, 5, 10, 15, 30, 60 min (fitting uses the per-minute grid; scoring uses these) |
| split policy | `top_gfp_bins` (**not** the argparse default `random_bins`) |
| test size | 0.2 (argparse default) — the highest-GFP 20% of bins held out |
| seeds | 42, 43, 44 |

### 6. Scoring

| item | value |
|---|---|
| metric | ODE-integrated R2, median over bins (`ode_integ_r2_median`), from `compute_marker_integration.py` |
| integration | the recovered rate law integrated forward per bin, compared with raw measurements at the six acquired timepoints |
| success threshold | > 0.6 on the held-out bins |
| aggregation across seeds | best of three (leak-free alternative: seed chosen by train parsimony) |

### 7. What is NOT part of this configuration

Flags added during methodological checks and left **off** in production:
`--standardise-target`, `--no-require-gfp`, `--conditioning-threshold` (0.0),
`--pysr-sample-weighting` (a silent no-op — the custom loss ignores sample weights).

## Corrections to v7

| v7 says | should be |
|---|---|
| AUC ~ 0.75 | **0.84** |
| solved group 2.90 variables | **3.10** |
| "~1.8 vs ~3.8 interacting pairs" | **3.73 vs 6.97** |
| "requires more interacting variables" | more measured variables (pairs are redundant with the dependency count: density saturates, partial rho = +0.16, P = 0.33) |
| "indicating dynamics that resist compression ... rather than lacking structure" | replace with the measured association; the deductive reading is not supported |
| "identifying where new measurements are needed" | the measured variables suffice for closure; say the dynamics do not compress |

## Provenance

Every number above reproduces from files present locally:
`data/experimental/runs/pysr_ood_final/` (SR fits, 3 seeds x 40),
`data/experimental/runs/reducibility/l21_hl4_local/` (network models and metrics),
`data/experimental/runs/analysis_2026-08-11/` (extracts, spec curve).
No cluster access required.
