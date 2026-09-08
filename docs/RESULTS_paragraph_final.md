# Results paragraph — final

## Specification

- **Symbolic regression.** Frozen production configuration, seeds 42/43/44; the seed is chosen
  by training R2 and its held-out score reported. Success at held-out ODE-integrated R2 > 0.6:
  13 of 40 contexts.
- **Learnability gate.** The 21 contexts where the sparse neural ODE cleared held-out R2 > 0.6
  on the same out-of-distribution split, taken as the best of three network seeds. **All
  statistics are computed on these 21.** The gate is not a convenience: the dependency count is
  read from the network's learned rate law, so it measures the dynamics only where that law
  demonstrably extrapolates.
- **Complexity.** Effective dependency count — inputs above 0.15 of the largest mean absolute
  input Jacobian — averaged over the three network seeds. Calibrated against synthetic dynamics
  of known driver count (slope 0.947, versus 0.688 for the participation ratio).
- **Statistics.** Two-sided. The unit of analysis is the context; permutations shuffle contexts,
  never seeds.
- **Aggregation principle** (state in Methods): existence questions take the best of three seeds
  (does a compact law exist; does a reduced description exist), quantities being estimated are
  averaged over them (complexity).

---

## Paragraph

To ask whether symbolic regression's failures reflect the method or the absence of an accessible
reduced description, we used a sparse neural ODE as a learnability control. Fitted to the same
variables and scored under the same out-of-distribution split — holding out the highest-GFP
bins — it cleared R2 > 0.6 in 21 contexts. In those, the measured variables demonstrably support
a reduced description that generalises, so a failure of symbolic regression cannot be attributed
to missing signal or noise, and the number of inputs the learned rate law depends on is a
meaningful readout of the dimension at which the dynamics close. That count is read from the
network, which never sees the symbolic search's objective, operator basis or output, so it is
independent of the quantity it is used to explain, and it was calibrated against synthetic
dynamics of known driver count (slope 0.947). We therefore restrict this analysis to those 21
contexts.

Among them, the contexts symbolic regression failed on closed over more measured variables than
those it solved (4.1 vs 3.1; two-sided Mann-Whitney P = 0.019; AUC = 0.81). The relationship is
graded rather than a group difference alone: the fraction of seeds recovering a generalising law
falls as the closure dimension rises (Spearman rho = -0.54, P = 0.011), an outcome that involves
no seed selection of any kind. The contrast is not explained by data quality; it persists when
the network's own held-out accuracy is controlled for, and when contexts with sparse measurement
coverage are excluded (SI).

That difference of roughly one variable is consequential because the learned rate laws are
densely coupled. Within each fit, the number of interacting input pairs is at or near the
maximum its dependency count allows (median exactly the maximum, mean 0.97 of it, never above),
so every variable that matters couples to every other that matters, and no context admits an
additive or separable rate law. The interaction terms a closed form must carry therefore grow as
the square of the closure dimension: equivalently, the contexts symbolic regression failed on
carry roughly 6.6 interaction terms against 3.8 where it succeeded. It is the closure dimension,
not the pattern of coupling, that varies across contexts.

Where symbolic regression finds a law, the dynamics coarse-grain compactly and the recovered law
is the demonstration. Where it does not, the dynamics still close over the measured variables —
the network closes them and extrapolates — but over more of them. Whether a compact
coarse-grained law exists is therefore something symbolic regression measures, not something one
assumes.

---

## Numbers, with sources

| statement | value |
|---|---|
| gate | 21 of 40 contexts, network best-of-3 held-out R2 > 0.6 |
| within the gate | 9 solved / 12 failed by symbolic regression |
| dependency count, failed vs solved | 4.06 vs 3.11 = 1.30x, P = 0.0191, AUC 0.806 |
| graded (seed recovery frequency) | rho = -0.543, P = 0.0109, n = 21 contexts |
| interaction terms (transform of the above, no separate test) | 6.58 vs 3.78 |
| per-seed interaction density, gated fits | mean 0.970, median 1.000, max 1.000 (n = 61) |
| calibration slope | 0.947 (participation ratio 0.688) |
| SR successes overall | 13 of 40 |

## For SI

- **Ungated, all 40 contexts:** the same contrast holds — 4.09 vs 3.36, P = 0.009; graded
  rho = -0.467, P = 0.002 — so the gate does not produce the effect. Note that outside the gate
  the dependency count is read from a network that failed to extrapolate, which is why the gated
  analysis is primary.
- **Gate definition.** With the network seed chosen on validation rather than best-of-three, the
  gate holds 17 contexts and the contrast is 4.12 vs 3.11, P = 0.051 (graded rho = -0.487,
  P = 0.050) — same direction and effect size, reduced power.
- **Covariate control.** Including the network's held-out accuracy leaves the complexity term
  significant (logistic beta = -1.02, P = 0.033).
- **Measurement coverage.** Excluding contexts with fewer than five held-out or twenty training
  bins leaves the contrast unchanged in direction.
- **Sensitivity threshold.** The contrast holds at Jacobian thresholds from 0.05 to 0.20 and
  vanishes only at 0.25-0.30, where the count degenerates toward two and stops discriminating.
- **Specification curve.** 83 of 84 pre-enumerated analyses run in the predicted direction; none
  contradicts.
- **Pairs are not independent evidence** and are reported without a test. Per-seed density is
  0.970 (median 1.000, never exceeding 1), so the pair count is a deterministic transform of the
  dependency count; partial rho = +0.16, P = 0.33 given the dependency count. Seed-averaged pair
  means can appear to exceed C(mean k, 2) purely because C is convex and k varies across seeds;
  compared correctly against mean C(k, 2), no marker exceeds it.
- **Contexts where symbolic regression outperforms the network.** Four under the primary rule:
  PIP5K3 (0.928 vs network 0.568), untransfected1 (0.794 vs 0.000), PTPN2 (P1) (0.698 vs 0.000),
  TBK1 (0.636 vs 0.594). In two the network scores exactly zero against SR at 0.70 and 0.79.

## Do not write

- "more variables than a compact expression can hold" — symbolic regression recovers laws at
  4.33 dependencies (TBK1) and writes five-variable expressions in these data.
- dependency count and interacting pairs as independent lines of evidence.
- any claim that a failure proves no compact law exists.
- a ceiling at 4.3 dependencies. It holds only under the production configuration and seeds; the
  train-selected configuration search recovered a compact law at closure dimension 4.67
  (PRKACA). Validating failures with the wide search while quoting a ceiling from the narrow one
  is inconsistent, and the graded correlation makes the same point protocol-independently.
