# Path A — what has to be true for a general-interest submission

Path A means keeping the strong claim: *SR failure, cross-checked against a
sparse Neural ODE, tells you whether a low-dimensional description exists in the
measured variables — and therefore what to measure next.* This file tracks what
that claim needs, what is built, and what is genuinely open.

The honest framing: **the re-analysis in [`FINDINGS.md`](FINDINGS.md) weakened
the current evidence.** The load-bearing statistic gives *P*=0.44 in the
main-text model once best-of-seed selection is removed. Path A is therefore not
"defend the existing result" — it is "replace the existing evidence with better
evidence". Everything below is built to do that, and each item can come back
negative. That is the point; a diagnostic that cannot fail is not a diagnostic.

---

## The multiverse result — and the reframe it makes available

Two instabilities are now documented in the same result: across **regulariser**
(P=0.036 under L1, P=0.442 under the main-text L21) and across **depth**
(P=0.032 vs P=0.87 between architectures that are statistically
indistinguishable on held-out data, paired P=0.28). Neither choice was argued
for; depth was a script default.

`multiverse.py` quantifies this over **1,008 defensible specifications**
(regulariser × PR aggregation × threshold × seed convention × objective ×
conditioning), and the answer separates cleanly into two parts:

| | Result |
|---|---|
| **Direction** | positive in **97.3%**; significantly negative in **0%**; median ΔPR = **+0.45** |
| **Significance** | clears P<0.05 in only **20.6%**; median P = **0.203**, range 0.0002–0.995 |
| **PR variance** | biological context 29.8%, regulariser 17.2%, seed 1.0% |

The manuscript's own specification, run under all nine equally-defensible
regulariser × PR-aggregation combinations, gives P from 0.037 to 0.443 — **the
published result is one of nine**, and it sits at the 43rd percentile of the
effect-size distribution, i.e. typical in magnitude and simply under-powered. Its
conditioning (perturbations + Neural-ODE-success) is also the *least* powered in
the whole space (5% of specifications significant, versus 35% using all
contexts).

**This is a better paper than one P-value, not a worse one.** The defensible
claim becomes:

> SR-fail contexts consistently lean on more measured inputs — the effect is
> positive in 97.3% of 1,008 analytic specifications and never significantly
> negative — but it is small relative to specification noise, and any single
> (architecture, penalty, threshold) choice yields P anywhere from 0.0002 to
> 0.995 on identical data.

That has two audiences instead of one. The signalling result survives, on much
harder-to-attack evidence than a single test. And the methodological finding —
**Jacobian-based dependency attribution in sparse neural ODEs is not
identifiable on correlated biological data** — generalises well beyond ERK, to a
class of readouts (participation ratios, L1/L21 dependency counts, "which inputs
matter" plots) that is used widely and almost never stability-checked. Breadth
is the criterion the current framing fails; this framing has it.

Three things must hold for that version to work, and all three are running:

1. **`arch_sweep.py`** must show depth behaves like the other nuisance factors —
   shifting significance while leaving direction intact. If depth genuinely
   *reverses* the sign, the negative result is stronger but the positive claim
   dies. 12 architectures sit within 0.05 validation R² of each other; that is
   the multiverse.
2. **`pr_calibration.py`** must supply the mechanism — correlated inputs making
   attribution non-identifiable — so the finding is explained, not just observed.
3. **`input_ablation.py`** must show **k\* is stable where PR is not**. Without a
   constructive alternative this is a purely negative paper, which is a much
   harder sell. Run `multiverse.py` against k\* once the ablation data lands.

Item 3 is the one that decides whether this is publishable as a positive
contribution. Prioritise the ablation run.

## The five claims, and what would sink each

| # | Claim | Test | Sinks it |
|---|---|---|---|
| A | The dependency metric means what we say | `pr_calibration.py` | PR saturates → PR cannot support a dimensionality reading; `k*` replaces it everywhere |
| B | SR-fail contexts genuinely need more variables | `input_ablation.py` → `k*` | `k*` does not separate → the dimensionality story is wrong |
| C | SR-fail contexts carry hidden state | `latent_node.py` | gain flat or uniform across contexts → drop the hidden-state language |
| D | SR-failure adds over the identifiability toolkit | `identifiability_baseline.py` | classical diagnostics predict SR outcome → novelty rests on practical advantages only |
| E | The diagnostic guides measurement | `rescue_demo.py` | dropping a random input does the same → the ranking is not specific |

`collect.py` adjudicates all five and writes `path_a_verdicts.csv`.

**A and E are the two that decide it.** A decides whether the existing
Table S8 / Fig. S3 argument survives at all. E is the generative promise the
abstract already makes and the paper never demonstrates — all three mock
reviewers flagged it, and it is the closest thing here to a headline result.

---

## Already known, and already load-bearing

From step 1–3 re-analysis (no new compute):

- The manuscript's n=12/6 comes from **perturbations only × conditional on
  Neural-ODE success × best-of-3**. State this in Methods regardless of what
  else happens.
- The PR contrast is significant in L1 (*P*=0.036) and not in the main-text L21
  (*P*=0.442, CI [−0.56, 1.38]). Direction is consistent everywhere.
- Best-of-3 inflates the perturbation SR success rate 25% → 6.3% (majority).
- Neural-ODE false-negative rate is 7.5% (3 contexts), dropping to 1 context
  when both sides are scored on matched objectives.

From the first real ablation (PTPN7, seed 42): **the full ten-input model
extrapolates at R²=0.08 while a four-input model reaches 0.78.** Held-out R² is
strongly non-monotone in input count. If this generalises it changes the paper:
the published Neural ODEs are overfitting the held-out GFP bins, which means
every participation ratio in Table S8 was read off an overfitted model. That is
either a serious problem or a much better result than the current one —
`k*` reported per context, with the overfitting shown, is a cleaner and more
useful claim than PR ever was.

---

## What Path A still needs that is *not* built here

These need decisions or data, not code:

1. **A real new measurement.** Everything here is in-silico hold-out. The
   strongest possible version — take a genuinely failing context, measure the
   variable the diagnostic names, show a compact law appears — needs bench work.
   `rescue_demo.py` establishes the inference is sound so that this is worth
   doing; it does not substitute for it.

2. **"Self-contained" has to go or be earned.** The p-ERK ODE is driven by
   measured trajectories of its own network neighbours, which for ERK (feeding
   back on MEK/Raf) is not defensible as self-containment. Either move to
   genuinely coupled multi-state discovery, or rename the claim. This is a
   scoping decision, not an analysis.

3. **The pseudo-trajectory validity gap.** Population-averaged CyTOF snapshots
   are treated as a dynamical system; mean-of-nonlinear-dynamics is not
   dynamics-of-the-mean, and within-bin stationarity is untested. At minimum
   this needs a stated limitation and ideally a simulation showing how much
   averaging distorts a known law — which the existing synthetic machinery in
   [`src/pipelines/regimes/`](../../regimes/) could do.

4. **A prospective hypothesis that is not textbook.** PTPN7/HePTP restates known
   ERK-phosphatase biology. One non-obvious, checkable prediction is worth more
   than two confirmations.

Items 1 and 4 are the difference between a strong specialist paper and a
general-interest one. Items 2 and 3 are things a referee will raise regardless
of venue.

---

## Order of operations

1. **Run `submit_nemo.sh all`** — five job arrays, one task per marker. Start
   with `identifiability` and `ablation`; they are the cheapest and they settle
   claims B and D.
2. **Read `path_a_verdicts.csv`.** If claim A comes back unsupported, stop and
   restructure around `k*` before writing anything.
3. **Then decide the venue.** With A, B and E supported there is a real
   general-interest case. With A unsupported, the honest paper is the specialist
   one — and the `k*` overfitting result makes it a better specialist paper than
   the current draft.

The cheap writing fixes in `FINDINGS.md` §"What this means" should be applied
regardless of which way this goes; none of them is wasted work.
