# h142 — Does the RTG/GP decoupling PREDICT regret, or is it another h118?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.
STATISTIC: per-run fractional within-run change in `rtg_gpbelief_corr_per_iter`
(first third -> last third) against that run's final regret as **rel% of
|optimum|** @cost_curve 200 via h83's frozen `sr_curve` + `grid`. Rank
correlation across runs, computed **within benchmark** so the two scales never
mix.

## Why this gate exists before any heuristic

h140 found `rtg_gpbelief_corr` — the correlation between the return-to-go the DT
is conditioned on and the GP's own posterior mean at those points — **rises ~4x
across a Borehole run and falls on Hartmann, effect 5.54**, the largest in this
project. The code's own comment says it "checks whether RTG tracks the model's own
aggregate notion of state quality at all." Read plainly: on Hartmann the DT is
conditioned on return targets its own surrogate does not believe.

**That is exactly the shape of h117.** h117 confirmed MF-DRO's boundary waste is
real and reproducible at fresh seeds — and h118 then showed **the waste does not
predict regret.** Real, reproducible, and irrelevant. A whole line of work was
spent before that was checked.

**So the check comes first this time.** If RTG/GP decoupling does not predict
regret, it is an epiphenomenon and no heuristic should be built on it, however
large its effect size.

## Predictions (locked)

**P1 (PRIMARY).** Within Hartmann, across seeds 42-46 control runs, the rank
correlation between RTG/GP degradation and final regret has |rho| >= 0.5, with
**worse degradation going with worse regret**. FALSIFIED if |rho| < 0.5.
(Partitioned; 0.5 is this project's effect-size bar applied to a rank statistic.)

**P2.** The same within Borehole. Reported whatever it shows — with only five
seeds per benchmark I have no grounds to predict Borehole separately, and a
rank correlation on n=5 is weak by construction.

**P3.** Pooled across both benchmarks with regret expressed in rel% so the scales
are commensurable, n=10. Reported; **not** the primary, because pooling two
benchmarks whose RTG behaviour differs in SIGN is the kind of aggregation that
has produced three artefacts in this project already.

## Stated before looking

- **n=5 per benchmark is very weak for a rank correlation.** A |rho| of 0.5 at
  n=5 arises by chance often. **P1 passing is therefore suggestive, not
  decisive**, and I am recording that in advance so a pass cannot be quoted as
  establishing the lever. A *failure* is the more informative outcome: it would
  retire the lead cheaply, exactly as h118 retired boundary waste.
- The direction of causation is not testable here at all. Even a strong
  correlation is consistent with both "bad RTG causes bad regret" and "a run
  going badly produces both".

## What this could RETRACT

**My recommendation to the user that the training-signal direction is where
MF-DRO should be improved.** h140's P1 (gradient coherence) survived its gate, so
the direction has one locked result behind it. But **if the sharpest diagnostic
in that audit turns out not to predict outcome, the direction is much weaker than
I presented it**, and I should say so plainly rather than falling back on the
grad-coherence result.
