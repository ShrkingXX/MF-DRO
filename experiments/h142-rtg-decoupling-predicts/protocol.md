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

---

## AMENDMENT 1 — P4, extend to every qualifying control run (registered before checking which qualify)

P1 passed at rho = -0.600 within Hartmann and -0.600 within Borehole, but at n=5
the 5% Spearman critical value is ~0.9, so those are consistent with chance. **The
cheapest way to strengthen or kill the lead is more seeds, and many control runs
already exist** — h75 (Borehole), h77 (Hartmann), h117 (Borehole), h93, h57.

**I already know the n=5 result. So the inclusion rule is fixed HERE, before I
check which runs qualify, and every qualifying run is used — no selection.**

### Inclusion criteria (a run is IN iff all four hold)

1. `_meta.method` is an unmodified control arm: `MF-DRO`, `NO-ROI`, or `ROI-OFF`.
2. `_meta` matches h83's for that benchmark on **all** of `budget`, `c_H`, `c_L`,
   `n_hf`, `n_lf`. A different initial design or cost ratio is a different object
   and pooling it is the error that broke h133's ordering.
3. `rtg_gpbelief_corr_per_iter` is present and has >= 9 entries (so thirds are
   meaningful).
4. The run completed — a `results/` file, never a `ckpt/` checkpoint. h126's
   killed runs are the precedent for why this must be explicit.

**Every excluded run is reported with its reason.** Duplicate seeds across
experiments are resolved by taking h83's, then the lowest experiment number, and
the choice is reported.

### P4 (locked)

Within each benchmark separately, across all qualifying runs: **|rho| >= 0.5 with
the same sign as P1** (worse RTG/GP degradation with worse regret). FALSIFIED if
|rho| < 0.5.

**This is the real test.** P1 at n=5 could not distinguish the lead from noise;
with n>=10 per benchmark, |rho| >= 0.5 starts to mean something. If P4 fails on
both benchmarks the lead is dead and the direction loses its sharpest diagnostic.

**Still no pooling across benchmarks.** P3 flipped sign at n=10 and that remains
the standing demonstration of why.

### What P4 failing would retract

Everything h142 currently supports: that RTG/GP decoupling is a lever rather than
an epiphenomenon, and with it the specific mechanism behind my training-signal
recommendation. h140's P1 (gradient coherence) would still stand, so the direction
would survive — but on a weaker and less specific footing than I have claimed.
