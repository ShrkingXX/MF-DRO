# h174 -- does rollout_length=1 work on Hartmann?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why

h172 is the front's only ACTIONABLE result -- one-step rollouts at no cost in
regret and a 6.26x wall-clock saving -- and it is **Borehole-only, n=5**. It is
also the result most likely to be quoted, because it changes what the code should
do rather than what we believe about it. A one-benchmark actionable claim is the
worst kind to leave unscoped.

Testing the extreme point (L=1) rather than the whole dose: the dose's shape is
established on Borehole, and what needs checking here is whether the extreme
survives a different benchmark, not whether the shape repeats.

## Predictions

P1 L=1 lands within ~3 rel% of the Hartmann control's 7.99, improving ~5/5.
P2 Wall-clock falls substantially (Borehole gave 6.26x; Hartmann's fixed
   per-iteration costs are a larger share, so expect less).

## What this can RETRACT

R1 L=1 fails on Hartmann -> the actionable claim is Borehole-specific and must be
   scoped in findings.md, research-state.yaml AND the published report, all of
   which currently state it without a benchmark qualifier. This is the exposure
   the arm exists to close.
R2 L=1 works -> the actionable claim holds on two benchmarks.
R3 Intermediate -> reported as such at n=5.

## Named confounds, checked before the numbers

SC1 realised HF fraction against the Hartmann control's 0.200. That reference is
    itself unstable there (per-seed 0.038-0.750, per h164), so the check is
    weaker than on Borehole and will be reported as such.
SC2 **The Hartmann saturation floor.** h173 seed 44 returned regret 0.7531 for
    HEAD, TAIL *and* h168's random arm alike -- the same value, because none
    improved on that seed's initial design. Any Hartmann arm can hit that floor,
    so per-seed agreement at 0.7531 is not evidence of anything and will be
    excluded from the improvement count rather than read as a tie.

## Design

Hartmann_6D seeds 42-46, rollout_length=1, n=5. 5 workers alongside h173's 7 = 12.
No code changes: h172's worker takes the benchmark as argv.
