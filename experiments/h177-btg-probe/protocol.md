# h177 -- is the OTHER conditioning input inert too?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Why this targets the right thing

h175 established that the tau=0 account's residual is **~5 SE on BOTH
benchmarks** (5.0/5.0 Borehole, 5.3/4.7 Hartmann). A stable incompleteness is a
better target than the benchmark-specific scope gap, because it is a property of
the mechanism rather than of one function.

The DT is conditioned on **two** scalars at inference: `rtg_tgt` and `btg_now`
(mf_dro.py:3224). **h168 showed the emitted action is essentially independent of
RTG** -- 0.0074 movement across the full 0-to-1 sweep, 357 probed iterations.
BTG has never been probed.

## The measurement

The h168 probe, which is verified **bit-identical to an unprobed run** on all
five seeds, gains a BTG axis: at every real iteration, re-query the DT at the
same state and the real RTG across a sweep of BTG values, and record each
emitted x. Same RNG save/restore.

BTG sweep spans the range the runs actually visit, read from the logged
`btg_now` values rather than guessed.

## Predictions

P1 The emitted action is also essentially independent of BTG -- movement across
   the sweep under ~15% of its mean distance from the box centre, matching what
   h168 found for RTG.
P2 It is NOT independent -- BTG moves the action materially. Then part of the
   residual is BTG-driven, and the tau=0 account is incomplete in a locatable way
   rather than an unlocatable one.

## What this can RETRACT

R1 P1 holds -> **both** conditioning inputs are inert, and the DT's inference
   output is a function of the state alone. That sharpens the tau=0 account (the
   conditioning does nothing at all) but leaves the residual unexplained, and I
   would then say so rather than look for a third scalar.
R2 P2 holds -> a locatable component of the residual, and the first thing on this
   front that would suggest a fix (condition on something the network actually
   responds to).
R3 The probe fails to populate -> caught by a smoke test BEFORE launch, as
   registered for h169 and h171. h169 was lost to skipping exactly this.

## Sanity checks, before any number is read

SC1 probe populates with the BTG axis present (smoke test, pre-launch)
SC2 bit-identity of the default path (probe OFF)
SC3 the probed run remains bit-identical to its unprobed twin -- checkable
    against h149's RANDOM-POOL run on the same seeds, as h168 was

## Design

Hartmann_6D RANDOM-POOL seeds 42-46 -- the same arm h168 used, so the RTG and BTG
results are directly comparable and SC3 has an exact reference. 5 workers.

## Sanity checks passed, and an early signal recorded before the arm ran

SC1 (the check h169 was lost to): **PASS** — 7 BTG probes per iteration, axis
populated. SC2 bit-identity of the default path: **PASS** (122.2906675273).

**Early signal, 5 iterations on Borehole, recorded now so it is not reported as a
discovery later.** The emitted action moves **0.0007 across the whole BTG sweep —
0.7% of its own distance from the box centre**, against h168's 8.9% for RTG at
full length:

```
BTG   20.0  24.0  26.0  28.0  30.0  32.0  36.0
d      .0982 .0985 .0986 .0987 .0988 .0988 .0989
```

Note the smoke ran on Borehole where the real `btg_now` is 10.9, so the sweep
sits entirely *outside* the range that run visits — and the action still barely
moves. If that holds at full length on Hartmann (where 26–30 is the visited
range), **P1 fires and both conditioning inputs are inert**.

Five iterations of a barely-trained network is not a verdict and none is taken.
