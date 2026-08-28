# H106 — does the ROI close the ONE gap that matters, seed-matched at n=10?

LOCKED BEFORE ANY RUN. 5 runs: Borehole ROI-Q10 at seeds 52-56.

## Why this is now the right experiment

The n=10 table (verified today, explicit paths) says MF-DRO's four-benchmark
picture is **tied on two, nil on one, and losing on exactly one**:

    Borehole   MF-DRO 15.42  vs  MF-MES 8.24   gap +7.18, median +8.30, 2/10

Borehole is the single real deficit, and its median gap EXCEEDS its mean, so it
is not one seed. **It is also the only benchmark where the ROI does anything at
all** -- every ROI effect measured (regret, relocation, waste, query quality) is
Borehole-specific (h104).

So the one intervention that works, works on the one deficit that is real. That
is either the most useful thing this investigation has produced or a coincidence
of a single benchmark, and the way to tell is to measure them on the SAME seeds.

**They never have been.** The ROI is measured at 42-46 (h84) and 47-51 (h90,
h94). The n=10 MF-DRO-vs-MF-MES comparison uses 42-46 + 52-56. The overlap is
half. Filling seeds 52-56 makes the ROI arm seed-matched to the comparison and
to MF-MES's own runs (h92 has Borehole MF-MES at 52-56).

## Design

    arm            Borehole_8D, ROI-Q10, seeds 52-56          5 NEW runs
    control        h89's CONTROL at 52-56                     reuse, explicit path
    comparator     h92's MF-MES at 52-56                      reuse, explicit path
    dev half       h84's ROI-Q10 at 42-46                     reuse, explicit path

Config is h84/h90's ROI-Q10 EXACTLY: use_roi=True, roi_beta_mode='quantile',
roi_target_accept=0.10, n_roi_candidates=600. No variant will be run on these
seeds. **Every cross-experiment path is named here; no globbing** (the rule
adopted after a glob returned a different experiment's run of the same config).

## Predictions, with EFFECT SIZE stated — the rule four bars failed today

M1 (h95) passed on a floored measure. T2 (h98) passed on an unseparable
ordering. W2 (h104) passed at a mean of -0.001. P1 (h94) passed at a mean one
sixth of its own sd. All four used counts and signs and none required the effect
to be large relative to its spread. These do.

**Q1 (PRIMARY). The ROI reduces MF-DRO's Borehole deficit against MF-MES at
seed-matched n=10, by a paired mean of at least ONE HALF of its own paired sd.**
Registered POSITIVE. The count is secondary and reported alongside; the bar is
the ratio. Prior: the ROI's three Borehole measurements are -4.22, -3.49, -3.86,
so the effect is ~3.7 with a paired sd around 2.4-2.7, a ratio near 1.4.

**Q2 (NEGATIVE, registered so a gain cannot be inflated). The ROI does NOT close
the gap: MF-DRO+ROI's n=10 mean stays ABOVE MF-MES's 8.24.** Expected residual
~11.5 against 8.24. If this is refuted, MF-DRO+ROI ties or beats MF-MES on the
one benchmark it loses, which would change the project's headline and must be
re-verified before it is believed.

**Q3. The dev half (42-46) and the new half (52-56) agree to within 2 points in
paired mean.** A larger split means the Borehole effect is seed-set-dependent
and Q1's pooled figure is not a single quantity. Registered POSITIVE: h90 and
h94 agreed to 0.72 pts per seed at 47-51.

## Falsifier

If Q1 fails, the ROI does not measurably help on the one benchmark where MF-DRO
has a real deficit, and its Borehole gain -- confirmed three times -- is a gain
against a control that does not matter for the comparison anyone cares about.

## Gate

8 workers running (peer's h97 and h102). 5 runs takes it to 13, inside 15.
