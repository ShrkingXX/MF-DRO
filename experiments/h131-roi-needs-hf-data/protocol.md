# h131 — Is Hartmann different in KIND, or merely TRUNCATED?

STATUS: LOCKED before any statistic is computed.
TYPE: CONFIRMATORY. ZERO NEW COMPUTE.
READ POINT / STATISTIC: simple regret from h83's frozen `sr_curve` + `grid`
(post-init cost axis, `cost_cum - init_cost`), paired ROI-Q10 vs h83 `MF-DRO`,
Borehole and Hartmann, seeds 42-46. Reading at several grid points uses the
frozen metric's own step-interpolator; it is not a new metric.

## Why

Four mechanisms have now been tested across four benchmarks (peer's h130 table,
my h129 P4-P6) and **every one has exactly one positive cell, always Borehole**.
That is a well-established fact and it is purely descriptive. Nothing so far
explains WHY Borehole, and "it is Borehole-specific" is the kind of statement
that ends an investigation without answering it.

One structural difference is very large and has not been used: **Borehole affords
94.0 post-init HF queries; Hartmann affords 11.6.** The ROI is a GP-posterior
confidence region, `{x | UCB(x) >= max LCB}`. A confidence region built from
twelve observations in 6D is close to uninformative, and restricting a training
distribution to an uninformative region should do nothing — which is exactly what
Hartmann shows.

**If that is right, Hartmann is not a different kind of benchmark. It is Borehole
stopped early.**

## The test

Borehole's own run passes through Hartmann's entire HF budget on its way to 94.
So compute the paired ROI effect as a function of **cumulative post-init HF
count**, on both benchmarks, and overlay them.

## Predictions (locked)

**P1 (PRIMARY).** At the point in Borehole's run where cumulative HF count first
reaches 12 — Hartmann's *full-run* total — the paired ROI effect is **below 0.5
sd**, matching Hartmann's full-run 0.48 rather than Borehole's final 1.74.

**P2.** Borehole's effect grows with HF count, exceeding 1.0 by HF count ~50.

**Falsified if** Borehole's effect at HF count 12 is already >= 1.0. That would
mean Borehole differs from Hartmann in kind from the very first queries, and
truncation explains nothing.

## The confound, stated before looking

Within a single benchmark, cumulative HF count and elapsed cost are nearly
collinear (Borehole's HF count fraction is roughly constant at ~0.88). **So this
design CANNOT distinguish "the ROI needs accumulated HF observations" from "the
ROI needs elapsed budget".** Any early effect is small partly because there has
been less run in which to accumulate one, whatever the mechanism.

What the design CAN do is decide a narrower and still-useful question: **is
Hartmann's null a different phenomenon from Borehole's early run, or the same
one?** P1 tests exactly that and nothing more. I am registering this limit now so
that a P1 pass is not later inflated into a causal claim about HF data volume.

## Reporting

Report both benchmarks at every grid point, including where the effect is null
or reverses. n=5, no p-values.
