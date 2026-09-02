# h173 -- does the HEAD/TAIL result replicate on Hartmann?

STATUS: protocol locked, nothing run (queued: compute at 15/15).
TYPE: CONFIRMATORY.

## Why this before the residual

h171 answered the front by intervention, and it is **Borehole-only**. Every other
load-bearing result on this front has been made to replicate on Hartmann —
the 2x2 (h165/h166), the dispersion collapse (h164), the open-loop penalty
(h154b) — and in two of those cases the second benchmark changed what could be
claimed (h164's HF-only slicing was unusable; h165's ordering flipped).

The alternative use of this compute is chasing h170's ~5 SE residual. That is a
refinement of an account already established at the level that matters; a
one-benchmark headline is a bigger exposure.

## The arms

Identical to h171, on Hartmann_6D seeds 42-46:

  HEAD-MES   tau=0 MES argmax, tau=1..7 uniform random
  TAIL-MES   tau=0 uniform random, tau=1..7 MES argmax

## Predictions

P1 HEAD works, near the Hartmann control's 7.99, improving ~5/5.
P2 TAIL fails, near RANDOM-POOL's 65.14, improving ~0/5 or 2/5.
P3 Query centroids: HEAD far from the box centre (control 0.6503, in 6D),
   TAIL near it (RANDOM-POOL 0.0830).

## What this can RETRACT

R1 The ordering does NOT reproduce (TAIL >= HEAD) -> h171's interventional
   result is Borehole-specific, and the front's ANSWER — currently stated in
   findings.md, research-state.yaml and the published report **without a
   benchmark qualifier** — must be scoped to one benchmark. This is the exposure
   that justifies spending the arm.
R2 It reproduces -> the answer holds on two benchmarks and two failing teachers.
R3 Both arms intermediate -> inconclusive at n=5, reported as such.

## Named confounds, checked before the numbers

SC1 realised HF fraction against the Hartmann control's **0.200**. Note this
    reference is itself unstable on Hartmann (per-seed 0.038-0.750, per h164),
    so the check is weaker here than on Borehole and will be reported as such
    rather than treated as decisive.
SC2 h171's TAIL had its HF fraction collapse to 0.217 and SC1 fired. Expect the
    same here; the attribution will again rest on the arms that fail WITHOUT a
    collapse (Hartmann ORACLE 0.261, RANDOM-POOL 0.281).

## Design

Hartmann_6D seeds 42-46, both arms, 10 workers. Launch when h172's L=1 group
(currently 90% done) frees capacity. No code changes needed: h171's worker takes
the benchmark as argv.
