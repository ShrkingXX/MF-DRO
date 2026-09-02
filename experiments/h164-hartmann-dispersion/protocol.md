# h164 -- does the dispersion collapse replicate on HARTMANN?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY. Zero compute:
existing serialised data only.

## Why

h162's dispersion split and h163's inversion are the learnability framing's only
observable predictions, and **both were measured on Borehole alone**. The whole
2x2, h158's dose and h162/h163 are Borehole. If the split does not replicate,
the framing's evidence base is one benchmark.

Hartmann has three arms with x_t_trace serialised at 5/5: control, ORACLE,
RANDOM-POOL. DIVERSE-GOOD was never run there.

## Prediction

Same split as Borehole: **control HIGH dispersion; ORACLE and RANDOM-POOL LOW**,
with arm-level separation. Borehole reference: working 0.2464-0.2889, failing
0.1115-0.1891.

Measured three ways as in h162 -- all queries, HF only, and first-100 (n matched)
-- because the failing arms carry more queries and different fidelity mixes.

## Named caveat, recorded before the numbers

**Hartmann's ORACLE arm is flagged confounded in findings.md** (h145's Hartmann
run; the degenerate-y* fix was not re-run there because RANDOM-POOL had already
answered the scope question). So the load-bearing contrast here is
**control vs RANDOM-POOL**. ORACLE is reported for completeness and explicitly
NOT counted as independent evidence.

## What this can RETRACT

R1 no split on Hartmann -> h162's dispersion collapse is Borehole-specific and
   the learnability framing loses its only cross-benchmark observable.
   findings.md must scope the framing to one benchmark.
R2 split replicates -> the framing's observable generalises. Still a correlate;
   h161 remains the causal test.
R3 split in the opposite direction -> actively contradicts, same as h162's R3.

## Caveat, unchanged

Query-level statistics at n=5: the class that produced h150 (retracted) and
h154's refuted M2. Replication on a second benchmark strengthens it within that
class; it does not move it to a stronger one.
