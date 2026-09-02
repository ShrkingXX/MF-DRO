# h156c -- does the tail account generalise to HARTMANN?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

h156/h156b reproduced all four Borehole rtg_targets from trajectory geometry.
That is one benchmark. The account is stated in findings.md without a
benchmark qualifier, so it owes a second one.

Hartmann_6D has three arms with rtg_target already serialised:
  control      0.8844   (h83, seeds 42-46)
  ORACLE       0.3622   (h145, seeds 42-46)
  RANDOM-POOL  (read at launch, h149, seeds 42-46)

Run the SAME harness, unchanged, on Hartmann control traces. Targets to match:
C1 -> control, C3 -> RANDOM-POOL, C4 -> ORACLE. C2 and C5 have no Hartmann
counterpart and are recorded as forecasts only.

Note Hartmann's control target (0.8844) is LOWER than Borehole's (0.9761) while
its ORACLE target (0.3622) is HIGHER than Borehole's (0.3113) -- a narrower
spread. If the harness tracks that narrowing rather than just reproducing two
extremes, that is stronger evidence than matching Borehole was.

## What this can RETRACT

R1 harness misses Hartmann's targets -> the account is Borehole-specific and
   findings.md must be narrowed to say so. The Borehole agreement would then
   be at risk of being a coincidence of one benchmark's geometry.
R2 harness matches the two extremes but NOT the narrowing -> it reproduces
   scale, not structure; report as partial.
R3 harness matches all three including the narrowing -> the account generalises
   and the h153 forecast stands on two benchmarks rather than one.

## Compute

1 worker, offline harness, minutes. h153 (5) + h155 (5) still running = 11 <= 15.
