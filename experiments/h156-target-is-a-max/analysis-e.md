# h156e -- mechanism CONFIRMED, gate FAILED, and the gate had a HOLE

## The gate did not cover the outcome

Pre-stated clauses were:
  PASS     mean |err| < 10% AND worst < 15%, both replicates
  FAIL     C4/C5 stay 20%+ under
  PARTIAL  errors shrink but C4/C5 remain outside 15%

Observed: mean |err| 9.4% (inside PASS), worst 29.4% (outside PASS), and C4/C5
came *inside* 15% -- so neither FAIL nor PARTIAL applies either. **None of the
three clauses matches.** That is a gate with a hole, the exact failure mode
tools/check_gate.py exists to catch, and I did not run it on this protocol.
Recorded as a discipline miss. The outcome is reported descriptively below
rather than forced into a clause it does not fit.

By the strict reading of PASS (an AND, both conditions required) this is a
**FAIL**.

## What the ensemble actually did

| condition | OLD (1 GP) | NEW (10-GP ensemble) | observed | OLD err | NEW err |
|---|---|---|---|---|---|
| C1 control | 0.906 / 0.962 | 0.954 / 1.008 | 0.9761 | −4.3% | **+0.5%** |
| C3 random | 0.274 / 0.257 | 0.403 / 0.365 | 0.2965 | −10.6% | **+29.4%** |
| C4 oracle | 0.302 / 0.238 | 0.270 / 0.308 | 0.3113 | −13.3% | **−7.1%** |
| C5 diverse-good | 0.265 / 0.247 | 0.310 / 0.341 | 0.3285 | −22.0% | **−0.8%** |

**The specific prediction was confirmed.** h156e predicted the missing
between-model variance would hurt C4/C5 most, because their within-condition
spread is smallest. It did, and adding the ensemble fixed exactly those:
C5 −22.0% → −0.8%, C4 −13.3% → −7.1%. The **one-sided** bias is gone: errors
are now two-sided (+0.5, +29.4, −7.1, −0.8) instead of uniformly negative.

**And it broke the condition that previously fit.** C3 (random) now
over-predicts by +29.4%. That is above the harness's own worst-case noise
(13.2%), so it is a real over-prediction and it is unexplained.

## Net: not a better instrument

  mean |error|   12.5% -> 9.4%      modest gain
  worst |error|  22.0% -> 29.4%     worse
  noise floor    6.1% / 10.9%  ->  8.4% / 13.2%     worse

Accuracy improved by about as much as the noise floor rose. **The ensemble
harness is not meaningfully more trustworthy than the single-GP one**, and I am
not adopting it as the reference. Both versions are retained.

## The honest characterisation of this instrument

Across four harness variants, per-arm errors run from 0.5% to 31% with an
8-13% noise floor. **This harness supports SCALE claims, not numeric ones.**
The finding it does support -- control 0.79-1.01 against every failing arm
0.24-0.40, a 3-4x gap on two benchmarks across six independent runs -- is far
outside every error and noise figure above. The claim that it "reproduces the
observed targets" should be read as reproducing their SEPARATION, not their
values.

## h153 forecast

C2/C1 over four measurements: 90.9, 95.5, 95.9, 87.3%. Range 87.3-95.9%,
against an 8.4% noise floor. Freezing costs 4-13% of the tail; the failing arms
lose 60-75%. **The forecast stands** -- h153 should not collapse its target --
but stated as 85-96%, not the tight 90.9% I quoted when it rested on one run.
