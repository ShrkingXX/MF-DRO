# H79 result — NULL fired. The identity is 8/10, not exact.

**CONFIRMATORY** against `protocol.md`.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | bit-for-bit on >= 9 of 10 seeds | **8/10** | **NOT MET** |
| SECONDARY | where they differ, \|diff\| < 0.5 pts | max **3.00** pts | **NOT MET** |
| NULL | matches <= 8 of 10 | fired | **restate the claim** |

| seed | SF-EI@1000 | MI-Greedy | match |
|---|---|---|---|
| 42 | 12.2269% | 12.2269% | EXACT |
| 43 | 7.7806% | 7.7806% | EXACT |
| 44 | 7.1494% | 7.1494% | EXACT |
| **45** | **8.7763%** | **7.5513%** | **differs by 1.23 pts** |
| 46 | 6.7591% | 6.7591% | EXACT |
| 47 | 10.0934% | 10.0934% | EXACT |
| 48 | 10.9018% | 10.9018% | EXACT |
| **49** | **11.9494%** | **8.9450%** | **differs by 3.00 pts** |
| 50 | 10.7353% | 10.7353% | EXACT |
| 51 | 10.7143% | 10.7143% | EXACT |

## The claim must be weakened

h70 concluded, from seeds 44/46/48, that **MI-Greedy's advantage over the
single-fidelity MES baseline is *entirely* candidate pool size**. Those three
seeds are among the eight that match exactly. At n=10 the identity holds on
**8 of 10**, and on the other two SF-EI@1000 is **worse** — by 1.23 and 3.00
points.

**Restated:** on Borehole, MI-Greedy reduces to single-fidelity EI with a
1000-point pool on **most** seeds but not all. Pool size explains the great
majority of its advantage; something else contributes on a minority of runs, and
that something makes MI-Greedy *better*, not worse.

This is a weakening, not a reversal. The mean claim survives — MI-Greedy 9.29%
vs SF-EI@1000's mean over these seeds — and the mechanism is still overwhelmingly
pool size. But "entirely" was an n=3 word and it is now wrong.

## EXPLORATORY — a proposed cause, checked and refuted immediately

The obvious candidate: MI-Greedy's LF phase, which is *mostly* inert at ~100% HF
on Borehole but need not be always. If it inserted LF queries on seeds 45 and 49,
the trajectories would diverge.

**Refuted by the recorded iteration counts.** Both methods run 100 iterations on
both divergent seeds. And seed 43 has MI-Greedy at **99** iterations against
SF-EI's 100 yet matches **exactly** — so iteration count does not track
divergence at all.

A remaining candidate, **untested**: MI-Greedy's `_explore_lf` samples candidates
and computes information gains *even when it selects no LF point*, consuming RNG
draws that SF-EI never makes. Once the two candidate-pool streams diverge, the
EI argmax can differ. That would explain why divergence is seed-dependent and
why it appears without any change in iteration count. It is a hypothesis, and
this project's record on untested mechanism hypotheses is six refuted out of six.

## What this changes downstream

`findings.md` and `to_human/h57_failure_modes.md` both state the "entirely pool
size" version. Both must be corrected to 8/10.
