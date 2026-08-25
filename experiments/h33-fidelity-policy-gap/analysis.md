# H33 — the fidelity channel is uninformative AND miscalibrated

24 decisions, same states and candidate pools for both.

| | value |
|---|---|
| teacher's `ell` (unconstrained argmax) | HF on **1/24 = 4.2%**, and it does vary |
| student's `p` | **0.5570 – 0.5577**, sd **0.000242** |
| `corr(p, teacher ell)` | **+0.1553** |
| implied student HF rate vs teacher's | **55.7% vs 4.2%** |

**PRED 1 PASS** — `|corr| = 0.155 < 0.2`. The student's fidelity probability
carries essentially no information about what the teacher would choose.
**PRED 2 holds** — the teacher's `ell` varies, so the comparison is not vacuous.

## Two separate defects

**Uninformative.** `p` varies by 0.0007 across 24 decisions (sd 2.4e-4) while the
teacher's choice genuinely varies. The fidelity head is a near-constant, so
`ell ~ Bernoulli(p)` is close to an unconditional coin flip. This is the same
inertness already established for the location head, now shown for fidelity on
the teacher-comparison axis rather than in isolation.

**Miscalibrated.** `p ≈ 0.557` against a teacher preference of `4.2%` on the same
pools — an order of magnitude apart in the *opposite* direction from what the
cost ratio implies (HF costs 8×).

## A caveat I wrote and then had to withdraw

I initially attributed the gap between this probe's 4.2% and H31's realised ~24%
HF to `minimum_hf_fraction = 0.25` acting as a floor. **That was wrong.** The
flag is applied only inside `simulate_mf_trajectory` (`mf_dro.py:1247`), i.e. to
the *rollouts*; the real proposal path applies no HF floor at all. Checked by
grep after writing the claim, and corrected here.

The comparison in this probe is therefore **like-for-like**: `p = 0.557` and the
teacher's `4.2%` are measured on the *same states and the same candidate pools*,
with no constraint mediating either. The ~13x gap at that operating point is
real.

What does not generalise is the *level*: the teacher's HF rate is state
dependent, rising from 4.2% at this early state to ~24% realised across a full
H31 run as the GP evolves. So the correct claim is a 13x gap **at the measured
state**, not a constant property of the two policies.

Also note `p` here (~0.557) differs from H21's measurement (~0.125) taken at a
later training state with different weights. `p` is near-constant *across states
at a given point in training*; it is not a fixed number across training.

## Bearing on H31

H32 named the fidelity choice as the remaining candidate for a student/teacher
gap. It is now confirmed as a real difference in *policy*, though the
`minimum_hf_fraction` floor damps how much of it survives into realised
behaviour. If H31 shows near-parity despite this, the floor is doing the work the
fidelity head fails to do.
