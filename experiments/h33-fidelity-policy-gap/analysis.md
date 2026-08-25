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

## Caveat that materially limits the second claim

The 4.2% figure is the teacher's **unconstrained** preference on fresh uniform
pools at an early training state. The full system applies
`minimum_hf_fraction = 0.25`, and the H31 run's realised mix is ~24% HF — so the
*system* does not actually query HF at 4.2%. The honest statement is therefore:

- **the student's fidelity head does not track the teacher's preference** (solid,
  `corr = 0.155`, and the near-zero sd is decisive on its own);
- **`p = 0.557` sits far from the teacher's raw preference of 4.2%** (measured),
  but a floor constraint sits between raw preference and realised behaviour, so
  this is not a 13x discrepancy in what the two *do*.

Also note `p` here (~0.557) differs from H21's measurement (~0.125) taken at a
later training state with different weights. `p` is near-constant *across states
at a given point in training*; it is not a fixed number across training.

## Bearing on H31

H32 named the fidelity choice as the remaining candidate for a student/teacher
gap. It is now confirmed as a real difference in *policy*, though the
`minimum_hf_fraction` floor damps how much of it survives into realised
behaviour. If H31 shows near-parity despite this, the floor is doing the work the
fidelity head fails to do.
