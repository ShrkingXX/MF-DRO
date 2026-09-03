# h192 — **P1. The mechanism is now INTERVENTIONAL, not correlational.**

**CONFIRMATORY** against the protocol committed before the code was written, with the
identity gate, the SC, and the readout script all committed before results. 5/5 runs,
no failures.

## SC read first, as registered

| | teacher τ=0 mean, dist from box centre |
|---|---|
| control (ROLLOUT1) | **0.7837** |
| shifted (TAU0SHIFT) | **0.3645** |

**SC PASS.** The imposed shift is **+0.4192**.

## The registered statistic

> **transfer ratio** = (control DT centroid − shifted DT centroid) ÷ (imposed teacher shift)
> **P1 ≥ 0.50 · P2 0.15–0.50 · P3 < 0.15 (mechanism falsified)**

| | value |
|---|---|
| DT query centroid, control | 0.8546 |
| DT query centroid, shifted | **0.3961** |
| teacher shift **imposed** | **+0.4192** |
| DT shift **observed** | **+0.4585** |
| **TRANSFER RATIO** | **1.094** |

**P1, and not marginally.** The gate required ≥0.50; the DT reproduced the imposed shift
**essentially one-for-one**. h191 measured ≈0.70 for ROI's naturally-occurring shift;
under a direct, deliberate translation the DT tracks it completely.

**This converts the central claim of the front from correlational to interventional.**
h185, h188, h182 and h191 all *observed* that the DT's query tracks its teacher's τ=0
mean. This *moved* that mean and the DT moved with it.

## The secondary prediction also fired, and hard

| | frozen rel% | improves on its OWN initial design |
|---|---|---|
| control (ROLLOUT1) | **13.69** | **5/5** (init best 173.54 → 267.20) |
| shifted (TAU0SHIFT) | **43.18** | **1/5** (init best 173.54 → **175.91**) |

Pushing the constant to the centre cost **+29.49 rel% points** and destroyed the
policy's ability to beat its own starting design — 5/5 → 1/5, with the final best value
barely moving off the initial one. 43.18 sits just under the 43.94 that h182 identified
as *being* the initial design.

So the full chain, established by intervention rather than observation:
**move the teacher's τ=0 mean → the DT's emitted query follows one-for-one → and where
that constant lands determines whether the policy contributes anything at all.**

## Limits, and the confound this design carries

- **The shift direction is not neutral.** I translated toward the box centre, which h182
  had already identified as a bad region. So the regret result is a **joint** test of
  "the DT follows" and "the centre is bad" — it does not independently establish the
  second. A shift of the same magnitude in a *neutral* direction would separate them,
  and is the obvious next test. The **transfer ratio** is unaffected by this: it measures
  only whether the query follows, not whether following is harmful.
- **Ratio 1.094 slightly exceeds 1.** At n=5 this should not be read as overshoot; the
  honest statement is "one-for-one within the resolution available".
- One shift magnitude (λ=0.5), one direction, one benchmark, n=5, rollout_length=1.
  L=1 was chosen so the recorded mean *is* the τ=0 mean; whether the same ratio holds at
  L=8, where seven other steps dilute the signal, is untested.

## What this RETRACTS

Nothing. P3 would have invalidated h185/h188/h191, findings.md's Phase 2 header and the
report's core section — that was named in the protocol before launch and did not occur.
The mechanism stands, and stands more strongly than before.
