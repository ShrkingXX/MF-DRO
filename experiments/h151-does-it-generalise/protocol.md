# h151 — Does "any non-MES teacher fails totally" hold beyond Borehole?

STATUS: LOCKED before running.
TYPE: CONFIRMATORY. A **boundary test of a measured result**, not a fourth
mechanism account.

## Why this and not another explanation

Three mechanism stories have been tried and discarded on this front: RCSL return
coverage (h147, falsified), information-gain-as-ranking (h149, corrected), policy
distillation (h150, falsified). The project's wider record is four-for-four the
same way. **Generating a fourth until one sticks is what h148's stopping rule was
written to prevent.**

What is NOT exhausted is the measured result's **boundary**. Everything —
h145, h146, h149, h150 — is Borehole. I have flagged that limitation in every
write-up without testing it. A negative result whose scope is untested is worth
much less than one whose scope is known.

## The test

`rollout_policy="random"` on **Hartmann_6D**, seeds 42-46, same h83 config. This
is the h149 arm on a second benchmark.

**Why RANDOM-POOL and not the oracle.** h145's Hartmann arm was confounded — its
degradation tracked fidelity collapse at corr -0.830 — because `forced_x` changed
which fidelity the MES criterion picked at the forced point. RANDOM-POOL uses
**pre-existing code** (`mf_dro.py:1567`) whose fidelity rule is a fixed 25% coin
flip, independent of location. It cannot reproduce that confound, and it does not
touch the hook I wrote.

Hartmann is also the harder case for the claim: it is HF-starved (11.6 post-init
HF queries against Borehole's 93.4), so its control has far less room to
demonstrate improvement in the first place.

## Predictions (locked)

**P1.** RANDOM-POOL on Hartmann never improves on its initial design: **0/5**.
FALSIFIED if it improves on 2 or more seeds.

**P2.** Its regret degradation versus the h83 MF-DRO control is separable, effect
>= 1.0. Reported separately from P1 because on Hartmann the control itself only
improves on some seeds — a degradation gate and an improvement gate can disagree
there, and I want both on the record.

**P3 (no direction).** `rtg_target`, oracle-style collapse or not. On Borehole
every non-MES teacher collapsed it to ~0.30 from 0.976. Reported whatever it does.

## What this could RETRACT

**The scope of the answer, not the answer.** If RANDOM-POOL improves on Hartmann
while failing totally on Borehole, then "any non-MES teacher fails totally" is a
**Borehole statement**, and the front's headline must be restated with that
boundary attached — the same correction the ROI results needed when four
benchmarks turned out to have one positive cell each.

**A specific risk I am naming in advance:** Hartmann's control improves on fewer
seeds than Borehole's, so a 0/5 for RANDOM-POOL there is **less informative** than
the same figure on Borehole. If the control also improves rarely, P1 passing
proves little. I will report the control's own improvement count beside it rather
than quoting P1 alone.
