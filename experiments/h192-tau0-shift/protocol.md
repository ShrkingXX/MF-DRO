# h192 — the causal test. Move the teacher's τ=0 mean and see whether the DT follows.

**CONFIRMATORY.** Committed before the code is written and before any result exists.

## Why this is the arm that matters

The mechanism — *the DT emits its teacher's τ=0 action mean* — is supported by h185
(best-constant identity, 10 arms, two benchmarks), h188 (the all-τ mean predicts the
emitted query when it equals the τ=0 mean, with two by-construction controls), h182 (the
HEAD/TAIL inversion) and h191 (every successful intervention moves that mean).

**All of it is correlational.** Nothing has ever moved the teacher's τ=0 mean directly
and checked that the DT's query moves with it. h191's limits section names exactly this
as the missing experiment. This is it, and it is the first arm capable of **falsifying**
the mechanism rather than adding to it.

## The intervention

At **τ=0 only**, after the teacher has chosen `x_tau` by its ordinary rule, translate it a
fixed fraction toward the box centre:

```
x' = x + λ (c − x),   c = ½(bounds[0] + bounds[1]),   λ = 0.5
```

Later steps are untouched. Disabled by default (`None`), so no existing configuration can
change behaviour. Run at **rollout_length = 1**, where the recorded all-τ teacher mean
**is** the τ=0 mean, so the shift is directly measurable rather than averaged away.

Borehole, seeds 42–46, frozen metric. Control is **ROLLOUT1**, already run:
teacher τ=0 mean **0.7788** from centre, DT query centroid **0.8484**, rel% **13.69**.

## SC before the regret

The teacher's τ=0 mean distance from the centre must **approximately halve** (0.7788 →
≈0.39). If it does not, the shift did not apply and nothing may be read.

## Gate — on the TRANSFER RATIO

> **statistic** = (control DT centroid − shifted DT centroid) ÷ (control teacher mean
> dist − shifted teacher mean dist)
>
> i.e. what fraction of the teacher's imposed shift the DT's own query reproduces.
> h191 measured **≈0.70** for ROI's naturally-occurring shift.

- **P1 — mechanism HOLDS**: ratio **≥ 0.50**
- **P2 — partial**: **0.15 ≤ ratio < 0.50**
- **P3 — mechanism FALSIFIED**: ratio **< 0.15**

Partitions the real line below 0.50 with no gap. A ratio near 1.0 is the mechanism's
strongest possible confirmation; near 0 means the DT does **not** track its teacher's
τ=0 mean and the account is wrong.

**Secondary, reported alongside:** regret should **degrade**, since the constant is being
moved toward the box centre and h182 established the centre cannot beat the initial
design. If regret does *not* degrade while the query does move, that is itself worth
recording — it would separate "the DT follows the mean" from "where the mean sits
determines performance".

## What this could RETRACT

- **P3 fires → the central claim of this entire front is false.** h185/h188/h191 would
  all need rewriting, `findings.md`'s Phase 2 header would need replacing, and the
  published report's core section would be wrong. This is by far the most consequential
  outcome available and it must be reported plainly and immediately if it occurs.
- **P2 fires → the mechanism is real but weaker than stated**; "the DT emits its
  teacher's τ=0 mean" would need qualifying to "partially tracks".
- P1 converts the front's central claim from correlational to **interventional**.

## Prerequisite

`use_roi=False` bit-identity (`tools/identity_gate.py`, reference
122.29066752728207) must PASS on the patched file before launch.

## Compute

5 workers × 1 thread. Machine idle.

## SC — PASS, recorded before launch

```
teacher tau=0 mean dist from centre = 0.3397
control (ROLLOUT1, full run)        = 0.7788
```

The shift applied: **0.7788 → 0.3397**, a factor of 0.436 against the 0.5 the λ=0.5
translation targets. (Not exactly 0.5 because the translation is applied per-action and
then averaged, and the mean of the shifted points is not the shifted mean's distance.)
The imposed teacher shift used in the gate's denominator is the **measured** one from
the full runs, not the nominal 0.5.
