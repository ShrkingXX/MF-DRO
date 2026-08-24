# H18 — GATE FAILED (G2). Stopped as pre-registered. The failure is the finding.

| arm | distinct trajectories / 200 | argmax moved |
|---|---|---|
| `sample` (current) | 131 | 0/12 |
| `mean` (deterministic) | **62** | 0/12 |

- **G1 determinism: PASS** — mean-mode repeat-difference `0.000e+00` vs
  sample-mode `1.6e+00`. The intervention did exactly what it claimed.
- **G2 diversity preserved: FAIL** — 62 distinct trajectories, against a
  pre-registered floor of 150.

**H18 stops here.** The `mean` arm's 0/12 is **not interpreted**: with fewer than
half the distinct trajectories, a null could be caused by training-data
starvation rather than by anything about conditioning. That confound is precisely
what the gate existed to catch.

(The `sample` arm's 0/12 does reproduce H8 exactly on the same probe — a useful
instrument-consistency check.)

## Where my design reasoning was wrong

The protocol asserted:

> "this does **not** make the behaviour policy deterministic ... so rollout
> diversity is preserved."

That was wrong, and measurably so: diversity nearly halved. The teacher
(`compute_joint_mf_mes`) is itself deterministic given the KO model, so once the
transition stops injecting noise the entire rollout is very nearly determined by
its starting model. The fantasy draw was not only the MDP's transition — it was
**the dominant source of behavioural diversity in the training data**.

## The structural tension this exposes

This is worth more than the experiment I intended to run. In MF-DRO the fantasy
draw does **double duty**:

1. it *is* the MDP's stochastic transition — Brandfonbrener's `eps`; and
2. it is the main generator of trajectory diversity — which is what supplies
   return coverage `alpha_f`.

Their Corollary 1 bound is `eps * (1/alpha_f + 3) * H^2`. In GP-fantasy-rollout
RCSL these two quantities are **not independent**: driving `eps -> 0` also
shrinks the support that `alpha_f` is measured over. The two conditions of the
theorem are in direct conflict *in this class of method*.

That is a statement about the method class, not about our implementation, and it
survives regardless of how H17 turns out.

## Next

`eps` and `alpha_f` must be decoupled: keep the deterministic transition, but
restore diversity through the **behaviour policy**, which the theory explicitly
permits (`beta` may be stochastic). `rollout_policy` already supports
`thompson`/`random`/`cost_ei`/`ucb_beta1`/`ucb_beta3` (`mf_dro.py:1188-1231`),
so the mechanism exists. Locked separately as H19.
