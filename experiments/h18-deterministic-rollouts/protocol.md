# H18 — the ONE intervention RCSL theory says should work

## The reasoning

Brandfonbrener et al. (2022, arXiv:2206.01079) Corollary 1 bounds RCSL's
suboptimality by `eps * (1/alpha_f + 3) * H^2`, where `eps` measures departure
from **deterministic dynamics**. Their Figure 1c shows that when `eps` is large
the bias survives **regardless of the conditioning function** — which is exactly
what our seven conditioning-side interventions found (H4, H5, H8, H9, H10, H11,
H12–H16).

MF-DRO's rollout transition is `y_tau = sample_fantasy(x_tau, .)`, a **draw from
the GP posterior**. `eps` is maximal by construction.

**So the theory predicts one intervention should work where all the
conditioning-side ones could not: make the dynamics deterministic.**

## Intervention (one variable)

`fantasy_mode="mean"` — the rollout transition uses the posterior **mean**
instead of a posterior **sample** (`ko_gp.sample_fantasy`). Certainty-equivalent
rollouts.

Critically, this does **not** make the behaviour policy deterministic.
Brandfonbrener's `eps` concerns the MDP's transition/reward, not `beta`, so
action-selection stochasticity (candidate draw, fidelity Bernoulli) is left
alone and rollout diversity is preserved. Getting this distinction wrong would
collapse 200 trajectories to 10.

## GATE — run first, stop on failure

- **G1 determinism**: replaying a fixed action sequence twice yields identical
  `y` values (max abs diff < 1e-12).
- **G2 diversity preserved**: distinct trajectories in a 200-rollout batch stays
  **> 150**. If determinism has collapsed the batch, the intervention has traded
  stochasticity for data starvation and the comparison is confounded — report
  and stop.

## Locked predictions (only if the gate passes)

1. **PRIMARY**: an in-band RTG sweep moves the argmax on **> 30%** of candidate
   pools, versus the **0/12** measured under stochastic rollouts (H8) on the
   identical probe. This would be the **first** intervention in the entire
   investigation to move the decision.
2. **NULL — and this is the more valuable outcome to state in advance**: if the
   argmax still does not move under deterministic dynamics, then near-determinism
   is *not* the binding constraint either, and the negative result becomes
   materially stronger: MF-DRO's transformer fails to condition **even in the
   regime where RCSL theory says it should succeed**. That points the cause at
   the score-head bottleneck (H5) rather than at RCSL's preconditions.

Both outcomes are informative; neither is a failed experiment.

## Scope

Probe only — no regret comparison, no change to `PROTOCOL.md`. Whether
certainty-equivalent rollouts help *regret* is a separate question, and would
carry a real cost (mean-fantasy under-represents posterior uncertainty, making
the rollout over-confident). Not claimed here.

## Compute

Single process, 1 thread, run alongside H17's 10 workers (11 <= 15). Reuses
H8's probe code so the 0/12 comparison is instrument-identical.
