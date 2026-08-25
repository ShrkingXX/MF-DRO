# H43 — is the inactive-DT finding landscape-specific? Test on Ackley 10D.

## Why

Every "DT is inactive" measurement was made on Hartmann 6D. The historical
Ackley evidence that motivated a landscape explanation is **pre-leak-fix**
(Aug 20 vs the fix on Aug 24), and post-fix Hartmann barely freezes (1/30), so
the original contrast has dissolved. Ackley has **never** been run post-fix.

## Design

Ackley 10D (`d=10`, `c_H=10`, `c_L=1`), 3 seeds, 100 iterations, both scoring
heads (`use_candidate_scoring` in {True, False}). After training, probe the DT
with the **same measurements used on Hartmann**, so the numbers are directly
comparable:

| metric | Hartmann value |
|---|---|
| argmax changed on genuine state swap | 0/12 |
| `‖δ‖ / ‖w̄‖` (state's share of the coefficient vector) | 0.00129 |
| `w̄` alone reproduces the argmax | 12/12 |
| relative spread: state → h → w | 0.2155 → 0.0745 → 0.0219 |
| fidelity `p` spread across states | 2.4e-4 |

For the **regression** arm the analogue is the spread of `action_head(h)` across
states — there is no `w` to decompose.

## Locked predictions

1. **PRIMARY (not landscape-specific)**: on Ackley, `‖δ‖/‖w̄‖ < 5%` and the
   state-swap argmax change is `< 30%`. That would make the inactive-DT finding
   architectural, not a property of the Hartmann landscape.
2. **FALSIFICATION (landscape-specific)**: if Ackley shows `‖δ‖/‖w̄‖` an order of
   magnitude larger AND state-swap movement `> 30%`, the DT *does* condition on
   Ackley and the Hartmann result is landscape-bound — which would substantially
   change the paper's scope claim.

## Compute note

12 workers already busy (H40, H42). This adds **3** → exactly 15, at the rule's
limit. Each seed runs both arms sequentially.
