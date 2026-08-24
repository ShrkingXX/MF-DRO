# H19 — decouple `eps` from `alpha_f`

## Why

H18 established (G1 PASS) that `fantasy_mode="mean"` genuinely determinises the
transition, but its gate failed (G2) because doing so halved trajectory
diversity: 131 -> 62 distinct. In MF-DRO the fantasy draw is *both*
Brandfonbrener's `eps` **and** the generator of return coverage `alpha_f`.

Their theory permits a **stochastic behaviour policy** — `eps` bounds the MDP's
transition/reward, not `beta`. So the two can be decoupled by moving the noise
from the transition into the policy.

## Intervention

`fantasy_mode="mean"` (deterministic transition, verified by H18's G1)
**+** `rollout_policy="thompson"` (stochastic action selection;
`mf_dro.py:1188`).

Control arms, all on the identical probe used by H8 and H18:

| arm | fantasy_mode | rollout_policy | role |
|---|---|---|---|
| S-mes | sample | mes | current default; must reproduce 0/12 |
| S-thom | sample | thompson | isolates the policy change alone |
| M-mes | mean | mes | = H18's failed arm, for continuity |
| **M-thom** | **mean** | **thompson** | **the candidate** |

## GATE (unchanged floor, so it cannot be moved to fit)

**G2 diversity**: M-thom distinct trajectories **> 150 / 200** — the same
threshold H18 failed, stated before running. If M-thom also fails it, then
`eps` and `alpha_f` cannot be decoupled by this mechanism in this method, which
is itself a reportable structural result and H19 stops.

G1 is inherited from H18 (determinism already verified, `0.000e+00`).

## Locked predictions (only if G2 passes)

1. **PRIMARY**: M-thom moves the argmax on **> 30%** of pools, vs 0/12 for
   S-mes. First movement in the investigation.
2. **ATTRIBUTION**: S-thom must be **< 30%**. If the policy change alone
   produces the movement, the result is about exploration, not determinism, and
   prediction 1 must be restated accordingly.
3. **NULL**: if M-thom is still ~0%, then with `eps ~ 0` and diversity restored
   — the regime RCSL theory says should work — the DT *still* does not
   condition. The cause is then the score-head bottleneck (H5: swapping `h`
   changes the argmax 0/12), not RCSL's preconditions. This is the strongest
   available form of the negative result.

## Scope

Probe only. No regret claim, `PROTOCOL.md` untouched. Single process, 1 thread
(runs alongside H17's 10 workers; 11 <= 15).
