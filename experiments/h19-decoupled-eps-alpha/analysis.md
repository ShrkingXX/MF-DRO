# H19 — gate PASSES, and the pre-registered NULL fires. The strongest negative yet.

Corrected instrument (signature = query coordinates + fidelity pattern):

| arm | fantasy_mode | rollout_policy | distinct traj | argmax moved |
|---|---|---|---|---|
| S-mes | sample | mes | 200/200 | 0/12 |
| S-thom | sample | thompson | 200/200 | 0/12 |
| M-mes | mean | mes | 200/200 | 0/12 |
| **M-thom** | **mean** | **thompson** | **200/200** | **0/12** |

- **G1 determinism**: inherited from H18 — mean-mode repeat-difference
  `0.000e+00` vs sample-mode `1.6e+00`. Verified.
- **G2 diversity**: **PASS**, 200/200 (threshold >150, unchanged).

**PRED 1 FAIL** (M-thom 0.0%, needed >30%).
**PRED 3 NULL FIRES** — and it was pre-registered as the more valuable outcome.

## What the null establishes

With **deterministic dynamics** (`eps ~ 0`, verified) **and** full trajectory
diversity (200/200 distinct), the Decision Transformer **still does not
condition**. That is the exact regime in which Brandfonbrener et al.'s theory
says RCSL should work — and it does not.

**Therefore near-determinism is not the binding constraint here.** The proximate
cause is the one H5 already isolated: the score head barely reads its hidden
state (swapping `h` for another state's changes the argmax 0/12). Nothing that
reaches `h` — RTG, BTG, state, history, or the reward that generates them — can
move a decision that is not a function of `h`.

## This TEMPERS my RCSL framing from earlier today

Last tick I wrote that RCSL theory "predicts everything we measured" and that
"no architectural conditioning fix could have worked here". That was too strong,
and H19 is the check that catches it:

- **Still true**: RCSL theory explains why conditioning-side fixes fail under
  *stochastic* dynamics, which is the regime every prior experiment ran in.
- **Now shown false**: it is not the *whole* explanation. Removing the
  stochasticity does not restore conditioning, so RCSL's near-determinism
  condition is not what binds in MF-DRO.

The honest claim is narrower and better supported: *the failure is architectural
(the score-head bottleneck), and RCSL theory explains why the many conditioning-
side remedies could not have rescued it.*

## RETRACTION — H18's structural claim was an instrument artifact

H18 concluded that in GP-fantasy-rollout RCSL the fantasy draw is *both* `eps`
and the generator of `alpha_f`, so the theorem's two conditions conflict. That
rested on measuring diversity collapse 131 -> 62.

**Both numbers were wrong.** The signature was `(rounded rtg, ell)` and omitted
the query locations entirely, so any two rollouts with a dead reward (63% of
them) and a matching fidelity pattern counted as identical. With locations
included, every arm is 200/200 — **diversity does not collapse under
deterministic dynamics at all**.

`eps` and `alpha_f` are **not** coupled in the way H18 claimed. That claim is
withdrawn from `findings.md` and `research-log.md`. H18's G1 (determinism
verified) stands; its G2 and everything built on it does not.

The threshold (>150) was never moved — only the measurement was corrected.
