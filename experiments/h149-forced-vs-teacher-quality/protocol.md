# h149 — Is it MY forcing, or is it any non-MES teacher? (exoneration control)

STATUS: LOCKED before running.
TYPE: CONFIRMATORY. **This is a control on my own implementation.**

## Why this must be run before anything else is concluded

h145 (ORACLE) and h146 (DIVERSE-GOOD) both fail **totally**: neither ever improves
on its random initial design, 0/5 and 0/4, while the control improves 5/5. Their
post-init best HF is *below* the initial best in every run.

**A binary failure across 9 of 9 runs is the signature of a broken implementation
at least as much as of a real effect.** Both arms route through the `forced_x`
hook I added. I have checked what I can by inspection — `actions_x` derives from
the overridden `x_tau`, the candidate-scoring block that would break on a stale
`cand_idx` is disabled, SC4 shows the default path bit-identical — but inspection
is what h136 exists to distrust.

## The control

`rollout_policy="random"` is **built-in, pre-existing, and does not touch
`forced_x` at all**. It selects a uniformly random candidate from the same
`roi_candidates` pool the MES teacher argmaxes over. So it is a **non-MES teacher
implemented by code I did not write**.

    CONTROL        MES argmax over the pool     improves 5/5
    RANDOM-POOL    uniform from the same pool   ???        <- this arm
    DIVERSE-GOOD   forced, diverse good ends    improves 0/4
    ORACLE         forced, all end at x*        improves 0/5

## Predictions (locked)

**P1 (PRIMARY, and it is a fork).**
- If **RANDOM-POOL also fails totally** (improves 0/5, post-init best below
  initial best), then total failure is caused by **any non-MES teacher**, my
  `forced_x` hook is exonerated, and the h145/h146 results stand as measurements
  of teacher choice rather than of my code.
- If **RANDOM-POOL improves** (>= 3/5) while both forced arms fail 0/5, then the
  failure tracks `forced_x` and **not** teacher quality. h145 and h146 would then
  be measuring a defect in my implementation, and both must be withdrawn.

FALSIFIED as a clean fork if RANDOM-POOL lands in between (1-2 of 5); that outcome
is reported as INDETERMINATE and neither branch is claimed.

**P2 (no direction).** RANDOM-POOL's `rtg_target`. Under the information-gain
account it should sit below the control's 0.976, since a random teacher earns less
information gain than the argmax. Reported whatever it shows.

## What this could RETRACT

**Everything h145 and h146 measured, and the synthesis built on them** — the
"teacher already optimal in the rewarded currency" account, the report section
published to the user, and the answer given to the question this front was opened
to settle. If P1's second branch holds, all of it is an artefact of my hook.

**This is the outcome I should want to find if it is true**, and it is the reason
this control runs before the POOL dose or any further arm. I have already reported
one confounded result as clean in this experiment (Hartmann) and one bug the user
caught (the degenerate y*). A third would be too many to keep asserting the
account without this check.
