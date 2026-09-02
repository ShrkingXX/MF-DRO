# h174 — **SC1 FIRES. No verdict.** The arm is not a like-for-like comparison.

CONFIRMATORY, n=3 of 5 at time of reading. The gate fired before the regret was
interpreted, which is what it was registered for.

## What SC1 found

| arm | HF fraction | post-init queries | HF queries | LF queries |
|---|---|---|---|---|
| control (L=8) | 0.200 | **120** | 12 | 108 |
| **h174 (L=1)** | **0.776** | **34** | **24** | 10 |

On Hartmann `c_H=8, c_L=1`, so a fidelity shift changes what the budget buys.
L=1 makes **twice the high-fidelity queries and a third of the total**. That is
not the same experiment run faster — it is a different allocation of the same
budget.

**No verdict is issued on whether rollout_length=1 works on Hartmann.** The
regret numbers (7.58 vs the control's 7.99, improving 3/3) are recorded but not
interpreted: the direction of the confound is ambiguous in principle — more HF
should help, far fewer queries should hurt — so it cannot even be argued to run
against the claim, which is what rescued h165's confounded arm.

## The wall-clock figure also needs correcting

3.3 min against 94.0 is **not** a 28× rollout saving. Normalised by queries:

  control  94.0 / 120 = 0.78 min per query
  L=1       3.3 /  34 = 0.097 min per query   → **8.1×**

and even that is cross-run rather than contention-matched. The honest statement
is 8.1× per query, with the remainder of the wall-clock gap explained by making
a third as many queries.

## Why this did not happen on Borehole

h172's SC1 **passed**: HF 0.939 against the control's 0.883, and comparable query
counts. So the Borehole result stands as a clean like-for-like comparison and
h174 does not weaken it.

## Consequence for scope

**The actionable claim stays Borehole-only.** findings.md, research-state.yaml
and the published report state it without a benchmark qualifier; that is now
known to be unscoped rather than merely untested, and must be corrected.

## What would test it properly

An arm that holds the fidelity mix fixed while shortening the rollout — the same
device h155 used to isolate the location rule from the fidelity channel. Not
attempted here; registered as the way to do it.
