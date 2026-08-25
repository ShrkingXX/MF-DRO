# H45 — does the DT's OWN proposal match its teacher, with no pool and no argmax?

## Hypothesis under test (user's)

MF-DRO's edge over MF-MES (0.4007 vs 0.4781) comes from the **fresh candidate
pool + acquisition argmax**, not from the Decision Transformer. Remove that
machinery -- let the DT's regression head emit the query point DIRECTLY -- and
performance should fall back to roughly teacher level, because all the DT can
contribute is its imitation of the teacher's choices.

## Design: one new arm completes an existing three-way comparison

Arms A and C already exist at IDENTICAL settings (Hartmann 6D, seeds 42-51,
cost budget 200, initial design n_HF=36 / n_LF=60):

| arm | who picks the point | pool + argmax? | status |
|---|---|---|---|
| **A** MF-DRO, candidate scoring | `argmax_k <w(h), cf_k>` | yes | **done** (h17): 0.4007 +/- 0.0475 |
| **C** MF-MES teacher, no DT | `argmax_k MES(cf_k)/c` | yes | **done** (h31): 0.4781 +/- 0.0414 |
| **B** MF-DRO, regression head | `x = action_head(h)` | **NO** | **this experiment** |

Only arm B is run here -- 10 jobs, not 30.

## What each contrast isolates

- **B vs C** -- the hypothesis. If B is no better than C, the DT's own proposal
  carries no advantage over the teacher it imitates.
- **A vs B** -- what the pool + argmax machinery adds on top of the same DT.
  If A beats B, the machinery is doing the work.
- A vs C is already known (+0.0774 for MF-DRO, p = 0.2324, n.s.).

## Locked predictions

1. **PRIMARY (hypothesis holds)**: arm B's mean regret is **>= arm C's 0.4781**,
   i.e. the direct proposal does NOT beat the teacher.
2. **MACHINERY**: arm A (0.4007) beats arm B, paired across the shared seeds.
3. **FALSIFICATION**: if B beats C clearly, the DT's direct proposal adds
   something beyond imitation and the "it only copies its teacher" account is
   incomplete.

## Caveats fixed in advance

- n = 10 is underpowered (82 seeds were needed for the A-vs-C effect). Read
  directions and per-seed win counts, not p-values.
- The regression path was **unrunnable** until a `has_soft` guard was fixed
  earlier today (`mf_dro.py:2541`), so arm B has never been measured at these
  settings by anyone.
