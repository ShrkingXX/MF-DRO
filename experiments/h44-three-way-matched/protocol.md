# H44 — does the DT alone (no pool, no argmax) match its teacher?

## The question

MF-DRO edges its teacher (0.4007 vs 0.4781, n.s.). Hypothesis under test: that
edge comes from the **candidate pool + argmax machinery**, not from the DT. If
so, a DT that proposes DIRECTLY -- regression head, no pool, no argmax -- should
fall back to roughly teacher-level performance, because all it can contribute is
its imitation of the teacher's choices.

## The three arms, all at IDENTICAL settings

| arm | who picks the point | pool + argmax? |
|---|---|---|
| **A** MF-DRO, candidate scoring | `argmax_k <w(h), cf_k>` | yes |
| **B** MF-DRO, regression head | `x = action_head(h)` | **no** |
| **C** MF-MES teacher, no DT | `argmax_k MES(cf_k)/c` | yes |

Hartmann 6D, seeds 42-51, cost budget 200, initial design `n_HF=6, n_LF=45`
(the literature-standard 10% sizing). Arms A and B are already running as H40;
**this experiment adds arm C at the same settings**, which is the piece that
makes the three-way comparison valid. H31's teacher numbers used a different
initial design (36/60) and are NOT comparable to H40's.

## What each contrast isolates

- **B vs C**: does the DT's own proposal match the teacher? (the user's question)
- **A vs B**: what does the pool + argmax machinery add on top of the DT?
- **A vs C**: the headline MF-DRO-vs-teacher comparison, at a sane init

## Locked predictions

1. **PRIMARY**: arm B (regression, no pool) is **no better than** arm C
   (teacher), paired across seeds. If B >= C in regret, the DT's own proposal
   carries no advantage over the teacher it imitates.
2. **MACHINERY**: arm A beats arm B. If so, the pool + argmax contributes the
   edge, not the DT -- confirming the hypothesis.
3. **FALSIFICATION**: if arm B beats arm C significantly, the DT's direct
   proposal *does* add something beyond imitation, and the "it only copies its
   teacher" account is incomplete.

## Not claimed

n = 10 is underpowered for these effect sizes (82 seeds needed for the A-vs-C
effect measured earlier). Directions and per-seed win counts are the readable
signal; p-values will mostly not clear 0.05.
