# H58 — interim: prediction 1 (mechanical) confirmed; regret WITHHELD

Hartmann 6D arm: FLOOR 3/3 complete, FREE 2/3. Currin arm not started.
**No regret comparison is reported here** — FREE seed 46 is still running and
lesson 19 forbids reporting the subset when the rest is coming.

## Prediction 1 (MECHANICAL): does the inference-time floor bind?

FREE seed 46's counterpart is h57's live MF-DRO checkpoint at the same config;
seeds 44 and 48 use h58's own completed FREE runs.

| seed | FREE HF% | FREE nq | FLOOR HF% | FLOOR nq | bound? |
|---|---|---|---|---|---|
| 44 | 81% | 31 | 81% | 31 | no — FREE already >= 25% |
| **46** | **2%** | **169** | **26%** | **74** | **YES** |
| 48 | 28% | 67 | 28% | 67 | no — FREE already >= 25% |

Confirmed, and narrowly: the floor changes behaviour on **1 of 3 seeds**. On the
other two the fidelity head already exceeded 25% on its own, so FLOOR and FREE
are bit-identical there — an intended no-op, not a failure.

## The cost of binding, which is locked prediction 3's mechanism

Seed 46 went from **169 queries to 74** — the floor removed **57% of the run's
queries**. At c_H=8 versus c_L=1 on Hartmann, forcing a quarter of the budget to
HF is a large reallocation, not a free correction. Prediction 3 (FLOOR is
harmful) named exactly this and remains live.

So the two arms on seed 46 are not "same search, better fidelity mix". They are
**74 queries with 19 HF** against **169 queries with 3 HF** — different budgets
in every sense except total cost. Whichever wins, the reason will be this
trade, and the writeup must say so rather than attributing it to the floor
"fixing starvation".

## What is still open

- FREE seed 46 (running) closes the Hartmann arm.
- The Currin arm (c_H=3, so a milder trade) has not started.
- Prediction 2 (FLOOR lower regret on >= 4 of 6 cells) untested.
