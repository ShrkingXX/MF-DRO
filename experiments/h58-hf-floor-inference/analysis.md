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

---

# H58 FINAL — 12/12. Prediction 2 not met; prediction 3 (my expectation) REFUTED.

| benchmark | arm | s44 | s46 | s48 | mean | nq (HF%) |
|---|---|---|---|---|---|---|
| Hartmann | FREE | 0.7531 | 0.2875 | 0.4228 | 0.4878 | 31(81%) **179(2%)** 67(28%) |
| Hartmann | **FLOOR** | 0.7531 | **0.2230** | 0.4228 | **0.4663** | 31(81%) **74(26%)** 67(28%) |
| Currin | FREE | **0.0001** | 0.0031 | 0.0011 | **0.0014** | 134(25%) 128(28%) 108(43%) |
| Currin | FLOOR | 0.0002 | 0.0031 | 0.0011 | 0.0015 | 127(29%) 128(28%) 108(43%) |

## Verdicts against the locked predictions

**Prediction 1 (mechanical, floor binds where HF% < 25)** — CONFIRMED, and
narrowly. It changed behaviour on **2 of 6 cells**: Hartmann s46 (2% -> 26%) and
Currin s44 (25% -> 29%, a marginal touch). The other four were already above the
floor, so the arms are bit-identical there.

**Prediction 2 (FLOOR lower regret on >= 4 of 6)** — NOT MET. FLOOR is better on
1, identical on 4, worse on 1. With only 2 binding cells, 4-of-6 was
unreachable by construction — a flaw in how I wrote the prediction, not a
property of the floor.

**Prediction 3 (FLOOR harmful, my stated expectation)** — **REFUTED on the one
cell that tests it.** Hartmann s46 went from 179 queries at 2% HF to 74 queries
at 26% HF and regret *improved* 22% (0.2875 -> 0.2230). I predicted the
reallocation would cost more in queries than it bought, because HF is 8x LF on
Hartmann. It did not.

## What the one informative cell actually says

**176 LF queries were worth less than 19 HF queries.** The free policy spent 179
queries and finished worse than a run given 74. That is a strong statement about
the fidelity head: left unconstrained it bought a large volume of near-worthless
LF evaluations, and a crude 25% floor — copied verbatim from the rollout
simulator, no tuning — recovered 22% of the regret.

Currin's binding cell (s44, 25% -> 29%) barely moved the mix and the regret
moved 0.0001 in FREE's favour. Uninformative, as expected at c_H=3 with the
policy already at the floor.

## Honest limits

- **n = 1 informative cell.** One seed, one benchmark. Everything above rests on
  Hartmann s46. This sizes an effect; it establishes nothing.
- The right follow-up is a floor sweep (0%, 25%, 50%) on seeds whose free HF
  fraction is low, not more seeds at 25%.
- The design flaw is worth recording: a 6-cell experiment where the treatment
  can only act on cells meeting a data-dependent condition cannot support an
  "N of 6" prediction. I should have predicted over *binding* cells.
