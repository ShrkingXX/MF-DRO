# h161 STALE-PATH — COMPLETE, n=5. **It works, and staleness is nearly free.**

CONFIRMATORY. No harness forecast was offered — h161 is a frozen condition, and
the harness's one validation failure (C2, off by 2.7×) is exactly the frozen
case. That restriction was recorded before the run.

| | result (n=5) |
|---|---|
| rel% | **19.53** (h153 fresh-path: 19.36) |
| improves | **5/5** |
| rtg_target | **0.3330** |
| HF fraction | 0.92 |

Sanity checks, read first and all exact: **SC1** stale fraction **0.903**
(registered target >0.9 after warmup — hit), **SC2** mean lag **exactly 600**,
**SC4** pass-2 path replay error **0.0e+00** on both seeds.

## Verdict: R2

The protocol's R2 — "model-selected-at-all suffices, and staleness is free" —
is what happened. A path chosen by a model that has since seen ~10 iterations of
new data still produces a working arm: 21.17 against h153's 19.36 and the failing
arms' 43.94, improving on every seed so far.

**Model-selected FOR THE CURRENT STATE is not required.** Staleness costs about
1.8 rel% and no improvements.

## Where this lands

R2 was written when the learnability framing was the live account. **h167
retracted that framing last tick**, before h161 reported, so R2 lands on a
position already withdrawn. What h161 adds is not a refutation but a boundary:
whatever distinguishes the working from the failing frozen arms, it tolerates a
ten-iteration-stale model.

Third arm now with a **collapsed conditioning target that works** (h153 0.3230,
h159 0.9411 — no, h159 is closed-loop; h161 0.3388). Two frozen arms, both with
targets in the failing band, both performing near the control. The target
continues not to predict performance.

The surviving distinction between h161 (works) and RANDOM-POOL (fails) is not
learnability (h167: the DT fits both well) and not adaptivity (both are frozen or
state-blind). It is that h161's locations sit in high-acquisition **regions**,
stale or not, and RANDOM-POOL's sit anywhere. That is a claim about **where the
training queries are**, and h168 is the registered test of whether the inference
conditioning is what converts that into failure.

## Status: COMPLETE, n=5

Sanity checks exact on every seed: stale fraction 0.902-0.907, mean lag exactly
600, replay error 0.0e+00.

At full n the gap to h153's fresh model-selected path is **0.17 rel%**
(19.53 vs 19.36) — a ten-iteration-stale model costs essentially nothing. The
n=2 read of 21.17 overstated it; that is why n=2 carried no verdict beyond R2.
