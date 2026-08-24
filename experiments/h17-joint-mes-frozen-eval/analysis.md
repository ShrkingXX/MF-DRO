# H17 — the joint-MES reward gives the best MF-DRO result on record, and still FAILS the frozen test

10/10 runs, **5876.6 s wall** (98 min), 10 workers x 1 thread.

## Per-seed final simple regret (required by PROTOCOL.md)

| seed | `improvement` | `mes_entropy` | diff |
|---|---|---|---|
| 42 | 0.3126 | 0.4491 | +0.1366 |
| 43 | 0.5218 | 0.4483 | −0.0735 |
| 44 | 0.6502 | 0.2507 | **−0.3994** |
| 45 | 0.4331 | 0.6018 | +0.1687 |
| 46 | 0.4412 | 0.5673 | +0.1261 |
| 47 | 0.7251 | 0.4307 | −0.2944 |
| 48 | 0.5429 | 0.2141 | −0.3288 |
| 49 | 0.5887 | 0.1996 | **−0.3891** |
| 50 | 0.4239 | 0.5459 | +0.1220 |
| 51 | 0.4073 | 0.2992 | −0.1081 |

## Summary at matched cost ~200

| method | final simple regret | sd |
|---|---|---|
| **MF-DRO / `mes_entropy`** | **0.4007 ± 0.0475** | 0.1501 |
| MF-DRO / `improvement` | 0.5047 ± 0.0395 | 0.1250 |
| MF-MI-Greedy | 0.5091 ± 0.1266 | 0.4004 |
| MF-GP-UCB | 1.7934 ± 0.1223 | 0.3868 |

## Locked predictions

| | result |
|---|---|
| **PRED 1** — lower regret than `improvement`, Wilcoxon p<0.05 | **FAIL** (−0.1040, 6/10, p = 0.375) |
| **PRED 2** — frozen success test, mean+SE < 0.3825 | **FAIL** (0.4481) |
| **PRED 3** — null composes into a coherent negative | fires |

## The honest reading

**The reward change produced the largest regret movement of the entire
investigation** — a 20.6% mean reduction, and the best MF-DRO number on record.
Against the baselines it is lower in mean than **both**, with 2.7× smaller sd
than MI-Greedy.

**And none of that is statistically significant at the pre-registered n=10.**

| paired comparison | diff | wins | Wilcoxon | t-test |
|---|---|---|---|---|
| vs `improvement` | −0.1040 | 6/10 | 0.375 | 0.193 |
| vs MF-MI-Greedy | −0.1085 | 6/10 | 0.432 | 0.352 |
| vs MF-GP-UCB | −1.3927 | 10/10 | **0.0020** | <0.0001 |

Per-seed differences are large in *both* directions (−0.399 to +0.169). Paired
sd is 0.2339, so detecting the observed effect at 80% power needs **~40 seeds**.

**I am not running 40 seeds.** `PROTOCOL.md` fixes the evaluation at 10 seeds;
extending it to chase significance would be exactly the optional-stopping the
frozen protocol exists to prevent. The pre-registered answer at n=10 is FAIL,
and that is the answer.

The only significant win is over MF-GP-UCB, which H3 showed is degenerate here —
it never queries HF at all (`mean_n_HF = 0.0` on all 10 seeds), so beating it is
not evidence of much.

## What this settles for the research question

`PROTOCOL.md` asks whether a within-DRO-frame fix beats MF-MI-Greedy and
MF-GP-UCB at matched real cost. **Answer: no.** Two rewards, both run under the
frozen evaluation, both FAIL the success test:

- `improvement`: 0.5442 ≥ 0.3825
- `mes_entropy`: 0.4481 ≥ 0.3825

The gap narrowed by roughly half. It did not close.

## Composition with the mechanism

H17 is the *only* intervention that moved regret at all — and it moved it by
changing the **training signal**, not the conditioning pathway. That is exactly
what the mechanism predicts: the state/RTG/BTG channel is attenuated ~10× and
leaves the argmax invariant (0/12, both within and across iterations), so the
only route to better behaviour is a better *target to re-fit toward*. MF-DRO
improves the way a re-fitted acquisition function improves, not the way a
conditioned policy improves.

## ETA note

Estimated 50–90 min; then revised to ~145 min; **actual 98 min**. Wrong in both
directions on the same run — the first estimate 8 min short of its upper bound,
the revision 47 min long. The revision over-corrected by extrapolating the
straggler's cost rate as if it were constant, when LF-heavy runs accelerate as
the GP sharpens. My estimates on this class are **noisy, not biased**.
