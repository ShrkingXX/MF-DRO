# H36 — PRED 1 FAILS. Stopping, as the protocol committed.

| tau | mean y* | **sd y*** | max mu_H | gap | LF/c_L |
|---|---|---|---|---|---|
| 0 | 1.4001 | 0.1448 | 1.2505 | 0.1496 | 0.0779 |
| 3 | 1.4156 | 0.1239 | 1.2966 | 0.1190 | 0.0546 |
| 7 | 1.4284 | 0.1110 | 1.3278 | 0.1006 | 0.0476 |
| **REAL** | 1.5158 | **0.1684** | 1.4644 | **0.0513** | **0.1351** |

**PRED 1 FAILS.** I predicted the gap `mean(y*) − max(mu_H)` would *grow* with
`tau`. It **shrinks**, `0.1496 → 0.1006`. And the real-inference state — where
LF is most valuable — has the *smallest* gap of all (0.0513), the opposite of
what the hypothesis requires.

So `y*` drift, in the sense I proposed, does not explain the LF collapse.

## What I am NOT going to claim

The table does contain a quantity that tracks `LF/c_L` closely: the **spread** of
the `y*` samples. `sd(y*)` falls monotonically `0.1448 → 0.1110` across the
rollout, and the real state — with by far the highest LF value (0.1351) — also
has by far the widest `y*` distribution (0.1684).

That is a real pattern. **It is also exactly the kind of post-hoc substitution
this experiment's protocol pre-committed against**, written before the run:

> "My mechanism guess has now been wrong twice in a row on this sub-question
> (H34's floor hypothesis, H35's `sigma_L` hypothesis). If this prediction also
> fails, the honest move is to stop proposing mechanisms and report the collapse
> as measured but unexplained."

Three failed mechanism guesses in a row is strong evidence that my intuitions
about this quadrature are unreliable. Swapping in the variable that happens to
correlate, *after* seeing the table, would be the fourth guess dressed as a
finding.

**Recorded status: the LF collapse is measured and unexplained.** `sd(y*)` is
logged as a candidate for a *future pre-registered* test, not as an answer.

## What is established without it

The causal chain from H34/H35 does not depend on this sub-mechanism:

1. Inside rollouts, `LF/c_L` collapses −37% while `HF/c_H` stays flat (H35).
2. The teacher therefore picks HF 50.6% in rollouts vs 4.2–11.9% at real
   inference (H34, H35).
3. The fidelity head learns the rollout rate to within 0.7 pp and applies it to
   the real regime (H34).

*Why* the LF branch loses value is one level below any of these claims. All three
stand on direct measurement.
