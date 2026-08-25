# H32 — the distillation is FAITHFUL. My prediction failed.

| metric | value |
|---|---|
| Spearman(student score, teacher score) | **+0.7707 ± 0.0091** |
| argmax agreement | 5/12 = 41.7% |
| top-10 overlap (of 10) | 6.83 |
| **teacher's rank of the student's pick (of 200)** | **median 2**, range [1, 12] |

Per-pool ranks: `[1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 12]`.

**PRED 1 FAILS.** I predicted a median teacher-rank worse than 10, i.e. a lossy
distillation. The student picks the teacher's **top-ranked candidate five times
out of twelve**, its median pick is the teacher's **second best of 200**, and its
worst pick is 12th of 200.

## The metric I had been using was the wrong one

H24 reported 8/12 argmax agreement and I called the distillation "lossy" on that
basis. **Argmax agreement is a poor metric when the top candidates are near-ties.**
Disagreeing about which of two near-equal candidates is best is not a meaningful
error; the teacher-rank measure shows the student is essentially always inside
the teacher's top few. Fourth instance of the same lesson in this project: pick
the decisive metric, not a proxy that correlates with it.

## Revised expectation for H31, recorded before it lands

Since the student is a faithful approximation, **H31 should show near-parity**
rather than a clear teacher win. If the teacher-only arm instead comes in much
better than $0.4007$, something outside the per-decision approximation quality is
responsible --- most plausibly the fidelity choice, which the student makes by a
Bernoulli draw from a near-constant probability while the teacher chooses
`ell` deterministically as part of its argmax.

## What this settles about the method

MF-DRO is not a broken copy of MF-MES. It is a **faithful** copy that
additionally cannot condition (H23: fixed rule to within 0.13%) and costs orders
of magnitude more compute to produce. The negative result is therefore not "the
transformer approximates badly" but the sharper:

> The transformer reproduces its teacher closely, adds no conditioning, and
> inherits the teacher's exploitative selection behaviour. The apparatus is
> redundant rather than harmful.
