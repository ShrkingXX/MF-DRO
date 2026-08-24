# H29 — the teacher's exploitativeness is INTRINSIC, not a tuning artefact

Chosen candidate's `sigma_H` percentile within its own pool (control: `mu_H`
percentile, which must exceed 50%).

| `c_H/c_L` | K=5 | K=10 | K=50 | `mu_H` control |
|---|---|---|---|---|
| 2 | 2.3% | 2.7% | 1.9% | 94.7% |
| 4 | 1.3% | 1.3% | 1.0% | 94.5% |
| **8** (frozen protocol) | 3.4% | **2.9%** | 3.3% | 94.2% |
| 16 | 5.2% | 5.5% | 5.3% | 93.6% |

**PRED 1 fires: intrinsic.** The chosen-`sigma_H` percentile never exceeds
**5.5%** in any of the twelve cells. The control holds everywhere (93.6–94.7%).

## Two things the sweep settles

**The `y*` sample count is irrelevant.** Within every cost ratio, K=5, 10 and 50
give effectively identical percentiles (e.g. 3.4 / 2.9 / 3.3 at ratio 8). Ten
times more Monte Carlo does **not** make MES's realised choices more
exploratory — the behaviour is not an estimation artefact.

**The cost ratio barely matters.** From 2 to 16 — an 8× swing — the percentile
moves only from ~1% to ~5%. It rises weakly with the ratio (expensive HF pushes
selection slightly toward higher-variance points), but never approaches
indifference, let alone exploration.

## What this means

Cost-normalised MF-MES is **exploitative as a demonstrator across the entire
swept regime**. Combined with H28, the conclusion is structural rather than
incidental:

> A student trained to imitate the *choices* of an argmax-of-MES teacher cannot
> learn to explore, at any operating point tested. The acquisition's score
> rewards uncertainty, but its argmax is dominated by the posterior mean, and in
> a GP the high-mean region is where the data already is.

This is a limitation of the **approach** — distilling a teacher's selections —
not of this implementation, and it is not fixable by tuning the teacher's cost
ratio or Monte Carlo budget. Fixing it requires the student to learn from the
teacher's *scores* over the whole candidate set rather than from its argmax, or
an explicitly exploration-augmented demonstrator.

## Scope

`PROTOCOL.md` untouched; no regret was measured and nothing here alters the
frozen evaluation, whose cell (`c_H/c_L = 8`, `K = 10`) is the one actually run.
Hartmann 6D only.
