# H35 — fantasy conditioning confirmed as the driver; my mechanism was wrong

| tau | HF/c_H | LF/c_L | ratio | sigma_H | sigma_L | sH/sL | HF rate |
|---|---|---|---|---|---|---|---|
| 0 | 0.0559 | 0.0775 | 0.720 | 0.2403 | 0.1826 | 1.316 | 33.0% |
| 3 | 0.0552 | 0.0582 | 0.948 | 0.2500 | 0.1951 | 1.282 | 47.5% |
| 4 | 0.0555 | 0.0518 | **1.072** | 0.2519 | 0.1972 | 1.277 | **55.5%** |
| 7 | 0.0566 | 0.0488 | 1.160 | 0.2597 | 0.2028 | 1.281 | 62.0% |
| **REAL** | 0.0633 | **0.1287** | **0.491** | 0.2365 | 0.1814 | 1.304 | `ell=0` |

**PRED 1 PASSES on both clauses.** The cost-normalised HF:LF ratio climbs
`0.720 → 1.160` across the rollout, crossing 1.0 at `tau=4` — exactly where the
HF choice rate crosses 50%. And `tau=7`'s ratio of 1.160 is more than double the
real-inference value of 0.491.

So fantasy accumulation *is* the driver of the regime gap H34 found.

## PRED 2 FAILS — my proposed mechanism is refuted

I predicted `sigma_L` would fall faster than `sigma_H`, tipping the balance.
Measured, neither falls at all: both **rise** slightly across the rollout
(`sigma_H` 0.2403→0.2597, `sigma_L` 0.1826→0.2028) and their ratio is essentially
**constant** (1.316→1.281). The effect is not a variance-reallocation.

## What actually drives it

**The LF branch collapses; the HF branch does not move.**

- `LF/c_L`: `0.0775 → 0.0488`, a **−37%** decline across the rollout
- `HF/c_H`: `0.0559 → 0.0566`, **+1%** — flat

The teacher does not become more attracted to high fidelity. Low fidelity simply
stops being worth its (small) cost. And at real inference `LF/c_L = 0.1287` —
**2.6x** the `tau=7` value and 1.7x even the `tau=0` value — so the real regime
is one where LF queries are far more valuable than anywhere inside a rollout.

Since `sigma_L` is flat while LF's MES collapses, the decline is **not** driven by
LF uncertainty being resolved. It must come from the LF branch's information
about `y*_H` degrading — the quantity Takeno's Lemma 3.1 quadrature computes
through the cross-fidelity relationship. Diagnosing that is a further step and is
**not** claimed here.

## The chain, as far as it is now measured

1. Fantasy conditioning makes LF queries progressively worthless *within a
   rollout* (−37% in cost-normalised MES), while HF holds flat.
2. The teacher therefore selects HF 50.6% of the time in rollouts versus
   4.2–11.9% at real inference (H34).
3. The student's fidelity head learns the rollout rate to within 0.7 pp
   (`p=0.557` vs labels `57.7%`) and applies it to the real regime (H34).
4. Its fidelity policy is consequently both miscalibrated *and* — separately, and
   still unexplained — uninformative (`sd 2.4e-4`, `corr 0.155`, H33).

## Method note

Fifth time in this project that checking a mechanism has corrected the
explanation I had already committed to. The phenomenon (PRED 1) held; my account
of it (PRED 2) did not.
