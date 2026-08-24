# H30 — `w[sigma_H] < 0` is a PARTIAL coefficient, not an aversion

**Status: EXPLORATORY** (unplanned diagnostic, run after H29 exposed a hole in
its own framing). Labelled as such.

## The hole H29 left

H29 concluded "a student imitating the *choices* of an argmax-of-MES teacher
cannot learn to explore". But MF-DRO does **not** train on the argmax — the loss
is a soft-KL to the teacher's *scores* over the whole K=20 candidate set. So the
student is already learning from scores, and the H29 framing was wrong about the
mechanism even if right about the outcome.

That raises a sharper question: if the teacher's score correlates **positively**
with `sigma_H` (Spearman +0.1585, H28), why does a student fitting those scores
learn a **negative** `sigma_H` weight?

## Answer: collinearity

1600 candidate sets:

| quantity | value |
|---|---|
| `corr(mu_H, sigma_H)` across candidates | **−0.4696 ± 0.0046** |
| **marginal** Spearman(teacher score, `sigma_H`) | **+0.1585 ± 0.0054** |
| **partial** OLS coefficient on `sigma_H`, `mu_H` held | **−0.0419 ± 0.0009** |
| partial coefficient negative on | **91.4%** of candidate sets |

In a GP the high-`mu` region is where the data sits, so `mu` and `sigma` are
strongly negatively correlated (−0.47). Under that collinearity the teacher's own
score function has a **negative partial coefficient on `sigma_H`** even though its
marginal correlation with `sigma_H` is positive — textbook suppression.

**The student's `w[sigma_H] < 0` reproduces the teacher's own partial
coefficient.** It is faithful linear approximation of MES in a collinear basis,
not an independently acquired aversion to uncertainty.

## What survives and what is corrected

**Survives — the behavioural claim.** Chosen candidates sit at the 2.9th
percentile of `sigma_H` (H28), in every one of twelve regimes (H29). That is a
statement about *selections* and does not depend on any coefficient
interpretation.

**Corrected — the coefficient claim.** The paper said the learned rule
"penalises exactly the quantity that UCB, EI and MES reward". That is **wrong**:
MES's own partial coefficient on `sigma_H` is negative here too. The student is
not doing anything its teacher does not.

The honest formulation:

> The policy reproduces its teacher's score function, including a negative
> partial coefficient on `sigma_H` induced by `mu`--`sigma` collinearity. The
> resulting *behaviour* is exploitative — selections land in the bottom 3% of
> posterior uncertainty — but this follows from `mu`-domination plus
> collinearity, not from a term that seeks out low-uncertainty points.

## Method note

This is the third correction in this arc: H28 retracted "the student inverted its
teacher", H29 relocated the fault to the teacher, and H30 now corrects what the
negative coefficient *means*. Each came from checking a mechanism rather than
accepting a plausible reading of a number.
