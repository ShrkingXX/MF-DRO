# H39 — reverting to the regression head does NOT bring back the freeze

**Status: EXPLORATORY** (user question, no pre-registered protocol).
**n = 1 seed, Currin 2D, 14 iterations.** Not conclusive; needs replication.

| | REGRESSION head (`action_head`) | CANDIDATE scoring |
|---|---|---|
| proposed-x mean pairwise spread | 0.050072 | 0.047927 |
| per-coordinate sd | [0.0394, 0.0178] | [0.0346, 0.0138] |
| **distinct proposals** | **14/14** | **14/14** |
| **incumbent improvements** | **2 — moving** | **1 — moving** |
| final regret | 0.0073 | 0.0013 |

## The prediction, and its refutation

I predicted the regression head would re-freeze, reasoning: the state channel is
nearly dead (`w` moves 0.13%), so `x = action_head(h)` with a near-constant `h`
would give a near-constant `x`, and the incumbent would stop moving.

**Wrong.** The regression head produced **14/14 distinct proposals** and moved
the incumbent **twice** — slightly more than candidate scoring did (once).

## The mechanism I proposed is also refuted

I claimed candidate scoring *masks* the conditioning failure: even with a fixed
scoring rule, the argmax moves because the 200 candidates are redrawn each
iteration. That predicted the regression head, which has no such external
randomness, would expose the failure.

It did not. Both heads produce essentially the same query spread (0.050 vs
0.048). So the per-iteration variation does **not** come from candidate-pool
resampling. It comes from `h` genuinely changing as real data accumulates
(measured elsewhere: real-iteration states differ by mean pairwise L2 = 1.4968)
together with the DT being fine-tuned each iteration.

## What this establishes

**The frozen incumbent was caused by the target-leakage bug, not by the
regression head.** Post-fix, both heads move the incumbent. This retires the
worry that the candidate-scoring rewrite was load-bearing for the freeze fix.

It also removes a confound worth naming: `use_candidate_scoring=True` was the
default when the freeze was declared resolved (h1, 0/10 frozen), and candidate
scoring landed *before* the leak fix — so h1 could not separate the two. This
experiment separates them, in the direction that credits the leak fix.

## A real bug found on the way

`use_candidate_scoring=False` **crashed** before this run:
`has_soft` is `None` on the regression path (no candidate sets exist to carry
soft targets) but `_train_dt` guarded only on `self.use_soft_score_target`,
giving `TypeError: 'NoneType' object is not subscriptable` (`mf_dro.py:2541`).
The regression path had been unrunnable. Fixed by guarding on the tensor itself.
