# H42 — the regression head does NOT freeze on Hartmann 6D

Regression head (`use_candidate_scoring=False`), Hartmann 6D, 3 seeds,
50 iterations each, initial design `n_HF=6, n_LF=45`.

| seed | improvements | frozen | distinct proposals | final regret | n_HF | wall |
|---|---|---|---|---|---|---|
| 42 | **3** | no | 50/50 | 0.4221 | 8 | 1582 s |
| 43 | **6** | no | 50/50 | 0.1340 | 35 | 1575 s |
| 44 | **3** | no | 50/50 | 0.4384 | 47 | 1557 s |

**0/3 frozen.** Every run improved the incumbent (3, 6, 3 times) and every run
proposed **50 distinct points out of 50** — no repetition at all.

## What this establishes

H39 found no freeze with the regression head on Currin 2D at n=1. Currin is easy
(regret 0.005 by iteration 6), so it could not stress the pathology. **Hartmann
6D is the benchmark where the freeze was originally observed — 9/12 (75%) of runs
froze pre-leak-fix — and the regression head does not reproduce it here.**

Combined with H39, the conclusion is that **the incumbent freeze was caused by the
target-leakage bug, not by the regression head**, and the candidate-scoring
rewrite was not load-bearing for the fix.

## Why this matters beyond the freeze question

It removes a confound that h1 could not resolve. `use_candidate_scoring=True` was
the default when the freeze was declared resolved (h1: 0/10 frozen), and
candidate scoring landed **before** the leak fix — so h1 could not tell which of
the two changes fixed the freeze. H39 + H42 separate them, and the credit goes to
the leak fix.

## Scope

3 seeds, one benchmark, 50 iterations, iteration-capped (not cost-matched to any
other arm). This answers the **freeze** question only. It is **not** a regret
comparison between the two heads: the per-seed spread here (0.134 to 0.438) is
wide and n=3, and the fidelity mix varies enormously across seeds (n_HF 8, 35,
47), so total cost differs by roughly 3x between seed 42 and seed 44.

## Note on an incident during this experiment

H40, the predecessor run, was repeatedly "restarted" because a broken process
check (`pgrep -fc 'h40-regression'`) reported zero workers while five grids were
in fact running concurrently -- `ps` shows the resolved interpreter path, so the
pattern never matched. The stacked grids exhausted swap and were killed. The
detection was replaced with a `ps`-based check that matches the script name, and
H42 ran as a single clean grid.
