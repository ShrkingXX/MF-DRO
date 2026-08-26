# H73 result — MET. The first claim in this project to survive n=10.

**CONFIRMATORY** against `protocol.md`; verdict script committed at 0/7 before any
number existed.

**CORRECTION — the "built-in reproduction control" was VACUOUS.** It reported
seeds 44/46/48 as reproducing h59's 11.49% / 21.39% exactly. Of course it did:
h73 ran only seeds 42/43/45/47/49/50/51, and `analyse.py` falls back to the h59
directory for the other three. **It compared h59's files to themselves.** The
claim "reproduction control passed exactly" was stated in the original version of
this file and in the commit message; it verified nothing.

What actually needed checking: h73's worker is a copy of h59's whose only
difference is the experiment name (`h59_{bench}_{seed}` -> `h73_{bench}_{seed}`),
which feeds `setup_dirs()` and `_build_dro_config()`. If that name reaches any
seeding path, the 7 new seeds would not be comparable to h59's 3. A **real**
control — h73's worker run on seed 44, compared against h59's 11.48% — is
recorded below.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | >= 5.0 pts **and** >= 7/10 wins | **+12.71 pts, 10/10** | **MET** |
| SECONDARY | SF-DRO sd below SF-MES | 4.08 vs 5.11 | **MET** |
| NULL | < 5.0 pts or <= 6/10 | — | did not fire |
| REVERSED | SF-MES ahead by >= 5.0 | — | did not fire |

| seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| SF-DRO | 13.14 | 10.45 | 11.48 | 3.30 | 13.26 | 3.43 | 9.73 | 3.01 | 6.39 | 10.42 |
| SF-MES | 21.81 | 23.47 | 22.67 | 19.25 | 30.74 | 24.70 | 10.76 | 18.52 | 19.28 | 20.48 |

SF-DRO **8.46%** (sd 4.08, worst 13.26%) vs SF-MES **21.17%** (sd 5.11, worst
30.74%). **10/10 paired wins.** Wilcoxon signed-rank **p = 0.0020**, reported
unconditionally.

## Why this one matters

Lesson 26 stood at three for three: every exploratory n=3 direction this project
took to n=10 had failed, and one reversed sign. h72 then showed Hartmann is
specifically where n=3 misleads most — published baseline entries there moved by
**+12.7** and **+21.5** points at n=10.

h73 is the first to survive, and it did not merely survive: the n=3 gap was 9.9
points at 3/3, and the n=10 gap is **larger** at 12.71 with a clean sweep. The
effect was real and n=3 had **understated** it.

**Lesson 26 is not weakened — it is confirmed in the other direction.** n=3 does
not estimate a direction reliably; here it happened to point the right way while
underestimating the size. That is the same instability, not an exception to it.

## Query-matched — verified, not assumed

Both arms run **exactly 25 optimization iterations on all 10 seeds** (Hartmann
c_H=8, budget 200 -> 200//8 = 25 for a single-fidelity method), plus the same
6-point HF initial design. The comparison is not confounded by query count or
budget accounting.

(SF-DRO's `n_queries` field reads 0 for seeds 44/46/48 — h59's known empty-trace
bug, where the worker filtered `if "x" in d` over `iteration_log_history`, which
carries no coordinates. It does not touch `final_regret`, which is read from the
regret curve.)

## Scope — what this does NOT establish

This is **Hartmann only**. At n=3 SF-DRO *loses* to SF-MES on Currin (0.4% vs
0.0%) and Borehole (15.1% vs 13.3%), and neither was re-run here (43.8 and
67.2 min per run). So the claim is: **SF-DRO beats its own MES counterpart on one
of three benchmarks under replication** — not that it is the better method.

It also says nothing about the multi-fidelity baselines, which are a separate
comparison and are not pre-registered here.

---

## POST HOC addendum — SF-DRO vs MF-MES. **Not confirmatory.**

The data already existed and both means had been seen before this comparison was
posed. It is reported in full because it bears directly on the north star, but it
**cannot be a claim** and no pre-registration is retro-fitted to it.

| | mean | sd | worst |
|---|---|---|---|
| SF-DRO | 8.46% | **4.08** | **13.26%** |
| MF-MES | **8.24%** | 6.85 | 25.25% |

**SF-DRO wins 4/10. Wilcoxon p = 0.4316.** Indistinguishable, with MF-MES
marginally ahead on the mean and SF-DRO ahead on variance and worst case.

**Full Hartmann standings at n=10** (every method with 10 seeds):

| method | n=10 | |
|---|---|---|
| MF-DRO + POOL600 | 6.64% | mean advantage withdrawn by h66 — 5/10 wins, one-seed artifact |
| MF-MES | 8.24% | |
| **SF-DRO** | **8.46%** | single fidelity, **no free LF initial design** |
| SF-EI | 18.61% | single fidelity |
| SF-MES | 21.17% | single fidelity |
| MF-MI-Greedy | 36.61% | |
| MF-GP-UCB | 66.81% | |

Three methods cluster at 6.6-8.5% and are not separable at n=10; then a 10-point
gap to everything else. SF-DRO reaches that cluster **without any low-fidelity
information at all**, while MF-MES receives a free LF initial design worth +45
cost units on Hartmann (22.5% of the optimisation budget).

**Does this meet the north star?** Not as stated. "At least as good as the
baselines" is a per-benchmark bar, and SF-DRO *loses* to SF-MES on Currin (0.4%
vs 0.0%) and Borehole (15.1% vs 13.3%) at n=3. Hartmann alone is not the north
star, and a post hoc tie is not a result. h74 pre-registers the generalisation
test.
