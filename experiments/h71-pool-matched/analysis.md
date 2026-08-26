# H71 result — all four bars MET, but one of them reads differently after h77

**CONFIRMATORY** against `protocol.md`; verdict script written at 1/6 before any
h71 regret was inspected.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | Borehole: >= 2.0 pts **and** >= 2/3 | **+6.05 pts, 3/3** | **MET** |
| SECONDARY | POOL1000 stays above 12% on Borehole | **17.66%** | **MET** |
| NULL | movement < 2.0 pts | — | did not fire |
| CONTROL | Hartmann not worse by > 2.0 pts | +4.37 | **MET** |

| benchmark | seed | BASE | POOL1000 | winner |
|---|---|---|---|---|
| Borehole | 44 | 24.43% | **17.65%** | POOL1000 |
| Borehole | 46 | 24.74% | **20.79%** | POOL1000 |
| Borehole | 48 | 21.95% | **14.55%** | POOL1000 |
| Borehole | mean | 23.71% | **17.66%** | **3/3** |
| Hartmann | 44 | 22.67% | **13.42%** | POOL1000 |
| Hartmann | 46 | **8.65%** | 10.22% | BASE |
| Hartmann | 48 | 12.73% | **7.30%** | POOL1000 |
| Hartmann | mean | 14.68% | 10.32% | 2/3 |

## The teacher pool IS load-bearing for MF-DRO — on Borehole

Widening the *rollout teacher's* candidate pool from 200 to 1000 — a change to
the training signal, not to inference, since MF-DRO's regression head does no
inference-time acquisition search — moves Borehole **23.71% -> 17.66%, 3/3**.
That is the largest improvement any intervention has produced on this benchmark.

**And it is not enough.** The SECONDARY was written to test exactly this:
POOL1000 stays at 17.66% against MI-Greedy's **9.29%** (n=10). For the *greedy
baselines* pool size was the whole story — h70 showed SF-EI at 1000 candidates
reproduces MI-Greedy exactly, seed for seed. For MF-DRO it closes about a third
of the gap and no more.

## The CONTROL passed against a reference that h77 later showed was wrong

h71's bars were locked against h57's **n=3** BASE. h77 has since measured that
BASE cell at n=10:

| benchmark | BASE n=3 (locked) | BASE n=10 | POOL1000 (n=3) | vs the n=10 BASE |
|---|---|---|---|---|
| Borehole | 23.71% | 22.89% | 17.66% | **+5.23 — better** |
| Hartmann | 14.68% | **8.91%** | 10.32% | **−1.41 — WORSE** |

**Borehole's reference was accurate to 0.82 points, so the PRIMARY reading
holds.** Hartmann's was off by **5.77**, and against the properly estimated BASE
POOL1000 is **worse by 1.41 points, not better by 4.37**. The CONTROL is
technically MET against what was locked, and substantively misleading.

**The paired comparison is unaffected.** Both arms ran the *same* seeds
44/46/48, so the win counts (3/3, 2/3) stand. What the unrepresentative draw
distorts is the **magnitude**, and only on Hartmann.

## What this cannot settle

**POOL1000 is n=3.** Lesson 26: three of four exploratory n=3 directions in this
project failed at n=10, one reversing sign. A 3/3 sweep with a +6.05-point margin
is the strongest such signal seen here, and it is still three seeds. **h78
pre-registers the n=10 replication on Borehole.**

It also does not touch the north star. Even taken at face value, 17.66% loses to
MI-Greedy's 9.29% — the SECONDARY says so explicitly, and that bar was met.
