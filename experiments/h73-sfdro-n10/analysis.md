# H73 result — MET. The first claim in this project to survive n=10.

**CONFIRMATORY** against `protocol.md`; verdict script committed at 0/7 before any
number existed. **Built-in reproduction control PASSED**: seeds 44/46/48 give
11.49% / 21.39%, exactly h59's published values.

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

## Scope — what this does NOT establish

This is **Hartmann only**. At n=3 SF-DRO *loses* to SF-MES on Currin (0.4% vs
0.0%) and Borehole (15.1% vs 13.3%), and neither was re-run here (43.8 and
67.2 min per run). So the claim is: **SF-DRO beats its own MES counterpart on one
of three benchmarks under replication** — not that it is the better method.

It also says nothing about the multi-fidelity baselines, which are a separate
comparison and are not pre-registered here.
