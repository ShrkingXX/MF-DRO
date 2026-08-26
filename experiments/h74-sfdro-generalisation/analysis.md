# H74 result — NULL fired. h73 is Hartmann-specific.

**CONFIRMATORY** against `protocol.md`. Verdict script authored from the protocol
alone, without inspecting any Borehole value, with both gates enforced in code.

**Reproduction control PASSED:** h74's worker on Currin seed 44 gives
0.0012218365, identical to h59's published value, **diff 0.000e+00**. The 7 new
seeds per benchmark are produced by the same code path as h59's 3, so the 10-seed
means do not mix code paths. (This is the check h73's "built-in control" only
appeared to perform.)

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | Borehole: >= 2.0 pts **and** >= 7/10 wins | **−1.84 pts, 2/10** | **NOT MET** |
| SECONDARY | Currin: within 1.0 pt | −0.22 pts | **MET** |
| NULL | Borehole < 2.0 pts or <= 6/10 | fired | **h73 is Hartmann-specific** |
| REVERSED | SF-MES ahead by >= 2.0 pts | −1.84, just short | did not fire |

## SF-DRO vs SF-MES across all three benchmarks, n=10

| benchmark | SF-DRO | SF-MES | gap | SF-DRO wins | Wilcoxon p |
|---|---|---|---|---|---|
| Hartmann 6D | **8.46%** | 21.17% | **+12.71** | **10/10** | 0.0020 |
| Borehole 8D | 14.60% | **12.76%** | −1.84 | **2/10** | 0.0840 |
| Currin 2D | 0.22% | **0.00%** | −0.22 | **2/10** | 0.0137 |

**One win, two losses.** SF-DRO is not generally better than its own MES
counterpart. The n=3 losses on Borehole and Currin were **not** noise — they
replicated at n=10 with the same sign and, on Currin, a small p-value.

## The SECONDARY "MET" overstates Currin

The Currin bar was "within 1.0 point", written to catch SF-DRO being *materially*
worse on a benchmark where every non-degenerate method finishes inside 0.6%. By
that magnitude test it passes at −0.22 points. But SF-DRO wins only **2/10** with
**p = 0.0137**: it is *reliably* slightly worse, and the magnitude bar cannot see
that. This is the third protocol in this session whose bar passed while the
underlying comparison went the other way (after h68's PRIMARY and h65's spread
test). **A magnitude bar and a win-count bar catch different failures, and a
prediction needs both** — which is exactly why the Borehole PRIMARY required
both, and why it correctly returned NOT MET.

## What this settles

h73 established SF-DRO beating greedy single-fidelity BO on Hartmann by a large,
replicated, verified margin — and a post hoc check showed it also beats SF-EI
there (9/10, p=0.0059), so the Hartmann advantage is not specific to MES.

h74 settles the scope of that: **it does not travel.** On the other two
benchmarks SF-DRO is behind. The honest one-line summary is that SF-DRO beats its
MES counterpart on **one of three** benchmarks, and Hartmann is also the
benchmark whose standings h72 showed to be least resolved at small n.

## The north star

Not met, and now not met even against SF-DRO's own single-fidelity counterpart on
2 of 3 benchmarks. "At least as good as the baselines" is a per-benchmark bar, and
SF-DRO clears it on Hartmann only.

## What this cannot settle

n=10 per benchmark. It compares SF-DRO to SF-MES, not to the multi-fidelity
baselines. And it cannot say *why* Hartmann differs — the mechanism behind the
one win remains unexplained, and this project's record on proposing mechanisms
without measuring them is poor.
