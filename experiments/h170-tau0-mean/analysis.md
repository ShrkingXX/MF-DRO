# h170 — the τ=0 mechanism survives its first NON-post-hoc test

CONFIRMATORY. 40 states per arm (8 cuts × 5 seeds), 120 reconstruction draws.

| arm | d(q, τ=0 mean) | d(q, centre) | d(q, random) | τ=0 mean → centre | est. SE |
|---|---|---|---|---|---|
| control (works) | **0.2428** | 0.7946 | 1.1200 | 0.6831 | 0.0487 |
| h155 UCB-LOC (works) | **0.2513** | 0.8135 | 1.0932 | 0.6955 | 0.0505 |
| RANDOM-POOL (fails) | 0.1535 | 0.1276 | 0.8143 | **0.0712** | 0.0731 |
| ORACLE (fails) | 0.2101 | 0.1928 | 0.8613 | **0.0686** | 0.0734 |

- **P1 HOLDS**, and it is the discriminating test: on the working arms — the only
  arms where "τ=0 teacher mean" and "box centre" are different targets — the DT's
  query is **3.3× and 3.2× closer** to the τ=0 mean.
- **P2 CALIBRATED**: the failing arms' reconstructed τ=0 mean lands 0.0712 /
  0.0686 from the box centre, matching the estimator's own floor at 120 draws.
- **P3 HOLDS on all four.**

## A gate I set below my own estimator's resolution

The first run used 120→**12** draws and P2 reported "NOT calibrated" at 0.2242 /
0.2380 against a 0.15 threshold. That was **my estimator, not the theory**: the
mean of 12 uniform draws in 8D sits an expected **0.2289** from the centre, so
the threshold was below the noise floor and could not have passed however right
the theory was.

Checked by direct simulation (4000 replicates): 12 draws → 0.2289, 50 → 0.1131,
200 → 0.0562. Re-run at 120 draws, P2 calibrates.

**Same error class as h156's "within 8%" over-read**: quoting an agreement
without first measuring what the instrument can resolve. Guard already adopted
after h156; not applied here. Recorded as a repeat.

## What is NOT claimed

The residual is real. d(q, τ=0 mean) = 0.2428 against an estimator SE of 0.0487
is **≈5 SE** — the DT's query is not *at* the τ=0 mean, it is much nearer to it
than to any alternative offered. So the τ=0 conditional mean explains the **bulk
of where the query lands, not all of it**.

Per the registered map this is **R2**: the mechanism survives its first
non-post-hoc test, is still one test, and is **not reported as established**.
Six accounts have been proposed on this front and five have fallen; this is the
first to make a quantitative prediction before being checked and have it hold.
