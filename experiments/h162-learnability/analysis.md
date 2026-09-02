# h162 — **R2 fires.** The DT's own queries collapse for the unlearnable teachers.

EXPLORATORY, post-hoc, existing data only. No new runs.

## M1 dispersion (mean pairwise distance, domain-normalised), three slicings

| arm | ALL queries | HF only | first 100 (n matched) |
|---|---|---|---|
| control MES (works) | 0.2766 | 0.2778 | 0.2819 |
| h155 UCB-LOC (works) | 0.2889 | 0.2998 | 0.2935 |
| h153 MES-FROZEN (works) | 0.2464 | 0.2565 | 0.2532 |
| ORACLE (fails) | 0.1891 | 0.2036 | 0.1997 |
| DIVERSE-GOOD (fails) | 0.1830 | 0.1988 | 0.1945 |
| RANDOM-POOL (fails) | 0.1115 | 0.1345 | 0.1244 |

**Every working arm is above every failing arm in all three slicings.** The
obvious confounds do not explain it: the failing arms have more queries (124–160
vs 107–109) and very different fidelity mixes, and restricting to HF only, or to
a matched first-100, leaves the split intact.

## Within-seed pairing — the honest statement

The seed effect is large (seed 43 is high for every arm: 0.40, 0.42, 0.34), so
the unpaired comparison understates the split. Paired within seed, asking whether
every working run beats every failing run:

```
seed 42  works [.254 .261 .233]  fails [.208 .203 .123]   separated
seed 43  works [.401 .417 .344]  fails [.205 .205 .127]   separated
seed 44  works [.266 .270 .244]  fails [.183 .168 .126]   separated
seed 45  works [.204 .237 .168]  fails [.184 .184 .124]   ONE CROSSING
seed 46  works [.285 .282 .278]  fails [.218 .213 .122]   separated
```
**4 of 5 seeds show complete separation.** Seed 45 crosses: h153 (0.168) falls
below ORACLE and DIVERSE-GOOD (0.184). Reported, not smoothed over. Pooled and
unpaired there is no complete separation (working min 0.168 < failing max 0.218).

## The L_loc puzzle is resolved, consistently

`L_loc` is LOWER for the forced teachers (0.018–0.022 vs 0.040) and this has sat
unexplained for many ticks. Low loss on **clustered** targets means the network
is predicting their mean, not learning a mapping. The dispersion split is what
that looks like from the outside, and M2 agrees: nearest-neighbour distance as a
fraction of dispersion is 0.44 for all three failing arms against 0.27–0.32 for
the working ones — points packed uniformly inside a small region.

The seed-to-seed **consistency** points the same way: the failing arms' M1 s.d.
is 0.003–0.015 against the working arms' 0.066–0.074. They do nearly the same
thing regardless of seed, which is what a collapsed predictor does.

## What is NOT claimed

R2 was pre-named as "the framing survives one test", not as establishing a
mechanism, and that stands. This is query-level statistics at n=5 — the same
evidence class that produced h150 (a published finding, retracted) and h154's M2
(registered direction, refuted). One consistent correlate is not a cause.

What it does do: it removes the standing contradiction. The learnability framing
predicted a specific, falsifiable pattern in data already on disk, the pattern is
there, and it survives three robustness slicings. h161 remains the arm that can
actually test the framing's causal claim.
