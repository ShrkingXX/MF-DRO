# h164 — **R2 fires.** The dispersion collapse replicates on Hartmann.

CONFIRMATORY, zero compute, existing serialised data.

| arm | ALL | first 70 (n matched) | Borehole reference |
|---|---|---|---|
| control MES (works) | **0.2144** | **0.2166** | 0.2766 |
| ORACLE (fails, **confounded**) | 0.1563 | 0.1680 | 0.1891 |
| RANDOM-POOL (fails) | **0.1153** | **0.1171** | 0.1115 |

M2 (nearest-neighbour / dispersion): control 0.282, ORACLE 0.409, RANDOM 0.443
— the same signature as Borehole (working 0.27–0.32, failing ~0.44).

## The load-bearing contrast: 5/5 paired

Hartmann's ORACLE arm is flagged confounded in findings.md, so control vs
RANDOM-POOL carries the result. Paired within seed on the matched-n slice:

```
seed42  control 0.159  RANDOM 0.097   ok
seed43  control 0.260  RANDOM 0.115   ok
seed44  control 0.237  RANDOM 0.113   ok
seed45  control 0.160  RANDOM 0.122   ok
seed46  control 0.266  RANDOM 0.139   ok
```
**5 of 5.** Against ORACLE it is 4 of 5 — seed 45 crosses (0.160 vs 0.177) — and
that arm is confounded anyway, so it is reported without being counted.

## A slicing that does NOT support the claim, reported

The HF-only slicing is **unusable on Hartmann**: the control has fewer than 12
post-init HF queries on three of five seeds, so it averages over 2 seeds
(0.3833) against RANDOM's 5 (0.1311). The direction agrees but the comparison
is not like-for-like and is not counted. On Borehole all three slicings were
usable; here only two are.

## Scope

The framing's observable now replicates on a second benchmark and a second
failing teacher. It remains a **correlate**: h161 is the causal test. And the
evidence class is unchanged — query-level statistics at n=5, which produced h150
(retracted) and h154's refuted M2 direction. Replication strengthens it within
that class, not beyond it.
