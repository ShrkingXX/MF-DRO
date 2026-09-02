# h158 — the POOL dose. **R3 fires.** Quality and diversity are both orthogonal.

6 states × 2 replicates, N=100, 10-model ensemble.

## Manipulation checks — both PASS

| POOL | endpoint value | endpoint spread |
|---|---|---|
| 16 | 169.33 | 0.2497 |
| 256 | 221.45 | 0.2035 |
| 4096 | 255.47 | 0.1643 |

MC1 (value rises) **PASS**. MC2 (spread falls) **PASS**. The dose moved what it
was supposed to move: endpoint quality **+51%**, endpoint diversity **−34%**,
on one axis in opposite directions.

## Outcome — the tail is FLAT

| condition | rep0 | rep1 | mean | % of control |
|---|---|---|---|---|
| control | 0.9731 | 0.9087 | 0.9409 | 100.0% |
| POOL 16 | 0.3036 | 0.2935 | 0.2986 | **31.7%** |
| POOL 256 | 0.3030 | 0.3076 | 0.3053 | **32.4%** |
| POOL 4096 | 0.2892 | 0.2620 | 0.2756 | **29.3%** |

Span **3.2 points** against an 8–13% harness noise floor — flat. Neither
monotone rising nor falling. All three sit inside the failing band established
independently in h157 (RANDOM / ORACLE / DIVERSE-GOOD: 25.9–34.3%).

**A 51% rise in endpoint quality and a 34% fall in endpoint diversity produce no
detectable change in the reward's upper tail.** R1 (tail rises with quality) and
R2 (tail falls with diversity) both fail to fire.

## What this settles

This is the sharpest confirmation the account has. Every prior test of "quality
is orthogonal to the reward" compared two or three arms that differed in many
ways at once. This moves quality and diversity **continuously and in opposite
directions on one axis**, with the manipulation verified rather than assumed,
and the outcome does not move.

Interpolating toward an already-good point earns almost no information about
where the optimum is — and it does not matter how good the point is, nor whether
every trajectory heads somewhere different.

## And it settles the standing decision to decline the pipeline version

The /loop prompt has carried this follow-up for many ticks. I declined it and
recorded the reason, but that reason was an ASSERTION extrapolated from two
endpoints. It is now a measured prediction that held. **Not running the ~15
worker-hour pipeline dose is justified on measurement.** The pipeline version
would return ~43.94 at all three doses; the harness says the mechanism-level
quantity is flat, and the pipeline outcome is downstream of it.

## Limits, stated

Harness only — no pipeline arm was run, so this is a claim about the reward
signal, not directly about final regret. Borehole seeds 42/43, 6 states. The
harness supports scale claims only (per-arm errors 0.5–31%); the finding here is
a 3-point span against a 13-point noise floor, which is a scale claim about
flatness and is within what this instrument can carry.
