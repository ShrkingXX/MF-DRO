# h154 -- EXPLORATORY. Gate: **MIXED**, exactly as pre-named.

Borehole seeds 42-46, post-init real queries only, all arms already complete.

| arm | M1 response-to-outcome | M2 lag-1 autocorr | mean step |
|---|---|---|---|
| control MES (closed-loop) | **-0.5027 ± 0.1013** | **+0.7026 ± 0.1001** | 0.1267 |
| ORACLE (frozen) | +0.1446 ± 0.2630 | +0.3903 ± 0.0752 | 0.1455 |
| DIVERSE-GOOD (frozen) | +0.1190 ± 0.1935 | +0.3475 ± 0.0959 | 0.1435 |
| RANDOM-POOL (non-adaptive) | -0.0363 ± 0.2554 | +0.2733 ± 0.0319 | 0.0904 |

## M1 -- CONFIRMED as registered

Per-seed:
```
control       [-0.5286, -0.3763, -0.4213, -0.5691, -0.6181]
ORACLE        [+0.1426, -0.1718, +0.4441, -0.0542, +0.3622]
DIVERSE-GOOD  [+0.1421, -0.1040, +0.3766, -0.0352, +0.2158]
RANDOM-POOL   [-0.0727, -0.3074, +0.1728, -0.2493, +0.2752]
```
**Every one of the 5 control runs sits below every one of the 15 frozen runs**,
gap +0.0689. No p-value computed and none needed: it is a complete separation.

The control's queries respond to what it observes -- a good outcome is followed
by a short step. The three substitute arms show no such relationship, and their
spread is 2-2.6x wider, which is what "no systematic relationship" looks like.
The three collapse together on M1 (+0.14 / +0.12 / -0.04) exactly as they
collapse together at 43.94 rel%.

## M2 -- registered prediction FAILED

I registered "control LOWEST". Control is **HIGHEST** (+0.7026 vs 0.27-0.39).
The direction is wrong, and it is recorded as a failed prediction.

The separation is still present and still splits control from all three frozen
arms. Post-hoc, the sign is coherent with M1: a policy that exploits stays in a
region, which RAISES lag-1 autocorrelation rather than lowering it. My
registered reasoning -- that a conditioning-token-driven policy would be
smoother -- was simply backwards. **That reinterpretation is post-hoc and is
not evidence.** It is written down so the failure is not quietly converted into
support.

## Verdict

MIXED. One registered measure confirmed with a complete separation; one
registered direction refuted. Per the locked gate this is reported as MIXED and
NOT as support.

Effect on h153: it continues. The prior on the adaptivity hypothesis is
somewhat raised by M1 and somewhat lowered by my having reasoned wrongly about
M2 -- I understood the mechanism less well than the M1 result alone suggests.

## Caveat carried forward from h150 (registered before running)

h150 tested "policy distillation of MES" with query-level statistics and its
result retracted an already-published finding. Query-level statistics at n=5
are weak evidence about what a network internally represents. **h154 does not
establish the mechanism and is not reported as having done so.** h153 is the
direct test.

---

# h154b -- the open-loop penalty on HARTMANN. Registered generality check. REPLICATES.

|  | Borehole | Hartmann |
|---|---|---|
| greedy CLOSED-loop | +0.4860 | +0.4375 |
| greedy OPEN-loop (identical rule, frozen) | +0.3266 | +0.3530 |
| **OPEN-LOOP PENALTY** | **+0.1594** | **+0.0845** |
| in noise floors | 0.72 | 0.51 |
| closed-loop wins | 16/21 | **18/21** |
| noise floor | 0.2218 | 0.1651 |
| beam vs greedy, BOTH open-loop | -0.0397 (10/21) | -0.0225 (10/21) |

The penalty is NOT Borehole-specific. It is roughly **half the magnitude** on
Hartmann (+0.0845 vs +0.1594) but MORE consistent by count (18/21 vs 16/21) --
Hartmann's noise floor is lower, so a smaller effect separates more reliably.
The registered narrowing ("if Hartmann shows no penalty, h152's generality
claim must be narrowed") does NOT fire.

Second, unregistered but worth recording: "joint optimisation buys nothing at
matched loop type" replicates almost exactly -- -0.0397 (10/21) on Borehole,
-0.0225 (10/21) on Hartmann. Two benchmarks, the same dead heat, the same
count. h152's central negative is not a Borehole artefact.

MAGNITUDE CAVEAT UNCHANGED: +0.08 to +0.16 rtg is real and replicated, and
still NOT large enough on its own to explain the 0.976 -> 0.311 rtg_target
collapse. Whether adaptivity matters through POLICY LEARNING rather than
through the reward magnitude is what h153 tests directly.
