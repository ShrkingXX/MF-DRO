# h166 — **R2 fires (n=3).** The target-collapse refutation is now two-benchmark.

CONFIRMATORY. No harness forecast was offered: h166 is a frozen condition, and
the harness's sole validation failure (C2, off 2.7×) is exactly the frozen case.

## The completed Hartmann 2×2

| | CLOSED-loop | OPEN-loop (frozen) |
|---|---|---|
| **MES** | control **7.99**, 5/5, rtg 0.8844 | **h166 6.75**, 3/3, rtg **0.4002** |
| **non-MES** | h165 **10.58**, 5/5, rtg 0.8486 | RANDOM-POOL **65.14**, 2/5, rtg 0.2924 |

**Three of four cells work**, the same structure as Borehole. Working arms
6.75–10.58 against a failing arm at 65.14.

## The split reproduces

h166's `rtg_target` is **0.4002** — far nearer RANDOM-POOL's 0.2924 than the
control's 0.8844 — while it **performs best of all four arms** (6.75 against the
control's 7.99). That is the h153 pattern exactly: collapsed conditioning target,
good performance.

**R2 as registered**: it works AND its target collapses, so the refutation of the
target-collapse account — previously resting on Borehole alone, and stated
without a benchmark qualifier in findings.md and the published report — now
holds on two benchmarks and two independent frozen arms (h153, h161, h166).

R3, the outcome that would have looked like success and carried no evidence
(works but target does NOT collapse), does not fire.

Sanity: SC1 path replay error **0.0** on all three seeds. SC2 open-loop penalty
0.146 / 0.283 / 0.324 — all positive, and averaging ~0.25 against Borehole's
~0.35, consistent with h154b's finding that the Hartmann penalty is roughly half
Borehole's.

## Status

n=3 of 5. The verdict is R2 and it is stable across all three seeds; the
remaining two are running.

## h165 completed at n=5

10.58 rel%, improving **5/5**, rtg 0.8486, HF fraction 0.290 (against the
control's 0.200). The confound noted at n=4 persists but shrank (0.353 → 0.290),
and its direction still runs *against* the claim: h165 buys more HF and performs
worse than the control, so the shift cannot manufacture its success.
