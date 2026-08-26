# H65 result — predictions technically met, effect substantively absent

**CONFIRMATORY.** Verdict script written and committed while the arm stood at 0/3.

| arm | s44 | s46 | s48 | mean | rel | spread |
|---|---|---|---|---|---|---|
| BASE | 0.7531 | 0.2875 | 0.4228 | 0.4878 | 14.7% | 0.4656 |
| REFINE | 0.6664 | 0.4034 | 0.2060 | 0.4253 | 12.8% | **0.4604** |

| prediction | result |
|---|---|
| 1 PRIMARY (variance): REFINE spread < BASE spread | **MET** — 0.4604 vs 0.4656 |
| 2 SECONDARY (mean): REFINE beats BASE >=2/3 | MET (2/3) |
| 3 BOREHOLE-SPECIFIC: neither moves | no |
| 4 HARMFUL: spread increases | no |

## Both predictions met, and the result is still negative

The PRIMARY passed on a **1.1% spread contraction**. On Borehole, h61 measured
BASE 8.62 -> REFINE 2.57, a **3.4x** contraction. That is the effect h65 was
built to test for generalisation, and 1.1% at n=3 is indistinguishable from noise.

**The bar was badly specified.** "REFINE spread < BASE spread" is satisfied by any
reduction whatsoever, including one far smaller than the sampling variation of a
3-seed spread. A prediction that cannot fail against a null effect is not a test.
The honest verdict: **h61's Borehole variance collapse does NOT generalise to
Hartmann**, and the locked criterion failed to say so.

This is the second protocol-design error in this session, after h68's PRIMARY
(necessary but not sufficient for the conclusion it licensed). Both passed while
the substantive claim failed. Recording the pattern: **a locked prediction needs a
minimum effect size, not just a direction**, whenever the quantity is noisy.

## Context

The broader "DRO buys variance not mean" hypothesis was already refuted across all
six DRO-vs-MES pairs (DRO better on worst-case in only 2/6). h65 was testing the
narrower h61-specific claim about REFINE's teacher refinement, and that too fails
to generalise off Borehole.
