# H6 — PRIMARY AND FINAL ANALYSIS (pre-registered n=30)

Per `protocol-extension.md` (b00dacd): n=30 was fixed in advance, this analysis
is primary **and** final, and **no further extension will be run** regardless of
outcome.

## Result

| | FROZEN (DT fixed after iter 5) | LIVE (retrained each iter) |
|---|---|---|
| mean regret | 0.5898 | 0.5146 |
| sd | 0.2298 | 0.1636 |

    paired diff = +0.0752   sd 0.2393   SE 0.0437
    95% CI      = [-0.0105, +0.1608]      <- contains zero
    Wilcoxon    p = 0.0795
    FROZEN better on 9/30 seeds

## Locked predictions

1. **CI still contains zero — MET.** This was deliberately a prediction of a
   *null*, registered in advance so that a null could not later be dressed up as
   a finding. It held.
2. **Levene p < 0.05 on the variance — NOT MET** (p = 0.1964). The variance ratio
   is 1.97x, but on right-skewed regret data the robust test governs and it does
   not reach significance. The variance story, which looked like the sturdiest
   part of H6 at n=10, does **not** survive at n=30 either.

## Conclusion, stated at the strength the evidence supports

**Freezing the DT after 5 of ~60 iterations is not distinguishable from
retraining it throughout**, at n=30, on this benchmark. The point estimate
(+0.075, i.e. FROZEN nominally *worse*) leans toward continued training helping
slightly, and Wilcoxon p = 0.0795 is suggestive without clearing 0.05, but the
CI includes zero and the effect is smaller than the paired sd.

This is **weaker** than the claim I was tempted to make at n=7 ("freezing costs
nothing — continued training contributes ~nothing"). The honest statement is
that the experiment **cannot distinguish** the two, and that the nominal
direction mildly favours continued training rather than disfavouring it.

## The estimate's trajectory is itself the cautionary result

    n=1   -0.208     "freezing helps 40%"
    n=5   -0.103     outside 1 SE, prediction refuted
    n=7   -0.010     within 1 SE, prediction supported
    n=10  +0.098     outside 1 SE, opposite sign
    n=30  +0.075     CI contains zero, p=0.08

Every intermediate reading was publishable-sounding and every one of them was
different. Only the pre-registration — n fixed in advance, primary-and-final,
null predicted — prevents this from becoming a story about whichever n happened
to be reached when someone stopped looking.

## Relationship to H7

H7 asks the same question with a far better instrument (decision agreement
between the live policy and its iteration-5 snapshot: ~50-200 paired decisions
per run instead of one regret scalar). Its smoke test already shows the live and
snapshot policies selecting **bit-identical** proposals (`dist = 0.000000`), with
an independence check running to rule out the trivial explanation.

If H7 confirms near-total decision agreement, the two experiments together give
the defensible version of the claim: *continued training changes the policy's
decisions almost not at all, and the regret consequence of that is too small for
30 seeds to resolve.* That is a more precise statement than either alone.
