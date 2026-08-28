# H95 — does the confirmed ROI gain come from wasting less HF budget?

ZERO NEW COMPUTE. Reads h90's completed Borehole runs (NO-ROI 5/5, ROI-Q10 5/5).
LOCKED BEFORE THE NUMBERS ARE COMPUTED.

## Why this is not already answered

h90 confirmed the ROI lowers final simple regret on Borehole (-3.49, 4/5).
**That is not the same claim as the one this investigation was commissioned to
test.** The commission was: find an ROI strategy that stops MF-DRO WASTING HF
BUDGET ON LOW-VALUE REGIONS. Simple regret is a MAX statistic (Lesson 22) -- a
method can lower it by landing one good query while wasting exactly as many
others. Nothing measured so far distinguishes those.

The established diagnosis was stated in query-quality terms: mean HF query score
0.336 vs MF-MES's 0.747 on Hartmann, 20.8% of HF queries landing WORSE than the
initial design, proposals 3x more dispersed. Those are the quantities the
commission names, and no ROI experiment has reported them.

## Measures (defined here, before computing)

Post-initial-design HF queries only. Per run, then paired across seeds 47-51.

  W  WASTE FRACTION: share of post-init HF queries whose y is worse than
     `best_init` = the best HF value already present in the initial design.
     A query landing below a value you already had bought you nothing.
  D  DISPERSION: mean over dimensions of the std of post-init HF query
     locations, in normalized [0,1]^d coordinates.
  Q  MEAN QUERY REGRET: mean over post-init HF queries of 100*(opt - y)/opt.

These are MY definitions and may not match whatever normalization produced the
0.336/0.747 figures. That is acceptable because every comparison here is
ROI vs no-ROI under ONE definition, on the same runs. It would NOT be acceptable
to quote 0.336 alongside these; I will not.

## Predictions (registered before computing)

**M1. W falls under the ROI on >= 4/5 seeds.** Registered POSITIVE. This is the
commission's own claim; if the ROI improves regret without lowering W, the gain
is not "less waste" and the commission's framing is wrong about the mechanism.

**M2. D falls under the ROI on >= 4/5 seeds.** Registered POSITIVE, and weakly.
The ROI restricts the teacher's pool, which should tighten the student's
proposals -- but the audit established that the ROI reaches the query only
through a lossy imitation channel, so the effect could be small.

**M3. Q falls under the ROI on >= 4/5 seeds.** Registered POSITIVE.

**M4 (THE DISCRIMINATOR). If M1 fails while the regret gain holds, then the ROI
improves the BEST query without reducing waste** -- and the honest report is
that it does not do the thing it was commissioned to do, even though it helps.
Registered explicitly so that outcome cannot be quietly reframed as success.

I am NOT confident in M1-M3. h88 found MF-DRO's dispersion produces a BETTER
global surrogate than MF-MES's concentration, so "less dispersed" is not
automatically better and M2 could be true while M3 is false, or vice versa.

## Falsifier

If M1 fails, the sentence "the ROI stops MF-DRO wasting HF budget" must not
appear in findings.md, the report, or the paper. The defensible sentence would
be "the ROI lowers final regret on Borehole", which is what was measured.

## Label

EXPLORATORY on mechanism, CONFIRMATORY on nothing. h90's regret result is the
confirmatory one; this asks what produced it and has no pre-existing bar.
