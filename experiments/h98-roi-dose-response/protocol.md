# H98 — is CENTRING the mediator between ROI tightness and regret?

ZERO NEW COMPUTE. h84's three ROI arms on Borehole seeds 42-46, plus a no-ROI
control. LOCKED BEFORE COMPUTING.

## Why

h96: the ROI helps by RELOCATING the query cloud toward x* in the sensitive dims.
h97: reach is governed by (distance from the head's centre to the bound) / sd,
     and the head centres correctly but too far short in dims 3/5/6.

Both are single-arm accounts: ROI on vs ROI off. h84 ran THREE tightness
settings on Borehole and their regret outcomes are already known, which gives a
four-level dose to test the mediation claim rather than assert it.

    arm        target accept   MEASURED accept   Borehole regret gain
    ROI-OFF        --              1.0                    0
    ROI-ANN     annealed          0.4934              -1.31
    ROI-FIX2    fixed beta        0.2141              -4.81
    ROI-Q10     q=0.10            0.0999              -4.22

**Note the regret is NOT monotone in tightness**: FIX2 (looser) beats Q10
(tighter). That is already on record and it is the reason this test is worth
running -- if centring is the mediator, centring should track REGRET, not
tightness, and should reproduce the FIX2 > Q10 inversion.

## Measures

Post-init HF queries, normalized, per arm pooled over seeds 42-46. Sensitive
dims 0/3/5/6 (99.6% of output variance; all four have x* on a bound).

  OFFSET  mean over sensitive dims of |mean coordinate - 0.5|  (centring)
  GAPSD   mean over sensitive dims of |mean - x*| / sd         (h97's predictor)
  REACH   mean over sensitive dims of the fraction within 0.05 of x*

## Predictions (registered before computing)

**T1. Centring is NOT monotone in tightness.** Registered POSITIVE. Regret is
not, and if centring is the mediator it must inherit regret's shape, not
tightness's. A monotone centring result would mean centring cannot explain the
FIX2 > Q10 inversion and would weaken the mediation story.

**T2 (PRIMARY). The arms ranked by GAPSD (ascending = better) match the arms
ranked by regret gain (descending = better), including the FIX2 > Q10
inversion.** Registered POSITIVE. This is the mediation claim in its strongest
checkable form.

**T3. REACH ranks the same way.** Registered POSITIVE but weaker: reach is a
tail statistic on 5-25 queries per dim per run and is noisier than GAPSD.

**T4 (FALSIFIER). If GAPSD is flat across arms while regret varies by 4.8
points, centring is NOT the mediator** and h96/h97's account explains the
ROI-on/ROI-off contrast but not the dose. It would then be a description of one
comparison rather than a mechanism, and must be labelled that way.

## What this cannot establish

**n=4 ARMS. No correlation coefficient will be computed and no p-value exists.**
The test is an ORDERING match between two rankings of four items, which under a
null has a 1/24 chance of matching exactly -- suggestive at best, and stated here
so it is not later dressed up. Each arm is n=5 seeds. Different arms differ in
more than acceptance rate (FIX2 holds beta fixed so its tightness DRIFTS within
a run, 250x on this benchmark, while Q10 and ANN recalibrate) -- so "tightness"
is not a clean scalar dose and FIX2 is not simply "looser than Q10".

That last point is the main threat to interpreting any result here and it is
registered before seeing numbers.

## Label

EXPLORATORY. Mediation on existing data.
