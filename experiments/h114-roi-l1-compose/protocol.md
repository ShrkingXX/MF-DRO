# H114 — do the two surviving Borehole interventions COMPOSE?

LOCKED BEFORE ANY RUN. 10 runs: Borehole ROI-Q05 + L1-LOSS at seeds 42-51.
ID claimed via tools/claim_id.sh. **Arm checked against the results tree first**
(`ls experiments/*/results/*L1*Q05*` -> none) -- the rule adopted after I nearly
duplicated the peer's h111.

## Why

Two interventions survived replication on Borehole, both from independent
sessions, both mechanistically unexplained, and both improving regret without
moving the quantity their proposed mechanism operates on:

    ROI-Q05   -5.40  sd 2.98  10/10   (mechanism: relocation, not concentration)
    L1-LOSS   -2.21  sd 1.78   9/10   (mechanism: boundary reach -- bar NOT cleared)

The peer raised the question neither of us had examined: **if there is a shared
explanation, it is here.** Nobody has run them together.

## The two hypotheses make DIFFERENT point predictions

Measured on the same 10 seeds against the same controls (control mean 15.78):

    ADDITIVE (independent channels)  combined ~ 15.78 - 5.40 - 2.21 = **8.17**
    OVERLAPPING (one channel)        combined ~ the better alone     = **10.38**

    separation 2.21 points, against a paired sd of ~3 -> ratio ~0.74 at n=10,
    which clears a 0.5-sd bar. **This experiment can distinguish them.**

## And the additive prediction lands on top of the baseline

**MF-MES on these seeds is 8.24.** The additive prediction is 8.17. So if the
two effects compose, MF-DRO+ROI+L1 essentially TIES the strongest comparator on
the one benchmark where MF-DRO has a real deficit.

That is a large claim and it is registered here as a PREDICTION OF A HYPOTHESIS,
not as an expectation. If it comes out that way it must be re-verified at fresh
seeds before it is reported as anything, per this project's record with results
that looked decisive at n=5 or n=10 and did not survive.

## Predictions

**C1 (PRIMARY, GENUINELY UNCERTAIN). The combined arm's mean is closer to 8.17
than to 10.38.** Registered with NO direction preferred. Both mechanisms are
unexplained, so there is no principled basis to expect independence or overlap,
and I decline to manufacture one.

**C2. The combined arm beats EACH intervention alone, paired, on >= 7/10 with
|mean| >= 0.5 sd against the better of the two (ROI).** Registered POSITIVE but
weakly -- it holds under additivity and fails under full overlap, so it is
largely a restatement of C1 and is reported alongside rather than as independent
support.

**C3 (NEGATIVE). The combined arm does NOT beat MF-MES paired on >= 6/10.**
Registered NEGATIVE deliberately: the mean-based additive prediction ties MF-MES,
but this project has repeatedly found mean-based and paired verdicts to
disagree, and MF-MES's own spread on these seeds is large. **The paired count
must be reported with any mean comparison.**

**C4 (GATE). Measured accept_frac in [0.045, 0.055] on all 10 runs, and the L1
flag verifiably active** -- L_loc should sit near 0.36 as the peer measured for
L1, against ~0.033-0.038 under MSE. If either fails, C1-C3 are uninterpretable.
Both are OBSERVED quantities, not config read-backs.

## Falsifier

If C1 favours overlap, the two interventions act through one channel, and the
"two unexplained mechanisms" framing collapses to one unexplained mechanism with
two implementations -- which would be the more informative outcome for the
write-up.

## Gate

2 workers running. 10 more takes it to 12, inside 15.
