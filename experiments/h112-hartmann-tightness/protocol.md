# H111 — is the ROI's benchmark-specificity a TUNING artifact?

LOCKED BEFORE ANY RUN. 5 runs: Hartmann_6D ROI-Q05 at seeds 42-46.

## The question nobody has asked

Every ROI effect measured in this project is Borehole-specific (h104): regret,
relocation, waste reduction, query quality all appear there and are absent on
Hartmann -- the benchmark the commissioning diagnosis was actually drawn from.

**But every Hartmann ROI run used q=0.10.** h84 tested ROI-Q10, ROI-FIX2 and
ROI-ANN there; h87 confirmed the failure at fresh seeds. **q=0.05 has never been
run on Hartmann.**

That gap matters because of this project's own central argument. A CONSTANT beta
gives wildly different acceptance across benchmarks -- 12.6% to 100% -- which is
why beta_t must be calibrated. **The same reasoning applies one level up: a
constant TARGET ACCEPTANCE may be as benchmark-inappropriate as a constant
beta.** q=0.10 was never chosen by measurement anywhere; it was the first value
tried, on Borehole.

So: does the ROI fail on Hartmann, or does q=0.10 fail on Hartmann?

## Design

    arm         Hartmann_6D, ROI-Q05, seeds 42-46          5 NEW runs
    comparator  h84's Hartmann ROI-Q10 at 42-46            reuse, explicit path
    control     h83's Hartmann MF-DRO at 42-46             reuse, explicit path

Config is h97/h107/h110's ROI-Q05 exactly (use_roi=True, quantile beta_t,
roi_target_accept=0.05). Runs on code h109 verified bit-identical.

## Predictions (effect sizes stated)

**S1 (PRIMARY). q=0.05 does NOT rescue Hartmann: paired vs no-ROI, |mean| < 3.0
OR ratio < 0.5.** Registered POSITIVE -- i.e. I expect the ROI to fail here too.
Grounds: h104 measured NO waste reduction and NO query-quality gain on Hartmann
at q=0.10 (-0.013 and -0.001), and h96 found NO relocation there. Three separate
mechanism measurements say nothing is happening, and tightening a region that is
not helping should not obviously start helping.

**S2 (THE DISCRIMINATOR, registered NEGATIVE). If S1 is REFUTED -- if q=0.05
delivers on Hartmann what q=0.10 did not -- then the ROI's Borehole-specificity
is a TUNING artifact, not a benchmark property**, and every "Borehole-only"
statement in findings.md needs rewriting. That would be the most consequential
result of this investigation and must be re-verified at fresh seeds before it is
believed anywhere.

**S3. Hartmann's measured acceptance lands in [0.045, 0.055].** The gate, in
h97's form: OBSERVED accept_frac, not the requested target. If calibration
cannot hit 0.05 on Hartmann, S1/S2 are uninterpretable and say so.

## Why this is worth 5 runs

If S1 holds, "the ROI is Borehole-specific" survives a real attempt to break it,
which it has not yet faced -- the claim currently rests on one setting never
chosen by measurement. If S1 fails, the central limitation of this
investigation dissolves. Either way the answer is worth more than the runs.

## Gate

2 workers running. 5 more takes it to 7, inside 15.
