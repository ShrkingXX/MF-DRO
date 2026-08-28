# H96 — does the ROI RELOCATE the query cloud rather than concentrate it?

ZERO NEW COMPUTE. Reads h90's completed Borehole runs (both arms 5/5).
LOCKED BEFORE ANY NUMBER IS COMPUTED.

## Why

h95 established that the ROI improves mean HF query regret (-4.15, 5/5) and
final regret (-3.49, 4/5) while **increasing** dispersion (+0.010, 4/5) --
replicated independently by the concurrent session under a different statistic.
So the ROI works, and it does NOT work by concentrating proposals.

That leaves the mechanism unexplained. This tests the natural alternative:

  **RELOCATION, NOT CONCENTRATION.** The ROI excludes low-value regions. The
  surviving mass can then spread over a high-value plateau -- which raises
  dispersion while moving the cloud CLOSER to the optimum. Exclusion is not
  the same operation as concentration, and only the second one narrows.

## The metric trap this protocol exists to avoid

`query_dist_to_xstar_per_iter` IS logged, and on Borehole it is **the misleading
metric**. findings.md:3174 records that unweighted Euclidean distance in
normalized 8-D space REVERSED the true ordering: MF-DRO looked closer to x* than
MF-MES (0.2535 vs 0.2998) purely by sitting near x* in the four dimensions that
carry 0.4% of the output variance. Sensitivity shares, measured by freezing each
dim at its midpoint over 3000 samples:

    dim 0: 81.6%   dim 6: 8.0%   dim 5: 5.4%   dim 3: 4.6%   dims 1,2,4,7: 0.4%

and all four sensitive dims have x* ON THE BOUNDARY.

So the PRIMARY measure here is the SENSITIVITY-WEIGHTED distance
sqrt(sum_i w_i (x_i - x*_i)^2) in normalized coordinates, with
w = [.816, .001, .001, .046, .001, .054, .080, .001].

**Both measures will be reported and the verdict turns on the WEIGHTED one.**
That is stated now, before computing, because h85's P4 was first reported MET on
whichever of two admissible measures happened to pass. Naming the deciding
measure in advance is the rule that came out of it.

## Predictions (registered before computing)

**R1 (PRIMARY). Weighted distance to x* falls under the ROI on >= 4/5 seeds.**
Registered POSITIVE. This is the relocation claim.

**R2. Unweighted distance need NOT fall, and if it moves the opposite way that
is EXPECTED, not a contradiction.** Registered explicitly so a divergence
between the two cannot later be presented as a surprise or as a failure.

**R3. The fraction of HF queries within 0.05 of x* in the four SENSITIVE dims
rises under the ROI.** h83 measured MF-DRO at 68%/1%/0%/2% on dims 0/3/5/6
against MF-MES's 99%/49%/34%/70%. If relocation is real, the ROI should move
these up. Registered POSITIVE but WEAKLY -- the boundary-aversion finding says
the head structurally struggles to reach bounds, and the ROI does not change the
output parameterisation.

**R4 (THE DISCRIMINATOR). If R1 holds while dispersion rises, the mechanism is
relocation-with-spread and "concentrate the proposals" is the wrong prescription.
If R1 FAILS, then the ROI's benefit is not explained by WHERE the queries are at
all**, and the remaining candidate is that it changes the fidelity mix or the
surrogate rather than the locations. That would be a genuinely open question and
must be reported as one, not papered over.

## Falsifier

If R1 fails, "the ROI relocates the query cloud toward the optimum" must not be
written. h95's measured facts (better mean query, more dispersion) stand
regardless; only the explanation dies.

## Label

EXPLORATORY. No pre-existing bar; this is mechanism, on n=5, one benchmark.

## Amendment 1 — the relocation account must be tested on HARTMANN, registered before computing

H96 concluded the ROI works by RELOCATING the query cloud toward x* in the
dimensions that matter. **That conclusion is Borehole-only, and the concurrent
session has just shown what goes wrong with Borehole-only conclusions here**: the
founding diagnosis was measured on Hartmann, and the dispersion story reversed
sign between the two benchmarks. The same objection applies to relocation with
equal force, so it gets a registered test rather than an assumption.

Hartmann is the natural falsifier because the ROI FAILED there (h87, 2/5,
withdrawn). If relocation is what makes the ROI work, then on the benchmark
where it did not work, relocation should be ABSENT or much weaker.

### Predictions (registered before computing)

**H1. On Hartmann, the ROI does NOT reduce weighted distance to x* on >= 4/5
seeds.** Registered POSITIVE for the relocation account -- i.e. I expect
relocation to be absent where the ROI failed.

**H2 (THE DISCRIMINATOR). If relocation IS present on Hartmann while the regret
result failed, then relocation is NOT SUFFICIENT** -- exactly the shape the
dispersion story just collapsed into, and the relocation account would then be a
Borehole description rather than a mechanism. That must be reported in those
words, not softened.

**H3. Hartmann's sensitivity profile must be measured, not assumed.** Borehole's
weights (dim0 81.6% etc.) are Borehole's. Hartmann's will be estimated by the
same procedure recorded in findings.md:3183 -- freeze each dimension at its
midpoint, measure the loss of output variance over 3000 samples -- and reported
before the distance numbers. If Hartmann's variance is spread evenly across
dims, then weighted and unweighted distance nearly coincide there and the
metric-choice caveat that mattered on Borehole does not apply; that would be
worth knowing on its own.

Hartmann's x* is INTERIOR in all six dimensions (0.15-0.657, findings.md:3217),
so the boundary-aversion mechanism that bounds the Borehole gain is absent
there. Relocation and boundary reach are therefore separable on Hartmann in a
way they are not on Borehole.

### Falsifier

If H1 fails (relocation present on Hartmann anyway), H96's mechanism claim is
downgraded from "the mechanism" to "a Borehole-specific description", and
findings.md must say so where the claim is made, not in a footnote.
