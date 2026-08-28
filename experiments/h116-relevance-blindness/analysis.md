# H116 — analysis

Data: h83-main-comparison only (4 bench x 5 method x 5 seeds, 42-46). Zero new runs.

## 1. VERDICT ON THE LOCKED PREDICTION: NOT SUPPORTED. GATE MISSED.

Predicted: on Borehole, median rho(MF-MES) < 0, median rho(MF-DRO) ~ 0,
with paired |mean|/sd >= 1.0.

Unit-cube result (post-Amendment-1), Borehole, n=5 paired seeds:

| profile stat | rho MF-DRO | rho MF-MES | paired mean | \|mean\|/sd | gate |
|---|---|---|---|---|---|
| SD (primary)  | +0.024 (sd 0.41) | -0.357 (sd 0.19) | -0.076 | **0.17** | MISSED |
| MAD (robust)  | -0.571 (sd 0.24) | -0.524 (sd 0.22) | +0.041 | **0.10** | MISSED |

The SD medians look like the prediction (DRO ~ 0, MES < 0). They do not
survive the pre-registered paired test: per-seed variance is large (DRO sd
0.41) and the effect is 0.17, not >= 1.0. The MAD variant REVERSES the sign
of the difference. Two declared variants disagreeing in sign, both far below
the gate, is not a result. Reported as a gate miss.

Hartmann (secondary): 4 of 5 seeds excluded by the Amendment-2 n>=15 floor
(MF-DRO non-init HF counts 8-24). One usable pair. NO INFERENCE.
Currin and Ackley were pre-declared degenerate; their numbers are in
results/ and are excluded from inference as planned.

## 2. SUPERSEDED FIRST RUN (disclosed)

The pre-amendment run reported Borehole rho = -0.405 (sd 0.02) for MF-DRO
and -0.429 (sd 0.03) for MF-MES. Those numbers are an artefact: stored `x`
is in RAW units on Borehole (box widths up to 5.25e4), so the "dispersion
profile" was mostly the fixed domain box -- the same constant for both
methods and all seeds, which is why the sd across seeds was 0.02. Superseded
by Amendment 1. See protocol.md.

## 3. EXPLORATORY (NOT pre-registered): where the excess dispersion sits

The Spearman measure ranks dimensions and is therefore insensitive to
magnitude. A magnitude measure on the same data, same seeds, unit cube:
total dispersion sum_j sd_j, aggregated weighted by S1 share (`perdim.agg`,
the project's default) versus unweighted.

MF-DRO / MF-MES total HF-query dispersion, mean over seeds:

| bench | weighted | unweighted | per-seed weighted/unweighted | \|mean log\|/sd |
|---|---|---|---|---|
| Borehole_8D | **3.96** | **1.06** | 2.95, 3.22, 2.96, 3.67, 5.53 | **4.84** |
| Currin_2D   | 2.84 | 1.65 | 1.48, 1.92 (n=2 after floor) | 2.83 |
| Ackley_10D  | 3.00 | 3.00 | 1.00 x5 | 0.30 |

On Borehole the two methods disperse their HF queries by essentially the
SAME total amount (1.06x) but MF-DRO's is ~4x larger once weighted by
variance share. Its excess spread sits in the dimensions that carry the
objective's variance, and it is correspondingly tighter in the ones that do
not. This is seed-paired and within-seed; 5/5 seeds exceed 2.9.

Ackley's exact 1.00 is DEFINITIONAL, not evidence: its S1 shares are uniform
(PR/d = 1.000), so weighted and unweighted aggregates are identically equal.
It is a check that the code does what it claims, nothing more.

Effect on the reading of h116: the hypothesis was that MF-DRO is BLIND to
relevance, i.e. allocates dispersion without regard to it. That is not what
the data show. Its allocation is not random with respect to relevance -- it
is systematically concentrated in the high-relevance dimensions, which is
the opposite of what MF-MES does (MES tightens them: rho -0.357 under SD).
"Blind" is the wrong word; "fails to localise along the dimensions that
determine the objective" is what is actually measured.

## 4. Consequence for the proposed intervention

The motivating code facts stand and are unchanged:
KO GP is ARD (`ko_gp.py:312`); `mf_dro.py:251` averages the lengthscale
vector; that is the only point at which lengthscales enter the pipeline.

But the intervention those facts suggested -- weight `L_loc` by ARD-derived
relevance so the policy stops mis-spending dispersion -- is NOT supported by
a passing pre-registered test. It rests on the exploratory Section 3 result
only. It must not be launched as though h116 had confirmed anything.

## 5. Limitations

- n=5 seeds. No p-values (project rule).
- Section 3 is EXPLORATORY: the measure was chosen after seeing Section 1 fail.
  It requires confirmation on independent seeds before it is load-bearing.
- Executed HF queries are NOT the DT's raw proposals. The founding
  diagnosis's "3x more dispersed" concerned proposals. The agreement between
  its 3.65x/3.73x and the 3.96x here is consistent but is NOT the same
  quantity measured twice.
- S1 shares are binned first-order estimates and ignore interactions.
- Hartmann is absent from Section 3 (n floor), so the anisotropy ordering
  Borehole > Currin > Ackley rests on 3 benchmarks, one of them degenerate.
