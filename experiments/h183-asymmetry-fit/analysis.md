# h183 — a THIRD account of the benchmark asymmetry. This one is supported, not refused.

**EXPLORATORY.** No new runs. Not pre-registered — it came out of a diagnostic scan,
and the scan's weaknesses are recorded below.

## How this was found, including the part that did not work

First attempt was a scan of 12 saved diagnostics across four Hartmann arms, asking
what separates the two best (7.93, 7.99) from HEAD-MES (25.16). **Ten of the twelve
flagged HEAD as outside the others' range** — the multiple-comparisons failure I
had named as a risk before running it. That scan is not evidence and is not used.

The principled filter uses the asymmetry's own structure: **HEAD has the same
teacher on Borehole, where it works fine.** So a diagnostic that is equally
disturbed on both benchmarks reflects the arm's design, not the asymmetry. Taking
each diagnostic's HEAD/control ratio on each benchmark:

| diagnostic | Borehole H/C | Hartmann H/C | tracks the asymmetry? |
|---|---|---|---|
| `L_fid_per_iter` | **1.020** | **1.483** | **yes** |
| `rtg_gpbelief_corr_per_iter` | 1.376 | **−2.254** (sign flip) | yes |
| `L_loc_per_iter` | 1.895 | 1.705 | no — disturbed on both |
| `grad_coherency_per_iter` | 0.383 | 0.195 | no — depressed on both |

## The structural fact behind it

The two benchmarks differ enormously in how much the method uses the cheap source:

| arm | Borehole `lf_fraction` | Hartmann `lf_fraction` |
|---|---|---|
| **MF-DRO (control)** | **0.1171** | **0.8004** |
| HEAD-MES | 0.1753 | 0.7544 |
| mean over all arms | **0.229** | **0.711** |

**Borehole's policy is high-fidelity by a factor of ~7.** Its fidelity decision
barely matters; Hartmann's method leans on LF for 80% of its queries and must keep
deciding fidelity well, all run long.

## The test, with the duplicate-inflation caught

If the asymmetry is about needing to fit the teacher, then how well the DT fits
should predict regret on Hartmann and not on Borehole. Across every arm with
per-iteration losses saved:

| | Borehole (25 arms) | Hartmann (18 arms) |
|---|---|---|
| ρ(fidelity loss, rel%) | **−0.240** | **+0.527** |
| ρ(location loss, rel%) | **+0.157** | **+0.608** |
| mean `lf_fraction` | 0.229 | 0.711 |

**Duplicate inflation, caught and removed.** Four Hartmann "arms" — PROBE-RANDOM,
EMB-RANDOM, BTG-RANDOM, RANDOM-POOL — carry *identical* losses and regret: they are
the same underlying run with RNG-neutral probes (bit-identity verified previously).
Counted separately they quadruple the worst point and inflated the correlations to
+0.699 / +0.750. Deduped they fall to **+0.527 / +0.608**. Borehole had one such
pair (ROI-FIX2 in h84 and h139). All figures above are post-dedupe.

## What this is, and what it is not

**It is:** a sharper *characterization*. On Hartmann, how well the DT fits its
teacher predicts how it performs. On Borehole, it does not — a Borehole arm can fit
badly and do well, which is exactly what "only the first step matters" implies.
This is the front's own answer showing up in the DT's training losses.

**It is not an explanation of the origin**, and saying "fit predicts performance
where fitting matters" is close to restating the asymmetry. The candidate *cause*
is the `lf_fraction` gap — Borehole 0.12 vs Hartmann 0.80 for the same method — which
is a genuine independent structural fact, not something inferred from the arms.
**That causal step is untested.**

## What could RETRACT this

- The correlations are moderate (0.53, 0.61) over 18 arms and the Borehole nulls
  are not exactly zero (−0.24, +0.16). A larger or differently-composed arm set
  could move either.
- Arms are not independent — many share code paths and initial designs.
- **The obvious test is not run:** force Borehole to a Hartmann-like `lf_fraction`
  (or the reverse) and see whether the asymmetry moves with it. That is the
  experiment this account calls for, and until it exists this is a hypothesis with
  supporting correlation, not a demonstrated cause.
- Unlike the two accounts before it, this one was *not* refused by its own numbers —
  but it was also not pre-registered, and it is the third attempt at the same
  question, which is itself a reason for caution.
