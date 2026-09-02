# h156d -- Gate **FAILED**. And it caught me over-claiming precision in h156.

## The gate

PASS required BOTH: C4's Hartmann error inside ~10%, AND the cross-benchmark
direction corrected. Result:

  direction   observed +0.0509 | old harness −0.0532 (WRONG) | new +0.0163 (CORRECT)
  magnitude   Hartmann C4 error 29.8%  (was 31.2%)

Direction fixed, magnitude essentially unmoved. Pre-stated gate was an AND, so
this is a **FAIL**, and the fidelity shortcut is NOT the explanation for the
interpolating-path misfit.

## The more important finding: my harness is noisier than I reported

h156d changed ONLY C4/C5. But `fid_of` now draws Thompson samples inside the
loop, shifting every later RNG draw — so C1/C2/C3 in tail5 vs tail6 are a **pure
replicate** of the same computation. Their spread is the harness's own noise:

| | run A | run B | Δ% |
|---|---|---|---|
| Borehole C1 | 0.9060 | 0.9615 | 5.9% |
| Borehole C2 | 0.8238 | 0.9185 | **10.9%** |
| Borehole C3 | 0.2739 | 0.2565 | 6.6% |
| Hartmann C1 | 0.7860 | 0.8147 | 3.6% |
| Hartmann C2 | 0.7148 | 0.7595 | 6.1% |
| Hartmann C3 | 0.2789 | 0.2892 | 3.6% |

**Harness noise on the MAX statistic: 6.1% mean, 10.9% worst.**

### Correction to h156's headline

I reported "reproduces all four observed rtg_targets, three of them within 8%,
not tuned to them" — and repeated it in the published report. **With a 6% noise
floor, "within 8%" is barely above noise and was not the tight quantitative
agreement I presented it as.** Re-reading both replicates against the noise:

| | errors (2 replicates) | verdict |
|---|---|---|
| Borehole C1 / control | −7.2%, −1.5% | within noise |
| Hartmann C1 / control | −11.1%, −7.9% | within noise |
| Hartmann C3 / RANDOM-POOL | −4.6%, −1.1% | within noise |
| Borehole C3 / RANDOM-POOL | −7.6%, −13.5% | borderline |
| Borehole C4 / ORACLE | −2.9%, −23.6% | **real misfit** |
| Hartmann C4 / ORACLE | −31.2%, −29.8% | **real misfit** |
| Borehole C5 / DIVERSE-GOOD | −19.3%, −24.7% | **real misfit** |

The −2.9% that made C4 look like the harness's best fit was **noise**; its
replicate is −23.6%.

## What survives, stated at the precision the data supports

**The scale separation is robust and is not in question.** Control 0.79–0.96
against every failing arm 0.24–0.29 — a 3–4× gap against a 6% noise floor, on
two benchmarks, in four independent runs. The qualitative account (the failing
arms have no informative tail for a max to find) stands.

**The quantitative claim does not.** The harness systematically
UNDER-predicts the two interpolating conditions by 20–30%; the real ORACLE and
DIVERSE-GOOD arms achieve higher targets than trajectory geometry alone
predicts. That is unexplained, and the fidelity shortcut — my leading
candidate — is now ruled out.

## The h153 forecast is unaffected and is now better supported

C2/C1 across two benchmarks × two replicates: **90.9%, 95.5%, 90.9%, 93.2%**.
Range 90.9–95.5%, tight relative to the noise floor. The forecast that h153
should NOT collapse its target, and should land near the control, stands on
four independent measurements rather than one.
