# h210 analysis — CONFIRMATORY. 25/25 finals, 0 tracebacks.

Frozen metric imported from h83's `grid`; finals only; endpoint only.

## Result

| arm | bench | regret | vs its control | dims-on-face | LF |
|---|---|---|---|---|---|
| **CTRL-K1-H** (h84 cfg on current code) | Hartmann | **5.93** | **+0.00, se 0.00, 5/5** vs h84's 5.93 | | |
| L1-NIR-H | Hartmann | 8.87 | −0.88 (se 1.77, 4/5) vs NIR-H 9.75 | | 0.62 |
| L1-NIR-B | Borehole | 16.50 | **+3.88** (se 2.30, 1/5) vs NIR-B 12.62 | 0.97 | 0.44 |
| **MIXR-P72-B** | Borehole | **7.90** | **−4.72** (se 0.96, **5/5**) vs NIR-B | **0.92** | 0.33 |
| L1-MIXR-B | Borehole | 10.61 | −2.01 (se 1.46, 3/5) vs NIR-B | 1.15 | 0.39 |
| (MIXR-B, h207) | Borehole | 6.22 | −6.40 (se 2.33, 4/5) | 1.28 | 0.27 |

## Scoring

**P-CTRL — SUPPORTED, exactly.** CTRL-K1-H = 5.93; paired difference against h84's
ROI-Q10 is **+0.00 with se 0.00 on 5/5 seeds** — bit-identical, seed for seed. There is
no code drift on Hartmann. **Consequence: the ~3.8-point Hartmann gap is the K=8
configuration itself, and "K=8 ≈ K=1" (h206) is a BOREHOLE-ONLY statement.** Recorded as a
scope correction, not a retraction: h206's Borehole measurements stand.

**P-L1-H — SUPPORTED.** L1-NIR-H − NIR-H = −0.88 (se 1.77, 4/5), inside the band. The
median head costs nothing on an interior optimum.

**P-L1-B — FAILED.** Predicted ~8; measured 16.50, *worse* than the MSE control on 4/5 and
10.28 worse than MIXR. The protocol's registered consequence fires:

> **RETRACTED: the mean-vs-median account of MIXR.** The diagnostic is decisive, not just
> the endpoint — L1-NIR-B's dims-on-face is **0.97**, indistinguishable from MSE's 0.95.
> The median head does not move queries toward the faces at all. The boundary-proximity
> correlation reported in h209 was correlation, exactly as the protocol warned.

**P-P72 — SUPPORTED, and it is the cleanest form of the effect.** With the random half's
HF fraction matched to the MES rollouts', MIXR-P72 still gains **−4.72 on 5/5** (se 0.96 —
tighter than the original MIXR's 2.33). And its dims-on-face is **0.92**, i.e. *no boundary
shift at all*, while keeping most of the gain.

> **RETRACTED: the fidelity-composition account of MIXR** (already corrected in its causal
> form at h210 amendment 1; now refuted outright as the source of the gain).

**P-L1MIXR — FAILED.** 10.61, better than NIR on only 3/5, worse than both plain MIXR and
MIXR-P72. Combined with P-L1-B, L1 is a drag on Borehole that the random half partly
overcomes; it is not the enabling ingredient.

## What survives

**A robust, unexplained, deployable effect.** Adding unselected random-in-ROI rollouts to
the MES training set improves Borehole by ~5–6 points across **15/15 seeds and three
configurations** (MIXR 42–46, MIXR 47–51, MIXR-P72 42–46). It is not data volume (h209
NIR120), not boundary geometry (L1-NIR-B, MIXR-P72 dims-on-face), not fidelity composition
(MIXR-P72). Remaining candidates, none tested: state coverage (random rollouts visit
posteriors MES never produces), variance of the action distribution rather than its mean,
or a Borehole-specific ROI-shape effect.

**A scope correction that changes the baseline for everything.** NIR is *worse* than plain
K=1 on both benchmarks (12.62 vs 11.59; 9.75 vs 5.93). MIXR's gains have only ever been
measured on top of that handicap. The honest reference for any claim is CTRL-K1, not NIR:
against it, MIXR's Borehole gain is 11.59 → 6.22 ≈ 5.4 points.

## Next (proposed, NOT registered — awaiting the user)

h211: MIXR on the **K=1 base** (both benchmarks, 10 runs) — the highest-value untested
cell, since the effect is a data-composition effect and should not depend on the window;
plus **K1 + terminal_improvement on Hartmann** (5 runs) to isolate the label from the
window inside the 3.82-point gap, which is currently attributed to the window on an
assumption.
