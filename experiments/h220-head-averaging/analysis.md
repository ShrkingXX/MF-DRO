# h220 analysis — CONFIRMATORY. 15/15 finals, 0 tracebacks. P-HEAD FAILED: the median head averages too.

## Primary readout — |x − x\*|∞ of the emitted query, first 10 iterations

| arm | mixture | loss | \|x − x\*\| | per seed |
|---|---|---|---|---|
| CTRL-K1 (reference) | no oracle | mse | 0.67 | — |
| h207 MIXO | 20 MES + 20 oracle | mse | 0.18 | the 50/50 midpoint |
| **MIXO-MSE** | 20 MES + 10 oracle | mse | **0.390** | 0.35 0.63 0.37 0.25 0.35 |
| **MIXO-L1** | 20 MES + 10 oracle | **l1** | **0.417** | 0.32 0.68 0.44 0.24 0.40 |

Registered P-HEAD: MIXO-L1 ≥ 0.45 ⇒ the median tracks the MES majority. **Observed 0.417.
FAILED.** L1 moves the emitted query 0.027 toward the MES mode relative to MSE — inside
seed-to-seed noise.

Two facts the numbers establish together:

1. **The MSE head is a mixture-weighted mean.** At 50/50 it emits 0.18; at 2:1 it emits
   0.39, exactly where a proportion-weighted average of the two modes lands. The head is
   not selecting on the return; it is averaging the action targets.
2. **The median does not select either.** With MES at two-thirds of the batch, an L1 head
   that picked the majority mode would emit ~0.67. It emits 0.42. Both losses produce a
   blend. The registered retraction fires: *"the problem is not the loss function, and the
   next suspect is the conditioning — the head has no mechanism to make its output depend
   on RTG at all."*

The inference target is the batch max, which under `mes_entropy` labels the MES half as
the better one (Stage 0: MES rtg[0] +0.03 to +0.13, oracle −0.11 to −0.25). A head that
conditioned on the target would therefore emit the MES action. Neither head does.

## Secondary — regret

| arm | regret | vs reference |
|---|---|---|
| MIXR-L1 | **6.43** | −0.74 vs MIXR-K1 7.17 (inside band); ties MF-MES 6.40 |
| MIXO-L1 | 10.26 | ceiling arm, reported not scored |
| MIXO-MSE | 11.97 | ceiling arm, reported not scored |

MIXR-L1 at 6.43 is the closest any deployable configuration has come to MF-MES on Borehole.
Inside the band against MIXR-K1, so not claimed as an improvement.

## With h221, the return channel is now characterised at both ends

| end | measurement | result |
|---|---|---|
| **label** | adj R² of rtg[0] on the first action (h221) | ≈ 0, and CRN does not raise it |
| **head** | emitted query under a bimodal target, MSE vs L1 (h220) | averages under both losses |

Neither is a plumbing defect. The label does not encode what a selecting head would need,
and the head would not select on it if it did. That is the mechanism behind every null on
the RTG thread — h180, h198 (once its confound was removed), the label swaps, the target
schema — and behind why added diversity (random half, oracle half, multi-teacher) gets
averaged rather than exploited.
