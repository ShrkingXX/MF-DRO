# h220 — Does the LOSS decide whether the head averages? MIXO as the probe.

Registered BEFORE any h220 run. CONFIRMATORY. No new code: `loc_loss='l1'` is H102's,
already gated; the oracle half is h207's `worker_MIXO` machinery.

## The blocker this targets

MIXO put an **oracle half** and a **MES half** at an *identical* τ=0 state with labels ~40
units apart — the cleanest selection problem this project can construct. The DT did not
select. It emitted a point **0.18 from x\*** *regardless of RTG* (the control emits 0.67),
i.e. it took the **midpoint** of the two modes. Under MSE that is exactly what a location
head must do: it fits the conditional **mean** of a bimodal target.

That is the standing explanation for why every diversity intervention gets washed out —
multi-teacher data (h180: coherent acquisitions are 0.044 apart, so their mean is the same
point), the random half, the oracle half. **Until the head stops averaging, added diversity
cannot be exploited.**

`loc_loss='l1'` fits the **median** instead. With an *unbalanced* mixture the median is the
majority mode rather than the midpoint — so an L1 head on 20 MES + 10 oracle should emit
**MES's action**, far from x\*, where the MSE head emits the midpoint. That is a sharp,
falsifiable difference and it needs no new architecture.

h210 tested L1 only on the K=8 base that has since been retracted as a valid comparison
(and its MIXR arm was confounded with the window). It has never been run on CTRL-K1.

## Arms — 15 runs, Borehole 42–46, CTRL-K1 base (K=1, `mes_entropy`, ROI-Q10)

| arm | mixture | loss | role |
|---|---|---|---|
| **MIXO-MSE** | 20 MES + 10 oracle | mse | matched control — the averaging baseline |
| **MIXO-L1** | 20 MES + 10 oracle | **l1** | the probe |
| **MIXR-L1** | 20 MES + 20 random | **l1** | does L1 help the *deployable* arm on the current base? |

Oracle half is CEILING/DIAGNOSTIC (uses x\* and true f); MIXR-L1 is deployable.
References at the same seeds: CTRL-K1 **11.59**, MIXR-K1 **7.17**.

## The primary readout is NOT regret

**|x − x\*|∞ of the emitted query**, from the H168 probe, averaged over the first 10
iterations before the GP has learned x\* on its own:

| | measured |
|---|---|
| CTRL-K1 (no oracle half) | **0.67** |
| MIXO under MSE (h207) | **0.18** ← the midpoint |
| MIXO-L1 | **?** |

- **P-HEAD (the point of the arm):** MIXO-L1's |x − x\*| ≥ 0.45, i.e. the median head tracks
  the MES majority instead of the midpoint. **Then the loss is the blocker**, a
  mode-selecting head is worth building, and multi-teacher data (including the user's
  multi-acquisition proposal) becomes worth generating.
- **P-HEAD-FAIL:** MIXO-L1 stays near 0.18. Then even the median averages here, the problem
  is not the loss function, and the next suspect is the *conditioning* — the head has no
  mechanism to make its output depend on RTG at all, which is an architecture change
  (gating / mixture density), not a loss change.

## Secondary predictions (regret, band ±1.26)

- **P-MIXR-L1:** |MIXR-L1 − MIXR-K1 (7.17)| ≤ 1.26. Lean: L1 neither helps nor hurts the
  deployable arm on this base. h210 measured L1 as a **drag** on the K=8 base (+3.88 alone),
  so a large regression here would say that drag is not window-specific.
- MIXO-MSE is a ceiling and its regret is reported, not scored.

## What each outcome RETRACTS

- **P-HEAD supported** → retracts nothing, but converts "the DT cannot select" from an
  architectural dead end into a fixable loss choice, and reopens the multi-teacher and
  oracle-contrast lines that were closed on the averaging result.
- **P-HEAD fails** → strengthens the averaging account to "not repairable by the loss", and
  the next arm is a gated or mixture-density head, which is a real build.

## Stage 0 (smoke)

(a) all three arms build and run at BUDGET=30; (b) `mf.dt.loc_loss` is `'l1'` on the two L1
arms and `'mse'` on the control; (c) the oracle half is present at the right count and its
τ=0 actions are within 0.1 of x\* in unit coords (h207's SC9 check). A silent default is
the failure mode.

## Compute

15 runs, cap 15, queued. 1 thread/worker.

## Evaluation

Primary: |x − x\*| from the probe (a mechanism readout, not the frozen metric).
Secondary: frozen final simple regret, rel% @ cost 200, imported from h83's `grid`.
Finals only. Every run reported.
