# h171 — **R3 fires.** Both forecast halves hold, on both arms.

CONFIRMATORY, n=5 each. Forecast committed with both arms at 0/5.

| | forecast | HEAD-MES | TAIL-MES |
|---|---|---|---|
| **F1** query centroid from box centre | HEAD > 0.5, TAIL < 0.2 | **0.7397** | **0.0313** |
| **F2** rel% | HEAD ~15.82, TAIL ~43.94 | **16.96** | **43.94** |
| **F2** improves | HEAD ~5/5, TAIL ~0/5 | **5/5** | **0/5** |

Reference: control 0.7604 / 15.82 / 5-of-5; RANDOM-POOL 0.0239 / 43.94 / 0-of-5.

**The teacher that follows the acquisition on seven of eight steps fails
completely. The one that follows it on a single step works.** F3 holds: the
ordering is inverted relative to teacher quality, which no trajectory-quality
account predicts and this one requires.

Neither R1 (HEAD fails) nor R2 (TAIL works) — the two individually fatal
outcomes — occurred.

## SC1 FIRES on TAIL, and the attribution has to be handled carefully

TAIL's realised HF fraction is **0.217** against the control's 0.883. SC1 was
registered as *"a collapse voids the arm"*, and by the letter it does.
**TAIL alone therefore cannot attribute its failure to the τ=0 location rule** —
its fidelity mix collapsed too, and the two are not separable within this arm.

**ORACLE and DIVERSE-GOOD separate them.** Their HF fractions are **0.626 and
0.604** — no collapse — and they still fail at 43.94 with centroids of 0.0394 and
0.0409. So a τ=0 location drawn independently of the model is **sufficient to
fail with the fidelity mix intact**. The attribution rests on those arms, not on
TAIL.

## The fidelity collapse is itself a second confirmation

The same mechanism applies to the fidelity head, which is also emitted at τ=0:

| τ=0 teacher fidelity rule | arms | realised HF fraction |
|---|---|---|
| MES's own choice / info-gain at a point | control, h155, h159, h153, h161, HEAD, ORACLE, DIVERSE | 0.60–0.90 |
| **uniform 25% HF** | **TAIL, RANDOM-POOL** | **0.217, 0.256** |

Both arms whose τ=0 fidelity is the 25%-HF coin flip land at 0.22–0.26. The
mechanism predicted the location head; the fidelity head obeys it too, on a
quantity nobody was aiming at.

## The compute consequence, now contention-matched

HEAD and TAIL ran **concurrently on the same machine under the same load**, which
is the comparison the earlier cross-run figure could not support:

| arm | MES calls per rollout | wall | min/query |
|---|---|---|---|
| HEAD-MES | 1 | 39.6 min | 0.360 |
| TAIL-MES | 7 | 72.2 min | 0.439 |

**TAIL takes 1.82× HEAD's wall time.** Against the control, HEAD costs **+1.14
rel%** (16.96 vs 15.82) while consulting the acquisition once per rollout instead
of eight times.

## What this establishes, and what it does not

Established: the teacher's **first step** is what reaches inference. Seven of
eight rollout steps do not affect the real query's location, its improvement
rate, or its regret — they affect only cost.

Not established: that the τ=0 *conditional-mean* story is the full mechanism.
h170's residual was ≈5 SE, and this arm does not address it. What h171 shows is
that the first step is the operative one, by intervention rather than
correlation — which is the claim five previous accounts could not make.
