# h202 — the window collapses the fidelity mix by SATURATING THE FIDELITY HEAD

**EXPLORATORY** (no new runs — measured from logs already on disk), but the key contrast
is **causal**: h194's two arms differ ONLY by `inference_context_k`.

## The measurement

> **CORRECTION (same session, before this was used for anything).** The first version of
> this file called `fid_mean` "the TEACHER's HF fraction (the target)". **It is not.**
> `decisionTransformer.py:581` computes `fid_mean = p_pred.detach().mean()` -- the DT's
> OWN mean prediction across all timesteps. The conclusion below survives, but the
> evidence for it had to be replaced with a direct measurement of the teacher's
> per-tau fidelity (1620 rollouts), which is what the table now reports.

Last third of each run, 5 seeds per arm. `fid_mean` is the DT's mean predicted p over all
timesteps; `p_pred_inference` is its prediction at the readout position.

| arm | teacher `fid_mean` | DT `p_pred` | gap | iters with p_pred>0.95 |
|---|---|---|---|---|
| **CTRL-K1** (no window) | 0.542 | **0.600** | +0.057 | 0.248 |
| h194 WINDOW-K8 | 0.519 | **0.996** | **+0.477** | 0.705 |
| h196 WINDOW-K8 (real actions) | 0.511 | **0.991** | +0.480 | 0.680 |
| h197 SPEC K=8 | 0.515 | **0.992** | +0.477 | 0.684 |

**The training target is the same in every arm (0.511-0.542).** What changes is the head.
Without a window the DT tracks its target within 0.06; with one it pins at ~0.99.

## The teacher's ACTUAL per-tau fidelity (1620 rollouts, direct measurement)

| tau | 0 | 1 | 2 | 3 | 4 | 5 | 6 | **7** |
|---|---|---|---|---|---|---|---|---|
| teacher HF frac | **0.909** | 0.696 | 0.726 | 0.736 | 0.715 | 0.741 | 0.698 | **0.731** |

So the window moves the readout to a position whose target uses **LESS** HF (0.731) than
tau=0 (0.909) -- yet the DT emits **0.99** there. Both readout positions are
miscalibrated, in OPPOSITE directions:

| readout | teacher target | DT predicts | error |
|---|---|---|---|
| K=1, position 0 | 0.909 | 0.600 | **under** by 0.31 |
| K=8, position 7 | 0.731 | **0.99** | **over** by 0.26 |

This is what licenses the word "saturation": the head is not tracking a high-HF target,
it is ignoring a LOWER one.

## Two hypotheses, both tested against this table

1. **"The teacher's fidelity at tau=7 is higher, so 0.99 is faithful."** REFUTED by the
   direct measurement above: tau=7 is 0.731, LOWER than tau=0's 0.909. The DT is not
   reproducing its target; it is over-shooting it.
2. **"The window feeds back the DT's own recent HF choices via the `ae` slots."**
   REFUTED. h194's window carried **zeroed** action tokens -- its history said LOW
   fidelity everywhere -- and it saturates at 0.996, indistinguishable from h196's and
   h197's real-action windows (0.991/0.992).

**What survives: the saturation tracks sequence LENGTH, not sequence CONTENT.** A
length-8 token sequence drives the fidelity logit to an extreme regardless of what
occupies the slots.

## Why it matters

It supplies a MECHANISM for the confound h197 uncovered:

    window -> fidelity head saturates -> LF collapses 0.261 -> 0.09
           -> fewer queries per unit budget -> worse endpoint

If that path dominates, then h194/h196/h197's harm is not about history or context at
all; it is the fidelity head going out of distribution on sequence length. That is
exactly what h200 was designed to isolate (h200 is HALTED on human instruction, so the
confound remains OPEN).

## Caveat this places on h201

h201 uses the same K=8 window, so the same saturation should apply. If h201's DT emits
~x* at position 7, that is the LOCATION head behaving as the mechanism predicts -- but
its fidelity head will very likely still be pinned near 1.0, and h201's endpoint will
carry the same LF-collapse penalty. The location result would still be interpretable;
the endpoint would not be a clean test of "does the window help".

## What this does NOT establish

That the saturation CAUSES the endpoint harm. The causal chain above is a hypothesis
with a measured first link (window -> saturation) and a measured correlate (LF collapse).
Holding the mix fixed -- h200's design -- is what would close it.

## Raw measurement
```
  teacher HF fraction BY TAU  (1620 rollouts of length 8)
==============================================================
    tau=0:  HF frac = 0.909
    tau=1:  HF frac = 0.696
    tau=2:  HF frac = 0.726
    tau=3:  HF frac = 0.736
    tau=4:  HF frac = 0.715
    tau=5:  HF frac = 0.741
    tau=6:  HF frac = 0.698
    tau=7:  HF frac = 0.731

    aggregate over all tau = 0.744   <- this is `fid_mean`
    tau=0  = 0.909      (what the K=1 readout targets)
    tau=7  = 0.731      (what the K=8 readout targets)
```
