# h202 — the window collapses the fidelity mix by SATURATING THE FIDELITY HEAD

**EXPLORATORY** (no new runs — measured from logs already on disk), but the key contrast
is **causal**: h194's two arms differ ONLY by `inference_context_k`.

## The measurement

Last third of each run, 5 seeds per arm. `fid_mean` is the TEACHER's HF fraction in the
training rollouts (the target); `p_pred_inference` is what the DT actually predicts.

| arm | teacher `fid_mean` | DT `p_pred` | gap | iters with p_pred>0.95 |
|---|---|---|---|---|
| **CTRL-K1** (no window) | 0.542 | **0.600** | +0.057 | 0.248 |
| h194 WINDOW-K8 | 0.519 | **0.996** | **+0.477** | 0.705 |
| h196 WINDOW-K8 (real actions) | 0.511 | **0.991** | +0.480 | 0.680 |
| h197 SPEC K=8 | 0.515 | **0.992** | +0.477 | 0.684 |

**The training target is the same in every arm (0.511-0.542).** What changes is the head.
Without a window the DT tracks its target within 0.06; with one it pins at ~0.99.

## Two hypotheses, both tested against this table

1. **"The teacher's fidelity at tau=7 differs from tau=0."** REFUTED. `fid_mean` is
   essentially identical across arms, so the training target did not move.
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
