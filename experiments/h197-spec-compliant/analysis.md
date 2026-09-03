# h197 — **P3. The full specification does not beat no-window.**

**CONFIRMATORY**, 5/5 finals, readout committed before any run completed. Quality is
compared ONLY by **final simple regret** (frozen rel% of |optimum| @ cost 200), per the
standing instruction of 2026-09-03.

## Result

| arm | final regret |
|---|---|
| h194 CTRL-K1 (no window) | **11.59** |
| h196 WINDOW (K=8, real actions) | 13.96 |
| **h197 SPEC** (K=8, L1, real-query IG labels) | **14.55** |

- **h197 − CTRL-K1: +2.96** (se 1.17), better on **1/5**. Threshold 1.26 → **P3**.
- h197 − h196: **+0.59** (se 1.71), better on 1/5 → **indistinguishable**.

Per-seed h197: 13.38, 17.19, 13.44, 17.45, 11.29.

## What this settles

The specification was implemented faithfully — all six SCs passed before launch, and the
real-query `b` was fixed to use the ROI-filtered 600-point pool so its information-gain
labels are computed exactly as training computes them (b falls −27.9% over a run, not
the +2.7% noise the wrong pool produced). **It still does not help.**

More sharply: **the spec's ingredients beyond h196's action-feeding fix bought nothing.**
L1 location loss, real-query information-gain RTG labels, and the window-relative
timestep together move the endpoint by +0.59 ± 1.71. Whatever is wrong is not addressed
by labelling the history correctly.

The early curve was actively misleading and would have been reported as encouraging
under the old habit: h197 leads CTRL-K1 at c=25 (−5.70, 4/5) and c=50 (−6.79, 4/5) and
**finishes worse**. This is why quality is now endpoint-only.

## The finding underneath the null — the window collapses the FIDELITY MIX

| arm | LF fraction |
|---|---|
| h194 CTRL-K1 | **0.261** |
| h196 WINDOW | 0.085 |
| h197 SPEC | 0.092 |

Both window arms spend **~3x less on low fidelity** than the control. On Borehole's 2:1
cost ratio that is materially fewer total queries for the same budget.

**So the window's remaining deficit is not established to be about history at all.** It
is equally consistent with "the window shifts spend toward expensive HF and buys fewer
queries". These two accounts predict the same endpoint and h197 cannot separate them —
the same structure as h60, where a Thompson teacher collapsed fidelity to 99% LF and
confounded every teacher comparison, and as h145's quality/diversity confound.

**This is a confound in h194/h196/h197 alike**, and it means the sliding-window null —
including h196's own P3 — is NOT safe to state as "history does not help". Registered as
h200: hold the fidelity mix fixed (the `max_hf_fraction` knob from h184 already exists)
and re-run the window arm. If the deficit vanishes, the window question is still open
and three arms need re-reading.

## What this RETRACTS

Nothing previously claimed — h196's P3 was stated as "the window still hurts", which
remains true as a measurement. What is now withdrawn is any **interpretation** of that
P3 as evidence about history or context length, mine included. The mechanism-level claim
("a constant does not depend on its inputs") is untouched by this, since it rests on
h185/h186/h192, not on the window arms.
