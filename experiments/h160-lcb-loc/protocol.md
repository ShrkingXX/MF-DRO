# h160 -- LCB-LOC: model-selected but ANTI-informative

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The ambiguity this targets

h153 completed the 2x2 and left the mechanism open. The post-hoc hypothesis in
findings.md is that the working arms all query **model-selected** locations while
the failing ones use criteria external to the model.

The h153-vs-RANDOM-POOL contrast already supports that more than "post-hoc"
suggests: both are state-blind at query time (one frozen, one uniformly
re-drawn from the SAME pool), and they differ mainly in whether the model chose
the points. 19.36 / 5-of-5 against 43.94 / 0-of-5.

**But "model-selected" and "informative" are still confounded.** h153's frozen
path is model-selected AND sits on high-acquisition points. Every working arm
does. No arm has ever been model-selected and deliberately UNinformative.

h159 was locked to break the loop-type/information correlation with beta=0, and
its blind forecast (91.5%) showed it does NOT: a mean-greedy teacher keeps
earning information because conditioning moves the mean. That was recorded
before h159 produced anything. This is the arm that actually breaks it.

## The arm

`ucb_loc_beta = -2.0`. The location becomes argmax of (mu - 2*sigma) -- the
LOWER confidence bound. It selects points that are **confidently mediocre**:
low predicted value AND low uncertainty, i.e. exactly where the model expects to
learn least. Closed-loop, adaptive, re-decided every step from the current
fantasy-conditioned model, model-selected in the strictest sense -- and
anti-informative by construction.

Implementation is one number in h159's existing kwarg wrapper. `src/policy/mf_dro.py`
is NOT modified, so h155's bit-identity gate stands unchanged.

## Predictions (opposed)

INFORMATION account: LCB-LOC **FAILS**, ~43.94, improves ~0/5.
MODEL-SELECTED account: LCB-LOC **WORKS**, ~15-20, improves ~5/5.

## Gate before spending the arm

A blind harness forecast (condition C8, beta=-2.0) is run FIRST. **If the
harness puts C8 near the control (>70% of its tail), this arm does not
discriminate either** -- the same failure mode as h159 -- and the pipeline arm
will NOT be launched. Spending ~10 worker-hours on a non-discriminating arm
twice in a row would be the definition of running arms.

NOTE the harness's own limits, established the hard way by h153: its C2
condition was off by 2.7x and was **the only condition never validated against a
finished arm**. C8 is likewise unvalidated. Its forecast is therefore used ONLY
as a go/no-go screen on whether the arm discriminates, NOT as a prediction of
the outcome, and that distinction is recorded before it is run.

## Named confound, checked before any number is read

Realised HF fraction against the control's 0.88. A teacher steering to
low-uncertainty points may collapse the fidelity mix (h60's `thompson` went to
2/196 HF). If it does, the arm is CONFOUNDED and no verdict is issued.

## What this can RETRACT

R1 LCB-LOC FAILS -> the "model-selected locations" hypothesis is REFUTED as
   stated. Being chosen by the model is not sufficient; the points must be
   informative. findings.md's post-hoc paragraph must be rewritten.
R2 LCB-LOC WORKS -> the information-seeking reading is refuted instead, and
   model-reference is the operative property. This would also mean h158's flat
   dose and the whole tail line are describing a correlate, not a cause -- a
   second such demotion in two ticks.
R3 Intermediate (25-35 rel%) -> inconclusive at n=5, reported as such.

## Design

Borehole_8D seeds 42-46, n=5, frozen metric, no p-values. Launched only if the
screen says it discriminates.
