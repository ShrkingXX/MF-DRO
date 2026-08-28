# H107 — does q=0.05's advantage survive a second seed set?

LOCKED BEFORE ANY RUN. ID claimed via `tools/claim_id.sh`.

## Why this and not "find the exact optimum"

h97 showed q=0.05 beats the confirmed q=0.10 on Borehole: paired −1.52, 4/5,
clearing a separability bar registered in advance. The tempting next step is to
bracket the turning point further (q=0.025, q=0.03). **That is the wrong next
step.**

Every single-seed-set result this session has produced and then re-tested has
lost something: the ROI's Hartmann flip was WITHDRAWN, the HF floor was WITHDRAWN
with its mechanism inverted, teacher refinement shrank 64% before recovering.
h97 is n=5 on seeds 47-51 only. Locating an optimum on an unreplicated effect
would be building on exactly the kind of result this project has repeatedly had
to retract.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| seeds | 42, 43, 44, 45, 46 — the ORIGINAL set, where q=0.10 scored −4.22 |
| new arm | ROI-Q05 (`roi_target_accept=0.05`), identical to h97's |
| comparators | h84's ROI-Q10 and h83's MF-DRO control at the same seeds |
| runs | 5 |

This gives Q05 **n=10 across two independent seed sets**, and compares it against
Q10 at matched seeds on the set where Q10's effect was first measured.

## Gate

Same G3 as h97, and non-vacuous for the same reason: `accept_frac` is
`_n_acc/_n_seen`, measured over real rejection draws, with `target_accept`
stored separately. **Every run must report accept_frac in [0.045, 0.055]** or the
arm is void regardless of regret. h97's runs came in at 0.0499-0.0500.

## Predictions

**P1.** Q05 beats the no-ROI control at these seeds (negative paired mean, ≥4/5).
Registered **POSITIVE**: q=0.10 managed −4.22 at 5/5 here and q=0.05 was stronger
at 47-51.

**P2 — the actual question.** Q05 still beats Q10 at these seeds, clearing the
same **|paired mean| > 0.59 AND ≥4/5** bar h97 used. Registered as **GENUINELY
UNCERTAIN.** h97's −1.52 is one seed set, and this session has watched a −5.85
become −2.11 and a 4/5 become 2/5 on exactly this kind of re-test. I am not
predicting it holds.

**P3.** Q05 does not make MF-DRO competitive with MF-MES on Borehole. Registered
**POSITIVE** — it has held for every intervention tried.

## What each outcome means

  - **P2 holds:** the tightness effect is real across two seed sets, q=0.10 is
    confirmed suboptimal, and every ROI figure this project reports understates
    the method. *Then* bracketing (0.02, 0.05] is worth the compute.
  - **P2 fails:** h97's ordering was seed-set specific, joining the withdrawn
    list. q=0.10 and q=0.05 would be indistinguishable pooled, and the honest
    claim reverts to "the mechanism is the region, not the threshold".
