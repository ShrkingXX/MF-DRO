# H111 — is "the ROI only works on Borehole" a fact about the ROI, or about q=0.10?

LOCKED BEFORE ANY RUN. ID claimed via `tools/claim_id.sh`.

## Why this is the right next experiment

The single most-repeated limitation in this investigation is that every ROI
result is Borehole-only. A peer session has flagged it three times, most recently
as "the mechanism remains unidentifiable from four benchmarks rather than merely
unknown".

**But that conclusion rests entirely on q=0.10**, and h97+h107 have now shown
q=0.10 is suboptimal on two independent seed sets (q=0.05 beats it by −1.54
pooled, 8/10, clearing both the 0.59 separability bar and the 0.5-sd effect-size
bar). Every non-Borehole ROI arm in the tree is q=0.10:

      Hartmann_6D   ROI-Q10 (also FIX2, ANN)   −1.62 pts, 3/5
      Ackley_10D    ROI-Q10                    −0.09 pts, 1/5
      Currin_2D     ROI-Q10                    +0.11 pts, 0/5

Hartmann's −1.62 is the interesting one: the **magnitude clears** the 0.59 bar
while the **split does not** reach 4/5. That is precisely the shape of an effect
too weak to resolve at n=5 — not the shape of no effect. Retesting it at a
setting known to be stronger is the obvious move, and nobody has made it.

## Design

| | |
|---|---|
| benchmarks | Hartmann_6D and Ackley_10D |
| seeds | 42, 43, 44, 45, 46 |
| arm | ROI-Q05 (`roi_target_accept=0.05`), identical to h97/h107's |
| comparators | h83's MF-DRO control and h84's ROI-Q10, both at the same seeds |
| runs | 10 |

Currin is excluded deliberately: h93 established both methods solve it (gap
0.0155 absolute on an optimum of 13.80, four of five seeds at exactly zero), so
no setting can produce a resolvable difference there.

## Gate

Same G3, non-vacuous: **every run must report accept_frac in [0.045, 0.055]**.
h97 and h107 came in at 0.0499–0.0500 across ten runs.

## Predictions

**P1 (Hartmann).** Registered **GENUINELY UNCERTAIN.** The q=0.10 result has a
clearing magnitude and a failing split, which is exactly ambiguous. I have been
wrong on nine mechanism predictions here; the honest position is that I do not
know which way this resolves.

**P2 (Ackley).** ROI-Q05 does not produce a separable improvement. Registered
**POSITIVE**: q=0.10 gave −0.09 at 1/5, which is not a weak effect but an absent
one, and halving the acceptance rate is not a plausible route from absent to
separable.

**P3.** Neither benchmark becomes competitive with its best baseline. Registered
**POSITIVE**; true of every intervention tried on any benchmark.

## What each outcome means

  - **Hartmann separates:** "the ROI is Borehole-only" is **false**, and it was an
    artefact of testing at an unoptimised setting. That would materially change
    the investigation's central limitation and make the mechanism question
    tractable — two benchmarks constrain it far more than one.
  - **Hartmann does not separate:** Borehole-only survives a real test rather than
    resting on an untested assumption, which is worth more than the current
    situation even though the headline does not move.

---

## PREMISE QUALIFIED after launch — recorded, not quietly dropped

This protocol argued that retesting the non-Borehole benchmarks at q=0.05 was
worthwhile because q=0.05 is "a setting known to be stronger", citing h97+h107
(−1.52 and −1.57 against q=0.10, both 4/5).

**That premise is now weaker than when written.** A peer session's h110 ran the
same comparison at seeds 52-56 and got **+0.30, 2/5** — the opposite direction.
Pooled over fifteen seeds the advantage is −0.93 at 10/15, which fails the
separability bar registered before any of those runs. The claim "q=0.05 beats
q=0.10" is WITHDRAWN.

**What this does and does not do to h111.**

  - The *motivation* is weakened: q=0.05 is not established as stronger, so
    "retest at a stronger setting" is no longer the right description.
  - The *question* is unaffected and still worth answering. Every non-Borehole ROI
    arm in the tree is q=0.10, and whether a different tightness changes the
    Borehole-only picture is a fact about the method that no experiment had
    tested. h111 tests it either way.
  - The *bars are untouched*. They were registered before any h111 run and are
    not adjusted in light of this.

The honest description of h111 is now: **a second tightness setting, not a better
one.** Which makes a null result here more informative than it would have been —
if neither setting works off Borehole, that is two settings failing, not one
setting failing at the wrong value.
