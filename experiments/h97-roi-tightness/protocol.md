# H97 — is q=0.10 the right ROI tightness, or did we stop at the first value that worked?

LOCKED BEFORE ANY RUN. Numbering: session B holds h90+; peer session holds h94-h96,
so this takes h97 to avoid the collisions that hit h88/h89.

## Why now

The ROI's Borehole gain is CONFIRMED (h90: P1/P2/P3 met, pooled 9/10, 83%
retention). Every ROI result this project has rests on **q = 0.10**, and that
value was never optimised — it was the first calibrated setting tried. The
project's own state file lists the tightness question as unclaimed and UNLOCATED:

  - h84 found tighter beat looser on Borehole under FIXED beta
  - teacher measurements imply a turning point BELOW q=0.10: at q=0.02 the
    closest approach to x* degrades 0.022 -> 0.110

Those two together bracket an optimum somewhere in (0.02, 0.10) but no experiment
tests it. With the ROI now confirmed to work, "how tight should it be" is the
direct next question for this session's primary aim.

## Design

| | |
|---|---|
| benchmark | Borehole_8D (the only benchmark with a confirmed ROI gain) |
| seeds | 47, 48, 49, 50, 51 |
| new arm | **ROI-Q05** — identical to ROI-Q10 but roi_target_accept=0.05 |
| comparators | ROI-Q10 and NO-ROI at the SAME seeds, already run in h90 |
| runs | 5 (only the new arm; nothing is re-run) |

Reusing h90's two arms is legitimate here in a way it would not normally be: they
are the same code, same seeds, same worker, same commit. The h90 worker is invoked
unmodified with one config value changed. Verified before launch: h90's ROI arm's
own logged accept_frac is 0.0998-0.1000 against its 0.10 target, so the
calibration demonstrably fires — this arm's q=0.05 must show accept_frac ~0.05 in
its own logs or the run is void (see gate).

## Predictions

**P1.** ROI-Q05 beats NO-ROI (negative paired mean, >=4/5). Registered POSITIVE
and expected to be easy: q=0.10 achieved -3.49 at 4/5, and 0.05 is still a
region rather than a point.

**P2 — the actual question.** ROI-Q05 vs ROI-Q10 is registered as **GENUINELY
UNCERTAIN, no direction predicted.** The two pieces of evidence point opposite
ways: h84's fixed-beta result says tighter is better, the teacher measurement at
q=0.02 says too tight destroys reach. q=0.05 sits between them and I have no
basis for calling which side of the turning point it lands on. Six mechanism
predictions in this session were refuted; declining to guess is the honest
posture, not a hedge.

**P3.** ROI-Q05 does not make MF-DRO competitive with MF-MES on Borehole.
Registered POSITIVE. Every intervention so far has failed to close that gap and
nothing about tightening a region addresses boundary aversion.

## Gate (G3, adopted from a peer session's failure this evening)

A bit-identity gate on the OFF path is structurally incapable of catching a
broken ON path. Before reading any regret number, this arm's `roi_summary` must
show **accept_frac in [0.045, 0.055]** — the side effect OBSERVED, not merely
the absence of a crash. If it shows ~0.10 the flag did not take and the runs are
void regardless of what the regret says.

## What each outcome means

  - **Q05 beats Q10:** the optimum is below 0.10 and every ROI number this project
    reports is from an unoptimised setting. Worth locating properly.
  - **Q10 beats Q05:** 0.10 is at or near a local optimum from below, and the
    turning point the teacher measurement implies lies between 0.02 and 0.05.
  - **Indistinguishable:** the gain is robust to tightness over 2x, which is a
    more useful claim than a tuned optimum — it says the mechanism is the region,
    not the threshold.

---

## Gate-validity check, done while the runs were in flight

A gate that reads back its own input proves nothing, so I verified that
`accept_frac` is a measurement and not an echo of the configured target before
trusting it as G3's criterion.

In `src/policy/mf_dro.py` the recorded value is

    _acc = (_n_acc / _n_seen) if _n_seen else 0.0

— the empirical acceptance rate over actual rejection-sampling draws. The
configured value is stored *separately* in the same record, as `target_accept`.
They are distinct fields, so a mis-applied config cannot make the two agree by
construction.

**Consequence: the gate does what it claims.** If the ARMS entry failed to apply,
`cfg.roi_target_accept` would fall back to its 0.10 default, the measured
acceptance would come out near 0.10, and the gate would reject the runs — which
is exactly the failure mode it was written for.

Confirmed empirically on h90's already-complete ROI arm, where the same field
reads 0.0998-0.1000 against a 0.10 target: the measurement tracks the target to
within 0.0002 when the config does apply.

**Known limit.** `roi_summary` aggregates `accept_frac` but not `target_accept`,
so the completed files support "the measured rate is 0.05" and not the stronger
"the measured rate equals the recorded target". The former is sufficient here
because the target is fixed by the arm definition and verified in the shim, but
carrying `target_accept` into the summary would be the better design and is worth
doing in any future arm.

---

## ADDENDUM — the separability bar, fixed before h97 results exist

Registered with 0/5 h97 runs on disk. Reason: two tightness levels are ALREADY on
record, and if they cannot be told apart, h97's P2 cannot be read as an ordering
either. Establishing that bar after seeing h97 would let me pick it to suit.

Recomputed from h84's Borehole arms, seeds 42-46:

      seed   FIX2 (acc 21.4%)   Q10 (acc 10.0%)   FIX2 - Q10
        42        11.13              11.50          -0.37
        43        10.93              12.27          -1.35
        44         8.26              11.37          -3.10
        45        11.99              11.19          +0.80
        46        12.70              11.62          +1.07
      paired mean -0.589, sd 1.707, FIX2 better 3/5, |mean|/sd = 0.35

**A 2.1x change in acceptance rate (21.4% -> 10.0%) moves regret by 0.59 points
against a per-seed sd of 1.71, and splits 3/2.** Those two levels are not
separable. A peer session reached the same conclusion from the mechanism side
(GAPSD paired -0.018 against sd 0.169) and recorded that q=0.10 was never shown
to beat q~0.21.

### The bar

**h97's q=0.05 vs q=0.10 difference is declared an ORDERING only if it exceeds
what already failed to separate: |paired mean| > 0.59 AND at least 4/5 in one
direction.** Anything smaller is recorded as INDISTINGUISHABLE, whatever its sign.

This is deliberately a bar the existing data would fail. That is the point: an
ordering claim has to beat the noise level that two settings 2.1x apart could not.

### What the likely outcome means

Three tightness levels spanning 21.4% -> 10.0% -> 5.0%, a **4.3x range**, all
mutually indistinguishable, would be a stronger and more useful result than a
tuned optimum: **the mechanism is the region, not the threshold.** It would also
retire the tightness question rather than leaving it open, and it says the
calibration's value is controllability across benchmarks — bounding rejection
cost and collapsing 12.6-100% acceptance to a single dial — not superiority at
any particular setting on Borehole.

If instead q=0.05 clears the bar in EITHER direction, that is the first evidence
of a real tightness effect and locates a turning point, which is worth following.
