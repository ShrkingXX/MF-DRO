# H130 — Does the ROI's QUALITY effect generalise, when its fidelity effect does not?

STATUS: LOCKED before any statistic was computed.
TYPE: CONFIRMATORY.
COMPUTE: zero new runs.
READ POINT / STATISTIC: count-matched mean non-init HF y, per seed, K = min HF
count across the two arms within that seed. Named per today's convention.

## Why this is the right next test

Two mechanisms have been measured for the ROI's effect. Today they diverged:

  FIDELITY  post-init HF count fraction. Borehole 1.65; Hartmann 0.78, Ackley
            0.49 (censored), Currin 0.41. **Does not generalise** (h120 scope
            correction, after the peer's h129 P6).
  QUALITY   h120 P3, count-matched mean HF y: +17.05, effect 3.54, 5/5 on
            Borehole. Peer's h129 P4 agrees from a different statistic (2.66).
            **Generality never tested.**

Quality is also the channel the founding diagnosis actually named ("mean HF
query score 0.336 vs MF-MES 0.747 on Hartmann"). So its generality is the single
most consequential untested cell: if quality generalises it is the first ROI
effect that does, and if it does not then nothing about the ROI does.

## Measure

Per benchmark x seed (42-46), ROI-Q10 vs h83 `MF-DRO` control, paired:
  K      = min(HF count control, HF count ROI) within that seed
  q_arm  = mean y over the FIRST K non-init HF queries of that arm
  delta  = q_ROI - q_control   (higher is better; all four objectives are
           maximisation in the stored traces)

Count-matching is required, not optional: h119's uncorrected version was
confounded by the ROI making ~9% fewer HF queries in a run that also converged
earlier, and h120's P3 exists precisely to remove it. The uncount-matched value
is reported alongside, labelled, so the confound's size stays visible.

Sources (all seeds 42-46): control h83 `MF-DRO`; ROI arms h84 `ROI-Q10`
(Borehole, Hartmann) and h86 `ROI-Q10` (Ackley, Currin). Control substitution
verified bit-identical at 5/5 Borehole seeds (h120 Amendment 3, discharged).

## Prediction (locked)

P1. Borehole reproduces: delta > 0 with |mean|/sd >= 1.0 in >= 4/5 seeds.
    (A re-measure of h120 P3 under the same statistic; a failure here would mean
    I have mis-specified the measure, not that P3 was wrong.)

P2 (PRIMARY). **Quality does NOT separate on Hartmann, Currin or Ackley** —
    each below |mean|/sd 1.0.

Grounds: every ROI effect measured on more than one benchmark has been
Borehole-specific — the regret benefit (h111, h121), the fidelity mix (h129 P6,
verified), the boundary mass (h118). Four benchmarks, several measures, one
positive cell each time.

P3. If quality DOES separate on any of the other three, that is the first
    generalising ROI effect found in this project and must be reported as
    prominently as the null, with fresh-seed confirmation required before it is
    believed. Three benchmarks are tested, so one clearing 1.0 is weak.

## Caveat on Ackley, carried forward

Ackley's control is one-sided censored on the FIDELITY measure (40 HF x c_H=5.0
= the entire budget, zero variance). That censoring does NOT apply to quality,
which is a value statistic and free to move. Ackley is therefore included here
on equal terms, and the reason it was excluded elsewhere is recorded so the
inconsistency is deliberate rather than sloppy.

## Limitations

- n=5 per benchmark, no p-values.
- Three benchmarks tested against one prediction; multiplicity handled by P3's
  requirement rather than by correction.
- Mean HF y is not the diagnosis's own normalised "score"; that normalisation is
  not recoverable from the record (h121). This measures the same construct on a
  stated scale rather than reproducing an unstated one.
