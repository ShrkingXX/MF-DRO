# h150 — If the DT distils MES, do its QUERIES look like MES's?

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.

## Why this and not the POOL dose

h146 and h149 showed the outcome is **flat** in both quality and diversity: a
perfect teacher, a good-and-diverse teacher and a uniformly random teacher all
return 43.94 rel%, +28.13, effect 4.49, 0/5. **A dose tracing quality against
diversity would return 43.94 at every point.** Running it would be motion, not
progress.

The account those experiments produced — *the DT inherits its teacher's quality as
a policy* — makes a sharper claim that has never been checked: **MF-DRO should
behave like a lossy copy of MES.** If it does, "policy distillation" is a
description of what the method IS, not just an explanation of why the oracle
failed.

## The test

Borehole, seeds 42-46, h83 runs (MF-DRO, MF-MES, MF-GP-UCB all present at the same
seeds). Post-init queries only, normalised to the unit cube.

For each MF-DRO query, the distance to its **nearest** query in the comparison
arm's cloud, averaged. Then the asymmetry:

    A = mean_nn_distance(MF-DRO -> MF-GP-UCB) - mean_nn_distance(MF-DRO -> MF-MES)

**A > 0** means MF-DRO's queries sit closer to MES's than to a different
acquisition's — i.e. it is imitating MES specifically, not merely "doing Bayesian
optimisation".

## Predictions (locked)

**P1.** `A > 0` on at least 4 of 5 seeds. FALSIFIED if `A <= 0` on 2 or more.
Deliberately a **sign test with no magnitude threshold**: no prior calibration
exists for this statistic, and inventing one would be the bar-design failure this
project has made three times.

**P2 (no direction).** MF-DRO -> MF-MES distance versus MF-MES -> MF-MES's own
internal spread. Reported; it says whether "close to MES" means close relative to
how spread MES's own queries are, which a raw distance cannot.

## Confound, stated before looking

Both arms optimise the same benchmark, so both concentrate near good regions
regardless of any imitation. **The asymmetry against MF-GP-UCB is what controls
for that** — GP-UCB is also drawn to good regions, so a positive `A` cannot be
explained by "both find the optimum".

**What it cannot rule out:** MES and MF-DRO may share machinery beyond the teacher
(both use the same KO-GP ensemble), so some similarity is architectural rather than
imitative. P1 passing is therefore consistent with distillation and does not
establish it.

## What this could RETRACT

**The "policy distillation" framing itself.** If MF-DRO's queries are no closer to
MES's than to GP-UCB's, then calling it a distillation of MES is unsupported —
the teacher would be determining *whether it works* without determining *what it
does*, which is a materially weaker and stranger claim than the one now in
findings.md and in the published report.
