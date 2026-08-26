# H75 — Calibrate MF-DRO, the last uncalibrated method

**CONFIRMATORY.** Locked before any h75 number exists. Bars name magnitudes.

## Why

h72 calibrated the cheap baselines and flagged its own main gap: MF-DRO and
SF-DRO were too expensive at n=10. SF-DRO has since been closed by h73/h74.
**MF-DRO is the last one left**, and it is the method this project is about — its
every standings entry is n=3 and of unknown reliability.

Two concrete reasons this matters now:

1. **Lesson 26.** Three of four exploratory n=3 directions taken to n=10 in this
   project failed, one reversing sign. MF-DRO's Borehole per-seed values are
   24.4% / 24.7% / 22.0% — tight — but its Hartmann values are 22.7% / 8.7% /
   12.7%, a spread as wide as methods whose means moved by >10 points at n=10.
2. **h71 is n=3 vs n=3.** Its PRIMARY compares POOL1000 (3 seeds) against h57's
   BASE (3 seeds). Whatever it returns is weak evidence by this project's own
   standard. A properly estimated BASE makes h71 interpretable rather than
   suggestive.

## Design

MF-DRO (h57 configuration, unchanged) on **Borehole 8D**, seeds 42-51. Seeds
44/46/48 already exist in h57 and are reused; the 7 new seeds run here. Borehole
is chosen over Hartmann because it is where h71's PRIMARY sits and where MF-DRO's
deficit is largest (23.7% vs MI-Greedy's 8.3%).

7 jobs at roughly 82-114 min each (h57's observed range), alongside h71's 5
remaining workers — 12 of 15 cores.

**Reproduction control, stated up front.** h75's worker is copied from h57's and
reuses seeds 44/46/48 from h57, so its built-in control would be vacuous in
exactly the way h73's was. h74's precedent applies: `diff` must show the worker
differs from h57's only in inert ways, and a direct control (h75's worker on
seed 44 vs h57's published value) is run and must pass bit-for-bit before the
verdict is reported. The verdict script enforces this in code.

## h57 safety — verified after launch, with a failed check redone

h75's worker is byte-identical to h57's, so it calls
`_build_mf_dro_config("h57", ...)` with the experiment name hardcoded. That is
only safe if nothing writes using it. Checked statically before launch (the
worker never calls `setup_dirs`/`save_result`/`log_global`; `mf_dro.py` never
imports `checkpoint`; `RES` resolves relative to `__file__`) and confirmed
dynamically after:

- h57's newest result file: **Aug 25 07:44** — 25+ hours before h75 started.
- h57's newest checkpoint: **Aug 25 13:18**.
- Global `RESULTS_ROOT` contains **no `h57` entry at all**.

A first attempt at the dynamic check used `find -newermt "-30 minutes"`, which
this system's `bfs` rejected as an invalid timestamp; it printed `0` from the
error, not from a match. That `0` was reported as evidence and was wrong. The
check was redone with `ls -lt` and the global-root inspection above. Recorded
because a verification that fails open is worse than none.

## Locked predictions

1. **PRIMARY.** |n=10 mean − published n=3 (23.7%)| **< 3.0 points**. Rationale
   is measured, not assumed: h72 found Borehole to be the *stable* benchmark —
   MI-Greedy shifted +1.0, MF-GP-UCB +2.6, SF-MES −0.5, and SF-DRO −0.5 — whereas
   Hartmann shifted +12.7 and +21.5.
2. **SECONDARY.** The C(10,3) three-seed span is **< 8.0 points**, comparable to
   the other Borehole entries (MI-Greedy 4.13, SF-EI 3.95, SF-MES 5.65, SF-DRO
   3.11) rather than to Hartmann's (26.07, 51.83).
3. **NULL / SURPRISE.** Shift >= 3.0 points. Then Borehole is not the stable
   benchmark h72 measured it to be for the *cheap* methods, MF-DRO's published
   entry is materially wrong, and every Borehole gap quoted in this project needs
   restating.

## What this cannot settle

Borehole only — MF-DRO's Hartmann and Currin entries stay n=3, and Hartmann is
the one h72 showed to be least resolved. It also does not re-rank anything: a
better estimate of MF-DRO's regret does not change that it loses on this
benchmark by a wide margin.
