# H77 — Calibrate MF-DRO on Hartmann, the last uncalibrated cell

**CONFIRMATORY.** Locked before any h77 number exists. Bars derived from prior
measurements, per lesson 28.

## Why

h75 closed Borehole: every entry there is now n=10 with a passed reproduction
control, and MF-DRO's published 23.71% held up (n=10 = 22.89%, shift −0.82).

**Hartmann is the opposite case and MF-DRO's entry there is the last one left at
n=3.** h72 measured how badly n=3 misleads on Hartmann: published baseline
entries moved **+12.7** (MI-Greedy) and **+21.5** (MF-GP-UCB) at n=10, and
three-seed spans there run 9.39 (SF-DRO) to 51.83 (MF-GP-UCB), against Borehole's
3.11-5.89. MF-DRO's own Hartmann seeds are **22.7% / 8.7% / 12.7%** — a spread
consistent with the unstable ones.

MF-DRO's Hartmann figure of 14.68% is quoted throughout this project's standings.
It has never been checked.

## Design

MF-DRO (h57 configuration, unchanged) on **Hartmann 6D**, seeds 42-51. Seeds
44/46/48 exist in h57 and are reused; the 7 new seeds run here. Worker copied
byte-identically from h57's, as in h75 — verified safe there (never calls
`setup_dirs`/`save_result`/`log_global`; `mf_dro.py` never imports `checkpoint`;
`RES` resolves relative to `__file__`; no `h57` entry in the global results root).

7 jobs alongside h71's 5 remaining workers — 12 of 15 cores.

**Reproduction control**, as in h75: h77's worker on seed 44 must reproduce h57's
published value bit-for-bit before any verdict prints. Enforced in the verdict
script, not left to judgement.

## Locked predictions

1. **PRIMARY.** The C(10,3) three-seed span is **>= 8.0 points**. Grounded in
   measurement, not intuition: every Hartmann span h72 and the SF-DRO calibration
   produced is >= 9.39 (SF-MES 10.13, SF-EI 15.32, SF-DRO 9.39, MI-Greedy 26.07,
   MF-GP-UCB 51.83), while every Borehole span is <= 5.89.
2. **SECONDARY.** |n=10 mean − published 14.68%| **>= 2.0 points** — a larger
   shift than Borehole's 0.82, because Hartmann is the unstable benchmark. Note
   this predicts *instability*, not a direction: the baselines shifted upward
   there, SF-DRO downward by 3.03.
3. **NULL.** Span < 8.0 **and** |shift| < 2.0. Then Hartmann is stable for MF-DRO
   specifically, despite being unstable for every other method measured, and the
   published entry stands as-is.

## What this cannot settle

Hartmann only; MF-DRO's Currin entry stays n=3 (Currin does not discriminate —
every non-degenerate method finishes inside 0.6%, so its span is bounded by
construction). It re-ranks nothing: MF-DRO is mid-table on Hartmann and a better
estimate does not change the north-star verdict, which is already settled negative.
