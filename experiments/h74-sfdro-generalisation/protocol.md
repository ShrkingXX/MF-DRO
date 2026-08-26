# H74 — Does h73's SF-DRO result generalise off Hartmann?

**CONFIRMATORY.** Locked before any h74 number exists.

## Why

h73 established, at n=10 with a passed reproduction control, that SF-DRO beats
SF-MES on Hartmann by 12.71 points, 10/10, p=0.0020 — the first claim in this
project to survive replication.

It is one benchmark. At n=3 SF-DRO **loses** to SF-MES on both others: Currin
0.4% vs 0.0%, Borehole 15.1% vs 13.3%. Those n=3 numbers are exactly the kind
lesson 26 says do not estimate a direction, so the losses are as unreliable as
the win was. The generalisation is untested in both directions.

A post hoc look at Hartmann also put SF-DRO (8.46%) level with MF-MES (8.24%,
4/10, p=0.43) — indistinguishable, and reached with **no** low-fidelity
information while MF-MES gets a free LF init worth +45 cost units. That is
suggestive and explicitly not a claim; it is a reason to find out whether the
single-fidelity result holds up anywhere else.

## Design

SF-DRO on **Currin 2D** and **Borehole 8D**, seeds 42-51 (n=10). Seeds 44/46/48
already exist in h59 and are reused as a built-in reproduction control; 7 new
seeds per benchmark run here. Identical config to h59/h73, cost budget 200, no LF
queries.

Comparator: **SF-MES at n=10 from h72**, already measured on both benchmarks
(Currin 0.00%, Borehole 12.76%) with its own reproduction control passed at +0.00.

14 jobs at 43.8 min (Currin) and 67.2 min (Borehole) per run. Runs at 7 workers
alongside h71's 6, inside the 15 cap.

## Locked predictions

1. **PRIMARY.** SF-DRO beats SF-MES on **Borehole** by **>= 2.0 points** and
   **>= 7/10** wins. Both required. This is the discriminating benchmark: n=3 had
   SF-DRO *losing* by 1.8 points, so a win here would mean the n=3 loss was noise
   and the Hartmann effect generalises.
2. **SECONDARY.** On **Currin**, SF-DRO within **1.0 point** of SF-MES. Currin
   does not discriminate — every non-degenerate method finishes inside 0.6% — so
   the only failure mode worth naming is SF-DRO being *materially* worse.
3. **NULL.** Borehole gap < 2.0 points or wins <= 6/10. Then h73's result is
   **Hartmann-specific**, SF-DRO is not generally better than its MES counterpart,
   and the honest summary is one benchmark of three.
4. **REVERSED.** SF-MES ahead on Borehole by >= 2.0 points, confirming the n=3
   loss. Then the Hartmann win is an outlier benchmark, not a method property.

## What this cannot settle

n=10 per benchmark. It compares SF-DRO to its own MES counterpart, not to the
multi-fidelity baselines — which get a free LF initial design and are a separate,
unregistered comparison. Even a clean sweep here would establish "SF-DRO beats
SF-MES on 3/3 benchmarks", not "SF-DRO beats the baselines".
