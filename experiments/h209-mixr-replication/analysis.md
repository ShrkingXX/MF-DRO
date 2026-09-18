# h209 analysis — CONFIRMATORY. Scored as arms land.

## P-H (Hartmann 42–46) — FAILED, with the registered LF signature

| arm | frozen regret | LF frac |
|---|---|---|
| MIXR-H  20 MES + 20 random | 12.08 | 0.444 |
| NIR-H   20 MES | 9.75 | 0.558 |
| h84 ROI-Q10 K=1 (last full run) | 5.93 | |
| h83 MF-MES | 6.62 | |

MIXR-H − NIR-H = **+2.33** (se 1.94), MIXR better on **1/5**; per-seed [+0.5, +7.2, +0.3,
−2.9, +6.5]. LF fraction dropped on **5/5** seeds (−0.20, −0.12, −0.09, −0.08, −0.08) — the
same shift MIXR showed on Borehole (0.44 → 0.27).

This is the discriminating outcome the protocol registered: *MIXR worse AND LF dropped ⇒
the random half retrains the fidelity head toward HF; cheap at 2:1, costly at 8:1.* By the
protocol's rule (real only if all three of P-Q / P-H / P-B47 hold), **MIXR as configured is
not a general method.** The Borehole gain is, at least in part, a cost-ratio artefact.

Not the whole story on Borehole, though: CTRL-K1 has MIXR-B's LF fraction (0.261 vs 0.270)
and scores 11.59 vs 6.22, so mix alone does not explain the Borehole number. The clean
separation is a `random_p_hf` dose that matches MIXR's mix to NIR's (h210, first arm).

## An unregistered warning from the same runs

NIR-H (K=8, no positional embedding, `terminal_improvement`) = 9.75 vs the last full run's
ROI-Q10 K=1 = 5.93 on the same seeds. Every window arm since h194 was validated on
Borehole only, where K=8-no-position reached parity with K=1. On Hartmann — the benchmark
where MF-DRO was strongest — the same configuration is ~4 points worse. The comparison
conflates K=8 vs K=1, the label, and code drift since h84, so it is not a result; it is
the reason a current-code Hartmann CTRL-K1 (K=1, `mes_entropy`) must run before any
tuning. EXPLORATORY, flagged.

## P-Q and P-B47 — pending

## P-Q (Borehole 42–46, data-quantity control) — quantity ruled out

| arm | regret |
|---|---|
| NIR120-B  40 MES/member (120 traj/iter) | 14.04 |
| NIR-B     20 MES/member (60) | 12.62 |
| MIXR-B    20 MES + 20 random (120) | 6.22 |

NIR120 − NIR = +1.42 (se 1.92, 2/5) — the first clause's band (±1.26) is missed by 0.16 in
the direction that STRENGTHENS the conclusion (more MES data is slightly worse). MIXR −
NIR120 = **−7.82** (se 3.48), MIXR better on **5/5**. Data volume does not explain MIXR.

## P-B47 (Borehole 47–51, fresh seeds) — SUPPORTED

MIXR-B47 **4.84** [1.76 6.98 5.11 9.32 1.05] vs NIR-B47 **10.41** [8.09 9.68 13.73 10.74
9.79]: −5.56 (se 1.51), MIXR better on **5/5**. Replicates more cleanly than the original.

## Verdict: 2 of 3 — "partial, unexplained" by the protocol's rule; now explained

The Borehole gain is real (10/10 seeds over two seed sets, ≈ −6 vs the matched control),
is not data volume, and does not transfer to Hartmann.

**Mechanism (EXPLORATORY, from the traces on disk):** mean number of dims within 0.02 of a
box face among real HF queries — Borehole: MIXR 1.28 / 1.75 vs NIR 0.95 / 0.51 vs CTRL-K1
0.60 vs MF-MES 2.52 (Borehole's optimum has 7 of 8 dims on faces; MF-MES queries faces 97%
of the time). Hartmann: MIXR 0.54 vs NIR 0.27 (optimum interior). Together with the LF
shift (−0.12 to −0.17 on every seed set):

> **The random half shifts the regression mean.** Random-in-ROI actions include edge
> points, so the MSE location head's mean moves toward the faces; the random half's 50% HF
> moves the fidelity head toward HF. Borehole rewards both (corner optimum, 2:1 cost);
> Hartmann punishes both (interior optimum, 8:1). Same mechanism MIXO exposed with a
> constant target. Not RTG selection; not "learning from diversity".

This is also MF-DRO's known Borehole boundary aversion, accidentally and crudely corrected.
The principled correction already exists in the code: H102's `loc_loss='l1'` (the median
sits AT a bound where the mean is pulled inward) — and it carries no fidelity cost.

## What this RETRACTS / leaves

- Retracted: "add random rollouts" as a general method (P-H).
- Retracted: the diversity account in its DT-paper form (RTG never selected; the mean moved).
- Stands: a real, replicated, deployable Borehole improvement with a legible mechanism —
  which is a *diagnosis* of boundary aversion, not a method.
- Missing control, flagged: a current-code Hartmann CTRL-K1 (K=1, `mes_entropy`) — NIR-H
  9.75 vs the last full run's 5.93 says the K=8 line has never been validated off Borehole.

## Next (to register): h210
1. Hartmann CTRL-K1 on current code (the missing control).
2. `random_p_hf` matched to NIR's mix — isolates location-shift from fidelity-shift.
3. `loc_loss='l1'` on NIR (no random data) — the principled boundary fix, both benchmarks.
