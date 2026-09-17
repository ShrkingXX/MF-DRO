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
