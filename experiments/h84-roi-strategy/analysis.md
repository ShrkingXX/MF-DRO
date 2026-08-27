# H84 analysis — 34/34 runs, 0 failures

Reproduction control: **PASSED 4/4 bit-identical** (|d(regret)| and max|dx| both
0.000e+00, both benchmarks, two seeds each). Every comparison below is measured
against a control verified end-to-end, not merely inherited from h83.

## Final numbers, paired vs ROI-OFF, all arms n=5

| benchmark | arm | acceptance | d(q-score) | d(rel.regret) |
|---|---|---|---|---|
| Hartmann_6D | ROI-Q10 | 10.0% | +0.001 (3/5) | **-1.62 pts (3/5)** |
| Hartmann_6D | ROI-FIX2 | 12.9% | -0.094 (2/5) | -0.26 pts (2/5) |
| Hartmann_6D | ROI-ANN | 49.8% | -0.055 (1/5) | +1.56 pts (1/5) |
| Borehole_8D | ROI-Q10 | 10.0% | **+0.114 (5/5)** | **-4.22 pts (5/5)** |
| Borehole_8D | ROI-FIX2 | 21.4% | +0.077 (5/5) | **-4.81 pts (5/5)** |
| Borehole_8D | ROI-ANN | 49.3% | +0.034 (4/5) | -1.31 pts (3/5) |

## Pre-registered bars

**P1 — FAILED.** Required +0.10 mean HF query score on >= 4/5 seeds on BOTH
benchmarks. Borehole met it (+0.114, 5/5); Hartmann did not (+0.001, 3/5). Not
renegotiated. Note Lesson 22: PRIMARY was set on the MEAN when simple regret is
a MAX statistic, so this bar records nothing on Hartmann even though regret
improves 1.62 pts there.

**P2 — REFUTED.** Predicted the fixed-beta arm would buy nothing on Borehole.
It posted the BEST regret of any arm there: -4.81 pts, 5/5. My prediction
reasoned from offline acceptance at two data-size extremes (100% at n_hf=10,
0.4% at n_hf=35); the run-averaged acceptance is 21.4%. Compounding this, the
graceful-degradation fallback I wrote for the pool fix converts the late-run
collapse into dilution toward no-ROI rather than collapse onto duplicated
points -- so my own fix is what made the arm I predicted would fail succeed.

**P3 — MET.** ROI-Q10's relative regret is lower than ROI-OFF's on Borehole:
-4.22 pts, better on 5/5 seeds.

**P4 — FAILED.** Required arm D not worse than arm C on either benchmark. ANN is
worse than Q10 on both (Hartmann +1.56 vs -1.62; Borehole -1.31 vs -4.22).
Note this bar was already compromised: arm D never annealed (Amendment 3), so it
tested a constant q~0.49 rather than the registered schedule.

Two of four registered bars met or refuted against me. The experiment's PRIMARY
prediction failed and its NEGATIVE prediction was refuted.

## What the experiment established anyway

1. **A calibrated ROI improves MF-DRO's regret on both benchmarks** -- Borehole
   -4.22 pts (5/5), Hartmann -1.62 pts (3/5) -- and on h83's own metric and bar,
   Hartmann's result puts MF-DRO past MF-MES (5.93% vs 6.62%, 4/5 paired wins),
   overturning h83's finding that MF-DRO beats no baseline anywhere.
2. **Fixed beta cannot set ROI tightness**, along three measured axes: across
   benchmarks (12.6%-100%), within a run (250x on Borehole), and across SEEDS of
   the same benchmark (6.9x on Hartmann). Calibration collapses all three to
   1.0x. This argument is independent of the regret numbers.
3. **The ROI acts on the upper tail**, not the bulk: Hartmann d(mean) +0.001 but
   d(p90) +0.031 and d(best) +0.022.

## CORRECTION carried forward

I wrote that "on Hartmann only q=0.10 helps; both looser settings HURT (+1.56,
+6.32)". At n=5 that is wrong for ROI-FIX2, which finishes at **-0.26 pts**,
i.e. roughly neutral rather than harmful. Its reported value moved +6.32 (n=2)
-> +0.36 (n=3) -> -0.26 (n=5) as seeds landed. Only ROI-ANN actually hurts
Hartmann. This is the sharpest illustration in the project of why partial arms
must not be quoted.

## What h84 does NOT establish

MF-DRO is still behind on Borehole (11.59% vs MF-MES 6.40%) and Ackley. The
Hartmann result rests on a benchmark where three ROI settings were run and the
winner is reported -- h87 exists to remove that, with q=0.10 fixed in advance on
seeds 47-51 and a falsifier requiring withdrawal if it fails.
