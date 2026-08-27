# H87 analysis — the Hartmann flip does not replicate. 10/10 runs, 0 failures.

| seed | MF-DRO+ROI | MF-MES | diff |
|---|---|---|---|
| 47 | 2.19 | 0.37 | +1.82 |
| 48 | 4.66 | 10.68 | -6.01 |
| 49 | 9.56 | 25.25 | **-15.69** |
| 50 | 7.95 | 7.20 | +0.74 |
| 51 | 6.98 | 5.95 | +1.03 |

paired mean **-3.62 pts**, sd **7.45**, wins **2/5**

## Bars

- **P1 FAILED.** Required the paired diff negative on >= 4/5 seeds. Got 2/5.
- **P2 NOT MET.** Predicted the margin would SHRINK from h84's -0.68 (selection
  inflates the original). It grew to -3.62 -- but only through seed 49's
  outlier, so this bar tells us nothing useful. Badly designed: a mean-based
  bar cannot detect shrinkage when one seed dominates the mean.
- **P3 FAILED.** h83's full bar needs lower mean AND >= 4/5 seeds. 2/5.

## The verdict was mechanical

`analyse.py` was written and committed while the treatment arm was still
running. It refuses to print a verdict below 5/5 paired seeds, compares only
within h87 at matched seeds, and prints the falsifier itself when P1 fails.
That was deliberate: six of eight bars had already failed across h84/h86, and
the temptation to reinterpret was real each time. Withdrawal here was triggered
by the script, not by my judgement after seeing the numbers.

## What the data actually show

MF-DRO + ROI wins big on the one seed where MF-MES collapses (seed 49: 25.25%
vs 9.56%) and loses narrowly on three of the remaining four (by 0.74, 1.03,
1.82). The favourable paired mean is an artifact of that single seed.

The defensible reading is **lower tail risk, not better typical performance** --
a real property, but not the one that was claimed, and not one h87 was designed
to test. Establishing it would need many more seeds and a pre-registered
tail-risk metric.

## What this experiment cost, and what it bought

Ten runs (~6 CPU-hours) to withdraw a claim that had already been published in
a report and written into findings.md, research-state.yaml and the research log.
The claim survived four separate caveats I had attached to it -- one benchmark of
four, selection over three ROI settings, post-hoc use of another experiment's
bar, and n=5 -- none of which was sufficient to stop me announcing it.

The caveats were correct and did not help. Only re-running with the setting
fixed in advance and fresh seeds settled it. That is the argument for building
confirmation into the plan rather than treating caveats as a substitute for it.
