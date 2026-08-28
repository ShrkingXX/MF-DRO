# H120 — analysis (superseding the n=2 no-verdict version)

Data: Borehole, seeds 42-46 (as locked). Control = h83 `MF-DRO`, ROI arm =
h84 `ROI-Q10`. The control substitution is MEASURED bit-identical to h84's own
`ROI-OFF` at both overlapping seeds, 137 and 132 queries, 0 differing, across
three commits (Amendment 3). Zero new runs for this table.

## Verdicts on the locked predictions

| | prediction | control | ROI-Q10 | paired | \|m\|/sd | dir | verdict |
|---|---|---|---|---|---|---|---|
| P1 | HF count LOWER | 94.0 | 84.8 | -9.20 | **1.55** | 5/5 | **PASS** |
| P1b | LF count HIGHER | 12.6 | 31.0 | +18.40 | **1.58** | 5/5 | **PASS** |
| P2 | time-to-incumbent LOWER | 0.855 | 0.792 | -0.063 | 0.29 | 3/5 | **FAIL** |
| P3 | count-matched HF y HIGHER | 224.31 | 241.36 | +17.05 | **3.54** | 5/5 | **PASS** |
| — | uncounted HF y (confounded) | 226.20 | 241.36 | +15.16 | 3.43 | 5/5 | — |
| P4 | frac worse than init — predicted NOT to separate | 0.079 | 0.030 | -0.050 | 0.98 | 5/5 | as predicted |

Per-seed HF counts: control [93, 99, 93, 91, 94] vs ROI [86, 98, 81, 82, 77].
Per-seed LF counts: control [14, 3, 14, 19, 13] vs ROI [29, 5, 39, 36, 46].

## What is confirmed

**The ROI reallocates fidelity.** 9.2 fewer HF queries and 18.4 more LF queries,
5/5 seeds, at the seeds this protocol locked, against a proper control. This is
the first confirmatory pass in this line of work.

**Its individual HF queries are better, and not because the run converged.**
P3 was written specifically to remove that confound by comparing the first K
queries with K matched within seed. The count-matched gain (+17.05) is LARGER
than the uncounted one (+15.16), so the effect is not an artefact of averaging
over 9% fewer queries — averaging over the matched prefix strengthens it
slightly.

## What did NOT replicate

**P2, time-to-incumbent, failed: effect 0.29, 3/5 seeds.** h119's screen found
this at effect 1.35 with 5/5 on the h90 seeds (0.938 -> 0.748). At the
independent seeds it is 0.855 -> 0.792 and not separable. The "converges
earlier" limb of the hypothesis is **not supported** and is dropped.

That is exactly what the screen was for: it generated three surviving
quantities, and confirmation at independent seeds kept two and killed one.

## P4, the negative control, behaved as predicted — narrowly

P4 was pre-registered as NOT expected to separate, and it did not: effect 0.98
against a 1.0 bar. But it is 5/5 in direction and 0.98 is a hair under the bar.
Reported as "did not separate, as predicted" while noting that this is the
weakest possible version of that verdict, and that a different seed set could
easily push it over. It should not be cited as evidence that the ROI leaves the
founding diagnosis's statistic untouched on Borehole; it is evidence that the
effect there, if any, is smaller than the ones in P1 and P3.

## Standing

The fidelity-mix account is now CONFIRMED in two of its three limbs at
independent seeds, with a measured control. It remains an account of WHAT the
ROI changes, not of why that produces a regret gain: no arm here manipulates the
fidelity mix directly, so causation is untested.

The peer's h113 supplies a relevant constraint from the other direction: adding
the L1 loss on top of the ROI moves the mix BACK (+1.9 HF, -3.5 LF) while
improving regret by 2.10. So the fidelity mix is not L1's channel, and two
interventions that both help do not share this one.

## Limitations

- n=5, one benchmark, no p-values.
- Control read from h83 rather than h84 under a measured-equivalence exception.
  Three ROI-OFF runs at seeds 44-46 are in flight and will test that
  substitution at the three seeds where it is currently untested; a failure
  voids this table.
- P1 and P1b are the same fact under a fixed cost budget, not two.
