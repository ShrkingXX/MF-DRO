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

## SCOPE CORRECTION (2026-08-28, after the peer's h129 P6) — Borehole only

h120 confirmed that the ROI reallocates fidelity. **That result is
Borehole-specific and must be stated as such.** The same measure across four
benchmarks, ROI-Q10 vs h83 control, seeds 42-46, paired, post-init HF count
fraction:

| bench | c_H | max HF affordable | control HF per seed | at ceiling? | shift | effect |
|---|---|---|---|---|---|---|
| Ackley_10D | 5.0 | 40.0 | 40, 40, 40, 40, 40 | **YES** | -0.033 | 0.49 |
| Currin_2D | 3.0 | 66.7 | 31, 40, 22, 21, 22 | no | -0.030 | 0.41 |
| **Borehole_8D** | 2.0 | 100.0 | 93, 99, 93, 91, 94 | no | **-0.144** | **1.65** |
| Hartmann_6D | 8.0 | 25.0 | 8, 24, 12, 6, 8 | no | +0.056 | 0.78 |

**Only Borehole clears 1.0, and it does so by 2x over the next benchmark.**
Currin is a fourth point I added to the peer's three; it is uncensored and
agrees (0.41).

**Ackley's cell is censored and should not be counted as evidence either way.**
Its control spends 40 HF x c_H=5.0 = exactly the 200 budget, in all five seeds
with zero variance. The fraction cannot rise, so the measure is one-sided
there.

**Headroom does not explain the ordering.** Ranked by HF queries affordable
(Hartmann 25, Ackley 40, Currin 67, Borehole 100) the effects run
0.78, 0.49, 0.41, 1.65 — not monotone. Ranked by how close the control sits to
its ceiling (Ackley 100%, Borehole 94%, Hartmann 46%, Currin 40%) they run
0.49, 1.65, 0.78, 0.41 — also not monotone. So there is no mechanical account
on this evidence; Borehole is simply the outlier.

**What survives unchanged.** P3 — count-matched mean HF y, +17.05 at effect
3.54, 5/5 — is a QUALITY measure, not a fidelity one, and nothing here touches
it. The peer's independent P4 (query score up, effect 2.66) agrees from a
different statistic. Quality is the mechanism that survives across measures;
fidelity is the one that does not survive across benchmarks.

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

## AMENDMENT 3'S INVALIDATION CONDITION: DISCHARGED (2026-08-28)

h120 read its control from h83's `MF-DRO` under a measured-equivalence exception,
and Amendment 3 registered a specific condition that would void the result: the
three ROI-OFF runs launched at seeds 44-46 must reproduce h83's MF-DRO
bit-identically at exactly the seeds where the substitution was untested.

**They do.**

| seed | fresh n | stored n | differing | fresh commit | dirty | stored commit |
|---|---|---|---|---|---|---|
| 44 | 137 | 137 | **0** | 26a9b9cd1 | True | 3654df07c |
| 45 | 140 | 140 | **0** | 15203e5cd | True | 3654df07c |
| 46 | 137 | 137 | **0** | ee5ecafcc | True | 3654df07c |

Bit-identical on fid, x and y at every query, across **three different commits**,
every one of them with the working tree dirty (the h94/h102 patches), against a
stored trace from a fourth.

This does more than clear h120:

1. **`MF-DRO` and `ROI-OFF` are the same arm**, now verified at 5 of 5 Borehole
   seeds (42-43 previously, 44-46 here) rather than 2.
2. **h117's GATE G0 is extended.** G0 established the current tree reproduces one
   stored trace (Ackley seed42, 83 queries). This adds three Borehole runs on
   three further commits — 414 more queries, still 0 differing.
3. **Everything downstream of the substituted control stands**: h120's confirmed
   P1/P1b/P3, the -4.22% ROI benefit row, and h128's control.

The check was registered before the runs finished and its failure condition was
stated in advance. It was not designed for purposes 1-3 above, which is what
makes it worth those conclusions.
