# H72 — How much can n=3 mislead? Calibrate the standings against their own noise.

**CONFIRMATORY.** Locked before any h72 number exists.

## Why

Lesson 26: every exploratory n=3 direction this project took to n=10 failed, and
one reversed sign (h45, h64, h70b). The uncomfortable corollary is that **the h57
standings table is itself n=3** — every per-benchmark number quoted all project,
including "MF-DRO 23.7% on Borehole" and "MF-MES best on Hartmann at 8.5%".

Those orderings have never been checked against their own sampling noise. The
baselines are cheap (MI-Greedy 0.4-0.7 min/run, SF arms under a second), so this
is answerable directly rather than by assertion.

## Design

Run the cheap methods at **seeds 42-51 (n=10)** on all three benchmarks:
MF-MI-Greedy, MF-GP-UCB, SF-MES, SF-EI. Reuse the n=10 sets that already exist
(MF-MES Hartmann from h66; SF-MES/SF-EI Hartmann+Borehole from h70b).

Then, for every method with n=10, enumerate **all C(10,3) = 120 three-seed
subsets** and record the distribution of subset means. This is exactly the
estimator the h57 table used, so its spread is a direct measure of how much an
n=3 standings entry could have differed by seed luck alone.

MF-DRO and SF-DRO are excluded: at 82-473 min per run, n=10 across benchmarks is
not affordable this session. Their n=3 entries therefore remain uncalibrated, and
that limitation is part of the result rather than a gap to paper over.

## Locked predictions

1. **PRIMARY.** For at least one method on at least one benchmark, the range of
   120 three-seed subset means spans **>= 5.0 points** of relative regret.
2. **SECONDARY.** At least one **ordering asserted in the h57 standings** between
   two methods measured here is **not resolved** at n=3 — i.e. their three-seed
   subset-mean ranges overlap, so some 3-seed draw would have reversed the
   published ordering.
3. **NULL.** All ranges span < 5.0 points and no asserted ordering overlaps. Then
   n=3 was adequate for the standings after all, lesson 26 applies only to the
   small exploratory effects that produced it, and the headline table stands
   as-published.

## What this cannot settle

It calibrates the *cheap* methods only. It cannot tell us how noisy MF-DRO's or
SF-DRO's n=3 entries are — the two methods the project is actually about — and it
does not re-rank anything: a wide subset range says an ordering is unresolved,
not that it is wrong. The n=10 means computed here are better estimates than the
n=3 ones they replace, but they are still n=10.
