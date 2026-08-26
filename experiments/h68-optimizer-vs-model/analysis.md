# H68 result — PRIMARY met, SECONDARY missed

**CONFIRMATORY** against predictions locked in `protocol.md` before any number existed.

## Verdict

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | `x_mig` above 50th pct | **9/9 cells**, mean 98.1 | **MET** — optimizer-failure signature |
| SECONDARY | `x_mig` outranks `x_dro` in >=5 of 9 | **4/9** | **MISSED** |
| NULL | mean abs gap < 10 pct | 17.4 | did not fire |

MF-DRO's own MES acquisition rates MI-Greedy's winning points at the **95th-100th
percentile of a 600-point Sobol pool, at every checkpoint, in every seed**
(mean 98.1, min 95.0). The acquisition is not blind to the points that win. So
Borehole is **not** a case of the acquisition dispreferring the right region.

The SECONDARY miss is the honest qualifier and it is not incidental: `x_mig`
outranks `x_dro` in only 4 of 9 cells, because from k=20 onward MF-DRO's own
choices score just as high.

## Where the difference actually lives (EXPLORATORY — not predicted)

| k (HF optimization queries) | x_dro | x_mig | gap | per-seed x_dro |
|---|---|---|---|---|
| 10 | **52.1%** | 99.2% | **+47.2** | 89.8 / **8.7** / 57.7 |
| 20 | 96.1% | 95.6% | −0.5 | 98.2 / 90.5 / 99.7 |
| 40 | 100.0% | 99.4% | −0.6 | 100.0 / 100.0 / 100.0 |

The entire discrepancy is at **k=10**. Seed 46 proposes a point its **own**
acquisition ranks below 91% of random draws (8.7th percentile). By k=20 the
policy's proposals are indistinguishable from MI-Greedy's on this measure, and
by k=40 both saturate at ~100 — where the percentile test stops discriminating
and carries no information either way.

This coincides with the already-measured fact that Borehole's outcome gap
**opens by HF query 20 and never closes** (MI-Greedy 264.41 at q20; MF-DRO not
reaching it in 109).

Read narrowly, this says: MF-DRO's learned policy sometimes proposes points that
its own acquisition would reject, and it does so in exactly the early window
where the run's fate is decided. That is a **policy/acquisition mismatch** —
neither of h60's two candidates as stated, since the acquisition is fine and the
teacher's pool is not what is being scored here.

## What this does NOT establish

- n=3 seeds. The k=10 column is 89.8 / 8.7 / 57.7 — one catastrophic seed and
  two mediocre ones. A single seed is doing most of the work in that mean.
- A percentile is not a regret. A well-ranked point is not a proof it would have
  been found, nor that finding it would have closed a 15-point regret gap.
- The k=40 saturation (both at 100.0) means the late-run test is uninformative
  by construction, not evidence of late-run health.
- Evaluating MI-Greedy's point under MF-DRO's surrogate conditions on different
  data than MI-Greedy actually had.

## Defect found and fixed mid-analysis

The first run lost 3 of 9 cells to "too few LF". The h57 trace stores the initial
design as all-HF-then-all-LF, so slicing positionally at the k-th HF query cuts
before the LF init block and leaves zero LF rows. Fixed by always including the
full initial design and counting k as HF queries **after** it. The bug was not
cosmetic: it suppressed every k=10 cell, which is where the entire finding lives,
and it inverted the NULL verdict (mean gap 2.6 -> 17.4).
