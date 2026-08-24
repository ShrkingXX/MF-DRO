# h1 analysis — first post-leak-fix run under the frozen evaluation

30/30 jobs, 6979 s wall. Cost matching verified before any regret comparison;
no run stopped on the iteration guard.

## Cost matching (checked first)

| method | mean cost | min | max |
|---|---|---|---|
| MF-DRO | 201.5 | 200.0 | 206.0 |
| MF-MI-Greedy | 207.2 | 202.0 | 215.0 |
| MF-GP-UCB | 200.0 | 200.0 | 200.0 |

MI-Greedy overshoots ~3.6% (it checks the budget at round start and a round
costs up to 2*c_H). Small, in MI-Greedy's favour, and disclosed.

## Headline

| method | mean +/- SE | sd | range | mean n_HF | mean n_improved |
|---|---|---|---|---|---|
| MF-DRO | **0.5047 +/- 0.0395** | **0.125** | [0.313, 0.725] | 17.5 | 3.00 |
| MF-MI-Greedy | 0.5091 +/- 0.1266 | 0.400 | [0.188, 1.540] | 14.2 | 3.30 |
| MF-GP-UCB | 1.7934 +/- 0.1223 | 0.387 | [1.182, 2.375] | **0.0** | 0.00 |

## FROZEN SUCCESS TEST: **FAIL**

    MF-DRO mean+SE        = 0.5442
    MI-Greedy mean-SE     = 0.3825
    0.5442 >= 0.3825  ->  FAIL

Reported exactly as pre-registered. `PROTOCOL.md` explicitly permits "no
within-frame fix closes the gap" and that is the correct headline: **the fixes
do not produce a method that strictly beats MF-MI-Greedy.**

Locked prediction (MF-DRO mean < 1.0): **MET** (0.5047 vs a pre-fix ~1.31 that
is not comparable).

## What DID change: the freeze is gone

**0/10 runs frozen** (`n_improved == 0`), against a pre-fix rate of 9/12 (75%).
Per-seed improvements: [5,1,2,4,5,1,3,3,5,1]. The pathology this investigation
is named after is resolved. Mean regret moved ~1.31 -> 0.505, though only the
post-fix number is trustworthy.

## Paired analysis (seeds are matched — more powerful than the unpaired SE test)

    MF-DRO better on 4/10 seeds, worse on 6/10
    mean paired diff  = -0.0045   (median +0.0915)
    Wilcoxon p        = 0.2754    -> no significant difference

**MF-DRO and MI-Greedy are statistically indistinguishable on this benchmark at
matched cost.** The near-zero mean difference is not evidence of parity in the
typical case — it is driven almost entirely by **seed45**, where MI-Greedy
failed badly (1.5396) and MF-DRO did not (0.4331). On the median seed MF-DRO is
slightly *worse* (+0.09).

## The real difference is variance, not mean

    sd:         MF-DRO 0.125  vs  MI-Greedy 0.400   (3.2x)
    worst case: MF-DRO 0.725  vs  MI-Greedy 1.540
    best case:  MF-DRO 0.313  vs  MI-Greedy 0.188

MI-Greedy has the better ceiling; MF-DRO has a much better floor and is 3.2x
more consistent. That is a genuine, reportable difference in behaviour that the
mean alone conceals.

**Caveat on the success test itself.** Its bar, `best-baseline mean-SE`, is set
by *the baseline's own instability*: MI-Greedy's large SE pushes the bar down to
0.3825. A method with an identical mean and one-third the variance is
structurally penalised. The test is frozen and reported as specified — this is
an observation about what it measures, not a reason to change it. But
"failed the criterion while being 3.2x more consistent" is a materially
different claim from "failed the criterion", and both belong in any write-up.

## H3 confirmed as a side result

MF-GP-UCB: `mean_n_HF = 0.0` and `mean_n_improved = 0.00` on **all 10 seeds**.
It never queries HF once, so its HF incumbent cannot move by construction. Its
"100% freeze" is definitional, categorically unlike the DRO family's
leakage-driven freeze. The old pooled freeze table conflated the two.

## Compute

30 jobs, 15 workers x 1 thread = 15 <= 15. MF-DRO 4565 s mean wall/run
(41 s/iter, unchanged from solo -> the thread cap eliminated contention);
MI-Greedy 18 s; MF-GP-UCB 140 s.
