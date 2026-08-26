# H68 result — MODEL failure, not optimizer failure

**CONFIRMATORY** against `protocol.md`. This file REPLACES an earlier version
whose exploratory conclusion was a Monte-Carlo artifact; the retraction is
recorded in full below.

## Verdict (after replication — median of 8 independent draws per cell)

| prediction | bar | result | verdict |
|---|---|---|---|
| PRIMARY | `x_mig` above 50th pct | **9/9**, min-over-draws >= 60.3 in every cell | **MET** |
| SECONDARY | `x_mig` outranks `x_dro` in >=5 of 9 | **1/9** | **MISSED** |
| NULL | mean abs gap < 10 pct | did not fire | — |

## The PRIMARY was the wrong test — say this plainly

`x_mig` above the 50th percentile is **necessary but not sufficient** for
optimizer failure, and I mapped it to that conclusion in the protocol. It is the
**SECONDARY** that discriminates, and it failed at 1/9.

Both points score at the top of the pool:

| seed | k | x_dro median | x_mig median |
|---|---|---|---|
| 44 | 10/20/40 | 98.5 / 100.0 / 100.0 | 95.5 / 94.6 / 99.5 |
| 46 | 10/20/40 | 81.9 / 99.9 / 100.0 | 96.8 / 99.3 / 69.8 |
| 48 | 10/20/40 | 97.9 / 99.9 / 100.0 | 96.0 / 97.8 / 64.2 |

**MF-DRO's own choices score HIGHER than MI-Greedy's in 8 of 9 cells.** So
MF-DRO is maximising its acquisition *well* — better, by its own measure, than
the points that actually win. And it still finishes 23.7% short while MI-Greedy
finishes at 8.3%.

> **Conclusion: MF-DRO successfully maximises an acquisition that does not
> correlate with outcome on Borehole.** That is a MODEL failure — the surrogate
> or the acquisition — not an optimizer failure. Searching harder inside this
> acquisition cannot close the gap, because the acquisition already prefers
> MF-DRO's losing points to MI-Greedy's winning ones.

This is the reverse of what the PRIMARY's framing would have licensed, and it
promotes **surrogate class** over **teacher optimisation quality** among h60's
two remaining candidates.

## RETRACTED: the "k=10 policy/acquisition mismatch"

The previous version of this file reported that at k=10 MF-DRO proposes points
its own acquisition ranks poorly — mean percentile 52.1, with seed 46 at the
**8.7th percentile**. **That is withdrawn. It was a single-draw Monte-Carlo
artifact.**

Re-running the identical quantity 12 times, varying only the Thompson/pool draw:

| seed | reported 8.7-style value | median of 12 | min | max | spread |
|---|---|---|---|---|---|
| 44 | 89.8 | 98.9 | 94.8 | 99.8 | 5.0 |
| 46 | **8.7** | **79.4** | 41.2 | 91.3 | **50.2** |
| 48 | 57.7 | 99.6 | 79.7 | 100.0 | 20.3 |

Seed 46's headline number does not even fall inside the range of 12 replications.
The MES acquisition uses K=10 Thompson samples, so a single-draw percentile on a
600-point pool is far noisier than the effect it was being used to detect.

**A second, systematic defect made it worse.** The original run scored `x_dro`
and `x_mig` in the SAME `compute_joint_mf_mes` call. The shared `y*` Thompson
samples are drawn over the candidate set, so including MI-Greedy's strong point
raises `y*` and depresses `x_dro`'s MES — biasing exactly the comparison the
analysis was built on. Fixed by scoring each point in its own call.

## H68b — locked prediction MISSED

Does the mismatch track MF-DRO's per-benchmark standing? Predicted Currin
highest, Borehole lowest, Hartmann between. Measured (single draw, so read the
ordering only, not the values):

| benchmark | rel regret | mean x_dro pct | missing cells |
|---|---|---|---|
| Currin 2D | 0.0% | 99.1 | 0/9 |
| Hartmann 6D | 14.7% | **71.3** | 2/9 |
| Borehole 8D | 23.7% | 94.3 | 0/9 |

Ordering is Currin > Borehole > **Hartmann**, not the predicted
Currin > Hartmann > Borehole. **PRIMARY MISSED.** Hartmann, where MF-DRO is
mid-table, has the lowest agreement with its own acquisition; Borehole, where it
loses worst, has high agreement. The quantity does not track performance, which
independently undercuts the retracted mechanism.

Missing cells reported, not dropped: Hartmann seed 46 at k=5 and k=10 — that run
makes only 3 HF optimization queries in total.

## What still cannot be settled

n=3 seeds. A percentile is not a regret. The k=40 cells saturate at 100.0 for
`x_dro` in every seed, so the late-run comparison carries no information by
construction. And evaluating MI-Greedy's point under MF-DRO's surrogate
conditions on data MI-Greedy never had.

## Lesson

A stochastic diagnostic needs its own noise floor measured **before** any effect
is read off it. The retracted effect (52.1 vs 99.2, a 47-point gap) was smaller
than the single-cell replication spread (50.2 points). Nothing in the original
run was mislabelled or miscomputed — it was simply one draw, reported as though
it were a measurement.
