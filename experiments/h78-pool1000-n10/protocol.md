# H78 — Does h71's POOL1000 result survive n=10 on Borehole?

**CONFIRMATORY.** Locked before any h78 number exists. Bars derived from
measurements, per lesson 28.

## Why

h71 found that widening MF-DRO's **rollout teacher** pool from 200 to 1000 moves
Borehole **23.71% -> 17.66%, 3/3** — the largest improvement any intervention has
produced on that benchmark, and the only one that has ever helped MF-DRO there.

It is **n=3**. Lesson 26: three of four exploratory n=3 directions in this project
failed at n=10, one reversing sign. This one is the strongest such signal seen
(3/3 with a +6.05-point margin against the locked reference, +5.23 against the
n=10 BASE), which is exactly why it must be replicated rather than assumed.

h75 has since measured the BASE cell at n=10 (**22.89%**, sd 2.94), so h78 can be
compared against a properly estimated, **paired** baseline rather than a
three-seed one — the weakness h71 was explicitly amended to acknowledge.

## Design

MF-DRO with `n_roi_candidates = 1000` on **Borehole 8D**, seeds 42-51. Seeds
44/46/48 exist in h71 and are reused; the 7 new seeds run here. Config otherwise
identical to h57/h71.

Comparator: **h75's BASE at n=10**, same seeds, paired.

Runs take ~9 hours each (h71's Borehole POOL1000 cells were 529-544 min). 7
workers on an idle machine.

**Reproduction control**, enforced in the verdict script as in h74/h75/h77:
h78's worker on seed 44 must reproduce h71's published POOL1000 value
bit-for-bit before any verdict prints. h78 reuses h71's seeds 44/46/48, so
without this the 10-seed mean mixes code paths.

## Locked predictions

1. **PRIMARY.** POOL1000 beats BASE by **>= 3.0 points** **and** wins **>= 7/10**
   seeds. Both required — h74's SECONDARY and h65's PRIMARY both passed a
   direction-only bar while the substantive comparison went the other way
   (lesson 27). The n=3 margin was 5.23 points against the n=10 BASE; requiring
   over half of it plus a clear win count guards against a single-seed mean.
2. **SECONDARY.** POOL1000 stays **above 12%** — it does not reach MI-Greedy's
   **9.29%**. h71's n=3 said 17.66%. If POOL1000 *did* reach 9.29%, the pool
   would be the whole story for MF-DRO as it was for the greedy baselines (h70),
   contradicting h71's SECONDARY.
3. **NULL.** Gain < 3.0 points **or** wins <= 6/10. Then h71's 3/3 was noise,
   lesson 26 goes **four of five**, and the only intervention that ever improved
   MF-DRO on Borehole is withdrawn.
4. **REVERSED.** BASE ahead by >= 3.0 points.

## What this cannot settle

Borehole only — h71's Hartmann cells are not replicated here, and against the
n=10 BASE POOL1000 was *worse* there by 1.41 points, so no Hartmann claim
follows either way. It also cannot meet the north star: even h71's face-value
17.66% loses to MI-Greedy's 9.29%, and prediction 2 expects that to persist.
This asks whether a real-but-insufficient improvement is real, not whether it is
sufficient.
