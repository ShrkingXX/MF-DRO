# H79 — Is SF-EI@1000 == MI-Greedy an identity, or a three-seed coincidence?

**CONFIRMATORY.** Locked before any h79 number exists.

## Why

h70's sharpest result: on Borehole, SF-EI with `n_candidates=1000` did not merely
close the 4.72-point gap to MI-Greedy — it **reproduced MI-Greedy's regret
exactly, seed for seed** (7.1 / 6.8 / 10.9 for both, residual **+0.00**). That
says MI-Greedy's entire advantage over the single-fidelity MES baseline is
candidate pool size, and it is the cleanest mechanistic claim this project has.

**It is n=3.** Lesson 26 applies to every three-seed result here. This one is
different in kind — a per-seed *identity* is far stronger evidence than a mean
difference, since three exact matches to ten significant figures cannot be
coincidence in the way three favourable means can. But "cannot be coincidence"
is an argument, and these runs cost **seconds**. There is no defensible reason to
leave it at three seeds.

The claim also matters beyond this project: it says a published baseline gap was
a harness parameter, not an algorithmic difference.

## Design

SF-EI with `n_candidates=1000` on **Borehole 8D**, seeds 42-51. Comparator:
MI-Greedy at n=10 from h72, already measured (9.29%) with its own reproduction
control passed at +0.00. Seeds 44/46/48 exist in h70 and are reused.

Runs cost seconds; this adds negligible load beside h78.

## Locked predictions

1. **PRIMARY.** SF-EI@1000 matches MI-Greedy **bit-for-bit on >= 9 of 10 seeds**
   (|difference| < 1e-9). Not "close" — identical. One seed of slack is allowed
   only for a tie-break or floating-point path difference, not for a genuine
   algorithmic divergence.
2. **SECONDARY.** Where they differ at all, |difference| < 0.5 points of relative
   regret.
3. **NULL.** Matches on <= 8 of 10. Then h70's three-seed identity was a
   coincidence of those seeds, the two methods are merely *similar* rather than
   equivalent, and the claim "MI-Greedy's advantage is entirely pool size"
   weakens to "mostly pool size" and must be restated everywhere it appears —
   including in the slide deck.

## What this cannot settle

Borehole only. On Hartmann MI-Greedy runs at 12% HF, so its LF phase is active
and the reduction to a single-fidelity EI loop does not hold there; no Hartmann
claim follows. It also says nothing about MF-DRO, whose own pool lever closes
only a third of its gap (h71, replication running as h78).
