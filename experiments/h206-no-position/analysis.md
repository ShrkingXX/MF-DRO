# h206 analysis -- CONFIRMATORY (P1/P2/P3 + falsifier registered in protocol.md before launch)

10/10 finals, 0 tracebacks, ~75-90 min/seed. Read from `results/*.json` only.
Metric IMPORTED from h83's `grid`: final simple regret, rel% of |optimum| @ cost 200.

## Result

| arm | positional signal | final regret | vs CTRL-K1 |
|---|---|---|---|
| h205B | absolute episode index | **10.68** | -0.91 (se 0.85, 4/5) |
| **N** | **none** | **11.39** | -0.20 (se 0.74, 3/5) |
| CTRL-K1 (h194) | arange, but K=1 so index 0 only | 11.59 | -- |
| **P** | arange, K=8 (matched, CURRENT code) | **14.44** | +2.85 (se 1.19, 0/5) |
| h196 | arange, K=8 (OLD code) | 13.96 | reference only |

## The falsifier, checked first

P - B = **+3.76** (se 1.71, B better 4/5), well outside the +/-1.26 band. Arange-K8 on
current code lands at 14.44, i.e. h196's 13.96 reproduced within noise. **h205's
separation holds; nothing re-opens.** The one assumption the whole h205 comparison rested
on is now measured rather than inherited.

## Scoring the registered predictions

- **P1 SUPPORTED by the registered criterion.** N - B = +0.71 (se 1.30, N better 2/5),
  inside the +/-1.26 band. Deleting the positional embedding outright reaches within
  noise of the absolute-labelling arm.
- **P2 not supported** (would need N - B > 1.26). **P3 not supported** (would need < -1.26).
- **Honest limit:** the paired se (1.30) is the size of the band, and the per-seed deltas
  are [-0.29, +5.1, +1.13, +0.59, -2.97] -- one seed (43) carries most of the mean. The
  result cannot distinguish "position carries exactly nothing" from "position carries a
  little". What it CAN say, with 4/5 agreement and a 3-point margin, is that both N and B
  are far better than P. No p-values at n=5.

## What this RETRACTS

The protocol registered: *"P1 retracts the framing in my h205 readout that 'absolute
labelling is the fix.'"* That fires. **Retracted:** B's mechanism is not the addition of
a useful absolute-time signal; it is the removal of a harmful one. N - P = **-3.05**
(se 1.69, 4/5) and B - P = -3.76 (se 1.71, 4/5) are the same effect. The `arange`
positional embedding, under a K=8 window, was an actively misleading input -- it welded
the readout index to "7 steps into a fantasy, RTG~0" -- and the fix is to stop supplying
it. The minimal change is deletion, which also removes the `max_seq_length` ceiling and
the overflow guard h205 needed.

Two consequences for how the result should be described:
- The DT paper's episodic timestep embedding is not doing here what it does in Atari.
  This is expected: the property that makes `embed_t(t)` meaningful in DT (random window
  offsets inside real episodes, decorrelating slot from time) is structurally absent when
  every training sequence is a fantasy anchored at "now". See protocol.md, reason 2.
- Budget progress is already in the state (`step_norm`, slot 5M+1), so deleting the
  embedding loses no information the model needs. Protocol reason 1, now supported.

## The fidelity collapse is now pinned to the arange signal (CONFIRMATORY here)

| arm | LF fraction |
|---|---|
| P arange-K8 (current code) | **0.091** |
| h196 / h197 arange-K8 (old code) | 0.085 / 0.092 |
| **N no position** | **0.462** |
| h205B absolute | 0.462 |
| CTRL-K1 | 0.261 |

h205 found (exploratory) that all three of its arms escaped the K=8 fidelity collapse.
h206 P reproduces the collapse exactly (0.091) on current code, and N escapes it exactly
as B did (0.462 to three decimals). Five window arms now line up with no exceptions:
**every arm carrying the arange positional embedding collapses to LF~0.09; every arm
without it does not.** The collapse is a consequence of the positional defect, full stop.
This closes the confound h200 was halted before attacking.

It remains true that fidelity and regret come apart (h205A had healthy LF and was the
worst arm), so "fix the mix" is necessary but not sufficient.

## Standing

With the harmful signal gone, the K=8 window is worth approximately nothing against no
window at all: N - CTRL = -0.20, B - CTRL = -0.91, both inside the band. The window is no
longer self-defeating -- that is Q1's premise repaired -- but it does not yet buy a gain.
That is consistent with the root cause h198 identified for RTG inertness: every rollout
comes from ONE teacher, so there is nothing in the history for the model to identify. The
paper's own hypothesis for why context helps (it lets the model tell which of many
policies produced the data) does not apply to a single-teacher dataset. Q2 is next.
