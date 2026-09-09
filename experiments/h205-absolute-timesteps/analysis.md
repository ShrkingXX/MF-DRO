# h205 analysis -- CONFIRMATORY (P1/P2/P3 + ordering registered in protocol.md before launch)

15/15 finals, 0 tracebacks, ~2h/seed. Read from `results/*.json` only (never `results/ckpt/`).
Metric IMPORTED from h83's `grid`: final simple regret, rel% of |optimum| @ cost 200.

## Result

| arm | final simple regret | vs CTRL-K1 | better on |
|---|---|---|---|
| **B  absolute only** | **10.68** | **-0.91** (se 0.85) | 4/5 |
| C  both | 12.27 | +0.68 (se 0.25) | 1/5 |
| A  prefix only | 13.12 | +1.53 (se 0.20) | 0/5 |
| CTRL-K1 (h194) | 11.59 | -- | -- |

Prior K=8 arms for scale: h196 13.96, h197 14.55. Saturation floor 43.94.

## Scoring the registered predictions

- **P1 (A beats CTRL, < -1.26): REFUTED.** A is +1.53, 0/5, se 0.20 -- consistently worse.
- **P3 on A and C: SUPPORTED.** Both are worse than CTRL with tight standard errors.
- **P2 on B: cannot be separated from a small real gain.** -0.91 with se 0.85 and 4/5 is
  inside the +/-1.26 band the protocol set for "no move", so B does not clear the
  registered bar. At n=5 no p-value is computed and none is implied.
- **Ordering prediction A >= C > B: REFUTED, and exactly inverted.** Observed B > C > A.
  A - B = +2.44 (se 0.96, A better on 1/5); A - C = +0.85 (se 0.22, A better on 0/5).

## What this RETRACTS

The protocol registered: "**B >> A** would retract this protocol's central argument (that
context depth is the binding constraint) and restore absolute labelling as the primary fix."
That condition fired. **Retracted: context depth at the readout is the binding constraint.**
The prefix, which was argued to fix the readout mismatch exactly and without touching the
embedding table, is the *worst* of the three arms and is reliably worse than no window at
all. Absolute labelling, which the revision demoted to "should contribute little on its
own", is the only arm that moves in the right direction -- and adding the prefix to it
(arm C) gives back 1.59 of B's advantage. The prefix is not neutral; it is harmful.

Why the prefix might hurt is not established here. One candidate, untested: the prefix
tokens are written with `valid_mask=False`, so they are attended but never supervised,
and they carry RTG values relabelled by the *inference* rule while the fragment carries
the *rollout* rule -- two different labelling conventions inside one sequence.

## EXPLORATORY -- the fidelity collapse is gone in all three arms

Not registered; discovered on readout, and therefore held to a lower standard of belief.

| arm | LF fraction |
|---|---|
| CTRL-K1 (no window) | 0.261 |
| h196 (K=8, defective labelling) | 0.085 |
| h197 (K=8, defective labelling) | 0.092 |
| **h205 A / C / B** | **0.253 / 0.305 / 0.462** |

Every previously measured K=8 arm cut LF usage ~3x versus no window; h202 established that
tax was paid identically regardless of teacher. All three h205 arms clear it, and B spends
*more* on LF than the no-window control. This is the first evidence that the window's
fidelity collapse was a consequence of the positional/phase defect rather than of the
window itself -- which is the confound h200 was built to attack before it was halted.

It does not follow that fixing the mix buys regret: A has a healthy LF fraction (0.253,
essentially CTRL's) and is still the worst arm. Fidelity saturation and final regret come
apart here.

## Standing

The K=8 family now reads 14.55 -> 13.96 -> 13.12 (A) -> 12.27 (C) -> 10.68 (B). B is the
first window arm at or below the no-window control. That is progress on Q1's premise --
the window is no longer self-defeating -- but it is not yet a win: B's margin does not
clear the band registered before the run.
