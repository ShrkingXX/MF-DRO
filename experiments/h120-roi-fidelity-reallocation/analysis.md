# H120 — analysis

**NO VERDICT ISSUED.** The confirmatory data does not exist (Amendment 1):
h84's ROI-OFF control was completed only at seeds 42 and 43. Every locked
prediction requires >= 4 of 5 seeds. What follows is n=2, DESCRIPTIVE, and is
not partial support for anything.

| | quantity | ROI-OFF | ROI-Q10 | paired | \|m\|/sd | dir |
|---|---|---|---|---|---|---|
| P1 | HF query count (predicted LOWER) | 96.0 | 92.0 | -4.00 | 0.94 | 2/2 |
| P1b | LF query count (predicted HIGHER) | 8.5 | 17.0 | +8.50 | 0.92 | 2/2 |
| P2 | time-to-incumbent (predicted LOWER) | 0.821 | 0.836 | +0.015 | 0.06 | 1/2 |
| P3 | **count-matched** mean HF y (HIGHER) | 223.20 | 239.93 | **+16.73** | 7.96 | 2/2 |
| — | uncounted mean HF y (confounded) | 223.87 | 239.93 | +16.05 | 10.00 | 2/2 |
| P4 | frac HF worse than init (control) | 0.102 | 0.042 | -0.060 | 0.83 | 2/2 |

Per-seed HF counts: OFF [93, 99] vs Q10 [86, 98]. LF: OFF [8.5 mean; 14, 3] vs
Q10 [29, 5]. Count-match K per seed: [86, 98].

## What can and cannot be said

- The fidelity direction (P1, P1b) is consistent with h119 in both seeds, but
  the seed-43 gap is small (99 vs 98 HF) and two points cannot establish it.
- **P3 is the one worth noting.** The count-matched quality gain (+16.7 in raw
  y, over the same number of queries in both arms) is close to the uncounted
  one (+16.1), which means h119's C5 was NOT merely an artefact of averaging
  over fewer queries in a more converged run. That was the specific confound
  P3 existed to remove, and at n=2 it survives removal. It still needs n=5.
- **P2 went the WRONG way** (1/2, +0.015). h119's C4 found the ROI converging
  substantially earlier (0.938 -> 0.748, effect 1.35, 5/5 on the h90 seeds);
  these two h84 seeds do not reproduce that direction at all. Reported because
  the rule is to report every result, and this one is against the hypothesis.
- P4, the pre-registered negative control, moved in both seeds (0.102 -> 0.042).
  It was predicted NOT to separate. At n=2 this says nothing, but it is flagged
  now so that a later n=5 result cannot be presented as if the control had
  always been expected to move.

## Registered compute (NOT launched)

Three runs: `Borehole_8D ROI-OFF` at seeds 44, 45, 46 under h84's exact config
(`dict(use_roi=False)`, BUDGET=200.0, n_hf=10, n_lf=20), which completes h84's
control arm and makes P1-P4 evaluable at n=5 exactly as locked. Estimated ~83
min each; three slots for one wall-clock hour and a half.

Not launched: the machine is at 15/15 (10 workers on the peer's h113, 5 on
h117). Queued for the first tick with free slots.

Note for whoever runs them: they must run on code that is empty-diff against
h84's `af5ec31b1` over `src/ dro_runner.py benchmarks.py`, or the completed arm
is not comparable to the ROI-Q10 runs already in hand. My working tree currently
carries the uncommitted h94/h102 patches, so this must be checked, not assumed.
