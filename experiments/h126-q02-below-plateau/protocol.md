# H126 — Is there anything below the plateau? q = 0.02 vs q = 0.10

STATUS: LOCKED before launch and before any h126 statistic was computed.
TYPE: CONFIRMATORY.
COMPUTE: 5 new runs (Borehole MF-DRO ~83 min each, 5 slots).
DATA: h126's own q=0.02 runs, paired against h84 `ROI-Q10` at the SAME seeds
      (42-46) under the measured-equivalence exception recorded below.

## Why, and why now

h125 established the first affirmative answer to the primary question: ROI
tightness IS a lever. Loosening from q=0.100 to q~0.495 costs **+9.018 regret,
effect 5.69, 5/5 seeds** on Borehole. Combined with h97/h107/h110 (q=0.05 vs
0.10 flat across three seed sets), the curve so far is:

  q = 0.05 ... 0.10   FLAT
  q = 0.10 ... 0.495  STEEP DEGRADATION

The unmeasured region is BELOW the plateau. That is where an actual improvement
on the current default could live, and the primary question asks for a strategy,
not just a warning about what is harmful.

q = 0.02 is a **5x** contrast against q = 0.10 — deliberately the same range that
h125 showed has power. The 2x contrasts that produced "tightness is a null axis"
are exactly what misled both sessions; this is not repeating that mistake.

## Feasibility check done BEFORE locking (no starvation)

`roi_raw_pool = 2000`, pool target `_N_POOL = 600`, `_MAX_DRAWS = 40`. Survivors
per draw ~ 2000q, so draws needed ~ 600/(2000q):

  measured  q=0.100 -> 3.51 draws, n_distinct 600
  measured  q=0.050 -> 6.55 draws, n_distinct 600
  predicted q=0.020 -> ~15 draws, well under 40

So q = 0.02 will not trigger the top-up-from-unfiltered path that commit 950fdd6
was written to fix. q = 0.01 would need ~30 of 40 and is NOT attempted here.

## GATE G1 (observed acceptance, blocking)

Every h126 run must report `roi_summary.accept_frac` within [0.018, 0.022] and
`n_distinct == 600`. If any run falls outside, the ROI did not realize its
target or the pool starved, and that run is reported and EXCLUDED rather than
analysed. This gate exists because ROI-ANN's whole failure was an arm whose
name asserted a tightness its realized acceptance denied.

## Measured-equivalence exception (declared before launch)

The comparator is h84's `ROI-Q10` at seeds 42-46, a different experiment.
Grounds, consistent with h120 Amendment 3 and h122:
  - h109 verified h84's `ROI-Q10` reproduces bit-identically on the patched
    tree, 2/2 seeds, 115 and 103 queries, |dregret| = 0.
  - h117's GATE G0 passed TODAY on this tree: 83 queries, 0 differing.
  - h126 copies h84's worker file byte-for-byte; only the arm dict differs.
Invalidation: if any h126 run's recorded commit is not empty-diff against
af5ec31b1 over `src/ dro_runner.py benchmarks.py`, the pairing is withdrawn and
h126 is reported as uncontrolled.

## Prediction (locked)

P1. q=0.02 does NOT differ from q=0.10 on final_regret at >= 4/5 seeds with
    |mean|/sd >= 1.0. **I predict the plateau extends.**

I am flagging that this is a genuine risk rather than a safe bet: h125 showed a
5x contrast HAS power, and this is a 5x contrast. If the plateau does not
extend, P1 fails and I will have predicted wrongly twice in a row on this axis.

P2 (pre-committed reading of the alternatives, so neither can be spun after):
  - q=0.02 BETTER  -> the useful region extends below 0.05 and the current
    default is not optimal. A strategy improvement, needing fresh-seed
    confirmation before it is believed.
  - q=0.02 WORSE   -> the curve is U-shaped and over-restriction has its own
    cost, most plausibly because a teacher confined to the top 2% by UCB stops
    supplying the policy with anything to learn exploration from. Equally
    interesting, and it would bound the useful region on BOTH sides.

## Limitations

- n=5, one benchmark, no p-values.
- Borehole only. h111 already showed the ROI's regret effect does not transfer
  to Hartmann or Ackley, so this tests the strategy where it exists and says
  nothing about generality.
- Comparator drawn from another experiment under the stated exception.
- One point below the plateau. A null bounds q=0.02, not the whole region.
