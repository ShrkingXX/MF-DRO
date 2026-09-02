# h152 Stage 0 -- CONFIRMATORY. Gate FAILED. R3 fired, with a sharper reason.

21 states (h83 Borehole traces, seeds 42/43/44, 7 cuts each), R=8 replays,
beam B=4 branch=4, cost-capped at greedy's own mean rollout cost.

| quantity | value | states |
|---|---|---|
| greedy realised rtg[0] | +0.4860 | per-state replay s.d. **0.2218** = noise floor |
| beam PLANNED rtg[0] | +1.0321 | |
| beam REALISED rtg[0] | +0.3641 | |
| **PLANNED lift over greedy** | **+0.5461** | positive **21/21** |
| **REALISED lift over greedy** | **-0.1219** | positive **6/21** |
| **winner's curse (planned-realised)** | **+0.6680** | positive **21/21** |

Gate required realised lift > 0 AND > noise floor. It is -0.55 noise floors.
**STAGE 1 DOES NOT RUN.**

## What actually happened

The joint optimum is NOT absent. The beam finds trajectories whose planned b_T
is much lower than greedy's, in every single state, by half an rtg unit. The
submodular gap the protocol predicted is real and large.

It does not survive fantasy resampling. Replay the SAME (x, ell) path with
fresh fantasy draws and the entire advantage disappears -- and then goes
NEGATIVE. Beam-selected paths are, on average, WORSE designs than greedy's.

This is a winner's curse, and the sign is the tell. The beam scores each
candidate path on ONE fantasy sample path and takes an argmin over B of them,
so it selects paths whose b_T was low because the fantasy draw was lucky, not
because the design was informative. Selecting on a noisy realisation of the
objective ANTI-selects on the underlying quantity. The beam preferred a
non-greedy path in 21/21 states and was wrong to, in 21/21.

## The number that matters beyond this experiment

The reward label's own signal-to-noise is **2.19** (mean +0.486, replay s.d.
0.222). rtg[0] = log(b_0) - log(b_T) is not a property of the trajectory; it is
a property of the trajectory AND the fantasy draw, and the draw contributes
roughly a third of the spread. Any procedure that OPTIMISES against b_T at this
noise level is fitting the draw.

This reframes R3. Greedy MES is not beatable here, but NOT because it is
jointly optimal -- the planned-lift column proves it is not. It is unbeatable
because the objective cannot be estimated precisely enough to search. Greedy
never selects on a realised b_T, so it never pays the curse.

## Retracted / not claimed

- The protocol's R3 as written ("greedy is already effectively the joint
  optimum") is NOT what happened and should not be reported that way. The
  joint optimum exists and is large in plan; it is unreachable in realisation.
- No claim is made about Stage 1, which did not run.
- This says nothing yet about whether label noise explains the MAIN puzzle
  (why teacher quality does not order outcomes). It is a candidate, untested.

## Registered follow-up (EXPLORATORY -- see protocol-followup.md)

The curse is a scoring artefact, not a fact about the objective. Score each
candidate path by its EXPECTED b_T over M fresh fantasy replays instead of one,
and the bias should vanish. If a positive realised lift appears, the arm is
alive and Stage 1 runs. If it does not, the joint teacher is dead on the
merits and the noise reading above stands.

---

# h152b -- EXPLORATORY. Debiasing worked. Gate still FAILS. And it exposed a confound.

Re-ran Stage 0 with C1 (select by mean log b_T over M=16 fresh replays) and C2
(prune by accumulated MES instead of single-path b_tau).

| | baseline M=1 | C1 only | C1+C2 |
|---|---|---|---|
| winner's curse | **+0.6680** | +0.0804 | **+0.0448** |
| SC2b (selection beats elite) | 0/21 | **21/21** | **21/21** |
| beam planned lift | +0.5461 | -0.0356 | -0.1218 |
| **beam realised lift** | -0.1219 | **-0.1160** | **-0.1666** |

C1 did exactly what it was designed to do: the curse collapsed 15x and
selection became provably honest (21/21). And the planned lift collapsed WITH
it, from +0.5461 to below zero. **The entire Stage 0 planned advantage was
selection bias.** The gate fails again, now on the merits.

## Correction to Stage 0's headline

Stage 0's analysis.md said the joint optimum "exists and is large in plan; it
is unreachable in realisation". That is WRONG and is retracted. Measured
honestly, the joint optimum over this candidate class is **not better than
greedy at all**. There was never a large joint advantage to be unreachable.

## The confound Stage 0 and h152b BOTH contained

Greedy is a CLOSED-LOOP policy -- it re-decides at every step against its own
realised fantasy draw. The beam emits a FROZEN plan delivered through forced_x,
which is OPEN-LOOP. Every "beam vs greedy" number above compares those two
things at once. Control: freeze greedy's OWN path and replay it.

|  | rtg[0] |
|---|---|
| greedy CLOSED-loop (re-decides each step) | **+0.4860** |
| greedy OPEN-loop (identical rule, own path frozen) | **+0.3266** |
| beam OPEN-loop (joint plan, frozen) | +0.2869 |

**OPEN-LOOP PENALTY = +0.1594**, 0.72 noise floors, closed-loop wins 16/21.

Re-reading every comparison at the SAME loop type:
- beam vs greedy, both open-loop: **-0.0397, beam wins 10/21** -- a dead heat.
- beam(open) vs greedy(closed): -0.1991, beam wins 3/21.

So ~80% of the joint teacher's apparent deficit is not the joint teacher at
all. It is the cost of freezing ANY plan.

## Why this matters far beyond h152

**Every substitute teacher tested so far is open-loop.** h145 ORACLE and h146
DIVERSE-GOOD are delivered through forced_x, a frozen [T,d] path. h149
RANDOM-POOL re-draws each step but ignores the state entirely, so it is
non-adaptive too. The control -- and ONLY the control -- adapts.

That is a systematic confound across the whole "teacher quality" programme. It
also predicts the one result that has stayed unexplained: L_loc is LOWER for
forced teachers (0.018-0.022 vs 0.040), i.e. the DT fits them BETTER and still
emits worse points. A state-independent teacher is exactly what you would
expect to be easy to fit and useless to imitate.

CAVEAT, stated plainly: the open-loop penalty measured here is +0.16 rtg units,
which is real but modest, and NOT on its own large enough to explain the
0.976 -> 0.311 rtg_target collapse. The policy-learning argument (a frozen
teacher teaches a state-independent policy) is a SEPARATE claim and is so far
untested. h153 tests it.
