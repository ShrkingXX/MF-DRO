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
