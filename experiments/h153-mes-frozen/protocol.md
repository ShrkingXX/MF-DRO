# h153 -- MES-FROZEN: is the channel trajectory QUALITY, or ADAPTIVITY?

STATUS: protocol locked, nothing run.
TYPE: CONFIRMATORY. Predictions and the retraction set are stated below.

## The confound h152b exposed

Every substitute teacher tested so far is OPEN-LOOP:
  h145 ORACLE        forced_x, a frozen [T,d] path
  h146 DIVERSE-GOOD  forced_x, a frozen [T,d] path
  h149 RANDOM-POOL   re-draws per step but ignores the state -- non-adaptive
The control -- and only the control -- RE-DECIDES each step against its own
realised fantasy draw. So "teacher quality" and "loop type" have never been
separated.

h152b measured the penalty for freezing directly, using greedy's OWN path:
  greedy CLOSED-loop  +0.4860 rtg[0]
  greedy OPEN-loop    +0.3266 rtg[0]     penalty +0.1594, 0.72 noise floors, 16/21

## The test

MES-FROZEN. Run the ordinary closed-loop MES rollout to derive a path, then
re-run the SAME rollout with that path frozen through forced_x.

The frozen teacher's trajectories have EXACTLY the quality of the control's --
same rule, same state, same candidate pool, same everything -- and differ from
it in one bit only: they cannot adapt to the fantasy draws of the run they are
being fed into. This is the cleanest available separation of the two channels.

## Predictions (opposed, so the result is decisive either way)

ADAPTIVITY hypothesis: MES-FROZEN lands with ORACLE/RANDOM near 43.94 rel%,
  improves ~0/5, rtg_target collapses toward ~0.3.
QUALITY hypothesis:    MES-FROZEN lands with the control near 15.82 rel%,
  improves ~5/5, rtg_target stays near 0.98.

## Implementation

Wrap simulate_mf_trajectory: pass 1 unmodified (closed-loop) to obtain the
path, pass 2 with forced_x set to that path.

CRITICAL UNIT CONVERSION: traj['actions_x'] is stored NORMALISED to [0,1]^d
(mf_dro.py:1883) while forced_x is consumed as RAW domain (mf_dro.py:1619).
The path must be de-normalised, x_raw = lo + a*(hi-lo), or the teacher is
silently placed in the wrong space. h145 was unaffected because it built its
path in raw coordinates from the start.

Fidelity is NOT forced -- it is re-chosen by the same info-gain criterion at
the frozen point, matching h145/h146 exactly.

Cost: the rollout phase runs twice, so wall time roughly doubles. Accepted.

## Sanity checks (before any result is read)

SC1  pass-2 actions_x reproduces pass-1 actions_x to float tolerance
     (the freeze must actually replay the same locations).
SC2  pass-2 rtg[0] < pass-1 rtg[0] on average, reproducing h152b's open-loop
     penalty inside the real pipeline rather than in the offline harness.
SC3  fidelity is allowed to differ between passes; record how often it does,
     since a large divergence would mean the frozen arm differs in HF/LF mix
     as well as in adaptivity, which would reintroduce a confound.
SC4  use_roi=False remains bit-identical.

## Design

Borehole_8D, seeds 42-46, n=5. Borehole because h145 v2 established Hartmann is
confounded and Borehole is the clean benchmark. Frozen metric: rel% of
|optimum| @cost_curve 200 via h83 sr_curve+grid. No p-values at n=5.

Reference arms, all n=5 Borehole, same metric:
  control (MES, closed-loop)  15.82  improves 5/5  rtg_target 0.9761
  ORACLE                      43.94  improves 0/5  rtg_target 0.3113
  DIVERSE-GOOD                43.94  improves 0/5  rtg_target 0.3178
  RANDOM-POOL                 43.94  improves 0/5  rtg_target 0.2904

## What this RETRACTS

RA  MES-FROZEN fails (~43.94) -> RETRACTS the standing interpretation of h145,
    h146 and h149. "Better trajectory quality does not improve MF-DRO" would be
    unsupported by those experiments, because all three varied loop type at the
    same time as quality. The finding becomes: the DT needs an ADAPTIVE teacher,
    and quality was never the variable under test. findings.md and the published
    report in to_human/ both need correcting, not supplementing.

RB  MES-FROZEN succeeds (~15.82) -> RETRACTS the adaptivity hypothesis outright
    and EXONERATES the forced_x delivery path. h145/h146 would then stand as
    clean quality tests, and the h152b open-loop penalty (+0.16 rtg) would be
    established as real but too small to matter downstream.

RC  MES-FROZEN lands in between -> both channels contribute; no clean
    attribution, and the honest report is that the programme cannot currently
    separate them at n=5. Named now so an intermediate result is not spun as
    supporting whichever hypothesis it sits closer to.
