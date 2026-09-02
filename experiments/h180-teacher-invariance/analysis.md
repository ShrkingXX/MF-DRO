# h180 — the emitted first query is invariant to the teacher's RULE

**EXPLORATORY.** No new runs; a re-analysis of already-saved Borehole traces
(seeds 42–46). Not pre-registered — it began as the premise check for a proposed
teacher-rotation arm, and the arm was declined on the result. Labelled
exploratory precisely because the finding came out of the check rather than a
protocol committed before it.

## Why this was run

A coherent 2-member rotation (`mes` + `ucb_loc`, the only two coherent teachers
available — 4 of the 5 shipped `TEACHER_POOL` members are incoherent) needs no
core change: rotation is already per-rollout at `mf_dro.py:2609`. Before building
it, the h174 lesson said check the premise. **If the two teachers emit the same
first query, rotation is null by construction.**

## Result (paired by seed, FROZEN metric imported from h83, not re-derived)

| teacher arm | dist to MES 1st query | frozen rel% |
|---|---|---|
| MES (control, h83) | 0.0000 | 15.82 |
| UCB-LOC (h155) | 0.0440 | 15.13 |
| MES-FROZEN (h153) | 0.0455 | 19.36 |
| STALE-PATH (h161) | 0.0455 | 19.53 |
| EXPLOIT-LOC (h159) | 0.0648 | 19.07 |
| TAIL-MES (h171) | 0.1567 | 43.94 |
| DIVERSE-GOOD (h146) | 0.2516 | 43.94 |
| **HEAD-MES (h171)** | **0.2714** | **16.96** |
| ORACLE-EXPERT (h145) | 0.3387 | 43.94 |
| RANDOM-POOL (h149) | 0.4342 | 43.94 |

Distances are in the unit box, paired by seed. The frozen metric reproduces every
independently-recorded value (control 15.82, UCB-LOC 15.13, HEAD 16.96, TAIL
43.94, MES-FROZEN 19.36), which is the check that it is the frozen metric.

### The positive control, without which this is just a null

Swapping MES → RANDOM moves the emitted query **9.9× further** than MES → UCB-LOC
(0.434 vs 0.044), with **disjoint** ranges (UCB-LOC max 0.065 < RANDOM min 0.312).
RANDOM's centroid also sits **nearer the box centre** (0.337 vs MES's 0.520) —
the direction the conditional-mean account predicts for a teacher whose first
move is uniform. **The instrument detects teacher effects.** The UCB-LOC null is
a measured invariance, not a blind probe.

### Noise floor

Same teacher, different seed: mean 0.773, min 0.454. The ≤0.065 cluster sits ~7×
below that — unambiguous. RANDOM-POOL's 0.434 is only just under it, so the
*unpaired* floor does not separate RANDOM; the paired statistic does.

## What it means

**Two clusters, and one exception.**

- **dist ≤ 0.065 → performance preserved (15.13–19.53).** Five arms whose *rules*
  differ as much as UCB vs exploit-only vs frozen-target vs stale-path all emit
  essentially the control's first query, and all work.
- **dist ≥ 0.157 → four of five arms sit at the 43.94 saturation floor.**
- **HEAD-MES is the exception:** it moves the query far (0.2714) and stays good
  (16.96). It is the one intervention that *selects* trajectories rather than
  randomising them.

So: changing the teacher's **rule** does not move the emitted action; changing
**what gets averaged** does. This is why six accounts on this front fell — all six
changed the rule — and why the two interventions that worked (h149 selection,
h172 shortening) both changed the averaging instead.

**No correlation coefficient is reported.** Four of the nine non-control arms are
pinned at the identical 43.94 saturation floor, so a Pearson r over this set
(+0.707) measures a ceiling effect, not a linear relation. The two-cluster
structure is the honest reading. (An earlier pass did compute and nearly report
that r, on a rel% formula that also disagreed with the control's known 15.82.)

## A bit-level confirmation of h177/h178

**MES-FROZEN and STALE-PATH emit a bit-identical first query on 5/5 seeds** —
and identical *first ten* queries — despite being different manipulations, run
in different experiments, for different reasons, with different output files
(different md5, different byte counts, and 101–114 vs 102–112 total queries).

Both manipulate the **RTG conditioning target**. h177/h178 found that channel
architecturally inert (trained response 0.5216 RTG vs 0.0056 BTG, 92.9× apart).
If the target does not reach the action, two different target manipulations must
emit the same action — and at bit level, for ten queries, they do. The runs
diverge only afterwards, once different queries have produced different GP state.

## Consequence: the rotation arm is DECLINED

Rotating between two teachers that agree to 0.044 cannot produce a resolvable
effect. **Not built.** This is the second arm declined on a measured premise this
tick (h174's was declined on a direction check).

## What could retract this

- A competent teacher whose rule differs *and* which moves the first query while
  staying good. HEAD-MES is the nearest thing, but it is a selection scheme, not
  a rule, so it does not refute the claim as stated.
- Borehole only, n=5. The tight cluster's 7× margin is wide, but the arms in it
  were not chosen to span rule-space systematically.
- The five ≤0.065 arms may be less rule-diverse than they look; "different rule"
  is my classification, not a measured quantity.
