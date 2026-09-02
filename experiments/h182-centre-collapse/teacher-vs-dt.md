# The DT tracks τ=0, not the teacher's average — a perfect inversion

**EXPLORATORY.** Re-analysis of `teacher_action_stats_per_iter` (mean vector of
`actions_x` over **all** τ, per iteration) against the DT's own realised queries.
Borehole, seeds 42–46. Distances are from the box centre in the unit box.

| arm | rel% | teacher mean over ALL τ | DT's own queries |
|---|---|---|---|
| **TAIL-MES** (random at τ=0, acquisition τ=1–7) | 43.94 | **0.571 → 0.614** | **0.290 → 0.089** |
| **HEAD-MES** (acquisition at τ=0, random τ=1–7) | 16.96 | **0.092 → 0.103** | **0.645 → 0.819** |
| ROLLOUT4 | 15.14 | 0.675 → 0.759 | 0.686 → 0.850 |
| ROLLOUT2 | 13.97 | 0.683 → 0.772 | 0.701 → 0.847 |
| ROLLOUT1 | 13.69 | 0.681 → 0.784 | 0.709 → 0.864 |

## The inversion

HEAD and TAIL are the pair built to dissociate "the first step" from "the other
seven", and on this measurement they invert **completely**:

- **HEAD-MES**: its teacher's average action sits essentially **at the box centre**
  (0.092) — seven of its eight steps are random, so the all-τ mean is centred. Its
  DT queries sit **far** from the centre (0.819), and it **works** (16.96).
- **TAIL-MES**: its teacher's average action sits **far** from the centre (0.571) —
  seven of eight steps are acquisition-chosen. Its DT queries sit **at** the centre
  (0.089), and it **fails** (43.94).

**The teacher's all-τ average is anti-predictive of what the DT does.** The one
step that predicts is τ=0. This is the front's answer measured directly, on the
arms designed to separate the two, rather than inferred from performance.

It is also a direct positive confirmation of h180: the DT reproduces the mean of
its teacher's **first move**, not of its teacher's moves.

## The feedback-loop hypothesis is NOT supported

The natural mechanism for h182's divergence was a loop: the DT's queries become
the GP's data, which seeds the teacher's rollouts, which set the DT's next
average — so collapse would compound through the teacher.

**Every teacher expands, including the failing arm's** (ratios 1.07, 1.12, 1.12,
1.13, 1.15 — TAIL-MES's teacher expands at 1.07 while its DT contracts at 0.31).
The teacher's action distribution does not collapse at all. **Whatever drives the
DT into the centre is internal to the DT**, not mediated by a degrading teacher
distribution. The loop hypothesis is dropped.

## Limits

- Five arms, one of them failing. `teacher_action_stats` was only ever recorded on
  h171 and h172, so this cannot be run across the 28-arm set.
- `teacher_action_stats` averages over all τ by construction; there is no per-τ
  breakdown saved, so "the τ=0 action specifically" is inferred from the
  HEAD/TAIL design rather than measured directly. Recording a τ=0 slice is a
  cheap addition for any future run.
- Correlational, like the rest of h182.
