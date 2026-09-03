# h194 — expert (joint information-gain) trajectory + LONGER INFERENCE + ROI

**CONFIRMATORY**, two-stage with a cheap pre-gate. Registered before any code is written
and before any result exists. **Human-proposed** (the combination, and the observation
that h27 is stale).

## Why this is not h27

h27 paired a **myopic** teacher (greedy MES) with a sliding window and found the
proposals bit-identical. But a myopic teacher has **no plan to read**: each step is
chosen from the current state alone, so extra history tokens carry nothing the current
state does not.

The joint-IG teacher is different **by construction**. h152 derived that the rollout
reward telescopes, `rtg[0] = log b_0 − log b_T`, so maximising joint information gain is
**argmin b_T** — a *path-independent set-selection problem* over the T query points. That
teacher emits a **plan**, and one-step inference reads only its **first element**. A
window is the only way to read more of it.

**So the pairing matters:** a window is worthless with a myopic teacher and is the *only*
way to access a planned one. Neither h27 nor h152 tested that pairing. h152 evaluated the
planned teacher under **one-step** inference, where its plan is invisible past step 0.

**And h27's evidence is stale regardless** — 33 commits have touched the DT/policy since,
three changing behaviour, including `950fdd6` (Aug 27) whose message records that the ROI
candidate-pool resolution bug "confounded every ROI A/B". Any window-plus-ROI arm predates
that fix.

## What the current mechanism predicts, stated against the arm

The mechanism (measured on **current** code) predicts a **null**: `loss/var` ∈
[0.750, 1.054] across 10 arms means the DT sits at the best-constant solution, and state
sensitivity is 0.0122 (h186). **A constant does not depend on its inputs.**

This arm is worth running anyway because the prediction has a specific escape hatch: the
DT is at the best constant **for the target it was given**. A planned teacher is a
*different target*, and if its actions carry more sequential structure, the
best-constant solution is no longer near-optimal and a window has something to exploit.

## Stage 0 — the cheap gate (RUN FIRST)

Using `tau0_state_action_anova` (added this session) plus a sequential term, generate
rollout batches under **both** teachers from the same states and measure:

| quantity | meaning |
|---|---|
| `frac_state` | fraction of the τ=0 action variance explained by the τ=0 state |
| `frac_seq` | fraction of the action variance at τ>0 explained by that trajectory's OWN earlier actions, beyond the state |

`frac_seq` is the quantity a window can exploit and one-step inference cannot.
Reference: the ordinary MES teacher's `frac_state` measured **0.0999** (preliminary,
8 iterations, 1 seed).

> **GATE**: proceed to Stage 1 only if the joint-IG teacher's `frac_seq` exceeds the MES
> teacher's by at least **2×** AND is at least **0.10** in absolute terms.
>
> - **G-PASS** → the plan carries sequential structure a window could read. Stage 1 runs.
> - **G-FAIL** → the planned teacher's actions are no more sequentially predictable than
>   the myopic one's. A window has nothing extra to read, the arm is predicted null on
>   the mechanism, and **Stage 1 does not run** — reported as a gate miss, not a result.

Stage 0 costs minutes. Stage 1 costs ~10 worker-hours. The gate exists so a predicted
null is not paid for twice.

## Stage 1 — the arm (only if Stage 0 passes)

Borehole, seeds 42–46, frozen metric, **ROI-Q10 in all arms**:

| arm | teacher | inference | purpose |
|---|---|---|---|
| **A** | joint-IG (M=16 replay-averaged) | K=1 | isolates the teacher |
| **B** | joint-IG (M=16 replay-averaged) | **K=8** | the full combination |
| control | MES | K=1 | **already run**: ROI-Q10 = **11.59** |

**B vs A isolates the window. A vs control isolates the teacher.** Two new arms, 10
workers — the control is in hand, so the factorial costs no extra compute.

The M=16 replay-averaged selection is mandatory: h152's first version scored each path on
**one** fantasy draw and argmin'd over the beam, selecting lucky draws. That winner's
curse was +0.6680 and the entire apparent advantage vanished when it was fixed.

## Gates for Stage 1

Threshold is the project's pre-existing harness noise floor, 10.9% worst-case on ROI-Q10's
11.59 = **1.26 rel% points**.

- **P1 — the window helps a planned teacher**: B − A < **−1.26** (B better)
- **P2 — no window effect**: |B − A| ≤ 1.26
- **P3 — the window hurts**: B − A > **+1.26**

Secondary, reported alongside: A − control, isolating the planned teacher under one-step
inference. h152 found the planned teacher a **dead heat** with greedy at matched loop type
(−0.0397, 10/21), so A ≈ control is the expectation and a large |A − control| would itself
need explaining.

## What this could RETRACT

- **P1 fires → the mechanism's scope narrows sharply.** "A constant does not depend on its
  inputs" would still hold *for the targets tested so far*, but a planned teacher would be
  a target for which the DT is **not** at the best constant, and the whole
  "input-side fixes cannot help" conclusion — which currently unifies three Phase-1 nulls
  (state, conditioning, history) — would need an explicit exception. findings.md's Phase 2
  header would need rewriting.
- **P3 fires** → a window actively harms, unpredicted by anything, and would reopen h27.
- **G-FAIL** is the most likely outcome on current evidence and must be reported as a gate
  miss with its numbers, not quietly dropped.

## Prerequisite

`tools/identity_gate.py` must PASS exactly on any patched core before Stage 1 launches.

## Compute

Stage 0: 1–2 processes, minutes. Stage 1: 10 workers × 1 thread.
