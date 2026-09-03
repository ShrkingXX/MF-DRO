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

---

# STAGE 0 REVISED, before running it — a sharper and cheaper gate

Reading the code rather than trusting the summary changed the hypothesis.

**`decisionTransformer.py:78`** — when `hist` is supplied, positions are
`torch.arange(T)` and the readout is the **final** state token. So h27's window did
**not** leave the DT stuck at position 0: it already supplied **in-distribution position
indices 0…T−1** and read out at T−1. My earlier guess that the window "fixes the timestep
problem" was wrong; h27 already did that.

**This makes h27's null contradict h185.** h185 measured that the DT is a *per-timestep*
constant predictor whose explained variance **is** the between-τ component — 13.3%
(STDCOND) and 25.0% (LFF-CTRL) for ordinary teachers. If the DT emits a different constant
per position, reading out at position 6 must give a **different action** than position 0.
h27 measured them **bit-identical** (max |Δx| = 0.000e+00, ctx growing 1→6).

Both cannot describe the same code. **h27 is the stale one** (33 commits since, 3
behaviour-changing). That tension is now the strongest reason to rerun, and it is a
sharper gate than the `frac_seq` statistic originally registered here.

## Revised Stage 0

Two short runs at the same seed, current code, ROI-Q10, differing **only** in
`inference_context_k` (1 vs 8). Compare the emitted queries directly.

> **statistic**: max |Δx| between the K=1 and K=8 emitted queries, per iteration.
>
> - **G-PASS — h27 is stale, the arm is live**: proposals **differ** (max |Δx| > 1e-6 on
>   at least half the iterations where ctx > 1). The window changes the decision on
>   current code, so Stage 1's factorial is worth 10 worker-hours.
> - **G-FAIL — h27 replicates**: proposals bit-identical. The window is dead on current
>   code too, Stage 1 does **not** run, **and h185 owes an explanation** — a per-timestep
>   constant predictor with 13–25% between-τ variance should not be position-invariant.
>   That tension would become an open item in its own right.

Either outcome is informative, which the original `frac_seq` gate was not: it measured a
property of the *teacher*, whereas this measures the property of the *DT* that the whole
arm depends on. Cost: two short runs, minutes.

**Registered before running.** The `frac_seq` gate above is superseded, not deleted — it
was written before I read line 78, and the record should show why it changed.

---

# STAGE 1 SPLIT, registered before launch — cost measured, cheap half first

The joint-IG (beam) teacher was costed directly rather than assumed:

| beam config | per rollout | per seed (60 rollouts × ~100 iters) |
|---|---|---|
| M=1 | 0.54 s | 0.9 h |
| **M=16** (mandatory — fixes h152's winner's curse) | **1.57 s** | **2.6 h** |

On top of the ~0.9 h/seed baseline that is **~3.5 h/seed**, so the two IG arms are
**~35 worker-hours**. Affordable, but heavy for an outcome the mechanism predicts fails.

**The two factors are separable, and one is 4× cheaper.** Splitting:

## Stage 1a — the WINDOW alone (cheap: 5 workers, ~1 h)

| arm | teacher | inference | ROI |
|---|---|---|---|
| **WINDOW** (new) | MES | **K=8** | Q10 |
| control | MES | K=1 | Q10 — **already run, 11.59** |

One new arm. Tests the window in isolation at the configuration that actually wins, and
it is the half Stage 0 has already shown *changes the decision*.

Gate unchanged in form; threshold is the pre-existing 10.9% worst-case floor on 11.59 =
**1.26 rel% points**:

- **P1 — the window HELPS**: WINDOW − control < −1.26
- **P2 — no effect**: |diff| ≤ 1.26
- **P3 — the window HURTS**: WINDOW − control > +1.26 ← **what the mechanism predicts**

## Stage 1b — the EXPERT TEACHER (expensive: ~35 worker-hours) — GATED on 1a

> Run Stage 1b **only if** Stage 1a returns **P1 or P2**.
>
> If Stage 1a returns **P3** (the window hurts, as predicted), then adding a costly
> planned teacher *on top of a harmful inference change* cannot be attributed: any
> result would confound the teacher with a known-harmful window. Report 1a and stop.

This is not a way of avoiding the expensive arm. It is that the arms are **ordered**: the
window is the shared component of both, and if it is harmful the IG arms measure it too.

## Why this ordering does not shortchange the human's proposal

The proposal was expert trajectory **+** longer inference. Stage 1a tests the second
factor alone at the winning configuration; Stage 1b tests the first, conditional on the
second not being actively harmful. If 1a returns P3 the honest report is *"the shared
component fails, so the combination cannot be read"* — which is a real answer, not a
dodge, and it costs 1 hour instead of 36.

## A monitoring near-miss, recorded

Stage 1a was first launched from `code/worker_window.py`. Five processes started
correctly — and **`tools/count_workers.sh` reported 0**, because its default pattern is
`code/worker.py` and does not match `worker_window.py`.

That is the exact failure class the tool's own docstring was written about: miscounting
the fleet. Here it under-counts, which is the dangerous direction — acting on "0 workers"
would have meant launching more and breaching the 15-worker cap.

**Fix:** the worker is renamed to `code/worker.py`, the convention every other experiment
uses, so the default pattern matches. Caught by cross-checking `ps` against the tool
rather than trusting the tool's zero.
