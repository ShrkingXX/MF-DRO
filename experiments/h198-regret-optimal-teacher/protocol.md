# h198 — a teacher that optimises the TASK, not the label

**CONFIRMATORY.** Locked before any code is written and before any result exists.
**Human-proposed**, and the proposal is a correction to my design, not an extension of it.

## The criticism this answers

I built the joint-IG teacher (h152) as `argmin b_T`, derived from the fact that the
labelled return telescopes to `log b_0 − log b_T`. That derivation is correct and the
teacher is a correct optimum **of the reward we wrote down**. That is exactly the
problem: it is the optimum of a *nominal standard*. A teacher that perfects the label
inherits whatever is wrong with the label, behind a derivation that looks rigorous.

The human's framing: an expert teacher should be one that **makes the optimal decision
at every step**, not one that maximises a proxy or one that merely *arrives* somewhere
good.

**Neither existing "good teacher" meets that bar, and I checked rather than assumed:**

| teacher | what its step actually optimises |
|---|---|
| MES (default) | per-step cost-normalised info gain — myopic, and about `y*`, not regret |
| joint-IG beam (h152) | `argmin b_T` — the labelled reward |
| h145 ORACLE | **nothing.** `x_τ = x_start + (x*−x_start)·τ/(T−1)`, `x_start ~ U(domain)` |
| UCB-LOC / EXPLOIT-LOC / Thompson / EI | posterior-greedy acquisition rules |
| RANDOM, HEAD/TAIL-MES | scripted ablations |

**No teacher in this repository has ever optimised final simple regret.** Verified by
grep over `src/policy/` and every `experiments/*/protocol.md`.

## Why the existing failures do NOT already answer this

This is the part that makes h198 worth compute rather than a fifth mechanism.

h145's oracle has `x_start ~ Uniform(domain)`, so **its τ=0 action is uniform noise**.
The confirmed mechanism (h171/h173/h192, transfer ratio 1.094) is that the DT emits its
teacher's **τ=0 action mean**. For a uniform start that mean is the box centre. So
h145's oracle destroyed precisely the signal the DT consumes — its 10/10 failure is
evidence about *that construction*, not about teacher quality.

The joint-IG beam has the opposite problem: its elite node's first child **is** the
greedy MES move, so its τ=0 is ~unchanged from the control. It could not have moved the
outcome either.

**So the mechanism does not predict a null here — it predicts the two previous arms
were both incapable of testing the claim.** That is the gap.

## Prediction, committed now

The mechanism says teacher quality helps **iff** it improves the τ=0 action mean. A
regret-optimal teacher's τ=0 is the optimal first query under `D_0` — a real decision,
low-variance across rollouts, and not the box centre.

- **P1** — τ=0 mean moves *and* frozen rel% improves vs the ROI-Q10 control (11.59).
  The mechanism is confirmed AND actionable: teacher design is a live lever.
- **P2** — τ=0 mean moves but rel% does not. The mechanism's *necessity* survives and
  its *sufficiency* is refuted: moving the mean is not enough.
- **P3** — τ=0 mean does not move. The teacher is not doing what it claims; a
  construction failure, not a result. Report as such.

## What a P1 would RETRACT

`THE_ANSWER`'s closing sentence is already withdrawn (state, 2026-09-03). A P1 would
further retract the *implicit* reading carried through ~10 arms — that teacher-shaping
is a dead lever — and would demote the "declined on measured premises" status of the
teacher-rotation arm. A **P2** retracts nothing but bounds the mechanism to necessity.

## Design

Per step, for each `(x, ℓ)` in a restricted candidate set: condition on a fantasy `y`,
run a cheap base policy (greedy MES) to the cost horizon, and score by **expected final
simple regret** under the frozen metric's own budget. Average over `M` fantasies. Pick
the argmin. Cost forces a restricted set — the top-`n_c` by MES — and small `M`.

**Mandatory SCs, registered here:**
1. At `n_c=1, M=1` the teacher reduces EXACTLY to greedy MES, step for step. (The beam
   has the same identity check; it caught a real elitism bug.)
2. `M` is large enough that the argmin is not selecting lucky fantasy draws. The beam's
   first version lost its entire apparent advantage to exactly this winner's curse
   (+0.6680). Report `planned − realised` as the curse magnitude, as h152 does.
3. The teacher's τ=0 action mean is measurably different from the MES control's, with
   its across-rollout SD reported. **If it is not, that is P3** and Stage 1 does not run.

## The label mismatch — named before running, not after

The RTG labels information gain (`log b_τ − log b_T`) while this teacher optimises
regret. The DT would be told "this trajectory earned return R" where `R` measures
something its teacher was not pursuing. Two coherent options:

- **(a)** keep MES labels — the comparison against every existing arm stays clean, but
  teacher and label pull in different directions;
- **(b)** relabel return-to-go with the task reward, closer to the original DT.

**Stage 1 runs (a)**, because it changes exactly one thing against controls already in
hand. (b) is registered as h198b and runs only if (a) gives P1 or P2 — otherwise the
teacher itself is unproven and relabelling would confound two changes at once.

PROTOCOL.md permits the method change; the evaluation stays frozen either way.

## Cost

Stage 0 (SCs + τ=0 mean) is minutes and gates everything. Stage 1 is Borehole seeds
42–46 against the ROI-Q10 control **already in hand** at 11.59, so 5 workers.
