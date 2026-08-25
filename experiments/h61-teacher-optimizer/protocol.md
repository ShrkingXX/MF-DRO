# H61 — is MF-DRO's teacher losing to SF-DRO's because it barely optimises the acquisition?

**CONFIRMATORY. Protocol committed before any run.**

## The gap in the code, located

The two DRO variants optimise their rollout acquisition very differently:

| | broad samples | restarts | local refinement |
|---|---|---|---|
| **SF-DRO** `_optimize_acquisition` | **1000** (`opt_samples`) | 10 | **100 samples** at noise 0.05 |
| **MF-DRO** `compute_joint_mf_mes` | **200** uniform | none | **none** — plain `scores.reshape(-1).argmax()` |

MF-DRO's teacher does a flat argmax over 200 random points. SF-DRO's does a
1000-point search with restarts *and* a local refinement pass.

## Why this is the lead

Three measurements converge on it:

1. **h47-variant-d**: a 200-point pool finds an acquisition value **4.3x worse**
   than a 4000-point one on Hartmann, and is not saturated at 4000. MF-DRO's
   teacher sits at the bottom of that curve.
2. **h60**: the rollout teacher is **load-bearing** — swapping MF-MES for
   `thompson` moved Borehole regret 23.7% -> 43.8% and collapsed the fidelity
   head to ~99% LF. Teacher quality has a 20-point lever on the result.
3. **h60 also excluded** the reward schema (0/3) and the LF initial design (1/3),
   leaving the teacher and the surrogate as the only live candidates. This tests
   the teacher without touching the surrogate.

The MF-DRO/SF-DRO comparison was never "multi vs single fidelity" — on Borehole
MF-DRO is 99-100% HF. This is a concrete, code-located difference that survives
that objection.

## Design

Borehole 8D (fidelity inert there, so the teacher is isolated), seeds 44/46/48,
cost budget 200. One change from h57's MF-DRO: the rollout teacher's candidate
pool and refinement.

| arm | teacher optimisation |
|---|---|
| **BASE** | 200 uniform, argmax — **reuses h57's cells** |
| **POOL1000** | 1000 uniform, argmax |
| **REFINE** | 1000 uniform + local refinement (100 samples, noise 0.05), matching SF-DRO |

9 new jobs. BASE reused (policy code byte-identical to its pin — re-verified at
launch and the hash recorded per result).

The change is confined to a new `n_roi_candidates` / refinement path used by
`simulate_mf_trajectory`'s teacher call. `src/policy/mf_dro.py` is no longer
frozen (h57 finished), but the edit is additive and gated behind config keys so
BASE's behaviour is unchanged by construction.

## Locked predictions

1. **PRIMARY**: REFINE's mean final HF simple regret vs BASE's 73.40 (23.7%),
   paired on 3 seeds. Direction and win counts only.
2. **PRE-REGISTERED EXPECTATION**: **REFINE beats BASE on >= 2/3**, and POOL1000
   lands between them. If teacher optimisation quality is what separates SF-DRO
   from MF-DRO, giving MF-DRO SF-DRO's optimiser should close part of the 8.6pp
   gap.
3. **NULL**: neither arm moves. Then teacher *optimisation quality* is excluded
   even though the teacher *identity* is load-bearing (h60), and the surrogate
   class is the last candidate standing.
4. **HARMFUL**: a sharper teacher makes it worse. Live, and not perverse — a
   better-optimised MES teacher concentrates its demonstrations, which could
   reduce the diversity the DT trains on. h45 showed this architecture is
   sensitive to demonstration diversity.

## What this cannot settle

n = 3, one benchmark. Borehole is chosen because fidelity is inert there, not
for generality. A teacher effect here may differ on Hartmann, where the fidelity
split does vary and the teacher drives it (h60).

---

## AMENDMENT (before any run, 0 result files on disk): arm sizes cut for compute

`compute_joint_mf_mes` is O(N) in the candidate count and is called once per
rollout step — 8 steps x 200 rollouts x ~100 iterations. So the pool size
multiplies the dominant cost almost exactly. BASE took **~92 min/seed** on
Borehole; a 1000-point pool would be ~5x that (~7.7 h/seed) and the REFINE arm
as originally written (1000 + 100) ~10x (~15 h/seed). That is not affordable and
I should have costed it before writing the arms.

Revised, keeping both mechanisms separable and each affordable:

| arm | broad | refinement | approx cost vs BASE |
|---|---|---|---|
| **BASE** | 200 | none | 1x (reused, not re-run) |
| **REFINE** | 200 | **100 samples, noise 0.05** | ~1.5x |
| **POOL600** | **600** | none | ~3x |

This still separates the two mechanisms — REFINE isolates SF-DRO's *local
refinement*, POOL600 isolates *broad-search resolution* — and REFINE is the more
distinctive of the two, since a local pass is the thing MF-DRO's teacher lacks
entirely. The predictions are unchanged in direction; only the magnitudes
available are smaller, so a null on POOL600 is weaker evidence against pool size
than the original 1000 would have been.

**Regression gate passed before any arm ran**: with the new config keys at their
defaults (`n_roi_candidates=200`, `teacher_refine_samples=0`), a 3-iteration
Currin run reproduces the pre-edit result bit-for-bit — regret curve
`[1.5398834959, 1.5398834959, 0.0495809348]` and `x_t[0] = [0.0, 0.0]` on both
sides of the edit. BASE's h57 cells therefore remain valid for reuse.
