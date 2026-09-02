# h187 — the no-DT control, on the FROZEN metric, on Borehole

**CONFIRMATORY.** Committed before the arm is launched and before any result exists.

## Why

The synthesis in findings.md ("the DT is an averager of its teacher's first move")
has three legs. Two are measured on this front (h185 constant-predictor, h186
input-insensitivity). The third — **that the averager is competitive with the teacher
it copies** — rests entirely on **h31**, which has three weaknesses:

- it is **Hartmann**, not Borehole;
- it reports **final simple regret**, not the frozen rel% at cost 200;
- its arms ran at **different fidelity mixes** (teacher-only 7–20 HF over 64–151
  iterations; MF-DRO 14–23 HF over 45–103), the same confound class that voided h174.

h31's own verdict was "not a net negative", 7/10 seeds, Wilcoxon p = 0.2324 — not
resolved. This arm supplies the missing like-for-like comparison.

## The arm

h31's mechanism verbatim: replace **only** `dt.propose_mf` so `compute_joint_mf_mes`
picks (x, ℓ) directly from the same candidate pool the DT would have scored. Initial
design, cost accounting and regret curve are the identical code path. Borehole, seeds
42–46, frozen metric, compared against the existing h83 MF-DRO control (15.82).

**Named asymmetry, not a defect of the design but the thing being compared.** The
teacher-only arm gets **pool + argmax**; MF-DRO emits a point directly. So this
compares "argmax over a candidate pool" against "a learned constant". That asymmetry
is exactly the contribution question, and it is why pool+argmax is legitimate *as a
control* here even though it is **not** an acceptable *fix* for MF-DRO itself.

## Gate

statistic: **paired (teacher-only − MF-DRO) frozen rel% points**, same seeds.
Threshold from the project's pre-existing harness noise floor (10.9% worst-case on a
15.82 base = **1.72 rel% points**), established long before this arm.

- **P1 — competitive**: |diff| ≤ 1.72. The synthesis's leg 3 holds on the frozen metric.
- **P2 — teacher BETTER**: diff < −1.72.
- **P3 — teacher WORSE**: diff > +1.72. The DT adds value beyond averaging.

Partitions the real line with no gap (`check_gate.py` verified on the |diff| form).

## What this could RETRACT

- **P2 fires → the synthesis's third leg breaks**, and worse: MF-DRO's Decision
  Transformer would be a **net negative** against its own teacher on Borehole under
  the frozen metric. findings.md's framing of the contribution would have to change,
  and so would the report's. This is the outcome most damaging to the project and it
  must be reported plainly if it occurs.
- **P3 fires → the "averager" account is incomplete**: the DT would be doing
  something beyond reproducing its teacher's first-move mean, which h185/h186 do not
  currently allow for.
- P1 leaves the synthesis standing on a clean comparison for the first time.

## Compute

5 workers × 1 thread. **Launched only when h184's head arms exit**, so the machine
stays at ≤ 10 concurrent and h184 — the registered priority — is not slowed. Running
both at 15/15 would roughly halve h184's throughput for no scientific gain, since
h187's readout is regret, not wall time.

## SC1 (is the teacher path actually taken?) — PASS, recorded before launch

Observable that a live DT cannot fake: the teacher sets `last_p_pred = float(ell)`,
so `p_pred_inference_per_iter` is exactly 0.0/1.0 every iteration, whereas a DT
emits a continuous probability.

```
SC1 p_pred values: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
SC1 teacher path taken: PASS (6 iters)
```

## A DEVIATION from h31 that the smoke test forced

h31's code read the candidate pool from `candidate_features`. In the current code the
call site passes `candidate_features=None` unless `use_candidate_scoring` is on — and
that flag must **not** be enabled, since pool+argmax is not an acceptable fix for
MF-DRO. The first smoke run crashed on exactly this (`'NoneType' object has no
attribute 'double'`), which is what the smoke test is for.

The teacher is an acquisition rule and inherently needs a pool to argmax over, so this
arm **draws its own**: `n_infer_candidates` (200) uniform draws over the bounds — the
same size and distribution the candidate path would have used, and the same
distribution `simulate_mf_trajectory`'s `roi_candidates` uses in training.

## A fidelity difference to check in the results, before the regret

SC1's `p_pred` was **1.0 on every probed iteration** — the teacher chose HF every
time, so this arm may run at `lf_fraction ≈ 0` against MF-DRO's 0.117. At **matched
cost** (which the frozen metric enforces) that is a strategy difference rather than a
metric confound, but it is exactly the h31 weakness this arm exists to avoid, so the
realised `lf_fraction` is to be read **before** the regret numbers and reported either
way.
