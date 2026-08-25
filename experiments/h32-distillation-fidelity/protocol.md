# H32 — how faithfully does the student approximate its teacher?

## Why now

H31 (running) asks whether the teacher alone beats MF-DRO. This experiment
measures the quantity that should *predict* that outcome, and is locked before
H31 lands so the prediction is genuine rather than retrofitted.

H24 found the student's argmax agrees with the teacher's on only **8/12** pools.
That is one coarse number. Here we measure the full approximation quality.

## Measurement

On 12 independently resampled 200-candidate pools, compare the student's score
vector `<w(h), cf_k> + b(h)` against the teacher's `compute_joint_mf_mes` score
for the same candidates:

1. Spearman correlation between the two score vectors
2. argmax agreement
3. top-10 overlap (does the student at least rank the teacher's good region high?)
4. the teacher's rank of the student's chosen candidate (how bad is the student's
   pick, in the teacher's own ordering?)

## Locked prediction, stated BEFORE H31 completes

If the student were a faithful distillation, its pick would sit near the top of
the teacher's ranking and H31 would show near-parity. We predict instead:

- **PRIMARY**: the median teacher-rank of the student's chosen candidate is
  **worse than 10** (out of 200), i.e. the student routinely picks something the
  teacher does not consider near-best.
- **CONSEQUENCE**: if so, we expect H31's teacher-only arm to have **lower**
  regret than MF-DRO's 0.4007. Recording this now so H31 can confirm or refute a
  prediction rather than merely produce a number.
- **NULL**: if the student's pick is typically in the teacher's top 10, the
  distillation is faithful, the loss is elsewhere, and H31 should show parity.

Single process, 1 thread (11 total with H31's 10). `PROTOCOL.md` untouched.
