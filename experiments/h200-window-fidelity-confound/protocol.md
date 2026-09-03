# h200 — is the sliding window's harm a HISTORY effect or a FIDELITY-MIX effect?

**CONFIRMATORY.** Locked before any code is run. Arises from an h197 measurement, not
from a hypothesis I went looking for.

## The confound, measured

| arm | LF fraction | final regret |
|---|---|---|
| h194 CTRL-K1 (no window) | **0.261** | **11.59** |
| h196 WINDOW (K=8, real actions) | 0.085 | 13.96 |
| h197 SPEC (K=8, L1, real-IG labels) | 0.092 | 14.55 |

Turning the window on cuts LF usage by **~3x**. On Borehole's 2:1 cost ratio that is
materially fewer total queries for the same budget. So every window arm varies TWO
things at once:

1. the DT sees history (the intended manipulation), and
2. the DT spends its budget on more expensive queries (unintended).

**Both predict the same endpoint.** h194, h196 and h197 cannot separate them, so the
sliding-window null is not currently safe to state as "history does not help" — a
sentence I have used, and which is not supported.

This is the third instance of this exact failure mode in this project: h60 (a Thompson
teacher collapsed fidelity to 99% LF, confounding teacher comparisons), h145
(quality confounded with endpoint diversity), and now the window arms.

## Design — hold the fidelity mix fixed, change nothing else

`max_hf_fraction` already exists (h184) and is inert at its default. Run the **h196
window arm** (the cleanest window arm: real actions fed, MSE loss, no extra spec
ingredients) with the HF ceiling set so its realised LF fraction matches CTRL-K1's
**0.261**, and compare to CTRL-K1.

Two arms, because constraining only the window arm would itself be a difference:

| arm | window | HF ceiling |
|---|---|---|
| **A** | K=8 (h196 config) | set to match CTRL-K1's LF fraction |
| **B** | K=1 (control) | **same ceiling as A** |

**B is mandatory.** Without it, arm A differs from CTRL-K1 in both the window AND the
constraint, which is the same error this experiment exists to fix.

## SC, registered before running

The ceiling must actually bind and land near the target. Measure realised LF fraction in
both arms.
- If arm A's realised LF fraction is not within **0.05** of arm B's, the constraint did
  not equalise the mix and the comparison is void — report as a **GATE MISS**.

## Prediction, and what each outcome RETRACTS

- **P1 — the deficit VANISHES** (|A − B| <= 1.26). The window's harm was a fidelity-mix
  artifact. This **RETRACTS the interpretation of h194/h196/h197's P3s** as evidence
  about history or context length, and **reopens the sliding-window question** that the
  human raised and that h27 was wrongly cited to close.
- **P2 — the deficit SURVIVES** (A − B > 1.26, same sign and rough size as +2.37). The
  window genuinely harms for a reason other than fidelity. Retracts nothing; it upgrades
  h196's P3 from "confounded" to "attributable", which is strictly more than it can
  currently claim.
- **P3 — the deficit REVERSES** (A − B < −1.26). The window HELPS once the mix is held
  fixed. Would retract h196's P3 outright.

Note the mechanism-level claim ("the DT is a per-timestep constant predictor") is NOT at
stake either way: it rests on h185/h186/h192, not on the window arms.

## Cost

Borehole seeds 42–46, both arms = 10 workers, ordinary MES teacher (cheap, ~80 min/seed
as h197 measured). Runs when h198 frees slots; the cap is 15 and h198 currently holds 10.
