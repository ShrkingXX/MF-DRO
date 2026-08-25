# H34 — the fidelity head is correctly calibrated. Its TRAINING REGIME is wrong.

| quantity | value |
|---|---|
| training fidelity labels (`actions_ell`) | **57.7% HF** (n=1600) |
| teacher's raw choice **inside rollouts**, no floor | **50.6% HF** |
| gap attributable to `minimum_hf_fraction` | **+7.1 pp** |
| **fidelity head's output `p` (H33)** | **0.557** |

**PRED 1 FAILS** (+7.1 pp, needed ≥10). The floor is *not* the cause.

## The finding is better than the hypothesis

`p = 0.557` against a label rate of `57.7%`. **The head is calibrated to its
training labels to within 0.7 percentage points.** It is doing exactly what BCE
asks of it.

The problem is that those labels come from a regime that does not resemble
inference:

| regime | teacher's HF rate |
|---|---|
| **inside rollouts** (fantasy-conditioned GP) | **50.6%** |
| at real inference, fresh pools (H33) | **4.2%** |
| over a full real run (H31, interim) | **11.9%** |

The same acquisition, on the same benchmark, chooses high fidelity **four to
twelve times more often inside rollouts than at real inference**. The student
faithfully learns the rollout rate and then applies it to the real regime.

## Why this matters more than a miscalibrated head

This is a **train/inference distribution shift in the fidelity labels**, not a
learning failure. It is the same class of problem as RCSL's return-coverage
condition: the behaviour distribution the student is trained on is not the one it
acts in. And it is invisible to any amount of better training — the head is
already optimal for the data it is given.

Rollouts condition the GP on `sample_fantasy` draws, which evidently keeps
posterior uncertainty in a regime where cost-normalised MES favours HF far more
than the real, data-conditioned posterior does. Diagnosing *why* the fantasy
posterior does this is the natural next question.

## Scope, stated in the protocol and still binding

This explains the **level** of `p`, not its **uninformativeness**. `p` has
standard deviation `2.4e-4` across 24 decisions and correlates `+0.155` with the
teacher's choice (H33). A well-calibrated constant is still a constant. These
remain two separate defects, and H34 addresses only the first.

## Position breakdown

HF label rate by rollout step: `tau=0: 33.0%`, then `81.0%, 40.0%, 47.5%, 55.5%,
78.5%, 64.0%, 62.0%`. Note `tau=0` — the only position whose state resembles real
inference — is by far the lowest and closest to the real regime, which is
consistent with the shift being driven by fantasy conditioning accumulating over
the rollout.
