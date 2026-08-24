# H25 — the negative sigma_H sign is robust; its magnitude is not

Ten independently trained models, one per frozen-protocol seed.

| seed | `w[mu_H]` | `w[sigma_H]` | `w[mu_L]` | `w[sigma_L]` |
|---|---|---|---|---|
| 42 | +1.0322 | −0.2977 | +0.7718 | −0.0682 |
| 43 | +0.6319 | −0.3788 | +0.6438 | −0.3779 |
| 44 | +1.0824 | −0.5487 | +0.8254 | −0.0177 |
| 45 | +1.0221 | −0.2987 | +0.5861 | −0.6361 |
| 46 | +1.2229 | −0.0181 | +0.3891 | −0.0611 |
| 47 | +0.5968 | −0.3607 | +0.6296 | +0.0828 |
| 48 | +0.4161 | −0.0548 | +0.9534 | −0.2227 |
| 49 | +0.9455 | −0.6871 | +0.3503 | −0.3718 |
| 50 | +1.0884 | **+0.1297** | +0.0929 | −0.1986 |
| 51 | +0.5362 | −0.0296 | +0.9201 | −0.5883 |

- **PRED 1 PASS** — `w[sigma_H] < 0` on **9/10** seeds (bar was 8).
- **PRED 2 PASS** — `w[mu_H] > 0` on **10/10**.
- mean `w[sigma_H]` = **−0.2544** (sd 0.2578).

## What holds and what does not

**The sign is robust.** Nine of ten independently trained models penalise
high-fidelity posterior uncertainty, and all ten reward the posterior mean. The
uncertainty-aversion claim survives and stays in the paper.

**The magnitude is not.** Values span −0.6871 to +0.1297, and the mean sits only
about one standard deviation from zero. Seed 50 is positive outright.

**H24's headline number was on the large side.** Seed 44's −0.5487 is the second
most negative of the ten; the typical value is roughly half that. The paper must
quote the distribution, not seed 44's coefficient, or it overstates the effect.

## Scope

Ten seeds of one objective. This licenses "the learned rule is consistently
uncertainty-averse on Hartmann 6D" and nothing broader; no claim is made about
other benchmarks.
