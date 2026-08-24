# H20 — is the attenuation architectural, or specific to the linear score head?

## Why this must be answered before writing

The measured mechanism is a ~10× attenuation of state variation
(0.346× encoder, 0.294× head), leaving the argmax invariant 0/12. The obvious
reviewer question — and the one thing that would change the paper from a pure
negative into a negative-plus-fix — is whether the **head's** half is an artefact
of the *linear* scoring parameterisation.

Under `use_linear_score_head=True` (the default) the score factors as

    score_k = <w(h), cf_k> + b(h)

so `h` influences the ranking **only** through an 11-dimensional coefficient
vector. Measured: that vector rotates 2.04° across a full run's state change.
A more expressive head has no such bottleneck:

    score_k = MLP([h ; cf_k])           # use_linear_score_head=False

`h` enters jointly and non-linearly with each candidate. **The flag already
exists** (`decisionTransformer.py:434, 665`); nothing new is being built.

## Design (one variable: the score parameterisation)

| arm | `use_linear_score_head` | role |
|---|---|---|
| L | True | current default; must reproduce 0/12 |
| **M** | **False** | MLP over `[h ; cf]` — the candidate |

Both arms: identical seed, identical training, identical candidate pools.

Probes (both use the **corrected** state swap — comparison states drawn from
*different* ensemble-model blocks, since H5's original swap compared a state
with itself):

1. **state swap**: does the argmax change when `h` comes from a genuinely
   different state?
2. **RTG sweep** in the realised band [0.5, 1.0].
3. **attenuation**: relative spread of `s`, `h`, and the score vector.

## Manipulation check (first)

Assert `dt.use_linear_score_head is False` in arm M **and** that arm M's score
vector is not an affine function of `cf` — verified by checking that arm M's
scores cannot be reproduced by any single coefficient vector (residual of a
least-squares fit of `score ≈ cf @ w + b` is materially non-zero). If arm M is
secretly still linear in `cf`, the comparison is void.

## Locked predictions

1. **PRIMARY**: arm M moves the argmax on **> 30%** of pools under a genuine
   state swap, vs arm L's 0/12.
2. **NULL — the outcome that finishes the paper**: if arm M is also ~0%, the
   conditioning failure is **architecture-independent within this family**. The
   linear bottleneck is then not the cause, the negative result is materially
   stronger, and the empirical programme is complete.

## Scope

Probe only. No regret run, `PROTOCOL.md` untouched. If arm M *does* condition, a
frozen-evaluation run on it becomes the next experiment — and the paper's claim
changes — which is exactly why this is being run **before** invoking the paper
skill rather than after.
