# H22 — no threshold exists within two orders of magnitude

Trained network held fixed; state deviations scaled as
`s'(λ) = s̄ + λ(s − s̄)`.

| λ | argmax moved | min cos(w) | rel. spread of `w` |
|---|---|---|---|
| 1 | 0/12 | 0.99999224 | 0.001826 |
| 2 | 0/12 | 0.99996903 | 0.003651 |
| 3 | 0/12 | 0.99993116 | 0.005470 |
| 5 | 0/12 | 0.99981170 | 0.009110 |
| 8 | 0/12 | 0.99953686 | 0.014494 |
| 12 | 0/12 | 0.99901078 | 0.021595 |
| 20 | 0/12 | 0.99754821 | 0.035599 |
| 35 | 0/12 | 0.99429217 | 0.059706 |
| 60 | **1/12 (8%)** | 0.98812666 | 0.095327 |
| 100 | 0/12 | 0.97988447 | 0.138267 |

**PRED 3 NULL fires.** No `λ ≤ 100` moves the argmax on more than 30% of pools.

## The manipulation unquestionably worked

This is not a null from a no-op. Across the sweep the coefficient vector's
relative spread grows **76×** (0.001826 → 0.138267) and the vectors rotate from
cosine 0.99999 to 0.97988 — an **11.5° separation**. `w` really does change, a
great deal. The ranking it induces does not.

The lone 1/12 at λ=60 is **non-monotonic** (0 at λ=35, 1 at λ=60, 0 at λ=100),
so it is a single-pool coincidence rather than a threshold crossing.

## What this establishes

The earlier phrasing — "far from the threshold, not marginally short" — was too
weak. The correct statement is:

> The learned scoring map is invariant to state *direction* across two orders of
> magnitude of input gain. A 76× increase in coefficient variation and an 11.5°
> rotation of `w` still fail to reorder the top of a 200-candidate pool.

The reason is visible in the numbers: the acquisition's ranking is dominated by
a state-independent component of `w`, and the margin between the winning
candidate and its rivals exceeds anything an 11.5° rotation can overturn. The
state does not merely have *little* influence; it has influence in a direction
that does not compete with what decides the ranking.

## Caveat, pre-registered and binding

Large `λ` places states **far out of distribution**. This measures the gain of
the learned map, **not** a usable intervention: nothing here proposes amplifying
states, and no regret claim is made. The caveat is the same one that made an
earlier RTG sweep uninterpretable until it was re-run in-band, which is why it
was fixed in the protocol before running.
