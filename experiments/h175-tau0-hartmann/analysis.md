# h175 — **P1 FAILS. R1 fires.** The mechanism is equally strong on both benchmarks.

CONFIRMATORY, 40 states, 120 draws. P2 checked first, as registered.

| Hartmann arm | d(q, τ0 mean) | d(q, centre) | d(q, random) | τ0 → centre | est. SE |
|---|---|---|---|---|---|
| control (works) | **0.1943** | 0.6521 | 0.9558 | 0.5329 | 0.0364 |
| h165 UCB-LOC (works) | **0.1683** | 0.6354 | 0.9797 | 0.5518 | 0.0355 |
| RANDOM-POOL (fails) | 0.1225 | 0.1048 | 0.6872 | **0.0605** | 0.0632 |
| ORACLE (fails) | 0.1569 | 0.1461 | 0.7235 | **0.0594** | 0.0633 |

**P2 calibrates** (0.0605, 0.0594 against a pre-computed 6D floor of 0.062 and a
threshold of 0.15 — the h170 mistake of setting the threshold *below* the floor is
not repeated). **P3 holds 4/4.**

## P1 fails, and in the opposite direction

The tightness ratio `d(q, τ0 mean) / d(q, centre)`:

```
Borehole (h170)   0.306   0.309
Hartmann (h175)   0.298   0.265
```

P1 predicted Hartmann would be **materially larger** — a weaker mechanism, leaving
room for the later steps to matter. It is slightly **smaller**. The τ=0 mean
determines the query at least as tightly on Hartmann as on Borehole.

## What this means, stated as a gap rather than an account

The strong form ("only the first step matters") holds on Borehole and fails on
Hartmann. **That difference is NOT explained by the τ=0 mechanism being weaker
there** — it isn't weaker. So the query lands in the same relation to its
teacher's first-step mean on both benchmarks, yet on one that suffices to match
the control and on the other it does not.

**I am recording this as unexplained rather than proposing a seventh account.**
Six have been proposed on this front and five fell, every one of them by fitting
the evidence available at the time. The available evidence here would support a
story about Hartmann's sharper optimum needing refinement that Borehole's does
not — and that is exactly the kind of story that has failed five times. It is not
being written down as a finding.

## One thing that did replicate

The residual is the same size on both benchmarks in SE units: **5.0 and 5.0 SE on
Borehole, 5.3 and 4.7 on Hartmann**. So the τ=0 account's known incompleteness is
a stable feature of the mechanism, not a Borehole artefact — which is worth
knowing, and is the opposite of what a benchmark-specific quirk would look like.
