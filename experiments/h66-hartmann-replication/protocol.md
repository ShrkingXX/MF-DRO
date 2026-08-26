# H66 — replicate the Hartmann result at n=10

**CONFIRMATORY. Protocol committed before any run.**

## Why

h64 produced this project's first arrival at the north star: on Hartmann 6D,
MF-DRO with `n_roi_candidates=600` reached **7.6%** relative regret against
MF-MES's **8.5%**, the best baseline — 3/3 over BASE, **2/3 over MF-MES**.

**2 of 3 paired wins is not a result.** With n=3 the probability of 2-or-better
under a coin flip is 0.5, and this project has already retracted a headline built
on an under-powered subset (lesson 19, the h45 5/6-then-7/8 sequence that
reversed at 10/10). The claim "a DRO variant beats the best baseline" is the most
consequential thing measured here and it currently rests on three seeds.

The result is also **mechanistically unexplained**: h64 pre-registered a NULL for
Hartmann because widening 200 -> 600 buys it **1.00x** acquisition value, and the
gain happened anyway. An unexplained effect at n=3 is exactly the profile that
should be replicated before it is built on.

## Design

Hartmann 6D, **seeds 42-51 (n=10)**, cost budget 200 post-init, everything else
identical to h57's MF-DRO.

| arm | change | seeds needed |
|---|---|---|
| **POOL600** | `n_roi_candidates=600` | 42,43,45,47,49,50,51 (44/46/48 reused from h64) |
| **MF-MES** | h48's standalone Takeno MF-MES | 42,43,45,47,49,50,51 (44/46/48 reused from h57) |
| BASE | none | reused from h57 where available |

17 new jobs. Existing cells are reused, not re-run — policy code verified
byte-identical at the h61 regression gate, commit hash recorded per result.

## Locked predictions

1. **PRIMARY**: POOL600's mean relative regret on Hartmann is **below MF-MES's**
   at n=10, and it wins **>= 6 of 10** paired seeds.
2. **REPLICATION FAILURE**: POOL600 wins <= 5 of 10. Then h64's 2/3 was noise,
   the north-star arrival is **withdrawn**, and that withdrawal is reported as
   prominently as the original claim.
3. **VARIANCE**: POOL600's seed sd is below BASE's 0.2395 at n=10. h64 measured
   0.0604 at n=3; a 3-point sd is a crude statistic and this is the check.
4. **Statistics**: at n=10 a Wilcoxon signed-rank test IS reportable and will be
   reported **whatever it shows** — including if it fails to reach significance
   while the win count looks favourable. h17-vs-h31 needed 82 seeds for 80%
   power on a smaller effect, so a non-significant result at n=10 is expected
   and is not a refutation on its own; the win count and direction carry the
   claim.

## What this cannot settle

One benchmark. Borehole remains at 19.5% against MI-Greedy's 8.3% and is not
addressed here. The mechanism stays unexplained regardless of outcome — this
tests whether the effect is real, not why.
