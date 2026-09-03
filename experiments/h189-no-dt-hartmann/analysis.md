# h189 — **P3. The opposite of Borehole, also at 5/5. The DT's value FLIPS SIGN between benchmarks.**

**CONFIRMATORY** against the protocol committed before launch, readout committed before
all runs finished, SC observation recorded before the regret. 5/5 runs, no failures.
h187's worker reused **unchanged**, so no code differs between the two benchmarks.

## lf_fraction first, as registered

| seed | 42 | 43 | 44 | 45 | 46 | mean |
|---|---|---|---|---|---|---|
| teacher `lf_fraction` | 0.989 | 0.368 | **0.000** | 0.934 | 0.924 | 0.643 |

The bistability recorded **before** the regret was read is confirmed: two seeds go
almost pure high-fidelity, three almost pure low-fidelity. MF-DRO's Hartmann control
sits stably at **0.800**.

## The registered statistic

> paired (teacher-only − MF-DRO) frozen rel%. Threshold 0.87 rel% points.
> **P1 |diff| ≤ 0.87 · P2 diff < −0.87 (generalises) · P3 diff > +0.87 (Borehole-specific)**

| seed | teacher lf | teacher evals | teacher rel% | MF-DRO rel% | gap |
|---|---|---|---|---|---|
| 42 | 0.989 | 186 | 47.09 | 16.41 | **+30.68** |
| 43 | 0.368 | 38 | 16.34 | 0.67 | **+15.67** |
| 44 | 0.000 | 25 | 16.41 | 10.16 | **+6.26** |
| 45 | 0.934 | 137 | 15.54 | 5.28 | **+10.26** |
| 46 | 0.924 | 131 | 14.02 | 7.42 | **+6.60** |
| **mean** | 0.643 | | **21.88** | **7.99** | **+13.89** (se 4.53) |

**MF-DRO better on 5/5 seeds. +13.89 → P3.**

## The headline: the sign flips, and both directions are unanimous

| benchmark | cost ratio | teacher-only | MF-DRO | paired | winner |
|---|---|---|---|---|---|
| **Borehole** | 2:1 | **12.97** | 15.82 | **−2.85** | teacher, **5/5** |
| **Hartmann** | 8:1 | 21.88 | **7.99** | **+13.89** | MF-DRO, **5/5** |

**The two benchmarks give exactly opposite answers, each unanimous across seeds.** The
Hartmann advantage (+13.89) is also nearly **5× larger** than the Borehole deficit.

**h31 is vindicated, not explained away.** It found MF-DRO ahead on Hartmann on 7/10
seeds using final simple regret with unmatched fidelity mixes. On the frozen metric
with the identical code path, MF-DRO is ahead **5/5**. My P2 branch would have
attributed h31's result to its metric; **P3 fired instead, and h31's direction was
right.**

## What this RETRACTS

**"The DT is a net negative" must never be stated unqualified.** h187's finding is
**Borehole-specific**, exactly as its scoping paragraph anticipated. And the
synthesis's third leg — "the averaging is about as good as running the teacher" — is
wrong in *both* directions, not just one: it is a **cost** on Borehole and a **large
gain** on Hartmann.

**Corrected:** *the DT reproduces the mean of its teacher's first move; the value of
that averaging is benchmark-dependent and flips sign — −2.85 on Borehole, +13.89 on
Hartmann.*

h185, h186, h188 and h182 are untouched — they describe *what* the DT does, not what
it is worth.

## A candidate account, explicitly NOT established

The benchmarks differ 4× in cost ratio (Hartmann 8:1, Borehole 2:1). On Hartmann a
fidelity mistake costs eight cheap evaluations; on Borehole it costs two. The teacher's
fidelity choice is **bistable on Hartmann** (0.000–0.989) while MF-DRO's is **stable at
0.800**. Stability in allocation would be worth much more where mistakes are expensive.

**This is a hypothesis and the data here cannot support it.** Within these five seeds
ρ(teacher `lf_fraction`, gap) = +0.700 and ρ(evaluations, gap) = +0.700 — but those two
are nearly collinear (more LF ⇒ more evaluations), so it is one correlation, over five
points, and is a *description*, not evidence. It is recorded as the next thing to test,
not as an explanation.

## What could RETRACT this result

- Seed 42 contributes +30.68 of the five gaps and the se is 4.53. Dropping it leaves
  +9.70, still P3 and still 4/4 — the verdict does not hinge on it, but the magnitude
  does.
- One teacher (MES), one metric, n=5 per benchmark.
- The teacher gets pool + argmax over 200 candidates on both benchmarks, so the
  asymmetry in *that* respect is constant and cannot explain a sign flip.
