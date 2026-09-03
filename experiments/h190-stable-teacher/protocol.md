# h190 — is MF-DRO's Hartmann advantage bought by FIDELITY STABILITY?

**CONFIRMATORY.** Committed before the arm is built and before any result exists.

## Why

h189 found MF-DRO beats teacher-only on Hartmann by **+13.89 on 5/5 seeds**, the exact
opposite of Borehole (−2.85, 5/5). Its analysis recorded a candidate account and
explicitly marked it **not established**:

> The teacher's fidelity choice is **bistable on Hartmann** (`lf_fraction` = 0.989,
> 0.368, **0.000**, 0.934, 0.924) while MF-DRO's is **stable at 0.800**. Hartmann prices
> an HF query at 8 cheap ones (Borehole: 2), so stable allocation should be worth more
> where mistakes cost more.

This arm tests it directly: **give the teacher MF-DRO's stable fidelity allocation and
see whether its disadvantage shrinks.**

## The arm

h187's worker (unchanged mechanism — teacher decides, DT bypassed) **plus**
`max_hf_fraction = 0.20`, the HF ceiling built and identity-gated in h184. That forces
`lf_fraction → ~0.80`, matching MF-DRO's Hartmann control. Hartmann, seeds 42–46,
frozen metric, against the same h83 MF-DRO control (7.99).

## The gate — and why it is NOT on the paired mean

h189's gap has **se 4.53**. A second arm carries a similar se, so the se on a
*difference of gaps* is ≈ **6.40**. Detecting even a **halving** (13.89 → 6.95, a shift
of 6.94) would be **1.08 se**. **A gate on the paired mean would be unreadable at n=5**,
and registering one anyway would produce an unfalsifiable result. Computed before
launch, not after.

> **statistic**: number of Hartmann seeds (0–5) where the STABLE teacher beats MF-DRO.
> Currently **0/5**.
>
> - **P1 — stability was the main cause**: **≥ 3**
> - **P2 — partial**: **1 or 2**
> - **P3 — stability was not the cause**: **0**

Partitions the integers 0–5 with no gap or overlap.

**Applying h184's lesson explicitly:** a sign-count gate can fire on an effect far
smaller than the one it models. So the **paired mean is reported alongside as a
secondary**, with the stated caveat that it cannot resolve a halving. If the sign count
moves while the mean does not, that is reported as *direction without magnitude*, as
h184 was.

## SC before the regret

Realised `lf_fraction` must be **≈0.80 on every seed**, replacing h189's 0.000–0.989
spread. **If the ceiling does not stabilise it, the arm tests nothing** and the regret
must not be read.

## What this could RETRACT

- **P3 fires → the stability account is REFUSED.** MF-DRO's Hartmann advantage would
  then have no proposed source, leaving *two* unexplained things (the benchmark
  asymmetry and the sign flip). This is the likelier outcome to have to report: it is
  the fourth mechanism I would have proposed on this project and three of the previous
  ones were refused by direct measurement.
- **P1 fires → a real mechanism for the sign flip**, and it would connect to h183/h184
  (fidelity matters on Hartmann, not Borehole) and to the 8:1 vs 2:1 cost ratios.
- Nothing here can overturn h189 itself — that comparison is committed and stands.

## Compute

5 workers × 1 thread. Machine idle.
