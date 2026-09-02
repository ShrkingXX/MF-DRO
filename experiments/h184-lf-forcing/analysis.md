# h184 — **P1 fires** on the registered statistic. Supported in DIRECTION, not in MAGNITUDE.

**CONFIRMATORY** against the protocol committed before launch, with the readout script
(`code/readout.py`) committed before the CTRL arm landed. 10/10 runs, no failures.

## SC read first, as registered

| arm | realised `lf_fraction` |
|---|---|
| LFF-CTRL | 0.75, 0.75, 0.75, 0.75, 0.75 |
| LFF-HEAD | 0.75, 0.75, 0.75, 0.75, 0.75 |

Unforced Borehole runs at **0.117**. The HF ceiling held its 25% target **exactly, on
every seed of both arms** — symmetric forcing, so the gap between arms is not
confounded by the intervention. **SC PASS.**

## The registered statistic

> number of Borehole seeds (0–5) where LF-forced HEAD is worse than LF-forced CTRL.
> **P1 ≥ 4 · P2 = 2 or 3 · P3 ≤ 1**

| | frozen rel% CTRL | frozen rel% HEAD | paired mean | seeds worse |
|---|---|---|---|---|
| unforced Borehole | 15.82 | 16.96 | +1.15 (se 1.34) | **2/5** |
| **LF-forced Borehole (h184)** | **15.76** | **17.35** | **+1.60 (se 0.79)** | **4/5** |
| Hartmann | 7.99 | 25.16 | +17.17 (se 11.10) | **5/5** |

Per-seed paired gap: −0.09, +1.12, +4.34, +0.41, +2.20.

**4/5 → P1. The `lf_fraction` account is SUPPORTED on its registered statistic.**
Forcing Borehole into Hartmann's fidelity mix moved the sign pattern from 2/5 to 4/5,
toward Hartmann's 5/5, exactly as the account required.

## The part the verdict line does not say, and which matters more

**The magnitude did not become Hartmann-like at all.**

- The paired mean moved from **+1.15 to +1.60** — a change of **+0.45**, which is
  **0.33 se** of the unforced gap. Not resolvable at n=5.
- Hartmann's gap is **+17.17**. Borehole's forced gap is **9% of the way** there.
- Two of the four "worse" seeds are worse by only 0.41 and 1.12, and the fifth is
  effectively tied at −0.09. The 4/5 count is real but it sits on small margins.

So the intervention moved the **direction** of the effect as predicted and left its
**size** essentially where it was. Reporting "P1, account supported" without this
would overstate the result substantially.

### On the gate design, honestly

The gate was placed on the sign pattern **because** Hartmann's mean is unresolved
(se 11.10 on a mean of 17.17, one seed at 60.98). That was the right call with the
information available, and it is why the gate was registered before launch. The cost
of that choice is now visible: **a sign-pattern gate can fire on an effect a tenth the
size of the one it is modelled on.** A magnitude clause registered alongside it would
have made this result sharper in both directions. Worth carrying forward.

## Standing

This is the **first of three** accounts of the benchmark asymmetry not refused by its
own numbers — the escape-fraction account and the geometric account were both refuted
by direct measurement. It is supported in direction, unsupported in magnitude, and
therefore **partial**: the fidelity operating point is *a* contributor to the
asymmetry, and cannot be the whole of it, since replicating Hartmann's fidelity mix
reproduces under a tenth of Hartmann's effect.

**The benchmark asymmetry is NOT closed.**

## What could still RETRACT it

- The 4/5 count rests on two margins under 1.2 rel% points. A repeat at fresh seeds
  could plausibly return 3/5, which is P2 — refusal.
- Only HEAD-vs-CTRL was forced. Whether forcing changes the *other* arms' orderings
  (TAIL, ORACLE, RANDOM) is untested and would test the account far more broadly.
- Forcing HF down necessarily costs Borehole absolute performance; CTRL at 15.76 is
  coincidentally close to the unforced 15.82, but that is not something the design
  guaranteed and should not be read as the intervention being free.
