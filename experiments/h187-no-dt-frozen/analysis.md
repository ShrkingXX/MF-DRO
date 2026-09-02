# h187 — **P2. On Borehole under the frozen metric, the DT is a NET NEGATIVE against its own teacher.**

**CONFIRMATORY** against the protocol committed before launch, with SC1 verified
before launch and the readout script committed before any run finished. 5/5 runs, no
failures. This is the outcome the protocol named as *"most damaging to the project"*
and committed to reporting plainly.

## lf_fraction first, as registered

| arm | realised `lf_fraction` |
|---|---|
| teacher-only | 0.400, 0.291, 0.413, 0.511, 0.561 (mean **0.435**) |
| MF-DRO control | **0.117** |

**A correction to the protocol's own prediction.** It warned this arm might run at
`lf_fraction ≈ 0`, based on SC1's first six iterations where the teacher chose HF
every time. Over full runs it chooses LF **43.5%** of the time. The prediction was
made on six samples and was wrong.

## The registered statistic

> paired (teacher-only − MF-DRO) frozen rel% points. Threshold = the pre-existing
> 10.9% worst-case harness noise floor on a 15.82 base = **1.72 rel% points**.
> **P1 |diff| ≤ 1.72 · P2 diff < −1.72 · P3 diff > +1.72**

| | frozen rel% |
|---|---|
| **teacher-only, no DT** | **12.97** |
| MF-DRO | 15.82 |

Paired per seed: **−1.23, −4.28, −0.12, −7.24, −1.37**
Mean **−2.85** (se 1.30). **Teacher better on 5/5 seeds.**

**−2.85 < −1.72 → P2.** Removing the Decision Transformer entirely and letting the
acquisition rule choose directly **improves** the frozen metric by 2.85 rel% points,
on every seed.

## The obvious alternative explanation is ruled out — by an arm already run

The teacher spends far more on the cheap source (0.435 vs 0.117), so the natural
objection is that it wins on **fidelity allocation**, not on decisions. h184 already
forced MF-DRO to a high LF share and supplies the control for free:

| arm | `lf_fraction` | frozen rel% |
|---|---|---|
| MF-DRO unforced | 0.117 | 15.82 |
| **MF-DRO LF-forced (h184)** | **0.750** | **15.76** |
| teacher-only | 0.435 | **12.97** |

**A 6.4× change in MF-DRO's LF share moves it 0.06 rel% points.** MF-DRO does not
improve when given the teacher's fidelity mix, or far more of it. **The teacher's
advantage is not a fidelity-allocation effect.**

## What this RETRACTS

**The synthesis's third leg falls.** findings.md and the published report state that
the DT's averaging is *"about as good as running the teacher"*, resting on **h31**
(Hartmann, final simple regret, unmatched fidelity mixes: MF-DRO better on 7/10, its
own reading "not resolved"). On Borehole, on the frozen metric, with the fidelity
objection controlled, **it is worse on 5/5**.

The corrected synthesis:

> The DT reproduces the mean of its teacher's first move, and on Borehole under the
> frozen metric **that average is worse than running the teacher**.

Everything else in the account stands — h185 (per-timestep constant), h186 (ignores
its inputs), h188 (the constant *is* the teacher's τ=0 mean), h182 (collapse to the
centre). What changes is the value of the averaging: it is not a wash, it is a cost.

## Caveats, stated because the result is unflattering and must not be overstated either

- **The teacher gets pool + argmax over 200 candidates; MF-DRO emits a point
  directly.** That asymmetry *is* the comparison — it is what replacing search with a
  learned policy means — but the honest phrasing is "the DT is a net negative against
  an acquisition rule performing explicit search", not "against something equivalent".
- **h31's opposite result is unreconciled.** It found MF-DRO ahead on Hartmann on
  7/10 seeds. That is a different benchmark *and* a different metric. **A Hartmann
  h187 is the obvious follow-up** and is registered below.
- n=5, one benchmark, one teacher (MES).
- −2.85 is 2.2 se; at n=5 the 5/5 sign pattern is the stronger part of the evidence.

## Registered follow-up

**h189: h187 on Hartmann**, same mechanism, frozen metric, seeds 42–46 — to determine
whether P2 is Borehole-specific or whether h31's 7/10 was an artifact of its metric
and unmatched fidelity mixes. Until that runs, the claim is scoped to **Borehole**.
