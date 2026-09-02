# h184 — does Borehole's HIGH-FIDELITY operating point cause the asymmetry?

**CONFIRMATORY.** Committed before any code is written and before any result exists.

## Why

h183 found that the DT's fit quality predicts regret on Hartmann (ρ = +0.53 fidelity
loss, +0.61 location loss over 18 arms) and not on Borehole (−0.24, +0.16 over 25),
and identified the candidate cause: the two benchmarks operate at completely
different fidelity mixes.

| | `lf_fraction`, control | `lf_fraction`, all arms |
|---|---|---|
| Borehole | **0.117** | 0.229 |
| Hartmann | **0.800** | 0.711 |

Borehole's policy is high-fidelity by ~7×. If that is *why* only the first step
matters there, then **forcing Borehole to a Hartmann-like fidelity mix should make
Borehole behave like Hartmann** — and the strong form should break.

h183's own limits section names this as the experiment it calls for. It is the
difference between a supporting correlation and a demonstrated cause.

## The intervention

Add `max_hf_fraction` to the real-query fidelity decision: when the run's realised
HF fraction is at or above the target, force the next real query to LF. It mirrors
the existing `real_hf_every` floor (mf_dro.py:3804) in structure and is **disabled
by default**, so no existing configuration can change behaviour.

Target `max_hf_fraction = 0.25`, taking Borehole from ~0.88 HF to ~0.25 HF, i.e.
into Hartmann's operating range (Hartmann control runs at 0.20 HF).

Two Borehole arms, seeds 42–46: **LFF-CTRL** (control + forcing) and **LFF-HEAD**
(HEAD-MES + forcing). Both arms get the identical intervention; the comparison is
the *gap between them*, so the forcing itself cancels.

## The statistic and the gate

Calibration of the existing, un-forced gaps (paired, same seeds):

| | paired HEAD−control | sd | se | seeds where HEAD is worse |
|---|---|---|---|---|
| Borehole | +1.15 | 3.01 | 1.35 | **2/5** |
| Hartmann | +17.18 | **24.81** | **11.09** | **5/5** |

**The Hartmann mean is not well resolved** — se 11.09, driven by one seed at 60.98
(median 6.26). The robust contrast is the **sign pattern**, so the gate is on that:

> **statistic**: number of Borehole seeds (0–5) where LF-forced HEAD is worse than
> LF-forced control.

- **P1 — Hartmann-like, account SUPPORTED**: **≥4**
- **P2 — unchanged, account REFUSED**: **2 or 3** (Borehole is at 2 now)
- **P3 — inverted**: **≤1**

Verified to partition the integers 0–5 with no overlap and no gap
(`tools/check_gate.py --int`, plus an explicit 0–5 coverage check).

## What this could RETRACT

- **P2 refuses h183's causal candidate.** h183 downgrades from "a supported account"
  to "a characterization of the asymmetry with no established cause", and the
  asymmetry will have survived **three** explanations of mine — the escape-fraction
  account, the geometric account, and this one. That is the outcome I should expect
  to have to report.
- **P3** would mean forcing LF *helps* HEAD relative to control on Borehole, which
  no current account predicts and would need its own investigation.
- Absolute regret will almost certainly worsen in both arms — Borehole needs HF
  observations. That is expected and is **not** the readout; the gap is.

## Prerequisite gate before launch

`use_roi=False` must remain **bit-identical** (regret 122.2906675273). The arm does
not launch until that passes on the modified file.

## Compute

2 arms × 5 seeds = 10 workers × 1 thread ≤ 15. Launched only after h181's five
workers exit.

## Prerequisite gate: PASSED (recorded before launch)

```
tools/identity_gate.py
  reference : 122.29066752728207
  this build: 122.29066752728207
  IDENTITY GATE: PASS
```

Verified **exactly**, not to a tolerance: the pre-patch file was checked out
(`git stash`) and run, giving `122.29066752728207`; the patched file gives the
same value to full precision.

Note for future gates: findings.md records this reference as `122.2906675273`,
which is the same number shown to 13 significant figures. Comparing against that
rounded value fails a genuinely identical build by ~1.8e-11 — which is exactly
what happened on the first attempt here. `tools/identity_gate.py` now carries the
full-precision value and compares exactly.

## SC1 (does the ceiling actually fire?) — PASS, recorded before launch

A silent no-op would look exactly like a null, which is how h169 was lost. Smoke
run, Borehole seed 42, control arm, short budget:

```
HF ceiling override fired on 5 iterations
realised fidelity split: 5 LF / 2 HF  ->  lf_fraction ~0.71
unforced Borehole control:                lf_fraction  0.117
```

The intervention moves Borehole from ~0.12 LF into Hartmann's operating range
(~0.71–0.80 LF), which is what the arm requires.
