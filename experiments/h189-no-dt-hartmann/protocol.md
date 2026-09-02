# h189 — h187 on Hartmann. Is the DT's net-negative result benchmark-specific?

**CONFIRMATORY.** Committed before launch, before any result exists.

## Why

h187 found the DT is a **net negative** against its own teacher on Borehole under the
frozen metric: 12.97 vs 15.82, **5/5 seeds**, paired −2.85 (se 1.30), with the
fidelity-allocation explanation ruled out by h184's control.

**h31 found the opposite on Hartmann** — MF-DRO ahead on 7/10 seeds — but on *final
simple regret*, not the frozen metric, and with the two arms at unmatched fidelity
mixes. The two results cannot be compared as they stand.

This arm runs h187's exact mechanism on Hartmann so the two benchmarks are measured
the same way. It reuses `experiments/h187-no-dt-frozen/code/worker.py` unchanged —
that worker already takes the benchmark as an argument, so **no new code is written**
and nothing can drift between the two arms.

## Gate

Same statistic and threshold construction as h187: paired (teacher-only − MF-DRO)
frozen rel% points, threshold = the 10.9% worst-case harness noise floor on the
Hartmann MF-DRO control's 7.99 base = **0.87 rel% points**.

- **P1 competitive**: |diff| ≤ 0.87
- **P2 teacher BETTER**: diff < −0.87 → the net-negative result **generalises**
- **P3 teacher WORSE**: diff > +0.87 → the net-negative result is **Borehole-specific**,
  and h31's direction is confirmed on the frozen metric

## What this could RETRACT

- **P3 fires → h187's finding is scoped to Borehole**, and the claim "the DT is a net
  negative" must never be stated without that qualifier. h31's 7/10 would then be
  vindicated rather than explained away.
- **P2 fires → the net-negative result generalises across two benchmarks**, and h31's
  opposite finding is attributable to its metric and unmatched fidelity mixes. This is
  the more damaging outcome for the method and must be reported plainly.
- P1 leaves Hartmann genuinely undecided.

Note the asymmetry in what is at stake: **h187 is already committed and reported.**
This arm can only widen or narrow its scope, not undo it.

## Compute

5 workers × 1 thread. Machine is otherwise idle (h184 and h187 both complete).

## Where the results land — recorded so the readout is not confused

Reusing h187's worker unchanged has one consequence: `RES` is hardcoded relative to
**h187's** code directory, so h189's Hartmann runs write to

```
experiments/h187-no-dt-frozen/results/Hartmann_6D__NODT__seed4*.json
experiments/h187-no-dt-frozen/results/ckpt/Hartmann_6D__NODT__seed4*.json
```

not to `h189-no-dt-hartmann/results/`. Only the run **logs** are in h189's directory.

This was noticed at launch and is left as-is deliberately: the alternative is editing
the worker, which would break the guarantee that h187 and h189 run *identical* code.
The `Hartmann_6D__NODT__` tag is unambiguous, so nothing is lost but tidiness.

## SC OBSERVATION, recorded BEFORE the regret is read

The teacher's realised `lf_fraction` on Hartmann, across the five seeds:

| seed | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| `lf_fraction` | **0.981** | 0.368 | **0.000** | 0.931 | 0.922 |

**The same acquisition rule, on five seeds of the same benchmark, spans essentially the
entire range** — two seeds go almost pure high-fidelity (0.000, 0.368) and three go
almost pure low-fidelity (0.92–0.98). For contrast, the same teacher on Borehole stayed
in 0.291–0.561, and Hartmann's MF-DRO control runs at 0.800.

**Two consequences, both stated before the outcome is known:**

1. **The teacher-only arm on Hartmann is not one strategy but five very different
   ones.** A 5-seed mean over a bimodal spread like this will be noisy, and the paired
   comparison may well be unresolvable at the registered 0.87 threshold. If the verdict
   comes back P1 ("competitive"), that should be read as *undecided*, not as evidence
   of equivalence.
2. **This is an independent finding about the method**, not just a nuisance: MES's
   cost-normalised fidelity criterion is **bistable on Hartmann** in a way it is not on
   Borehole. It is worth recording whichever way the regret falls, and it may bear on
   the benchmark asymmetry — Hartmann is where fidelity choice was already shown to
   matter (h183: fit quality predicts regret there, `lf_fraction` 0.80 vs 0.12).
