# H28 — the uncertainty aversion is INHERITED. My "inversion" claim is retracted.

1600 teacher decisions from a real 200-trajectory rollout batch.

| quantity | value |
|---|---|
| **CONTROL** — chosen `mu_H` percentile within its pool | **94.2% ± 0.1%** ✓ |
| **PRIMARY** — chosen `sigma_H` percentile within its pool | **2.9% ± 0.2%** |
| Spearman(teacher score, `sigma_H`) | **+0.1585** |
| Spearman(teacher score, `mu_H`) | **+0.8517** |

The control passes decisively, so the extraction is sound.

## The teacher picks the bottom 3% of uncertainty

**PRED 1 fires.** The MF-MES teacher's realised choices sit at the **2.9th
percentile** of `sigma_H` within their own candidate pools. The student's
negative `w[sigma_H]` is therefore **faithful imitation**, not corruption.

## RETRACTION

Two ticks ago I wrote, and put in `paper/main.tex`:

> "trained on rollouts from an information-seeking MF-MES teacher, it converges
> to a fixed rule that inverts the sign of its teacher's defining term."

**That is wrong.** The student inverts nothing. It reproduces what its teacher
actually did, on 97% of decisions. The claim is retracted from the paper and
from `findings.md`.

## The real mechanism, which is more interesting

The teacher's *scoring function* mildly rewards uncertainty
(Spearman `+0.1585` with `sigma_H`) — MES genuinely is an information-seeking
acquisition. But its *argmax* is overwhelmingly driven by the posterior mean
(Spearman `+0.8517` with `mu_H`), and in a GP the high-`mu` regions are precisely
where data already sits, hence low `sigma`. The mild uncertainty bonus never
survives the mean term.

So **MF-MES's realised behaviour on this benchmark is exploitative even though
its scoring rule is not.** A student trained by imitating *choices* rather than
*scores* can only learn the exploitative part.

This relocates the fault. It is not the distillation and not the transformer:
**a better student cannot fix an exploitative teacher.** Any remedy has to change
what the teacher selects — cost ratio, `y*` sampling, or the acquisition itself —
not how faithfully the student copies it.

## Why the earlier claim was tempting and wrong

"Information-seeking teacher, exploitative student" is a clean story, and the
teacher's *definition* supports the first half. Checking what the teacher
actually *did* took one probe and reversed it. The lesson repeats: a component's
specification is not its behaviour.
