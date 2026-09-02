# h163 — **R2 fires decisively.** Teacher and student dispersion are INVERTED.

CONFIRMATORY. Prediction locked before the teacher-side numbers were computed.

| teacher rule | TEACHER M1 | STUDENT M1 (h162) |
|---|---|---|
| MES (control / h153) | 0.6730 | 0.2766 |
| UCB β=2 (h155) | 0.7443 | 0.2889 |
| ORACLE | 0.8445 | 0.1891 |
| DIVERSE-GOOD | 0.8818 | 0.1830 |
| **RANDOM** | **1.1295** | **0.1115** |

**Spearman(teacher, student) = −0.900** over 5 arms.

The prediction was NEGATIVE; the natural null — "the student inherits its
teacher's spread" — predicts POSITIVE. The observed sign is the opposite of the
null, and strongly so. R1 does not fire.

The extreme case is the clearest: RANDOM-POOL's teacher is the **most** dispersed
of all six (uniform draws, M1 1.13) and its student is the **least** dispersed of
all six (0.111). A network imitating its teacher cannot do that. A network
falling back on the mean of an unlearnable target does exactly that, and the more
scattered the target, the flatter the fallback.

## An OUT-OF-SAMPLE prediction, recorded before it is checked

h159 (UCB β=0) was excluded from the correlation above because its student M1 had
not been computed. Its **teacher** M1 is **0.3624 — the lowest of all six rules**,
which under the inversion means its student should be the **most dispersed of any
arm, above h155's 0.2889**.

This is a genuine out-of-sample test: the relationship was fitted on five arms,
h159 was not among them, and its student number is computed only after this
paragraph is committed.

If h159's student M1 comes in **below ~0.24** (i.e. in the failing arms' range or
merely mid-pack), the inversion does not extrapolate and is a within-sample
pattern rather than a relationship. That would be recorded as a partial
retraction of this analysis.

## Caveat, unchanged

Query-level statistics at n=5 — the evidence class that produced h150
(retracted) and h154's refuted M2 direction. The inversion is worth weight
because its sign is opposite to the natural null and because it now has an
out-of-sample test, not because the evidence class has improved.

---

## Out-of-sample result: **PARTIAL**. The point prediction failed; the threshold held.

h159's student M1 = **0.2639** (per-seed 0.254, 0.394, 0.243, 0.166, 0.263).

- Predicted: **highest of any arm**, above h155's 0.2889 → **FAILED**. It is
  third of seven, below h155 (0.2889) and the control (0.2766).
- Retraction threshold (below ~0.24 = does not extrapolate): **NOT crossed**.
  0.2639 sits in the working-arm band.

Spearman falls from **−0.900** (5 arms) to **−0.771** with h159 included. Still
strongly negative, but the rank relationship is looser than the 5-arm fit
suggested, and h159 is the arm that breaks it: the lowest teacher dispersion of
all six (0.3624) should have produced the highest student dispersion and did not.

## The honest restatement

**The inversion holds as a GROUP SEPARATION, not as a rank relationship.**

```
working arms:  h155 0.2889 | control 0.2766 | h159 0.2639 | h153 0.2464
failing arms:  ORACLE 0.1891 | DIVERSE 0.1830 | RANDOM 0.1115
```
Complete arm-level separation, now with **four** working arms against three
failing ones, and a clear gap (0.2464 vs 0.1891). But *within* the working
group the ordering does not follow teacher dispersion, so the mechanism does not
support a graded prediction — only a categorical one.

That is weaker than what the 5-arm Spearman implied, and the out-of-sample test
is what showed it. The claim is downgraded accordingly: dispersion collapse
distinguishes learnable from unlearnable teachers; it does not measure "how
unlearnable".
