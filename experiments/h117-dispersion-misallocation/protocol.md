# H117 — Is MF-DRO's excess dispersion concentrated in the dimensions that matter?

STATUS: LOCKED before any h117 run was launched.
TYPE: CONFIRMATORY replication of h116 section 3, which was EXPLORATORY.
DATA: fresh runs, Borehole_8D, seeds 52-56, MF-DRO and MF-MES, one experiment.
      Independent of h83's seeds 42-46. No cross-experiment pairing.

## What h116 found (exploratory, needs confirmation)

On Borehole (h83, seeds 42-46) MF-DRO and MF-MES disperse their non-init HF
queries by nearly the same TOTAL amount on the unit cube (ratio 1.06), but
MF-DRO's dispersion is 3.96x larger once each dimension is weighted by its
first-order variance share. Per-seed weighted/unweighted ratio (wR/uR):
2.95, 3.22, 2.96, 3.67, 5.53; |mean log|/sd = 4.84.

Reading: MF-DRO's excess spread sits in the high-variance-share dimensions
and it is correspondingly tighter in the irrelevant ones -- the opposite of
MF-MES, which tightens the dimensions that matter.

That measure was chosen AFTER h116's pre-registered Spearman test failed its
gate. It is therefore exploratory and cannot be load-bearing until replicated
on seeds that played no part in choosing it. That is this protocol.

## GATE G0 (precondition, blocking)

The working tree carries an uncommitted h94 patch and an h102 loc_loss
selector. Both are inert by inspection. Inspection is what missed h94's
NameError, so before h117 is analysed, `code/identity_gate.py` must show that
Ackley_10D MF-DRO seed42 on this tree reproduces h83's stored trace
bit-identically (fid, x, y at every query).

If G0 FAILS, h117 is VOID regardless of what its numbers say, and the h116
section-3 result reverts to unreplicated. G0 runs concurrently with the
h117 arms; a failure discards them.

## Measure (identical to h116 section 3, no changes)

Per seed, over non-init HF queries (`fid==1 and not is_init`), on the unit
cube Z = (X - domain_min)/(domain_max - domain_min):
  s_j = sd(Z[:,j]);  uR = sum_j s_j(DRO) / sum_j s_j(MES);
  wR = agg(s(DRO), bench) / agg(s(MES), bench)   [`tools/perdim.agg`, S1-weighted]
Statistic: wR/uR per seed. Amendment-2 floor of 15 non-init HF queries applies
(Borehole ran 79-99 in h83, so no exclusions are expected; any are reported).

## Prediction (locked)

1. wR/uR >= 2.0 in at least 4 of 5 seeds.
2. |mean log(wR/uR)| / sd >= 2.0.
3. Direction: wR > uR (excess dispersion in the HIGH-share dimensions).

Failing 1 or 2 refutes the replication. A result in the opposite direction
(wR < uR) refutes it outright and would mean h116 section 3 was a seed
artefact.

## What this does and does not establish

Confirms, if it passes: MF-DRO mis-allocates dispersion across dimensions
relative to MF-MES, on Borehole, reproducibly.

Does NOT establish: that this causes the regret gap, or that correcting the
allocation would close it. Both remain untested. The founding diagnosis's
"3x more dispersed" concerned the DT's raw PROPOSALS; this measures EXECUTED
HF queries. They are related but not the same quantity.

## Limitations

- n=5 seeds. No p-values (project rule).
- One benchmark. Borehole is the most anisotropic of the four (PR/d = 0.168),
  and every ROI effect in this project has been Borehole-specific. This
  protocol does NOT test generality and must not be reported as if it did.
- Hartmann cannot be added: MF-DRO's non-init HF counts there fall below the
  n floor in 4/5 (h83) and 3/5 (h87) seeds.
- Ackley is definitionally excluded: uniform S1 shares make wR == uR exactly.

---

## AMENDMENT 1 (2026-08-28) — the effect is ONE dimension. Filed before any
## h117 run finished; zero h117 result files existed when this was written.

Further exploratory analysis of h83 (seeds 42-46 — NOT h117's seeds) shows the
"misallocation across dimensions" framing in the parent protocol is wrong.
Per-dimension sd ratios DRO/MES on Borehole:

  dim 0 (S1 = 0.858):  18.15x
  all seven others:    0.67 - 1.89, no consistent direction
  weighted ratio excluding dim 0:   0.97
  unweighted ratio excluding dim 0: 0.97

The two methods are indistinguishable on seven of eight dimensions. The whole
wR/uR effect is dim 0. "Mis-allocates dispersion across dimensions" overstates
it and is withdrawn.

### What is actually happening

Borehole dim 0 is r_w in [0.05, 0.15], and its optimum is at the UPPER BOUNDARY.

  MF-MES: mean z0 0.997-1.000, sd 0.001-0.013, 100% of HF queries at z0 >= 0.9.
  MF-DRO: mean z0 0.948-0.965, sd 0.032-0.104,  84-97% at z0 >= 0.9, min 0.128.

MF-MES pins to the boundary and never leaves. MF-DRO approaches it and keeps
drifting off. The 18x is therefore partly a small-denominator artefact
(MF-MES's sd is ~0.001) and the RATIO is not the quantity to report.

The absolute quantity is: MF-DRO spends **8.9% of its HF queries** (per seed
3.2, 14.1, 6.5, 16.5, 4.3) off the boundary; MF-MES spends 0.0%. Those queries
are unambiguously wasted -- the BEST off-boundary y (175-208) is below the MEAN
on-boundary y (226-238) in all five seeds, so no off-boundary query could ever
have become the incumbent.

This is the "boundary aversion" already recorded for Borehole, now localised to
the single dimension that carries 86% of the variance and priced in HF budget.

### ADDITIONAL LOCKED PREDICTION for h117 (seeds 52-56)

4. MF-DRO's off-boundary fraction (z0 < 0.9, non-init HF queries) EXCEEDS
   MF-MES's in 5/5 seeds.
5. MF-DRO's mean off-boundary fraction >= 3%.
6. Paired |mean|/sd >= 1.0.

Gate 6 is deliberately set BELOW the h83 point estimate (1.49): with n=5 and a
per-seed spread of 3.2-16.5% this measure cannot support a stringent magnitude
gate, and pretending otherwise would be false precision. Predictions 4-6 test
DIRECTION and rough SCALE, not magnitude.

Predictions 1-3 in the parent protocol stand unchanged, but their INTERPRETATION
is now "dim 0 boundary-locking failure", not "allocation across dimensions".

---

## AMENDMENT 2 (2026-08-28) — baseline confound. Filed before any h117 run finished.

`mf_mes_takeno.py:297` refines its query with box-constrained L-BFGS-B over a
2048-point Sobol pool. MF-DRO has no continuous refinement: its query is a
regression onto 600 i.i.d.-uniform teacher candidates (`_draw_raw`, torch.rand).

A box-constrained quasi-Newton method lands ON active constraints, so MF-MES's
0.0% off-boundary rate is largely a property of its optimiser.

Predictions 4-6 are UNCHANGED and still worth measuring -- they quantify a real
end-to-end deficit. But the conclusion h117 may draw is narrowed: a PASS shows
MF-DRO reproducibly fails to reach a boundary optimum and pays HF budget for it.
It does NOT license "the DT is boundary-averse" or "MF-MES searches better",
because the two arms differ by construction in exactly the way that produces
this contrast.

Separating "absence of continuous refinement" from "something about the DT/ROI/
loss" requires an arm this protocol does not contain and which is not registered
anywhere. Noting it as the open question rather than smuggling it into h117.

---

## GATE G0 RESULT (2026-08-28): **PASS** — 83 queries, 0 differing.

`Ackley_10D MF-DRO seed42` on the current working tree reproduces h83's stored
trace bit-identically (fid, x, y at every one of 83 queries). Log:
`results/identity_gate.log`.

The uncommitted h94 (`roi_inference_mode` hook, `_roi_snap`, actions_x variance
logging) and h102 (`loc_loss` selector) patches are therefore inert BY
EXECUTION, not merely by inspection — which is what this gate existed to
establish, after inspection alone missed h94's NameError.

h117's ten Borehole runs are cleared to count. This also discharges the
dependency recorded in h120 Amendment 2: the three ROI-OFF control runs are no
longer contingent on an unverified tree.

---

## AMENDMENT 3 (2026-08-28) — P4's verdict boundary, closed BEFORE the last run
## landed. seed56 was still executing when this was written.

Ran the peer's `tools/check_gate.py` against my own registered gates. The gaps it
first reported were artefacts of how I stated them TO the tool, not defects in
the protocol — P1/P2/P3 and P4/P5/P6 each partition their outcome space under the
natural reading. I record that so the tool is not credited with a catch it did
not make.

**But it forced a real inconsistency into view.** P1 passes at **>= 4 of 5**
seeds. P4 requires **5 of 5 exactly**. Two gates in one protocol using different
consistency thresholds, with no stated reason for the difference.

Under the natural reading a 4/5 result FAILS P4, and I am fixing that reading
NOW, in writing, while seed56 is still running and no h117 number exists:

  **P4 FALSIFIER: fewer than 5 of 5. A 4/5 result is a FAIL, not a partial pass
  and not an indeterminate.**

Why P4 is stricter than P1, stated late but stated before the data: P4 is a
DIRECTIONAL claim about a quantity whose control value is exactly 0.0% — MF-MES
spent 0.0% of its HF queries off-boundary in all five h83 seeds. Against a floor
of zero, "MF-DRO exceeds" should hold in every seed or the direction is not what
was claimed. P1's threshold concerns a ratio with genuine seed-to-seed spread,
where 4/5 is the appropriate bar.

That reasoning should have been in the protocol when P4 was written. It was not,
and a reader would have been entitled to ask why the two differ.

**No number is being fitted here.** h117's analysis script was committed at
2309876, before this amendment, and implements P4 as `(d>0).sum() == len(rows)` —
i.e. it already treats 4/5 as a fail. This amendment states in prose what the
committed code already does.
