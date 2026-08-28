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
