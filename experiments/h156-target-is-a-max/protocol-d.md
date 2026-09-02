# h156d -- is the interpolating-path misfit MY harness's fidelity shortcut?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The pattern being explained

| condition | Borehole err | Hartmann err |
|---|---|---|
| C3 random path | −7.6% | −4.6% |
| C4 oracle (interpolating) | −2.9% | −31.2% |
| C5 diverse-good (interpolating) | −19.3% | — |

Random fits well on both benchmarks. The two interpolating conditions fit worst
and inconsistently, and C4 gets the cross-benchmark DIRECTION wrong.

## The suspected cause is my own shortcut, not the arms

C4/C5 assign fidelity in run.py as `1 if torch.rand(1) < 0.75 else 0` -- a
hardcoded HF-heavy coin flip standing in for the real rule. The actual arms
(h145, h146) do NOT force fidelity: they choose it by the cost-normalised
info-gain criterion evaluated AT the forced point (mf_dro.py:1618ff, the block
h145 built and h145 v2 corrected after the degenerate-y* bug).

C3 should be insensitive to this -- random locations carry no information at
either fidelity. An interpolating path walking into a high-value region is
exactly where the HF/LF decision should bite. So the shortcut predicts the
observed pattern.

## The change

Replace the coin flip in C4/C5 with the arms' own criterion, verbatim: build the
HF proxy, Thompson-sample y* over the SAME roi_candidates pool (never a
one-point pool -- that was the h145 v1 bug), evaluate MES-HF and MES-LF at the
path point, cost-normalise, argmax. C1/C2/C3 are untouched.

## GATE (pre-stated)

PASS  C4 error on Hartmann falls well inside the C3 band (|err| < ~10%) AND the
      cross-benchmark direction for C4 becomes correct (Hartmann ORACLE ABOVE
      Borehole ORACLE, matching observed +0.0509).
FAIL  C4 stays far off -> the misfit is NOT the fidelity shortcut. The tail
      account then reproduces scale but demonstrably not fine structure, and
      findings.md must say so without hedging.

Named now: a fix that improves the MAGNITUDE but leaves the DIRECTION wrong is a
FAIL, not a partial pass. Direction is the thing h156c actually failed on.

## What this can RETRACT

- PASS retracts h156c's "reproduces scale, not structure" as a statement about
  the ACCOUNT; it would become a statement about a fixed bug in the harness, and
  h156c's analysis must be corrected rather than merely appended to.
- FAIL retracts nothing already claimed but hardens h156c's limitation into a
  permanent caveat on the tail account, which currently appears in findings.md
  and in the published report without one.

Either way the C1/C2/C3 numbers -- and therefore the h153 forecast -- are
unaffected, since those conditions are not touched.

## Compute

1 worker per benchmark, offline. h153 (5) + h155 (5) + 2 = 12 <= 15.
