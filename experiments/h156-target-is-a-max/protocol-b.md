# h156b -- does the tail account explain ORACLE and DIVERSE-GOOD too?

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

h156 showed the offline harness reproduces two real rtg_targets:
  C1 closed-loop      MAX 0.9583  vs control's observed     0.9761
  C3 open, random     MAX 0.2825  vs RANDOM-POOL's observed 0.2965

If the tail account is right it must ALSO reproduce the other two arms, which it
has not yet been asked to. Adding:

  C4 ORACLE       random start, linear interpolation to the true x*
                  (h145's _expert_path, verbatim)
  C5 DIVERSE-GOOD random start, linear interpolation to argmax of POOL=256 true
                  objective draws -- a DIFFERENT high-quality endpoint per
                  trajectory (h146's _diverse_good_path, verbatim)

Targets to match: ORACLE 0.3113, DIVERSE-GOOD 0.3285.

## Prediction

Both collapse to ~0.3, near C3, DESPITE C5 having genuine endpoint diversity.
Under the tail account the operative variable is not endpoint diversity but the
across-trajectory upper tail of INFORMATION GAIN: walking toward an
already-good point earns little information whatever the destination, so the
batch has no informative tail for the max to find.

## What this can RETRACT

R1 C4/C5 do NOT collapse -> the harness does not explain the arms it was built
   to explain, and h156's agreement with C1/C3 is coincidence. The whole tail
   account would be withdrawn, including the h153 forecast that rests on it.
R2 C5 collapses but C4 does not (or vice versa) -> the account is partial and
   must be stated as covering only some arms.
R3 both collapse -> the harness reproduces all FOUR observed rtg_targets from
   trajectory geometry alone, and the tail account stands as the quantitative
   bridge that findings.md currently lacks.

Note this is a strong test: the harness was NOT tuned to these arms, and C4/C5
paths are copied from the arms' own generators rather than re-derived.
