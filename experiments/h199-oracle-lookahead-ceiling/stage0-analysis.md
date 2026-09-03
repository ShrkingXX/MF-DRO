# h199 Stage 0 — PASS. The ceiling arm is attributable.

| teacher | tau=0 action SD (normalised) | vs MES |
|---|---|---|
| MES (reference) | 0.1890 | — |
| h198 GP-lookahead | 0.1973 | 104.4% |
| **h199 ORACLE** | **0.1971** | **104.3%** |

**SC-DIVERSITY PASS with room to spare** (gate was >=25%). This is the SC that
matters, because h146 established that h145's oracle confounded QUALITY with
ZERO ENDPOINT DIVERSITY and could not distinguish them. h199 does not collapse:
the candidate shortlist still comes from MES on the CURRENT GP STATE, so rollouts
still start from different places even though the EVALUATION of those candidates
is now truthful. A result here is attributable to quality.

SC-ORACLE-USED: differs from GP-lookahead on 7/14 starts -> not a silent no-op.
SC1: reduces exactly to greedy MES at n_c=1, 6/6 -- the identity survives the
oracle substitution.

```
======================================================================
SC-DIVERSITY (GATE): across-rollout SD of the tau=0 action, normalised units
   MES teacher       : 0.1890   (reference)
   h198 GP-lookahead : 0.1973   (104.4% of MES)
   h199 ORACLE       : 0.1971   (104.3% of MES)
   -> PASS

SC-ORACLE-USED: does the oracle change the choice vs h198?
   differs from GP-lookahead on 7/14 starts
   -> PASS

SC1: n_c=1 still reduces exactly to greedy MES under the oracle
   identical on 6/6 -> PASS
======================================================================
STAGE 0: PASS
```
