# H117 — analysis

**ALL SIX LOCKED PREDICTIONS PASS.** The h116 exploratory finding replicates at
seeds 52-56, which played no part in generating it.

Analysis script committed at `2309876` and P4's verdict boundary at `ba724fe`,
both **before** seed56 finished. GATE G0 passed earlier (83 queries, 0 differing).
No seeds excluded by the n>=15 floor.

| seed | wR | uR | wR/uR | MF-DRO off-bnd | MF-MES off-bnd | nDRO | nMES |
|---|---|---|---|---|---|---|---|
| 52 | 3.065 | 1.347 | 2.276 | 14.6% | 1.1% | 89 | 89 |
| 53 | 3.002 | 1.715 | 1.750 | 13.1% | 0.0% | 99 | 85 |
| 54 | 4.804 | 1.342 | 3.580 | 4.1% | 0.0% | 98 | 90 |
| 55 | 2.796 | 1.159 | 2.413 | 7.1% | 0.0% | 99 | 82 |
| 56 | 2.277 | 0.911 | 2.499 | 11.2% | 0.0% | 98 | 79 |

| prediction | required | observed | |
|---|---|---|---|
| P1 | wR/uR >= 2.0 in >= 4/5 | **4/5** | PASS |
| P2 | \|mean log(wR/uR)\|/sd >= 2.0 | **3.48** | PASS |
| P3 | direction wR > uR | **5/5** | PASS |
| P4 | DRO off-bnd > MES in 5/5 | **5/5** | PASS |
| P5 | DRO mean off-bnd >= 3% | **10.0%** | PASS |
| P6 | paired \|mean\|/sd >= 1.0 | **2.40** | PASS |

## Replication quality, stated honestly

The effect is **smaller at the confirmatory seeds than at the exploratory ones**,
which is the expected direction and worth recording rather than glossing:

  wR/uR   h116 (42-46, exploratory): 2.95, 3.22, 2.96, 3.67, 5.53 — mean **3.67**
          h117 (52-56, confirmatory): 2.28, 1.75, 3.58, 2.41, 2.50 — mean **2.50**

  off-bnd h116 mean **8.9%**  vs  h117 mean **10.0%**

The wasted-budget fraction reproduces closely (8.9% -> 10.0%); the dispersion
ratio comes in about a third smaller. P1 passes 4/5 rather than 5/5 because seed
53 lands at 1.750, below the 2.0 bar. Both are consistent with the exploratory
estimate having been the high end of its own sampling distribution.

## What this establishes, and what h118 already took away

ESTABLISHED: MF-DRO reproducibly fails to reach a boundary optimum in Borehole's
dominant dimension and pays HF budget for it — **10.0% of its high-fidelity
queries against MF-MES's 0.2%**, at independent seeds, on a pre-registered test.

NOT ESTABLISHED, and the protocol said so in advance:

- **Amendment 2**: MF-MES refines with box-constrained L-BFGS-B
  (`mf_mes_takeno.py:297`), which converges onto active constraints. Its ~0%
  off-boundary rate is largely a property of its optimiser, so this licenses
  neither "the DT is boundary-averse" nor "MF-MES searches better".
- **h118**: the waste does not predict regret. ROI-Q10 and REFINE-100 differ
  2.8x in wasted budget and reach the same final value; r(waste, outcome) = -0.26
  over 15 runs.

So h117 confirms a real, reproducible inefficiency that **is not the reason
MF-DRO loses**. That combination — a clean confirmatory pass on a quantity
already shown not to matter — is the honest summary, and it is why the
replication was still worth running: "the inefficiency is a seed artefact" and
"the inefficiency is real but irrelevant" are different states of knowledge, and
only the second is now supported.
