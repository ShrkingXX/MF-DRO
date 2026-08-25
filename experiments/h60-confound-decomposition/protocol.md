# H60 — decompose the SF-DRO vs MF-DRO gap into its four confounded factors

**CONFIRMATORY. Protocol committed before any run.**

## Why

h59 reported SF-DRO beating MF-DRO 3/3 on Borehole (15.1% vs 23.7%) and I
attributed it to dropping multi-fidelity. That attribution is retracted: on
Borehole MF-DRO is **99-100% HF post-init**, so fidelity provably did not act.
The two configurations differ in four ways:

| | MF-DRO | SF-DRO |
|---|---|---|
| initial design | 10 HF + **20 LF** | 10 HF only |
| surrogate | `KennedyOHaganGP` (rho + discrepancy GP) | `SingleTaskGP` ensemble |
| rollout teacher | joint MF-MES | EI |
| reward / RTG | `mes_entropy` | `use_mes_reward=False`, `rtg_schema="fixed"` |

Borehole is the right test bed precisely because fidelity is inert there — any
gap that survives is caused by the other three.

## Design

Borehole 8D, seeds 44/46/48, cost budget 200 post-init. One variable changed per
arm, all else identical to h57's MF-DRO.

| arm | change from MF-DRO | isolates |
|---|---|---|
| **BASE** | none — **reuses h57's cells** | reference |
| **NOLFINIT** | `initial_lf=0` (10 HF init only) | the 20 LF init points' effect on the KO fit |
| **REWARD** | `rollout_reward="improvement"` | the MES-entropy reward |
| **TEACHER** | `rollout_policy="thompson"` | the joint-MF-MES rollout teacher |

9 new jobs (3 arms x 3 seeds); BASE is reused, not re-run.

**Baseline reuse is legitimate**: `git diff 57b65ad HEAD` over
`src/policy/mf_dro.py`, `src/model/decisionTransformer.py`, `src/models/ko_gp.py`,
`src/utils/init_design.py` and `dro_runner.py` is empty — the only `src/` change
since h57 is the addition of `src/analysis/*`, which the policy never imports.
Each new result records its own commit hash so this stays checkable.

## What this CANNOT separate

**The surrogate.** KO-GP vs `SingleTaskGP` is a different class, not a flag, so
it needs a code change and is out of scope here. It is also the lead with prior
support — h48 found "the surrogate init, not the acquisition, is what survives
correction". If all three arms below come back null, the surrogate is what is
left, and that is an informative outcome rather than a failed experiment.

Note `rollout_policy="thompson"` is *not* EI — mf_dro offers mes/thompson/random
only. It tests "is the MES teacher load-bearing", not "does EI specifically help".

## Locked predictions

1. **PRIMARY**: each arm's mean final HF simple regret vs BASE's 73.40 (23.7%),
   paired on 3 seeds. Direction and win counts only, no p-values at n=3.
2. **PRE-REGISTERED EXPECTATION**: **NOLFINIT is the arm that moves.** On
   Borehole the KO model must identify rho and a discrepancy GP from 20 LF + 10
   HF points, where SF-DRO's plain GP fits 10 HF directly. If rho is weakly
   identified the KO posterior is worse than the simpler model, which would
   produce a gap with fidelity behaviour held constant — exactly what is
   observed. REWARD and TEACHER are expected to be near-null.
3. **NULL-ON-ALL-THREE**: if none of the three moves regret materially, the gap
   is attributable to the surrogate class itself, and the next experiment is a
   code-level swap rather than a flag.

## What this cannot settle

n = 3 per arm, one benchmark. Borehole is chosen for the inert-fidelity property,
not for generality; a factor that matters here may not on Hartmann, where the
fidelity split does vary.
