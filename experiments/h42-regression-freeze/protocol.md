> **ID DISAMBIGUATION (added 2026-08-27).** The id `H42` names TWO distinct
> protocols in this tree. This file is `h42-regression-freeze` (this one has 3 result files).
> The other is `h42-fixed-rule-control` — "is "not frozen" caused by the pipeline, not the DT?" (no results -- registered and abandoned).
> They were registered minutes apart by the same session, which reissued the
> number because nothing checked. Any bare reference to "H42" elsewhere is
> ambiguous and must be resolved by content. Numbers are now claimed via
> `tools/claim_id.sh`, which blocks reissue.

# H42 — does the REGRESSION head freeze the incumbent on Hartmann 6D?

Focused replacement for H40 (stopped). Single question, single arm.

## Question

With `use_candidate_scoring=False` (the DT regresses the query point directly via
`action_head` instead of scoring a candidate pool), does the incumbent freeze on
Hartmann 6D?

**Freeze** = zero incumbent improvements over the run: the best HF value never
moves after initialisation.

## Design

- Hartmann 6D, `use_candidate_scoring=False`
- **3 seeds** (42, 43, 44), **50 iterations each**, iteration-capped (no cost cap)
- Initial design `n_HF=6, n_LF=45` (literature-standard ~10% sizing, same as H40)

## Reference point

No control arm is run — freeze is self-contained (zero improvements or not). The
comparison is historical: **9/12 (75%) of runs froze pre-leak-fix**. H39 found no
freeze with the regression head on Currin 2D (n=1): 14/14 distinct proposals, 2
improvements. Hartmann is the benchmark where the pathology was actually observed.

## Outcomes

- **0/3 frozen** → H39 replicates on the hard benchmark; the regression head does
  not reintroduce the freeze, and the target-leakage fix gets the credit.
- **≥1/3 frozen** → the freeze can return under the regression head, and the
  candidate-scoring rewrite is doing more than H39 suggested.

## Reported per seed

incumbent improvements, frozen yes/no, distinct proposals, query spread, final
simple regret, inference regret.
