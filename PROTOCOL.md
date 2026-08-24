# Pre-Registered Protocol — MF-DRO on Hartmann 6D

Registered before any fix work. **Frozen**: the autoresearch loop may change the
method, not this file. Any change here must be made by the user, explicitly, and
logged in the Amendments section with a reason.

## Research question

Does the incumbent-freeze pathology in MF-DRO on Hartmann 6D have a fix within
the DRO frame (simulated trajectory + Decision Transformer), and does that fix
produce lower final simple regret than MF-MI-Greedy and MF-GP-UCB at matched
real cost?

This is a question, not a target. "MF-DRO does not win" is a permitted and
reportable outcome.

## Frozen evaluation protocol

| Item | Value |
|---|---|
| Benchmark | Hartmann 6D |
| Baselines | MF-MI-Greedy, MF-GP-UCB |
| Seeds | 10, fixed set, identical across all methods |
| Budget | Matched real cost across methods |
| Metric | Final simple regret at matched cost |
| Success test | MF-DRO mean+SE strictly below best-baseline mean−SE |
| Initial design | Identical for all methods within a comparison |

Report individual seed traces alongside mean ± SE. Report every run, including
runs that fail a gate.

## Fix scope

**In scope** — the DRO frame:
- Simulated trajectory generation and rollout
- Decision Transformer architecture, training, conditioning, reward/RTG
- DRO acquisition and its ambiguity set
- Initial design (e.g. `sequential_max_variance_design` vs `lhs_design`),
  provided the same design is applied to MF-DRO and all baselines

**Out of scope:**
- Replacing MF-DRO with an existing method. Not a fix.
- Changing anything in the frozen evaluation protocol above.
- Selecting seeds, budgets, or basin widths on the basis of results.

## Prior evidence carried in

1. **Incumbent-freeze is benchmark-specific, not architectural.** On Ackley 10D,
   MF-DRO incumbent-improvement counts are 7, 9, 8 vs 3–4 for all baselines
   (`results/mfdro_ackley_test`). MF-DRO does not freeze there.

2. **The "unfixable initialization" claim in REVISION_LOG.md does not hold.**
   Verified by 2000-seed simulation:
   - P(any of 18 LHS points within L2=0.3 of x*) = **6.2%**, not the logged 12%.
   - The `max > 2.0` gate fails for **86%** of seeds; median init max is 1.24.
   - seed=42's max of 0.9632 is the **30th percentile** — an ordinary draw, not
     an unlucky one. It is not evidence of a rare initialization failure.

3. **A mitigation exists and has never been benchmarked.**
   `src/utils/init_design.py:43` `sequential_max_variance_design`, written for
   "better coverage of narrow-basin benchmarks than LHS."

## Leading hypothesis

Incumbent-freeze on Hartmann 6D is driven by narrow-basin coverage failure at
initialization, not by a DT/DRO architectural defect.

**Caveat that must be tested, not assumed:** improving the initial design helps
the baselines too. Confirming the hypothesis may explain the freeze without
closing the performance gap. If that happens, report it as such — do not
reintroduce an asymmetric initialization to manufacture a win.

## Amendments

None.
