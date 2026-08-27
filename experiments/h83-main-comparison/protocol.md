# H83 — main comparison: SF-DRO, MF-DRO, MF-MES, MF-MI-Greedy, MF-GP-UCB

LOCKED BEFORE ANY RUN. Requested directly by the human for the pitch talk.

## Design

| | |
|---|---|
| methods | SF-DRO, MF-DRO, MF-MES (Takeno), MF-MI-Greedy, MF-GP-UCB |
| benchmarks | Ackley_10D (5:1), Currin_2D (3:1), Hartmann_6D (8:1), Borehole_8D (2:1) |
| seeds | 42, 43, 44, 45, 46 |
| cost budget | 200, POST-INIT (initial design is not charged against it) |
| jobs | 5 x 4 x 5 = 100 |

MF-DRO defaults in force: **M = 3, n_roi_candidates = 600, refinement OFF**
(adopted after h81; see findings.md). MF-MES surrogate is **KO, not SLFM** --
see "MF-MES surrogate" below.

Initial design (n_hf / n_lf), matching h57 where the benchmark already existed:

| benchmark | n_hf | n_lf | c_H : c_L |
|---|---|---|---|
| Currin_2D | 5 | 15 | 3 : 1 |
| Hartmann_6D | 6 | 45 | 8 : 1 |
| Borehole_8D | 10 | 20 | 2 : 1 |
| Ackley_10D | 10 | 30 | 5 : 1 |

Ackley_10D is NEW to this comparison and its initial design is a judgement call,
stated here rather than buried: n_hf = d, n_lf = 3d at d = 10. No Ackley run
informed this choice; nothing has been run on Ackley_10D_{HF,LF} before.

BENCHMARK KEY TRAP. `Ackley_10D_HF` / `Ackley_10D_LF` are the pair used here
(both on [0,1]^10). The bare `Ackley_10D` entry in benchmarks.py is an OLDER,
UNRELATED single-fidelity entry on [-32.768, 32.768]^10. The worker only ever
forms `f"{bench}_HF"` / `f"{bench}_LF"`, never the bare key, so the trap cannot
fire; `_build_dro_config` receives `benchmark_spec` explicitly and uses
`benchmark_name` only as a logging label (dro.py:84,306-309,362).

## MF-MES surrogate: KO, deliberately, and why that is not a deviation

MF-MES's derivation requires only that outputs across fidelities be jointly
multivariate normal. The paper names three surrogates satisfying it:

  "Standard multi-output extensions of GPR such as multi-task GPR (Bonilla et
   al., 2008), co-kriging (Kennedy & O'Hagan, 2000), and semiparametric latent
   factor model (SLFM) (Teh et al., 2005), satisfy this condition."
                                                        -- MF-MES.pdf, Sec. 2

Kennedy-O'Hagan co-kriging IS our KO GP, so KO is sanctioned by the method.
SLFM appears only in their experiments ("For MF-GPR, we used SLFM in GP-based
methods, unless otherwise noted", C = 2). Holding the surrogate fixed across
MF-DRO and MF-MES makes the ACQUISITION the only variable; under SLFM any
difference would confound acquisition with surrogate. Decision carried forward
from h48 D1. Cost of this choice, stated: a reviewer may ask whether MF-MES was
denied its published surrogate; the quote above is the answer, and an SLFM
robustness check on one benchmark remains available.

MF-MI-Greedy runs at the reference's shipped lambda = 1 and is near-single-
fidelity by construction (1-4% LF). That is the authors' own code's behaviour,
established in findings.md Lesson 20, not a defect introduced here. lambda=0.95
would make it somewhat stronger on Borehole; we did not tune per benchmark.

## Metric (frozen)

Simple regret SR = f(x*) - best HF value observed so far, plotted against
CUMULATIVE COST. Relative regret = SR / |f(x*)|, reported as a percentage.

SR is recomputed IN ANALYSIS from each run's recorded query trace and the
benchmark's `known_optimal_value`, uniformly across all five methods, rather
than trusting five different in-optimizer curve conventions. Each method's own
final regret is ALSO stored and the analysis asserts the two agree to 1e-6; a
mismatch is a hard failure, not a warning. This is the check that would have
caught the negation-convention bugs this project has hit before.

Initial-design points are recorded with `is_init=True` and are excluded from
cost accounting (budget is post-init) but INCLUDED in the running best, since
every method sees its own initial design.

## Predictions (pre-registered, evaluated in analysis.md)

PRIMARY. MF-DRO does NOT beat the best baseline on any of the four benchmarks,
where "beat" = strictly lower mean relative regret at cost 200 AND >= 4/5 seeds.
This is a NEGATIVE prediction and it is the one the project's evidence supports
(the north star has settled negative; see findings.md). Stating it as the
primary bar means a positive result would be a genuine surprise, not a
post-hoc rescue.

SECONDARY. SF-DRO beats MF-DRO on Hartmann_6D (h73 found SF-DRO strong there at
n=10). Direction only, no threshold.

TERTIARY. MF-MI-Greedy's LF fraction is under 10% on all four benchmarks,
confirming Lesson 20 generalises to Ackley.

FALSIFIABLE AND SEPARATE: these are three independent bars. Meeting one says
nothing about the others; each is reported whether or not it passes. No
p-values at n=5.

## Deliverables

1. mean SR-vs-cost, one panel per benchmark, all five methods
2. seed-by-seed SR-vs-cost (5 seeds x 4 benchmarks)
3. MF-DRO per-seed per-iteration query/fidelity scatter

The IR (inference regret) plot was dropped at the human's explicit direction.

## Compute

<= 15 concurrent workers, 1 thread each (15-core M5 Pro). Every job checkpoints
its trace every 20 s to results/ckpt/ so a killed run is not total loss and a
run in flight can be inspected with freeze_watch.py.
