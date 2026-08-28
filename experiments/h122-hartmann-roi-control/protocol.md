# H122 — Does ANY ROI variant reduce waste on the benchmark where the waste is?

STATUS: LOCKED before launch and before any h122 statistic was computed.
TYPE: CONFIRMATORY. **The prediction is a NULL** (stated below), and I expect to
      confirm it. It is registered because a null here is the direct answer to
      the primary question and should be on record as a prediction, not as a
      post-hoc shrug.
COMPUTE: 3 new runs (~113 min each, 3 slots). Machine at 10/15; this takes it
         to 13.

## Why this experiment exists

h121 established that the waste the founding diagnosis names lives on
**Hartmann** (12.5% median) and not on Borehole (3.2%), while every demonstrated
ROI benefit is Borehole-only. Nobody has ever tested whether an ROI variant
reduces waste ON HARTMANN against a proper control, because h84's Hartmann
`ROI-OFF` arm was never completed: it holds seeds 42 and 43 only, exactly the
same shortfall found on its Borehole side.

So: complete the control, then test the question on the benchmark that matters.

## What is actually being compared (corrected today)

The three ROI arms in h84 form a tightness ladder, but NOT the one the arm names
suggest. Per today's bug finding:

  ROI-ANN   nominally q: 0.50 -> 0.05. ACTUALLY constant q ~ 0.498 on Hartmann
            (moves 0.13 points across the run). The LOOSEST ROI in the project.
  ROI-Q10   constant acceptance q = 0.10.
  ROI-FIX2  constant beta = 2.0 (acceptance floats; not pinned).

This protocol treats ROI-ANN as a constant q~0.50 arm and says so. Any write-up
calling it "annealed" is wrong.

## Runs

`Hartmann_6D ROI-OFF` at seeds 44, 45, 46, under h84's exact worker and config
(`dict(use_roi=False)`, bo_iterations=4000, BUDGET=200, n_hf=6, n_lf=45),
completing that arm to seeds 42-46.

Provenance: h117's GATE G0 passed today (83 queries, 0 differing), so the current
tree reproduces stored MF-DRO traces bit-identically. ROI-OFF is the
use_roi=False path, which h105 separately measured byte-identical.

## Measure

waste_frac = fraction of non-init HF queries with y < max(initial-design HF y),
per arm x seed, exactly as h121. Also reported: final best HF y.

## Prediction (locked) — a NULL

P1. **No ROI arm reduces Hartmann waste_frac relative to ROI-OFF in >= 4/5
    seeds.** I expect all three to fail that bar.

Grounds: h111 showed the ROI's regret benefit fails on Hartmann and Ackley at
two tightness settings spanning 2x; h118 showed the ROI barely moves the
spatial quantity it was supposed to move; h121 showed the ROI's only demonstrated
benefit is on the benchmark where waste is smallest.

P2. If any arm DOES clear 4/5, that is a positive result against my stated
    expectation and must be reported at least as prominently as the null —
    with the caveat that three arms are being tested against one control, so a
    single arm clearing the bar is weak evidence and would need its own
    confirmation at fresh seeds.

## Why a null is worth three runs

Because the primary question asks for an ROI strategy that stops MF-DRO wasting
HF budget, and the honest answer so far is assembled from three experiments on
the wrong benchmark. A properly controlled null ON HARTMANN converts "the ROI
has never been shown to help where the waste is" into "the ROI was tested where
the waste is, with a complete control, and did not help." Those are different
claims and only the second one is worth writing in a paper.

## Limitations

- n=5, no p-values. Hartmann HF counts are 6-24 per run, so waste_frac rests on
  very few queries and this test has LOW POWER. A null here is weak evidence of
  absence and will be reported as such.
- Three arms against one control: multiplicity, addressed by P2's requirement
  that any winner be confirmed separately rather than believed.
- Hartmann only. Says nothing about Borehole, where the ROI does work.


---

## AMENDMENT — these are h84 ARM COMPLETIONS, not a separate experiment.
## Filed before any result file from these runs existed.

PROBLEM I CREATED. These ROI-OFF runs write into this experiment's results
directory, while the ROI arms they must be compared against live in h84. Pairing
across those directories is exactly the CROSS-EXPERIMENT PAIRING this project
bans, and the ban exists for good reason: different experiments can differ in
code, config, spec or budget in ways that are invisible at analysis time.

DECLARED EXCEPTION, with grounds. These runs are not a different experiment;
they are the missing seeds of an h84 arm, produced by:

  - the SAME worker file (`code/worker.py` is a byte copy of h84's),
  - the SAME arm config, `dict(use_roi=False)`,
  - the SAME `bo_iterations=4000`, `BUDGET=200.0`, and benchmark SPEC,
  - a tree that passed h117's GATE G0 today (83 queries, 0 differing) and whose
    use_roi=False path h105 separately measured byte-identical (md5 ff70f008c0ac).

HOW THIS IS HANDLED. On completion the JSON files are COPIED into
`experiments/h84-roi-strategy/results/` (and `results/ckpt/`) so that h84 holds
one complete arm and the analysis reads a single experiment directory. A
manifest `experiments/h84-roi-strategy/results/COMPLETED_ARMS.md` records, for
each copied run: which experiment launched it, when, its git commit, its
`code.dirty` state, and this justification. The originals stay in place so the
launch record is not rewritten.

WHAT WOULD INVALIDATE THIS. If any copied run's recorded `code.commit` turns out
to differ from h84's in a way that is NOT empty-diff over
`src/ dro_runner.py benchmarks.py`, the merge is withdrawn and the runs are
reported separately as their own experiment with the pairing caveat stated. This
is checked at merge time, per run, and the check is recorded in the manifest.

I am declaring this BEFORE seeing any of these runs so that the exception cannot
be shaped by what they show.
