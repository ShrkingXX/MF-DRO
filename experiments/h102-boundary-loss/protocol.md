# H102 — is boundary aversion caused by the L2 loss fitting a conditional MEAN?

LOCKED BEFORE ANY RUN. ID claimed via `tools/claim_id.sh`, so it cannot collide.

## Why this, and why now

research-state.yaml has carried this as an open question for hours — *"Boundary
aversion: causal test (change the output parameterisation) not run"* — and every
line of evidence since has pointed at it as the remaining lever:

  - Borehole is MF-DRO's ONE substantive deficit of four benchmarks (h93).
  - The calibrated ROI closes 37% of the gap and structurally cannot close more:
    it selects *where to look*, not *what the head can emit*.
  - Teacher refinement does not close it either.

**The mechanism is specific and checkable.** The regression head is trained with

    L_loc = F.mse_loss(x_pred, actions_x)

An L2 loss fits the **conditional mean**. A mean is pulled inward from a
boundary whenever the target distribution has any mass away from it. A
conditional **median** is not: it sits *at* the bound once more than half the
mass is there. That is a one-line difference in the loss and a large difference
in what the head can express at a bound.

And Borehole is where it bites hardest — **x\* lies on the domain boundary in 7
of 8 dimensions** (normalised [1, 0, 0.61, 1, 1, 0, 0, 1]), while the head
actually emits a boundary value in only **5.3%** of its proposals (6.5% with the
ROI). Measured here from h90's completed runs, not assumed.

## Design

| | |
|---|---|
| benchmark | Borehole_8D |
| seeds | 47, 48, 49, 50, 51 |
| new arm | **L1-LOSS** — `use_roi=False`, regression head, L1 instead of MSE |
| control | h90's NO-ROI at the SAME seeds: same code, same commit, same worker |
| runs | 5 (control is reused, not re-run) |

`use_candidate_scoring` stays **False**. This changes the loss the existing
regression head is trained under; it does not introduce a pool or an argmax, so
the standing constraint is untouched.

Implementation is held as a patch outside `src/` until compute clears, following
the convention a peer session adopted for h94: a worker starting mid-edit would
import a half-changed module.

## Gate (G3), and why it is not vacuous

Reading back a config flag proves nothing — that is the failure this project has
already catalogued. The observable side effect here is the **scale of L_loc
itself**: for residuals below 1, |r| > r², so an L1 objective reports a
systematically larger loss than MSE at equal fit quality. h90's completed runs
end at L_loc ≈ 0.033–0.038 under MSE; the same fit under L1 lands near
√0.035 ≈ 0.19.

**Gate: L1-LOSS runs must report final L_loc > 0.10.** An MSE objective cannot
produce that given every fit observed on this benchmark. If the value looks like
h90's, the flag did not take and the runs are void regardless of regret.

## Predictions

**P1 (mechanism).** L1-LOSS raises the fraction of proposals landing on a bound
in the 7 boundary dimensions, above NO-ROI's 5.3%. Registered **POSITIVE**: this
is close to definitional given median-vs-mean, and if it fails the intervention
did not do the thing it was chosen for.

**P2 (regret).** Registered as **GENUINELY UNCERTAIN, no direction predicted.**
Reaching bounds more often is not obviously good: dim 2's optimum is interior
(0.61), and a median-seeking head could over-commit to bounds where the mean was
right. Seven mechanism predictions have been refuted in this investigation;
declining to guess is the calibrated posture, not a hedge.

**P3 (the falsifier, and the reason this is worth running either way).** If P1
passes and P2 shows no improvement, then **boundary aversion is not the cause of
the residual gap** — the head reaches bounds more and still does not win. That
would retire the explanation this project has been building toward for hours, and
it is the single most informative outcome available.

## Separability bar

Adopted from h97's addendum, and from two bar-design failures a peer recorded.
Two ROI tightness settings 2.1x apart moved Borehole regret by 0.59 points
against a per-seed sd of 1.71. **A regret difference here counts as real only if
|paired mean| > 0.59 AND at least 4/5 seeds agree in direction.** Anything
smaller is INDISTINGUISHABLE whatever its sign.

## Known limitation, stated before results

Swapping the loss changes training **globally**, not only near boundaries. A
regret change cannot be attributed to boundary behaviour alone without P1's
mechanism check also passing. That is why P1 and P2 are registered separately and
why P3 is written as a conjunction.

---

## G3 pre-launch smoke test — PASSED, run before the patch touches `src/`

The peer session's h94 crashed all 8 jobs in 45s on a `NameError` because the ON
path had never been executed; their bit-identity gate could not catch it, since
with the flag off the new code never runs. h102 has exactly that exposure, so the
patch was applied to a **sandbox copy of `src/`** and exercised there.

    loc_loss='mse'  ->  model.loc_loss = 'mse'   L_loc = 0.078530
    loc_loss='l1'   ->  model.loc_loss = 'l1'    L_loc = 0.219020
    masked branch (the one training uses):  mse 0.063352   l1 0.195967
    ratios: 2.79 unmasked, 3.09 masked

Three things confirmed that a config read-back could not have shown:

  1. The flag **reaches the model** — `dt_cfg` forwarding works, which is the
     exact failure the H20 scar in `mf_dro.py` records.
  2. **Both** loss branches execute — the masked one is what training actually
     calls, and it is a separate call site.
  3. The objective genuinely **changes**, in the predicted direction and by the
     predicted rough factor. L1 exceeds MSE for sub-unit residuals, which is the
     same fact the G3 result gate relies on.

Verified after the test that the real `src/` still contains **zero** H102
markers: the sandbox was a copy and the patch remains un-applied in the tree.

**Process note.** Midway through, `git status` showed `src/policy/mf_dro.py`
modified and I read it as my own sandbox leaking into the tree. It was not — all
five pending `src/` hunks are the peer's H94 instrumentation, applied on their
side. Checking whose change it was took one command; asserting it was mine would
have been a false alarm about a shared file while 15 workers ran.

## Pre-flight: P1's measurement function verified, and the reference values it will be judged against

Checked before results, since P1's verdict depends entirely on `bound_frac` and a
bug there would produce a confident wrong answer.

    boundary dims: [0, 1, 3, 4, 5, 6, 7]  -- 7 of 8
    NO-ROI    per-seed  0.0467 0.0622 0.0639 0.0320 0.0594   mean 0.0529
    ROI-Q10   per-seed  0.0506 0.0673 0.0556 0.0810 0.0680   mean 0.0645

Both reproduce, to four decimals, the figures measured independently earlier in
this investigation. The control path (`h90/.../NO-ROI`) resolves.

**A reference point that sharpens P1's reading.** The ROI *already* raises
boundary-reaching from **5.29% to 6.45%** — a 1.16-point, 22% relative increase,
purely from constraining where the teacher looks. So the question h102 answers is
not "can anything raise it" but **how the loss function compares to the ROI as a
lever on the same quantity**:

  - L1 raising it by **much more than 1.16 points** would say the output
    parameterisation is the bigger lever, and that the ROI was working around a
    limitation of the head rather than fixing it.
  - L1 raising it by **about the same** would say the two act on the same
    bottleneck and are unlikely to compose.
  - L1 **not** raising it would fail P1 outright, and P2 would then say nothing
    about boundary aversion in either direction.

None of these were registered as predictions — P1 registers only the direction,
and P2 registers no direction at all. This is stated so the result is read
against a number rather than against an impression.
