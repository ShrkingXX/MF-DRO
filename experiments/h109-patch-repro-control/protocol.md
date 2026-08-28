# H109 — a true reproduction control for the patched `src/` on the use_roi=True path

LOCKED BEFORE ANY RUN.

## The gap this fills

A peer session argued from h106's halves that the working-tree patches (their h94
instrumentation, my `loc_loss` selector) did not perturb ROI runs: the 42-46
half gives −4.22 and the 52-56 half −3.93, a 0.30 split, and "if either patch had
perturbed a use_roi=True run, a 0.30 split is not what it would look like."

**That design cannot support the conclusion.** h106 ran five NEW runs, all at
seeds **52-56**; its 42-46 half is h84's stored data. So:

  - "the 42-46 half reproduces h84's −4.22 exactly" is true by construction —
    it *is* h84's numbers, not a re-run.
  - the 0.30 split compares **pre-patch seeds 42-46 against post-patch seeds
    52-56**, confounding any patch effect with the seed-set difference. h89
    measured up to 3.67 points of seed-set difficulty on this benchmark, so 0.30
    is reassuring about seed similarity and says little about the patches.

The conclusion may well be right. My own sandbox smoke test showed the MSE path
unchanged, and the diff resolves to `F.mse_loss` when `loc_loss` is unset. But
that is a *reasoned* argument, and the peer explicitly wanted it measured.

## The design that measures it

h84's own reproduction control is the template: **re-run an existing
configuration on the current code and compare traces to the stored run.**

| | |
|---|---|
| arm | Borehole_8D, **ROI-Q10**, seeds 42 and 43 — the exact config h84 ran |
| comparator | h84's stored `ROI-Q10` runs at those seeds |
| runs | 2 |

ROI-Q10 rather than Q05 deliberately: it exercises the **use_roi=True** path,
which is where a perturbation would matter and which the OFF-path byte-identity
argument (md5 ff70f008c0ac) does *not* cover.

## Prediction

**P1.** The re-runs are **bit-identical** to h84's: `|dregret| = 0` and
`max|dx| = 0`. Registered **POSITIVE**. Two independent reasons — the sandbox
smoke test, and the patch resolving to the original function object when the flag
is unset.

**This is a control, not a hypothesis.** If P1 fails, every post-patch ROI result
in this project (h106, h107, h108) is contaminated and must be re-run on clean
code. That is why it is worth two runs to settle rather than argue.
