# H136 — GATE: does the per-iteration logging patch perturb an ROI run?

STATUS: LOCKED before the check runs.
TYPE: GATE. Blocking for h123 and for anything else launched on the patched tree.

## The patch

`src/policy/mf_dro.py`, one added dict field: `'n_real_iter': int(n_real_iter)`
inside the `if roi_stats is not None:` record. `roi_stats` already carries
`accept_frac` AND `beta_sqrt` per record; the only thing missing was which
iteration produced it, so one tag makes both recoverable as per-iteration series.

It closes M1's shape blind spot (does a configured ramp actually ramp?) and
answers whether ROI-FIX2's fixed beta is an implicit schedule — the peer's
question, which run-level aggregates could not resolve.

## Why h117's GATE G0 does NOT cover it

G0 ran `Ackley_10D MF-DRO seed42` — a **use_roi=False** arm. The patched line
sits inside `if roi_stats is not None:`, which that path never enters, and
`self.roi_stats` is None there by construction. **G0 could not have exercised
this patch.** Neither could h120's 414-query extension, which was also ROI-OFF.

Reusing a gate that structurally cannot reach the changed line is the h94 failure
in a new costume, and it is the reason this protocol exists rather than a reuse.

## The check

Run `Ackley_10D ROI-Q10 seed42` on the patched tree and compare against h86's
stored trace (83 queries, `roi_summary.n_records` = 2580) — an arm that
**does** enter the patched branch, 2580 times.

Ackley chosen over Borehole because its ROI-Q10 runs take ~39 min against
Borehole's ~83, and both exercise the same line.

PASS: 83 queries, identical `fid`, `x`, `y` at every one.
FAIL: any difference. The patch is then reverted and h123 does not launch.

## Prediction

PASS. The field is additive to a dict already being built: no control flow, no
RNG draw, no value consumed downstream, and `n_real_iter` is already an int in
scope. I expect inertness — and I am running the check anyway, because
"provably inert by inspection" is exactly the claim that failed twice today.

## What a PASS licenses

That this specific patch does not perturb an ROI run. Nothing more. It does not
re-license the h94/h102 patches, which G0 and h120's extension already cover on
the use_roi=False path.
