# h186 — the conditioning is not *uniquely* inert. It moves the output MORE than the state does.

**EXPLORATORY.** No new runs; re-analysis of the saved h168-style probe, which
sweeps **four states** (`real`, `train0/1/2`) × **three RTG values** (0.0, 0.5, 1.0)
at every iteration.

## Measurement

- **conditioning sensitivity** = |x(rtg=0) − x(rtg=1)| holding the state fixed
- **state sensitivity** = |x(state_i) − x(state_j)| holding rtg fixed

| arm | conditioning | state | state ÷ conditioning |
|---|---|---|---|
| Borehole unstandardised (h181) | 0.0782 | **0.0122** | 0.16 |
| Borehole standardised (h179) | 0.1010 | **0.0028** | 0.03 |
| Hartmann unstandardised (h177) | 0.0406 | **0.0069** | 0.17 |

Scale references: changing only the **random seed** moves the emitted query ~0.82;
swapping the teacher's whole decision **rule** moves it 0.044.

**The state moves the output 6× less than the conditioning does** — and after
standardisation, 36× less. The output barely moves for *anything*, which is what a
per-timestep constant predictor (h185) looks like from the input side.

## THE CONFOUND, named first because it limits the claim

**These are not comparable perturbations.**

- RTG is swept **across its full observed range** (0 → 1) — a deliberate, maximal
  perturbation of that input.
- The states are **whatever the run actually produced**, and they are separately
  documented as near-degenerate: `uniq_tau0_states = 3` of 60, on every seed, with
  the real inference state **bit-identical** to one of them.

So a small state sensitivity is partly a statement about **how little the states
differ**, not only about how little the model responds to them. The probe records
the state's *label*, not its vector, so the input distance cannot be recovered from
the saved files and the two sensitivities cannot be normalised against each other.

**What therefore cannot be concluded:** that the DT ignores its state. That would
need the state-vector distances, which were never serialised.

**What can be concluded:** across the variation the model *actually encounters at
inference*, the conditioning channel is not the uniquely dead input. It is the
**more** influential of the two.

## Why this matters — it corrects a framing I have been carrying

findings.md has described the conditioning as an architectural **defect**: raw
scalars into `Linear(1→H)` + LayerNorm, BTG's response 92.9× below RTG's, "a defect
with a fix". h181 already scoped the fix (only ~1.1% of h178's 336× transfers in
situ). This goes further:

**The conditioning is not uniquely broken. It is the most responsive input the model
has, and all of them are nearly irrelevant to its output.** That is a different
diagnosis with a different implication — fixing the embedding cannot help much,
because the model has settled on a solution that barely uses any input. Which is
exactly h185's per-timestep-constant result, seen from the input side rather than
the loss side.

## What could RETRACT this

- **Serialising the τ=0 state vectors** and finding the three states are in fact far
  apart would restore "the model ignores state" as the reading — or, if they are
  as close as suspected, would let the two sensitivities finally be normalised. That
  is a cheap addition (d floats per probe) and is the obvious next measurement.
- Three states on one benchmark family; the probe exists only on h177/h178/h179/h181.
- The 0.0028 figure for standardised Borehole rests on a single arm.
