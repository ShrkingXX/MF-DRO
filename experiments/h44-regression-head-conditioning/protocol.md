# H44 — is the DT still input-inactive under the REGRESSION head?

## Why H42 does not answer this

H42 showed the regression head does not freeze: 0/3, 50/50 distinct proposals.
But *movement of queries* is exactly the behavioural evidence that misled this
project before. Queries move when the **weights** are re-fit each iteration; that
is not the same as the policy **conditioning** on its inputs.

Under candidate scoring the conditioning channel was measured inert (argmax 0/12
across state, RTG, BTG, and history). This asks whether the regression head is
different.

## A measurement difference that matters

Candidate scoring emits `argmax_k` over a discrete pool, so a small change in `h`
can be **absorbed** — the same candidate stays on top. The regression head emits
`x = action_head(h)` **continuously**, so *any* change in `h` produces *some*
change in `x`. This probe is therefore strictly more sensitive, and a null here
would be stronger than the 0/12 nulls.

Because any change is non-zero, the question is not "does x move" but "does it
move **enough to matter**". We normalise against the run's own query spread.

## Design

Train one DT with `use_candidate_scoring=False` on Hartmann 6D (seed 42), hold
the **weights fixed**, then vary only the inputs:

1. **state**: substitute each of the real τ=0 states from the run
2. **RTG**: sweep the realised band
3. **BTG**: sweep the realised band

Report `‖Δx‖` for each, divided by the run's **own** mean pairwise query spread.

## Locked predictions

1. **INACTIVE**: input-induced `‖Δx‖` is **< 10%** of the run's query spread for
   all three channels — i.e. re-fitting explains essentially all query movement,
   and the regression head is inactive like the scoring head.
2. **ACTIVE**: any channel induces **> 50%** of the query spread — the regression
   head conditions where candidate scoring did not, which would make the head
   choice load-bearing after all and change the paper's Section on the mechanism.
3. Between 10% and 50%: partial conditioning; report the ratio, claim neither.

## Scope

One seed, one benchmark. This is a mechanism probe, not a performance claim.
