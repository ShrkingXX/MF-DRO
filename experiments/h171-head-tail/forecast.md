# h171 BLIND FORECAST — committed with HEAD 0/5 and TAIL 0/5 finished

The τ=0 mechanism predicts more than the regret ordering: it predicts **where
each arm's DT will query**, quantitatively, from numbers h170 already measured.

## The derivation, from h170's measured τ=0 action means (Borehole, 120 draws)

| teacher rule | its τ=0 action mean, distance from box centre |
|---|---|
| MES argmax | **0.6831** |
| uniform draw | **0.0712** |

h171's arms have those exact τ=0 rules by construction:

  HEAD-MES  τ=0 = MES argmax   → predicted τ=0 mean ≈ 0.68 from centre
  TAIL-MES  τ=0 = uniform      → predicted τ=0 mean ≈ 0.07 from centre

h170 also measured that the DT's query lands ≈ 0.24–0.25 from its teacher's τ=0
mean on working arms.

## The forecast

**F1 (query location).** HEAD-MES's real queries sit **FAR** from the box centre
— centroid distance > 0.5, comparable to the control's 0.7604. TAIL-MES's sit
**NEAR** it — centroid distance < 0.2, comparable to RANDOM-POOL's 0.0239.

**F2 (regret).** HEAD-MES near the control (~15.82, improving ~5/5); TAIL-MES
near the failing floor (~43.94, ~0/5).

**F3 (the ordering is inverted relative to teacher quality).** TAIL-MES follows
the acquisition on 7 of 8 steps and HEAD-MES on 1 of 8, so every
trajectory-quality account predicts the opposite of F2.

## Why F1 is the stronger half

F2 is a two-way ordering and could come out right by luck. **F1 is a numeric
prediction of a location in 8-dimensional space**, derived from a measurement
made on *different* arms (control, UCB, RANDOM, ORACLE) before these two existed.
If F1 holds and F2 does not, the mechanism describes where the DT queries but not
why that matters. If F2 holds and F1 does not, the regret ordering is right for
some other reason and the mechanism is not the explanation.

## Retraction map is unchanged

R1 (HEAD fails) and R2 (TAIL works) each remain individually fatal, as locked in
protocol.md. This forecast adds precision; it does not soften anything.
