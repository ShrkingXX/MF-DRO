# h169 -- STATE or CONDITIONING? Decomposing the training/inference gap

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## Where this stands

h167c: the DT fits its teacher's actions **2.5-4x better than any constant**
during training. h167: yet its real queries sit within 0.04 (Borehole) / 0.08
(Hartmann) of the box centre while working arms sit at 0.62-0.78. h168: and that
collapse is **independent of the conditioning RTG** across the full range 0 to 1
(movement 0.0074, ratio 0.986).

So the network can produce good actions, and RTG is not what stops it. The
remaining differences between the training setting and the inference setting are
the **state** and the auxiliary conditioning (BTG, timestep, context).

## The design: a 2x2, probed

The h168 probe already re-queries the DT at arbitrary (state, RTG) pairs and is
verified bit-identical to an unprobed run. Extend it to cross two factors:

  STATE     real (the actual trajectory state) | training (a tau=0 state from
            THIS iteration's rollout batch, i.e. exactly what the DT was fit on)
  RTG       the realised rtg_target | an in-support value (0.02)

Four cells, all measured at the same iteration on the same network.

## Predictions

P1 STATE is the operative factor: at TRAINING states the emitted action is far
   from the box centre (d comparable to the working arms' 0.62+, or at minimum
   >2x the real-state value), at REAL states it collapses -- and this holds at
   both RTG values.
P2 Neither factor matters: the action is near the box centre in all four cells.
   Then the network is globally collapsed at inference time regardless of input,
   and its good training loss must come from something the inference path does
   not reproduce -- which would point at the inference code path itself
   (propose_mf) rather than at any distributional story.
P3 An interaction: collapse only at (real state, target RTG). h168 already makes
   this unlikely, since RTG had no effect at real states, but it is named
   because a 2x2 can show interactions a 1-D sweep cannot.

## What this can RETRACT

R1 P1 holds -> state-distribution shift is the mechanism, and every account
   framed around the teacher's ACTIONS (quality, diversity, learnability,
   model-selection) has been looking at the wrong half of the problem. The
   teacher would matter only through the STATES its rollouts visit.
R2 P2 holds -> RETRACTS the entire "distribution shift" family at once,
   including the state version, and moves the suspect to the inference code path.
   That is the cheapest possible outcome to act on and the most embarrassing to
   have missed for seventeen ticks; it is named first for that reason.
R3 P3 holds -> h168's conclusion needs qualifying: RTG would matter, but only
   jointly with state.

## Implementation

`self._last_batch_tau0_states` is captured in `_generate_rollout_batch` (purely
additive, read-only). The probe gains a state axis. Same RNG save/restore, so the
arm must remain bit-identical to its unprobed twin -- **that equality is checked
against h149's run and is a gate, not a diagnostic.**

## Design

Hartmann_6D, RANDOM-POOL arm (the clearest failing arm: per-query d(centre)
0.0830), seeds 42-46. 5 workers.
