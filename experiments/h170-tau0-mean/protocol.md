# h170 -- the tau=0 conditional-mean mechanism, tested QUANTITATIVELY

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.

## The claim under test

Inference always queries `timestep=0`. At tau=0 the training states are
near-degenerate (`uniq_tau0_states=3` of 60, and the real inference state is
bit-identical to one of them). So the DT can only emit **the conditional mean of
the teacher's tau=0 action**.

h169 showed this reproduces the SIGN of all seven arms' collapse. That is
post-hoc. This tests it as a NUMBER.

## The test

Serialise the **tau=0-only** teacher action mean per iteration (the existing
`teacher_action_stats_per_iter` aggregates over all tau, which is the wrong
marginal -- exactly the error that made h167's P2 fail). Then compare, per
iteration and per seed:

    d( DT's actual emitted query ,  mean of that iteration's tau=0 teacher actions )

against two baselines computed on the same iterations:

    d( emitted query , box centre )          -- what h167/h168 measured
    d( emitted query , a random pool point ) -- a null with no privileged location

## Predictions

P1 The emitted query is CLOSER to the tau=0 teacher mean than to the box centre,
   on the WORKING arms. This is the discriminating half: for the failing arms
   the two targets coincide (their tau=0 mean IS the centre), so only a working
   arm can separate them.
P2 On the failing arms the tau=0 mean and the box centre agree to within the
   measurement's own noise -- a consistency check, not evidence.
P3 The emitted query is closer to the tau=0 mean than to a random pool point on
   every arm.

## What this can RETRACT

R1 P1 fails -- the working arms' queries are NOT near their tau=0 teacher mean
   -> the mechanism is WRONG despite matching all seven signs, and becomes the
   sixth account to fall. The seven-arm agreement would then be explained by
   something coarser (working teachers simply query away from the centre for
   unrelated reasons), and h169's analysis must be rewritten, not appended to.
R2 P1 holds -> the mechanism survives its first non-post-hoc test. It would
   still be one test, and it would NOT be reported as established.
R3 P3 fails -> the whole framing is measuring nothing; any agreement is an
   artefact of the distance measure.

**R1 is the live risk and it is named first.** Five accounts on this front have
matched the available evidence and then failed. Matching seven signs is exactly
the kind of agreement that has misled me before, which is why the number, not
the sign, is the test.

## Design

Two arms on Borehole: the **control** (a working arm -- P1's discriminating
case) and **RANDOM-POOL** (a failing arm -- P2's consistency case), seeds 42-46.
Requires the tau=0 serialisation, so it needs fresh runs; the existing results
predate it.

Compute: 10 workers.

## AMENDMENT — no fresh runs needed, recorded before running

The protocol above says this "requires the tau=0 serialisation, so it needs
fresh runs". That is wrong and I found a cheaper route before launching.

The tau=0 teacher action is **reconstructible offline** from the existing
traces: refit the KO model on the data up to iteration t (the h156/h163 harness
pattern), draw the ensemble members and candidate pools, and take the
acquisition argmax. That IS the tau=0 teacher action, and averaging over
members/pools gives its mean — the quantity under test. No new runs, and it can
be applied to arms that are already complete.

Nothing about the predictions or the retraction map changes. Compute drops from
10 workers x ~2 hours to one worker for minutes, and the test can use the
five arms that already exist rather than two fresh ones.

**One caveat this introduces**, stated now: the reconstruction is not the
literal tau=0 action the run used — it is a fresh draw from the same generating
process (same model, same rule, different pool/ensemble RNG). So it estimates
the tau=0 action *distribution*, which is what the mechanism is about, but it
cannot be compared to the run's own realised draw. The mechanism predicts the
DT emits the DISTRIBUTION's mean, so this is the right object; but if the
reconstruction is biased, the test is biased, and the honest check is whether
the failing arms' reconstructed mean lands on the box centre as theory says
(P2 -- now serving as a calibration of the reconstruction, not just a
consistency check).
