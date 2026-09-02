# h167 -- WHERE does the collapse land? A point prediction, not a spread.

STATUS: protocol locked, nothing run. TYPE: CONFIRMATORY.
Zero compute: existing serialised data only.

## Why this is sharper than h162/h164

h162 and h164 showed the DT's queries CLUSTER for the failing teachers. That is
a statement about spread. The mechanism claims something more specific: the
network is predicting the **mean of its teacher's action distribution**. If so,
the collapse should land at a **computable location**, and each failing teacher
predicts a DIFFERENT one.

Teacher action distributions, from the arms' own generators:

  RANDOM-POOL  x ~ Uniform(domain)                  -> mean = the DOMAIN CENTRE
  ORACLE       x_t = x_start + (x*-x_start)*t/(T-1),
               x_start ~ Uniform, t = 0..T-1
               mean(t/(T-1)) = 0.5                  -> mean = MIDPOINT(centre, x*)

Those are exact and they differ. Spread measures cannot distinguish them; a
centroid can.

## Predictions

P1 RANDOM-POOL's centroid is near the domain centre, and nearer to it than any
   working arm's centroid is.
P2 ORACLE's centroid is nearer to midpoint(centre, x*) than to the centre.
P3 Working arms' centroids are near NEITHER predicted location -- they go where
   the model sends them, which has no reason to be a distribution mean.

## What this can RETRACT

R1 P1 fails -> the "predicting the mean" resolution is WRONG as stated, and with
   it the only current account of why the DT fits the failing teachers BETTER
   while emitting worse points (the L_loc puzzle).
R2 P1 holds but P2 fails -> the collapse is toward the centre GENERICALLY, not
   toward each teacher's own mean. That is a weaker and different claim -- e.g.
   a bias toward the centre of the normalised box from the output
   parameterisation -- and must be stated that way.
R3 P3 fails (working arms also sit at the centre) -> the measure does not
   discriminate; nothing is learned.

**R2 is the outcome I consider most likely, and it is NOT support for the
mechanism as stated.** Named first so it cannot be read as support later.

## Caveat

Query-level statistics at n=5 -- the class that produced h150 (retracted) and
h154's refuted M2 direction. What makes it worth running is that this is a POINT
prediction against a computable target, not a direction.
