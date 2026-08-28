# H112 — WITHDRAWN BEFORE LAUNCH: duplicate of the peer's h111

**NOT RUN. NO RESULTS. This directory exists to record a near-duplication.**

I designed this experiment -- Hartmann ROI-Q05 at seeds 42-46, to test whether
the ROI's Borehole-specificity is a TUNING artifact rather than a benchmark
property -- wrote its protocol with registered bars, and was about to launch 5
runs.

**The concurrent session had already done it**, as `h111-q05-generality`, and
had done it BETTER: Hartmann *and* Ackley, 10 runs, with Ackley already complete
5/5 and Hartmann at 3/5 with the last two running. Its protocol makes the same
argument I did and adds an observation I had missed -- that Hartmann's -1.62 at
q=0.10 "clears the magnitude bar while the split does not reach 4/5, which is
the shape of an effect too weak to resolve at n=5, not the shape of no effect."

I caught it only because `tools/claim_id.sh` handed me h112 and I noticed 3
Hartmann ROI-Q05 runs already on disk while checking what existed.

## The gap this exposes, which the ID tool does not close

`claim_id.sh` makes ID collisions impossible. **It does nothing about WORK
duplication.** I followed the protocol discipline correctly -- claimed an ID,
wrote bars before running -- and still nearly spent 5 runs reproducing an
experiment already 80% complete. The check that would have caught it is not
"is this number free" but "does this experiment already exist", and nothing in
the workflow asks that.

**Standing rule added: before writing a protocol, grep the results tree for the
arm you intend to run.** One command:

    ls experiments/*/results/{BENCH}__{ARM}__seed*.json

I ran exactly that command in the same tick -- but only AFTER writing the
protocol, as part of checking what comparators existed. Two minutes earlier and
this directory would not exist.

## Also recorded: my protocol was mislabelled

The file written here was titled "H111" -- the peer's actual ID -- because I had
drafted the number before claiming one. The tool gave me h112. A protocol
carrying another experiment's ID is exactly the ambiguity the h42/h44 collisions
created, arrived at by a different route.
