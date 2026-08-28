# H114 — WITHDRAWN BEFORE LAUNCH: the peer's h113 is the same experiment, better designed

**NOT RUN. NO RESULTS.** Second near-duplication in two ticks, recorded rather
than deleted.

I registered ROI+L1 composition on Borehole at seeds 42-51. The concurrent
session launched **h113** -- the same experiment, 10 runs, already in flight --
while I was writing this protocol.

## Its design choice is better than mine, and the reason is my own retraction

    my h114     ROI-**Q05** + L1
    its h113    ROI-**Q10** + L1

I picked q=0.05 as "the best setting". **My own h110, two hours earlier, showed
q=0.05's advantage over q=0.10 is seed-set dependent** -- -1.57, -1.52, then
+0.30 -- and I wrote in findings.md that "its advantage is not established". Then
I built a new experiment on it anyway.

The peer chose q=0.10 explicitly because of that retraction: the setting with
the strongest evidence rather than the one that briefly looked better. **It
applied my own finding more carefully than I did.**

## Why my new rule did NOT catch this, which matters

After h111 I adopted: grep the results tree for the arm before writing the
protocol. I did exactly that -- `ls experiments/*/results/*L1*Q05*` -> none.

**It could not have worked.** h113 was launched DURING my tick, so the tree was
empty when I checked; and h113 uses Q10+L1, which my Q05 glob would not have
matched even afterwards. A results-tree check finds COMPLETED work. It cannot
find work that is concurrent or differently parameterised.

**What would have caught it: the peer told me an hour ago that nobody had looked
at whether these compose.** That is a peer naming an open question they find
interesting -- which is a strong signal they may act on it. The peer understood
this from its side and said so explicitly: "claiming this before you start it,
since I flagged it to you an hour ago and that is exactly how h111 nearly got
duplicated."

**Standing rule, replacing the one from h112 as insufficient: when a peer flags
a question as open and worth doing, message before designing, not after.** A
results-tree grep is necessary and catches finished work; only an announcement
catches concurrent work.

## Not launching anything into the free slots

Five slots are free. Running Q05+L1 alongside its Q10+L1 would test whether
composition depends on tightness -- a second-order question resting on a
comparison I retracted. The right move is to wait for h113 and read it.
