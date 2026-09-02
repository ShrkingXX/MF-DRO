# h173 — **R2 fires.** The front's answer holds on a second benchmark.

CONFIRMATORY. TAIL n=5, HEAD n=4 (one seed still running).

| arm | rel% | improves | HF frac | centroid | at floor | per-seed rel% |
|---|---|---|---|---|---|---|
| control MES | **7.99** | 5/5 | 0.200 | 0.6287 | 0/5 | 16.4, 0.7, 10.2, 5.3, 7.4 |
| **HEAD-MES** | **12.10** | 3/4 | 0.304 | **0.5632** | 1/4 | 2.3, 22.7, 11.5, 11.9 |
| **TAIL-MES** | **46.45** | 4/5 | 0.281 | **0.0404** | 1/5 | 50.0, 21.5, 22.7, 77.7, 60.3 |
| RANDOM-POOL | 65.14 | 2/5 | 0.281 | 0.0237 | 1/5 | 78.6, 77.7, 22.7, 78.2, 68.6 |

**P1 holds**: HEAD works (12.10, far from TAIL's 46.45 and RANDOM's 65.14).
**P2 holds**: TAIL fails, in the failing band.
**P3 holds decisively**: HEAD's query centroid sits at 0.5632 (control 0.6287),
TAIL's at **0.0404** — the box centre (RANDOM 0.0237).

**The teacher that follows the acquisition on seven of eight steps fails on a
second benchmark too.** R1 does not fire; the front's answer does not need
scoping to Borehole.

## SC1 PASSES here — and that closes the gap Borehole left open

On Borehole, h171's TAIL collapsed its fidelity (HF 0.217 against 0.883), SC1
fired, and the attribution had to rest on ORACLE and DIVERSE-GOOD instead.

**On Hartmann TAIL's HF fraction is 0.281 against the control's 0.200 — no
collapse.** So TAIL itself supplies the clean attribution here: a τ=0 location
drawn independently of the model is sufficient to fail **with the fidelity mix
intact**, demonstrated within the arm rather than borrowed from others.

The second benchmark supplied exactly what the first could not. That is the
argument for replicating at all, and it paid here.

## SC2 (the Hartmann saturation floor) checked, and it does not drive the result

Seed 44 sits at the 0.7531 floor for HEAD, TAIL and RANDOM alike — 1 of 4, 1 of
5, 1 of 5. Registered in advance so it could not be read as a tie. Dropping that
seed leaves the ordering unchanged (HEAD 2.3/22.7/11.9, TAIL 50.0/21.5/77.7/60.3).

## Caveats

HEAD is n=4. Hartmann's per-seed spread is large on every arm (control ranges
0.7–16.4), so the HEAD-vs-control gap (12.10 vs 7.99) is not resolvable at this
sample size — what is resolvable is HEAD vs TAIL, a factor of nearly four, and
TAIL vs the control, a factor of six.

---

# CORRECTION at n=5 — the verdict is **R3 (partial), not R2**

HEAD's fifth seed (42) returned raw regret **2.5713** — not at the 0.7531
saturation floor, so a genuine bad run, and close to RANDOM-POOL's 2.6100 on the
same seed. It moves HEAD from 12.10 (n=4) to **25.16 (n=5)**.

| Hartmann arm | rel% | improves | centroid | per-seed rel% |
|---|---|---|---|---|
| control | **7.99** | 5/5 | 0.6287 | 16.4, 0.7, 10.2, 5.3, 7.4 |
| **HEAD-MES** | **25.16** | 4/5 | 0.5687 | **77.4**, 2.3, 22.7, 11.5, 11.9 |
| TAIL-MES | 46.45 | 4/5 | 0.0404 | 50.0, 21.5, 22.7, 77.7, 60.3 |
| RANDOM-POOL | 65.14 | 2/5 | 0.0237 | 78.6, 77.7, 22.7, 78.2, 68.6 |

**P1 FAILS.** It required HEAD "near the control's 7.99". Observed 25.16, and
paired per-seed HEAD is worse on **5 of 5**. P2 holds (TAIL fails). P3 holds
decisively (centroids 0.5687 vs 0.0404).

**I reported R2 at n=4 last tick. That was premature and is corrected here.**
The n=4 caveat was recorded, but I led with "R2 fires" anyway.

## What survives, and what must now be scoped

**Survives on both benchmarks:** HEAD < TAIL (25.16 vs 46.45 here; 16.96 vs 43.94
on Borehole), and the centroid split. So the interventional core — *a bad first
step is fatal regardless of the other seven, and a good first step buys most of
what a good trajectory buys* — holds on two benchmarks.

**Must be scoped to Borehole:** the strong form, *"only the first step matters"*.
On Borehole HEAD matched the control (16.96 vs 15.82), so the seven later steps
contributed ~nothing. **On Hartmann they contribute substantially** — HEAD is
worse on every seed.

That also re-scopes h172's actionable result independently of h174's voided arm:
if the later steps do real work on Hartmann, shortening the rollout there should
not be expected to be free, and h174's SC1 firing left that untested anyway.
