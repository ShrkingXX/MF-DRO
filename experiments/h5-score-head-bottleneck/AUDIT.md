# Audit of H5 — the original probe was invalid; the conclusion survives re-measurement

**Status: EXPLORATORY** (unplanned self-audit, triggered by a `STATE-DIAG` line
reading `uniq_tau0_states=10` for 200 trajectories).

## The defect

H5's h-sensitivity test drew its comparison state as

    st_other = batch[(p + 7) % len(batch)]["states"][0]     # p = 0..11

so it used indices **7..18**. The batch is built
`for ko in ko_ensemble: for _ in range(rollouts_per_model)` with
`rollouts_per_model=20`, so **indices 0..19 all come from model 0 and share a
bit-identical τ=0 state**. Measured directly:

| check | result |
|---|---|
| H5's actual comparisons (idx 7..18) identical to `batch[0]` | **12/12** |
| across model blocks (idx 20, 40, …) identical to `batch[0]` | **0/9** |
| unique τ=0 states in a 200-trajectory batch | **10** |

**H5 swapped a state for itself.** Its "argmax unchanged 12/12" is exactly what
feeding the identical state must produce, and carried no information about
whether the score head reads `h`.

## The corrected measurement

Comparison states drawn from *different* ensemble-model blocks (verified
distinct, 0/9 identical):

    argmax changed when h comes from a genuinely different state : 0/12 = 0.0%
    mean score-vector correlation across states                  : 1.000000

**The conclusion survives.** The score head really is insensitive to `h` — a
correlation of `1.000000` between score vectors computed from different states
is stronger evidence than the original probe could ever have given.

## What must be said carefully

The claim "the score head barely reads `h`" is **true**, but until now it rested
on an **invalid probe**. Everything I built on it over the last two ticks — the
H19 conclusion, the intervention-ladder figure, the `to_human` report's lead —
was correct by luck, not by evidence. It is now correctly evidenced.

Also note: H8/H19's RTG results are **unaffected** by this defect. Those probes
varied the RTG target, not the state, so they never used the broken swap.

## A second hypothesis, refuted in passing

I suspected the insensitivity was **feature-scale domination**: one candidate
feature so much larger in spread that it fixes the ranking regardless of the
state-dependent coefficients. Measured contribution `|w_f| · sd_k(cf[:,f])`:

| feature | share of ranking spread |
|---|---|
| `mu_H` | 31.8% |
| `mu_L` | 22.8% |
| `x[1]` | 13.8% |
| `dist_inc` | 8.0% |
| others | ≤ 6.7% each |

No domination — `mu_H`'s across-candidate sd is **1.0×** the median feature's.
And `argmax(score) = 22` vs `argmax(mu_H) = 152`, so the old finding that the
score tracks `mu_H` on 67–75% of pools **does not reproduce** post-fix.

## Lesson

When a probe compares "two different X", assert that they are different **inside
the probe**. H5 would have failed a one-line assertion. This is the third
instrument defect this phase (H13's dead-signal metric, H19's diversity
signature, now H5's state swap) — all three found by checking the instrument
against the data rather than trusting it.

---

# WHY: `coef_head` is a constant function on the states inference actually visits

The corrected probe's score-vector correlation of **exactly 1.000000** was too
clean to be indifference. Measuring `w = coef_head(h)` across the 10 genuinely
distinct τ=0 states:

| quantity | value |
|---|---|
| pairwise cosine(w_i, w_j) | min **0.99999224**, mean 0.99999793 |
| ‖w‖ across states (min / max) | 1.5974 / 1.5983 — ratio **1.001×** |
| singular values of W | [**5.05324**, 0.00628, 0.00231, 0.0015] |
| sv₁ / Σsv | **0.997691** |
| bias across states | −0.5485 / −0.5468 |

`W` is 10 rows of norm ≈1.597; ten *identical* such rows would give
sv₁ = √10 · 1.597 = 5.05. That is what we observe.

**`coef_head` emits the same coefficient vector for every state the policy
actually encounters.** Not a fixed direction with a state-dependent gain — a
constant. The ranking is therefore state-invariant *by construction of the
learned head*, which is why no conditioning-side intervention could ever have
moved it.

## And why the head is constant there

The states are barely distinguishable in the first place. From `STATE-DIAG` on
the same batch:

    ref_block_std ACROSS trajectories at tau=0 : 0.007612
    ref_block_std ACROSS tau WITHIN a trajectory: 0.016865

Real inference states vary **2.2× less** than the fantasy states *within* a
single rollout, and there are only **10 unique τ=0 states per 200-trajectory
batch** — one per ensemble member, all views of the *same real dataset*.

So at any real iteration the policy is handed essentially one state. There is
nothing to condition *on*. This is not an architecture that refuses to
condition; it is an architecture given a degenerate conditioning input.

## This is the mechanistic form of the project's central claim

`findings.md` already said **"MF-DRO re-fits rather than conditions."** That was
a behavioural description. The mechanism is now explicit:

> At each real iteration there is exactly one state, drawn from a distribution
> with almost no spread, so `coef_head` is constant on it and the emitted
> ranking cannot depend on the state, the RTG, the BTG, or the history. All
> observed adaptation therefore comes from **re-fitting the weights between
> iterations** — which H6/H7 measured as changing ~18% of decisions and buying
> ~0 regret.

Note the earlier `[W-DIAG] across-batch VAR(w)` numbers (0.004–0.018) are not in
conflict: that variance is dominated by the τ>0 **fantasy** states, which do
differ substantially. Variation exists in the head — just not over the inputs
inference ever presents.

---

# Does the state channel work ACROSS real iterations? No.

The "constant `coef_head`" result above compared the 10 **ensemble members** at a
single iteration — states that are nearly identical by construction. The
practically important question is different: over a real run, the state changes a
great deal as data accumulates. Does the head respond *then*?

Captured all 12 real-iteration states from an actual MF-DRO run and evaluated the
final weights on each:

| quantity | across ensemble members | **across real iterations** |
|---|---|---|
| mean pairwise state L2 | ~0 (10 unique, tiny spread) | **1.4968** |
| pairwise cosine(w_i, w_j), min | 0.99999224 | 0.99936628 |
| implied rotation of `w` | ~0° | **2.04°** |
| ‖w‖ ratio | 1.001× | 1.0230× |
| sv₁/Σsv | 0.997691 | 0.973401 |
| **argmax moved** | **0/12** | **0/12** |
| distinct argmaxes per pool | 1.00 | **1.00 (max 1)** |

**A state change of L2 = 1.4968 — the full spread of a real run — rotates the
coefficient vector by 2.04° and changes the decision on 0 of 12 candidate
pools.** The head is not literally constant across iterations; its response is
simply far too small to move an argmax over 200 candidates.

So the state channel is non-functional at decision level **both within and
across** iterations. The claim is broader than the ensemble-member result
established, not narrower.

## RETRACTION of this script's own auto-verdict

The script printed **"NARROW: the head DOES vary across real iterations"**. That
verdict is **wrong** and is retracted. It keyed on a cosine threshold
(`min > 0.9999`) rather than on the argmax measurement that actually decides the
question — and the argmax says 0/12.

This is the **fourth** auto-printed verdict of this phase to be wrong:

| script | verdict keyed on | decisive quantity |
|---|---|---|
| H11 | "all arms ~0%" without requiring the manipulation to pass | manipulation check |
| H13 | LF-nonzero fraction, which saturated at 100% in both arms | mean/CV of `rtg[0]` |
| H19 | a diversity signature omitting query locations | signature *with* locations |
| this | cosine similarity of `w` | **argmax movement** |

**Lesson: a script's auto-verdict must key on the decisive measurement, not on a
proxy that correlates with it.** Every one of these proxies looked reasonable
when written and produced a confident, wrong sentence when run.

---

# Localising the attenuation: it is DISTRIBUTED, not one broken module

Relative spread (mean pairwise distance / mean norm) of each representation
across the 12 real-iteration states:

| stage | relative spread | attenuation |
|---|---|---|
| state `s` | 0.215545 | — |
| hidden `h` | 0.074498 | **0.346×** (encoder) |
| coefficients `w` | 0.021864 | **0.294×** (head) |

End-to-end `s → w`: **0.101×**, i.e. a **~10× attenuation** of state variation,
split almost evenly between the transformer encoder (3×) and `coef_head` (3×).

**There is no single broken layer.** The signal decays through the stack, which
is why interventions aimed at one module (H4's AdaLN conditioning, H5's feature
denial) could not recover it — each addresses at most half of a compounding loss.

## The fidelity head is state-invariant too

Holding the weights fixed and varying only the state across all 12 real
iterations:

    fidelity_head p : min 0.1248, max 0.1286  (spread 0.0038)

So *both* readouts are effectively state-invariant. The `fid_mean` values that
visibly move during a run (0.27–0.58 in the H17 logs) move because the network is
**retrained** between iterations — not because it responds to its input. That is
the same re-fit-not-condition signature, now shown for the fidelity head as well
as the location head.

*(This script still prints the stale "NARROW" auto-verdict, already retracted
above; the argmax measurement is decisive and reads 0/12.)*
