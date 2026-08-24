# H5 analysis — REFUTED on the real tests, and the refutation is the finding

## Result

12 resampled candidate pools, `rtg_conditioning="token"`, one variable changed.

| measurement | A: with GP features (d+5) | B: coords only (d) | locked prediction |
|---|---|---|---|
| (1) swapping `h` changes argmax | **0/12 = 0.0%** | **0/12 = 0.0%** | >30% in B — **FAIL** |
| (2) argmax == argmax(mu_H) | 8/12 = 66.7% | **0/12 = 0.0%** | <30% in B — check ok |
| (3) RTG sweep moves argmax | 2/12 = 16.7% | 1/12 = 8.3% | rise — **FAIL** |

**H5 NOT SUPPORTED.** The manipulation worked exactly as intended — the `mu_H`
shortcut was fully removed (66.7% -> 0.0%) — and the head *still* ignores `h`
completely, and became *less* RTG-sensitive, not more.

## This is pre-registered outcome #2, not an unanticipated result

The protocol stated in advance:

> "2 holds but 1 and 3 fail -> the head becomes *arbitrary*, not `h`-driven.
> H5 refuted in its useful form; the conditioning signal in `h` is genuinely too
> weak to rank candidates, which would be a substantive negative result about
> return-conditioned BO."

That is precisely what happened, which is why it is worth taking seriously
rather than explaining away.

## The stronger claim this licenses

Denying the score head every GP feature — leaving it *only* candidate
coordinates, so that any non-arbitrary ranking **must** flow through `h` — does
not make it use `h`. Combined with the converging prior evidence:

- swapping `h` for a different state's hidden vector: argmax unchanged 12/12 (here)
- batch-mean or shuffled `h`: argmax changed only ~8% of the time (earlier)
- state perturbation at 1x batch std: argmax unchanged, score corr 0.9997 (earlier)
- RTG sweep 0.1x-10x: argmax essentially pinned in every configuration tried

**Within a single trained model, MF-DRO's proposal is very nearly independent of
its own conditioning** — state, RTG and BTG all arrive through `h`, and `h` is
close to inert at the point of decision.

### Reconciling this with the fact that queries *do* move

`x_t_trace` has per-dimension std 0.166-0.213 across a run, so the policy is
plainly not emitting one constant point. The reconciliation matters:

> The network is **retrained from scratch-ish every BO iteration**. Within an
> iteration it is close to a fixed function of the candidate set; across
> iterations it changes because its *weights* were re-fit on new rollouts.

So MF-DRO's apparent adaptivity comes from **re-fitting**, not from
**conditioning**. It is behaving as a per-iteration acquisition function that
happens to be parameterised by a transformer — not as a return-conditioned
policy. That is a substantive statement about the method, and it explains the
entire failed sequence of conditioning-side interventions (H4's AdaLN included):
they were all trying to strengthen a pathway that is not being used at all.

## Caveat, stated plainly

Same confound as H4: the probe trains 10 epochs on a single rollout batch. A
model trained far longer might begin to use `h`. This measures the regime MF-DRO
*actually runs in* (`num_epochs=10` is the production setting), so it is the
operationally relevant answer — but it is not a claim about transformers in
general, or about this architecture at large training scale.

## What to do next

Stop trying to strengthen conditioning. Two honest directions:

1. **Test the re-fitting interpretation directly.** Freeze the DT weights after
   iteration k and continue the BO run. If regret behaviour is largely
   unchanged, the conditioning contributes ~nothing and the "DT policy" framing
   is not doing work. Cheap, decisive, and it would be the sharpest result in
   the project.
2. **If (1) shows the DT is inert, the honest write-up is a negative result**:
   within this frame, the return-conditioned policy adds little over the
   acquisition it distills, and the freeze was a compound of implementation
   defects (target leakage chief among them) rather than a conditioning-strength
   problem. PROTOCOL.md explicitly permits this outcome.
