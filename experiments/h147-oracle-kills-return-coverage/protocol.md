# h147 — Does the oracle teacher destroy RETURN COVERAGE? (theory-predicted)

STATUS: LOCKED before computing. ZERO NEW COMPUTE.
TYPE: CONFIRMATORY.

## The theory, already in our own literature notes

`literature/rcsl-necessary-conditions.md` frames MF-DRO as **return-conditioned
supervised learning** (Brandfonbrener et al., NeurIPS 2022, arXiv:2206.01079).
Theorem 2 needs two conditions, and the bound is

    J(pi_f^RCSL) - J(pi~_f)  <=  (C_f / alpha_f) H^2 sqrt(2 L(pi~))

    C_f      bounded occupancy mismatch   P_pi(s) / P_beta(s) <= C_f
    alpha_f  RETURN COVERAGE              P_beta(g = f(s) | s) >= alpha_f

**Every intervention this project tried on the conditioning side changed `f`.**
The note lists seven, all null, and explains why they had to be.

**h145 changed something different: the behaviour policy `beta` itself.** An
oracle teacher is expert-only data. In RCSL, expert-only data is the textbook way
to **destroy return coverage** — the model never observes what a low return looks
like at a given state, so `alpha_f` shrinks and the bound degrades as `1/alpha_f`.

**So the theory predicts h145's result**, and predicts it through a mechanism
nobody in this project has measured: not "the teacher is too good", but "the
teacher's returns are too uniform to condition on".

## Predictions (locked)

Borehole, seeds 42-46, ORACLE (h145) vs CONTROL (h83), paired.

**P1 (PRIMARY).** `rtg_frac_between_traj_var_per_iter` — the share of RTG variance
lying BETWEEN trajectories rather than within them — is **lower** in the oracle
runs, effect >= 1.0. This is the closest available proxy for return coverage: if
every trajectory earns a similar return, between-trajectory variance collapses and
conditioning carries no information. FALSIFIED if effect < 1.0.

**P2 (no direction registered).** `neg_rtg_frac_per_iter`, reported whatever it
shows.

## The complication I must state first, because it cuts against P1

**h145's RTG is the `mes_entropy` label**, `log(b_tau) - log(b_T)` — an
information-gain quantity computed from the GP, **not** the trajectory's y-quality.
An oracle teacher improves *where the trajectory goes*, which need not make its
*information gain* uniform. So P1 is NOT guaranteed by the design, and a failure
would be genuinely informative: it would say the oracle collapsed the policy
WITHOUT collapsing return coverage, which is a different mechanism than the theory
supplies.

**This is the honest test of a borrowed explanation.** The theory fits the shape of
our result; that is not evidence it is the operative mechanism here. P1 is the
measurement that decides.

## What this could RETRACT

**The RCSL framing as the explanation of h145.** If P1 fails, return coverage is
not what the oracle destroyed, and importing Brandfonbrener's account would be
exactly the error this project has made four times already — a mechanism story
that fits and does not predict. `literature/rcsl-necessary-conditions.md` would
need a note saying it explains the *conditioning-side* nulls and **not** the
teacher-side result.
