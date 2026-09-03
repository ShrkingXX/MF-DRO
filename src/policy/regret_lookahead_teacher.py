"""
h198 -- a teacher that optimises the TASK, not the label.

Every other teacher in this repo optimises an information or acquisition
criterion: MES (per-step cost-normalised info gain), UCB/EI/Thompson
(posterior-greedy), or -- in the case of h152's beam -- ``argmin b_T``, which is
the exact optimum of the LABELLED reward. The last one is the problem this
module answers: perfecting the reward we wrote down is a nominal standard, and a
teacher that achieves it inherits whatever is wrong with the label.

h145's "oracle" is not a counter-example: its path is
``x_tau = x_start + (x* - x_start) * tau/(T-1)`` with ``x_start ~ U(domain)``,
so it only ARRIVES at x* -- every intermediate step is a point on an arbitrary
line segment, and its tau=0 action is uniform noise. Since the confirmed
mechanism is that the DT emits its teacher's tau=0 action MEAN, that
construction destroyed precisely the signal the DT consumes.

WHAT THIS TEACHER OPTIMISES

The rollout acquisition function (Lam, Willcox & Wolpert 2016, "Bayesian
optimization with a finite budget"): for each candidate ``(x, ell)``, condition
on a fantasy ``y``, then follow a cheap BASE POLICY (greedy cost-normalised MES,
i.e. the existing teacher) to the cost horizon, and score by the EXPECTED
TERMINAL BEST HF VALUE. Average over M fantasies; pick the argmax.

Maximising expected terminal best value is minimising expected final simple
regret, because the two differ by ``E[y*]``, which the choice at this step does
not control. Terminal best value is used directly rather than a regret estimate
because it needs no y* draw at all -- one less estimator, and no dependence on
the Gumbel fit whose noise dominated the b measurements elsewhere.

REDUCTION TO GREEDY (SC1). At ``n_c=1`` the candidate set is the greedy argmax
alone, so the argmax over it is the greedy choice regardless of M, and this
teacher is greedy MES step for step. Asserted in sanity.py, not just claimed --
the beam's analogous identity check caught a real elitism bug.

WINNER'S CURSE (SC2). Scoring each candidate on ONE fantasy and taking the
argmax selects lucky draws, not good decisions. The beam's first version lost
its ENTIRE apparent advantage (+0.6680) to exactly this. M>1 is therefore not a
tuning knob but a correctness requirement, and ``score_spread`` is returned so
the curse can be reported rather than assumed absent.
"""
import numpy as np
import torch

from src.policy.mf_dro import compute_joint_mf_mes


def _draw_y(ko, x, ell, fantasy_mode, oracle_f):
    """
    h199: the ONE substitution separating the oracle ceiling from h198. With
    oracle_f given, the imagined future is evaluated on the TRUE function instead
    of drawn from the GP posterior; everything else about the decision rule is
    unchanged, so (h199 - h198) isolates fantasy quality and nothing else.
    NOT a method -- f_H is unavailable at run time in any real setting.
    """
    if oracle_f is None:
        return ko.sample_fantasy(x, "LH"[ell], mode=fantasy_mode)
    return float(oracle_f["H" if ell == 1 else "L"](x.reshape(1, -1)).reshape(-1)[0])


def _greedy_base_rollout(ko, roi_candidates, c_H, c_L, best_hf,
                         steps_left, fantasy_mode, device, dtype, oracle_f=None):
    """
    Follow greedy cost-normalised MES for exactly ``steps_left`` STEPS,
    returning the best HF value reached. This is the BASE POLICY: what the
    lookahead assumes will happen after the step being scored, and deliberately
    the SAME rule as the default teacher, so the only thing h198 changes is
    WHICH STEP IS TAKEN NOW, not how the future is imagined.

    STEPS, not a cost budget. The first version spent a budget of
    (steps_left-1)*c_H, which on Borehole's 2:1 ratio lets an all-LF
    continuation run FOURTEEN steps inside an eight-step rollout -- imagining a
    horizon the trajectory does not have, and making the lookahead's cost
    scale with the cost ratio for no principled reason. simulate_mf_trajectory
    is step-bounded (rollout_length) and enforces no budget, so the base policy
    must be too.
    """
    _b = float(best_hf)
    _n = int(steps_left)
    for _i in range(_n):
        x, ell, _ = compute_joint_mf_mes(ko, roi_candidates, c_H, c_L)
        _last = (_i == _n - 1)
        # On the LAST step the conditioned GP is never read again, and on an LF
        # last step the fantasy value cannot move _b either (only HF observations
        # do). Both are therefore skipped. This is not an approximation: the
        # returned _b is bit-identical, the work was simply discarded. Measured
        # 16 wasted conditionings and up to 16 wasted fantasy draws per teacher
        # decision.
        if _last and ell != 1:
            break
        y = _draw_y(ko, x, ell, fantasy_mode, oracle_f)
        if ell == 1:
            _b = max(_b, float(y))
        if _last:
            break
        ko = ko.make_fantasy_ko(
            x.unsqueeze(0), torch.tensor([y], device=device, dtype=dtype),
            "LH"[ell])
    return _b


def choose_regret_lookahead(ko_model, roi_candidates, c_H, c_L,
                            best_sim_hf, steps_left, n_c=8, M=4,
                            base_pool=150, max_horizon=0, oracle_f=None,
                            fantasy_mode='sample', device='cpu',
                            dtype=torch.float64):
    """
    Pick the (x, ell) maximising the expected terminal best HF value.

    Returns (x_tau, ell_tau, scores, info). ``scores`` is compute_joint_mf_mes's
    own score matrix, returned unchanged so every downstream consumer
    (teacher_action_stats, diagnostics) behaves exactly as under the MES teacher.
    """
    _x_g, _e_g, scores = compute_joint_mf_mes(ko_model, roi_candidates, c_H, c_L)
    if n_c <= 1 or steps_left <= 1:
        # SC1: the candidate set is the greedy argmax alone. Also the last step,
        # where there is no future for a lookahead to differ over.
        return _x_g, _e_g, scores, {'n_scored': 1, 'reduced_to_greedy': True}

    # Top n_c (candidate, fidelity) pairs by the base criterion. Restricting the
    # set is what makes this affordable; it also guarantees the greedy choice is
    # always among the options, so the lookahead can only re-rank, never lose it.
    _flat = scores.reshape(-1)
    _k = int(min(int(n_c), _flat.numel()))
    _idx = torch.topk(_flat, _k).indices
    _cands = [(int(i) // scores.shape[1], int(i) % scores.shape[1]) for i in _idx]

    # TRUNCATED ROLLOUT. Every candidate is followed by the SAME base policy, so
    # their imagined futures converge as the horizon grows and the discriminative
    # signal is concentrated in the early steps. Cost is linear in the horizon
    # (n_c * M * horizon MES calls, and MES has a ~6 ms floor that pool-size cuts
    # cannot get under), so truncating is the only lever with real leverage.
    # max_horizon=0 means "no truncation" and reproduces the untruncated teacher.
    _horizon = int(steps_left) - 1
    if max_horizon and _horizon > int(max_horizon):
        _horizon = int(max_horizon)

    # COMMON RANDOM NUMBERS. Scoring each candidate against its OWN independent
    # fantasy stream makes the comparison unpaired, and the argmax then partly
    # selects whichever candidate drew the luckiest futures. Measured before
    # this was added: the argmax was UNSTABLE between M=4 and M=8, and greedy
    # came out the worst of 8 candidates by a margin equal to the entire score
    # spread -- the signature of selection noise, not of a real ranking.
    # Seeding every candidate's m-th replication identically makes the
    # comparison PAIRED: all candidates face the same futures, so differences
    # reflect the decision rather than the draw. This is the same failure the
    # beam hit (winner's curse, +0.6680); CRN attacks it at lower cost than
    # raising M, which only shrinks noise as 1/sqrt(M).
    #
    # The RNG state is saved and restored around the whole search so the
    # teacher's internal deliberation does not perturb the rollout's own
    # fantasy draws. One int is consumed first, so the CRN seed still varies
    # from step to step and rollout to rollout.
    # The base rollout's MES calls dominate cost (n_c * M * horizon of them per
    # teacher step; measured 127x the greedy teacher at n_c=8, M=4, which is
    # 16.7 h/seed and not runnable). They are evaluated on a SUBSAMPLED pool,
    # because the base rollout is the IMAGINED FUTURE used to rank this step's
    # options -- not the decision itself, which still ranges over the full pool
    # via the top-n_c selection above. Subsampling the future costs resolution
    # in a quantity that is already a coarse approximation; subsampling the
    # decision would not be acceptable, and is not done.
    _bp = roi_candidates
    if base_pool and int(base_pool) < roi_candidates.shape[0]:
        _bp = roi_candidates[torch.randperm(roi_candidates.shape[0])[:int(base_pool)]]

    _crn = int(torch.randint(0, 2**31 - 1, (1,)).item())
    _rng_state = torch.get_rng_state()

    _vals = []
    for _ci, _ell in _cands:
        _x = roi_candidates[_ci]
        _acc = []
        for _m in range(int(M)):
            torch.manual_seed(_crn + _m)
            _y = _draw_y(ko_model, _x, _ell, fantasy_mode, oracle_f)
            _b0 = max(float(best_sim_hf), float(_y)) if _ell == 1 else float(best_sim_hf)
            _ko2 = ko_model.make_fantasy_ko(
                _x.unsqueeze(0),
                torch.tensor([_y], device=device, dtype=dtype), "LH"[_ell])
            _acc.append(_greedy_base_rollout(_ko2, _bp, c_H, c_L,
                                             _b0, _horizon, fantasy_mode,
                                             device, dtype, oracle_f))
        _vals.append(float(np.mean(_acc)))

    torch.set_rng_state(_rng_state)

    _best = int(np.argmax(_vals))
    _ci, _ell = _cands[_best]
    _info = {'n_scored': len(_cands),
             'reduced_to_greedy': False,
             'chose_greedy': bool(_ci == int(_idx[0]) // scores.shape[1]
                                  and _ell == int(_idx[0]) % scores.shape[1]),
             'score_spread': float(max(_vals) - min(_vals)),
             'greedy_minus_best': float(_vals[0] - _vals[_best])}
    return roi_candidates[_ci], int(_ell), scores, _info
