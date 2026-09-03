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


def _greedy_base_rollout(ko, roi_candidates, c_H, c_L, best_hf,
                         budget_left, fantasy_mode, device, dtype):
    """
    Follow greedy cost-normalised MES until the cost budget is exhausted,
    returning the best HF value reached. This is the BASE POLICY: what the
    lookahead assumes will happen after the step being scored, and deliberately
    the SAME rule as the default teacher, so the only thing h198 changes is
    WHICH STEP IS TAKEN NOW, not how the future is imagined.
    """
    _b = float(best_hf)
    while budget_left > 0:
        x, ell, _ = compute_joint_mf_mes(ko, roi_candidates, c_H, c_L)
        _c = c_H if ell == 1 else c_L
        if _c > budget_left:
            break
        y = ko.sample_fantasy(x, "LH"[ell], mode=fantasy_mode)
        if ell == 1:
            _b = max(_b, float(y))
        ko = ko.make_fantasy_ko(
            x.unsqueeze(0), torch.tensor([y], device=device, dtype=dtype),
            "LH"[ell])
        budget_left -= _c
    return _b


def choose_regret_lookahead(ko_model, roi_candidates, c_H, c_L,
                            best_sim_hf, steps_left, n_c=8, M=4,
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

    _budget = float(steps_left - 1) * float(c_H)
    _vals = []
    for _ci, _ell in _cands:
        _x = roi_candidates[_ci]
        _acc = []
        for _m in range(int(M)):
            _y = ko_model.sample_fantasy(_x, "LH"[_ell], mode=fantasy_mode)
            _b0 = max(float(best_sim_hf), float(_y)) if _ell == 1 else float(best_sim_hf)
            _ko2 = ko_model.make_fantasy_ko(
                _x.unsqueeze(0),
                torch.tensor([_y], device=device, dtype=dtype), "LH"[_ell])
            _acc.append(_greedy_base_rollout(_ko2, roi_candidates, c_H, c_L,
                                             _b0, _budget, fantasy_mode,
                                             device, dtype))
        _vals.append(float(np.mean(_acc)))

    _best = int(np.argmax(_vals))
    _ci, _ell = _cands[_best]
    _info = {'n_scored': len(_cands),
             'reduced_to_greedy': False,
             'chose_greedy': bool(_ci == int(_idx[0]) // scores.shape[1]
                                  and _ell == int(_idx[0]) % scores.shape[1]),
             'score_spread': float(max(_vals) - min(_vals)),
             'greedy_minus_best': float(_vals[0] - _vals[_best])}
    return roi_candidates[_ci], int(_ell), scores, _info
