"""
h152 -- the JOINT information-gain teacher.

The rollout reward in mf_dro.py is labeled (mf_dro.py:1953)

    rtg[tau] = log(b_tau) - log(b_T)

so the total return telescopes to rtg[0] = log(b_0) - log(b_T). b_0 is fixed
at rollout start, therefore

    argmax  (total information gain)   ==   argmin  b_T

The return depends ONLY on the terminal posterior and is PATH-INDEPENDENT:
maximizing joint information gain is a SET-selection problem over the T query
points. The existing teacher -- a per-step argmax of cost-normalised MES
(compute_joint_mf_mes) -- is the GREEDY approximation to that set problem.

This module searches the set problem directly, under a cost cap so that
"minimize b_T" cannot degenerate into "spend more" (rollout_length is fixed
and the rollout itself enforces no budget, so unconstrained argmin b_T is
just 8 HF queries).

At beam_width=1, branch=1 the search reduces EXACTLY to the greedy teacher.
"""
import math
import numpy as np
import torch

from src.policy.mf_dro import (
    compute_joint_mf_mes, _build_hf_proxy_model,
)
from gumbel_thompson import thompson_sample_y_star, fit_gumbel_to_samples


def gumbel_b(ko_for_b, roi_candidates, K_rtg=100,
             use_rtg_grounding=False, real_best_hf=None):
    """
    Replica of simulate_mf_trajectory's private _rollout_gumbel_b closure
    (mf_dro.py:1371), including the grounding clip and the degenerate-variance
    floor that guards scipy's gumbel_r.fit MLE crash. Kept byte-faithful to
    that body so the b this search optimises is the same b the rollout labels
    the trajectory with.
    """
    hf_proxy = _build_hf_proxy_model(ko_for_b)
    y_star_arr = thompson_sample_y_star(hf_proxy, roi_candidates, K=K_rtg)
    if use_rtg_grounding and real_best_hf is not None:
        y_star_arr = np.maximum(y_star_arr, real_best_hf)
    if np.ptp(y_star_arr) < 1e-9:
        b = 1e-12
    else:
        _, b = fit_gumbel_to_samples(y_star_arr)
    return max(b, 1e-12)


def _expand(node_scores, k, cost_cap, cost_so_far, c_H, c_L):
    """
    Top-k (candidate, fidelity) pairs by cost-normalised MES, dropping any
    whose fidelity cost would breach the cap. Returns list of (idx, ell).
    node_scores is [N, 2] (col0=LF, col1=HF), exactly compute_joint_mf_mes's
    `scores`; its flat argmax is the greedy choice, so the top-1 of this list
    IS the greedy choice.
    """
    flat = node_scores.reshape(-1)
    order = torch.argsort(flat, descending=True)
    out = []
    for f in order.tolist():
        ell = f % 2
        c = c_H if ell == 1 else c_L
        if cost_so_far + c > cost_cap + 1e-9:
            continue
        out.append((f // 2, ell))
        if len(out) >= k:
            break
    return out


def beam_search_trajectory(ko_model, roi_candidates, T, c_H, c_L,
                           cost_cap, beam_width=4, branch=4,
                           select_M=1, prune_by="b",
                           K_rtg=100, use_rtg_grounding=False,
                           real_best_hf=None, fantasy_mode='sample',
                           device='cpu', dtype=torch.float64,
                           return_diagnostics=False):
    """
    Cost-constrained beam search over (x, ell) sequences, SELECTED by terminal
    b_T. Mid-search pruning uses accumulated cost-normalised MES as a cheap
    surrogate (b is Thompson-sampled + Gumbel-fit, far too expensive to
    evaluate per node per step); b_T is computed only for the beam_width
    survivors, and the returned trajectory is the argmin over those.

    Returns (x_path [T,d], ell_path [T], info dict).
    """
    # The ELITE node carries the pure-greedy path and is NEVER pruned. Without
    # it a wider beam can prune the greedy path away (measured: B=2,k=2 returned
    # b_T=14.53 against B=1,k=1's 13.62, SC2 FAIL), because pruning compares
    # PARTIAL paths and greedy is only optimal step-by-step, not in prefix. With
    # elitism the final argmin runs over a set that always contains the greedy
    # trajectory, so b_T_beam <= b_T_greedy holds by construction.
    nodes = [dict(ko=ko_model, xs=[], ells=[], cost=0.0, b=None, acc=0.0, elite=True)]

    for _tau in range(T):
        children = []
        for nd in nodes:
            _, _, scores = compute_joint_mf_mes(nd["ko"], roi_candidates, c_H, c_L)
            exp = _expand(scores, branch, cost_cap, nd["cost"], c_H, c_L)
            for j, (idx, ell) in enumerate(exp):
                # j==0 is that node's own greedy (argmax) choice, so the elite
                # parent's j==0 child is the pure-greedy continuation.
                children.append((nd, idx, ell, nd["elite"] and j == 0,
                                 float(scores[idx, ell])))
        if not children:
            break  # cost cap exhausted; trajectory ends short, as BES would

        # Condition every child, then score by the ACTUAL objective b_tau
        # (2.1ms -- cheaper than the MES call, so no surrogate is needed).
        nxt = []
        for nd, idx, ell, is_elite, s_mes in children:
            x = roi_candidates[idx]
            y = nd["ko"].sample_fantasy(x, "LH"[ell], mode=fantasy_mode)
            ko2 = nd["ko"].make_fantasy_ko(
                x.unsqueeze(0),
                torch.tensor([y], device=device, dtype=dtype),
                "LH"[ell])
            nxt.append(dict(
                ko=ko2, xs=nd["xs"] + [x], ells=nd["ells"] + [ell],
                cost=nd["cost"] + (c_H if ell == 1 else c_L),
                b=(gumbel_b(ko2, roi_candidates, K_rtg,
                            use_rtg_grounding, real_best_hf)
                   if prune_by == "b" else None),
                acc=nd.get("acc", 0.0) + s_mes,
                elite=is_elite,
            ))

        elite = [n for n in nxt if n["elite"]]
        _key = (lambda n: n["b"]) if prune_by == "b" else (lambda n: -n["acc"])
        rest = sorted((n for n in nxt if not n["elite"]), key=_key)
        nodes = elite + rest[:max(0, beam_width - len(elite))]

    # C1 (h152b). Only the (x, ell) SEQUENCE transfers to the rollout -- the
    # fantasy y's are redrawn there. So the quantity that actually transfers is
    # E_y[log b_T | path], and scoring each survivor by its OWN single realised
    # b_T is an argmin over B noisy estimates: a winner's curse that measured
    # +0.6680 rtg units in 21/21 states (h152 Stage 0). select_M > 1 replays
    # each survivor's path M times with FRESH fantasies from the ORIGINAL model
    # and selects on the mean, which is unbiased for the transferable quantity.
    if select_M > 1:
        bs = []
        for nd in nodes:
            lb = []
            for _ in range(select_M):
                cur = ko_model
                for x, e in zip(nd["xs"], nd["ells"]):
                    yy = cur.sample_fantasy(x, "LH"[e], mode=fantasy_mode)
                    cur = cur.make_fantasy_ko(
                        x.unsqueeze(0),
                        torch.tensor([yy], device=device, dtype=dtype), "LH"[e])
                lb.append(math.log(gumbel_b(cur, roi_candidates, K_rtg,
                                            use_rtg_grounding, real_best_hf)))
            bs.append(float(np.mean(lb)))
    else:
        bs = [math.log(nd["b"]) if nd["b"] is not None
              else math.log(gumbel_b(nd["ko"], roi_candidates, K_rtg,
                                     use_rtg_grounding, real_best_hf))
              for nd in nodes]
    # COST CONSTRAINT (fixed after a crash). It was a hard cap inside _expand
    # set to greedy's MEAN rollout cost -- but the greedy path's own realised
    # cost exceeds its mean on roughly half of states, so the elite was priced
    # out and pruned, leaving b_greedy=None. The cap now binds at SELECTION
    # time against the elite's OWN realised cost: only paths that spent no more
    # than greedy actually spent may compete, and the elite is always eligible.
    # Tighter than the mean-cap and feasible by construction.
    e_idx = next((i for i, nd in enumerate(nodes) if nd["elite"]), None)
    if e_idx is None:                      # defensive; unreachable with cap=inf
        elig = list(range(len(nodes))); b_greedy = None
    else:
        ec = nodes[e_idx]["cost"]
        elig = [i for i, nd in enumerate(nodes)
                if nd["elite"] or nd["cost"] <= ec + 1e-9]
        b_greedy = bs[e_idx]
    best = elig[int(np.argmin([bs[i] for i in elig]))]
    win = nodes[best]

    # bs is in LOG units throughout (rtg's own units); b_T/b_T_greedy are
    # exponentiated back so callers reading a raw Gumbel scale are unaffected.
    info = dict(b_T=math.exp(bs[best]), logb_T=bs[best],
                b_T_all=[math.exp(v) for v in bs],
                b_T_greedy=(math.exp(b_greedy) if b_greedy is not None else None),
                logb_T_greedy=b_greedy, n_eligible=len(elig),
                cost=win["cost"], n_survivors=len(nodes),
                T_actual=len(win["xs"]), won_by_elite=bool(win["elite"]),
                ells=list(win["ells"]))
    return torch.stack(win["xs"]), win["ells"], info
