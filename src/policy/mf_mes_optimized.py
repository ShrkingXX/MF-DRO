"""
Variant D -- MF-MES with the acquisition ACTUALLY optimized (Takeno et al. 2020,
arXiv:1901.08275, Algorithm 1 line 4).

Why this file exists
--------------------
Our in-pipeline `compute_joint_mf_mes` is handed a 200-point uniform pool and
argmaxes over it (mf_dro.py:2644). Takeno's Section 3.4 says instead:

    "For the acquisition function maximization (argmax in line 4), if the
    candidate space X is a discrete set, we simply calculate the acquisition
    values for all x in X. For a continuous space, popular approaches such as
    DIRECT (Jones et al., 1993) and gradient-based optimizers are applicable."

Hartmann 6D / Currin 2D / Borehole 8D are continuous, so the published method
optimizes; we sample 200 points. Measured handicap on Hartmann 6D with y* held
fixed (so scores are comparable across pool sizes):

    N=200 -> 0.02275   N=1000 -> 0.06344   N=4000 -> 0.09778  (4.30x, still rising)

So the pool costs the acquisition a factor >4 and has not saturated by 4000.
Calling our 200-point version "MF-MES" overstates the baseline's strength.

Second, subtler handicap fixed here: `compute_joint_mf_mes` Thompson-samples
y* FROM the same 200-point pool it then argmaxes over, so the estimate of the
global optimum's value is built from the same sparse set. Takeno estimates y*
over the search space. We draw y* from an independent n_ref reference set.

Optimizer
---------
Two stages, both batched (the MES kernels are vectorized; per-point scipy calls
would be dominated by Python overhead):
  1. dense global stage over n_dense uniform points;
  2. shrinking-ball local refinement from the top n_starts, n_refine rounds of
     n_pert Gaussian perturbations each, radius halving per round.
Search is over the JOINT max across fidelities -- argmax_{x,ell} of the
cost-normalized score -- so the fidelity is recovered at the end, exactly as
Algorithm 1 line 4 specifies.
"""
import torch

from src.policy.mf_dro import (
    _build_hf_proxy_model,
    thompson_sample_y_star,
    _compute_mes_hf_vectorized,
    _compute_mes_lf_vectorized,
)


def optimize_mf_mes(ko_model, bounds, c_H, c_L, K=10, n_ref=2000, n_dense=2000,
                    n_starts=8, n_refine=4, n_pert=32, radius0=0.10, gen=None):
    """Returns (x_raw [d], ell int 0=L/1=H, best_score float)."""
    lo, hi = bounds[0], bounds[1]
    dtype = lo.dtype
    d = lo.numel()
    span = hi - lo

    def _u(n):
        return lo + span * torch.rand(n, d, dtype=dtype, generator=gen)

    hf_proxy = _build_hf_proxy_model(ko_model)
    # y* over the search space, NOT over the proposal pool.
    y_star = thompson_sample_y_star(hf_proxy, _u(n_ref), K=K)

    def acq(X):
        """[N,d] -> [N,2] cost-normalized scores, col0=LF, col1=HF."""
        h = torch.as_tensor(_compute_mes_hf_vectorized(X, hf_proxy, y_star), dtype=dtype)
        l = torch.as_tensor(_compute_mes_lf_vectorized(X, ko_model, y_star, n_quad=32),
                            dtype=dtype)
        return torch.stack([l / c_L, h / c_H], dim=1)

    # --- stage 1: dense global ---
    X = _u(n_dense)
    S = acq(X)
    joint = S.max(dim=1).values                       # argmax over (x, ell)
    k = min(n_starts, X.shape[0])
    top = joint.argsort(descending=True)[:k]
    cur_X, cur_J, cur_S = X[top].clone(), joint[top].clone(), S[top].clone()

    # --- stage 2: shrinking-ball local refinement ---
    radius = radius0
    for _ in range(n_refine):
        P = cur_X.repeat_interleave(n_pert, dim=0)
        P = P + radius * span * torch.randn(P.shape, dtype=dtype, generator=gen)
        P = torch.min(torch.max(P, lo), hi)
        SP = acq(P)
        JP = SP.max(dim=1).values.view(k, n_pert)
        bi = JP.argmax(dim=1)
        rows = torch.arange(k) * n_pert + bi
        better = JP[torch.arange(k), bi] > cur_J
        cur_X[better] = P[rows][better]
        cur_S[better] = SP[rows][better]
        cur_J[better] = JP[torch.arange(k), bi][better]
        radius *= 0.5

    b = int(cur_J.argmax())
    ell = int(cur_S[b].argmax())                      # 0=L, 1=H
    return cur_X[b], ell, float(cur_J[b])
