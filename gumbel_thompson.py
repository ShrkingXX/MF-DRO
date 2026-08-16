"""
Shared, dependency-free Thompson-sampling + Gumbel-fit helpers.

Extracted from validation/phase1_thompson_gumbel.py so that src/policy/dro.py
can import these functions for the "joint" RTG schema without creating a
circular import (validation/phase1_thompson_gumbel.py -> validation/
phase1_gumbel_quality.py -> src.policy.dro would otherwise cycle back to
dro.py itself). This module has no project-internal imports besides numpy/
torch/scipy, so it is safe to import from anywhere.
"""
import numpy as np
import torch
from scipy.stats import gumbel_r


def thompson_sample_y_star(model, roi_candidates: torch.Tensor, K: int = 500) -> np.ndarray:
    """
    Draw K exact samples of y* = max_x f(x) using Thompson sampling: a joint
    rsample() from the GP posterior at roi_candidates (which respects the
    posterior's true cross-candidate covariance), then take the max of each
    sample. No independence assumption anywhere in this procedure.
    """
    with torch.no_grad():
        posterior = model.posterior(roi_candidates)
        samples = posterior.rsample(torch.Size([K])) # [K, N, 1]
        samples = samples.reshape(K, -1)
        y_star = samples.max(dim=-1).values.cpu().numpy()
    return y_star


def fit_gumbel_to_samples(samples: np.ndarray):
    """
    Fit Gumbel(a, b) to samples by MLE via scipy.stats.gumbel_r (the
    right-skewed / maximum Gumbel; a=loc, b=scale).

    Returns: (a: float, b: float)
    """
    a, b = gumbel_r.fit(samples)
    return float(a), float(b)
