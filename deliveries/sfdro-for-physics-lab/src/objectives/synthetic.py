from __future__ import annotations

import math
import numpy as np

# # Define test functions (no changes needed)
# def ackley_function(x):
#     a, b, c = 20, 0.2, 2 * np.pi
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     sum1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
#     sum2 = -np.exp(np.mean(np.cos(c * x)))
#     return -(sum1 + sum2 + a + np.exp(1))  # Negative for maximization

# def rosenbrock_function(x):
#     """Rosenbrock function (minimization)"""
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     result = 0
#     for i in range(len(x) - 1):
#         result += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
#     # Convert to maximization
#     return -result

# def levy_function(x):
#     """Levy function (minimization)"""
#     # Ensure input is numpy array for calculations
#     x = np.array(x)
#     w = 1 + (x - 1) / 4
#     term1 = np.sin(np.pi * w[0])**2
#     term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
#     term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
#     # Convert to maximization
#     return -(term1 + term2 + term3)


import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional

# Define constants used in Ackley
ACKLEY_A = 20.0
ACKLEY_B = 0.2
ACKLEY_C = 2 * torch.pi


class Ackley(SyntheticTestFunction):
    r"""Ackley synthetic test function.

    d-dimensional function usually evaluated on the hypercube
    `[-32.768, 32.768]^d` or `[-5, 5]^d`. The function has many local
    minima and one global minimum at `f(x*) = 0` located at `x* = (0, ..., 0)`.

    f(x) = -a * exp(-b * sqrt(1/d * sum_{i=1}^d x_i^2)) -
           exp(1/d * sum_{i=1}^d cos(c * x_i)) + a + exp(1)
    """

    dim = 2 # Default dimension
    _optimizers = [(0.0, 0.0)] # Global minimum at the origin
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Ackley function.

        Args:
            dim: Dimension of the function.
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-5, 5]^d.
        """
        self.dim = dim
        if bounds is None:
            # Using bounds from the original code context
            self._bounds = [(-5.0, 5.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(0.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Ackley function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Ensure calculations happen on the correct device and dtype as X
        a = torch.tensor(ACKLEY_A, device=X.device, dtype=X.dtype)
        b = torch.tensor(ACKLEY_B, device=X.device, dtype=X.dtype)
        c = torch.tensor(ACKLEY_C, device=X.device, dtype=X.dtype)
        exp_1 = torch.exp(torch.tensor(1.0, device=X.device, dtype=X.dtype))
        inv_dim = torch.tensor(1.0 / self.dim, device=X.device, dtype=X.dtype)

        # Calculate terms - operate on the last dimension (dim=-1)
        term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(X**2, dim=-1)))
        term2 = -torch.exp(torch.mean(torch.cos(c * X), dim=-1))

        # Standard minimization form
        result = term1 + term2 + a + exp_1
        return result


class Rosenbrock(SyntheticTestFunction):
    r"""Rosenbrock synthetic test function.

    d-dimensional function usually evaluated on the hypercube `[-5, 10]^d`.
    Its global minimum `f(x*) = 0` is located at `x* = (1, ..., 1)`. For d=2,
    often evaluated on `[-2, 2]^2` or similar.

    f(x) = sum_{i=1}^{d-1} [100 * (x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    """

    dim = 2 # Default dimension
    _optimizers = [(1.0, 1.0)] # Global minimum at (1, 1, ...)
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Rosenbrock function.

        Args:
            dim: Dimension of the function (must be >= 2).
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-2, 2]^d.
        """
        if dim < 2:
            raise ValueError("Rosenbrock function dimension must be >= 2")
        self.dim = dim
        if bounds is None:
            # Using bounds from the original code context
            self._bounds = [(-2.0, 2.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(1.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Rosenbrock function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Standard calculation using tensor slicing
        # Calculate sum_{i=0}^{d-2} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
        # X[..., :-1] selects all but the last element along the dim dimension
        # X[..., 1:] selects all but the first element along the dim dimension
        term1 = 100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2
        term2 = (X[..., :-1] - 1.0) ** 2
        result = torch.sum(term1 + term2, dim=-1) # Sum across the feature dimension

        return result


class Levy(SyntheticTestFunction):
    r"""Levy synthetic test function.

    d-dimensional function usually evaluated on the hypercube `[-10, 10]^d`.
    Its global minimum `f(x*) = 0` is located at `x* = (1, ..., 1)`.

    f(x) = sin^2(pi * w_1) +
           sum_{i=1}^{d-1} (w_i - 1)^2 * [1 + 10 * sin^2(pi * w_i + 1)] +
           (w_d - 1)^2 * [1 + sin^2(2 * pi * w_d)]
    where w_i = 1 + (x_i - 1) / 4
    """

    dim = 2 # Default dimension
    _optimizers = [(1.0, 1.0)] # Global minimum at (1, 1, ...)
    optimal_value = 0.0 # Value at the global minimum

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = True, # True matches original code's maximization goal
        bounds: Optional[list[tuple[float, float]]] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize Levy function.

        Args:
            dim: Dimension of the function.
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [-10, 10]^d.
        """
        self.dim = dim
        if bounds is None:
             # Using bounds from the original code context
            self._bounds = [(-10.0, 10.0)] * self.dim
        else:
            self._bounds = bounds
        # Ensure optimizer matches dimension
        self._optimizers = [(1.0,) * self.dim]
        self.shift = shift

        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Levy function (standard minimization form).

        Args:
            X: A (batch_shape) x d tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
             raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        if self.shift is not None:
            # Shift the input tensor if a shift is provided
            X = X - self.shift

        # Calculate w = 1 + (x - 1) / 4
        w = 1.0 + (X - 1.0) / 4.0

        # Calculate terms using tensor slicing
        term1 = torch.sin(torch.pi * w[..., 0]) ** 2

        w_mid = w[..., :-1] # w_1 to w_{d-1}
        term2_sum = (w_mid - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(torch.pi * w_mid + 1.0) ** 2)
        term2 = torch.sum(term2_sum, dim=-1) # Sum across the feature dimension

        w_last = w[..., -1] # w_d
        term3 = (w_last - 1.0) ** 2 * (1.0 + torch.sin(2.0 * torch.pi * w_last) ** 2)

        result = term1 + term2 + term3
        return result

class CurrinExpHF(SyntheticTestFunction):
    r"""Currin Exponential function, high-fidelity (single-fidelity) form.

    2-dimensional benchmark from Currin et al. (1988), commonly used as the
    high-fidelity member of a two-fidelity pair in the multi-fidelity BO
    literature (e.g. Song et al. 2019). Only the high-fidelity function is
    implemented here -- the low-fidelity pair is not needed.

    Evaluated on [0, 1]^2:

        f_H(x1, x2) = (1 - exp(-1 / (2*x2))) *
                      (2300*x1^3 + 1900*x1^2 + 2092*x1 + 60) /
                      (100*x1^3 + 500*x1^2 + 4*x1 + 20)

    x2 is clamped away from 0 (min=1e-6) to avoid division by zero in the
    exp(-1/(2*x2)) term.

    Following this module's convention (evaluate_true returns a
    minimization-style value, with negate=True flipping it back to the
    natural to-be-maximized form -- see Ackley/Rosenbrock/Levy above),
    evaluate_true returns the NEGATIVE of f_H, so its minimum coincides with
    f_H's true maximum.

    NOTE ON THE GLOBAL OPTIMUM: for fixed x1, the x2-multiplier
    (1 - exp(-1/(2*x2))) is monotonically *decreasing* in x2 (it approaches 1
    as x2 -> 0+ and 0 as x2 -> inf), so f_H's true global max over [0,1]^2 is
    attained at the x2 boundary (x2 -> 0, i.e. at the clamp value 1e-6 given
    the guard above), not in the interior. Numerically (see self-test at the
    bottom of this file / _diag scripts): argmax x1 ~= 0.21667, giving
    f_H ~= 13.7987 at x2=1e-6. This does NOT match evaluating at
    x=(0.0813, 0.8997) (which gives f_H ~= 4.39) -- that point is NOT this
    function's argmax under the formula above; flagged for the user rather
    than silently picked one way or the other. _optimal_value is set to the
    (negated) TRUE global optimum, i.e. -13.7987..., computed numerically at
    the bottom of this file's self-test, since known_optimal_value must equal
    the actually-achievable best value for DRO's simple_regret (dro.py) to
    reach 0 at the true optimum.
    """

    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    # True global argmax of f_H (see NOTE above): x2 pinned to the clamp
    # boundary, x1 found by maximizing the cubic ratio alone.
    _optimizers = [(0.21666517564327006, 1e-6)]
    # Negated value of f_H at the argmax above (computed numerically, see
    # module-level self-test at the bottom of this file).
    _optimal_value = -13.798722044512703

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [0,1]^2.
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Currin Exponential function (minimization-style, i.e.
        -f_H; see class docstring).

        Args:
            X: A (batch_shape) x 2 tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")

        x1 = X[..., 0]
        x2 = X[..., 1].clamp(min=1e-6)

        numerator = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        denominator = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
        f_h = (1.0 - torch.exp(-1.0 / (2.0 * x2))) * (numerator / denominator)

        return -f_h


class CurrinExpLF(SyntheticTestFunction):
    r"""Low-fidelity Currin Exponential function, for pairing with CurrinExpHF
    in a two-fidelity benchmark.

    FORMULA SOURCE: NOT the "perturbed alpha"-style change used for Hartmann
    below -- this is a corner-averaging construction from Xiong et al.
    (2013), "A Better Understanding of Model Updating Strategies in
    Simulating Dynamic Systems" (originally for Kriging model-updating
    experiments), and reused as-is across the multi-fidelity BO literature
    that also uses Hartmann/Currin as a two-fidelity pair -- see e.g. the
    reference implementation in the open-source `mf2` multi-fidelity
    benchmark package (https://mf2.readthedocs.io/en/latest/functions/
    currin.html). This module could not independently confirm the exact
    Appendix-A text of Kandasamy et al. 2016 (NIPS) beyond secondary
    descriptions, so credit is given to Xiong et al. as the function's
    original source rather than asserting it verbatim matches that
    specific appendix.

    f_L is a 4-corner average of the high-fidelity f_H (the SAME f_H as
    CurrinExpHF -- see that class), each corner offset by delta=0.05 in
    both x1 and x2, with the x2-decreasing corners floored at 0 (f_H's
    domain requires x2 > 0):

        f_L(x1, x2) = 1/4 * [ f_H(x1+d, x2+d) + f_H(x1+d, max(0, x2-d))
                             + f_H(x1-d, x2+d) + f_H(x1-d, max(0, x2-d)) ]
        where d = 0.05

    This is a genuine, distinct surface from f_H (NOT a relabeled copy of
    CurrinExpHF's formula) -- averaging over 4 offset corners smooths and
    shifts the function, so f_L != f_H almost everywhere, and the two are
    only moderately correlated (empirically r ~ 0.6-0.9 depending on
    sample region), which is the entire point of using it as a cheap
    proxy fidelity.

    Following CurrinExpHF's convention, evaluate_true returns the NEGATIVE
    of f_L (minimization-style, matching this module's SyntheticTestFunction
    subclasses generally), with negate=True flipping it back to the natural
    to-be-maximized form.

    NOTE ON OPTIMAL VALUE: unlike CurrinExpHF, this class deliberately does
    NOT set `_optimal_value`/`_optimizers` -- f_L's global optimum has no
    known closed form (it's a numerical artifact of the corner-averaging
    construction, not derivable the way f_H's cubic-ratio argmax was).
    Accessing `.optimal_value` on an instance of this class will raise
    AttributeError; benchmarks.py instead determines it empirically via
    dense grid search (`_find_lf_optimum`) and stores that as
    known_optimal_value in the BENCHMARKS registry, not via this property.
    """

    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _CORNER_DELTA = 0.05

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [0,1]^2.
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _f_h(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Natural (to-be-maximized) high-fidelity Currin value -- the same
        formula as CurrinExpHF's f_H (see that class's docstring), duplicated
        here (rather than routed through a CurrinExpHF instance) so this
        corner-averaging helper can be called directly on arbitrary,
        possibly-already-offset (x1, x2) pairs, including exactly x2=0 at
        the floored corners, without going through CurrinExpHF.__call__'s
        own negate/bounds/noise handling.
        """
        x2c = x2.clamp(min=1e-6)
        numerator = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
        denominator = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
        return (1.0 - torch.exp(-1.0 / (2.0 * x2c))) * (numerator / denominator)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the low-fidelity Currin Exponential function (minimization-
        style, i.e. -f_L; see class docstring).

        Args:
            X: A (batch_shape) x 2 tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")

        x1 = X[..., 0]
        x2 = X[..., 1]
        d = self._CORNER_DELTA

        f_pp = self._f_h(x1 + d, x2 + d)
        f_pm = self._f_h(x1 + d, (x2 - d).clamp(min=0.0))
        f_mp = self._f_h(x1 - d, x2 + d)
        f_mm = self._f_h(x1 - d, (x2 - d).clamp(min=0.0))

        f_l = 0.25 * (f_pp + f_pm + f_mp + f_mm)
        return -f_l


# Hartmann-6 A/P matrices and HF alpha, mirrored from BoTorch's own
# botorch.test_functions.synthetic.Hartmann(dim=6) so HartmannLF shares
# IDENTICAL A/P/formula structure with the HF function actually used
# elsewhere in this codebase (benchmarks.py's Hartmann_6D/Hartmann_6D_HF
# entries construct botorch.test_functions.synthetic.Hartmann directly, not
# a local class) -- only ALPHA differs between fidelities.
_HARTMANN6_A = [
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
]
_HARTMANN6_P = [
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
]
_HARTMANN6_ALPHA_HF = [1.0, 1.2, 3.0, 3.2]
# Low-fidelity alpha: sourced from the open-source `mf2` multi-fidelity
# benchmark package's documented Hartmann6 low-fidelity variant
# (https://mf2.readthedocs.io/en/latest/functions/hartmann.html), which
# pairs this alpha with an additionally-modified exponential transform.
# HartmannLF below uses ONLY this alpha substitution against the SAME
# exponential form as HF (per this task's "formula structure identical to
# HF" requirement) -- a simplification of mf2's own LF construction, not a
# verbatim reproduction of it. Not asserted to be the exact Kandasamy et
# al. 2016 (NIPS) Appendix A values -- this module could not independently
# confirm that appendix's text beyond secondary descriptions.
_HARTMANN6_ALPHA_LF = [0.5, 0.5, 2.0, 4.0]


class HartmannLF(SyntheticTestFunction):
    r"""Low-fidelity Hartmann-6 function, for pairing with BoTorch's own
    `botorch.test_functions.synthetic.Hartmann(dim=6)` in a two-fidelity
    benchmark. See the module-level `_HARTMANN6_ALPHA_LF` comment above for
    exactly where these alpha values come from and how this differs from
    the source they're adapted from.

    Same A, P matrices and same formula structure as HF Hartmann-6:

        H_L(x) = - sum_{i=1}^4 ALPHA_LF_i * exp(- sum_{j=1}^6 A_ij (x_j - 1e-4*P_ij)^2)

    with ALPHA_LF = [0.5, 0.5, 2.0, 4.0] in place of HF's
    ALPHA_HF = [1.0, 1.2, 3.0, 3.2] -- the ONLY difference from HF's formula
    (A, P, and the exponential form itself are all identical to HF).

    evaluate_true returns H_L directly (already minimization-style, matching
    BoTorch's own Hartmann.evaluate_true convention -- no extra negation
    inside evaluate_true itself); negate=True (the default here, for
    consistency with every other class in this module) flips it to the
    natural to-be-maximized form via the base class's own negate handling.

    NOTE ON OPTIMAL VALUE: unlike HF Hartmann (whose global optimum is the
    well-known z=(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    H(z)=-3.32237), this LF variant's global optimum has no established
    closed form under this alpha substitution. `_optimal_value`/
    `_optimizers` are deliberately left unset -- accessing `.optimal_value`
    on an instance will raise AttributeError. benchmarks.py determines the
    LF optimum empirically via dense grid search (`_find_lf_optimum`)
    instead.
    """

    dim = 6
    _bounds = [(0.0, 1.0)] * 6

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [0,1]^6.
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer("ALPHA", torch.tensor(_HARTMANN6_ALPHA_LF))
        self.register_buffer("A", torch.tensor(_HARTMANN6_A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(_HARTMANN6_P, dtype=torch.float))

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the low-fidelity Hartmann-6 function (already minimization-
        style; see class docstring).

        Args:
            X: A (batch_shape) x 6 tensor of inputs.

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")

        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        return H


class HartmannWidened6D(SyntheticTestFunction):
    """
    Hartmann 6D with controllable basin width. alpha_basin scales the WHOLE
    A matrix by a single scalar: smaller alpha_basin = wider basin.

    evaluate_true returns the already-positive, maximization-ready sum
    directly (negate defaults to False here, unlike HartmannLF above which
    returns minimization-style + negate=True) -- verified against BoTorch's
    own Hartmann(dim=6, negate=True) (the HF objective this codebase's plain
    Hartmann_6D_HF entry actually uses): evaluate_true(x*) with
    alpha_basin=1.0 equals +3.32237, matching Hartmann(negate=True)(x*)
    exactly, so this class's un-negated evaluate_true is equivalent to the
    rest of this module's negate=True convention, just implemented directly.

    CAUTION (see benchmarks.py's Hartmann6D_w* registration comment): despite
    _optimal_value=3.3224 below, evaluate_true(x_star) is NOT actually
    invariant to alpha_basin for this construction -- scaling the full A
    matrix by one scalar is not a coordinate reparametrization around x_star,
    since Hartmann-6 is a weighted mixture of 4 anisotropic Gaussian bumps at
    4 different centers (P rows), not a single bump. Verified numerically:
    f(x_star; alpha_basin=1.0/0.3/0.1) = 3.322/4.127/5.683, and the true
    argmax location also shifts away from x_star as alpha_basin shrinks.
    _optimal_value/_optimizers here describe ONLY the alpha_basin=1.0 case
    (identical to standard Hartmann-6); benchmarks.py computes each
    variant's actual optimum empirically instead of trusting this constant.
    """

    dim = 6
    _bounds = [(0.0, 1.0)] * 6
    _optimal_value = 3.3224

    A_base = torch.tensor([
        [10.0,  3.0,  17.0,  3.5,  1.7,  8.0],
        [0.05, 10.0,  17.0,  0.1,  8.0, 14.0],
        [3.0,   3.5,   1.7, 10.0, 17.0,  8.0],
        [17.0,  8.0,  0.05, 10.0,  0.1, 14.0],
    ], dtype=torch.float64)

    alpha_coef = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=torch.float64)

    P = torch.tensor([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ], dtype=torch.float64)

    def __init__(self, alpha_basin=1.0):
        super().__init__()
        self.alpha_basin = alpha_basin

    def evaluate_true(self, X):
        A = self.A_base * self.alpha_basin
        result = torch.zeros(X.shape[0], dtype=torch.float64)
        for i in range(4):
            exponent = -(A[i] * (X - self.P[i]) ** 2).sum(dim=-1)
            result += self.alpha_coef[i] * torch.exp(exponent)
        return result


class HartmannWidenedLF6D(SyntheticTestFunction):
    """
    Low-fidelity counterpart to HartmannWidened6D: identical construction
    (same A_base, same alpha_basin scaling of the FULL A matrix, same P),
    with HartmannLF's existing ALPHA_LF=[0.5,0.5,2.0,4.0] (see
    _HARTMANN6_ALPHA_LF above) in place of HartmannWidened6D's
    ALPHA=[1.0,1.2,3.0,3.2] -- the same fidelity-differentiating
    substitution HartmannLF makes against standard Hartmann-6.

    Mirrors HartmannWidened6D's own evaluate_true convention (already
    maximization-ready, negate defaults to False) rather than HartmannLF's
    (minimization-style + negate=True), so the paired HF/LF widened classes
    stay structurally symmetric. Like HartmannWidened6D, has no reliable
    closed-form optimum per alpha_basin -- benchmarks.py determines each
    variant's LF optimum empirically, matching HartmannLF's own convention.
    """

    dim = 6
    _bounds = [(0.0, 1.0)] * 6

    A_base = HartmannWidened6D.A_base
    P = HartmannWidened6D.P
    alpha_coef = torch.tensor(_HARTMANN6_ALPHA_LF, dtype=torch.float64)

    def __init__(self, alpha_basin=1.0):
        super().__init__()
        self.alpha_basin = alpha_basin

    def evaluate_true(self, X):
        A = self.A_base * self.alpha_basin
        result = torch.zeros(X.shape[0], dtype=torch.float64)
        for i in range(4):
            exponent = -(A[i] * (X - self.P[i]) ** 2).sum(dim=-1)
            result += self.alpha_coef[i] * torch.exp(exponent)
        return result


# Borehole bounds (Harper & Gupta 1983 / Kandasamy et al. 2016), shared by
# both fidelities: rw, r, Tu, Hu, Tl, Hl, L, Kw, in that order.
_BOREHOLE_BOUNDS = [
    (0.05, 0.15),      # rw: radius of borehole (m)
    (100.0, 50000.0),  # r:  radius of influence (m)
    (63070.0, 115600.0),  # Tu: transmissivity, upper aquifer (m^2/yr)
    (990.0, 1110.0),   # Hu: potentiometric head, upper aquifer (m)
    (63.1, 116.0),     # Tl: transmissivity, lower aquifer (m^2/yr)
    (700.0, 820.0),    # Hl: potentiometric head, lower aquifer (m)
    (1120.0, 1680.0),  # L:  length of borehole (m)
    (9855.0, 12045.0),  # Kw: hydraulic conductivity of borehole (m/yr)
]


class BoreholeFunctionHF(SyntheticTestFunction):
    r"""Standard 8-dimensional Borehole function (water flow rate, m^3/yr).
    x = [rw, r, Tu, Hu, Tl, Hl, L, Kw]:

        f_H(x) = 2*pi*Tu*(Hu-Hl) /
                 ( ln(r/rw) * (1 + 2*L*Tu/(ln(r/rw)*rw^2*Kw) + Tu/Tl) )

    Following this module's convention (CurrinExpHF/HartmannLF above),
    evaluate_true returns the NEGATIVE of f_H (minimization-style), with
    negate=True flipping it back to the natural to-be-maximized form (i.e.
    treating higher borehole flow rate as the optimization target, matching
    every other benchmark in this module being set up as a maximization
    problem via negate=True).

    NOTE ON OPTIMAL VALUE: unlike Hartmann (published closed-form optimum)
    or CurrinExpHF (analytically derived here), Borehole has no established
    closed-form optimum used in this codebase. `_optimal_value`/
    `_optimizers` are deliberately left unset, matching CurrinExpLF/
    HartmannLF's convention -- benchmarks.py determines it empirically via
    dense grid search instead.
    """

    dim = 8
    _bounds = list(_BOREHOLE_BOUNDS)

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses the
                standard Borehole bounds (NOT [0,1]^8 -- each dimension has
                its own physical range).
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _unpack(X: torch.Tensor):
        return (X[..., 0], X[..., 1], X[..., 2], X[..., 3],
                X[..., 4], X[..., 5], X[..., 6], X[..., 7])

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Borehole function (minimization-style, i.e. -f_H; see
        class docstring).

        Args:
            X: A (batch_shape) x 8 tensor of inputs [rw, r, Tu, Hu, Tl, Hl, L, Kw].

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")

        rw, r, Tu, Hu, Tl, Hl, L, Kw = self._unpack(X)
        log_r_rw = torch.log(r / rw)
        numerator = 2.0 * math.pi * Tu * (Hu - Hl)
        denominator = log_r_rw * (
            1.0 + (2.0 * L * Tu) / (log_r_rw * rw ** 2 * Kw) + Tu / Tl
        )
        f_h = numerator / denominator
        return -f_h


class AckleyFunctionHF(SyntheticTestFunction):
    r"""10-dimensional Ackley function, high-fidelity member of the
    Ackley_10D two-fidelity pair. Domain [0,1]^10, internally rescaled to
    Ackley's natural [-5,5]^10 range (so this class's own _bounds/domain
    stays [0,1]^10, matching Currin_2D/Hartmann_6D's convention rather than
    Borehole_8D's native-physical-range one).

        f_H(x) = -20*exp(-0.2*sqrt(mean(x_scaled^2)))
                 - exp(mean(cos(2*pi*x_scaled))) + 20 + e
        x_scaled = x*10 - 5

    Global minimum (of this minimization-style formula) is 0 at
    x_scaled=0, i.e. x=[0.5]*10 -- see AckleyFunctionLF for why LF shares
    this exact optimum location.

    Following this module's convention (evaluate_true returns a
    minimization-style value, with negate=True flipping it back to the
    natural to-be-maximized form -- see CurrinExpHF/BoreholeFunctionHF
    above), evaluate_true does NOT negate internally.
    """

    dim = 10
    _bounds = [(0.0, 1.0)] * 10
    _optimizers = [(0.5,) * 10]
    _optimal_value = 0.0

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [0,1]^10.
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)

    @staticmethod
    def _raw_ackley(X: torch.Tensor, b: float = 0.2, c: float = 2.0 * math.pi) -> torch.Tensor:
        """Standard (minimization-style) Ackley value on the *rescaled*
        [-5,5]^10 domain, shared by HF and LF -- see class docstrings.
        b/c are Ackley's own decay-rate/frequency parameters (HF uses the
        textbook defaults 0.2/2*pi); AckleyFunctionLF overrides them to
        reshape the LF landscape -- see its docstring for why. Note that
        for ANY b>0, c>0 this formula's minimum is always exactly 0 at
        x_scaled=0 (x=[0.5]*10), so HF/LF keep the identical global
        optimum regardless of b/c."""
        x = X * 10.0 - 5.0
        term1 = -20.0 * torch.exp(-b * (x ** 2).mean(dim=-1).sqrt())
        term2 = -torch.exp(torch.cos(c * x).mean(dim=-1))
        return term1 + term2 + 20.0 + math.e

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Ackley function (minimization-style; see class
        docstring). X is on [0,1]^10; internally rescaled to [-5,5]^10.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        return self._raw_ackley(X)


class AckleyFunctionLF(SyntheticTestFunction):
    r"""Low-fidelity Ackley_10D: same formula family as AckleyFunctionHF but
    with reshaped decay-rate (b) and frequency (c) parameters, instead of an
    additive distance-from-center bias.

    DEVIATION FROM THE ORIGINALLY SPECIFIED DESIGN, FLAGGED EXPLICITLY: the
    originally requested mechanism was an additive bias, f_L(x) = f_H(x) -
    bias*||x-0.5||_2 (continuous-fidelity formulation from
    arXiv:2009.05700, distance-based bias per arXiv:2410.00544), targeting
    Pearson r(HF,LF) in [0.70, 0.85]. Empirically (see benchmarks.py's
    module-level self-test / _ackley_sanity_checks.py CHECK B) that
    mechanism could not hit the target range in this codebase's d=10, iid-
    uniform-on-[0,1]^10 sampling setting even at bias=20 (40x the suggested
    0.5), only reaching r~=0.98: in 10D, Ackley's value is dominated by its
    term1 (-20*exp(-b*rms_distance)), which concentration-of-measure makes
    an almost deterministic function of ||x-0.5|| for random x -- so ANY
    perturbation that is itself primarily a function of ||x-0.5|| (as the
    additive distance bias is, by construction) stays highly correlated
    with HF regardless of its scale, rather than decorrelating.

    Reshaping b (decay rate) and c (cosine frequency) instead changes the
    LOCAL curvature/oscillation structure of the landscape -- a genuinely
    different source of variation from HF's, not just a scaled copy of the
    same radial trend -- while leaving the analytic global minimum
    UNCHANGED (see _raw_ackley's docstring: for any b>0, c>0 the minimum
    stays exactly 0 at x=[0.5]*10), preserving the "LF and HF share the
    same true optimum" property the original design was after. b=0.05,
    c=0.5*pi (vs HF's b=0.2, c=2*pi) empirically gives Pearson r ~= 0.75
    (range 0.68-0.75 across several random seeds at n=500), inside the
    target [0.70, 0.85] at the seed this module's own sanity check uses.
    """

    dim = 10
    _bounds = [(0.0, 1.0)] * 10
    _b = 0.05
    _c = 0.5 * math.pi
    # No known closed-form optimal_value/optimizers stored for the LF
    # surface (matching CurrinExpLF/HartmannLF/BoreholeFunctionLF's
    # convention) even though its analytic minimum location IS known (see
    # class docstring) -- LF is never the target of simple_regret, only HF
    # is, so benchmarks.py's Ackley_10D_LF entry sets known_optimal_value
    # directly rather than relying on an _optimal_value class attribute here.

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses [0,1]^10.
            b: Override the class default decay-rate parameter (0.05).
            c: Override the class default frequency parameter (0.5*pi).
        """
        if bounds is not None:
            self._bounds = bounds
        if b is not None:
            self._b = b
        if c is not None:
            self._c = c
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the low-fidelity Ackley function (minimization-style; see
        class docstring).
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")
        return AckleyFunctionHF._raw_ackley(X, b=self._b, c=self._c)


class BoreholeFunctionLF(SyntheticTestFunction):
    r"""Low-fidelity Borehole approximation (Kandasamy et al. 2016), same
    domain and input order as BoreholeFunctionHF:

        f_L(x) = 5*Tu*(Hu-Hl) /
                 ( ln(r/rw) * (1.5 + 2*L*Tu/(ln(r/rw)*rw^2*Kw) + Tu/Tl) )

    Differs from f_H in exactly two places: the numerator coefficient
    (5 vs 2*pi) and the denominator constant (1.5 vs 1) -- these reduce the
    LF version's sensitivity to the borehole radius terms and introduce a
    systematic bias relative to HF, making it a useful but imperfect proxy.

    Same evaluate_true convention as BoreholeFunctionHF (returns -f_L,
    negate=True flips to the natural to-be-maximized form). See
    BoreholeFunctionHF's docstring for why `_optimal_value`/`_optimizers`
    are deliberately left unset here too.
    """

    dim = 8
    _bounds = list(_BOREHOLE_BOUNDS)

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = True,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        """
        Args:
            noise_std: Standard deviation of observation noise. Default: None.
            negate: If True, negate the function for maximization. Default: True.
            bounds: Custom bounds for the function domain. If None, uses the
                standard Borehole bounds (NOT [0,1]^8).
        """
        if bounds is not None:
            self._bounds = bounds
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the low-fidelity Borehole function (minimization-style,
        i.e. -f_L; see class docstring).

        Args:
            X: A (batch_shape) x 8 tensor of inputs [rw, r, Tu, Hu, Tl, Hl, L, Kw].

        Returns:
            A (batch_shape) tensor of function values.
        """
        if X.ndim < 1 or X.shape[-1] != self.dim:
            raise ValueError(f"Input tensor X must have last dimension equal to {self.dim}")

        rw, r, Tu, Hu, Tl, Hl, L, Kw = BoreholeFunctionHF._unpack(X)
        log_r_rw = torch.log(r / rw)
        numerator = 5.0 * Tu * (Hu - Hl)
        denominator = log_r_rw * (
            1.5 + (2.0 * L * Tu) / (log_r_rw * rw ** 2 * Kw) + Tu / Tl
        )
        f_l = numerator / denominator
        return -f_l


# Example Usage (Optional)
if __name__ == "__main__":
    # Ackley Example
    ackley2d = Ackley(dim=2)
    print(f"Ackley dim: {ackley2d.dim}, bounds: {ackley2d.bounds}")
    test_X_ackley = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
    # BoTorch functions return negated value by default (for maximization)
    print(f"Ackley values (negated): {ackley2d(test_X_ackley)}")
    # To get standard minimization value:
    print(f"Ackley values (min form): {-ackley2d(test_X_ackley)}")
    print("-" * 20)

    # Rosenbrock Example
    rosen2d = Rosenbrock(dim=2)
    print(f"Rosenbrock dim: {rosen2d.dim}, bounds: {rosen2d.bounds}")
    test_X_rosen = torch.tensor([[1.0, 1.0], [0.0, 0.0], [2.0, 3.0]])
    print(f"Rosenbrock values (negated): {rosen2d(test_X_rosen)}")
    print(f"Rosenbrock values (min form): {-rosen2d(test_X_rosen)}")
    print("-" * 20)

    # Levy Example
    levy2d = Levy(dim=2)
    print(f"Levy dim: {levy2d.dim}, bounds: {levy2d.bounds}")
    test_X_levy = torch.tensor([[1.0, 1.0], [0.0, 0.0], [-5.0, 5.0]])
    print(f"Levy values (negated): {levy2d(test_X_levy)}")
    print(f"Levy values (min form): {-levy2d(test_X_levy)}")
    print("-" * 20)

    # Example with higher dimension and custom bounds
    rosen4d_custom = Rosenbrock(dim=4, bounds=[(-3.0, 3.0)] * 4)
    print(f"Rosenbrock dim: {rosen4d_custom.dim}, bounds: {rosen4d_custom.bounds}")
    test_X_rosen4d = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    print(f"Rosenbrock 4D values (negated): {rosen4d_custom(test_X_rosen4d)}")
    print("-" * 20)

    # CurrinExpHF self-test
    currin = CurrinExpHF(negate=True)
    print(f"CurrinExpHF dim: {currin.dim}, bounds: {currin.bounds}")
    test_X_currin = torch.tensor([[0.0813, 0.8997], [0.5, 0.5]], dtype=torch.float64)
    print(f"CurrinExpHF values (negated, i.e. natural f_H): {currin(test_X_currin)}")
    print(f"CurrinExpHF true argmax {currin.optimizers[0].tolist()}: "
          f"value={currin(currin.optimizers[0].unsqueeze(0).double())}")
    print(f"CurrinExpHF.optimal_value: {currin.optimal_value:.6f}")
    # x2 near/at 0 must not produce NaN or Inf (clamped internally)
    edge_X = torch.tensor([[0.2, 0.0], [0.2, 1e-9]], dtype=torch.float64)
    edge_vals = currin(edge_X)
    print(f"CurrinExpHF near x2=0: {edge_vals}  "
          f"(NaN={torch.isnan(edge_vals).any().item()}, Inf={torch.isinf(edge_vals).any().item()})")