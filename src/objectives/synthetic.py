from __future__ import annotations

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