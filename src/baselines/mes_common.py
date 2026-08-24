"""
Model-agnostic MF-MES acquisition (Takeno et al. 2020, arXiv:1901.08275).

WHY THIS MODULE EXISTS -- and an important correction to the "novel KO
formula" framing this experiment was designed around:

Takeno's Lemma 3.1 is NOT derived for, or restricted to, an additive
fidelity model. Its Section 2 assumes only that the set of outputs
{f_x^(m)} is jointly multivariate normal, and explicitly names co-kriging
(Kennedy & O'Hagan, 2000) alongside multi-task GPR and SLFM as admissible
surrogates. Their Eq. (5) is the plain bivariate Gaussian conditional

    f^(M) | f^(m) ~ N(u(x), s^2(x))
    u(x)   = sigma^2(mM) * (f^(m) - mu^(m)) / sigma^2(m) + mu^(M)
    s^2(x) = sigma^2(M) - (sigma^2(mM))^2 / sigma^2(m)

written in terms of whatever cross-fidelity covariance sigma^2(mM) the
chosen MF-GPR happens to produce. Substituting the KO covariances
(sigma^2(mM) = rho*sigma_L^2, sigma^2(M) = rho^2*sigma_L^2 + sigma_delta^2)
gives u = rho*(f_L - mu_L) + mu_H and s^2 = sigma_delta^2 -- exactly what
src/policy/mf_dro.py's _compute_mes_lf_vectorized already computes. That is
an INSTANTIATION of Takeno's lemma at a surrogate the paper already covers,
not a new derivation, and the same is true of the additive case.

What genuinely differs between the two methods this module supports is
therefore the SURROGATE, not the acquisition: KO/co-kriging
(f_H = rho*f_L + delta, rho fitted) versus the Song-2019-style additive
model (f_L = f_H + e). Keeping one shared, model-agnostic implementation of
the acquisition here -- fed only by (mu, sigma, cross-covariance) triples --
is what makes that comparison clean: KO-MES and Additive-MES run
character-for-character the same acquisition code, so any gap between them
is attributable to the surrogate alone.

Both functions here are vectorized over candidates and take precomputed
Thompson samples of y*_H, matching the calling convention of
_compute_mes_hf_vectorized / _compute_mes_lf_vectorized.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import norm as scipy_norm

LOG_SQRT_2PIE = 0.5 * np.log(2.0 * np.pi * np.e)


def mes_hf(mu_H, sigma_H, y_star_arr):
    """
    Highest-fidelity MES, Takeno Eq. (4): H0 - E_{f*}[H1] where H1 is the
    entropy of the normal truncated at f*.

    Identical math to mf_dro._compute_mes_hf_vectorized (including the
    LOG_SQRT_2PIE term that must appear in BOTH H0 and H1 -- they do not
    cancel otherwise), but takes plain arrays instead of a BoTorch posterior
    so the additive and single-fidelity models can reuse it.

    mu_H, sigma_H: np.ndarray [N].  y_star_arr: np.ndarray [K].
    Returns np.ndarray [N], all >= 0.
    """
    mu_H = np.asarray(mu_H, dtype=float)
    sigma_H = np.maximum(np.asarray(sigma_H, dtype=float), 1e-12)

    gamma = (y_star_arr[None, :] - mu_H[:, None]) / sigma_H[:, None]  # [N, K]
    phi = scipy_norm.pdf(gamma)
    Phi = np.maximum(scipy_norm.cdf(gamma), 1e-300)

    H0 = np.log(sigma_H) + LOG_SQRT_2PIE
    H1 = np.mean(
        np.log(sigma_H[:, None]) + LOG_SQRT_2PIE + np.log(Phi)
        - gamma * phi / (2.0 * Phi),
        axis=1,
    )
    return np.maximum(H0 - H1, 0.0)


def mes_lf(mu_m, sigma_m, mu_M, sigma_M, cov_mM, y_star_arr, n_quad=32):
    """
    Lower-fidelity MES, Takeno Lemma 3.1, for ANY jointly-Gaussian
    two-fidelity surrogate: the caller supplies the marginal moments at the
    query fidelity m and the target fidelity M plus their cross-covariance,
    and this evaluates

        I(f^(m)(x); f*_M | D) = H(f^(m)(x) | D) - E_{f*}[H(f^(m)(x) | f^(M) <= f*, D)]

    by 1D Gauss-Hermite quadrature over f^(m), with the truncated conditional
    density q(v) = Phi((f* - u(v))/s) * phi_m(v) / Phi(gamma_M) from the
    lemma's Bayes-rule decomposition.

    QUADRATURE NORMALIZATION: the expectation E_{v~N(mu_m, sigma_m^2)}[g(v)]
    is computed as sum(w_i*g(v_i))/sum(w_i) on probabilist's-Hermite nodes
    (hermegauss, weight e^{-t^2/2}) rather than via an assumed closed-form
    scaling constant -- the same correction documented at length in
    mf_dro._lf_mes_info_gain, carried over here deliberately so the two
    implementations agree numerically.

    All of mu_m, sigma_m, mu_M, sigma_M, cov_mM: np.ndarray [N].
    y_star_arr: np.ndarray [K]. Returns np.ndarray [N], all >= 0.
    """
    mu_m = np.asarray(mu_m, dtype=float)
    sigma_m = np.maximum(np.asarray(sigma_m, dtype=float), 1e-12)
    mu_M = np.asarray(mu_M, dtype=float)
    sigma_M = np.maximum(np.asarray(sigma_M, dtype=float), 1e-12)
    cov_mM = np.asarray(cov_mM, dtype=float)

    # Takeno Eq. (5). The conditional variance is analytically non-negative
    # (it is a Schur complement of a PSD 2x2 covariance), but the marginals
    # here come from two separately-fitted GPs whose implied joint can drift
    # very slightly outside PSD, so clamp rather than take sqrt of a
    # negative.
    beta = cov_mM / (sigma_m ** 2)                                   # [N]
    s = np.sqrt(np.maximum(sigma_M ** 2 - cov_mM ** 2 / sigma_m ** 2, 1e-24))

    H0 = np.log(sigma_m) + LOG_SQRT_2PIE  # [N]

    xi, wi = hermegauss(n_quad)
    wi_sum = wi.sum()
    v_nodes = mu_m[:, None] + sigma_m[:, None] * xi[None, :]          # [N, Q]
    phi_m = scipy_norm.pdf(v_nodes, loc=mu_m[:, None], scale=sigma_m[:, None])

    H1 = np.zeros_like(mu_m)
    for f_star in y_star_arr:
        gamma_M = (f_star - mu_M) / sigma_M                           # [N]
        Phi_M = np.maximum(scipy_norm.cdf(gamma_M), 1e-300)           # [N]

        u_v = beta[:, None] * (v_nodes - mu_m[:, None]) + mu_M[:, None]
        Phi_cond = scipy_norm.cdf((f_star - u_v) / s[:, None])        # [N, Q]

        q_v = Phi_cond * phi_m / Phi_M[:, None]

        # g(v) = (Phi_cond/Phi_M) * log q(v); x*log(x) -> 0 as x -> 0, so skip
        # negligible entries rather than risk 0 * (-inf) = nan.
        safe = q_v > 1e-300
        g_v = np.zeros_like(q_v)
        g_v[safe] = (Phi_cond[safe] / Phi_M[np.where(safe)[0]]) * np.log(q_v[safe])

        H1 += -np.sum(wi[None, :] * g_v, axis=1) / wi_sum

    H1 /= len(y_star_arr)
    return np.maximum(H0 - H1, 0.0)
