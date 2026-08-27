"""Joint additive multi-fidelity GP, ported from the authors' MATLAB reference
(mfBO/sqExpKernelAdditive.m, AdditiveGPRegression.m, optimizeNoiseKernel.m).

MODEL.  f_i(x) = f_M(x) + eps_i(x) for i < M, with f_M ~ GP(0, k_M) SHARED across
fidelities and eps_i ~ GP(0, k_i) independent per lower fidelity. Hence

    Cov(f_i(x), f_j(x')) = k_M(x, x') + [i == j and i < M] * k_i(x, x')

which is exactly sqExpKernelAdditive's `kernel_base + kernel_offset`: one shared
target-fidelity block over the POOLED data plus a block-diagonal discrepancy term.
Inference is a single Cholesky over all fidelities' data together, so low-fidelity
observations inform the target posterior directly.

This replaces a two-separate-GPs approximation (gp_H on HF data, gp_error on LF
residuals), under which LF data could not inform the target at all.

HYPERPARAMETERS, following optimizeNoiseKernel.m: the target kernel (bw_M,
scale_M) is fit first on target-fidelity data; then each lower fidelity's
(bw_i, scale_i) is fit by maximising the marginal likelihood of THAT fidelity's
data under k_M(fixed) + k_i, i.e. with the target kernel frozen.

NOISE, per acqMFMIGreedy.m: noiseFuncs{i}(x, x) is k_i(x, x) = scale_i for a
stationary kernel, and 1e-4 * std(Y)^2 at the target fidelity.

sqExpKernel.m is `scale * exp(-D / (2*bw^2))`, so `scale` multiplies the kernel
directly -- it is a variance, and k(x,x) = scale. Kept in that convention.
"""
import math
import numpy as np


def _sqexp(bw, scale, X, Y):
    D = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return scale * np.exp(-D / (2.0 * bw * bw))


def _chol(K, jitter=1e-8):
    n = K.shape[0]
    for k in range(8):
        try:
            return np.linalg.cholesky(K + (jitter * (10 ** k)) * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(K + 1e-2 * np.eye(n))


def _nlml_noise(bw, scale, X, Y, mean_val, base_bw, base_scale):
    """normMargLikelihoodFixNoise / normMargLikelihoodFixTarget: Ky is the SUM of
    the target kernel and the discrepancy kernel, with no separate noise term."""
    K = _sqexp(base_bw, base_scale, X, X) + _sqexp(bw, scale, X, X)
    return _nlml_core(K, X, Y, mean_val)


def _nlml_iid(bw, scale, X, Y, mean_val, noise_var):
    """normMargLikelihood: Ky is the kernel plus iid noise."""
    K = _sqexp(bw, scale, X, X) + noise_var * np.eye(X.shape[0])
    return _nlml_core(K, X, Y, mean_val)


def _nlml_core(K, X, Y, mean_val):
    y = (Y - mean_val).reshape(-1, 1)
    try:
        L = _chol(K)
    except Exception:
        return 1e12
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    # the reference's `sum(log(diag(L + 0.001)))` -- the offset guards log(0)
    v = float(-0.5 * (y.T @ a) - np.log(np.diag(L) + 1e-3).sum()
              - 0.5 * X.shape[0] * math.log(2 * math.pi))
    return -v if np.isfinite(v) else 1e12


def _fit_joint(obj, bw_range, sc_range, seed=0):
    """Joint 2-D fit over (log bw, log scale), mirroring maximizeScore.m:
    200 random seeds inside the box, then a local refine from the best."""
    rng = np.random.RandomState(seed)
    lo = np.array([math.log(bw_range[0]), math.log(sc_range[0])])
    hi = np.array([math.log(bw_range[1]), math.log(sc_range[1])])
    P = lo + (hi - lo) * rng.rand(200, 2)
    vals = np.array([obj(math.exp(t[0]), math.exp(t[1])) for t in P])
    t0 = P[int(np.argmin(vals))]
    try:
        from scipy.optimize import minimize
        r = minimize(lambda t: obj(math.exp(t[0]), math.exp(t[1])), t0,
                     method='L-BFGS-B', bounds=list(zip(lo, hi)))
        t0 = r.x if np.isfinite(r.fun) and r.fun <= vals.min() else t0
    except Exception:
        pass
    return math.exp(t0[0]), math.exp(t0[1])


class AdditiveMFGP:
    def __init__(self, d, n_fidels=2):
        self.d = d
        self.M = n_fidels
        self.bw = [1.0] * n_fidels
        self.scale = [1.0] * n_fidels
        self.mean_val = 0.0
        self.noise_var = [1.0] * n_fidels
        self._fitted = False

    def fit(self, X_list, Y_list, bounds):
        X_list = [np.asarray(x, float).reshape(-1, self.d) for x in X_list]
        Y_list = [np.asarray(y, float).reshape(-1) for y in Y_list]
        allY = np.concatenate([y for y in Y_list if y.size]) if any(y.size for y in Y_list) else np.zeros(1)
        stdY = float(allY.std()) if allY.size > 1 else 1.0
        rangeY = float(allY.max() - allY.min()) if allY.size > 1 else 1.0
        # mfboPreProcessParams.m: priorMeanVal = maxY + 2*rangeY for MI-Greedy
        self.mean_val = float(allY.max() + 2.0 * rangeY) if allY.size else 0.0
        span = float(np.mean(np.asarray(bounds[1]) - np.asarray(bounds[0])))
        rd = math.sqrt(self.d)
        # mfboPreProcessParams.m: ranges differ by fidelity. Lower fidelities get
        # an ABSOLUTE scale box [0.1, 100]; the target gets [10, 1e3]*rangeY,
        # linear in rangeY because sqExpKernel's `scale` multiplies the kernel
        # directly, so k(x,x) = scale is already a variance.
        bw_lo_box = (0.01 * rd, 100.0 * rd)
        bw_tg_box = (1e-3 * rd * span, 10.0 * rd * span)
        sc_lo_box = (0.1, 100.0)
        sc_tg_box = (10.0 * max(rangeY, 1e-6), 1e3 * max(rangeY, 1e-6))
        noise_tg = max(1e-4 * stdY ** 2, 1e-12)

        ok = [x.shape[0] > 2 for x in X_list]
        if not any(ok):                       # too little data to fit anything
            self.bw = [bw_lo_box[0]] * (self.M - 1) + [bw_tg_box[0]]
            self.scale = [sc_lo_box[0]] * (self.M - 1) + [sc_tg_box[0]]
        else:
            # multipleGPRegressionML.m 'coorLearn': two rounds of coordinate
            # ascent alternating optimizeTargetKernel and optimizeNoiseKernel,
            # warm-started from the previous iteration's noise hyperparameters.
            nbw = [self.bw[i] if self._fitted else bw_lo_box[0] for i in range(self.M - 1)]
            nsc = [self.scale[i] if self._fitted else sc_lo_box[0] for i in range(self.M - 1)]
            bwM = self.bw[-1] if self._fitted else bw_tg_box[0]
            scM = self.scale[-1] if self._fitted else sc_tg_box[0]

            def target_obj(b, sc):
                # allNormMargLikelihoodFixNoise: every lower fidelity under
                # k_target + k_i, plus the target's own term weighted 10x.
                tot = 0.0
                for i in range(self.M - 1):
                    if ok[i]:
                        tot += _nlml_noise(nbw[i], nsc[i], X_list[i], Y_list[i],
                                           self.mean_val, b, sc)
                if ok[-1]:
                    tot += 10.0 * _nlml_iid(b, sc, X_list[-1], Y_list[-1],
                                            self.mean_val, noise_tg)
                return tot

            for rnd in range(2):
                bwM, scM = _fit_joint(target_obj, bw_tg_box, sc_tg_box, seed=rnd)
                for i in range(self.M - 1):
                    if not ok[i]:
                        continue
                    nbw[i], nsc[i] = _fit_joint(
                        lambda b, sc: _nlml_noise(b, sc, X_list[i], Y_list[i],
                                                  self.mean_val, bwM, scM),
                        bw_lo_box, sc_lo_box, seed=10 * rnd + i + 1)
            self.bw = list(nbw) + [bwM]
            self.scale = list(nsc) + [scM]

        for i in range(self.M - 1):
            self.noise_var[i] = self.scale[i]   # noiseFuncs{i}(x,x) = scale_i
        self.noise_var[-1] = noise_tg           # target: 1e-4*stdY^2
        self._fitted = True

        # --- joint Cholesky over pooled data ---
        self._X = X_list
        self._idx = []
        Xj, Yj = [], []
        p = 0
        for i in range(self.M):
            n = X_list[i].shape[0]
            self._idx.append((p, p + n)); p += n
            if n:
                Xj.append(X_list[i]); Yj.append(Y_list[i])
        if not Xj:
            self._L = None; return self
        Xj = np.concatenate(Xj, 0); Yj = np.concatenate(Yj, 0)
        self._Xj = Xj
        K = _sqexp(self.bw[-1], self.scale[-1], Xj, Xj)          # shared base
        for i in range(self.M - 1):                              # block-diagonal offset
            a, b = self._idx[i]
            if b > a:
                K[a:b, a:b] += _sqexp(self.bw[i], self.scale[i], X_list[i], X_list[i])
        K = K + 1e-6 * np.eye(K.shape[0])
        self._L = _chol(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, (Yj - self.mean_val).reshape(-1, 1)))
        return self

    def posterior(self, X, fidelity):
        """(mu, var) of f_fidelity at X under the joint model."""
        X = np.asarray(X, float).reshape(-1, self.d)
        if self._L is None:
            return (np.full(X.shape[0], self.mean_val),
                    np.full(X.shape[0], self.scale[-1]))
        Kte = _sqexp(self.bw[-1], self.scale[-1], X, self._Xj)
        if fidelity < self.M - 1:
            a, b = self._idx[fidelity]
            if b > a:
                Kte[:, a:b] += _sqexp(self.bw[fidelity], self.scale[fidelity], X, self._X[fidelity])
        mu = self.mean_val + (Kte @ self._alpha).reshape(-1)
        kss = self.scale[-1] + (self.scale[fidelity] if fidelity < self.M - 1 else 0.0)
        V = np.linalg.solve(self._L, Kte.T)
        var = np.maximum(kss - (V ** 2).sum(0), 1e-12)
        return mu, var
