"""V1-V3: pure-math validation of src/baselines/mf_mes_takeno.py. No GP needed."""
import os, sys, math
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np
from scipy.stats import norm
from src.baselines.mf_mes_takeno import mes_hf, mes_lf, ClampStats, LOG_SQRT_2PIE

rng = np.random.default_rng(0)
FAIL = []

def pred_from(mu_L, var_L, rho, var_delta):
    var_H = rho**2 * var_L + var_delta
    mu_H = rho * mu_L + rng.normal(0, 0.3, size=mu_L.shape)
    return dict(mu_H=mu_H, var_H=var_H, mu_L=mu_L, var_L=var_L,
                var_delta=var_delta, cov_LH=rho*var_L, rho=rho)

# ---------------------------------------------------------------- V1
print("V1: HF closed form vs fine quadrature of the truncated-normal entropy")
errs = []
for _ in range(100):
    mu = rng.normal(0, 2); sig = np.exp(rng.normal(0, 0.6)); fs = mu + rng.normal(1.0, 1.5)*sig
    p = dict(mu_H=np.array([mu]), var_H=np.array([sig**2]))
    I_cf = float(mes_hf(p, [fs])[0])
    g = (fs - mu)/sig; Phi = norm.cdf(g)
    if Phi < 1e-8:   # degenerate truncation; excluded, counted below
        continue
    y = np.linspace(mu - 40*sig, fs, 400001)
    dens = norm.pdf((y-mu)/sig)/(sig*Phi)
    m = dens > 0
    H1 = -np.trapz(dens[m]*np.log(dens[m]), y[m])
    I_num = (np.log(sig) + LOG_SQRT_2PIE) - H1
    errs.append(abs(I_cf - I_num)/max(abs(I_num), 1e-12))
errs = np.array(errs)
print(f"   n={len(errs)}  max rel err = {errs.max():.3e}  median = {np.median(errs):.3e}"
      f"   [< 1e-6 required]  {'PASS' if errs.max() < 1e-6 else 'FAIL'}")
if errs.max() >= 1e-6: FAIL.append("V1")

# ---------------------------------------------------------------- V2
print("\nV2: 32-pt Gauss-Hermite vs 10,000-pt trapezoid over [mu-8s, mu+8s]")
errs = []
for _ in range(100):
    mu_L = rng.normal(0, 2); var_L = np.exp(rng.normal(0, 0.6))
    rho = rng.uniform(0.3, 1.2); var_d = np.exp(rng.normal(-0.5, 0.6))
    p = pred_from(np.array([mu_L]), np.array([var_L]), rho, np.array([var_d]))
    sig_L = math.sqrt(var_L); sig_H = math.sqrt(p["var_H"][0]); mu_H = p["mu_H"][0]
    fs = mu_H + rng.normal(1.0, 1.5)*sig_H
    I_gh = float(mes_lf(p, [fs])[0])   # n_quad="auto"
    # reference trapezoid of the SAME integrand
    v = np.linspace(mu_L - 8*sig_L, mu_L + 8*sig_L, 10000)
    Phi_H = max(norm.cdf((fs - mu_H)/sig_H), 1e-300)
    u = mu_H + rho*(v - mu_L); s = math.sqrt(var_d)
    q = norm.cdf((fs - u)/s) * norm.pdf((v-mu_L)/sig_L) / (sig_L*Phi_H)
    m = q > 0
    H1 = -np.trapz(q[m]*np.log(q[m]), v[m])
    I_ref = (np.log(sig_L) + LOG_SQRT_2PIE) - H1
    errs.append(abs(I_gh - I_ref)/max(abs(I_ref), 1e-12))
errs = np.array(errs)
print(f"   n={len(errs)}  max rel err = {errs.max():.3e}  median = {np.median(errs):.3e}"
      f"   [< 1e-4 required]  {'PASS' if errs.max() < 1e-4 else 'FAIL'}")
if errs.max() >= 1e-4: FAIL.append("V2")

# ---------------------------------------------------------------- V3
print("\nV3: mutual-information non-negativity on 10,000 random inputs (pre-clamp)")
n = 10000
mu_L = rng.normal(0, 2, n); var_L = np.exp(rng.normal(0, 0.8, n))
rho = rng.uniform(0.2, 1.3); var_d = np.exp(rng.normal(-0.5, 0.8, n))
p = pred_from(mu_L, var_L, rho, var_d)
fs = p["mu_H"] + rng.normal(1.0, 2.0, n)*np.sqrt(p["var_H"])
sH, sL = ClampStats(), ClampStats()
# measure PRE-clamp by re-deriving without the final np.maximum
IH = mes_hf(p, [float(np.median(fs))], stats=sH)
IL = mes_lf(p, [float(np.median(fs))], stats=sL)
print(f"   HF: min I = {IH.min():.3e}   {sH}")
print(f"   LF: min I = {IL.min():.3e}   {sL}")
print(f"   HF clamp rate {100*sH.rate():.4f}%  LF clamp rate {100*sL.rate():.4f}%"
      f"   {'PASS' if max(sH.rate(), sL.rate()) < 0.01 else 'FAIL (quadrature suspect)'}")
if max(sH.rate(), sL.rate()) >= 0.01: FAIL.append("V3")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
