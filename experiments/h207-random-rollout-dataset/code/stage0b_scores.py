"""h207 Stage 0b -- WHICH SCORE? Same batches, four scores side by side.

SC5 flipped between v1 (per-rollout ROI pools: MES beats random winners 3/3) and v2
(fixed Sobol-600: random winners beat MES 3/3, MES IR_T 12 -> 28). A sparse global
pool cannot resolve the peak MES sharpens, so it under-credits MES -- a bias toward
the arm under test. This measures it instead of arguing about it.

Scores per rollout (final posterior P_T, retained via ir_keep_final_ko):
  IR600   E[max f_H] - max mu_H on the run-fixed Sobol-600            (v2, in hand)
  IR3000  same on Sobol-3000 (resolution check; too slow for the arms)
  IRROI   same on ONE ROI-600 pool per member built from the member's
          starting posterior with a fixed seed (shared within member)
  IMP     best OBSERVED fantasy HF value in the rollout minus the real
          incumbent -- the frozen metric's own quantity; needs no pool
"""
import os, sys, io, contextlib, importlib.util, time
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp"); BUDGET = 40.0
import src.policy.mf_dro as MF
from src.policy.mf_dro import _build_hf_proxy_model, build_roi_pool
from gumbel_thompson import thompson_sample_y_star
CAP = {}
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self, *a, **k): _oi(self, *a, **k); CAP["mf"] = self
MF.DirectMFRegretOptimization.__init__ = _spy
_s = importlib.util.spec_from_file_location("h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8; c.max_seq_length = 256
    c.absolute_timesteps = False; c.real_prefix_training = False
    c.disable_position_embedding = True; c.random_p_hf = 0.5
    c.ir_keep_final_ko = True
    return c
w._build_mf_dro_config = _b; w.ROLLOUT_REWARD = "inference_regret"; w.BUDGET = BUDGET
with contextlib.redirect_stdout(io.StringIO()):
    w.run("Borehole_8D", "MF-DRO", 42, os.path.join(SCR, "h207_sc0b.json"))
mf = CAP["mf"]; M = len(mf.ko_ensemble)
with contextlib.redirect_stdout(io.StringIO()):
    mf.config.rollout_mix = [('mes', 20, None)];    b_mes = mf._generate_rollout_batch()
    mf.config.rollout_mix = [('random', 100, None)]; b_rnd = mf._generate_rollout_batch()
    mf.config.rollout_mix = None
print("=" * 78); print(f"batches: MES {len(b_mes)}  random {len(b_rnd)}   M={M}", flush=True)

def ir_on(ko, pool, seed, K=100):
    proxy = _build_hf_proxy_model(ko); rs = torch.get_rng_state()
    try:
        torch.manual_seed(seed); ys = thompson_sample_y_star(proxy, pool, K=K)
    finally: torch.set_rng_state(rs)
    with torch.no_grad(): mu, _ = ko.hf_posterior(pool)
    return float(np.mean(ys)) - float(mu.max())

# pools
bounds = mf.bounds; d = mf.d
sob3000 = bounds[0] + (bounds[1] - bounds[0]) * torch.quasirandom.SobolEngine(d, scramble=True, seed=99).draw(3000).to(torch.float64)
roi_pools = []
for m, ko in enumerate(mf.ko_ensemble):
    rs = torch.get_rng_state(); torch.manual_seed(1234 + m)
    try:
        cand, _ = build_roi_pool(ko, bounds, 600, use_roi=True, roi_beta_mode='quantile',
                                 roi_beta_sqrt=2.0, roi_target_accept=0.10, roi_raw_pool=2000)
    finally: torch.set_rng_state(rs)
    roi_pools.append(cand)
incumbent = max(mf.data_hf_y) if mf.data_hf_y else -np.inf

def score_all(batch, n_per):
    out = []
    t0 = time.time()
    for i, t in enumerate(batch):
        m = i // n_per; ko = t['_final_ko']
        ell = t['actions_ell'].numpy(); ys = t['y_values'].numpy()
        best_hf = max([float(y) for y, e in zip(ys, ell) if e == 1], default=-np.inf)
        out.append(dict(m=m, IR600=t['ir_T'],
                        IR3000=ir_on(ko, sob3000, 7 + 0 * i),
                        IRROI=ir_on(ko, roi_pools[m], 11 + 0 * i),
                        IMP=(best_hf - incumbent) if np.isfinite(best_hf) else 0.0,
                        lf=float(t['lf_fraction'])))
    print(f"   scored {len(batch)} in {time.time()-t0:.0f}s", flush=True)
    return out
S_mes = score_all(b_mes, 20); S_rnd = score_all(b_rnd, 100)
KEYS = ("IR600", "IR3000", "IRROI", "IMP")
lower_is_better = {"IR600": True, "IR3000": True, "IRROI": True, "IMP": False}

print("\nSC5 under each score: MES-20 vs top-20-of-100 random (selected BY THAT score), per member")
for k in KEYS:
    row = []
    for m in range(M):
        a = np.array([s[k] for s in S_mes if s['m'] == m]); r = np.array([s[k] for s in S_rnd if s['m'] == m])
        lf = np.array([s['lf'] for s in S_rnd if s['m'] == m])
        order = np.argsort(r if lower_is_better[k] else -r); top = order[:20]
        mes_v, win_v = a.mean(), r[top].mean()
        better = (win_v < mes_v) if lower_is_better[k] else (win_v > mes_v)
        row.append(f"m{m}: MES {mes_v:7.3f}  win20 {win_v:7.3f}  LFwin {lf[top].mean():.2f}  {'RND' if better else 'MES'}")
    print(f"   {k:6s} ({'lower' if lower_is_better[k] else 'higher'} better)  " + " | ".join(row))

print("\nMES IR_T level by pool (resolution check; if IR600 >> IRROI the global pool under-credits the peak)")
for k in ("IR600", "IR3000", "IRROI"):
    print(f"   {k:6s}  MES mean {np.mean([s[k] for s in S_mes]):8.3f}    random mean {np.mean([s[k] for s in S_rnd]):8.3f}")

from scipy.stats import spearmanr
print("\nSpearman rank agreement between scores on the 300 random rollouts (within member, mean over members)")
for i, a in enumerate(KEYS):
    for b_ in KEYS[i+1:]:
        rs_ = []
        for m in range(M):
            x = [s[a] for s in S_rnd if s['m'] == m]; y = [s[b_] for s in S_rnd if s['m'] == m]
            sgn = 1 if lower_is_better[a] == lower_is_better[b_] else -1
            rs_.append(sgn * spearmanr(x, y).correlation)
        print(f"   {a:6s} vs {b_:6s}: rho = {np.mean(rs_):+.3f}")

print("\nAction-informativeness at tau=0 (the CORRECT Layer-3 quantity): R^2 of score on (x_0, ell_0) within member")
def r2_action(batch, S, n_per):
    r2 = {k: [] for k in KEYS}
    for m in range(M):
        idx = [i for i in range(len(batch)) if i // n_per == m]
        X = np.stack([np.concatenate([batch[i]['actions_x'][0].numpy() if batch[i].get('actions_x') is not None
                                       else np.zeros(d), [float(batch[i]['actions_ell'][0])]]) for i in idx])
        X = np.hstack([X, np.ones((len(idx), 1))])
        for k in KEYS:
            y = np.array([S[i][k] for i in idx])
            if y.std() < 1e-12 or len(idx) <= X.shape[1]: r2[k].append(float('nan')); continue
            beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
            r2[k].append(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {k: float(np.nanmean(v)) for k, v in r2.items()}
ra, rr = r2_action(b_mes, S_mes, 20), r2_action(b_rnd, S_rnd, 100)
for k in KEYS:
    print(f"   {k:6s}  MES (n=20/member, {d+2} params -> in-sample, inflated) R^2={ra[k]:.3f}    random (n=100/member) R^2={rr[k]:.3f}")
print("   (MES at n=20 with 10 params is near-saturated by construction; the random column is the usable number.)")
print("=" * 78)
