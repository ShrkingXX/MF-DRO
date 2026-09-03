"""h198 Stage 0 -- the SCs registered in protocol.md, run BEFORE any arm."""
import os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.regret_lookahead_teacher import choose_regret_lookahead

torch.manual_seed(0); np.random.seed(0)
hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
f_h, f_l = hf["make_objective"](), lf["make_objective"]()

def draw(n):
    return bounds[0] + (bounds[1]-bounds[0])*torch.rand(n, d, dtype=torch.float64)

def mk(n_hf=8, n_lf=24):
    X_h, X_l = draw(n_hf), draw(n_lf)
    Y_h, Y_l = f_h(X_h).reshape(-1), f_l(X_l).reshape(-1)
    ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
    ko.fit(X_l, Y_l, X_h, Y_h, bounds)
    return ko, float(Y_h.max())

ko, best = mk()
pool = draw(600)
ok = True

print("="*68)
print("SC1: n_c=1 reduces EXACTLY to greedy MES, step for step")
# compute_joint_mf_mes THOMPSON-SAMPLES y*, so it consumes RNG and is itself
# stochastic. The first version of this SC called it once for the reference and
# once inside the teacher and compared -- which measured MES's own sampling
# noise and "failed" 2/6 for a reason having nothing to do with the teacher.
# (h152's sanity documents the same trap for the beam's gumbel_b calls.) Both
# calls are therefore seeded identically.
_agree = 0
for _t in range(6):
    torch.manual_seed(500+_t)
    gx, ge, _ = compute_joint_mf_mes(ko, pool, c_H, c_L)
    torch.manual_seed(500+_t)
    rx, re_, _, inf = choose_regret_lookahead(ko, pool, c_H, c_L, best,
                                              steps_left=8-_t, n_c=1, M=1)
    same = bool(torch.equal(gx, rx) and ge == re_)
    _agree += int(same)
    print(f"   step {_t}: greedy ell={ge}  lookahead ell={re_}  x identical={bool(torch.equal(gx,rx))}")
sc1 = (_agree == 6); ok &= sc1
print(f"   -> {'PASS' if sc1 else 'FAIL'}  ({_agree}/6 identical)")

print("\nSC2: winner's curse -- is the argmax stable as M grows?")
res = {}
for M in (1, 2, 4, 8):
    torch.manual_seed(7)
    _x, _e, _, inf = choose_regret_lookahead(ko, pool, c_H, c_L, best,
                                             steps_left=8, n_c=4, M=M)
    res[M] = (_x.clone(), _e, inf)
    print(f"   M={M}: spread={inf['score_spread']:.4f}  "
          f"greedy_minus_best={inf['greedy_minus_best']:+.4f}  "
          f"chose_greedy={inf['chose_greedy']}")
print(f"   argmax stable M=4 vs M=8: "
      f"{bool(torch.equal(res[4][0], res[8][0]) and res[4][1]==res[8][1])}")
print("   -> reported, not gated (M=4 is the arm's setting)")

print("\nSC3 (GATE): does the tau=0 action differ from greedy MES?  [ARM CONFIG n_c=4 M=4 base_pool=150]")
_n, _diff, _dists, _fid = 12, 0, [], 0
for _s in range(_n):
    torch.manual_seed(100+_s); np.random.seed(100+_s)
    _ko, _b = mk()
    _p = draw(600)
    torch.manual_seed(900+_s)
    gx, ge, _ = compute_joint_mf_mes(_ko, _p, c_H, c_L)
    torch.manual_seed(900+_s)
    rx, re_, _, _i = choose_regret_lookahead(_ko, _p, c_H, c_L, _b,
                                             steps_left=8, n_c=4, M=4)
    _nd = float(((gx-rx)/(bounds[1]-bounds[0])).abs().max())
    _dists.append(_nd); _diff += int(_nd > 1e-12 or ge != re_); _fid += int(ge != re_)
print(f"   differs from greedy on {_diff}/{_n} rollout starts "
      f"(fidelity differs on {_fid}/{_n})")
print(f"   normalised |x_greedy - x_lookahead|_inf: mean={np.mean(_dists):.4f} "
      f"max={np.max(_dists):.4f}")
sc3 = _diff >= max(3, _n//4); ok &= sc3
print(f"   -> {'PASS' if sc3 else 'FAIL (P3: the teacher is not doing what it claims)'}")
print("="*68); print("STAGE 0:", "PASS" if ok else "FAIL")
