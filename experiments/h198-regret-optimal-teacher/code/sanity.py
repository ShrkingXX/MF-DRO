"""h198 Stage 0 -- the SCs registered in protocol.md, run BEFORE any arm."""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.regret_lookahead_teacher import choose_regret_lookahead
from benchmarks import get_benchmark
from src.model.kennedy_ohagan import KennedyOHaganGP

torch.manual_seed(0); np.random.seed(0)
hf = get_benchmark("Borehole_8D_HF"); lf = get_benchmark("Borehole_8D_LF")
bnds = torch.tensor(hf.bounds, dtype=torch.float64).T if np.ndim(hf.bounds) == 2 else None
lo = torch.tensor([b[0] for b in hf.bounds], dtype=torch.float64)
hi = torch.tensor([b[1] for b in hf.bounds], dtype=torch.float64)
d = lo.numel(); c_H, c_L = 2.0, 1.0

def _mk(n_hf=12, n_lf=30):
    xh = lo + (hi-lo)*torch.rand(n_hf, d, dtype=torch.float64)
    xl = lo + (hi-lo)*torch.rand(n_lf, d, dtype=torch.float64)
    yh = torch.tensor([float(hf.f(x.numpy())) for x in xh], dtype=torch.float64)
    yl = torch.tensor([float(lf.f(x.numpy())) for x in xl], dtype=torch.float64)
    ko = KennedyOHaganGP(d=d)
    ko.fit(xl, yl, xh, yh)
    return ko, float(yh.max())

ko, best = _mk()
pool = lo + (hi-lo)*torch.rand(600, d, dtype=torch.float64)
ok = True

print("="*66)
print("SC1: n_c=1 reduces EXACTLY to greedy MES, step for step")
_agree = 0
for _t in range(6):
    gx, ge, _ = compute_joint_mf_mes(ko, pool, c_H, c_L)
    rx, re_, _, inf = choose_regret_lookahead(ko, pool, c_H, c_L, best,
                                              steps_left=8-_t, n_c=1, M=1)
    same = bool(torch.equal(gx, rx) and ge == re_)
    _agree += int(same)
    print(f"   step {_t}: greedy=({ge}) lookahead=({re_}) identical_x={bool(torch.equal(gx,rx))}")
sc1 = (_agree == 6); ok &= sc1
print(f"   -> {'PASS' if sc1 else 'FAIL'}  ({_agree}/6 identical)")

print("\nSC2: winner's curse -- does the argmax survive more fantasies?")
res = {}
for M in (1, 2, 4, 8):
    torch.manual_seed(7)
    _x, _e, _, inf = choose_regret_lookahead(ko, pool, c_H, c_L, best,
                                             steps_left=8, n_c=8, M=M)
    res[M] = (_x.clone(), _e, inf)
    print(f"   M={M}: spread={inf['score_spread']:.4f}  "
          f"greedy_minus_best={inf['greedy_minus_best']:+.4f}  "
          f"chose_greedy={inf['chose_greedy']}")
_stable = bool(torch.equal(res[4][0], res[8][0]) and res[4][1] == res[8][1])
print(f"   argmax stable between M=4 and M=8: {_stable}")
print(f"   -> reported, not gated (M=4 is the arm's setting)")

print("\nSC3 (GATE): does the tau=0 action differ from greedy MES?")
_n, _diff, _dists = 12, 0, []
for _s in range(_n):
    torch.manual_seed(100+_s)
    _ko, _b = _mk()
    _p = lo + (hi-lo)*torch.rand(600, d, dtype=torch.float64)
    gx, ge, _ = compute_joint_mf_mes(_ko, _p, c_H, c_L)
    rx, re_, _, _i = choose_regret_lookahead(_ko, _p, c_H, c_L, _b,
                                             steps_left=8, n_c=8, M=4)
    _nd = float(((gx-rx)/(hi-lo)).abs().max())
    _dists.append(_nd); _diff += int(_nd > 1e-12 or ge != re_)
print(f"   differs from greedy on {_diff}/{_n} rollout starts")
print(f"   normalised |x_greedy - x_lookahead|_inf: mean={np.mean(_dists):.4f} "
      f"max={np.max(_dists):.4f}")
sc3 = _diff >= max(3, _n//4); ok &= sc3
print(f"   -> {'PASS' if sc3 else 'FAIL (P3: teacher is not doing what it claims)'}")
print("="*66); print("STAGE 0:", "PASS" if ok else "FAIL")
