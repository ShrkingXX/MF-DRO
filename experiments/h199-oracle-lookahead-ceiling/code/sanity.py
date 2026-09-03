"""h199 Stage 0. SC-DIVERSITY is a GATE, registered in protocol.md before running."""
import os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.regret_lookahead_teacher import choose_regret_lookahead

hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
f_h, f_l = hf["make_objective"](), lf["make_objective"]()
ORACLE = {"H": f_h, "L": f_l}
draw = lambda n: bounds[0] + (bounds[1]-bounds[0])*torch.rand(n, d, dtype=torch.float64)

def mk(sd):
    torch.manual_seed(sd); Xh, Xl = draw(8), draw(24)
    ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
    ko.fit(Xl, f_l(Xl).reshape(-1), Xh, f_h(Xh).reshape(-1), bounds)
    return ko, draw(600)

N = 14; ok = True
mes_x, gp_x, or_x, or_e, gp_e = [], [], [], [], []
for sd in range(N):
    ko, pool = mk(300+sd)
    torch.manual_seed(21); gx, ge, _ = compute_joint_mf_mes(ko, pool, c_H, c_L)
    torch.manual_seed(21); px, pe, _, _ = choose_regret_lookahead(
        ko, pool, c_H, c_L, 0.0, steps_left=8, n_c=4, M=4, base_pool=150)
    torch.manual_seed(21); ox, oe, _, _ = choose_regret_lookahead(
        ko, pool, c_H, c_L, 0.0, steps_left=8, n_c=4, M=1, base_pool=150, oracle_f=ORACLE)
    nrm = lambda t: ((t - bounds[0])/(bounds[1]-bounds[0])).reshape(-1)
    mes_x.append(nrm(gx)); gp_x.append(nrm(px)); or_x.append(nrm(ox))
    or_e.append(oe); gp_e.append(pe)

M_ = torch.stack(mes_x); G_ = torch.stack(gp_x); O_ = torch.stack(or_x)
sd_m = float(M_.std(dim=0).mean()); sd_g = float(G_.std(dim=0).mean()); sd_o = float(O_.std(dim=0).mean())
print("="*70)
print("SC-DIVERSITY (GATE): across-rollout SD of the tau=0 action, normalised units")
print(f"   MES teacher       : {sd_m:.4f}   (reference)")
print(f"   h198 GP-lookahead : {sd_g:.4f}   ({100*sd_g/sd_m:5.1f}% of MES)")
print(f"   h199 ORACLE       : {sd_o:.4f}   ({100*sd_o/sd_m:5.1f}% of MES)")
gate = (sd_o / sd_m) >= 0.25
print(f"   -> {'PASS' if gate else 'GATE MISS -- degenerate, h146 confound; Stage 1 does NOT run'}")
ok &= gate

print("\nSC-ORACLE-USED: does the oracle change the choice vs h198?")
diff = sum(1 for a, b, ea, eb in zip(or_x, gp_x, or_e, gp_e)
           if float((a-b).abs().max()) > 1e-12 or ea != eb)
print(f"   differs from GP-lookahead on {diff}/{N} starts")
sc2 = diff >= max(3, N//4); ok &= sc2
print(f"   -> {'PASS' if sc2 else 'FAIL (silent no-op)'}")

print("\nSC1: n_c=1 still reduces exactly to greedy MES under the oracle")
agr = 0
ko, pool = mk(777)
for t in range(6):
    torch.manual_seed(600+t); gx, ge, _ = compute_joint_mf_mes(ko, pool, c_H, c_L)
    torch.manual_seed(600+t); rx, re_, _, _ = choose_regret_lookahead(
        ko, pool, c_H, c_L, 0.0, steps_left=8-t, n_c=1, M=1, oracle_f=ORACLE)
    agr += int(bool(torch.equal(gx, rx) and ge == re_))
print(f"   identical on {agr}/6 -> {'PASS' if agr==6 else 'FAIL'}"); ok &= (agr==6)
print("="*70); print("STAGE 0:", "PASS" if ok else "FAIL")
