"""h152 sanity checks SC1-SC3, SC5. Builds a real KO GP on a Borehole
initial design and exercises the beam against the greedy teacher."""
import os, sys, json, math
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.joint_ig_teacher import beam_search_trajectory, gumbel_b

torch.manual_seed(0); np.random.seed(0)
hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
f_h, f_l = hf["make_objective"](), lf["make_objective"]()

def draw(n):
    return bounds[0] + (bounds[1]-bounds[0])*torch.rand(n, d, dtype=torch.float64)

X_h, X_l = draw(8), draw(24)
Y_h, Y_l = f_h(X_h).reshape(-1), f_l(X_l).reshape(-1)
ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
ko.fit(X_l, Y_l, X_h, Y_h, bounds)
pool = draw(200)
T = 8

print("=== SC1: beam(B=1,k=1) == greedy teacher, step for step ===")
# The beam calls gumbel_b while pruning, which CONSUMES RNG. So the reference
# replay must interleave the identical call order -- MES, fantasy, condition,
# gumbel -- or the fantasy draws diverge and the paths differ for reasons that
# have nothing to do with the selection rule.
torch.manual_seed(7)
xb, eb, info = beam_search_trajectory(ko, pool, T, c_H, c_L,
                                      cost_cap=1e18, beam_width=1, branch=1)
torch.manual_seed(7)
cur, xg, eg = ko, [], []
for _ in range(T):
    x, e, _s = compute_joint_mf_mes(cur, pool, c_H, c_L)
    y = cur.sample_fantasy(x, "LH"[e], mode="sample")
    cur = cur.make_fantasy_ko(x.unsqueeze(0),
                              torch.tensor([y], dtype=torch.float64), "LH"[e])
    _ = gumbel_b(cur, pool)          # matches the beam's pruning call
    xg.append(x); eg.append(e)
xg = torch.stack(xg)
sc1 = bool(torch.equal(xb, xg)) and eb == eg
print(f"  x identical: {bool(torch.equal(xb,xg))}   ell identical: {eb==eg}   -> SC1 {'PASS' if sc1 else 'FAIL'}")
print(f"  greedy ells={eg}  cost={sum(c_H if e else c_L for e in eg):.1f}")
greedy_cost = sum(c_H if e else c_L for e in eg)

print("\n=== SC2: beam b_T <= its OWN in-run greedy b_T (elite never pruned) ===")
rows = []
for B, k in [(1,1), (2,2), (4,4), (6,4), (8,6)]:
    torch.manual_seed(7)
    _x, _e, inf = beam_search_trajectory(ko, pool, T, c_H, c_L,
                                         cost_cap=greedy_cost, beam_width=B, branch=k)
    ok = inf["b_T"] <= inf["b_T_greedy"] + 1e-12
    rows.append((B, k, inf["b_T"], inf["b_T_greedy"], inf["cost"], ok))
    lift = math.log(inf["b_T_greedy"]) - math.log(inf["b_T"])
    print(f"  B={B} k={k}: b_T={inf['b_T']:8.5f}  greedy={inf['b_T_greedy']:8.5f}  "
          f"lift(rtg)={lift:+.4f}  cost={inf['cost']:.1f}  elite_won={inf['won_by_elite']}  "
          f"{'ok' if ok else 'VIOLATION'}")
sc2 = all(r[5] for r in rows)
print(f"  -> SC2 {'PASS' if sc2 else 'FAIL'}")

print("\n=== SC3: cost cap respected ===")
ok3 = all(r[4] <= greedy_cost + 1e-9 for r in rows)
print(f"  cap={greedy_cost:.1f}  max observed={max(r[4] for r in rows):.1f}  -> SC3 {'PASS' if ok3 else 'FAIL'}")

print("\n=== SC5: emitted x drawn from pool and in bounds ===")
torch.manual_seed(7)
xb4, _, _ = beam_search_trajectory(ko, pool, T, c_H, c_L, greedy_cost, 4, 4)
inpool = all(any(torch.equal(xb4[i], p) for p in pool) for i in range(xb4.shape[0]))
inb = bool(((xb4 >= bounds[0]).all() and (xb4 <= bounds[1]).all()))
print(f"  all in pool: {inpool}   in bounds: {inb}   -> SC5 {'PASS' if inpool and inb else 'FAIL'}")
