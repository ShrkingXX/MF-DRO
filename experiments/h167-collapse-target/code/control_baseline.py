"""h167c -- best-constant MSE for the MES (control) teacher's action distribution,
generated on real Borehole states, to make L_loc comparable across arms."""
import os, sys, json
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.joint_ig_teacher import gumbel_b

T, POOL_N, NTRAJ = 8, 200, 60
hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
lo, hi = bounds[0], bounds[1]
A = []
for seed in (42, 43):
    tr = json.load(open(f"{REPO}/experiments/h83-main-comparison/results/"
                        f"Borehole_8D__MF-DRO__seed{seed}.json"))["queries"]
    n_init = sum(1 for e in tr if e.get("is_init"))
    for cut in np.linspace(n_init, len(tr), 4)[:-1].astype(int):
        torch.manual_seed(4200 + seed + int(cut)); np.random.seed(4200 + seed + int(cut))
        t = lambda z: torch.tensor(z, dtype=torch.float64)
        Xh = [e["x"] for e in tr[:cut] if e["fid"] == 1]; Yh = [e["y"] for e in tr[:cut] if e["fid"] == 1]
        Xl = [e["x"] for e in tr[:cut] if e["fid"] == 0]; Yl = [e["y"] for e in tr[:cut] if e["fid"] == 0]
        if len(Xh) < 3 or len(Xl) < 3: continue
        ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
        ko.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds)
        pool = lo + (hi - lo) * torch.rand(POOL_N, d, dtype=torch.float64)
        for _ in range(NTRAJ):
            cur = ko
            for _s in range(T):
                x, e, _ = compute_joint_mf_mes(cur, pool, c_H, c_L)
                A.append(((x - lo) / (hi - lo)).numpy())
                y = cur.sample_fantasy(x, "LH"[e], mode="sample")
                cur = cur.make_fantasy_ko(x.unsqueeze(0), torch.tensor([y], dtype=torch.float64), "LH"[e])
                _ = gumbel_b(cur, pool)
A = np.array(A)
m = A.mean(axis=0)
mse_const = float(((A - m) ** 2).mean())
print(f"\n  MES (control) teacher actions: {len(A)} samples over 6 real states")
print(f"  target mean (first 3 dims) {np.round(m,3)[:3]}")
print(f"  best-constant MSE          {mse_const:.4f}")
print(f"  observed L_loc (control)   0.040")
print(f"  ratio (constant/observed)  {mse_const/0.040:.2f}\n")
print(f"  {'arm':16s} {'best const':>11s} {'observed':>10s} {'ratio':>7s}")
for k, c, o in (("control (works)", mse_const, 0.040), ("RANDOM (fails)", 0.0834, 0.020),
                ("ORACLE (fails)", 0.0533, 0.020), ("DIVERSE (fails)", 0.0544, 0.020)):
    print(f"  {k:16s} {c:11.4f} {o:10.3f} {c/o:7.2f}")
json.dump(dict(mse_const=mse_const, n=len(A)),
          open(f"{REPO}/experiments/h167-collapse-target/results/control_baseline.json", "w"))
