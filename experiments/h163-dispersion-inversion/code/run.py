"""h163 -- TEACHER-side action dispersion, to set against h162's student side."""
import os, sys, json, math
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.joint_ig_teacher import gumbel_b

T, POOL_N, NTRAJ = 8, 200, 40
XSTAR = [0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0]
GOOD_POOL = 256

hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
lo, hi = bounds[0], bounds[1]; f_hf = hf["make_objective"]()
xstar = torch.tensor(XSTAR, dtype=torch.float64)
interp = lambda a, b: torch.stack([a + (b - a) * (t / (T - 1)) for t in range(T)])


def adaptive(ko, pool, kind, beta=2.0):
    """MES argmax, or UCB(beta) argmax, re-decided each step (closed-loop)."""
    cur, xs = ko, []
    for _ in range(T):
        if kind == "mes":
            x, e, _ = compute_joint_mf_mes(cur, pool, c_H, c_L)
        else:
            with torch.no_grad():
                mu, var = cur.hf_posterior(pool)
            x = pool[(mu + beta * var.clamp_min(0).sqrt()).argmax()]; e = 1
        y = cur.sample_fantasy(x, "LH"[e], mode="sample")
        cur = cur.make_fantasy_ko(x.unsqueeze(0), torch.tensor([y], dtype=torch.float64), "LH"[e])
        xs.append(x)
    return torch.stack(xs)


def disp(P):
    X = (P - lo) / (hi - lo)
    D = torch.cdist(X, X); iu = torch.triu_indices(len(X), len(X), 1)
    return float(D[iu[0], iu[1]].mean())


out, allpts = {}, {}
for seed in (42, 43):
    tr = json.load(open(f"{REPO}/experiments/h83-main-comparison/results/"
                        f"Borehole_8D__MF-DRO__seed{seed}.json"))["queries"]
    n_init = sum(1 for e in tr if e.get("is_init"))
    for cut in np.linspace(n_init, len(tr), 4)[:-1].astype(int):
        torch.manual_seed(900 + seed + int(cut)); np.random.seed(900 + seed + int(cut))
        t = lambda z: torch.tensor(z, dtype=torch.float64)
        Xh = [e["x"] for e in tr[:cut] if e["fid"] == 1]; Yh = [e["y"] for e in tr[:cut] if e["fid"] == 1]
        Xl = [e["x"] for e in tr[:cut] if e["fid"] == 0]; Yl = [e["y"] for e in tr[:cut] if e["fid"] == 0]
        if len(Xh) < 3 or len(Xl) < 3: continue
        ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
        ko.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds)
        pool = lo + (hi - lo) * torch.rand(POOL_N, d, dtype=torch.float64)
        acc = {k: [] for k in ("MES (control/h153)", "UCB b=2 (h155)", "UCB b=0 (h159)",
                               "ORACLE", "DIVERSE-GOOD", "RANDOM")}
        for _ in range(NTRAJ):
            acc["MES (control/h153)"].append(adaptive(ko, pool, "mes"))
            acc["UCB b=2 (h155)"].append(adaptive(ko, pool, "ucb", 2.0))
            acc["UCB b=0 (h159)"].append(adaptive(ko, pool, "ucb", 0.0))
            x0 = lo + (hi - lo) * torch.rand(d, dtype=torch.float64)
            acc["ORACLE"].append(interp(x0, xstar))
            cand = lo + (hi - lo) * torch.rand(GOOD_POOL, d, dtype=torch.float64)
            acc["DIVERSE-GOOD"].append(interp(lo + (hi - lo) * torch.rand(d, dtype=torch.float64),
                                              cand[f_hf(cand).reshape(-1).argmax()]))
            acc["RANDOM"].append(pool[torch.randint(0, POOL_N, (T,))])
        for k, v in acc.items():
            out.setdefault(k, []).append(disp(torch.cat(v)))
        print(f"  s{seed} cut{cut:3d} " + "  ".join(f"{k.split()[0]} {out[k][-1]:.3f}" for k in acc), flush=True)

STUDENT = {"MES (control/h153)": 0.2766, "UCB b=2 (h155)": 0.2889, "UCB b=0 (h159)": None,
           "ORACLE": 0.1891, "DIVERSE-GOOD": 0.1830, "RANDOM": 0.1115}
print(f"\n  {'teacher rule':22s} {'TEACHER M1':>11s} {'STUDENT M1':>11s}")
rows = []
for k in ("MES (control/h153)", "UCB b=2 (h155)", "UCB b=0 (h159)", "ORACLE", "DIVERSE-GOOD", "RANDOM"):
    tm = float(np.mean(out[k])); sm = STUDENT[k]
    print(f"  {k:22s} {tm:11.4f} {(f'{sm:11.4f}' if sm else '      (h159)')}")
    if sm: rows.append((tm, sm))
a = np.array(rows)
from scipy.stats import spearmanr
r, _ = spearmanr(a[:, 0], a[:, 1])
print(f"\n  Spearman(teacher M1, student M1) over {len(a)} arms = {r:+.3f}")
print(f"  prediction: NEGATIVE (inversion).  natural null 'student copies teacher': POSITIVE")
print(f"  -> {'INVERTED as predicted (R2)' if r < -0.3 else ('POSITIVE -- R1 fires' if r > 0.3 else 'no clear ordering (R3)')}")
json.dump({k: float(np.mean(v)) for k, v in out.items()},
          open(f"{REPO}/experiments/h163-dispersion-inversion/results/teacher_m1.json", "w"), indent=1)
