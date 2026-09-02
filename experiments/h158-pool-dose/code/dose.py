"""h158 -- POOL dose on the tail axis. Sweeps h146's endpoint pool
{16,256,4096}: quality UP, diversity DOWN, on one axis."""
import os, sys, json, math, argparse
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import (compute_joint_mf_mes, _build_hf_proxy_model,
                                _compute_mes_hf_vectorized, _compute_mes_lf_vectorized)
from gumbel_thompson import thompson_sample_y_star
from src.policy.joint_ig_teacher import gumbel_b

T, POOL_N = 8, 200
DOSE = [16, 256, 4096]


def _interp(a, b, T):
    return torch.stack([a + (b - a) * (t / max(T - 1, 1)) for t in range(T)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--states", type=int, default=3); ap.add_argument("--models", type=int, default=10)
    a = ap.parse_args()
    hf, lf = get_benchmark("Borehole_8D_HF"), get_benchmark("Borehole_8D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
    f_hf = hf["make_objective"](); lo, hi = bounds[0], bounds[1]

    def fid_of(P, ko0, pool):
        proxy = _build_hf_proxy_model(ko0); ys = thompson_sample_y_star(proxy, pool, K=10)
        out = []
        for i in range(P.shape[0]):
            xf = P[i].reshape(1, -1)
            mh = float(_compute_mes_hf_vectorized(xf, proxy, ys)[0])
            ml = float(_compute_mes_lf_vectorized(xf, ko0, ys, n_quad=32)[0])
            out.append(1 if (mh / c_H) >= (ml / c_L) else 0)
        return out

    def replay(ko, pool, xs, els):
        cur = ko
        for x, e in zip(xs, els):
            y = cur.sample_fantasy(x, "LH"[e], mode="sample")
            cur = cur.make_fantasy_ko(x.unsqueeze(0), torch.tensor([y], dtype=torch.float64), "LH"[e])
        return gumbel_b(cur, pool)

    recs = []
    for seed in a.seeds:
        tr = json.load(open(f"{REPO}/experiments/h83-main-comparison/results/"
                            f"Borehole_8D__MF-DRO__seed{seed}.json"))["queries"]
        n_init = sum(1 for e in tr if e.get("is_init"))
        for cut in np.linspace(n_init, len(tr), a.states + 1)[:-1].astype(int):
            sd = 31000 + 13 * seed + int(cut) + 100003 * a.rep
            torch.manual_seed(sd); np.random.seed(sd)
            t = lambda z: torch.tensor(z, dtype=torch.float64)
            Xh = [e["x"] for e in tr[:cut] if e["fid"] == 1]; Yh = [e["y"] for e in tr[:cut] if e["fid"] == 1]
            Xl = [e["x"] for e in tr[:cut] if e["fid"] == 0]; Yl = [e["y"] for e in tr[:cut] if e["fid"] == 0]
            if len(Xh) < 3 or len(Xl) < 3: continue
            ens = []
            for m in range(a.models):
                torch.manual_seed(sd + 7919 * m)
                k = KennedyOHaganGP(d=d, dkl_threshold=9999)
                k.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds); ens.append(k)
            pool = lo + (hi - lo) * torch.rand(POOL_N, d, dtype=torch.float64)
            rec = dict(seed=int(seed), cut=int(cut))
            # control reference on the same states
            c1 = []
            per = max(1, a.N // len(ens))
            for i in range(per * len(ens)):
                ko = ens[i // per]; lb0 = math.log(gumbel_b(ko, pool))
                cur, xs, els = ko, [], []
                for _ in range(T):
                    x, e, _ = compute_joint_mf_mes(cur, pool, c_H, c_L)
                    y = cur.sample_fantasy(x, "LH"[e], mode="sample")
                    cur = cur.make_fantasy_ko(x.unsqueeze(0), torch.tensor([y], dtype=torch.float64), "LH"[e])
                    _ = gumbel_b(cur, pool); xs.append(x); els.append(e)
                c1.append(lb0 - math.log(gumbel_b(cur, pool)))
            rec["C1_closed"] = dict(mean=float(np.mean(c1)), max=float(np.max(c1)))
            for P in DOSE:
                vals, ends = [], []
                for i in range(per * len(ens)):
                    ko = ens[i // per]; lb0 = math.log(gumbel_b(ko, pool))
                    cand = lo + (hi - lo) * torch.rand(P, d, dtype=torch.float64)
                    yv = f_hf(cand).reshape(-1); x_end = cand[yv.argmax()]
                    ends.append((x_end, float(yv.max())))
                    x0 = lo + (hi - lo) * torch.rand(d, dtype=torch.float64)
                    path = _interp(x0, x_end, T)
                    vals.append(lb0 - math.log(replay(ko, pool, path, fid_of(path, ko, pool))))
                E = torch.stack([e[0] for e in ends])
                Enorm = (E - lo) / (hi - lo)
                rec[f"POOL{P}"] = dict(mean=float(np.mean(vals)), max=float(np.max(vals)),
                                       endpoint_value=float(np.mean([e[1] for e in ends])),
                                       endpoint_spread=float(Enorm.std(0).mean()))
            recs.append(rec)
            print(f"  s{seed} cut{cut:3d} C1max {rec['C1_closed']['max']:+.3f} | " + " | ".join(
                f"P{P}: max {rec[f'POOL{P}']['max']:+.3f} val {rec[f'POOL{P}']['endpoint_value']:8.2f} "
                f"spread {rec[f'POOL{P}']['endpoint_spread']:.3f}" for P in DOSE), flush=True)
    out = f"{REPO}/experiments/h158-pool-dose/results/dose_r{a.rep}.json"
    json.dump(recs, open(out, "w")); print(f"\nwrote {out} ({len(recs)} states)")


if __name__ == "__main__":
    main()
