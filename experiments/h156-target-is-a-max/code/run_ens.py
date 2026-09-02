"""h156 -- is the rtg_target collapse a TAIL effect rather than a MEAN effect?

rtg_target = max(batch_max, 0.5*running_max) over rtg[0] (mf_dro.py:2056-2060).
Every penalty measured so far is a penalty on the MEAN. This measures the MAX.
"""
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

T = 8
POOL_N = 200
# h145's x* (raw domain) and h146's endpoint pool size, copied verbatim.
XSTAR = {"Borehole_8D": [0.15, 100.0, 95090.9777, 1110.0, 116.0, 700.0, 1120.0, 12045.0],
         "Hartmann_6D": [0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573]}
GOOD_POOL = 256


def _interp(x_start, x_end, T):
    return torch.stack([x_start + (x_end - x_start) * (t / max(T - 1, 1)) for t in range(T)])


def greedy_path(ko, pool, c_H, c_L):
    """One adaptive greedy rollout; returns its terminal b and its own path."""
    cur, xs, els = ko, [], []
    for _ in range(T):
        x, e, _ = compute_joint_mf_mes(cur, pool, c_H, c_L)
        y = cur.sample_fantasy(x, "LH"[e], mode="sample")
        cur = cur.make_fantasy_ko(x.unsqueeze(0),
                                  torch.tensor([y], dtype=torch.float64), "LH"[e])
        _ = gumbel_b(cur, pool)
        xs.append(x); els.append(e)
    return gumbel_b(cur, pool), torch.stack(xs), els


def replay(ko, pool, xs, els):
    cur = ko
    for x, e in zip(xs, els):
        y = cur.sample_fantasy(x, "LH"[e], mode="sample")
        cur = cur.make_fantasy_ko(x.unsqueeze(0),
                                  torch.tensor([y], dtype=torch.float64), "LH"[e])
    return gumbel_b(cur, pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="Borehole_8D")
    ap.add_argument("--models", type=int, default=1,
                    help="h156e: ensemble size (gp_num_models=10 in the real run)")
    ap.add_argument("--rep", type=int, default=0, help="replicate id -> seed offset + filename")
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--states", type=int, default=3)
    a = ap.parse_args()

    hf, lf = get_benchmark(f"{a.bench}_HF"), get_benchmark(f"{a.bench}_LF")
    f_hf = hf["make_objective"]()
    xstar = torch.tensor(XSTAR[a.bench], dtype=torch.float64)
    # forced paths take the SAME fidelity rule the arms used: fidelity is not
    # forced, it follows the cost-normalised criterion. Approximated here by the
    # arms' realised HF-heavy mix so the cost profile is comparable.
    def fid_of(P, ko0, pool):
        """h156d: the arms' OWN fidelity rule, verbatim (mf_dro.py:1618ff).
        y* is Thompson-estimated over the FULL pool -- never a one-point pool,
        which was the h145 v1 degenerate-y* bug. Replaces a hardcoded
        `1 if rand()<0.75 else 0` coin flip that fit the interpolating
        conditions badly (C4 Hartmann -31.2%, C5 Borehole -19.3%) while leaving
        the random condition, which is insensitive to fidelity, fitting well.
        Note this is evaluated on the START model for the whole path rather
        than re-derived per step, since the replay conditions sequentially --
        an approximation, but of the RULE rather than of a coin."""
        proxy = _build_hf_proxy_model(ko0)
        ys = thompson_sample_y_star(proxy, pool, K=10)
        out = []
        for i in range(P.shape[0]):
            xf = P[i].reshape(1, -1)
            mh = float(_compute_mes_hf_vectorized(xf, proxy, ys)[0])
            ml = float(_compute_mes_lf_vectorized(xf, ko0, ys, n_quad=32)[0])
            out.append(1 if (mh / c_H) >= (ml / c_L) else 0)
        return out
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
    recs = []

    for seed in a.seeds:
        p = f"{REPO}/experiments/h83-main-comparison/results/{a.bench}__MF-DRO__seed{seed}.json"
        tr = json.load(open(p))["queries"]
        n_init = sum(1 for e in tr if e.get("is_init"))
        for cut in np.linspace(n_init, len(tr), a.states + 1)[:-1].astype(int):
            _sd = 7000 + 13 * seed + int(cut) + 100003 * a.rep
            torch.manual_seed(_sd); np.random.seed(_sd)
            t = lambda z: torch.tensor(z, dtype=torch.float64)
            Xh = [e["x"] for e in tr[:cut] if e["fid"] == 1]; Yh = [e["y"] for e in tr[:cut] if e["fid"] == 1]
            Xl = [e["x"] for e in tr[:cut] if e["fid"] == 0]; Yl = [e["y"] for e in tr[:cut] if e["fid"] == 0]
            if len(Xh) < 3 or len(Xl) < 3: continue
            # h156e: the real batch draws from gp_num_models=10 KO-GPs fit on
            # IDENTICAL data (mf_dro.py:_update_ko_ensemble), differing only by
            # fitting randomness. A single GP omits that between-model variance,
            # which a MAX over the batch feeds on -- the suspected source of the
            # one-sided under-prediction.
            ens = []
            for _m in range(a.models):
                torch.manual_seed(_sd + 7919 * _m)
                _k = KennedyOHaganGP(d=d, dkl_threshold=9999)
                _k.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds)
                ens.append(_k)
            ko = ens[0]
            pool = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(POOL_N, d, dtype=torch.float64)
            c1, c2, c3, c4, c5 = [], [], [], [], []
            per = max(1, a.N // len(ens))
            for _i in range(per * len(ens)):
                ko = ens[_i // per]                    # 20 rollouts per member
                lb0 = math.log(gumbel_b(ko, pool))     # b_0 is member-specific
                bT, xs, els = greedy_path(ko, pool, c_H, c_L)   # C1 closed-loop
                c1.append(lb0 - math.log(bT))
                c2.append(lb0 - math.log(replay(ko, pool, xs, els)))   # C2 freeze its OWN path
                lo, hi = bounds[0], bounds[1]
                xs0 = lo + (hi - lo) * torch.rand(d, dtype=torch.float64)
                _p4 = _interp(xs0, xstar, T)
                c4.append(lb0 - math.log(replay(               # C4 ORACLE path
                    ko, pool, _p4, fid_of(_p4, ko, pool))))
                _cand = lo + (hi - lo) * torch.rand(GOOD_POOL, d, dtype=torch.float64)
                x_end = _cand[f_hf(_cand).reshape(-1).argmax()]
                xs1 = lo + (hi - lo) * torch.rand(d, dtype=torch.float64)
                _p5 = _interp(xs1, x_end, T)
                c5.append(lb0 - math.log(replay(               # C5 DIVERSE-GOOD
                    ko, pool, _p5, fid_of(_p5, ko, pool))))
                ridx = torch.randint(0, POOL_N, (T,))
                rell = [1 if torch.rand(1).item() < 0.25 else 0 for _ in range(T)]
                c3.append(lb0 - math.log(replay(ko, pool, pool[ridx], rell)))  # C3 random path
            rec = dict(seed=int(seed), cut=int(cut), N=a.N)
            for nm, v in (("C1_closed", c1), ("C2_open_own", c2), ("C3_open_rand", c3),
                          ("C4_oracle", c4), ("C5_diverse_good", c5)):
                v = np.array(v)
                rec[nm] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                               max=float(v.max()), p90=float(np.percentile(v, 90)))
            recs.append(rec)
            print(f"  s{seed} cut{cut:3d}  "
                  + "  ".join(f"{n.split('_')[0]}: mean {rec[n]['mean']:+.3f} sd {rec[n]['sd']:.3f} "
                              f"max {rec[n]['max']:+.3f}"
                              for n in ("C1_closed", "C2_open_own", "C3_open_rand",
                                        "C4_oracle", "C5_diverse_good")), flush=True)

    out = f"{REPO}/experiments/h156-target-is-a-max/results/tail_e{a.models}r{a.rep}_{a.bench}.json"
    json.dump(recs, open(out, "w")); print(f"\nwrote {out}  ({len(recs)} states)")


if __name__ == "__main__":
    main()
