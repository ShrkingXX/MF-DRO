"""h170 -- is the DT's emitted query at the MEAN of its teacher's tau=0 action?

Reconstructs the tau=0 teacher action distribution offline at real iterations,
takes its mean, and compares the DT's ACTUAL query at that iteration against
three targets: the tau=0 mean, the box centre, and a random pool point.
"""
import os, sys, json, argparse
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes

POOL_N, N_DRAW = 200, 120          # pools/members drawn per state

ARMS = {
 "control (works)":     ("experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json", "mes"),
 "h165 UCB-LOC (works)":("experiments/h165-hartmann-ucbloc/results/{b}__UCB-LOC__seed{s}.json", "ucb"),
 "RANDOM-POOL (fails)": ("experiments/h149-forced-vs-teacher-quality/results/{b}__RANDOM-POOL__seed{s}.json", "random"),
 "ORACLE (fails)":      ("experiments/h145-oracle-expert-ceiling/results/{b}__ORACLE-EXPERT__seed{s}.json", "oracle"),
}
XSTAR_RAW = np.array([0.2017, 0.1500, 0.4769, 0.2753, 0.3116, 0.6573])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cuts", type=int, default=8)
    a = ap.parse_args()
    hf, lf = get_benchmark("Hartmann_6D_HF"), get_benchmark("Hartmann_6D_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
    lo, hi = bounds[0], bounds[1]
    CENTRE = np.full(d, 0.5)
    out = {}
    for name, (pat, rule) in ARMS.items():
        rows = []
        for seed in range(42, 47):
            p = os.path.join(REPO, pat.format(b="Hartmann_6D", s=seed))
            if not os.path.exists(p): continue
            R = json.load(open(p)); tr = R["queries"]
            X = (np.asarray(R["x_t_trace"], float) - lo.numpy()) / (hi - lo).numpy()
            n_init = sum(1 for e in tr if e.get("is_init"))
            cuts = np.linspace(n_init, min(len(tr), n_init + len(X)), a.cuts + 1)[:-1].astype(int)
            for ci, cut in enumerate(cuts):
                qi = int(cut - n_init)
                if qi < 0 or qi >= len(X): continue
                torch.manual_seed(5000 + seed + int(cut)); np.random.seed(5000 + seed + int(cut))
                t = lambda z: torch.tensor(z, dtype=torch.float64)
                Xh = [e["x"] for e in tr[:cut] if e["fid"] == 1]; Yh = [e["y"] for e in tr[:cut] if e["fid"] == 1]
                Xl = [e["x"] for e in tr[:cut] if e["fid"] == 0]; Yl = [e["y"] for e in tr[:cut] if e["fid"] == 0]
                if len(Xh) < 3 or len(Xl) < 3: continue
                ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
                ko.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds)
                acts = []
                for _ in range(N_DRAW):
                    pool = lo + (hi - lo) * torch.rand(POOL_N, d, dtype=torch.float64)
                    if rule == "mes":
                        x, _e, _ = compute_joint_mf_mes(ko, pool, c_H, c_L)
                    elif rule == "ucb":
                        with torch.no_grad(): mu, var = ko.hf_posterior(pool)
                        x = pool[(mu + 2.0 * var.clamp_min(0).sqrt()).argmax()]
                    elif rule == "random":
                        x = pool[torch.randint(0, POOL_N, (1,)).item()]
                    else:                                   # oracle: tau=0 is x_start ~ Uniform
                        x = lo + (hi - lo) * torch.rand(d, dtype=torch.float64)
                    acts.append(((x - lo) / (hi - lo)).numpy())
                mu0 = np.mean(acts, axis=0)                 # the tau=0 action MEAN
                q = X[qi]                                   # the DT's ACTUAL query
                rnd = np.random.rand(d)
                se = float(np.mean(np.linalg.norm(np.array(acts)-mu0,axis=1))/np.sqrt(len(acts)))
                rows.append((float(np.linalg.norm(q - mu0)),
                             float(np.linalg.norm(q - CENTRE)),
                             float(np.linalg.norm(q - rnd)),
                             float(np.linalg.norm(mu0 - CENTRE)), se))
        if rows:
            A = np.array(rows)
            out[name] = dict(n=len(A), d_tau0=float(A[:, 0].mean()), d_centre=float(A[:, 1].mean()),
                             d_random=float(A[:, 2].mean()), tau0_to_centre=float(A[:, 3].mean()), se=float(A[:, 4].mean()))
    print(f"\n  Hartmann, {ARMS and ''}reconstructed tau=0 teacher action mean vs the DT's actual query\n")
    print(f"  {'arm':22s} {'d(q, tau0 mean)':>16s} {'d(q, centre)':>13s} {'d(q, random)':>13s} {'tau0->centre':>13s} {'est. SE':>9s} {'n':>4s}")
    for k, v in out.items():
        print(f"  {k:22s} {v['d_tau0']:16.4f} {v['d_centre']:13.4f} {v['d_random']:13.4f} "
              f"{v['tau0_to_centre']:13.4f} {v['se']:9.4f} {v['n']:4d}")
    print("\n  P1 (discriminating, WORKING arms): d(q, tau0 mean) < d(q, centre)?")
    for k in [k for k in out if "works" in k]:
        v = out[k]; print(f"    {k:22s} {v['d_tau0']:.4f} vs {v['d_centre']:.4f}  -> "
                          f"{'HOLDS' if v['d_tau0'] < v['d_centre'] else 'FAILS'}")
    print("  P2 (calibration, FAILING arms): reconstructed tau0 mean lands on the centre?")
    for k in [k for k in out if "fails" in k]:
        v = out[k]; print(f"    {k:22s} tau0 mean is {v['tau0_to_centre']:.4f} from centre  -> "
                          f"{'CALIBRATED' if v['tau0_to_centre'] < 0.15 else 'NOT calibrated'}")
    print("  P3 (all arms): d(q, tau0 mean) < d(q, random point)?")
    for k, v in out.items():
        print(f"    {k:22s} {v['d_tau0']:.4f} vs {v['d_random']:.4f}  -> "
              f"{'HOLDS' if v['d_tau0'] < v['d_random'] else 'FAILS'}")
    json.dump(out, open(f"{REPO}/experiments/h175-tau0-hartmann/results/tau0.json", "w"), indent=1)


if __name__ == "__main__":
    main()
