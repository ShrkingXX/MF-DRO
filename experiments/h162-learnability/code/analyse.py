"""h162 M1/M2 -- is the DT collapsed to a near-constant for unlearnable teachers?"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark

BENCH, SEEDS = "Borehole_8D", [42, 43, 44, 45, 46]
ARMS = {
 "control MES (works)":   "experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json",
 "h155 UCB-LOC (works)":  "experiments/h155-ucb-loc/results/{b}__UCB-LOC__seed{s}.json",
 "h153 MES-FROZEN (works)":"experiments/h153-mes-frozen/results/{b}__MES-FROZEN__seed{s}.json",
 "ORACLE (fails)":        "experiments/h145-oracle-expert-ceiling/results/{b}__ORACLE-EXPERT__seed{s}.json",
 "DIVERSE-GOOD (fails)":  "experiments/h146-why-oracle-hurts/results/{b}__DIVERSE-GOOD__seed{s}.json",
 "RANDOM-POOL (fails)":   "experiments/h149-forced-vs-teacher-quality/results/{b}__RANDOM-POOL__seed{s}.json",
}
hf = get_benchmark(f"{BENCH}_HF")
lo = np.array(hf["domain_min"], float); hi = np.array(hf["domain_max"], float)


def measures(path):
    X = (np.asarray(json.load(open(path))["x_t_trace"], float) - lo) / (hi - lo)
    if len(X) < 12:
        return None
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)   # pairwise
    iu = np.triu_indices(len(X), 1)
    m1 = float(D[iu].mean())                                     # dispersion
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1).mean()                                    # nearest-neighbour
    m2 = float(nn / m1) if m1 > 0 else np.nan                    # support / spread
    return m1, m2, len(X)


print(f"\n{BENCH}, seeds {SEEDS}, post-init real queries, domain-normalised\n")
print(f"  {'arm':26s} {'M1 dispersion':>16s} {'M2 nn/disp':>14s} {'n queries':>10s}")
out = {}
for name, pat in ARMS.items():
    rows = [measures(os.path.join(REPO, pat.format(b=BENCH, s=s)))
            for s in SEEDS if os.path.exists(os.path.join(REPO, pat.format(b=BENCH, s=s)))]
    rows = [r for r in rows if r]
    if not rows:
        print(f"  {name:26s} {'NO DATA':>16s}"); continue
    a = np.array(rows, float)
    out[name] = dict(m1=float(a[:, 0].mean()), m1_sd=float(a[:, 0].std(ddof=1)),
                     m2=float(a[:, 1].mean()), n=int(a[:, 2].mean()),
                     m1_per_seed=[round(v, 4) for v in a[:, 0]])
    o = out[name]
    print(f"  {name:26s} {o['m1']:9.4f}±{o['m1_sd']:.4f} {o['m2']:14.4f} {o['n']:10d}")

W = [k for k in out if "works" in k]; F = [k for k in out if "fails" in k]
if W and F:
    wv = np.array([v for k in W for v in out[k]["m1_per_seed"]])
    fv = np.array([v for k in F for v in out[k]["m1_per_seed"]])
    print(f"\n  WORKING arms  M1 {wv.mean():.4f}  (n={len(wv)} runs, min {wv.min():.4f})")
    print(f"  FAILING arms  M1 {fv.mean():.4f}  (n={len(fv)} runs, max {fv.max():.4f})")
    print(f"  prediction was: working HIGH, failing LOW (collapsed)")
    sep = wv.min() > fv.max()
    print(f"  complete separation in predicted direction: {sep}")
    print(f"  direction: {'AS PREDICTED' if wv.mean() > fv.mean() else 'OPPOSITE (R3)'}")
json.dump(out, open(os.path.join(REPO, "experiments/h162-learnability/results/m1m2.json"), "w"), indent=1)
