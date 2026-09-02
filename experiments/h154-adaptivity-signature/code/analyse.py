"""h154 M1/M2 -- adaptivity signature in the DT's OWN real queries."""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark

BENCH = "Borehole_8D"
SEEDS = [42, 43, 44, 45, 46]
ARMS = {
    "control MES (closed-loop)": "experiments/h83-main-comparison/results/{b}__MF-DRO__seed{s}.json",
    "ORACLE (frozen)":           "experiments/h145-oracle-expert-ceiling/results/{b}__ORACLE-EXPERT__seed{s}.json",
    "DIVERSE-GOOD (frozen)":     "experiments/h146-why-oracle-hurts/results/{b}__DIVERSE-GOOD__seed{s}.json",
    "RANDOM-POOL (non-adaptive)":"experiments/h149-forced-vs-teacher-quality/results/{b}__RANDOM-POOL__seed{s}.json",
}

hf = get_benchmark(f"{BENCH}_HF")
lo = np.array(hf["domain_min"], float); hi = np.array(hf["domain_max"], float)


def measures(path):
    r = json.load(open(path))
    X = (np.asarray(r["x_t_trace"], float) - lo) / (hi - lo)   # -> [0,1]^d
    y = np.asarray(r["y_t_trace"], float)
    f = np.asarray(r["fidelity_trace"], int)
    if len(X) < 12:
        return None
    step = np.linalg.norm(X[1:] - X[:-1], axis=1)              # ||x_{t+1}-x_t||
    # M1: standardise y WITHIN fidelity (HF and LF live on different scales),
    # then correlate with the step that FOLLOWS the observation.
    ys = np.full(len(y), np.nan)
    for fid in (0, 1):
        m = f == fid
        if m.sum() > 2 and np.std(y[m]) > 1e-12:
            ys[m] = (y[m] - y[m].mean()) / y[m].std()
    ok = ~np.isnan(ys[:-1])
    m1 = np.corrcoef(ys[:-1][ok], step[ok])[0, 1] if ok.sum() > 5 else np.nan
    # M2: lag-1 autocorrelation of the query sequence, averaged over dimensions.
    ac = [np.corrcoef(X[:-1, j], X[1:, j])[0, 1]
          for j in range(X.shape[1]) if np.std(X[:, j]) > 1e-12]
    return m1, float(np.mean(ac)), float(step.mean())


print(f"{BENCH}, seeds {SEEDS}, post-init real queries only\n")
print(f"  {'arm':30s} {'M1 resp-to-outcome':>20s} {'M2 lag-1 autocorr':>19s} {'mean step':>10s}")
out = {}
for name, pat in ARMS.items():
    rows = []
    for s in SEEDS:
        p = os.path.join(REPO, pat.format(b=BENCH, s=s))
        if not os.path.exists(p):
            continue
        v = measures(p)
        if v: rows.append(v)
    if not rows:
        print(f"  {name:30s} {'NO DATA':>20s}"); continue
    a = np.array(rows, float)
    out[name] = dict(n=len(rows), m1=float(np.nanmean(a[:, 0])), m1_sd=float(np.nanstd(a[:, 0], ddof=1)),
                     m2=float(np.nanmean(a[:, 1])), m2_sd=float(np.nanstd(a[:, 1], ddof=1)),
                     step=float(np.nanmean(a[:, 2])))
    o = out[name]; o["m1_per_seed"] = [round(v, 4) for v in a[:, 0]]
    print(f"  {name:30s} {o['m1']:+9.4f}+-{o['m1_sd']:.4f}  {o['m2']:+9.4f}+-{o['m2_sd']:.4f}  "
          f"{o['step']:9.4f}   (n={o['n']})")

json.dump(out, open(os.path.join(REPO, "experiments/h154-adaptivity-signature/results/m1m2.json"), "w"), indent=1)
# At n=5 the honest statement is a separation, not a p-value: does EVERY
# control seed sit below EVERY frozen seed?
import itertools
_c = np.array(out["control MES (closed-loop)"]["m1_per_seed"])
_f = np.concatenate([np.array(v["m1_per_seed"]) for k, v in out.items()
                     if k != "control MES (closed-loop)"])
print("\nper-seed M1:")
for k, v in out.items():
    print(f"  {k:30s} {v['m1_per_seed']}")
print(f"\n  control max {_c.max():+.4f}   frozen min {_f.min():+.4f}   "
      f"gap {_f.min()-_c.max():+.4f}")
print(f"  every control seed below every frozen run: {bool(_c.max() < _f.min())}"
      f"   ({len(_c)} control vs {len(_f)} frozen runs)")
print("\nPREDICTED if adaptivity hypothesis holds:")
print("  M1: control clearly NEGATIVE, frozen arms ~0")
print("  M2: control LOWEST, frozen arms higher and similar")
