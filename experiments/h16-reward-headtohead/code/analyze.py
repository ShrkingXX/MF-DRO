import os, json, numpy as np
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "..", "results", "h16.json")))
by = {(r["seed"], r["reward"]): r for r in res}
seeds = sorted({r["seed"] for r in res})
AX = [("f_x0", "M3 f_hf(x_0)  [ORIGINAL axis]"),
      ("best_hf", "M1 best HF pt [PRIMARY]"),
      ("best_all", "M1b best any  [robustness]")]

print(f"seeds completed: {len(seeds)}  ({seeds})\n")
summary = {}
for key, lab in AX:
    imp, mes, ok = [], [], []
    for s in seeds:
        a, b = by.get((s, "improvement")), by.get((s, "mes_entropy"))
        if not a or not b or a.get(key) is None or b.get(key) is None:
            continue
        imp.append(a[key]["mean"]); mes.append(b[key]["mean"]); ok.append(s)
    imp, mes = np.array(imp), np.array(mes)
    d = mes - imp
    try:
        p = wilcoxon(mes, imp).pvalue
    except Exception:
        p = float("nan")
    summary[key] = dict(n=len(ok), imp=imp.mean(), mes=mes.mean(),
                        diff=d.mean(), p=float(p),
                        mes_better=int((d > 0).sum()))
    print(f"{lab}   (n={len(ok)} seeds)")
    print(f"   improvement  {imp.mean():+.4f}  (SE {imp.std(ddof=1)/np.sqrt(len(imp)):.4f})")
    print(f"   mes_entropy  {mes.mean():+.4f}  (SE {mes.std(ddof=1)/np.sqrt(len(mes)):.4f})")
    print(f"   paired diff  {d.mean():+.4f}   mes better on {(d>0).sum()}/{len(d)} seeds"
          f"   Wilcoxon p={p:.4f}")
    print(f"   per-seed diff: {np.round(d,3).tolist()}\n")

print("=" * 74)
m1 = summary["best_hf"]; m3 = summary["f_x0"]
p1 = (m1["diff"] > 0) and (m1["p"] < 0.05)
print(f"PRED 1 PRIMARY (mes_entropy > improvement on M1, Wilcoxon p<0.05): "
      f"{'PASS' if p1 else 'FAIL'}  (diff {m1['diff']:+.4f}, p={m1['p']:.4f})")
repro_old = m3["imp"] > m3["mes"]
print(f"PRED 2 REPRODUCTION (original ordering improvement>mes_entropy on M3 "
      f"does NOT reappear): {'FAIL -- it DID reappear; H15 was a one-seed fluke and my retraction was premature' if repro_old else 'PASS -- non-reproduction confirmed at n=10'}")
allnull = all(summary[k]["p"] > 0.05 for k, _ in AX)
if allnull:
    print("\nPRED 3 NULL RULE FIRES: indistinguishable on all three axes.")
    print("  Pre-committed decision: fall back to SIGNAL HEALTH, where")
    print("  mes_entropy dominates (0.0% vs 63.0% dead rtg[0]; 100% vs 0.0% LF")
    print("  credit; CV 0.66 vs 1.96). Reward -> mes_entropy on that basis,")
    print("  explicitly NOT on a teacher-quality difference.")
print("=" * 74)
json.dump(summary, open(os.path.join(HERE, "..", "results", "summary.json"), "w"),
          indent=2, default=float)
