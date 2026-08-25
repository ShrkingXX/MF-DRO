"""H31 analysis. Tests the expectation locked in H32 BEFORE H31 landed:
near-parity on regret, since the location distillation is faithful. A clear
teacher win would implicate the fidelity channel measured in H33."""
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon
HERE=os.path.dirname(os.path.abspath(__file__))
H31=os.path.join(HERE,"..","results")
H17=os.path.join(HERE,"..","..","h17-joint-mes-frozen-eval","results")
H1 =os.path.join(HERE,"..","..","h1-leak-fix-validation","results")
S=list(range(42,52))
def rec(p):
    d=json.load(open(p))
    c=d.get("hf_regret_curve") or d.get("regret_curve")
    fid=d.get("fidelity_trace") or []
    return dict(regret=float(c[-1]), iters=len(c),
                n_hf=sum(1 for e in fid if e==1), n_lf=sum(1 for e in fid if e==0),
                cost=(d.get("cost_curve") or [0])[-1])
def load(dirp, meth):
    out={}
    for s in S:
        f=os.path.join(dirp,f"{meth}__seed{s}.json")
        if os.path.exists(f): out[s]=rec(f)
    return out
T=load(H31,"MF-DRO"); M=load(H17,"MF-DRO"); G=load(H1,"MF-MI-Greedy")
print(f"teacher-only complete: {len(T)}/10")
if len(T)<10:
    print("NOT the final analysis -- runs outstanding."); raise SystemExit
print(f"\n{'seed':>5} {'teacher':>9} {'MF-DRO':>9} {'diff':>9} | "
      f"{'t:nHF':>6} {'m:nHF':>6} {'t:iters':>8} {'m:iters':>8}")
for s in S:
    d=T[s]['regret']-M[s]['regret']
    print(f"{s:>5} {T[s]['regret']:>9.4f} {M[s]['regret']:>9.4f} {d:>+9.4f} | "
          f"{T[s]['n_hf']:>6} {M[s]['n_hf']:>6} {T[s]['iters']:>8} {M[s]['iters']:>8}")
t=np.array([T[s]['regret'] for s in S]); m=np.array([M[s]['regret'] for s in S])
g=np.array([G[s]['regret'] for s in S])
se=lambda a:a.std(ddof=1)/np.sqrt(len(a))
print(f"\n  MF-MES teacher, no DT : {t.mean():.4f} +/- {se(t):.4f}   sd {t.std(ddof=1):.4f}")
print(f"  MF-DRO / joint MES    : {m.mean():.4f} +/- {se(m):.4f}   sd {m.std(ddof=1):.4f}")
print(f"  MF-MI-Greedy          : {g.mean():.4f} +/- {se(g):.4f}   sd {g.std(ddof=1):.4f}")
d=t-m
print(f"\n  paired teacher - MF-DRO: {d.mean():+.4f}  teacher better on {(d<0).sum()}/10"
      f"  Wilcoxon p={wilcoxon(t,m).pvalue:.4f}")
th=np.mean([T[s]['n_hf'] for s in S]); mh=np.mean([M[s]['n_hf'] for s in S])
ti=np.mean([T[s]['iters'] for s in S]); mi=np.mean([M[s]['iters'] for s in S])
print(f"\n  FIDELITY MIX (H33 predicted these differ):")
print(f"    teacher mean n_HF {th:.1f} over {ti:.1f} iters = {th/ti:.1%} HF")
print(f"    MF-DRO  mean n_HF {mh:.1f} over {mi:.1f} iters = {mh/mi:.1%} HF")
print("\n"+"="*70)
near = abs(d.mean())<0.05 or wilcoxon(t,m).pvalue>=0.05
print(f"H32's LOCKED EXPECTATION (near-parity): "
      f"{'CONFIRMED' if near else 'REFUTED'}  (diff {d.mean():+.4f}, p={wilcoxon(t,m).pvalue:.4f})")
if not near and d.mean()<0:
    print("  Teacher clearly better -> the transformer is a NET NEGATIVE, and H33's")
    print("  fidelity gap is the leading mechanism.")
elif not near:
    print("  Teacher clearly WORSE -> the fixed-rule account is incomplete; the DT")
    print("  contributes something the linear analysis missed. Sections need revisiting.")
mm,ss=t.mean(),se(t)
print(f"FROZEN SUCCESS TEST (teacher alone): mean+SE {mm+ss:.4f} < 0.3825 -> "
      f"{'PASS' if mm+ss<0.3825 else 'FAIL'}")
print("="*70)
