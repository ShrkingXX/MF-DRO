import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
import json, sys, itertools, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,"tools"); sys.path.insert(0,".")
from perdim import shares
from benchmarks import get_benchmark

CK="experiments/h83-main-comparison/results/ckpt"
BENCH=["Borehole_8D","Hartmann_6D","Currin_2D","Ackley_10D"]; METH=["MF-DRO","MF-MES"]; SEEDS=[42,43,44,45,46]
NFLOOR=15
def unit(b):
    hf=get_benchmark(f"{b}_HF")
    lo=np.asarray(hf["domain_min"],float); hi=np.asarray(hf["domain_max"],float); return lo,hi-lo
def load(b,m,s,lo,w):
    d=json.load(open(f"{CK}/{b}__{m}__seed{s}.json"))
    X=np.array([q["x"] for q in d["queries"] if q["fid"]==1 and not q.get("is_init",False)],float)
    return (X-lo)/w if len(X) else X
def prof(Z,robust):
    s=np.median(np.abs(Z-np.median(Z,axis=0)),axis=0) if robust else Z.std(axis=0,ddof=1)
    return (s/s.sum() if s.sum()>0 else np.full_like(s,np.nan)), s.sum()

res={}
for b in BENCH:
    lo,w=unit(b); sh=np.asarray(shares(b),float); res[b]={}
    for robust in (False,True):
        st="mad" if robust else "sd"
        for m in METH:
            for s in SEEDS:
                Z=load(b,m,s,lo,w)
                if len(Z)<3: res[b][(st,m,s)]=(None,len(Z),None,None); continue
                p,tot=prof(Z,robust); r=spearmanr(p,sh).statistic
                res[b][(st,m,s)]=(None if not np.isfinite(r) else float(r), len(Z), p, float(tot))
DEG={"Currin_2D":"DEGENERATE d=2","Ackley_10D":"DEGENERATE S1~uniform"}
for st in ("sd","mad"):
    print(f"\n===== unit-cube dispersion profile, {st.upper()} (gate: |mean|/sd >= 1.0) =====")
    print(f"{'bench':12s} {'rho DRO':>16s} {'rho MES':>16s} {'paired':>8s} {'|mean|/sd':>9s} {'excl':>4s}  note")
    for b in BENCH:
        pr=[]; ex=0
        for s in SEEDS:
            a=res[b][(st,"MF-DRO",s)]; c=res[b][(st,"MF-MES",s)]
            if a[0] is None or c[0] is None: ex+=1; continue
            if a[1]<NFLOOR or c[1]<NFLOOR: ex+=1; continue
            pr.append((a[0],c[0]))
        if len(pr)<2:
            print(f"{b:12s} {'insufficient':>16s} (excluded {ex}/5 by n<{NFLOOR} floor)"); continue
        D=np.array([x for x,_ in pr]); M=np.array([y for _,y in pr]); df=M-D
        eff=abs(df.mean())/df.std(ddof=1) if df.std(ddof=1)>0 else float("nan")
        note=DEG.get(b,"PRIMARY" if b=="Borehole_8D" else "secondary")
        print(f"{b:12s} {np.median(D):>7.3f}(sd{D.std(ddof=1):.2f}) {np.median(M):>7.3f}(sd{M.std(ddof=1):.2f})"
              f" {df.mean():>+8.3f} {eff:>9.2f} {ex:>4d}  {note} n={len(pr)}")
print("\n===== shape vs scale (unit cube, SD) =====")
print(f"{'bench':12s} {'L1 btw':>7s} {'L1 within':>9s} {'ratio':>6s} {'total-disp DRO/MES':>19s}")
for b in ["Borehole_8D","Hartmann_6D","Ackley_10D"]:
    ok=[s for s in SEEDS if res[b][("sd","MF-DRO",s)][1]>=NFLOOR and res[b][("sd","MF-MES",s)][1]>=NFLOOR]
    if len(ok)<2: print(f"{b:12s} insufficient after n floor ({len(ok)} seeds)"); continue
    P={m:{s:res[b][("sd",m,s)] for s in ok} for m in METH}
    btw=np.mean([np.abs(P["MF-DRO"][s][2]-P["MF-MES"][s][2]).sum() for s in ok])
    wi=np.mean([np.abs(P[m][a][2]-P[m][c][2]).sum() for m in METH for a,c in itertools.combinations(ok,2)])
    sc=np.mean([P["MF-DRO"][s][3]/P["MF-MES"][s][3] for s in ok])
    print(f"{b:12s} {btw:>7.3f} {wi:>9.3f} {btw/wi:>6.2f} {sc:>19.2f}  (seeds={len(ok)})")
