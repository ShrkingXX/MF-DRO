"""H98: is centring the mediator between ROI tightness and regret? Zero compute."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Borehole_8D"; SEEDS=(42,43,44,45,46); SENS=[0,3,5,6]
H84=os.path.join(REPO,"experiments","h84-roi-strategy","results")
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
hf=get_benchmark(f"{B}_HF")
LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float)
XS=(np.array([0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0])-LO)/(HI-LO)
ARMS=[("ROI-OFF",1.0,0.00),("ROI-ANN",0.4934,-1.31),("ROI-FIX2",0.2141,-4.81),("ROI-Q10",0.0999,-4.22)]

def path(arm,s):
    p=os.path.join(H84,f"{B}__{arm}__seed{s}.json")
    if os.path.exists(p): return p
    if arm=="ROI-OFF":
        q=os.path.join(H83,f"{B}__MF-DRO__seed{s}.json")
        if os.path.exists(q): return q
    return None

def stats(arm):
    off=[];gap=[];rch=[];n=0
    for s in SEEDS:
        p=path(arm,s)
        if not p: continue
        qs=json.load(open(p))["queries"]
        post=[e for e in qs if not e.get("is_init") and e["fid"]]
        if not post: continue
        X=(np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO); n+=1
        off.append(np.mean([abs(X[:,d].mean()-0.5) for d in SENS]))
        gap.append(np.mean([abs(X[:,d].mean()-XS[d])/max(X[:,d].std(),1e-9) for d in SENS]))
        rch.append(np.mean([np.mean(np.abs(X[:,d]-XS[d])<=0.05) for d in SENS]))
    return n,np.mean(off),np.mean(gap),np.mean(rch)

print("H98 -- centring as mediator. Borehole seeds 42-46, sensitive dims 0/3/5/6.\n")
print(f"  {'arm':<10}{'accept':>8}{'regret':>9}{'n':>4}{'OFFSET':>9}{'GAPSD':>8}{'REACH':>8}")
rows=[]
for arm,acc,reg in ARMS:
    n,o,g,r=stats(arm)
    rows.append((arm,acc,reg,n,o,g,r))
    print(f"  {arm:<10}{acc:>8.4f}{reg:>9.2f}{n:>4}{o:>9.3f}{g:>8.2f}{r:>8.3f}")

acc=np.array([r[1] for r in rows]); reg=np.array([r[2] for r in rows])
gap=np.array([r[5] for r in rows]); rch=np.array([r[6] for r in rows]); off=np.array([r[4] for r in rows])
names=[r[0] for r in rows]
def rank(v,asc=True):
    o=np.argsort(v if asc else -v); return [names[i] for i in o]
print("\n  RANKINGS (best first)")
print(f"    by regret gain (most negative first): {rank(reg,asc=True)}")
print(f"    by GAPSD       (smallest first)     : {rank(gap,asc=True)}")
print(f"    by REACH       (largest first)      : {rank(rch,asc=False)}")
print(f"    by OFFSET      (largest first)      : {rank(off,asc=False)}")
print(f"    by tightness   (lowest accept first): {rank(acc,asc=True)}")
print("\n  REGISTERED BARS")
mono = list(rank(gap,asc=True))==list(rank(acc,asc=True))
print(f"    T1 (centring NOT monotone in tightness): {'MET' if not mono else 'FAILED -- it IS monotone'}")
t2 = list(rank(gap,asc=True))==list(rank(reg,asc=True))
print(f"    T2 PRIMARY (GAPSD ranking == regret ranking, incl. FIX2>Q10): "
      f"{'MET' if t2 else 'FAILED'}")
t3 = list(rank(rch,asc=False))==list(rank(reg,asc=True))
print(f"    T3 (REACH ranking == regret ranking): {'MET' if t3 else 'FAILED'}")
spread=gap.max()-gap.min()
print(f"    T4 falsifier (GAPSD flat while regret varies 4.8 pts): "
      f"GAPSD spread {spread:.2f} -> {'TRIGGERED (flat)' if spread<0.15 else 'not triggered'}")
print("\n  n=4 arms, ordering only. No correlation, no p-value. 1/24 under a null.")
if not t2:
    print("\n  *** T2 FAILED -> centring does not reproduce the dose. h96/h97 explain")
    print("      the ROI-on/ROI-off contrast but NOT how much ROI you want. Label")
    print("      them a description of one comparison, not a mechanism for tuning. ***")
