"""H97: parameterisation vs signal strength. Zero compute; h90 runs."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Borehole_8D"; SEEDS=(47,48,49,50,51); UNI=np.sqrt(1/12)
H90=os.path.join(REPO,"experiments","h90-borehole-confirm","results")
hf=get_benchmark(f"{B}_HF")
LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float)
XS=(np.array([0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0])-LO)/(HI-LO)
SHARE={0:81.6,1:0.1,2:0.1,3:4.6,4:0.1,5:5.4,6:8.0,7:0.1}

def coords(arm):
    out=[]
    for s in SEEDS:
        p=os.path.join(H90,f"{B}__{arm}__seed{s}.json")
        if not os.path.exists(p): continue
        q=json.load(open(p))["queries"]
        post=[e for e in q if not e.get("is_init") and e["fid"]]
        if post: out.append((np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO))
    return out

for arm in ("NO-ROI","ROI-Q10"):
    runs=coords(arm)
    if len(runs)<5: print(f"{arm}: only {len(runs)}/5"); continue
    print(f"\n=== {arm}  (n={len(runs)} runs) ===")
    print(f"  {'dim':>4}{'share%':>8}{'x*':>7}{'MEAN':>8}{'|M-.5|':>8}{'SD':>7}{'SHRINK':>8}  centre moved?")
    for d in range(8):
        m=np.mean([r[:,d].mean() for r in runs]); sd=np.mean([r[:,d].std() for r in runs])
        off=abs(m-0.5); shr=sd/UNI
        flag="YES" if off>0.25 else ("partial" if off>0.12 else "no  (sits at centre)")
        print(f"  {d:>4}{SHARE[d]:>8.1f}{XS[d]:>7.2f}{m:>8.3f}{off:>8.3f}{sd:>7.3f}{shr:>8.2f}  {flag}")

print("\n=== VERDICT ===")
runs=coords("NO-ROI")
m=np.array([np.mean([r[:,d].mean() for r in runs]) for d in range(8)])
sd=np.array([np.mean([r[:,d].std() for r in runs]) for d in range(8)])
off=np.abs(m-0.5); shr=sd/UNI
SENS=[0,3,5,6]; INSENS=[1,2,4,7]
print(f"  S1 shrinkage severe everywhere: SD/uniform ranges {shr.min():.2f}-{shr.max():.2f}"
      f"  -> {'MET' if shr.max()<0.6 else 'FAILED'}")
print(f"  S2 centre offset: dim0={off[0]:.3f}   dims3/5/6={off[3]:.3f}/{off[5]:.3f}/{off[6]:.3f}")
s2 = off[0]>0.25 and max(off[3],off[5],off[6])<0.25
print(f"     -> {'MET (dim0 moved, 3/5/6 did not)' if s2 else 'FAILED'}")
print(f"  S3 shrinkage similar across dims: sd of SHRINK = {shr.std():.3f}"
      f"  -> {'MET' if shr.std()<0.15 else 'FAILED'}")
print(f"  S4 falsifier (dim0 centre also unmoved -> mechanism A): "
      f"{'TRIGGERED' if off[0]<=0.25 else 'not triggered'}")
print()
if s2:
    print("  => MECHANISM B (SIGNAL STRENGTH). The head DOES relocate its centre")
    print("     where the signal is strong (dim 0) and sits at the domain centre")
    print("     where it is weak. The fix is a sensitivity-weighted L_loc, NOT an")
    print("     architecture change. Boundary aversion is a LOSS-WEIGHTING artefact.")
else:
    print("  => MECHANISM A (PARAMETERISATION) or neither. Read the table before")
    print("     prescribing anything.")
print(f"\n  reference: sensitive dims {SENS} mean offset {off[SENS].mean():.3f};"
      f" insensitive {INSENS} mean offset {off[INSENS].mean():.3f}")
