"""H96: relocation vs concentration. Zero compute; reads h90's completed runs.
PRIMARY measure is the SENSITIVITY-WEIGHTED distance (see protocol)."""
import sys, os, json
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
B="Borehole_8D"; SEEDS=(47,48,49,50,51)
H90=os.path.join(REPO,"experiments","h90-borehole-confirm","results")
hf=get_benchmark(f"{B}_HF")
LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float)
XSTAR_RAW=np.array([0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0])
XS=(XSTAR_RAW-LO)/(HI-LO)
W=np.array([.816,.001,.001,.046,.001,.054,.080,.001])   # sensitivity shares
SENS=[0,3,5,6]

def stats(path):
    q=json.load(open(path))["queries"]
    post=[e for e in q if not e.get("is_init") and e["fid"]]
    if not post: return None
    X=(np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO)
    dw=np.sqrt(((X-XS)**2*W).sum(axis=1))
    du=np.linalg.norm(X-XS,axis=1)
    near={d: float((np.abs(X[:,d]-XS[d])<=0.05).mean()) for d in SENS}
    return dict(n=len(post), dw_mean=float(dw.mean()), dw_min=float(dw.min()),
                du_mean=float(du.mean()), du_min=float(du.min()), near=near)

if __name__=="__main__":
    rows=[]
    for s in SEEDS:
        a=os.path.join(H90,f"{B}__ROI-Q10__seed{s}.json"); c=os.path.join(H90,f"{B}__NO-ROI__seed{s}.json")
        if os.path.exists(a) and os.path.exists(c):
            sa,sc=stats(a),stats(c)
            if sa and sc: rows.append((s,sa,sc))
    print("H96 -- relocation vs concentration. Borehole seeds 47-51, h90 runs.\n")
    if len(rows)<5: print(f"  INCOMPLETE {len(rows)}/5, no verdict"); sys.exit()
    def block(key,title,primary=False):
        print(f"  {title}{'   <-- PRIMARY (verdict turns on this)' if primary else ''}")
        print(f"    {'seed':>5}{'ROI':>10}{'no-ROI':>10}{'diff':>10}")
        d=[]
        for s,sa,sc in rows:
            print(f"    {s:>5}{sa[key]:>10.4f}{sc[key]:>10.4f}{sa[key]-sc[key]:>+10.4f}"); d.append(sa[key]-sc[key])
        d=np.array(d); print(f"    paired mean {d.mean():+.4f}   ROI closer {int((d<0).sum())}/5\n")
        return d
    dw=block("dw_mean","WEIGHTED distance to x* (mean over HF queries)",primary=True)
    dwm=block("dw_min","WEIGHTED distance, BEST single query (min)")
    du=block("du_mean","UNWEIGHTED distance (the metric findings.md:3174 flags as misleading)")
    print("  NEAR-BOUNDARY FRACTION in the four SENSITIVE dims (within 0.05 of x*)")
    print(f"    {'dim':>5}{'share':>8}{'ROI':>9}{'no-ROI':>9}{'diff':>9}")
    r3=[]
    for d_ in SENS:
        a=np.mean([sa["near"][d_] for _,sa,_ in rows]); c=np.mean([sc["near"][d_] for _,_,sc in rows])
        sh={0:"81.6%",3:"4.6%",5:"5.4%",6:"8.0%"}[d_]
        print(f"    {d_:>5}{sh:>8}{a:>9.3f}{c:>9.3f}{a-c:>+9.3f}"); r3.append(a-c)
    print(f"    h83 reference -- MF-DRO 68%/1%/0%/2% vs MF-MES 99%/49%/34%/70% on dims 0/3/5/6\n")
    r1 = int((dw<0).sum())>=4
    print("  REGISTERED BARS")
    print(f"    R1 PRIMARY (weighted distance falls >=4/5): {'MET' if r1 else 'FAILED'}")
    print(f"    R2 (unweighted may diverge -- registered, not a surprise): "
          f"unweighted falls {int((du<0).sum())}/5, mean {du.mean():+.4f}")
    print(f"    R3 (near-boundary fraction rises in sensitive dims): "
          f"{'MET' if sum(1 for v in r3 if v>0)>=3 else 'FAILED'}  ({sum(1 for v in r3 if v>0)}/4 dims up)")
    print(f"    R4: dispersion ROSE (h95: +0.010, 4/5).")
    if r1:
        print("        -> R1 MET with dispersion UP = RELOCATION-WITH-SPREAD.")
        print("           'Concentrate the proposals' is the wrong prescription.")
    else:
        print("        -> R1 FAILED. The ROI's benefit is NOT explained by where the")
        print("           queries are. Remaining candidates: the fidelity mix, or the")
        print("           surrogate. OPEN QUESTION -- report it as one.")
