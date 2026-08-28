"""H99: does weighted headroom gate where the ROI can help? Zero new compute."""
import os,sys,json,glob
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import numpy as np, torch
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); torch.set_num_threads(1)
from benchmarks import get_benchmark
XSTAR={"Borehole_8D":[0.15,100.0,95090.978,1110.0,116.0,700.0,1120.0,12045.0],
       "Hartmann_6D":[0.2017,0.15,0.4769,0.2753,0.3116,0.6573],
       "Ackley_10D":[0.5]*10,
       "Currin_2D":[0.21666666328646256,0.008707968518137932]}
# control arm (no ROI) per benchmark, and the ROI's measured regret effect
CTRL={"Borehole_8D":("experiments/h90-borehole-confirm/results","NO-ROI",(47,48,49,50,51)),
      "Hartmann_6D":("experiments/h83-main-comparison/results","MF-DRO",(42,43,44,45,46)),
      "Ackley_10D":("experiments/h83-main-comparison/results","MF-DRO",(42,43,44,45,46)),
      "Currin_2D":("experiments/h83-main-comparison/results","MF-DRO",(42,43,44,45,46))}
ROI_EFFECT={"Borehole_8D":"WORKED -3.49","Hartmann_6D":"failed (withdrawn)",
            "Ackley_10D":"negligible -0.09","Currin_2D":"saturated ~0"}

def sens(bench,LO,HI,obj,d,N=40000,B=40):
    rng=np.random.RandomState(7); X=rng.rand(N,d)
    Y=np.asarray(obj(torch.as_tensor(LO+(HI-LO)*X,dtype=torch.float64)),float).reshape(-1)
    VY=Y.var(); out=[]
    for i in range(d):
        b=np.clip((X[:,i]*B).astype(int),0,B-1)
        m=np.array([Y[b==k].mean() if (b==k).any() else Y.mean() for k in range(B)])
        c=np.array([(b==k).sum() for k in range(B)])
        out.append(np.average((m-Y.mean())**2,weights=c)/VY)
    o=np.array(out); return o/o.sum()

print("H99 -- weighted headroom vs where the ROI helped.\n")
rows=[]
for B_ in ("Borehole_8D","Hartmann_6D","Ackley_10D","Currin_2D"):
    hf=get_benchmark(f"{B_}_HF"); obj=hf["make_objective"]()
    LO=np.asarray(hf["domain_min"],float); HI=np.asarray(hf["domain_max"],float); d=len(LO)
    XS=(np.asarray(XSTAR[B_],float)-LO)/(HI-LO)
    W=sens(B_,LO,HI,obj,d)
    root,arm,seeds=CTRL[B_]; hr=np.zeros(d); n=0
    for s in seeds:
        p=os.path.join(REPO,root,f"{B_}__{arm}__seed{s}.json")
        if not os.path.exists(p): continue
        q=json.load(open(p))["queries"]
        post=[e for e in q if not e.get("is_init") and e["fid"]]
        if not post: continue
        X=(np.array([np.asarray(e["x"],float) for e in post])-LO)/(HI-LO)
        y=np.array([float(e["y"]) for e in post])
        hr += np.abs(X[np.argmax(y)]-XS); n+=1
    if n==0: print(f"  {B_}: no control runs"); continue
    hr/=n; prod=W*hr; tot=prod.sum()
    rows.append((B_,tot,W,hr,prod,n))
    print(f"  === {B_}  (n={n} control runs)   ROI: {ROI_EFFECT[B_]} ===")
    order=np.argsort(-prod)[:4]
    print(f"    {'dim':>4}{'weight':>9}{'headroom':>10}{'product':>10}")
    for i in order: print(f"    {i:>4}{W[i]:>9.3f}{hr[i]:>10.3f}{prod[i]:>10.4f}")
    print(f"    TOTAL weighted headroom = {tot:.4f}\n")

print("  === ORDERING ===")
for B_,t,_,_,_,_ in sorted(rows,key=lambda r:-r[1]):
    print(f"    {B_:<14}{t:.4f}   ROI: {ROI_EFFECT[B_]}")
print("\n  REGISTERED BARS")
d_={r[0]:r[1] for r in rows}
print(f"    P1 (Currin's TOTAL near zero): {d_.get('Currin_2D',float('nan')):.4f}"
      f" -> {'MET' if d_.get('Currin_2D',1)<0.05 else 'FAILED'}")
top=max(d_,key=d_.get)
p2 = top=="Borehole_8D"
print(f"    P2 PRIMARY (TOTAL highest where the ROI worked): highest is {top}"
      f" -> {'MET' if p2 else 'FAILED'}")
bh=[r for r in rows if r[0]=="Borehole_8D"]
if bh:
    _,_,W,hr,prod,_=bh[0]
    print(f"    P3 (Borehole dim0 contributes ~0 despite carrying {100*W[0]:.0f}%): "
          f"product {prod[0]:.4f}, headroom {hr[0]:.4f} -> "
          f"{'MET' if prod[0]<0.02*prod.sum()+1e-9 or hr[0]<0.02 else 'FAILED'}")
if not p2:
    print("\n    *** P2 FAILED -> P4 applies. Weighted headroom does NOT gate where")
    print("        the ROI helps. The four-benchmark relocation table remains a true")
    print("        pattern with no mechanism beneath it. Say that, do not smooth it. ***")
