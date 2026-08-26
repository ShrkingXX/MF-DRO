"""H68b: does the early policy/acquisition mismatch generalise beyond Borehole?

Self-contained readout: x_dro's percentile under MF-DRO's OWN MES acquisition.
No baseline needed -- the question is whether the policy agrees with its teacher.
Locked prediction in protocol.md.
"""
import os,sys,json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes

BENCHES=("Currin_2D","Hartmann_6D","Borehole_8D"); SEEDS=(44,46,48); KS=(2,5,10); NPOOL=600
REL={"Currin_2D":0.0,"Hartmann_6D":14.7,"Borehole_8D":23.7}
RES=os.path.join(REPO,"experiments/h57-baseline-comparison/results")

def opt_hf(q): return [e for e in q if e["fid"]==1 and not e.get("is_init")]

out=[]
for bench in BENCHES:
    hf=get_benchmark(f"{bench}_HF"); lf=get_benchmark(f"{bench}_LF")
    lo=torch.tensor(hf["domain_min"],dtype=torch.float64); hi=torch.tensor(hf["domain_max"],dtype=torch.float64)
    d=len(lo); bnds=torch.stack([lo,hi]); c_H=float(hf["cost"]); c_L=float(lf["cost"])
    for seed in SEEDS:
        qd=json.load(open(f"{RES}/{bench}__MF-DRO__seed{seed}.json"))["queries"]
        hd=opt_hf(qd)
        for k in KS:
            if len(hd)<k+1:
                out.append(dict(bench=bench,seed=seed,k=k,pct=None,note=f"only {len(hd)} HF opt queries"))
                continue
            cut=qd.index(hd[k-1])
            hist=[e for e in qd if e.get("is_init")]+[e for e in qd[:cut+1] if not e.get("is_init")]
            Xl=torch.tensor([e["x"] for e in hist if e["fid"]==0],dtype=torch.float64)
            Yl=torch.tensor([e["y"] for e in hist if e["fid"]==0],dtype=torch.float64)
            Xh=torch.tensor([e["x"] for e in hist if e["fid"]==1],dtype=torch.float64)
            Yh=torch.tensor([e["y"] for e in hist if e["fid"]==1],dtype=torch.float64)
            if len(Xl)<2 or len(Xh)<2:
                out.append(dict(bench=bench,seed=seed,k=k,pct=None,note="too few points")); continue
            torch.manual_seed(seed); np.random.seed(seed)
            ko=KennedyOHaganGP(d=d,dkl_threshold=9999); ko.bounds=bnds
            ko.fit(Xl,Yl,Xh,Yh,bnds)
            x=torch.tensor(hd[k]["x"],dtype=torch.float64)
            pool=lo+(hi-lo)*torch.rand(NPOOL,d,dtype=torch.float64)
            _,_,sc=compute_joint_mf_mes(ko,torch.cat([x[None,:],pool],0),c_H,c_L,K=10)
            s=sc[:,1].detach().cpu().numpy()
            pct=float((s[1:]<s[0]).mean()*100)
            out.append(dict(bench=bench,seed=seed,k=k,pct=pct,note=""))
            print(f"  {bench:<13} seed {seed} k={k:>2}  x_dro pct={pct:6.1f}",flush=True)

json.dump(out,open(os.path.join(os.path.dirname(__file__),"..","results","h68b.json"),"w"),indent=1)
print("\n  H68b -- x_dro percentile under MF-DRO's OWN acquisition (600-pt pool)\n")
print(f"  {'benchmark':<13}{'rel regret':>11}{'k=2':>9}{'k=5':>9}{'k=10':>9}{'mean':>9}{'missing':>9}")
for bench in BENCHES:
    r=[x for x in out if x["bench"]==bench]
    cells=[]; 
    for k in KS:
        v=[x["pct"] for x in r if x["k"]==k and x["pct"] is not None]
        cells.append(np.mean(v) if v else float('nan'))
    allv=[x["pct"] for x in r if x["pct"] is not None]
    miss=sum(1 for x in r if x["pct"] is None)
    print(f"  {bench:<13}{REL[bench]:>10.1f}%"+"".join(f"{c:>9.1f}" if c==c else f"{'--':>9}" for c in cells)
          +f"{np.mean(allv) if allv else float('nan'):>9.1f}{miss:>6}/{len(r)}")
print("\n  MISSING cells (reported, not dropped):")
for x in out:
    if x["pct"] is None: print(f"    {x['bench']:<13} seed {x['seed']} k={x['k']:<3} {x['note']}")
