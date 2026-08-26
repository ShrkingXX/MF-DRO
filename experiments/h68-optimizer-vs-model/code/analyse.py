"""H68: does MF-DRO's OWN acquisition rank MI-Greedy's winning points highly?

Refit MF-DRO's KO surrogate on MF-DRO's own data through its k-th HF query, then
score under its MES acquisition: the point MF-DRO queried next, the point
MI-Greedy queried at its (k+1)-th HF query, and a 600-point Sobol pool.
Percentile of the pool is the readout. See protocol.md for locked predictions.
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

BENCH="Borehole_8D"; SEEDS=(44,46,48); KS=(10,20,40); NPOOL=600
RES=os.path.join(REPO,"experiments/h57-baseline-comparison/results")
hf=get_benchmark(f"{BENCH}_HF"); lf=get_benchmark(f"{BENCH}_LF")
lo=torch.tensor(hf["domain_min"],dtype=torch.float64); hi=torch.tensor(hf["domain_max"],dtype=torch.float64)
d=len(lo); bnds=torch.stack([lo,hi]); c_H=float(hf["cost"]); c_L=float(lf["cost"])

def trace(method,seed):
    return json.load(open(f"{RES}/{BENCH}__{method}__seed{seed}.json"))["queries"]

def hf_points(q):
    return [e for e in q if e["fid"]==1]

def opt_hf(q):
    """HF OPTIMIZATION queries (init excluded). The h57 trace stores the initial
    design as all-HF then all-LF, so slicing positionally at the k-th HF query
    lands before the LF init block and yields zero LF rows. The full init is
    therefore always included and k counts HF queries AFTER it."""
    return [e for e in q if e["fid"]==1 and not e.get("is_init")]

rows=[]
for seed in SEEDS:
    qd=trace("MF-DRO",seed); qm=trace("MF-MI-Greedy",seed)
    hd=hf_points(qd); hm=hf_points(qm)
    for k in KS:
        if len(hd)<k+1 or len(hm)<k+1: 
            rows.append((seed,k,None,None,"insufficient trace")); continue
        cut=qd.index(hd[k-1])            # index of the k-th HF OPTIMIZATION query
        hist=[e for e in qd if e.get("is_init")] + [e for e in qd[:cut+1] if not e.get("is_init")]
        Xl=torch.tensor([e["x"] for e in hist if e["fid"]==0],dtype=torch.float64)
        Yl=torch.tensor([e["y"] for e in hist if e["fid"]==0],dtype=torch.float64)
        Xh=torch.tensor([e["x"] for e in hist if e["fid"]==1],dtype=torch.float64)
        Yh=torch.tensor([e["y"] for e in hist if e["fid"]==1],dtype=torch.float64)
        if len(Xl)<2: rows.append((seed,k,None,None,"too few LF")); continue
        torch.manual_seed(seed); np.random.seed(seed)
        ko=KennedyOHaganGP(d=d,dkl_threshold=9999); ko.bounds=bnds
        ko.fit(Xl,Yl,Xh,Yh,bnds)
        x_dro=torch.tensor(hd[k]["x"],dtype=torch.float64)
        x_mig=torch.tensor(hm[k]["x"],dtype=torch.float64)
        pool=lo+(hi-lo)*torch.rand(NPOOL,d,dtype=torch.float64)
        cand=torch.cat([x_dro[None,:],x_mig[None,:],pool],dim=0)
        _,_,scores=compute_joint_mf_mes(ko,cand,c_H,c_L,K=10)
        hfsc=scores[:,1].detach().cpu().numpy()      # HF column, cost-normalised
        s_dro,s_mig,s_pool=hfsc[0],hfsc[1],hfsc[2:]
        p_dro=float((s_pool<s_dro).mean()*100); p_mig=float((s_pool<s_mig).mean()*100)
        rows.append((seed,k,p_dro,p_mig,""))
        print(f"  seed {seed} k={k:>2}  x_dro pct={p_dro:6.1f}  x_mig pct={p_mig:6.1f}",flush=True)

print("\n  H68 RESULT -- percentile of MF-DRO's own MES acquisition pool (600 Sobol)\n")
print(f"  {'seed':>5}{'k':>5}{'x_dro pct':>12}{'x_mig pct':>12}{'note':>22}")
ok=[r for r in rows if r[2] is not None]
for seed,k,pd_,pm,note in rows:
    if pd_ is None: print(f"  {seed:>5}{k:>5}{'--':>12}{'--':>12}{note:>22}")
    else: print(f"  {seed:>5}{k:>5}{pd_:>11.1f}%{pm:>11.1f}%{'':>22}")
if ok:
    mig=[r[3] for r in ok]; dro=[r[2] for r in ok]
    above=sum(1 for m in mig if m>50); out=sum(1 for r in ok if r[3]>r[2])
    print(f"\n  cells evaluated: {len(ok)}/{len(rows)}")
    print(f"  mean x_mig percentile = {np.mean(mig):.1f}   mean x_dro percentile = {np.mean(dro):.1f}")
    print(f"  PRIMARY : x_mig above 50th pct in {above}/{len(ok)} cells -> "
          f"{'OPTIMIZER failure' if above>len(ok)/2 else 'MODEL failure'}")
    print(f"  SECONDARY: x_mig outranks x_dro in {out}/{len(ok)} (locked bar: >=5 of 9)")
    band=[abs(r[2]-r[3]) for r in ok]
    print(f"  NULL check: mean |pct difference| = {np.mean(band):.1f} (NULL if <10)")
    json.dump([dict(seed=r[0],k=r[1],p_dro=r[2],p_mig=r[3],note=r[4]) for r in rows],
              open(os.path.join(os.path.dirname(__file__),"..","results","h68.json"),"w"),indent=1)
