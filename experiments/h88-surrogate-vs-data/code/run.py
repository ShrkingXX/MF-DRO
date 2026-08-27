"""H88: same surrogate, same recommender, different data."""
import sys, os, json
import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
H83=os.path.join(REPO,'experiments','h83-main-comparison','results')
SEEDS=(42,43,44,45,46); OUT=os.path.join(os.path.dirname(__file__),"..","results")

def load(b,m,s):
    r=json.load(open(f'{H83}/{b}__{m}__seed{s}.json')); q=r["queries"]
    H=[(e["x"],e["y"]) for e in q if e["fid"]==1]; L=[(e["x"],e["y"]) for e in q if e["fid"]==0]
    return H,L

def fit_and_recommend(b,H,L,pool,fH):
    lo=torch.tensor(get_benchmark(f"{b}_HF")["domain_min"]); hi=torch.tensor(get_benchmark(f"{b}_HF")["domain_max"])
    d=len(lo)
    XH=torch.tensor(np.array([x for x,_ in H])); YH=torch.tensor(np.array([y for _,y in H]))
    XL=torch.tensor(np.array([x for x,_ in L])) if L else torch.zeros(0,d)
    YL=torch.tensor(np.array([y for _,y in L])) if L else torch.zeros(0)
    # dkl_threshold=9999 MATCHES h83's runs (worker.py passes it), so DKL never
    # activates. The default of 30 would activate a deep kernel that h83 never
    # used -- a different surrogate than the one this diagnostic is explaining.
    ko=KennedyOHaganGP(d=d, dkl_threshold=9999); ko.fit(XL,YL,XH,YH,torch.stack([lo,hi]))
    with torch.no_grad(): mu,_=ko.hf_posterior(pool)
    xh=pool[int(mu.reshape(-1).argmax())]
    with torch.no_grad(): f=float(fH(xh.unsqueeze(0)).reshape(-1)[0])
    return f, float(YH.max())

if __name__=="__main__":
    res={}
    for b in ("Hartmann_6D","Borehole_8D"):
        spec=get_benchmark(f"{b}_HF"); fH=spec["make_objective"]()
        opt=abs(float(spec["known_optimal_value"])); yopt=-float(spec["known_optimal_value"])
        lo=torch.tensor(spec["domain_min"]); hi=torch.tensor(spec["domain_max"]); d=len(lo)
        pool=lo+(hi-lo)*torch.quasirandom.SobolEngine(d,scramble=True,seed=7).draw(4096).to(torch.float64)
        rows=[]
        for s in SEEDS:
            out={}
            for cond,m in (("A_MFDRO_data","MF-DRO"),("B_MFMES_data","MF-MES")):
                H,L=load(b,m,s)
                f,best=fit_and_recommend(b,H,L,pool,fH)
                out[cond]=dict(rec_regret=100*(yopt-f)/opt, own_best_regret=100*(yopt-best)/opt,
                               n_hf=len(H), n_lf=len(L))
            rows.append((s,out)); print(f"  {b} s{s} done",flush=True)
        res[b]=rows
    json.dump(res,open(os.path.join(OUT,"h88.json"),"w"),indent=1,default=float)
    print("\n=== H88: same GP, same 4096-pt recommender, different DATA ===\n")
    for b,rows in res.items():
        print(f"  {b}")
        print(f"    {'seed':>5}{'A: fit on MF-DRO':>19}{'B: fit on MF-MES':>19}"
              f"{'A own best':>12}{'B own best':>12}")
        A=[];B=[];Ab=[];Bb=[]
        for s,o in rows:
            a,bb=o["A_MFDRO_data"],o["B_MFMES_data"]
            A.append(a["rec_regret"]); B.append(bb["rec_regret"])
            Ab.append(a["own_best_regret"]); Bb.append(bb["own_best_regret"])
            print(f"    {s:>5}{a['rec_regret']:>18.2f}%{bb['rec_regret']:>18.2f}%"
                  f"{a['own_best_regret']:>11.2f}%{bb['own_best_regret']:>11.2f}%")
        A,B,Ab,Bb=map(np.array,(A,B,Ab,Bb))
        print(f"    {'mean':>5}{A.mean():>18.2f}%{B.mean():>18.2f}%{Ab.mean():>11.2f}%{Bb.mean():>11.2f}%")
        print(f"    P1 (B better than A, >=4/5): {'MET' if (B<A).sum()>=4 else 'FAILED'}  [{int((B<A).sum())}/5]")
        print(f"    P2 (A no better than MF-DRO's own best): "
              f"{'MET' if (A>=Ab-1e-9).sum()==5 else 'FAILED'}  [{int((A>=Ab-1e-9).sum())}/5]")
        print(f"    P3 (B does NOT beat MF-MES's own best): "
              f"{'MET' if (B>=Bb-1e-9).sum()==5 else 'REFUTED'}  [{int((B>=Bb-1e-9).sum())}/5]")
        print()
