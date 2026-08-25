import os, sys, json, copy
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
RES=os.path.join(os.path.dirname(__file__),"..","results")

hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(42); np.random.seed(42)
cfg=_build_mf_dro_config("h46","Hartmann_6D","reg",42,bo_iterations=20,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e12,initial_hf=6,
    initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=42; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=False
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)

# ARM U: the trained DT's OWN initial weights, snapshotted before any training.
dt_untrained=copy.deepcopy(mf.dt)

hist=[]
_orig=mf.dt.propose_mf
def _rec(state,rtg,btg,**kw):
    hist.append(dict(state=state.detach().clone(),rtg=float(rtg),btg=float(btg)))
    return _orig(state,rtg,btg,**kw)
mf.dt.propose_mf=_rec
r=mf.run(); mf.dt.propose_mf=_orig

xs=np.array(r["x_t_trace"])
span=(bounds[1]-bounds[0]).numpy()
xs_n=(xs-bounds[0].numpy())/span
d=[np.linalg.norm(xs_n[i]-xs_n[j]) for i in range(len(xs_n)) for j in range(i+1,len(xs_n))]
run_spread=float(np.mean(d))
print(f"run's own mean pairwise query spread (normalised): {run_spread:.6f}")
print(f"iterations: {len(xs)}   distinct: {len({tuple(np.round(x,6)) for x in xs})}\n")

np.savez(os.path.join(RES,"states.npz"),
         states=np.stack([h['state'].numpy().reshape(-1) for h in hist]),
         rtg=np.array([h['rtg'] for h in hist]),
         btg=np.array([h['btg'] for h in hist]),
         xs=xs, run_spread=run_spread)

def sweeps(dt):
    dt.eval()
    def propose(state,rtg,btg):
        with torch.no_grad():
            x,_=dt.propose_mf(state.float(),rtg,btg,timestep=0,
                              use_candidate_scoring=False,fidelity_sampling=False)
        return x.double().numpy()
    base=hist[-1]
    x0=propose(base['state'],base['rtg'],base['btg'])
    ds=[float(np.linalg.norm(propose(h['state'],base['rtg'],base['btg'])-x0)) for h in hist]
    rl,rh=min(h['rtg'] for h in hist), max(h['rtg'] for h in hist)
    dr=[float(np.linalg.norm(propose(base['state'],v,base['btg'])-x0)) for v in np.linspace(rl,rh,9)]
    bl,bh=min(h['btg'] for h in hist), max(h['btg'] for h in hist)
    db=[float(np.linalg.norm(propose(base['state'],base['rtg'],v)-x0)) for v in np.linspace(bl,bh,9)]
    return {"state":ds,"RTG":dr,"BTG":db}, (rl,rh,bl,bh)

T,rng=sweeps(mf.dt)
U,_  =sweeps(dt_untrained)
rl,rh,bl,bh=rng

out={}
print(f"{'channel':<8}{'arm':<12}{'max||dx||':>11}{'mean||dx||':>12}{'max/spread':>12}{'mean/spread':>13}")
for nm in ("state","RTG","BTG"):
    row={}
    for arm,S in (("T trained",T),("U untrained",U)):
        a=np.array(S[nm]); mx=float(a.max()); mn=float(a.mean())
        row[arm.split()[0]]=dict(max=mx,mean=mn,ratio_max=mx/run_spread,ratio_mean=mn/run_spread)
        print(f"{nm:<8}{arm:<12}{mx:>11.6f}{mn:>12.6f}{mx/run_spread:>11.1%}{mn/run_spread:>13.1%}")
    out[nm]=row
    print()

RT=out["state"]["T"]["ratio_max"]; RU=out["state"]["U"]["ratio_max"]
frac=RU/RT if RT>0 else float('inf')
print("="*74)
print(f"state channel:  R_T = {RT:.1%}   R_U = {RU:.1%}   R_U/R_T = {frac:.3f}")
if frac>=1.5:
    v="SUPPRESSED"; msg="training REDUCED responsiveness -- learning toward a state-insensitive rule."
elif frac>=0.7:
    v="GENERIC"; msg="untrained is ~as responsive as trained. h44's 49.9% is architecture, NOT learned conditioning -- withdraw it as evidence."
elif frac<=0.3:
    v="LEARNED"; msg="training substantially created the responsiveness."
else:
    v="AMBIGUOUS"; msg="between the locked bands; report both ratios, claim neither."
print(f"VERDICT: {v} -- {msg}")
print("="*74)
json.dump(dict(run_spread=run_spread,channels=out,R_T=RT,R_U=RU,frac=frac,verdict=v,
               raw={"T":T,"U":U},bands=dict(rtg=[rl,rh],btg=[bl,bh])),
          open(os.path.join(RES,"h46.json"),"w"),indent=2)
