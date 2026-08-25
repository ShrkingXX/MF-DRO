import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
FD=torch.float32

hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(42); np.random.seed(42)
cfg=_build_mf_dro_config("h44","Hartmann_6D","reg",42,bo_iterations=20,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e12,initial_hf=6,
    initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=42; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=False
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
hist=[]
_orig=mf.dt.propose_mf
def _rec(state,rtg,btg,**kw):
    hist.append(dict(state=state.detach().clone(),rtg=float(rtg),btg=float(btg)))
    return _orig(state,rtg,btg,**kw)
mf.dt.propose_mf=_rec
r=mf.run(); mf.dt.propose_mf=_orig

xs=np.array(r["x_t_trace"])                      # RAW domain scale
span=(bounds[1]-bounds[0]).numpy()
xs_n=(xs-bounds[0].numpy())/span                 # normalise to [0,1]^d for comparability
d=[np.linalg.norm(xs_n[i]-xs_n[j]) for i in range(len(xs_n)) for j in range(i+1,len(xs_n))]
run_spread=float(np.mean(d))
print(f"run's own mean pairwise query spread (normalised): {run_spread:.6f}")
print(f"iterations: {len(xs)}   distinct: {len({tuple(np.round(x,6)) for x in xs})}\n")

mf.dt.eval()
def propose(state,rtg,btg):
    with torch.no_grad():
        x,_=mf.dt.propose_mf(state.float(),rtg,btg,timestep=0,
                              use_candidate_scoring=False,fidelity_sampling=False)
    return x.double().numpy()                     # already [0,1]^d

base=hist[-1]
x0=propose(base['state'],base['rtg'],base['btg'])

# 1. STATE: substitute each real tau=0 state from the run
ds=[np.linalg.norm(propose(h['state'],base['rtg'],base['btg'])-x0) for h in hist]
# 2. RTG: sweep the realised band
rl,rh=min(h['rtg'] for h in hist), max(h['rtg'] for h in hist)
dr=[np.linalg.norm(propose(base['state'],v,base['btg'])-x0)
    for v in np.linspace(rl,rh,9)]
# 3. BTG: sweep the realised band
bl,bh=min(h['btg'] for h in hist), max(h['btg'] for h in hist)
db=[np.linalg.norm(propose(base['state'],base['rtg'],v)-x0)
    for v in np.linspace(bl,bh,9)]

out={}
print(f"{'channel':<10}{'max ||dx||':>12}{'mean ||dx||':>13}{'max / run spread':>19}")
for nm,arr,rng in [("state",ds,f"{len(hist)} real states"),
                   ("RTG",dr,f"[{rl:.3f}, {rh:.3f}]"),
                   ("BTG",db,f"[{bl:.1f}, {bh:.1f}]")]:
    mx=float(np.max(arr)); mn=float(np.mean(arr)); ratio=mx/run_spread
    out[nm]=dict(max=mx,mean=mn,ratio=ratio,range=rng)
    print(f"{nm:<10}{mx:>12.6f}{mn:>13.6f}{ratio:>18.1%}   swept {rng}")

worst=max(v["ratio"] for v in out.values())
print("\n"+"="*66)
if worst < 0.10:
    print(f"PRED 1 INACTIVE: largest input effect is {worst:.1%} of the run's own")
    print("  query spread. Re-fitting explains essentially all query movement;")
    print("  the regression head is inactive like the scoring head.")
elif worst > 0.50:
    print(f"PRED 2 ACTIVE: {worst:.1%} of run spread -- the regression head DOES")
    print("  condition. The head choice is load-bearing; mechanism section needs revising.")
else:
    print(f"PARTIAL: largest input effect {worst:.1%} of run spread (10-50% band).")
    print("  Report the ratio; claim neither inactive nor active.")
print("="*66)
json.dump(dict(run_spread=run_spread,channels=out),
          open(os.path.join(os.path.dirname(__file__),"..","results","h44.json"),"w"),indent=2)
