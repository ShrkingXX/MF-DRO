"""H82 gate: memoisation must be BIT-FOR-BIT inert.

Runs a short real MF-DRO trajectory and prints every quantity the DT is trained
on. Executed once on the working tree and once on git HEAD; the two dumps must
be byte-identical. Memoisation caches pure functions, so anything other than an
exact match means the purity assumption was wrong.
"""
import os,sys,json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

BENCH,SEED="Hartmann_6D",44
hf=get_benchmark(f"{BENCH}_HF"); lf=get_benchmark(f"{BENCH}_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h82gate",BENCH,"MF-DRO",SEED,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=6,initial_lf=45,
    dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"; cfg.use_candidate_scoring=False
cfg.M=3                      # small ensemble keeps the gate quick
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf.run()
out=[dict(fid=int(e["ell_t"]),
          x=[float(v) for v in np.asarray(e["x_t"],dtype=float).reshape(-1)],
          y=float(e["y_t"]), regret=float(e["regret"]),
          inf=float(e.get("inference_regret",float("nan"))))
     for e in mf.iteration_log]
print("GATEJSON "+json.dumps(out,sort_keys=True))
