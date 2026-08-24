"""H28 (see ../protocol.md). Single process, thread-capped."""
import os, sys, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v]="1"
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from scipy.stats import spearmanr
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
SEED=44
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(SEED); np.random.seed(SEED)
cfg=_build_mf_dro_config("h28","Hartmann_6D","g",SEED,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=SEED; cfg.rollout_reward="mes_entropy"
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch()
d=mf.d
pct_sig, pct_mu, rho_sig, rho_mu = [], [], [], []
n=0
for t in batch:
    if "candidates" not in t: continue
    cnd=t["candidates"].double().numpy()          # [T,K,F]
    idx=t["chosen_idx"].long().numpy().reshape(-1)
    ts=t.get("teacher_scores")
    ts=ts.double().numpy() if ts is not None else None
    for j in range(min(len(idx),cnd.shape[0])):
        pool=cnd[j]; k=int(idx[j])
        if k<0 or k>=pool.shape[0]: continue
        sg=pool[:,d+1]; mu=pool[:,d]
        if sg.std()<1e-12 or mu.std()<1e-12: continue
        pct_sig.append(float((sg<sg[k]).mean()))
        pct_mu.append(float((mu<mu[k]).mean()))
        if ts is not None and j<ts.shape[0]:
            s=ts[j]
            if np.std(s)>1e-12:
                rho_sig.append(spearmanr(s,sg).correlation)
                rho_mu.append(spearmanr(s,mu).correlation)
        n+=1
f=lambda a: (np.mean(a), np.std(a)/np.sqrt(len(a))) if a else (float('nan'),)*2
ms,es=f(pct_sig); mm,em=f(pct_mu)
print(f"decisions analysed: {n}")
print(f"\nCONTROL  chosen mu_H percentile within its pool : {mm:.1%} +/- {em:.1%}")
print(f"         -> {'OK, teacher prefers high mu_H' if mm>0.5 else '*** CONTROL FAILED ***'}")
print(f"\nPRIMARY  chosen sigma_H percentile within pool  : {ms:.1%} +/- {es:.1%}")
if rho_sig:
    print(f"         Spearman(teacher score, sigma_H)      : {np.mean(rho_sig):+.4f}")
    print(f"         Spearman(teacher score, mu_H)         : {np.mean(rho_mu):+.4f}")
print("\n"+"="*68)
if mm<=0.5:
    print("CONTROL FAILED -- nothing here is interpretable.")
elif ms<0.40:
    print(f"PRED 1: teacher PREFERS LOW sigma_H ({ms:.1%}). The student's negative")
    print("  weight is INHERITED, not an inversion. The paper's 'the student")
    print("  inverted the sign of its teacher's defining term' must be RETRACTED.")
elif ms>0.60:
    print(f"PRED 2: teacher prefers HIGH sigma_H ({ms:.1%}). The student genuinely")
    print("  inverted it; current wording stands.")
else:
    print(f"PRED 3 AMBIGUOUS: teacher is ~indifferent to sigma_H ({ms:.1%}). The")
    print("  paper should say only that the teacher supplies no uncertainty-seeking")
    print("  signal to imitate -- neither inherited nor inverted.")
print("="*68)
json.dump({"n":n,"pct_sigma":ms,"pct_mu":mm,
           "rho_sigma":float(np.mean(rho_sig)) if rho_sig else None,
           "rho_mu":float(np.mean(rho_mu)) if rho_mu else None},
          open(os.path.join(os.path.dirname(__file__),"..","results","h28.json"),"w"),indent=2)
