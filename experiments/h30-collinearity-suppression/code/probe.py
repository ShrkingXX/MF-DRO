"""Is w[sigma_H]<0 a PREFERENCE, or a partial coefficient under mu-sigma collinearity?
In a GP, high-mu regions are where data sits => low sigma. If mu and sigma are
strongly negatively correlated across candidates, a linear rule fitted on both can
carry a negative sigma weight WITHOUT any behavioural aversion to uncertainty."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from scipy.stats import spearmanr, pearsonr
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg=_build_mf_dro_config("c","Hartmann_6D","c",44,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"
mf=DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch=mf._generate_rollout_batch(); d=mf.d
rho, part_sig, marg_sig = [], [], []
for t in batch:
    if "candidates" not in t or "teacher_scores" not in t: continue
    cnd=t["candidates"].double().numpy(); ts=t["teacher_scores"].double().numpy()
    for j in range(min(cnd.shape[0], ts.shape[0])):
        mu=cnd[j][:,d]; sg=cnd[j][:,d+1]; sc=ts[j]
        if mu.std()<1e-12 or sg.std()<1e-12 or np.std(sc)<1e-12: continue
        rho.append(pearsonr(mu,sg)[0])
        marg_sig.append(spearmanr(sc,sg).correlation)
        # partial: OLS of score on [mu, sigma, 1] -> sign of the sigma coefficient
        A=np.column_stack([mu,sg,np.ones_like(mu)])
        b,*_=np.linalg.lstsq(A,sc,rcond=None)
        part_sig.append(b[1])
f=lambda a:(np.mean(a),np.std(a)/np.sqrt(len(a)))
print(f"candidate sets analysed: {len(rho)}")
m,e=f(rho);      print(f"\ncorr(mu_H, sigma_H) across candidates : {m:+.4f} +/- {e:.4f}")
m2,e2=f(marg_sig);print(f"MARGINAL Spearman(score, sigma_H)     : {m2:+.4f} +/- {e2:.4f}")
m3,e3=f(part_sig);print(f"PARTIAL OLS coef on sigma_H (mu held) : {m3:+.4f} +/- {e3:.4f}")
print(f"  sign agrees with learned w[sigma_H]<0 : {(np.array(part_sig)<0).mean():.1%} of sets")
print()
if m < -0.3 and m3 < 0 and m2 > 0:
    print(">>> SUPPRESSION CONFIRMED: mu and sigma are negatively correlated, the")
    print(">>> teacher's score rises with sigma MARGINALLY but its PARTIAL coef is")
    print(">>> negative. The student's w[sigma_H]<0 reproduces the TEACHER'S OWN")
    print(">>> partial coefficient -- it is not an independent aversion.")
elif m3 > 0:
    print(">>> The teacher's partial sigma coef is POSITIVE, so the student's")
    print(">>> negative weight is NOT inherited from the score structure.")
