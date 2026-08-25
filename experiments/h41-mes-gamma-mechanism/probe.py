"""What actually drives MES's argmax? MES is a monotone-decreasing function of
gamma=(y*-mu)/sigma alone (sigma cancels from H0-H1). Measure gamma at the chosen
point vs the pool, and the sign of (y*-mu) there."""
import os,sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
sys.path.insert(0,"/Users/yurucui/Desktop/DRO-Code/DRO-aistats-submission")
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
import src.policy.mf_dro as M
hf=get_benchmark("Hartmann_6D_HF"); lf=get_benchmark("Hartmann_6D_LF")
bounds=torch.tensor([hf["domain_min"],hf["domain_max"]],dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg=_build_mf_dro_config("g","Hartmann_6D","g",44,bo_iterations=1,num_epochs=1,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=6,
    initial_lf=45,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44; cfg.rollout_reward="mes_entropy"
mf=M.DirectMFRegretOptimization(cfg,hf["make_objective"](),lf["make_objective"](),bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
ko=mf.ko_ensemble[0]
g_ch, g_pool, frac_above, pct_sig, pct_mu = [],[],[],[],[]
for p in range(40):
    X=bounds[0]+(bounds[1]-bounds[0])*torch.rand(200,mf.d,dtype=torch.float64,
        generator=torch.Generator().manual_seed(1000+p))
    proxy=M._build_hf_proxy_model(ko)
    ys=np.asarray(M.thompson_sample_y_star(proxy,X,K=10),dtype=float).reshape(-1)
    with torch.no_grad():
        post=proxy.posterior(X)
        mu=post.mean.reshape(-1).numpy(); sg=np.sqrt(post.variance.clamp_min(1e-12).reshape(-1).numpy())
    gam=((ys[None,:]-mu[:,None])/sg[:,None]).mean(axis=1)     # mean gamma per candidate
    x_raw,ell,_=M.compute_joint_mf_mes(ko,X,mf.c_H,mf.c_L)
    k=int((X-x_raw.unsqueeze(0)).norm(dim=1).argmin())
    g_ch.append(gam[k]); g_pool.append(gam.mean())
    frac_above.append(float((ys.mean()-mu[k])<0))             # is mu_chosen ABOVE mean y*?
    pct_sig.append(float((sg<sg[k]).mean())); pct_mu.append(float((mu<mu[k]).mean()))
f=lambda a:(np.mean(a),np.std(a)/np.sqrt(len(a)))
print(f"pools: {len(g_ch)}\n")
m,e=f(g_ch);  print(f"  gamma at the CHOSEN candidate     : {m:+.4f} +/- {e:.4f}")
m2,e2=f(g_pool);print(f"  gamma averaged over the pool      : {m2:+.4f} +/- {e2:.4f}")
print(f"  -> chosen gamma is {'LOWER' if m<m2 else 'HIGHER'} than pool mean, as MES requires"
      f" (MES decreases in gamma)")
print(f"\n  fraction of pools where mu_chosen > mean(y*)  : {np.mean(frac_above):.1%}")
print(f"     (when mu > y*, gamma<0 and SMALLER sigma makes gamma MORE negative -> higher MES)")
m3,_=f(pct_sig); m4,_=f(pct_mu)
print(f"\n  chosen sigma percentile : {m3:.1%}")
print(f"  chosen mu    percentile : {m4:.1%}")
