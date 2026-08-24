"""Does ONE candidate feature dominate the linear score by SCALE, making the
state-dependent coefficients irrelevant to the argmax?

score_k = sum_f w_f(h) * cf[k,f] + b(h)
The argmax over k is driven by feature f's contribution SPREAD across
candidates: |w_f| * sd_k(cf[:,f]). If one feature's spread dwarfs the rest,
the ranking is that feature's ranking regardless of h.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import (DirectMFRegretOptimization,
                                build_candidate_features, _y_star_for_model)

hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg = _build_mf_dro_config("scale","Hartmann_6D","d",44,bo_iterations=3,num_epochs=10,
    minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,initial_hf=36,
    initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44
mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)
mf._sample_initial_points(); mf._update_ko_ensemble()
batch = mf._generate_rollout_batch(); mf._train_dt(batch)

ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=44)
Xc = bounds[0] + (bounds[1]-bounds[0])*torch.rand(200, mf.d, dtype=torch.float64,
        generator=torch.Generator().manual_seed(500))
cf = build_candidate_features(mf.ko_ensemble[0], Xc, bounds, mf.c_H, mf.c_L,
        torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysa)
NAMES = [f"x[{i}]" for i in range(mf.d)] + ["mu_H","sigma_H","mu_L","sigma_L","dist_inc"]

st = batch[0]["states"][0].double()
FD = torch.float32
H = mf.dt.hidden_size
def hid(st):
    s=st.unsqueeze(0).unsqueeze(0).to(FD)
    r=torch.tensor([[[0.7]]],dtype=FD); b_=torch.tensor([[[22.0]]],dtype=FD)
    ax=torch.zeros(1,1,mf.d,dtype=FD); ae=torch.zeros(1,1,dtype=FD)
    e=[mf.dt.reward_ln(mf.dt.reward_embedding(r)), mf.dt.btg_ln(mf.dt.btg_embed(b_)),
       mf.dt.state_ln(mf.dt.state_embedding(s)),
       mf.dt.action_ln(mf.dt.action_embed_mf(torch.cat([ax,ae.unsqueeze(-1)],dim=-1)))]
    pos=mf.dt.position_embedding(torch.tensor([[0]],dtype=torch.long)).repeat_interleave(4,dim=1)
    seq=torch.stack(e,dim=2).reshape(1,4,H)+pos
    cm=torch.triu(torch.ones(4,4,dtype=torch.bool),diagonal=1)
    return mf.dt.transformer(seq,mask=cm)[0,2::4,:][0]

mf.dt.eval()
with torch.no_grad():
    h = hid(st)
    w = mf.dt.coef_head(h).double().numpy()
    sd = cf.std(dim=0).double().numpy()
    contrib = np.abs(w) * sd
    order = np.argsort(-contrib)
    print(f"{'feature':>10} {'|w|':>9} {'sd_k(cf)':>11} {'|w|*sd':>10} {'share':>8}")
    tot = contrib.sum()
    for i in order:
        print(f"{NAMES[i]:>10} {abs(w[i]):>9.4f} {sd[i]:>11.4f} "
              f"{contrib[i]:>10.4f} {contrib[i]/tot:>7.1%}")
    top = order[0]
    print(f"\nTop feature '{NAMES[top]}' carries {contrib[top]/tot:.1%} of the "
          f"total ranking spread.")
    print(f"Its across-candidate sd is {sd[top]/np.median(sd):.1f}x the median feature's.")
    # Does argmax(score) == argmax of that single feature?
    sc = (cf.to(FD)*mf.dt.coef_head(h).unsqueeze(0)).sum(-1)+mf.dt.bias_head(h)
    print(f"\nargmax(score) == argmax(top feature)?  "
          f"{int(sc.argmax())==int(cf[:,top].argmax()*1)}")
    print(f"argmax(score)={int(sc.argmax())}  argmax({NAMES[top]})={int(cf[:,top].argmax())}"
          f"  argmax(mu_H)={int(cf[:,mf.d].argmax())}")

# ---------------------------------------------------------------------------
# CRITICAL CHECK: H5's h-swap used batch[(p+7) % len(batch)] for p in 0..11,
# i.e. indices 7..18. The batch is built `for ko in ensemble: for _ in
# range(rollouts_per_model)`, so indices 0..19 ALL come from model 0 and share
# a bit-identical tau=0 state (STATE-DIAG reports uniq_tau0_states=10 for 200
# trajectories). If so, H5 swapped a state for ITSELF.
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("Was H5's h-swap a no-op?")
s0 = batch[0]["states"][0].double()
same_within = [bool(torch.allclose(s0, batch[(p+7) % len(batch)]["states"][0].double()))
               for p in range(12)]
print(f"  H5's actual comparisons (idx 7..18) identical to batch[0]: "
      f"{sum(same_within)}/12")
rpm = mf.config.rollouts_per_model
cross = [bool(torch.allclose(s0, batch[k*rpm]["states"][0].double()))
         for k in range(1, len(batch)//rpm)]
print(f"  ACROSS model blocks (idx {rpm}, {2*rpm}, ...): identical to batch[0]: "
      f"{sum(cross)}/{len(cross)}")
print(f"  rollouts_per_model={rpm}, batch={len(batch)}, "
      f"unique tau=0 states={len({tuple(t['states'][0].double().tolist()) for t in batch})}")
if sum(same_within) == 12:
    print("\n  >>> CONFIRMED: H5 compared batch[0] against 12 trajectories that")
    print("  >>> share its EXACT tau=0 state. The 'argmax unchanged 12/12' result")
    print("  >>> is what feeding the SAME state must produce. H5's conclusion")
    print("  >>> that the score head ignores h is UNSUPPORTED by that probe.")

# ---------------------------------------------------------------------------
# CORRECTED h-sensitivity probe: draw the comparison state from a DIFFERENT
# ensemble-model block (indices rpm, 2*rpm, ...), which the check above shows
# are genuinely distinct tau=0 states (0/9 identical to batch[0]).
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("CORRECTED h-sensitivity: comparison states from different model blocks")
moved = 0; tot = 0; corrs = []
with torch.no_grad():
    for p in range(12):
        Xp = bounds[0] + (bounds[1]-bounds[0])*torch.rand(200, mf.d,
                dtype=torch.float64, generator=torch.Generator().manual_seed(300+p))
        ysp = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=44)
        cfp = build_candidate_features(mf.ko_ensemble[0], Xp, bounds, mf.c_H, mf.c_L,
                torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysp)
        h0 = hid(s0)
        sc0 = (cfp.to(FD)*mf.dt.coef_head(h0).unsqueeze(0)).sum(-1)+mf.dt.bias_head(h0)
        a0 = int(sc0.argmax())
        k = 1 + (p % 9)
        hk = hid(batch[k*rpm]["states"][0].double())
        sck = (cfp.to(FD)*mf.dt.coef_head(hk).unsqueeze(0)).sum(-1)+mf.dt.bias_head(hk)
        tot += 1
        if int(sck.argmax()) != a0:
            moved += 1
        corrs.append(float(np.corrcoef(sc0.numpy(), sck.numpy())[0,1]))
print(f"  argmax CHANGED when h comes from a genuinely different state: "
      f"{moved}/{tot} = {moved/tot:.1%}")
print(f"  mean score-vector correlation across states: {np.mean(corrs):.6f}")
print()
if moved/tot > 0.30:
    print("  >>> The score head DOES read h. H5's 'ignores h' conclusion is")
    print("  >>> REFUTED -- it was an artefact of swapping a state for itself.")
else:
    print("  >>> Insensitivity SURVIVES the correction: the score head really")
    print("  >>> does barely read h, and H5's conclusion stands on better evidence.")
