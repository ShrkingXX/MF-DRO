"""Is coef_head constant across REAL ITERATIONS too, or only across ensemble
members within one iteration?

This distinguishes two very different claims:
  (a) narrow: the head can't tell the 10 ensemble members apart at a fixed
      iteration -- true but arguably harmless, since inference sees one state.
  (b) broad : the head emits the same coefficients at iteration 1 and
      iteration 12, i.e. it never adapts through the STATE at all, and
      re-fitting is the sole adaptation channel.
"""
import os, sys
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
from benchmarks import get_benchmark
from dro_runner import _build_mf_dro_config
from src.policy.mf_dro import DirectMFRegretOptimization

hf = get_benchmark("Hartmann_6D_HF"); lf = get_benchmark("Hartmann_6D_LF")
bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
torch.manual_seed(44); np.random.seed(44)
cfg = _build_mf_dro_config("coefiter","Hartmann_6D","d",44,bo_iterations=12,
    num_epochs=10,minimum_hf_fraction=0.25,real_hf_warmup=2,cost_budget=1e9,
    initial_hf=36,initial_lf=60,dkl_threshold=9999,bes_delta=0.0,rollout_length=8)
cfg.seed=44
mf = DirectMFRegretOptimization(cfg, hf["make_objective"](), lf["make_objective"](), bounds)

# capture the real state AND the live weights at each real iteration
snaps = []
_orig = mf.dt.propose_mf
def _rec(state, rtg_target, btg_target, **kw):
    with torch.no_grad():
        h_hidden = None
    snaps.append({'state': state.detach().clone().double(),
                  'coef_w': None})
    return _orig(state, rtg_target, btg_target, **kw)
mf.dt.propose_mf = _rec
mf.run()
mf.dt.propose_mf = _orig
print(f"\ncaptured {len(snaps)} real iterations")

FD = torch.float32; H = mf.dt.hidden_size
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

# FINAL weights, states from every real iteration -> isolates the STATE channel
mf.dt.eval()
with torch.no_grad():
    Hs = np.stack([hid(s['state']).double().numpy() for s in snaps])
    W = np.stack([mf.dt.coef_head(torch.tensor(h_,dtype=FD)).double().numpy() for h_ in Hs])
    FID = np.array([float(mf.dt.fidelity_head(torch.tensor(h_,dtype=FD))) for h_ in Hs])

# LOCALISE the collapse: is it the state ENCODER (h barely varies) or the HEAD
# (h varies but coef_head compresses it)? Compare relative spreads.
def relspread(M):
    n = np.linalg.norm(M, axis=1)
    d = [np.linalg.norm(M[i]-M[j]) for i in range(len(M)) for j in range(i+1,len(M))]
    return np.mean(d)/np.mean(n)
Sm = np.stack([sn['state'].numpy() for sn in snaps])
print("\nLOCALISING THE COLLAPSE (mean pairwise distance / mean norm):")
print(f"  state s   : {relspread(Sm):.6f}")
print(f"  hidden h  : {relspread(Hs):.6f}   <- does the ENCODER pass state through?")
print(f"  coef w    : {relspread(W):.6f}   <- does the HEAD pass h through?")
print(f"  attenuation s->h : {relspread(Hs)/relspread(Sm):.4f}x")
print(f"  attenuation h->w : {relspread(W)/relspread(Hs):.4f}x")
print(f"  fidelity_head p across real iterations: "
      f"min {FID.min():.4f} max {FID.max():.4f} (spread {FID.max()-FID.min():.4f})")
S = np.stack([s['state'].numpy() for s in snaps])
print(f"state vectors across iterations: mean pairwise L2 = "
      f"{np.mean([np.linalg.norm(S[i]-S[j]) for i in range(len(S)) for j in range(i+1,len(S))]):.4f}")
n = np.linalg.norm(W, axis=1); Wn = W/n[:,None]
cos = Wn @ Wn.T; off = cos[~np.eye(len(W),dtype=bool)]
sv = np.linalg.svd(W, compute_uv=False)
print(f"||w|| across REAL iterations min/max : {n.min():.4f} / {n.max():.4f} "
      f"(ratio {n.max()/n.min():.4f}x)")
print(f"pairwise cosine(w_i,w_j)             : min {off.min():.8f}  mean {off.mean():.8f}")
print(f"sv1/sum(sv)                          : {sv[0]/sv.sum():.6f}")

# ---- the decisive test: does the ARGMAX move across real-iteration states? ----
from src.policy.mf_dro import build_candidate_features, _y_star_for_model
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_coefiter_states.npz"), S=S, W=W)
ysa = _y_star_for_model(mf.ko_ensemble[0], mf.y_star_pool, seed=44)
moved_pools = 0; n_distinct = []
with torch.no_grad():
    for pidx in range(12):
        Xp = bounds[0] + (bounds[1]-bounds[0])*torch.rand(200, mf.d,
                dtype=torch.float64, generator=torch.Generator().manual_seed(700+pidx))
        cfp = build_candidate_features(mf.ko_ensemble[0], Xp, bounds, mf.c_H, mf.c_L,
                torch.zeros(mf.d, dtype=torch.float64), y_star_arr=ysa)
        ams = []
        for sn in snaps:
            hk = hid(sn['state'])
            sc = (cfp.to(FD)*mf.dt.coef_head(hk).unsqueeze(0)).sum(-1)+mf.dt.bias_head(hk)
            ams.append(int(sc.argmax()))
        n_distinct.append(len(set(ams)))
        if len(set(ams)) > 1:
            moved_pools += 1
print(f"\nARGMAX across the 12 REAL-ITERATION states, per candidate pool:")
print(f"  pools where the argmax MOVED : {moved_pools}/12 = {moved_pools/12:.1%}")
print(f"  distinct argmaxes per pool   : mean {np.mean(n_distinct):.2f}  max {max(n_distinct)}")
print(f"  angle implied by min cosine  : {np.degrees(np.arccos(min(off.min(),1.0))):.2f} deg")

print()
if off.min() > 0.9999:
    print(">>> BROAD: coef_head is constant ACROSS REAL ITERATIONS too. The state")
    print(">>> channel never adapts during a run; re-fitting is the SOLE channel.")
else:
    print(">>> NARROW: the head DOES vary across real iterations, so the earlier")
    print(">>> 'constant' result is about ensemble members at a fixed iteration only.")
