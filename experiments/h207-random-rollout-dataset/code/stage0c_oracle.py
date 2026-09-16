"""h207 Stage 0c -- SC9 (GATE) for MIX-oracle: are the halves separated by the label,
and is the oracle half labelled as GOOD? Also verifies the identical-tau0-state
property the selectivity argument rests on, and that forced_x actually applied."""
import os, sys, io, contextlib, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp"); BUDGET = 40.0; H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import worker_MIXO as WO          # reuse the EXACT wrapper the arm runs
import src.policy.mf_dro as MF
CAP = {}
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self, *a, **k): _oi(self, *a, **k); CAP["mf"] = self
MF.DirectMFRegretOptimization.__init__ = _spy
WO._ORACLE["x_star"] = torch.tensor(WO.XSTAR["Borehole_8D"], dtype=torch.float64)
WO._ORACLE["rng"] = torch.Generator().manual_seed(42 * 7919 + 207)
MF.simulate_mf_trajectory = WO._mixed_sim
WO.h83.BUDGET = BUDGET
with contextlib.redirect_stdout(io.StringIO()):
    WO.h83.run("Borehole_8D", "MF-DRO", 42, os.path.join(SCR, "h207_sc0c.json"))
mf = CAP["mf"]; M = len(mf.ko_ensemble)
with contextlib.redirect_stdout(io.StringIO()):
    mf.config.rollout_mix = [('mes', 20, None), ('oracle', 20, None)]
    batch = mf._generate_rollout_batch()
    mf.config.rollout_mix = None
print("=" * 78)
print(f"Stage 0c: {len(batch)} rollouts, policies: "
      f"{ {p: sum(1 for t in batch if t['_policy']==p) for p in set(t['_policy'] for t in batch)} }")
xs = WO._ORACLE["x_star"]; lo, hi = mf.bounds[0], mf.bounds[1]
xs_unit = ((xs - lo) / (hi - lo)).numpy()
ok = True; per = []
for m in range(M):
    mem = batch[m*40:(m+1)*40]
    mes = [t for t in mem if t['_policy'] == 'mes']; orc = [t for t in mem if t['_policy'] == 'oracle']
    # (a) identical tau=0 state across the two halves
    s_mes = torch.stack([t['states'][0] for t in mes]); s_orc = torch.stack([t['states'][0] for t in orc])
    same_state = float((s_mes.mean(0) - s_orc.mean(0)).abs().max())
    within = float(max(s_mes.std(0).max(), s_orc.std(0).max()))
    # (b) forced_x applied: oracle tau=0 action near x* (unit coords), MES far
    d_orc = float(np.mean([np.abs(t['actions_x'][0].numpy() - xs_unit).max() for t in orc]))
    d_mes = float(np.mean([np.abs(t['actions_x'][0].numpy() - xs_unit).max() for t in mes]))
    # (c) label separation
    a = np.array([t['term_imp'] for t in mes]); b = np.array([t['term_imp'] for t in orc])
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2); dsep = (b.mean() - a.mean()) / max(sd, 1e-9)
    print(f"  m{m}: tau0-state max|mean diff|={same_state:.2e} (within-half max sd {within:.2e})   "
          f"|a0 - x*|inf  oracle {d_orc:.3f}  MES {d_mes:.3f}")
    print(f"       label  MES mean {a.mean():8.2f} (sd {a.std(ddof=1):6.2f}, frac>0 {np.mean(a>0):.2f})   "
          f"ORACLE mean {b.mean():8.2f} (sd {b.std(ddof=1):6.2f}, frac>0 {np.mean(b>0):.2f})   d = {dsep:5.2f}")
    per.append(dsep)
    ok &= (same_state < 1e-6) and (d_orc < 0.1) and (d_mes > 0.1) and (dsep > 0.5) and (b.mean() > a.mean())
inc = max(mf.data_hf_y)
print(f"\n  calibration run (seed 42, cost {BUDGET:.0f}+init): REAL incumbent = {inc:.2f}  (|OPT| 309.58; NIR-config run reached 255.92)")
print(f"\nSC9 v3 (GATE): identical tau0 state, forced_x applied, oracle half labelled better (d > 0.5, mean higher) on every member.")
print(f"  (v2's >90%-positive criterion dropped: an arm that drives the incumbent to the optimum makes every label ~0 by construction)")
print(f"     -> {'PASS' if ok else 'FAIL'}   d per member = {[round(x,2) for x in per]}")
print(f"STAGE 0c: {'PASS' if ok else 'FAIL'}"); print("=" * 78)
sys.exit(0 if ok else 1)
