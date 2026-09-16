"""h207 Stage 0 v3 -- label = terminal_improvement (the frozen metric's own quantity).

Gates: SC0 parity on the K=8/ROI path, SC1 the fork is real, SC3 there is a
well-defined ranking among random rollouts, SC5 the MES and random halves are
separated by the label (MIX is only a test of RTG selectivity if they are).
Diagnostics: SC4 action-informativeness, SC6 probe, SC7 timing.
"""
import os, sys, json, time, subprocess, importlib.util, io, contextlib
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp"); H = os.path.dirname(os.path.abspath(__file__))
BUDGET = 40.0; PY = sys.executable; LABEL = "terminal_improvement"
SWEEP = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
ok = {}
def say(*a): print(*a, flush=True)
say("=" * 78)

WT = os.path.join(SCR, "h207_wt_head")
if os.path.isdir(WT):
    subprocess.run(["git", "worktree", "remove", "--force", WT], cwd=REPO, capture_output=True)
subprocess.run(["git", "worktree", "add", "--detach", WT, "8b25650"], cwd=REPO, check=True, capture_output=True)  # last PRE-implementation commit
def ctrl(repo, tag, reward):
    out = os.path.join(SCR, f"h207_ctrl_{tag}.json"); t0 = time.time()
    subprocess.run([PY, os.path.join(H, "ctrl_run.py"), repo, out, str(BUDGET), reward],
                   check=True, capture_output=True, cwd=repo)
    d = json.load(open(out)); d["wall"] = time.time() - t0; return d
old = ctrl(WT, "old_mes", "mes_entropy"); new = ctrl(REPO, "new_mes", "mes_entropy")
say(f"SC0 parity on control path (K=8, nopos, ROI-Q10, mes_entropy, BUDGET={BUDGET:.0f}; old = 8b25650, pre-implementation):")
say(f"     old n_q={old['n_q']} md5={old['md5']}   new n_q={new['n_q']} md5={new['md5']}")
ok["SC0"] = old["md5"] == new["md5"]; say(f"     -> {'PASS' if ok['SC0'] else 'FAIL'}")

ti = ctrl(REPO, "new_ti", LABEL)
ra, rb = np.array(new["rtg_target"], float), np.array(ti["rtg_target"], float); n = min(len(ra), len(rb))
same = np.allclose(ra[:n], rb[:n])
say(f"\nSC1 reward fork is REAL: rtg_target mes[:4]={np.round(ra[:4],4)} {LABEL}[:4]={np.round(rb[:4],4)}")
say(f"     md5 mes={new['md5']} ti={ti['md5']}   rtg_target identical over {n}: {same}")
ok["SC1"] = (not same) and (new["md5"] != ti["md5"]); say(f"     -> {'PASS' if ok['SC1'] else 'FAIL'}")

import src.policy.mf_dro as MF
CAP = {}
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self, *a, **k): _oi(self, *a, **k); CAP["mf"] = self; self._h168_probe = SWEEP
MF.DirectMFRegretOptimization.__init__ = _spy
_s = importlib.util.spec_from_file_location("h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8; c.max_seq_length = 256
    c.absolute_timesteps = False; c.real_prefix_training = False
    c.disable_position_embedding = True; c.random_p_hf = 0.5
    return c
w._build_mf_dro_config = _b; w.ROLLOUT_REWARD = LABEL; w.BUDGET = BUDGET
with contextlib.redirect_stdout(io.StringIO()):
    w.run("Borehole_8D", "MF-DRO", 42, os.path.join(SCR, "h207_sc_ti.json"))
mf = CAP["mf"]
say(f"\n     effective mf.rollout_reward = {mf.rollout_reward!r}")
ok["SC1"] = ok["SC1"] and mf.rollout_reward == LABEL

p168 = getattr(mf, "h168_probe_per_iter", [])
fin = all(np.isfinite([v for rec in p168 for pr in rec["probes"] for v in pr["x"]])) if p168 else False
def sens(recs):
    out = []
    for rec in recs:
        xs = {pr["rtg"]: np.array(pr["x"]) for pr in rec["probes"] if pr["state"] == "real"}
        if len(xs) >= 2: ref = xs[min(xs)]; out.append(max(np.abs(x - ref).max() for x in xs.values()))
    return float(np.mean(out)) if out else float("nan")
say(f"\nSC6 H168 probe: {len(p168)} iters, finite={fin}, mean RTG-sensitivity max|dx| = {sens(p168):.3e}")
ok["SC6"] = bool(p168) and fin; say(f"     -> {'PASS' if ok['SC6'] else 'FAIL'}")

with contextlib.redirect_stdout(io.StringIO()):
    t0 = time.time(); mf.config.rollout_mix = [('mes', 20, None)];    b_mes = mf._generate_rollout_batch(); t_mes = time.time() - t0
    t0 = time.time(); mf.config.rollout_mix = [('random', 100, None)]; b_rnd = mf._generate_rollout_batch(); t_rnd = time.time() - t0
    t0 = time.time(); mf.config.rollout_mix = [('random', 100, 20)];   b_w = mf._generate_rollout_batch(); t_w = time.time() - t0
    mf.config.rollout_mix = None
M = len(mf.ko_ensemble); d = mf.d
def by_member(b, n_per): return [b[i*n_per:(i+1)*n_per] for i in range(M)]
inc = max(mf.data_hf_y)
say(f"\nSC3 a well-defined ranking among random rollouts (label = best observed HF - incumbent {inc:.2f}, raw units):")
sc3 = True
for m, trs in enumerate(by_member(b_rnd, 100)):
    v = np.array([t['term_imp'] for t in trs]); lf = np.array([t['lf_fraction'] for t in trs])
    o = np.argsort(-v); top, bot = o[:20], o[-20:]
    distinct = len(np.unique(np.round(v, 9))) / len(v)
    say(f"     m{m}: distinct={distinct:.2f}  mean={v.mean():8.2f}  top20={v[top].mean():8.2f}  bot20={v[bot].mean():8.2f}  "
        f"frac>0 (beat incumbent)={np.mean(v > 0):.2f}  LF top20={lf[top].mean():.2f} all={lf.mean():.2f}")
    sc3 &= distinct > 0.9 and v[top].mean() > v[bot].mean()
ok["SC3"] = bool(sc3); say(f"     -> {'PASS' if ok['SC3'] else 'FAIL'}  (distinct > 0.9, top20 > bot20)")

say(f"\nSC5 (GATE) halves separation for MIX: MES-20 vs UNSELECTED random-20 per member, in label units")
sc5 = True; seps = []
for m, (tm, tr) in enumerate(zip(by_member(b_mes, 20), by_member(b_rnd, 100))):
    a = np.array([t['term_imp'] for t in tm]); b_ = np.array([t['term_imp'] for t in tr[:20]])
    sd = np.sqrt((a.var(ddof=1) + b_.var(ddof=1)) / 2); sep = (a.mean() - b_.mean()) / max(sd, 1e-9)
    say(f"     m{m}: MES mean {a.mean():8.2f} (sd {a.std(ddof=1):6.2f}, frac>0 {np.mean(a>0):.2f})   "
        f"random mean {b_.mean():8.2f} (sd {b_.std(ddof=1):6.2f})   separation d = {sep:5.2f}")
    seps.append(sep); sc5 &= sep > 1.0
ok["SC5"] = bool(sc5); say(f"     -> {'PASS' if ok['SC5'] else 'FAIL'}  (Cohen d > 1.0 on every member; otherwise MIX cannot test selectivity)")
say(f"     W's dataset vs NIR's: top20-of-100-random mean {np.mean([t['term_imp'] for t in b_w]):8.2f}  "
    f"vs MES-20 mean {np.mean([t['term_imp'] for t in b_mes]):8.2f}")

say(f"\nSC4 action-informativeness at tau=0 (Layer 3, restated): R^2 of label on (x_0, ell_0) within member")
def r2_action(batch, n_per):
    out = []
    for m in range(M):
        idx = [i for i in range(len(batch)) if i // n_per == m]
        X = np.stack([np.concatenate([batch[i]['actions_x'][0].numpy(), [float(batch[i]['actions_ell'][0])], [1.0]]) for i in idx])
        y = np.array([batch[i]['term_imp'] for i in idx])
        if y.std() < 1e-12 or len(idx) <= X.shape[1] + 5: out.append(float('nan')); continue
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
        out.append(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return out
r2r = r2_action(b_rnd, 100)
say(f"     random (n=100/member, {d+2} params): R^2 per member = {[round(x,3) for x in r2r]}  mean {np.nanmean(r2r):.3f}")
say(f"     MES (n=20/member): not estimable at n=20 with {d+2} params; by construction ~0 (one action per state)")
say(f"     (diagnostic, not a gate; the first of 8 steps cannot explain much of an 8-step outcome)")
def eta2(groups):
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups); sst = ((allv - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")
say(f"     eta^2 (member identity): MES {eta2([np.array([t['term_imp'] for t in g]) for g in by_member(b_mes,20)]):.3f}   "
    f"random {eta2([np.array([t['term_imp'] for t in g]) for g in by_member(b_rnd,100)]):.3f}")

say(f"\nSC7 wall per iteration: MES-20/member {t_mes:.1f}s   random-100/member {t_rnd:.1f}s   random-100->top20 {t_w:.1f}s   (W/ctrl = {t_w/max(t_mes,1e-9):.2f}x)")
ok["SC7"] = True
say(f"\nSC2 identity gate: run separately (PASS at 122.29066752728207)")
allok = all(ok.values())
say(f"\nSCs: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in sorted(ok.items())))
say(f"STAGE 0: {'PASS' if allok else 'FAIL'}"); say("=" * 78)
sys.exit(0 if allok else 1)
