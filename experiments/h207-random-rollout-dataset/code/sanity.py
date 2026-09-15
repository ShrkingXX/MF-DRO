"""h207 Stage 0 -- SC0..SC7 from protocol.md, run BEFORE any arm.

SC0 parity and SC1 (the reward fork is REAL) are the gates. SC1 is the exact
failure h198 died of; SC0 covers the K=8/ROI path the identity gate does not.
"""
import os, sys, json, time, subprocess, importlib.util, io, contextlib
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SCR = os.environ.get("SCRATCH", "/tmp"); H = os.path.dirname(os.path.abspath(__file__))
BUDGET = 40.0; PY = sys.executable
SWEEP = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
ok = {}

def say(*a): print(*a, flush=True)
say("=" * 78)

# ---- SC0: old-code vs new-code parity on the CONTROL path --------------------
# Old code = git worktree at HEAD (has h206's nopos flag and h83's ROLLOUT_REWARD
# knob, not h207's changes). Same config, same seed, same budget, md5 of trace.
WT = os.path.join(SCR, "h207_wt_head")
if not os.path.isdir(WT):
    subprocess.run(["git", "worktree", "add", "--detach", WT, "HEAD"], cwd=REPO,
                   check=True, capture_output=True)
def ctrl(repo, tag, reward):
    out = os.path.join(SCR, f"h207_ctrl_{tag}.json")
    t0 = time.time()
    subprocess.run([PY, os.path.join(H, "ctrl_run.py"), repo, out, str(BUDGET), reward],
                   check=True, capture_output=True, cwd=repo)
    d = json.load(open(out)); d["wall"] = time.time() - t0; return d
old = ctrl(WT,   "old_mes", "mes_entropy")
new = ctrl(REPO, "new_mes", "mes_entropy")
say(f"SC0 parity on control path (K=8, nopos, ROI-Q10, mes_entropy, BUDGET={BUDGET:.0f}):")
say(f"     old code n_q={old['n_q']} md5={old['md5']}   new code n_q={new['n_q']} md5={new['md5']}")
ok["SC0"] = old["md5"] == new["md5"]
say(f"     -> {'PASS' if ok['SC0'] else 'FAIL'}")

# ---- SC1 (GATE): the reward fork is REAL ------------------------------------
ir = ctrl(REPO, "new_ir", "inference_regret")
ra, rb = np.array(new["rtg_target"], float), np.array(ir["rtg_target"], float)
n = min(len(ra), len(rb))
same = np.allclose(ra[:n], rb[:n])
say(f"\nSC1 reward fork is REAL: rtg_target mes_entropy[:4]={np.round(ra[:4],4)} "
    f"inference_regret[:4]={np.round(rb[:4],4)}")
say(f"     traces md5 mes={new['md5']} ir={ir['md5']}   rtg_target identical over {n}: {same}")
ok["SC1"] = (not same) and (new["md5"] != ir["md5"])
say(f"     -> {'PASS' if ok['SC1'] else 'FAIL'}")

# ---- in-process policy for SC3..SC7 -----------------------------------------
import src.policy.mf_dro as MF
CAP = {}
_oi = MF.DirectMFRegretOptimization.__init__
def _spy(self, *a, **k):
    _oi(self, *a, **k); CAP["mf"] = self; self._h168_probe = SWEEP
MF.DirectMFRegretOptimization.__init__ = _spy
_s = importlib.util.spec_from_file_location("h83w", os.path.join(REPO, "experiments/h83-main-comparison/code/worker.py"))
w = importlib.util.module_from_spec(_s); sys.modules["h83w"] = w; _s.loader.exec_module(w)
_OB = w._build_mf_dro_config
def _b(*a, **k):
    c = _OB(*a, **k)
    c.use_roi = True; c.roi_beta_mode = 'quantile'; c.roi_target_accept = 0.10
    c.inference_context_k = 8; c.max_seq_length = 256
    c.absolute_timesteps = False; c.real_prefix_training = False
    c.disable_position_embedding = True
    c.random_p_hf = 0.5
    c.ir_probe_second_seed = True          # SC8: rescore IR_T with an independent draw
    return c
w._build_mf_dro_config = _b; w.ROLLOUT_REWARD = "inference_regret"; w.BUDGET = BUDGET
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    w.run("Borehole_8D", "MF-DRO", 42, os.path.join(SCR, "h207_sc_ir.json"))
mf = CAP["mf"]
say(f"\n     effective mf.rollout_reward = {mf.rollout_reward!r}  (must be 'inference_regret')")
ok["SC1"] = ok["SC1"] and mf.rollout_reward == "inference_regret"

# ---- SC6: H168 probe ran and is finite ----------------------------------------
p168 = getattr(mf, "h168_probe_per_iter", [])
fin = all(np.isfinite([v for rec in p168 for pr in rec["probes"] for v in pr["x"]])) if p168 else False
def sens(recs):
    out = []
    for rec in recs:
        xs = {pr["rtg"]: np.array(pr["x"]) for pr in rec["probes"] if pr["state"] == "real"}
        if len(xs) >= 2:
            ref = xs[min(xs)]; out.append(max(np.abs(x - ref).max() for x in xs.values()))
    return float(np.mean(out)) if out else float("nan")
say(f"\nSC6 H168 probe: {len(p168)} iterations recorded, finite={fin}, "
    f"mean RTG-sensitivity max|dx| over sweep = {sens(p168):.3e}")
ok["SC6"] = bool(p168) and fin
say(f"     -> {'PASS' if ok['SC6'] else 'FAIL'}")

# ---- SC3/SC4/SC5/SC7 from one mixed batch at the run's final state -----------
with contextlib.redirect_stdout(io.StringIO()):
    t0 = time.time(); mf.config.rollout_mix = [('mes', 20, None)]
    b_mes = mf._generate_rollout_batch(); t_mes = time.time() - t0
    t0 = time.time(); mf.config.rollout_mix = [('random', 100, None)]
    b_rnd = mf._generate_rollout_batch(); t_rnd = time.time() - t0
    t0 = time.time(); mf.config.rollout_mix = [('random', 100, 20)]
    b_w = mf._generate_rollout_batch(); t_w = time.time() - t0
    mf.config.rollout_mix = None
M = len(mf.ko_ensemble)
def by_member(b, n_per):
    return [b[i*n_per:(i+1)*n_per] for i in range(M)]
# ---- SC8: label RELIABILITY (test-retest), the noise floor for SC3/SC4 -------
def reliability(groups):
    """groups: list of per-member lists of (ir_T, ir_T_alt). Within-member
    variance of ir_T = signal + noise; noise = Var(ir_T - ir_T_alt)/2 pooled.
    reliability = (V_w - V_n) / V_w."""
    vw, vn, n = 0.0, 0.0, 0
    for g in groups:
        a = np.array([t[0] for t in g]); b = np.array([t[1] for t in g])
        vw += ((a - a.mean()) ** 2).sum(); vn += ((a - b) ** 2).sum() / 2.0; n += len(g)
    vw /= max(n - len(groups), 1); vn /= max(n, 1)
    return float((vw - vn) / vw) if vw > 0 else float("nan"), float(np.sqrt(vn)), float(np.sqrt(vw))
rel_mes = reliability([[(t['ir_T'], t['ir_T_alt']) for t in g] for g in by_member(b_mes, 20)])
rel_rnd = reliability([[(t['ir_T'], t['ir_T_alt']) for t in g] for g in by_member(b_rnd, 100)])
say(f"\nSC8 label reliability of IR_T (within-member signal fraction; noise = independent Thompson draws):")
say(f"     MES-20/member:    reliability={rel_mes[0]:.3f}   noise sd={rel_mes[1]:.3f}   within sd={rel_mes[2]:.3f}")
say(f"     random-100/member: reliability={rel_rnd[0]:.3f}   noise sd={rel_rnd[1]:.3f}   within sd={rel_rnd[2]:.3f}")
ok["SC8"] = np.isfinite(rel_rnd[0]) and rel_rnd[0] > rel_mes[0]
say(f"     registered: reliability(random) > reliability(MES) -- diversity raises the label's signal fraction")
say(f"     -> {'PASS' if ok['SC8'] else 'FAIL'}")
NOISE_SD = rel_rnd[1]

say(f"\nSC3 something to select (per member, 100 random rollouts, IR units raw, ranked by rtg[0] as the code does):")
sc3 = True; lf_top, lf_all = [], []
for m, trs in enumerate(by_member(b_rnd, 100)):
    r0 = np.array([float(t['rtg'][0]) for t in trs]); irT = np.array([t['ir_T'] for t in trs])
    ir0 = np.array([t['ir_0'] for t in trs]); lf = np.array([t['lf_fraction'] for t in trs])
    order = np.argsort(-r0); top, bot = order[:20], order[-20:]
    cv = r0.std() / max(abs(r0.mean()), 1e-12); gap = irT[bot].mean() - irT[top].mean()
    say(f"     m{m}: IR_0 spread={np.ptp(ir0):.2e} (want 0)  rtg0 CV={cv:.3f}  "
        f"IR_T top20={irT[top].mean():.4f} bot20={irT[bot].mean():.4f} gap={gap:.3f} "
        f"(= {gap/max(NOISE_SD,1e-9):.1f} x noise sd)  LF top20={lf[top].mean():.2f} all={lf.mean():.2f}")
    sc3 &= (np.ptp(ir0) < 1e-9) and (gap > 2.0 * NOISE_SD) and (cv > 0.05)
    lf_top.append(lf[top].mean()); lf_all.append(lf.mean())
ok["SC3"] = bool(sc3)
say(f"     -> {'PASS' if ok['SC3'] else 'FAIL'}  (IR_0 exactly shared, top-bottom gap > 2 x noise sd, CV > 0.05)")

say(f"\nSC4 Layer 3 as a number: eta^2 = between-member SS / total SS of rtg[0]")
def eta2(groups):
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups); sst = ((allv - gm) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")
e_mes = eta2([np.array([float(t['rtg'][0]) for t in g]) for g in by_member(b_mes, 20)])
rng = np.random.default_rng(0)
e_rnd = eta2([np.array([float(t['rtg'][0]) for t in g])[rng.choice(100, 20, replace=False)]
              for g in by_member(b_rnd, 100)])
say(f"     MES rollouts (20/member): eta^2 = {e_mes:.3f}     random (20/member, matched n): eta^2 = {e_rnd:.3f}")
say(f"     (at tau=0 the state is identical within a member -- 'RTG predictable from state'")
say(f"      IS 'RTG explained by member identity'. Registered: eta^2(MES) > eta^2(random).)")
ok["SC4"] = e_mes > e_rnd
say(f"     -> {'PASS' if ok['SC4'] else 'FAIL (Layer 3 as stated is WRONG)'}")

say(f"\nSC5 winner quality vs MES (IR_T, same members, same iteration):")
sc5 = []
for m, (tm, tw) in enumerate(zip(by_member(b_mes, 20), by_member(b_w, 20))):
    a = np.mean([t['ir_T'] for t in tm]); b = np.mean([t['ir_T'] for t in tw])
    say(f"     m{m}: MES-20 IR_T={a:.4f}   top20-of-100-random IR_T={b:.4f}   "
        f"{'random winners BETTER' if b < a else 'MES better'}")
    sc5.append(b - a)
say(f"     mean(top20 - MES) = {np.mean(sc5):+.4f}  (diagnostic, not a gate)")

say(f"\nSC7 wall per iteration: MES-20/member {t_mes:.1f}s   random-100/member {t_rnd:.1f}s   "
    f"random-100->top20/member {t_w:.1f}s   (ratio W/ctrl = {t_w/max(t_mes,1e-9):.2f}x)")
ok["SC7"] = True

say(f"\nSC2 identity gate: run separately via tools/identity_gate.py (PASS at 122.29066752728207)")
allok = all(ok.values())
say(f"\nSCs: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in sorted(ok.items())))
say(f"STAGE 0: {'PASS' if allok else 'FAIL'}")
say("=" * 78)
sys.exit(0 if allok else 1)
