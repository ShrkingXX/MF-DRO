"""h209 readout. Frozen metric imported from h83's grid; finals only; endpoint only.
Scores P-Q, P-H (with the registered LF discriminator), P-B47 as each lands."""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid
G = np.linspace(0, 200, 201); BAND = 1.26
def rel(fn, b):
    OPT = float(get_benchmark(f"{b}_HF")["known_optimal_value"]); q = json.load(open(fn))["queries"]
    ini = [float(e["cost_cum"]) for e in q if e.get("is_init")]
    init = max(ini) if ini else float(q[0]["cost_cum"]) - (2.0 if q[0]["fid"] else 1.0)
    c, s, bb = [], [], -np.inf
    for e in q:
        if e["fid"]: bb = max(bb, float(e["y"]))
        if not e.get("is_init") and float(e["cost_cum"]) > init: c.append(float(e["cost_cum"]) - init); s.append(float(-bb - OPT))
    return 100.0 * grid(np.asarray(c), np.asarray(s), G)[-1] / abs(OPT)
def arm(pat, b): return {int(f.split("seed")[1].split(".")[0]): rel(f, b) for f in sorted(glob.glob(pat))}
def lf(pat): return {int(f.split("seed")[1].split(".")[0]): json.load(open(f)).get("lf_fraction") for f in sorted(glob.glob(pat))}
def line(nm, a):
    if not a: return f"    {nm:30s} {'--':>7s}  0/5"
    return f"    {nm:30s} {np.mean(list(a.values())):7.2f}  {len(a)}/5   " + "  ".join(f"{k}:{v:.2f}" for k, v in sorted(a.items()))
def paired(x, xn, y, yn):
    sh = sorted(set(x) & set(y))
    if len(sh) < 2: return None
    d = [x[s] - y[s] for s in sh]
    print(f"    {xn} - {yn}  {np.mean(d):+7.2f}   se {np.std(d, ddof=1)/np.sqrt(len(d)):5.2f}   {xn} better on {sum(1 for v in d if v < 0)}/{len(d)}   per-seed {[round(v,2) for v in d]}")
    return float(np.mean(d)), sum(1 for v in d if v < 0)
E = f"{REPO}/experiments"; RR = f"{E}/h209-mixr-replication/results"
H = "Hartmann_6D"; B = "Borehole_8D"
MH = arm(f"{RR}/{H}__H209MIXR__seed4[2-6].json", H); NH = arm(f"{RR}/{H}__H209NIR__seed4[2-6].json", H)
ROI_H = arm(f"{E}/h84-roi-strategy/results/{H}__ROI-Q10__seed4[2-6].json", H); MES_H = arm(f"{E}/h83-main-comparison/results/{H}__MF-MES__seed4[2-6].json", H)
N120 = arm(f"{RR}/{B}__H209NIR120__seed4[2-6].json", B)
MB47 = arm(f"{RR}/{B}__H209MIXR__seed4[7-9].json", B); MB47.update(arm(f"{RR}/{B}__H209MIXR__seed5[01].json", B))
NB47 = arm(f"{RR}/{B}__H209NIR__seed4[7-9].json", B); NB47.update(arm(f"{RR}/{B}__H209NIR__seed5[01].json", B))
MIXR_B = arm(f"{E}/h207-random-rollout-dataset/results/{B}__H207MIXR__seed4[2-6].json", B)
NIR_B = arm(f"{E}/h207-random-rollout-dataset/results/{B}__H207NIR-MES-IR__seed4[2-6].json", B)

print(f"\n  P-H -- HARTMANN 42-46 (frozen rel% @ cost 200; |OPT| 3.32)\n")
print(line("MIXR-H  20 MES + 20 random", MH)); print(line("NIR-H   20 MES", NH))
print(line("h84 ROI-Q10 K=1 (ref)", ROI_H)); print(line("h83 MF-MES (ref)", MES_H))
if MH and NH:
    d, nb = paired(MH, "MIXR-H", NH, "NIR-H")
    lfm, lfn = lf(f"{RR}/{H}__H209MIXR__seed4[2-6].json"), lf(f"{RR}/{H}__H209NIR__seed4[2-6].json")
    print(f"    LF fraction  MIXR-H {np.mean(list(lfm.values())):.3f}  NIR-H {np.mean(list(lfn.values())):.3f}   per-seed drop: "
          + "  ".join(f"{s}:{lfm[s]-lfn[s]:+.2f}" for s in sorted(lfm)))
    if d < -BAND and nb >= 4: print("    -> P-H SUPPORTED: replicates on Hartmann")
    elif d > BAND and np.mean(list(lfm.values())) < np.mean(list(lfn.values())) - 0.05:
        print("    -> P-H FAILED with the REGISTERED LF SIGNATURE: MIXR worse AND LF dropped -> fidelity-head mechanism, Borehole-specific")
    elif d > BAND: print("    -> P-H FAILED (worse), without the LF signature")
    else: print("    -> P-H: inside the band -- no replication, no reversal")

print(f"\n  P-Q -- BOREHOLE 42-46 data-quantity control\n")
print(line("NIR120-B  40 MES/member", N120)); print(line("NIR-B (h207) 20 MES", NIR_B)); print(line("MIXR-B (h207) 20+20", MIXR_B))
if N120 and NIR_B and MIXR_B:
    d1, _ = paired(N120, "NIR120", NIR_B, "NIR-B"); d2, nb2 = paired(MIXR_B, "MIXR-B", N120, "NIR120")
    print("    -> " + ("P-Q SUPPORTED: quantity does not explain MIXR" if abs(d1) <= BAND and d2 < -BAND
                       else "P-Q FAILED: NIR120 within band of MIXR -> h207 headline RETRACTED as data volume" if abs(float(np.mean(list(MIXR_B.values()))) - float(np.mean(list(N120.values())))) <= BAND
                       else "P-Q: mixed -- see numbers"))

print(f"\n  P-B47 -- BOREHOLE fresh seeds 47-51\n")
print(line("MIXR-B47", MB47)); print(line("NIR-B47", NB47))
if MB47 and NB47 and len(set(MB47) & set(NB47)) >= 2:
    d, nb = paired(MB47, "MIXR-B47", NB47, "NIR-B47")
    print("    -> " + ("P-B47 SUPPORTED" if d < -BAND and nb >= 4 else "P-B47 FAILED" if d >= -BAND else "P-B47: better but < 4/5"))
print()
