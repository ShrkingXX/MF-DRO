import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[v]="1"
import json, numpy as np
CK="experiments/h84-roi-strategy/results/ckpt"; B="Borehole_8D"; S=[42,43]   # AMENDMENT 1: h84 ROI-OFF exists only at seeds 42-43. n=2, DESCRIPTIVE ONLY.
OFF,ON="ROI-OFF","ROI-Q10"
def parts(a,s):
    d=json.load(open(f"{CK}/{B}__{a}__seed{s}.json")); Q=d["queries"]
    return (Q,[q for q in Q if q["fid"]==1 and q.get("is_init")],
              [q for q in Q if q["fid"]==1 and not q.get("is_init")],
              [q for q in Q if q["fid"]==0 and not q.get("is_init")])
def eff(d):
    d=np.asarray(d,float)
    return abs(d.mean())/d.std(ddof=1) if len(d)>1 and d.std(ddof=1)>0 else float("nan")
def line(tag,name,a,b,lower_is_pred,gate=True):
    d=np.array(b)-np.array(a); e=eff(d)
    n=int((d<0).sum()) if lower_is_pred else int((d>0).sum())
    ok=None  # n=2: no gate verdict is issued (Amendment 1)
    print(f"  {tag} {name:38s} OFF {np.mean(a):8.3f}  Q10 {np.mean(b):8.3f}  paired {d.mean():+8.3f}"
          f"  sd {d.std(ddof=1):6.3f}  |m|/sd {e:6.2f}  {n}/{len(d)}  {'n=2 no verdict'}")
    return ok
hf_o=[];hf_n=[];lf_o=[];lf_n=[];t_o=[];t_n=[];q_o=[];q_n=[];u_o=[];u_n=[];w_o=[];w_n=[]
for s in S:
    Qo,io,ho,lo_=parts(OFF,s); Qn,inn,hn,ln=parts(ON,s)
    hf_o.append(len(ho)); hf_n.append(len(hn)); lf_o.append(len(lo_)); lf_n.append(len(ln))
    for (Q,h,tl) in ((Qo,ho,t_o),(Qn,hn,t_n)):
        tot=max(q["cost_cum"] for q in Q); best=max(q["y"] for q in h)
        tl.append(next(q["cost_cum"] for q in h if q["y"]>=best)/tot)
    K=min(len(ho),len(hn))
    q_o.append(np.mean([q["y"] for q in ho[:K]])); q_n.append(np.mean([q["y"] for q in hn[:K]]))
    u_o.append(np.mean([q["y"] for q in ho]));      u_n.append(np.mean([q["y"] for q in hn]))
    b0o=max(q["y"] for q in io); b0n=max(q["y"] for q in inn)
    w_o.append(np.mean([q["y"]<b0o for q in ho]));  w_n.append(np.mean([q["y"]<b0n for q in hn]))
print(f"  per-seed HF counts  OFF {hf_o}  Q10 {hf_n}")
print(f"  per-seed LF counts  OFF {lf_o}  Q10 {lf_n}")
print(f"  count-match K per seed: {[min(a,b) for a,b in zip(hf_o,hf_n)]}\n")
p1=line("P1","HF query count (predict LOWER)",hf_o,hf_n,True)
line("P1b","LF query count (predict HIGHER)",lf_o,lf_n,False,gate=False)
p2=line("P2","time-to-incumbent (predict LOWER)",t_o,t_n,True,gate=False)
p3=line("P3","COUNT-MATCHED mean HF y (HIGHER)",q_o,q_n,False)
line("--","uncounted mean HF y (confounded)",u_o,u_n,False,gate=False)
p4=line("P4","frac HF worse than init (control)",w_o,w_n,True)
print('\n  NO VERDICT ISSUED: n=2 cannot evaluate criteria requiring >=4/5 seeds (Amendment 1).')
