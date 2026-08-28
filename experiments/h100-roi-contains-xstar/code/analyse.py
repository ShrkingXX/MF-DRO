"""H100: does the ROI region contain x* where it helps? Zero compute."""
import os,sys,json,glob
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO)
SRC={"Borehole_8D":("experiments/h90-borehole-confirm/results","ROI-Q10",(47,48,49,50,51),8),
     "Hartmann_6D":("experiments/h84-roi-strategy/results","ROI-Q10",(42,43,44,45,46),6),
     "Ackley_10D":("experiments/h86-roi-full/results","ROI-Q10",(42,43,44,45,46),10),
     "Currin_2D":("experiments/h86-roi-full/results","ROI-Q10",(42,43,44,45,46),2)}
EFFECT={"Borehole_8D":"WORKED -3.49","Hartmann_6D":"failed (withdrawn)",
        "Ackley_10D":"negligible -0.09","Currin_2D":"saturated ~0"}
def uniform_ref(d,N=600,trials=400,seed=3):
    """Expected min distance from N uniform points to a fixed target, at this d.
    The baseline the protocol requires beside every cross-benchmark number."""
    rng=np.random.RandomState(seed); t=rng.rand(d); out=[]
    for _ in range(trials):
        P=rng.rand(N,d); out.append(np.linalg.norm(P-t,axis=1).min())
    return float(np.mean(out))
print("H100 -- does the ROI region contain x*?\n")
print(f"  {'benchmark':<14}{'d':>3}{'min_dist':>10}{'uniform ref':>13}{'ratio':>8}{'frac<0.2':>10}  ROI outcome")
rows={}
for B,(root,arm,seeds,d) in SRC.items():
    md=[];fr=[]
    for s in seeds:
        p=os.path.join(REPO,root,f"{B}__{arm}__seed{s}.json")
        if not os.path.exists(p): continue
        rs=json.load(open(p)).get("roi_summary") or {}
        if rs.get("min_dist_to_xstar") is not None: md.append(rs["min_dist_to_xstar"])
        if rs.get("frac_within_02") is not None: fr.append(rs["frac_within_02"])
    if not md: print(f"  {B:<14} no data"); continue
    ref=uniform_ref(d)
    rows[B]=(np.mean(md),np.mean(fr) if fr else float('nan'),ref,d)
    print(f"  {B:<14}{d:>3}{np.mean(md):>10.4f}{ref:>13.4f}{np.mean(md)/ref:>8.2f}"
          f"{(np.mean(fr) if fr else float('nan')):>10.4f}  {EFFECT[B]}")
print("\n  'ratio' = ROI's min_dist / the min_dist a UNIFORM 600-point draw would give at that d.")
print("  <1 means the ROI is CLOSER to x* than an unfiltered pool; >1 means FARTHER.\n")
print("  REGISTERED BARS")
if "Borehole_8D" in rows and "Hartmann_6D" in rows:
    b=rows["Borehole_8D"]; h=rows["Hartmann_6D"]
    q1 = (b[0]/b[2]) < (h[0]/h[2]) and b[1] > h[1]
    print(f"    Q1 PRIMARY (Borehole's ROI closer to x* than Hartmann's, ratio AND frac):")
    print(f"       ratio  Borehole {b[0]/b[2]:.2f} vs Hartmann {h[0]/h[2]:.2f}")
    print(f"       frac   Borehole {b[1]:.4f} vs Hartmann {h[1]:.4f}")
    print(f"       -> {'MET' if q1 else 'FAILED'}")
    if not q1:
        print("\n    *** Q3 APPLIES. The surrogate-quality hypothesis is DEAD. Two")
        print("        candidate mechanisms (headroom, containment) are now eliminated")
        print("        by measurement, and h96's four-benchmark relocation pattern has")
        print("        NO surviving explanation. State that. ***")
if len(rows)==4:
    order=sorted(rows,key=lambda k:-rows[k][1])
    print(f"\n    Q2 (frac_within_0.2 ordering vs regret benefit): {order}")
    print(f"       expected best = Borehole_8D -> {'MET' if order[0]=='Borehole_8D' else 'FAILED'}")
    print("       (n=4 benchmarks, ordering only, no p-value -- and H99's ordering test")
    print("        already failed once at this power)")
