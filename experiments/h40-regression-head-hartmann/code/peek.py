"""Read whatever H40 has finished so far. Safe to run mid-grid."""
import os, json, glob
import numpy as np
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
rows=[json.load(open(f)) for f in glob.glob(os.path.join(R,"*__seed*.json"))]
if not rows:
    print("no completed jobs yet"); raise SystemExit
print(f"completed {len(rows)}/20\n")
print(f"{'arm':<22}{'n':>3}{'frozen':>9}{'mean impr':>11}{'min impr':>10}{'mean regret':>13}")
for cs,lab in [(True,"CANDIDATE scoring"),(False,"REGRESSION head")]:
    g=[r for r in rows if r["use_cs"]==cs]
    if not g: 
        print(f"{lab:<22}{0:>3}{'--':>9}"); continue
    imp=np.array([r["n_improvements"] for r in g]); reg=np.array([r["final_regret"] for r in g])
    print(f"{lab:<22}{len(g):>3}{sum(r['frozen'] for r in g):>4}/{len(g):<4}"
          f"{imp.mean():>11.2f}{imp.min():>10}{reg.mean():>13.4f}")
r=[x for x in rows if not x["use_cs"]]
if r:
    nf=sum(x["frozen"] for x in r)
    print(f"\nPRED 1 (regression arm improves on >=8/10): "
          f"{sum(1 for x in r if x['n_improvements']>0)}/{len(r)} so far")
    print(f"PRED 3 falsification (regression frozen on >=3/10): {nf}/{len(r)} so far "
          f"-> {'TRIPPED' if nf>=3 else 'not tripped'}")
    print(f"\nper-seed improvements  reg: {sorted((x['seed'],x['n_improvements']) for x in r)}")
c=[x for x in rows if x["use_cs"]]
if c: print(f"per-seed improvements   cs: {sorted((x['seed'],x['n_improvements']) for x in c)}")
