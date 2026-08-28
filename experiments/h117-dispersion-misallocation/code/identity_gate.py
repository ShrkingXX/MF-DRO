"""GATE G0: prove the current working tree reproduces h83's MF-DRO path.

The tree carries an uncommitted h94 patch (roi_inference_mode hook, _roi_snap,
actions_x variance logging) and an h102 loc_loss selector. Both are inert BY
INSPECTION (getattr defaults, _roi_snap never called, roi_inference_mode set by
no config). Inspection is exactly what missed h94's NameError, so this executes
the MF-DRO path and compares against a stored trace.

Ackley_10D MF-DRO is the cheapest arm that exercises mf_dro.py (32 min vs 83).
"""
import os,subprocess,sys,json
H=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.join(H,"..","..","..",".venv","bin","python")
OUT=os.path.join(H,"..","results","IDENTITY__Ackley_10D__MF-DRO__seed42.json")
REF="experiments/h83-main-comparison/results/ckpt/Ackley_10D__MF-DRO__seed42.json"
if not os.path.exists(OUT):
    r=subprocess.run([PY_,"-u",os.path.join(H,"worker.py"),"Ackley_10D","MF-DRO","42"],
                     capture_output=True,text=True,cwd=os.path.join(H,"..","..",".."))
    src=os.path.join(H,"..","results","Ackley_10D__MF-DRO__seed42.json")
    if os.path.exists(src): os.replace(src,OUT)
    else: print(f"[GATE ERROR] worker produced nothing\n{r.stderr[-1500:]}"); sys.exit(2)
a=json.load(open(OUT)); b=json.load(open(os.path.join(H,"..","..","..",REF)))
qa,qb=a["queries"],b["queries"]
if len(qa)!=len(qb): print(f"[GATE FAIL] length {len(qa)} vs {len(qb)}"); sys.exit(1)
bad=[i for i,(u,v) in enumerate(zip(qa,qb))
     if u["fid"]!=v["fid"] or u["x"]!=v["x"] or u["y"]!=v["y"]]
print(f"[GATE {'PASS' if not bad else 'FAIL'}] {len(qa)} queries, {len(bad)} differing"
      + (f"; first at i={bad[0]}" if bad else ""))
sys.exit(0 if not bad else 1)
