"""Repair MF-MES traces written before the is_init fix.

Those runs recorded ONLY optimization queries (the initial design is drawn
through mf_dro's internal path, not the wrapped objectives), yet rec() labelled
the first n_init of them as init. The queries themselves are correct and
complete for the optimization phase; only the flag is wrong, and the initial
design is absent. This sets is_init=False everywhere and records that the init
is missing, rather than fabricating it.
"""
import json,glob,os,sys
R=os.path.join(os.path.dirname(__file__),"..","results")
n=0
for f in sorted(glob.glob(os.path.join(R,"*MF-MES*.json"))):
    d=json.load(open(f))
    q=d.get("queries") or []
    if not q or d.get("_trace_repaired"): continue
    if not any(e.get("is_init") for e in q): continue
    for e in q: e["is_init"]=False
    d["_trace_repaired"]="is_init pinned False; initial design NOT in this trace"
    tmp=f+".tmp"; json.dump(d,open(tmp,"w"),default=float); os.replace(tmp,f)
    print(f"  repaired {os.path.basename(f)}  ({len(q)} queries)"); n+=1
print(f"{n} file(s) repaired")
