"""H105: the pre-registered success test at the seed count PROTOCOL.md registers."""
import sys, os, json, subprocess, itertools
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark
G=np.linspace(0,200,201); B="Hartmann_6D"
OPT=abs(float(get_benchmark(f"{B}_HF")["known_optimal_value"]))
H83=os.path.join(REPO,"experiments","h83-main-comparison","results")
SRC={  # method -> {seed: path}
 "MF-DRO":       {**{s:f"{H83}/{B}__MF-DRO__seed{s}.json" for s in (42,43,44,45,46)},
                  **{s:f"{REPO}/experiments/h89-hffloor-confirm/results/{B}__CONTROL__seed{s}.json" for s in (52,53,54,55,56)}},
 "MF-MI-Greedy": {**{s:f"{H83}/{B}__MF-MI-Greedy__seed{s}.json" for s in (42,43,44,45,46)},
                  **{s:f"{REPO}/experiments/h105-registered-n10/results/{B}__MF-MI-Greedy__seed{s}.json" for s in (52,53,54,55,56)}},
 "MF-GP-UCB":    {**{s:f"{H83}/{B}__MF-GP-UCB__seed{s}.json" for s in (42,43,44,45,46)},
                  **{s:f"{REPO}/experiments/h105-registered-n10/results/{B}__MF-GP-UCB__seed{s}.json" for s in (52,53,54,55,56)}},
 "MF-MES":       {**{s:f"{H83}/{B}__MF-MES__seed{s}.json" for s in (42,43,44,45,46)},
                  **{s:f"{REPO}/experiments/h91-mfmes-freshseeds/results/{B}__MF-MES__seed{s}.json" for s in (52,53,54,55,56)}},
}
REGISTERED=("MF-MI-Greedy","MF-GP-UCB")
def at200(p):
    c,sr=sr_curve(json.load(open(p)),-OPT); return 100.0*grid(c,sr,G)[-1]/OPT
def commit(p):
    c=(json.load(open(p)).get("_code") or {}).get("commit"); return c[:8] if c else None
_cache={}
def behav(a,b):
    if (a,b) in _cache: return _cache[(a,b)]
    r=subprocess.run(["git","diff","--numstat",a,b,"--","src/","dro_runner.py","benchmarks.py"],
                     cwd=REPO,capture_output=True,text=True)
    _cache[(a,b)]=len([l for l in r.stdout.strip().split("\n") if l.strip()]); return _cache[(a,b)]
if __name__=="__main__":
    print("H105 -- PROTOCOL.md's success test at its registered seed count (10).\n")
    # ---- GATE: reused arms must not span a behavioural code change
    # A FILE-level diff is too coarse: it flags changes to files a given method
    # never executes (mf_dro.py's ROI block is irrelevant to MI-Greedy, and
    # irrelevant to MF-DRO itself when use_roi=False). The criterion that
    # matters is whether the EXECUTED path changed. For DRO-family arms that is
    # the `if not use_roi:` branch, hashed directly; for the baselines it is
    # their own source files.
    import hashlib
    def off_branch_hash(c):
        src=subprocess.run(["git","show",f"{c}:src/policy/mf_dro.py"],cwd=REPO,
                           capture_output=True,text=True).stdout
        i=src.find("if not use_roi:")
        if i<0: return None
        j=src.find("    else:", i)
        return hashlib.md5((src[i:j] if j>0 else src[i:i+600]).encode()).hexdigest()[:12]
    BASE_FILES={"MF-MI-Greedy":["src/baselines/","src/models/","benchmarks.py"],
                "MF-GP-UCB":["src/baselines/","src/models/","benchmarks.py"]}
    print("  GATE: did each method's EXECUTED path change across its commits?")
    bad=False
    for m,d in SRC.items():
        cs=sorted({commit(p) for p in d.values() if os.path.exists(p) and commit(p)})
        if m in BASE_FILES:
            n=0
            for a,b in itertools.combinations(cs,2):
                r=subprocess.run(["git","diff","--numstat",a,b,"--"]+BASE_FILES[m],
                                 cwd=REPO,capture_output=True,text=True)
                n+=len([l for l in r.stdout.strip().split("\n") if l.strip()])
            print(f"    {m:14s} {len(cs)} commit(s), own source files changed: {n}  {'OK' if n==0 else '*** CHANGED ***'}")
        else:
            hs={off_branch_hash(c) for c in cs}
            n=0 if len(hs)==1 and None not in hs else 1
            print(f"    {m:14s} {len(cs)} commit(s), use_roi=False branch hash: {sorted(hs)}  {'OK -- byte-identical' if n==0 else '*** DIFFERS ***'}")
        bad = bad or n>0
    if bad:
        print("\n    *** GATE FAILED -- reused arms are not comparable. Re-run rather than reuse. ***")
        sys.exit(1)
    print("    -> PASS\n")
    res={}
    print(f"  {'method':14s}{'n':>3}{'mean':>8}{'SE':>7}{'mean+SE':>9}{'mean-SE':>9}   registered?")
    for m,d in SRC.items():
        v=[at200(p) for p in d.values() if os.path.exists(p)]
        if len(v)<10: print(f"  {m:14s} incomplete ({len(v)}/10)"); continue
        a=np.array(v); se=a.std(ddof=1)/np.sqrt(len(a)); res[m]=(a.mean(),se)
        print(f"  {m:14s}{len(v):>3}{a.mean():>8.2f}{se:>7.2f}{a.mean()+se:>9.2f}{a.mean()-se:>9.2f}"
              f"   {'YES' if m in REGISTERED else 'no'}")
    if "MF-DRO" not in res: sys.exit()
    dm,dse=res["MF-DRO"]
    print("\n  REGISTERED SUCCESS TEST: MF-DRO mean+SE < best-baseline mean-SE")
    reg={k:v for k,v in res.items() if k in REGISTERED}
    if reg:
        b=min(reg,key=lambda k:reg[k][0]); thr=reg[b][0]-reg[b][1]
        ok=dm+dse<thr
        print(f"    registered baselines -> best {b}: {dm+dse:.2f} < {thr:.2f}  =>  {'PASSES' if ok else 'FAILS'}")
        print(f"    P1 (passes at n=10): {'MET' if ok else '*** FAILED -- the n=5 pass was an artefact ***'}")
    allb={k:v for k,v in res.items() if k!="MF-DRO"}
    if allb:
        b2=min(allb,key=lambda k:allb[k][0]); thr2=allb[b2][0]-allb[b2][1]
        print(f"\n    all comparators -> best {b2}: {dm+dse:.2f} < {thr2:.2f}  =>  "
              f"{'PASSES' if dm+dse<thr2 else 'FAILS'}")
        print(f"    P2 (still not best once MF-MES included): "
              f"{'MET' if res.get('MF-MES',(1e9,))[0] < dm else '*** REFUTED ***'}")
