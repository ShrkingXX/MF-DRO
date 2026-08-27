"""Combined h83-vs-ROI table: does the calibrated ROI change h83's verdict?

Every number is computed with h83's OWN sr_curve/grid functions, imported
directly, so all methods sit on one metric definition: simple regret
grid-interpolated at exactly cost 200. An earlier throwaway version of this
table mixed `final_regret` (SR where a run actually ended, cost possibly >200)
for MF-DRO with grid-at-200 for the baselines, which read h83's own MF-DRO as
7.55 instead of 7.99 and overstated the ROI's effect. Do not reintroduce that.

The MF-DRO+ROI arm is assembled from two experiments: h84 already ran
Hartmann/Borehole under this exact configuration, h86 ran Currin/Ackley.
"""
import sys, json, os
import numpy as np
REPO=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"experiments","h83-main-comparison","code"))
from analyse import sr_curve, grid
from benchmarks import get_benchmark

G=np.linspace(0,200,201)
H83=os.path.join(REPO,'experiments','h83-main-comparison','results')
ROI={"Hartmann_6D":os.path.join(REPO,'experiments','h84-roi-strategy','results','{b}__ROI-Q10__seed{s}.json'),
     "Borehole_8D":os.path.join(REPO,'experiments','h84-roi-strategy','results','{b}__ROI-Q10__seed{s}.json'),
     "Currin_2D":  os.path.join(REPO,'experiments','h86-roi-full','results','{b}__ROI-Q10__seed{s}.json'),
     "Ackley_10D": os.path.join(REPO,'experiments','h86-roi-full','results','{b}__ROI-Q10__seed{s}.json')}
BENCH=("Currin_2D","Hartmann_6D","Borehole_8D","Ackley_10D")
BASE=("SF-DRO","MF-MES","MF-MI-Greedy","MF-GP-UCB")
SEEDS=(42,43,44,45,46)

def at200(b,p):
    opt=float(get_benchmark(f"{b}_HF")["known_optimal_value"])
    c,sr=sr_curve(json.load(open(p)),opt); v=grid(c,sr,G)[-1]
    return v if abs(opt)<1e-9 else 100.0*v/abs(opt)

if __name__=="__main__":
    print("Simple regret at cost 200, h83's own metric. Ackley is ABSOLUTE SR "
          "(its optimum is 0).\n")
    print(f"  {'benchmark':13s}{'n':>3s}{'MF-DRO h83':>12s}{'+ROI-Q10':>10s}{'delta':>8s}"
          f"{'wins':>7s}   best baseline          verdict")
    flips=[]
    for b in BENCH:
        base={}
        for m in BASE:
            f=[f'{H83}/{b}__{m}__seed{s}.json' for s in SEEDS]
            if all(os.path.exists(x) for x in f):
                v=[at200(b,x) for x in f]; base[m]=(float(np.mean(v)),v)
        bn=min(base,key=lambda k:base[k][0]); bv,bseeds=base[bn]
        old=[];new=[];idx=[]
        for i,s in enumerate(SEEDS):
            p83=f'{H83}/{b}__MF-DRO__seed{s}.json'; pn=ROI[b].format(b=b,s=s)
            if os.path.exists(p83) and os.path.exists(pn):
                old.append(at200(b,p83)); new.append(at200(b,pn)); idx.append(i)
        if not new:
            print(f"  {b:13s}{0:>3d}{'--':>12s}{'pending':>10s}{'':>8s}{'':>7s}   {bn} {bv:.2f}")
            continue
        wins=sum(1 for k,v in zip(idx,new) if v<bseeds[k])
        beats=(np.mean(new)<bv and wins>=4)
        verdict=("BEATS BASELINE" if beats else
                 "mean below, seeds short" if np.mean(new)<bv else "no")
        if beats: flips.append(b)
        print(f"  {b:13s}{len(new):>3d}{np.mean(old):>11.2f}{np.mean(new):>10.2f}"
              f"{np.mean(new)-np.mean(old):>+8.2f}{wins:>4d}/{len(new)}   {bn} {bv:.2f}"
              f"   {verdict}")
    print(f"\n  'BEATS BASELINE' = h83's own bar: strictly lower mean AND >= 4/5 paired seed wins.")
    print(f"  h83's PRIMARY finding was that MF-DRO beats no baseline on ANY benchmark.")
    print(f"  With the calibrated ROI it beats the best baseline on: "
          f"{', '.join(flips) if flips else 'none'}.")
    print("\n  CAVEATS (findings.md): one benchmark of four; selection over three ROI")
    print("  settings on Hartmann; post-hoc use of a bar registered for a different")
    print("  configuration (h84's own P1 FAILED); n=5 with no p-values.")
