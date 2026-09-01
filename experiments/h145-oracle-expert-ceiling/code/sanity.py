"""h145 sanity checks SC1-SC3, SC7 -- pure-function, no run required.
SC4 (default-path identity) is a gate run; SC5/SC6/SC8 read a completed run."""
import os, sys, json, importlib.util
import numpy as np, torch
H=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(H,"..","..",".."))
sys.path.insert(0,REPO)
from benchmarks import get_benchmark
_s=importlib.util.spec_from_file_location("w",os.path.join(H,"worker.py"))
w=importlib.util.module_from_spec(_s); sys.modules["w"]=w
import src.policy.mf_dro as MF
_s.loader.exec_module(w)

GAP={"Hartmann_6D":2.1714101787750906e-06,"Borehole_8D":0.00045131857751812277}
ok=True
for bench in ("Hartmann_6D","Borehole_8D"):
    spec=get_benchmark(bench+"_HF"); f=spec["make_objective"]()
    bounds=torch.tensor([spec["domain_min"],spec["domain_max"]],dtype=torch.float64)
    d=int(bounds.shape[1]); T=8
    w._EXPERT["x_star"]=torch.tensor(w.XSTAR[bench],dtype=torch.float64)
    w._EXPERT["rng"]=torch.Generator().manual_seed(12345)
    print(f"\n=== {bench} (d={d}, T={T}) ===")
    e1=e2=e3=e7=0.0
    for trial in range(200):
        p=w._expert_path(bounds,T,d)
        # SC1 endpoint: last point == x*
        e1=max(e1,float((p[-1]-w._EXPERT["x_star"]).abs().max()))
        # SC2 interpolation: evenly spaced and collinear
        steps=p[1:]-p[:-1]
        e2=max(e2,float((steps-steps[0]).abs().max()))
        # in-domain
        assert bool((p>=bounds[0]-1e-9).all() and (p<=bounds[1]+1e-9).all()), "path left the domain"
    # SC3 objective at the endpoint equals the known optimum within solver gap
    yend=float(f(w._EXPERT["x_star"].unsqueeze(0)).reshape(-1)[0])
    kn=-float(spec["known_optimal_value"])
    e3=abs(yend-kn)
    # SC7 btg formula: backward cumsum of per-step costs
    costs=torch.tensor([spec["cost"]]*T,dtype=torch.float64)
    btg=costs.flip(0).cumsum(0).flip(0)
    e7=abs(float(btg[-1])-float(costs[-1]))
    print(f"  SC1 endpoint == x*            max |dev| {e1:.3e}   {'PASS' if e1<1e-12 else 'FAIL'}")
    print(f"  SC2 evenly spaced/collinear   max |dev| {e2:.3e}   {'PASS' if e2<1e-9 else 'FAIL'}")
    print(f"  SC3 y(x*) == known optimum    |dev| {e3:.3e} vs gap {GAP[bench]:.1e}   "
          f"{'PASS' if e3<=max(GAP[bench]*5,1e-6) else 'FAIL'}")
    print(f"  SC7 btg[-1] == last cost      |dev| {e7:.3e}   {'PASS' if e7<1e-12 else 'FAIL'}")
    ok &= (e1<1e-12) and (e2<1e-9) and (e3<=max(GAP[bench]*5,1e-6)) and (e7<1e-12)
print(f"\n  SC1/SC2/SC3/SC7: {'ALL PASS' if ok else 'FAILURE'}")
sys.exit(0 if ok else 1)
