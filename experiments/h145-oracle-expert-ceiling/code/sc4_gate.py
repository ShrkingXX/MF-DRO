"""SC4: with forced_x=None the tree must reproduce h83's stored trace exactly."""
import os,sys,importlib.util
H=os.path.dirname(os.path.abspath(__file__)); REPO=os.path.abspath(os.path.join(H,"..","..",".."))
sys.path.insert(0,REPO)
_s=importlib.util.spec_from_file_location("h83w",os.path.join(REPO,"experiments/h83-main-comparison/code/worker.py"))
h83=importlib.util.module_from_spec(_s); sys.modules["h83w"]=h83; _s.loader.exec_module(h83)
RES=os.path.join(H,"..","results","sc4"); h83.RES=RES
b,seed="Borehole_8D",42
r=h83.run(b,"MF-DRO",seed,os.path.join(RES,"ckpt",f"{b}__MF-DRO__seed{seed}.json"))
h83._atomic(os.path.join(RES,f"{b}__MF-DRO__seed{seed}.json"),r)
print(f"[sc4 done] regret={r['final_regret']:.6f}",flush=True)
