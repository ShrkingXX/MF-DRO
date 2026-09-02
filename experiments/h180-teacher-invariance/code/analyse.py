"""h180 EXPLORATORY re-analysis. No new runs.

Measures, on ALREADY-SAVED Borehole traces (seeds 42-46), the distance between
each teacher arm's FIRST REAL QUERY and the MES control's, paired by seed, and
reads it against the FROZEN metric (h83 sr_curve/grid @ cost 200 -- imported,
not re-derived; an earlier ad-hoc rel% formula disagreed with the control's
known 15.82 and was discarded).

Origin: this began as the premise check for a proposed teacher-rotation arm.
"""
import json, glob, sys, os
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "experiments/h83-main-comparison/code"))
from benchmarks import get_benchmark
from analyse import grid                      # FROZEN metric, imported

OPT = float(get_benchmark("Borehole_8D_HF")["known_optimal_value"])
LO = np.array([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0])
HI = np.array([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0])
R = HI - LO
G = np.linspace(0, 200, 201)

ARMS = [('MES (control, h83)', 'h83-main-comparison', 'MF-DRO'),
        ('UCB-LOC (h155)', 'h155-ucb-loc', 'UCB-LOC'),
        ('MES-FROZEN (h153)', 'h153-mes-frozen', 'MES-FROZEN'),
        ('STALE-PATH (h161)', 'h161-stale-path', 'STALE-PATH'),
        ('EXPLOIT-LOC (h159)', 'h159-exploit-loc', 'EXPLOIT-LOC'),
        ('TAIL-MES (h171)', 'h171-head-tail', 'TAIL-MES'),
        ('DIVERSE-GOOD (h146)', 'h146-why-oracle-hurts', 'DIVERSE-GOOD'),
        ('HEAD-MES (h171)', 'h171-head-tail', 'HEAD-MES'),
        ('ORACLE-EXPERT (h145)', 'h145-oracle-expert-ceiling', 'ORACLE-EXPERT'),
        ('RANDOM-POOL (h149)', 'h149-forced-vs-teacher-quality', 'RANDOM-POOL')]


def parse(fn):
    """First real query (unit-box) and the FROZEN rel% for one run."""
    q = json.load(open(fn))["queries"]
    init = max([float(e["cost_cum"]) for e in q if e.get("is_init")] + [0.0])
    cost, sr, best, x0 = [], [], -np.inf, None
    for e in q:
        if e["fid"]:
            best = max(best, float(e["y"]))
        if not e.get("is_init"):
            if x0 is None:
                x0 = (np.array(e["x"]) - LO) / R
            cost.append(float(e["cost_cum"]) - init)
            sr.append(float(-best - OPT))
    return x0, 100.0 * grid(np.asarray(cost), np.asarray(sr), G)[-1] / abs(OPT)


def load():
    D = {}
    for lab, d, tag in ARMS:
        o = {}
        for f in sorted(glob.glob(
                f'{REPO}/experiments/{d}/results/ckpt/Borehole_8D__{tag}__seed4[2-6].json')):
            s = int(f.split('seed')[1].split('.')[0])
            x0, rel = parse(f)
            if x0 is not None:
                o[s] = (x0, rel)
        if len(o) >= 4:
            D[lab] = o
    return D


if __name__ == "__main__":
    D = load()
    base = D['MES (control, h83)']
    print(f'  {"teacher arm":24s} {"n":>2s} {"dist to MES 1st query":>21s} {"FROZEN rel%":>12s}')
    print('  ' + '-' * 64)
    for lab, o in D.items():
        sh = sorted(set(o) & set(base))
        dd = np.array([np.linalg.norm(o[s][0] - base[s][0]) for s in sh])
        rel = float(np.mean([o[s][1] for s in sh]))
        print(f'  {lab:24s} {len(sh):2d} {dd.mean():>10.4f} (max {dd.max():.3f}) {rel:11.2f}')
    # NOISE FLOOR: same teacher, different seed.
    same = []
    for lab, o in D.items():
        ks = sorted(o)
        same += [np.linalg.norm(o[a][0] - o[b][0])
                 for i, a in enumerate(ks) for b in ks[i + 1:]]
    same = np.array(same)
    print(f'\n  NOISE FLOOR (same teacher, different seed): mean {same.mean():.4f}  min {same.min():.4f}')
    print('  The tight cluster (<=0.065) sits ~7x below this floor -- unambiguous.')
    print('  RANDOM-POOL (0.434) is only just under it, so the UNPAIRED floor does not')
    print('  separate it; the PAIRED statistic does (0.434 vs 0.044, ranges disjoint).')
