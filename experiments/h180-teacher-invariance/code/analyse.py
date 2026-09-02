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

BENCH = sys.argv[1] if len(sys.argv) > 1 else "Borehole_8D"
_B = get_benchmark(f"{BENCH}_HF")
OPT = float(_B["known_optimal_value"])
LO = np.asarray(_B["domain_min"], dtype=float)
HI = np.asarray(_B["domain_max"], dtype=float)
R = HI - LO
G = np.linspace(0, 200, 201)

ARMS_BY_BENCH = {
    "Borehole_8D": [
        ('MES (control, h83)', 'h83-main-comparison', 'MF-DRO'),
        ('UCB-LOC (h155)', 'h155-ucb-loc', 'UCB-LOC'),
        ('MES-FROZEN (h153)', 'h153-mes-frozen', 'MES-FROZEN'),
        ('STALE-PATH (h161)', 'h161-stale-path', 'STALE-PATH'),
        ('EXPLOIT-LOC (h159)', 'h159-exploit-loc', 'EXPLOIT-LOC'),
        ('TAIL-MES (h171)', 'h171-head-tail', 'TAIL-MES'),
        ('DIVERSE-GOOD (h146)', 'h146-why-oracle-hurts', 'DIVERSE-GOOD'),
        ('HEAD-MES (h171)', 'h171-head-tail', 'HEAD-MES'),
        ('ORACLE-EXPERT (h145)', 'h145-oracle-expert-ceiling', 'ORACLE-EXPERT'),
        ('RANDOM-POOL (h149)', 'h149-forced-vs-teacher-quality', 'RANDOM-POOL')],
    # REPLICATION set. Only arms that exist on Hartmann with matched seeds; the
    # rule-varying arms are UCB-LOC and MES-FROZEN (STALE-PATH/EXPLOIT-LOC/
    # DIVERSE-GOOD were never run here), so the tight cluster is thinner.
    "Hartmann_6D": [
        ('MES (control, h83)', 'h83-main-comparison', 'MF-DRO'),
        ('UCB-LOC (h165)', 'h165-hartmann-ucbloc', 'UCB-LOC'),
        ('MES-FROZEN (h166)', 'h166-hartmann-frozen', 'MES-FROZEN'),
        ('TAIL-MES (h173)', 'h173-head-tail-hartmann', 'TAIL-MES'),
        ('HEAD-MES (h173)', 'h173-head-tail-hartmann', 'HEAD-MES'),
        ('ORACLE-EXPERT (h145)', 'h145-oracle-expert-ceiling', 'ORACLE-EXPERT'),
        ('RANDOM-POOL (h149)', 'h149-forced-vs-teacher-quality', 'RANDOM-POOL')],
}
ARMS = ARMS_BY_BENCH[BENCH]


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
                f'{REPO}/experiments/{d}/results/ckpt/{BENCH}__{tag}__seed4[2-6].json')):
            s = int(f.split('seed')[1].split('.')[0])
            x0, rel = parse(f)
            if x0 is not None:
                o[s] = (x0, rel)
        if len(o) >= 4:
            D[lab] = o
    return D


if __name__ == "__main__":
    D = load()
    print(f'  === {BENCH} ===')
    base = D['MES (control, h83)']
    print(f'  {"teacher arm":24s} {"n":>2s} {"dist to MES 1st query":>21s} {"FROZEN rel%":>12s}')
    print('  ' + '-' * 64)
    stats = []
    for lab, o in D.items():
        sh = sorted(set(o) & set(base))
        dd = np.array([np.linalg.norm(o[s][0] - base[s][0]) for s in sh])
        rel = float(np.mean([o[s][1] for s in sh]))
        stats.append((lab, float(dd.mean()), rel))
        print(f'  {lab:24s} {len(sh):2d} {dd.mean():>10.4f} (max {dd.max():.3f}) {rel:11.2f}')
    # NOISE FLOOR: same teacher, different seed.
    same = []
    for lab, o in D.items():
        ks = sorted(o)
        same += [np.linalg.norm(o[a][0] - o[b][0])
                 for i, a in enumerate(ks) for b in ks[i + 1:]]
    same = np.array(same)
    print(f'\n  NOISE FLOOR (same teacher, different seed): mean {same.mean():.4f}  min {same.min():.4f}')
    tight = [d for _, d, _ in stats if d <= 0.10 and d > 0]
    far = [d for _, d, _ in stats if d > 0.10]
    if tight:
        print(f'  tight cluster (rule-varying) max {max(tight):.4f} -> '
              f'{same.min() / max(tight):.1f}x below the floor MINIMUM.')
    if far:
        print(f'  far cluster (averaging-varying) min {min(far):.4f}; the unpaired floor')
        print('  does not separate the far arms -- the PAIRED per-seed statistic does.')
