"""h152 STAGE 0 -- the cheap gate.

Does the joint (beam) teacher actually earn more total information gain than
the greedy MES teacher, once replay noise is accounted for?

rtg[0] = log(b_0) - log(b_T) and b_0 is fixed at rollout start, so rtg[0] is a
function of the TERMINAL posterior alone. The terminal posterior is fully
determined by the (x, ell, y) path, so realised rtg[0] can be computed by
replaying a path and Gumbel-fitting at the end -- no need for the full
simulate_mf_trajectory machinery (states/btg do not enter rtg[0]).

Three quantities per state:
  greedy_realised   R independent greedy rollouts        -> mean, and s.d. = NOISE FLOOR
  beam_planned      the beam's own internal b_T          -> WINNER'S-CURSE BIASED HIGH
                    (an argmin over B noisy b_T estimates on their own sample paths)
  beam_realised     the beam's x/ell path, REPLAYED with fresh fantasies
                    -> the honest number, and the only one the DT could be trained on

GATE: beam_realised - greedy_realised must be positive AND exceed the noise floor.
beam_planned - beam_realised is reported as the winner's curse: if the lift lives
only there, the "joint optimum" is fitting fantasy noise, not finding information.
"""
import os, sys, json, math, argparse
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from benchmarks import get_benchmark
from src.models.ko_gp import KennedyOHaganGP
from src.policy.mf_dro import compute_joint_mf_mes
from src.policy.joint_ig_teacher import beam_search_trajectory, gumbel_b

BENCH = "Borehole_8D"
T = 8
R = 8                      # replays per state
POOL_N = 200               # matches simulate_mf_trajectory's n_roi_candidates


def build_state(trace, upto, d, bounds):
    """Refit a KO GP on the first `upto` real queries of an h83 trace."""
    Xh = [e["x"] for e in trace[:upto] if e["fid"] == 1]
    Yh = [e["y"] for e in trace[:upto] if e["fid"] == 1]
    Xl = [e["x"] for e in trace[:upto] if e["fid"] == 0]
    Yl = [e["y"] for e in trace[:upto] if e["fid"] == 0]
    if len(Xh) < 3 or len(Xl) < 3:
        return None
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    ko = KennedyOHaganGP(d=d, dkl_threshold=9999)
    ko.fit(t(Xl), t(Yl).reshape(-1), t(Xh), t(Yh).reshape(-1), bounds)
    return ko


def greedy_rollout(ko, pool, c_H, c_L):
    cur, xs, els = ko, [], []
    for _ in range(T):
        x, e, _ = compute_joint_mf_mes(cur, pool, c_H, c_L)
        y = cur.sample_fantasy(x, "LH"[e], mode="sample")
        cur = cur.make_fantasy_ko(x.unsqueeze(0),
                                  torch.tensor([y], dtype=torch.float64), "LH"[e])
        _ = gumbel_b(cur, pool)       # the rollout computes b each step too
        xs.append(x); els.append(e)
    return gumbel_b(cur, pool), sum(c_H if e else c_L for e in els)


def replay(ko, pool, xs, els):
    """Follow a FIXED (x, ell) path with FRESH fantasy draws -> realised b_T."""
    cur = ko
    for x, e in zip(xs, els):
        y = cur.sample_fantasy(x, "LH"[e], mode="sample")
        cur = cur.make_fantasy_ko(x.unsqueeze(0),
                                  torch.tensor([y], dtype=torch.float64), "LH"[e])
    return gumbel_b(cur, pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--states", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hf, lf = get_benchmark(f"{BENCH}_HF"), get_benchmark(f"{BENCH}_LF")
    bounds = torch.tensor([hf["domain_min"], hf["domain_max"]], dtype=torch.float64)
    d = bounds.shape[1]; c_H, c_L = float(hf["cost"]), float(lf["cost"])
    recs = []

    for seed in a.seeds:
        p = f"{REPO}/experiments/h83-main-comparison/results/{BENCH}__MF-DRO__seed{seed}.json"
        trace = json.load(open(p))["queries"]
        n_init = sum(1 for e in trace if e.get("is_init"))
        cuts = np.linspace(n_init, len(trace), a.states + 1)[:-1].astype(int)

        for ci, cut in enumerate(cuts):
            torch.manual_seed(1000 * seed + cut); np.random.seed(1000 * seed + cut)
            ko = build_state(trace, int(cut), d, bounds)
            if ko is None:
                print(f"  [skip] seed{seed} cut{cut}: too few points", flush=True)
                continue
            pool = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(POOL_N, d, dtype=torch.float64)
            b0 = gumbel_b(ko, pool)
            lb0 = math.log(b0)

            g = [greedy_rollout(ko, pool, c_H, c_L) for _ in range(R)]
            g_rtg = [lb0 - math.log(b) for b, _ in g]
            cap = float(np.mean([c for _, c in g]))

            _x, _e, inf = beam_search_trajectory(ko, pool, T, c_H, c_L, cap,
                                                 a.beam, a.branch)
            planned = lb0 - math.log(inf["b_T"])
            b_rtg = [lb0 - math.log(replay(ko, pool, _x, _e)) for _ in range(R)]

            rec = dict(seed=seed, cut=int(cut),
                       greedy_mean=float(np.mean(g_rtg)), greedy_sd=float(np.std(g_rtg, ddof=1)),
                       beam_planned=float(planned),
                       beam_mean=float(np.mean(b_rtg)), beam_sd=float(np.std(b_rtg, ddof=1)),
                       cap=cap, beam_cost=float(inf["cost"]), elite_won=bool(inf["won_by_elite"]))
            recs.append(rec)
            print(f"  seed{seed} cut{cut:3d}: greedy {rec['greedy_mean']:+.4f}±{rec['greedy_sd']:.4f}  "
                  f"beam_planned {planned:+.4f}  beam_realised {rec['beam_mean']:+.4f}±{rec['beam_sd']:.4f}  "
                  f"elite_won={rec['elite_won']}", flush=True)

    out = a.out or f"{REPO}/experiments/h152-joint-ig-teacher/results/stage0_B{a.beam}k{a.branch}.json"
    json.dump(recs, open(out, "w"))
    print(f"\nwrote {out}  ({len(recs)} states)")


if __name__ == "__main__":
    main()
