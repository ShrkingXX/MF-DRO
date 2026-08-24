"""Inventory every result JSON: parse method/benchmark/seed from path, extract
final simple regret, incumbent-improvement count, and final cost. Writes a
tidy CSV to data/ for downstream analysis."""
import json, glob, os, re, csv

def final(v):
    if isinstance(v, list) and v:
        return v[-1]
    return None

rows = []
for f in sorted(glob.glob("results/**/*.json", recursive=True)):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    if not isinstance(j, dict):
        continue
    base = os.path.basename(f)[:-5]
    exp = f.split(os.sep)[1] if len(f.split(os.sep)) > 2 else ""
    parts = base.split("__")
    method = parts[0] if parts else ""
    bench = next((p for p in parts if re.search(r"_\d+D", p)), "")
    m = re.search(r"seed(\d+)", base)
    seed = int(m.group(1)) if m else j.get("seed")
    rc = j.get("regret_curve") or j.get("hf_regret_curve")
    rows.append(dict(
        exp=exp, method=method, benchmark=bench, seed=seed,
        final_regret=final(rc),
        n_iters=len(rc) if isinstance(rc, list) else None,
        distinct_regret=len(set(rc)) if isinstance(rc, list) else None,
        incumbent_improved=j.get("incumbent_improved_count"),
        final_cost=final(j.get("cumulative_cost_curve") or j.get("cost_curve")),
        lf_fraction=j.get("lf_fraction"),
        path=f,
    ))

os.makedirs("data", exist_ok=True)
with open("data/results_inventory.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"inventoried {len(rows)} runs -> data/results_inventory.csv")
