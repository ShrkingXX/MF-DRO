#!/bin/zsh
# Runs gate.py on the working tree AND on git HEAD, diffs the two.
set -e
cd "$(dirname "$0")/../../.."
R=experiments/h67-unbounded-rho/results
mkdir -p "$R"
cp src/models/ko_gp.py /tmp/h67_ko_edited.py
cp src/policy/mf_dro.py /tmp/h67_mf_edited.py
echo "[gate] running EDITED working tree ..."
.venv/bin/python experiments/h67-unbounded-rho/code/gate.py 2>/dev/null | grep '^GATEJSON ' | sed 's/^GATEJSON //' > "$R/gate_edited.json"
echo "[gate] restoring HEAD versions ..."
git show HEAD:src/models/ko_gp.py  > src/models/ko_gp.py
git show HEAD:src/policy/mf_dro.py > src/policy/mf_dro.py
echo "[gate] running PRE-EDIT HEAD ..."
.venv/bin/python experiments/h67-unbounded-rho/code/gate.py 2>/dev/null | grep '^GATEJSON ' | sed 's/^GATEJSON //' > "$R/gate_head.json"
echo "[gate] restoring edits ..."
cp /tmp/h67_ko_edited.py src/models/ko_gp.py
cp /tmp/h67_mf_edited.py src/policy/mf_dro.py
if cmp -s "$R/gate_edited.json" "$R/gate_head.json"; then
  echo "[gate] PASS -- bit-for-bit identical ($(wc -c < "$R/gate_edited.json") bytes)"
else
  echo "[gate] FAIL -- edited and HEAD differ. h67 MUST NOT LAUNCH."; diff "$R/gate_edited.json" "$R/gate_head.json" | head -20; exit 1
fi
