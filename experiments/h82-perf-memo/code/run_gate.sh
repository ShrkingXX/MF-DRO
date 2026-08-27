#!/bin/zsh
set -e
cd "$(dirname "$0")/../../.."
R=experiments/h82-perf-memo/results
mkdir -p "$R"
REF="${1:-HEAD}"
cp src/policy/mf_dro.py /tmp/h82_mf.py
cp src/models/ko_gp.py  /tmp/h82_ko.py
echo "[gate] EDITED working tree ..."
.venv/bin/python experiments/h82-perf-memo/code/gate.py 2>/dev/null | grep '^GATEJSON ' | sed 's/^GATEJSON //' > "$R/gate_edited.json"
echo "[gate] restoring $REF ..."
git show $REF:src/policy/mf_dro.py > src/policy/mf_dro.py
git show $REF:src/models/ko_gp.py  > src/models/ko_gp.py
echo "[gate] PRE-EDIT $REF ..."
.venv/bin/python experiments/h82-perf-memo/code/gate.py 2>/dev/null | grep '^GATEJSON ' | sed 's/^GATEJSON //' > "$R/gate_ref.json"
echo "[gate] restoring edits ..."
cp /tmp/h82_mf.py src/policy/mf_dro.py
cp /tmp/h82_ko.py src/models/ko_gp.py
if cmp -s "$R/gate_edited.json" "$R/gate_ref.json"; then
  echo "[gate] PASS -- bit-for-bit identical ($(wc -c < "$R/gate_edited.json") bytes)"
else
  echo "[gate] FAIL -- differs from $REF"; diff "$R/gate_edited.json" "$R/gate_ref.json" | head -20; exit 1
fi
