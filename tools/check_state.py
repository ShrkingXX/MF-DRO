"""Guard: research-state.yaml must actually parse.

It did not, for at least 15 hours across two sessions, because nothing read it
programmatically. The file is the cross-session coordination mechanism -- its
claims block is what the h88/h89 experiment-ID collisions were meant to prevent
-- and a claims block no tool can read cannot do that job.

Usage:  .venv/bin/python tools/check_state.py     (exit 1 on failure)

The commonest breakage: a list item containing ": " or ending in ":" becomes an
accidental mapping, and its continuation lines then read as sibling keys. Write
such an item as a block scalar (- >-) or quote it.
"""
import sys, os, yaml
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research-state.yaml")
def main():
    try:
        d = yaml.safe_load(open(P))
    except Exception as e:
        print("FAIL research-state.yaml does not parse:\n ", str(e).replace("\n", "\n  "))
        return 1
    if not isinstance(d, dict):
        print("FAIL top level is not a mapping"); return 1
    def find(o, k):
        if isinstance(o, dict):
            for a, b in o.items():
                if a == k: return b
                r = find(b, k)
                if r is not None: return r
        return None
    bad = []
    for key in ("constraints", "unclaimed_and_open",
                "claimed_by_session_A", "claimed_by_session_B"):
        v = find(d, key)
        if v is None: continue
        if not isinstance(v, list): bad.append(f"{key} is {type(v).__name__}, expected list")
        else:
            for x in v:
                if not isinstance(x, str):
                    bad.append(f"{key} item is {type(x).__name__}, expected str: {str(x)[:60]}")
    if bad:
        print("FAIL structure:"); [print("  -", b) for b in bad]; return 1
    print(f"OK research-state.yaml parses; {len(d)} top-level keys")
    return 0
if __name__ == "__main__":
    sys.exit(main())
