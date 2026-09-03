#!/usr/bin/env python3
"""Guard the PRIMARY research record against stale retracted claims.

WHY THIS EXISTS
  tools/check_report.py guards to_human/*.html against retracted claims surviving
  in the published surface. Nothing guarded findings.md -- the paper backbone --
  and its RETRACTED list had gone stale: it still held only the h135-era claims
  and none of the ~9 retractions made since.

  findings.md is append-only across ~50 ticks. A retraction appended at line
  17000 does not un-write the assertion at line 9000, and a reader (or a future
  session) hitting the earlier text first gets the withdrawn claim.

  Same lesson as check_report.py: a substring search cannot tell an ASSERTION
  from a CITATION inside a correction, which is exactly where a retracted phrase
  SHOULD appear. Retraction contexts are stripped before searching.

USAGE
  python tools/check_findings.py [path]     # default findings.md
"""
import re, sys

# (phrase, why it was retracted). Phrases must be distinctive enough that a hit
# outside a retraction context is a genuine stale assertion.
RETRACTED = [
    ("about as good as running the teacher",
     "h187: WORSE on Borehole 5/5; h189: flips sign, +13.89 on Hartmann"),
    ("exists only on h171/h172",
     "false limit -- it is on 18 arms across both benchmarks; blocked 3 tests"),
    ("exists on h171/h172 alone",
     "same false limit"),
    ("the centre is not a bad place",
     "best f within 0.10 of centre is 85.76 vs 273.00 for the whole box"),
    ("collapse is a marker rather than a cause",
     "retracted: failing arms improve on their own initial design 0/5 x 4 arms"),
    ("a 0.70-wide gap",
     "Borehole bimodality was a 10-arm sampling artefact; 28 arms fill it"),
    ("the channel is genuinely irrelevant either way",
     "h179 R3 withdrawn -- h181 showed the channel was never made responsive"),
    ("restoring BTG responsiveness 336",
     "module-level only; ~1.1% transfers in situ (h181)"),
]
# Must be RETRACTION ANNOUNCEMENTS, matched at word boundaries. The first version
# used loose substrings including "correct" and "wrong", which matched ordinary
# prose ("with no correction language", "went the wrong way") and stripped 23% of
# the file -- so the guard passed a deliberately planted live assertion. Caught by
# the negative test, which is the only reason it was found.
RETRACTION_PATTERNS = [
    r'\bretract(?:ed|ion|s)?\b', r'\bwithdraw(?:n|s)?\b',
    r'\bRETRACTION\b', r'\[CORRECTED\b', r'\bCORRECTION\b',
    r'\bI (?:previously |earlier )?wrote\b', r'\bis (?:now )?(?:retracted|withdrawn)\b',
    r'\bfalse limit\b', r'\bself-correction\b', r'\bthat (?:was|is) wrong\b',
]
_RETRACT_RE = re.compile("|".join(RETRACTION_PATTERNS), re.I)


def strip_retraction_contexts(md):
    """Drop blocks that are themselves retractions, and quoted spans."""
    out = []
    # split on markdown headings; a section announcing a retraction is a context
    blocks = re.split(r'(?m)^(?=#{2,4} )', md)
    for b in blocks:
        head = b.split("\n", 1)[0]          # the HEADING only, not 400 chars of body
        if _RETRACT_RE.search(head):
            continue
        out.append(b)
    s = "\n".join(out)
    s = re.sub(r'(?m)^>.*$', ' ', s)            # blockquotes are citations
    s = re.sub(r'\[CORRECTED:.*?\]', ' ', s, flags=re.S)   # inline correction notes
    s = re.sub(r'"[^"\n]{0,200}"', ' ', s)      # quoted spans
    s = re.sub(r'\*"[^"\n]{0,200}"\*', ' ', s)
    # A LINE that itself announces a correction is a context, not an assertion.
    # Heading-level stripping misses inline corrections mid-paragraph, which is
    # how both of this tool's first two hits were false positives.
    s = "\n".join(l for l in s.splitlines() if not _RETRACT_RE.search(l))
    return s


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "findings.md"
    md = open(path).read()
    stripped = strip_retraction_contexts(md)
    print(f"== {path} ==  ({len(md.splitlines())} lines, "
          f"{len(stripped.splitlines())} after stripping retraction contexts)")
    bad = 0
    for phrase, why in RETRACTED:
        if phrase.lower() in stripped.lower():
            # report every surviving line so it can be judged
            for i, line in enumerate(md.splitlines(), 1):
                if phrase.lower() in line.lower():
                    print(f"  STALE  L{i}: {phrase!r}\n         why retracted: {why}")
            bad += 1
    if not bad:
        print(f"  retracted .. OK ({len(RETRACTED)} claims absent outside retraction contexts)")
    else:
        print(f"  retracted .. {bad} claim(s) may survive as live assertions -- review above")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

# NEGATIVE TEST, run and recorded 2026-09-02 -- a guard that cannot fail is worthless.
#   planted live assertion  -> exit 1, reports "STALE L18009"   PASS
#   same phrase as a quoted citation inside a correction -> exit 0   PASS
#   real findings.md -> exit 0
# The FIRST version of this file passed the planted assertion. Its word list used
# loose substrings ("correct", "wrong") which matched ordinary prose and stripped
# 23% of the file. Only the negative test found it.
