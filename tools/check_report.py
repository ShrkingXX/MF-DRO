#!/usr/bin/env python3
"""Pre-publish check for to_human/*.html. Run BEFORE every Artifact publish.

WHY
  1. WRAPPER CONTAMINATION. The served page is wrapped in
     <!doctype html>...<base href="/_f/<version>/">...</body></html>. Rebuilding
     the working file from a saved served copy bakes that wrapper in as source.
     A peer session hit it twice. Tell: file starts with <!doctype, not <title>.
  2. STALE RETRACTED CLAIMS. A retraction that reaches findings.md and stops
     there leaves the original live on the only surface anyone else reads. One
     jointly-retracted claim survived four sections and many republishes.

WHY THIS IS PYTHON AND NOT GREP
  The first shell version flagged "0.11 from additive" as a live claim. It was
  inside a callout headed "A phrase from our own notes, withdrawn" -- a
  CITATION in a retraction, which is exactly where a retracted phrase SHOULD
  appear. A substring check cannot tell an assertion from a citation, so it
  fails on correctly-written corrections. This version strips retraction
  callouts and quoted spans before searching.
"""
import re, sys

# phrase that must not be ASSERTED | why it was retracted
RETRACTED = [
    ("moves the wrong way twice over", "dispersion sign flip: unweighted-averaging artefact"),
    ("That is what independence looks like", "asserts what the interaction's sd only permits"),
    ("0.11 from additive", "point estimate quoted as precision; sd is 2.420"),
    ("95% of the way to fully additive", "same, see h135"),
    ("up 9.5%", "retracted dispersion sign flip"),
    ("down 10.6%", "retracted dispersion sign flip"),
    # --- added after the h180-h189 run; this list had gone stale and held only
    # --- h135-era claims while ~9 further retractions accumulated.
    ("about as good as running the",
     "h187: worse on Borehole 5/5; h189: flips sign, +13.89 on Hartmann"),
    ("only ever recorded on two of the experiments",
     "false limit -- recorded on 18 runs across both benchmarks"),
    ("a gap 0.70 wide",
     "Borehole bimodality was a 10-arm sampling artefact"),
]
RETRACTION_WORDS = ("withdrawn", "retracted", "correction", "corrected", "we are not going to")

def strip_retraction_contexts(html):
    """Remove callouts that are themselves retractions, and quoted spans."""
    out = html
    # whole callout divs whose text announces a retraction
    for m in list(re.finditer(r'<div class="(?:callout|note|open)">(.*?)</div>', out, re.S)):
        if any(w in m.group(1).lower() for w in RETRACTION_WORDS):
            out = out.replace(m.group(0), " ")
    # any sentence-ish span inside typographic quotes is a citation, not a claim
    out = re.sub(r'&ldquo;.*?&rdquo;', ' ', out, flags=re.S)
    out = re.sub(r'“.*?”', ' ', out, flags=re.S)
    return out

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "to_human/mfdro_progress.html"
    html = open(path).read()
    bad = []
    print(f"== {path} ==")

    if html.lstrip()[:40].startswith("<title>"):
        print("  head ....... OK (<title>)")
    else:
        print(f"  head ....... FAIL -- got {html.lstrip()[:40]!r}"); bad.append("head")

    for marker in ("<!doctype", "<base href", "</body></html>", "<html"):
        n = html.lower().count(marker.lower())
        if n: print(f"  wrapper .... FAIL -- {marker} x{n}"); bad.append(marker)
    if not any(b in bad for b in ("<!doctype", "<base href", "</body></html>", "<html")):
        print("  wrapper .... OK (no served-page wrapper)")

    body = strip_retraction_contexts(html)
    for phrase, why in RETRACTED:
        n = body.lower().count(phrase.lower())
        if n: print(f"  retracted .. FAIL -- '{phrase}' asserted x{n} -- {why}"); bad.append(phrase)
    if not any(p for p, _ in RETRACTED if p in bad):
        print(f"  retracted .. OK ({len(RETRACTED)} claims absent outside retraction contexts)")

    for t in ("section", "table", "tr", "td", "p", "div"):
        o = len(re.findall(r"<%s[ >]" % t, html)); c = len(re.findall(r"</%s>" % t, html))
        if o != c: print(f"  tags ....... FAIL -- <{t}> {o} open / {c} close"); bad.append(t)
    if not any(t in bad for t in ("section","table","tr","td","p","div")):
        print("  tags ....... OK (balanced)")

    print("  ALL CHECKS PASS" if not bad else "  FAILED: " + ", ".join(map(str, bad)))
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
