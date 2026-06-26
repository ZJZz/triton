#!/usr/bin/env python3
"""Mark which passes actually changed the IR ("took effect") vs. were no-ops.

Works on a `mlir-pass-dump.log` produced with MLIR_ENABLE_DUMP=1.

Two modes, auto-detected:

* BEFORE-only log (default Triton build): consecutive "Before" snapshots are
  compared. The IR going into pass N equals the IR coming out of pass N-1, so
  diff(Before[N], Before[N+1]) == the effect of pass N.

* BEFORE+AFTER log (built with MLIR_DUMP_AFTER_PASS=1): each pass has an adjacent
  Before/After pair, so the effect of a pass is diff(its Before, its After) and we
  can attribute changes exactly, even for the last pass of every stage.

Output: an ordered list of passes, each marked [CHANGED] or [no-op], with a
line-delta count. No-ops are the passes you can usually ignore while learning.

Usage: python learn_triton/tools/mark_effective_passes.py <mlir-pass-dump.log>
"""
import re
import sys
from pathlib import Path

HEADER = re.compile(r"// -+// IR Dump (Before|After) (\w+)(?:Pass)?[: ].*?//-+ //")


def parse_sections(lines):
    """Return list of (when, pass_name, body_lines)."""
    idxs = [i for i, ln in enumerate(lines) if HEADER.search(ln)]
    idxs.append(len(lines))
    out = []
    for a, b in zip(idxs, idxs[1:]):
        m = HEADER.search(lines[a])
        # body excludes the header line itself
        out.append((m.group(1), m.group(2), lines[a + 1:b]))
    return out


def norm(body):
    # Ignore pure location noise and blank lines so we compare structure, not #locNN renumbering.
    keep = []
    for ln in body:
        s = ln.rstrip("\n")
        if s.strip().startswith("#loc"):
            continue
        s = re.sub(r"loc\(#?loc[0-9]*\)", "", s)  # strip trailing loc(...) attributes
        if s.strip():
            keep.append(s)
    return keep


def main():
    lines = Path(sys.argv[1]).read_text().splitlines(keepends=True)
    secs = parse_sections(lines)
    if not secs:
        print("No pass-dump headers found. Did you set MLIR_ENABLE_DUMP=1?")
        return

    has_after = any(w == "After" for w, _, _ in secs)
    changed = total = 0
    rows = []

    if has_after:
        # Pair each Before with the following After of the same pass.
        i = 0
        while i < len(secs):
            when, name, body = secs[i]
            if when == "Before" and i + 1 < len(secs) and secs[i + 1][0] == "After":
                a, b = norm(body), norm(secs[i + 1][2])
                rows.append((name, a != b, len(b) - len(a)))
                i += 2
            else:
                i += 1
    else:
        # Before-only: diff consecutive snapshots; the pass that ran is secs[i].
        bodies = [norm(b) for _, _, b in secs]
        for i in range(len(secs) - 1):
            name = secs[i][1]
            rows.append((name, bodies[i] != bodies[i + 1], len(bodies[i + 1]) - len(bodies[i])))
        # last snapshot's pass effect isn't observable in before-only mode
        rows.append((secs[-1][1], None, 0))

    mode = "BEFORE+AFTER (exact)" if has_after else "BEFORE-only (consecutive diff)"
    print(f"# Effective-pass report  [mode: {mode}]  ({len(rows)} passes)\n")
    for n, (name, eff, delta) in enumerate(rows):
        total += 1
        if eff is None:
            tag = "[? last-of-stream]"
        elif eff:
            tag = "[CHANGED]"
            changed += 1
        else:
            tag = "[no-op]  "
        d = f"{delta:+d} lines" if eff else ""
        print(f"  {n:03d} {tag} {name:<45} {d}")
    print(f"\n{changed}/{total} passes changed the IR.")


if __name__ == "__main__":
    main()
