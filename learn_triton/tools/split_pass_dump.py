#!/usr/bin/env python3
"""Split an MLIR_ENABLE_DUMP log (IR after each pass) into one file per pass.

Usage: python learn_triton/tools/split_pass_dump.py <mlir-pass-dump.log> <out_dir>

Each emitted file is named like  012_TritonGPUCoalesce.mlir  so you can scan the
directory in pass order and diff neighbouring passes to see what each one changed.
"""
import re
import sys
from pathlib import Path

HEADER = re.compile(r"// -+// IR Dump (Before|After) (\w+)")


def main():
    log = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.mlir"):
        f.unlink()

    lines = log.read_text().splitlines(keepends=True)
    # Find the start line of every pass-dump section.
    starts = [i for i, ln in enumerate(lines) if HEADER.search(ln)]
    starts.append(len(lines))

    index = []
    for n, (a, b) in enumerate(zip(starts, starts[1:])):
        m = HEADER.search(lines[a])
        when, pass_name = m.group(1), m.group(2)
        name = f"{n:03d}_{when}_{pass_name}.mlir"
        (out / name).write_text("".join(lines[a:b]))
        index.append(name)

    print(f"wrote {len(index)} pieces to {out}")
    for name in index:
        print("  ", name)


if __name__ == "__main__":
    main()
