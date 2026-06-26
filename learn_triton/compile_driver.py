#!/usr/bin/env python3
"""Compile a @triton.jit kernel for a chosen GPU arch WITHOUT torch / without a GPU.

This is a thin, correct replacement for triton/tools/compile.py whose --target path
has a bug (it passes arch as a string, so GPUTarget arch >= 100 comparisons crash).

We only care about the IR dumps, so we just call triton.compile() and let the
TRITON_KERNEL_DUMP / MLIR_ENABLE_DUMP env vars do the dumping.

Usage:
  python compile_driver.py <file.py> <kernel> "<signature>" <arch> [num_warps] [num_stages]

  <signature>  comma-separated, declaration order, same grammar as tools/compile.py:
               *fp16:16  -> 16-byte-aligned pointer to fp16
               i32       -> scalar
               64        -> a constexpr literal (for tl.constexpr params)
  <arch>       integer SM, e.g. 75 / 80 / 86 / 90
"""
import importlib.util
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location("_k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def constexpr(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return None


def main():
    path, name, sig_str, arch = sys.argv[1:5]
    num_warps = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    num_stages = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    arch = int(arch)

    kernel = load_kernel(path, name)
    sig = [s.strip() for s in sig_str.split(",")]

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(sig) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(sig)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for (i, ), v in hints.items():
        if v == 1:
            constants[kernel.arg_names[i]] = v
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(sig)}
    for k in constants:
        signature[k] = "constexpr"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}

    kernel.create_binder()
    src = kernel.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)

    target = GPUTarget("cuda", arch, 32)  # <-- arch is an int here (the bug fix)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    cc = triton.compile(src, target=target, options=options.__dict__)
    print(f"compiled {name} for sm_{arch}: {len(cc.asm)} stages -> {list(cc.asm.keys())}")


if __name__ == "__main__":
    main()
