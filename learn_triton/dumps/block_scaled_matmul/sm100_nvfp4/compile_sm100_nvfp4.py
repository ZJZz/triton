import importlib.util
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


ROOT = Path(__file__).resolve().parent
KERNEL_FILE = ROOT / "tutorial_kernel_only.py"


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location("_k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main():
    kernel = load_kernel(KERNEL_FILE, "block_scaled_matmul_kernel")
    signature = {
        "a_desc": "tensordesc<u8[128,128]>",
        "a_scale_desc": "tensordesc<fp8e4nv[1,1,4,2,256]>",
        "b_desc": "tensordesc<u8[256,128]>",
        "b_scale_desc": "tensordesc<fp8e4nv[1,2,4,2,256]>",
        "c_desc": "tensordesc<fp16[128,256]>",
        "M": "constexpr",
        "N": "constexpr",
        "K": "constexpr",
        "output_type": "constexpr",
        "ELEM_PER_BYTE_A": "constexpr",
        "ELEM_PER_BYTE_B": "constexpr",
        "VEC_SIZE": "constexpr",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_K": "constexpr",
        "rep_m": "constexpr",
        "rep_n": "constexpr",
        "rep_k": "constexpr",
        "NUM_STAGES": "constexpr",
    }
    constexprs = {
        "M": 8192,
        "N": 8192,
        "K": 8192,
        "output_type": 1,
        "ELEM_PER_BYTE_A": 2,
        "ELEM_PER_BYTE_B": 2,
        "VEC_SIZE": 16,
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 256,
        "rep_m": 1,
        "rep_n": 2,
        "rep_k": 4,
        "NUM_STAGES": 4,
    }

    src = ASTSource(fn=kernel, signature=signature, constexprs=constexprs)
    target = GPUTarget("cuda", 100, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": 4, "num_stages": 4, "num_ctas": 1})
    cc = triton.compile(src, target=target, options=options.__dict__)
    print(f"compiled {kernel.fn.__name__} for sm_100: {len(cc.asm)} stages -> {list(cc.asm.keys())}")


if __name__ == "__main__":
    main()
