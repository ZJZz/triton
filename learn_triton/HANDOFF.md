# HANDOFF — Triton device-side compilation learning setup

Audience: the next agent picking up this task. Read this top-to-bottom first.

## Who/what this is for
The user is learning Triton's **device-side (backend) compilation** for the first
time. Their leader's guidance: understand *what each pass does* conceptually (ignore
optimization passes at first), look at many real dumps, compile for different chips,
and mark which passes actually take effect. Everything below already exists and works.

## Environment (important constraints)
- Repo: `/LocalRun/jiangzhe.zhao/triton` — Triton 3.8.0, built editable. **NOT a git repo.**
- Python venv: `/LocalRun/jiangzhe.zhao/triton/.venv` (use `.venv/bin/python`).
- GPU: **NVIDIA RTX 3080, Ampere, `sm_86`** (Triton capability 86).
- **`torch` is NOT installed.** So kernels are *compiled*, not *run*. Compilation alone
  exercises the entire pass pipeline and needs no GPU/torch — do NOT add torch or try to
  launch kernels. Use the compile path only.
- `ptxas` is bundled at `third_party/nvidia/backend/bin/ptxas`.
- Build command: `make` (= `ninja -C <build_dir>`), rebuilds `python/triton/_C/libtriton.so`.
  Per AGENTS.md: only run `make` for native/C++ changes, not for pure-Python changes.

## The 5 device-side stages (the mental model taught to the user)
```
@triton.jit  →  TTIR (triton ir, hw-agnostic)
             →  TTGIR (triton gpu ir, layouts assigned)   ← most interesting passes
             →  LLIR  (llvm ir)
             →  PTX   (nvidia virtual isa)
             →  cubin (real SASS via ptxas)
```
Pipeline wiring: `third_party/nvidia/backend/compiler.py`
(`add_stages`, `make_ttir`, `make_ttgir`, `make_llir`, `make_ptx`, `make_cubin`).
Three "must-learn" passes: **ConvertTritonToTritonGPU**, **TritonGPUCoalesce**,
**ConvertTritonGPUToLLVM**.

## Files created in `learn_triton/` (all working, all verified)
| File | Purpose |
|------|---------|
| `GUIDE.md` | The main conceptual writeup. Point the user here. |
| `vec_add.py` | Minimal kernel (simplest example). |
| `matmul.py` | Chip-sensitive kernel (`tl.dot` → tensor cores differ per arch). |
| `compile_and_dump.sh` | Compile ONE kernel for the current GPU, emit all dumps. |
| `compile_driver.py` | No-torch compile for ANY int arch. **Use this, not tools/compile.py.** |
| `dump_multi_chip.sh` | Compile one kernel for sm_75/80/86/90, dump each. |
| `split_pass_dump.py` | Split an MLIR_ENABLE_DUMP log into one file per pass. |
| `mark_effective_passes.py` | Mark `[CHANGED]`/`[no-op]` per pass (which passes take effect). |
| `HANDOFF.md` | This file. |

## Dump knobs (env vars)
- `TRITON_ALWAYS_COMPILE=1` — bypass cache, always recompile.
- `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=<dir>` — write the 5 stage files
  (`<kernel>.ttir/.ttgir/.llir/.ptx/.sass/.cubin`) under `<dir>/<hash>/`.
- `MLIR_ENABLE_DUMP=1` — print IR for every pass to stderr (the "pass dump").
- `MLIR_DUMP_AFTER_PASS=1` — **(custom, see source change below)** also print IR *after*
  each pass, not just before.

## Dump outputs live in `learn_triton/dump/`
- `dump/vecadd/`       — vector-add, current chip. `stage_dump/<hash>/*.{ttir,ttgir,llir,ptx,sass,cubin}`,
  `mlir-pass-dump.log`, `mlir-pass-dump.split/NNN_*.mlir`.
- `dump/vecadd_ba/`    — vector-add with BEFORE+AFTER dumps (MLIR_DUMP_AFTER_PASS=1).
- `dump/matmul/sm{75,80,86,90}/` — matmul per chip.

`mlir-pass-dump.log` = full log (read top-to-bottom).
`mlir-pass-dump.split/` = same content, one file per pass (diff neighbours).
These two are the "log vs split" the leader referred to.

## ⚠️ LOCAL SOURCE CHANGE (not git-tracked — easy to lose)
To dump IR *both before and after* each pass, I patched two files and rebuilt:
1. `python/src/ir.cc`, in `pass_manager::enable_debug`: added
   `bool dumpAfter = ::triton::tools::getBoolEnv("MLIR_DUMP_AFTER_PASS");`
   and changed the `enableIRPrinting(...)` arg `printAfterOnlyOnFailure` from `true`
   to `!dumpAfter`. (Default behavior unchanged unless the env var is set.)
2. `include/triton/Tools/Sys/GetEnv.h`: added `"MLIR_DUMP_AFTER_PASS"` to
   `CACHE_INVALIDATING_ENV_VARS`.
3. Ran `make` → relinked `libtriton.so`. **This .so is already rebuilt and current.**
If anyone re-clones/rebuilds from a clean tree, these edits must be re-applied.

## Known bug worked around
`python/triton/tools/compile.py --target cuda:80:32` constructs
`GPUTarget(*"cuda:80:32".split(":"))`, so `arch` is the **string** `"80"`. Later
`arch >= 100` raises `TypeError: '>=' not supported between str and int`.
`compile_driver.py` avoids this by building `GPUTarget("cuda", int(arch), 32)`.
(Not yet reported/fixed upstream — could be a good first contribution.)

## Verified results (so the next agent knows what "correct" looks like)
- vec-add: 13/69 passes change the IR (before-only mode); 12/65 in before+after mode.
- matmul `tl.dot` lowering per chip:
  - sm_75 → no `#mma`, falls back to 1024× `fma.rn` (CUDA cores).
  - sm_80/sm_86 → `nvidia_mma v2` `instrShape [16,8]` → PTX `mma.sync.m16n8k16`.
  - sm_90 → `nvidia_mma v3` `instrShape [16,64,16]` → PTX `wgmma.mma_async.m64n64k16` + fences.

## How to reproduce / extend
```bash
cd /LocalRun/jiangzhe.zhao/triton/learn_triton

# one kernel, current chip, full dumps:
./compile_and_dump.sh vec_add.py add_kernel "*fp32:16, *fp32:16, *fp32:16, i32, 1024" "1024,1,1" vecadd

# one kernel across chips:
./dump_multi_chip.sh matmul.py matmul_kernel \
  "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
  unused matmul 3

# which passes took effect:
.venv/bin/python mark_effective_passes.py dump/matmul/sm90/mlir-pass-dump.log

# before+after a single pass, read in place:
MLIR_ENABLE_DUMP=1 MLIR_DUMP_AFTER_PASS=1 \
  .venv/bin/python compile_driver.py matmul.py matmul_kernel "<sig>" 90
```

## Suggested next step (where I left off)
The user was offered, but hasn't yet done, a walkthrough of the **sm_90 warp-specialization
+ `wgmma`** pass dump in `dump/matmul/sm90/` — that's where Hopper's pipeline diverges most
from Ampere and is the richest "optimization pass" example. Good place to continue.

## Signature grammar cheat-sheet (for compile_driver.py / compile_and_dump.sh)
Comma-separated, in kernel arg order:
- `*fp16` pointer to fp16 ; `*fp16:16` pointer 16-byte aligned (divisibility hint).
- `i32` / `i64` / `fp32` scalar.
- a bare number like `64` = a `tl.constexpr` literal (must match constexpr params).
