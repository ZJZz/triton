# Learning Triton's device-side (backend) compilation

Goal: understand **what each compiler stage/pass does** (conceptually), using real dumps.
We target the GPU in this box: **NVIDIA RTX 3080, Ampere, `sm_86`** (Triton "capability 86").

## The 5 stages (the big picture)

Triton lowers a `@triton.jit` kernel through five IRs. Each is one file in
`dump/<name>/stage_dump/<hash>/`:

```
 your Python kernel
        │  (front-end: Python AST -> MLIR)
        ▼
   add_kernel.ttir    Triton IR        hardware-agnostic. Tensors, no GPU layout yet.
        │  make_ttgir
        ▼
   add_kernel.ttgir   Triton GPU IR    layouts assigned (#blocked etc.), num_warps baked in.
        │  make_llir                    THIS is where most "interesting" passes live.
        ▼
   add_kernel.llir    LLVM IR          scalar/SIMT, no Triton ops left.
        │  make_ptx
        ▼
   add_kernel.ptx     PTX              NVIDIA virtual ISA (text asm).
        │  make_cubin (ptxas)
        ▼
   add_kernel.cubin   cubin / SASS     real Ampere machine code.
```

Wiring is in `third_party/nvidia/backend/compiler.py` (`add_stages`,
`make_ttir`, `make_ttgir`, `make_llir`, `make_ptx`, `make_cubin`).

## The two dump views (what your leader meant)

- `mlir-pass-dump.log` — the FULL log: IR snapshot **after every MLIR pass**, in order.
  Produced by `MLIR_ENABLE_DUMP=1` (an MLIR/LLVM env var) captured to a file.
- `mlir-pass-dump.split/NNN_<Pass>.mlir` — the SAME content, **one file per pass**, numbered.
  Produced by `split_pass_dump.py`. Diff neighbours to see exactly what a pass changed:
      diff dump/vecadd/mlir-pass-dump.split/009_*.mlir dump/vecadd/mlir-pass-dump.split/010_*.mlir

The two are "essentially the same content" — `.log` is for reading top-to-bottom,
`.split` is for diffing pass N vs N+1.

## KEY passes to understand first (ignore the optimization passes)

Read these in `make_ttgir` / `make_llir` order. The ones that change the *shape* of the
program (not just clean it up) are the ones worth learning:

### Front-end / TTIR (`make_ttir`)
- **Inliner** — inline called `@triton.jit` functions into one body.
- **Canonicalizer / CSE / SymbolDCE** — *optimization/cleanup; skip for now.*

### TTIR -> TTGIR (the most important transition)
- **ConvertTritonToTritonGPU** (`009`) — THE pivotal pass. Attaches a **layout**
  (`#blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>`) to every tensor.
  This decides how tensor elements map onto threads/warps. Compare `008` vs `009`.
- **TritonGPUCoalesce** (`010`) — pick layouts so neighbouring threads touch
  neighbouring memory => coalesced (fast) global loads/stores.
- **AccelerateMatmul** — map `tt.dot` onto tensor-core (`mma`) instructions. (No-op here,
  vector-add has no dot — but it's central for GEMM/attention.)
- **RemoveLayoutConversions** — delete redundant `ttg.convert_layout` ops. *(optimization)*
- **Pipeline / ScheduleLoops / AssignLatencies / WarpSpecialize** — software-pipeline
  loops to overlap global loads with compute (the `num_stages` knob). *(optimization, but
  the single biggest perf lever; learn it after the basics.)*

### TTGIR -> LLIR (`make_llir`)
- **AllocateSharedMemory** — assign shared-memory offsets to ops that need scratch.
- **ConvertTritonGPUToLLVM** (`057`) — THE other big pass. Lowers every remaining
  Triton/TritonGPU op to LLVM dialect: address computation, masked loads/stores,
  reductions, layout conversions become real per-thread instructions.
- **ConvertNVGPUToLLVM / NVVMToLLVM** — final dialect cleanup to plain LLVM IR.

So if you only learn three passes, learn:
**ConvertTritonToTritonGPU**, **Coalesce**, **ConvertTritonGPUToLLVM**.

## How to regenerate dumps for any kernel

    ./compile_and_dump.sh <file.py> <kernel_name> "<signature>" "<grid>" <out_subdir>

Example:

    ./compile_and_dump.sh vec_add.py add_kernel "*fp32:16, *fp32:16, *fp32:16, i32, 1024" "1024,1,1" vecadd

Env knobs it sets (worth memorizing):
- `TRITON_ALWAYS_COMPILE=1` — bypass the cache so you always get fresh dumps.
- `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=...` — write the 5 stage files.
- `MLIR_ENABLE_DUMP=1` — print IR after every pass (the pass-dump log).

## Looking at optimization passes (leader's follow-up asks)

Four techniques, each with a tool in this folder:

### 1. Dump the SAME kernel for different chips
`tl.dot` lowers to different tensor-core instructions per architecture. Run:

    ./dump_multi_chip.sh matmul.py matmul_kernel \
      "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
      unused matmul 3

What the dumps show for this matmul (`dump/matmul/sm{75,80,86,90}/`):
| chip | TTGIR `#mma` layout            | PTX tensor instr                         |
|------|--------------------------------|------------------------------------------|
| sm_75 Turing | (none) — fell back to FMA | 1024x `fma.rn` (CUDA cores)         |
| sm_80 Ampere | `nvidia_mma v2, instrShape [16,8]` | `mma.sync.m16n8k16`            |
| sm_86 Ampere | `nvidia_mma v2, instrShape [16,8]` | `mma.sync.m16n8k16`            |
| sm_90 Hopper | `nvidia_mma v3, instrShape [16,64,16]` | `wgmma.mma_async.m64n64k16` + fences |

(NOTE: tools/compile.py's `--target` is buggy — it passes arch as a string and
crashes on `arch >= 100`. Use `compile_driver.py`, which builds `GPUTarget("cuda", <int>, 32)`.)

### 2. Dump all passes at each stage
`MLIR_ENABLE_DUMP=1` already prints every pass of every stage (ttir/ttgir/llir pass
managers) into `mlir-pass-dump.log`; `split_pass_dump.py` makes it one file per pass.

### 3. Mark which passes actually take effect
Most passes are no-ops for a given kernel. To see which ones changed the IR:

    python mark_effective_passes.py dump/vecadd/mlir-pass-dump.log

It diffs neighbouring snapshots (ignoring `#loc` noise) and prints `[CHANGED]`/`[no-op]`
plus a line delta. For vector-add only 13/69 passes do anything; for matmul ~27/71.
The `[CHANGED]` list IS your shortlist of passes worth studying for that kernel.

### 4. Dump BOTH before and after each pass (needed a source change)
By default Triton only prints the IR *before* each pass: in `python/src/ir.cc`,
`enableIRPrinting(...)` was called with `printAfterOnlyOnFailure=true`, which suppresses
the "After" snapshot. I added an env var `MLIR_DUMP_AFTER_PASS` that flips it:

    # python/src/ir.cc  (inside pass_manager::enable_debug)
    bool dumpAfter = ::triton::tools::getBoolEnv("MLIR_DUMP_AFTER_PASS");
    self.enableIRPrinting(... , /*printAfterOnlyOnFailure*/ !dumpAfter, ...);
    # and whitelisted "MLIR_DUMP_AFTER_PASS" in include/triton/Tools/Sys/GetEnv.h

After `make` (rebuilds libtriton.so), enable it:

    MLIR_ENABLE_DUMP=1 MLIR_DUMP_AFTER_PASS=1 python compile_driver.py ...

Now every pass emits an `IR Dump Before X` AND an `IR Dump After X`, so you can read a
single pass's effect in place (and `mark_effective_passes.py` auto-switches to its exact
before/after mode). Why bother vs. diffing consecutive Befores? It attributes the change
to the *exact* pass (no ambiguity at stage boundaries where pass managers switch).

## Suggested first exercises

1. `diff` split files `008` -> `009`: watch every `tensor<1024xi32>` gain `, #blocked`.
2. Open `add_kernel.ttgir` and read the `#blocked` layout line; relate
   `sizePerThread=[4] * threadsPerWarp=[32] * warpsPerCTA=[1] = 128`... vs BLOCK_SIZE=1024.
3. Recompile with `num_warps` changed (e.g. 4 or 8) and see `warpsPerCTA` / layout change.
4. Skim `add_kernel.ptx` for `ld.global` / `st.global` and find the `.visible .entry`.
```
```
