# Learning Triton's device-side (backend) compilation

Goal: understand **what each compiler decision does** using real dumps.

For every important pass, we want to reconstruct this chain:

```text
Input IR
  -> compiler decision
  -> output IR
  -> hardware / execution motivation
  -> next-pass contract
```

A pass is the implementation unit. The learning target is the decision it makes and the
contract it establishes for later passes.

The target NVIDIA architectures for the current learning path are:

| Architecture | Main dump target | Why we study it |
|---|---:|---|
| Ampere | `sm80` / `sm86` | `mma.sync`, Ampere-era tensor core path, useful baseline |
| Hopper | `sm90` | `wgmma`, TMA, warp specialization, Hopper-era tensor memory path |
| Blackwell | `sm100` | Blackwell-specific tensor core / TMEM / cluster features; `sm120` is a related consumer Blackwell variant when needed |

The local GPU in this box is **NVIDIA RTX 3080, Ampere, `sm_86`**. For Hopper and
Blackwell we use ahead-of-time compiler dumps; no GPU execution is required for the IR study.

## The 5 stages (the big picture)

Triton lowers a `@triton.jit` kernel through five IRs. Each is one file in
`dumps/<name>/stage_dump/<hash>/`:

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

## What TTGIR is actually for

Use this mental model:

```text
TTGIR = distributed execution mapping
      + layout / data-movement organization
      + target-driven scheduling
```

In practice:

- **Distributed execution mapping**:
  decide which thread / warp / CTA owns which logical tensor elements.
- **Layout / data-movement organization**:
  decide how those values are organized as distributed tensors, shared-memory
  tiles, descriptor-backed tiles, and layout conversions.
- **Target-driven scheduling**:
  when needed, decide when to load, when to compute, when to overlap, and when
  to synchronize.

For deeper dives on the three terms, see:

- [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md:1)
- [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md:1)
- [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md:1)

In TTGIR, **layout is not only a storage format**. It is also one of the main
carriers of the execution mapping.

This is visible in the TTIR -> TTGIR conversion itself:
[include/triton/Conversion/TritonToTritonGPU/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)
describes `ConvertTritonToTritonGPU` as enhancing tensor types with layout
encodings that include `numWarps`, `threadsPerWarp`, and `numCTAs`.

So when you see:

```text
tensor<... , #blocked<...>>
```

do not read it as only "memory layout changed". Read it as:

```text
the compiler has chosen a distributed ownership / execution pattern
and encoded it into the tensor type
```

## TTGIR vs. TTNG boundary

When learning the NVIDIA backend, it is easy to blur:

- `ttg` = TritonGPU dialect, the generic GPU-level tensor/distribution layer
- `ttng` = TritonNvidiaGPU dialect, the NVIDIA-specific layer for things such as
  TMA, TMEM, async proxy fences, cluster barriers, and other target-specific ops

The boundary matters because many "memory movement / synchronization" examples
that show up in real Hopper or Blackwell kernels are no longer pure `ttg`
concepts.

Examples from the current repository:

- TMA lowering, descriptor-encoding optimization, and TMEM passes live in
  [include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:109)
- NVIDIA-specific async copy, barrier, cluster, and fence ops live in
  [include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:52)

So a better way to think about the backend split is:

```text
TTIR
  -> TTGIR: establish GPU-distributed tensor semantics and generic GPU scheduling structure
  -> TTNGIR / NVIDIA-specific TTGIR region: materialize NVIDIA-specific transport,
     tensor-memory, barrier, fence, and cluster contracts
  -> LLVM IR
```

That means statements like "TTGIR mainly does TMA / TMEM / cp.async barrier
protocol" are too broad unless you explicitly include the NVIDIA-specific
dialect layer in your definition of "TTGIR".

## The two dump views (what your leader meant)

- `mlir-pass-dump.log` — the FULL log: IR snapshot **after every MLIR pass**, in order.
  Produced by `MLIR_ENABLE_DUMP=1` (an MLIR/LLVM env var) captured to a file.
- `mlir-pass-dump.split/NNN_<Pass>.mlir` — the SAME content, **one file per pass**, numbered.
  Produced by `tools/split_pass_dump.py`. Diff neighbours to see exactly what a pass changed:
      diff dumps/vecadd/mlir-pass-dump.split/009_*.mlir dumps/vecadd/mlir-pass-dump.split/010_*.mlir

The two are "essentially the same content" — `.log` is for reading top-to-bottom,
`.split` is for diffing pass N vs N+1.

For the detailed pass-by-pass study method, use
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md).

The main question is not only "what changed?" but:

- What compiler question is this pass answering?
- Why is this decision made here in the pipeline?
- What contract does the after-IR establish for later passes?
- What invariants must remain unchanged?

## KEY passes to understand first (ignore the optimization passes)

Read these in `make_ttgir` / `make_llir` order. The ones that change the *shape* of the
program (not just clean it up) are the ones worth learning:

Think of the early backend as a dependency graph, not just a flat list:

```text
TTIR tensor program
  -> ConvertTritonToTritonGPU
       establishes GPU-distributed ownership / layouts / encodings
  -> Coalesce / layout-oriented passes
       refine memory-facing layouts without changing tensor meaning
  -> PlanCTA
       establishes CTA ownership / CGA layout when num_ctas > 1
  -> RemoveLayoutConversions / OptimizeThreadLocality / OptimizeDotOperands
       clean and refine layout contracts around compute, communication, and memory
  -> AccelerateMatmul / MMA-WGMMA-oriented passes
       establish tensor-core instruction contracts
  -> Pipeline / ScheduleLoops / WarpSpecialize
       establish execution ordering and latency-hiding contracts
  -> NVIDIA-specific descriptor / TMA / TMEM / fence / barrier passes
       materialize target-specific transport, memory-space, and synchronization contracts
  -> ConvertTritonGPUToLLVM
       lowers the established contracts into LLVM/NVVM-level code
```

When studying a pass, always ask which previous contract it consumes and which later pass
depends on its output.

### Front-end / TTIR (`make_ttir`)
- **Inliner** — inline called `@triton.jit` functions into one body.
- **Canonicalizer / CSE / SymbolDCE** — *optimization/cleanup; skip for now.*

### TTIR -> TTGIR (the most important transition)
- **ConvertTritonToTritonGPU** (`009`) — THE pivotal pass. Attaches a **layout**
  (`#blocked<{sizePerThread, threadsPerWarp, warpsPerCTA, order}>`) to every tensor.
  This is not only a storage decision. It decides how tensor elements map onto
  threads/warps/CTAs and encodes that distributed ownership into the type.
  Compare `008` vs `009`.
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

## A better 3-question framework for TTGIR

When reading any TTGIR pass, avoid starting from "what op did it rewrite?".
Start from which compiler question it is answering:

1. **Who computes / owns these tensor elements?**
   This is the execution-mapping question.
   Typical carriers: `#blocked`, `#slice`, CTA tiling, warp specialization tags.

2. **In what form should these values exist between uses?**
   This is the layout / movement question.
   Typical carriers: `ttg.convert_layout`, `local_load`, `local_store`,
   coalesced layouts, shared layouts, dot-operand layouts.

3. **When should the work happen, and what ordering is required?**
   This is the schedule / synchronization question.
   Typical carriers: pipelining, loop scheduling, warp specialization,
   later target-specific barriers / fences / waits.

This framework is stronger than "TTGIR mostly changes layout" because it tells
you **why** layout changes exist:

```text
optimization goal
  -> choose execution ownership
  -> choose value organization / movement
  -> if needed, choose execution order
```

`ttg.convert_layout` is often just the visible symptom of one of those deeper
decisions.

## How to regenerate dumps for any kernel

    ./scripts/compile_and_dump.sh <file.py> <kernel_name> "<signature>" "<grid>" <out_subdir>

Current scope: dump compiler IR only. We do not run the kernel or validate numerical results
at this stage.

Example:

    ./scripts/compile_and_dump.sh kernels/vec_add.py add_kernel "*fp32:16, *fp32:16, *fp32:16, i32, 1024" "1024,1,1" vecadd

Env knobs it sets (worth memorizing):
- `TRITON_ALWAYS_COMPILE=1` — bypass the cache so you always get fresh dumps.
- `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=...` — write the 5 stage files.
- `MLIR_ENABLE_DUMP=1` — print IR after every pass (the pass-dump log).

## Looking at optimization passes (leader's follow-up asks)

Four techniques, each with a tool in this folder:

### 1. Dump the SAME kernel for different architectures
`tl.dot` lowers to different tensor-core instructions per architecture. Run:

    ./scripts/dump_multi_chip.sh kernels/matmul.py matmul_kernel \
      "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
      unused matmul 3

For the current learning path, compare the same kernel across:

- Ampere: `dumps/<kernel>/sm80` or `dumps/<kernel>/sm86`
- Hopper: `dumps/<kernel>/sm90`
- Blackwell: `dumps/<kernel>/sm100`

Current canonical matmul dumps use exact before/after pass snapshots:

- `dumps/matmul/sm86_num_ctas1`
- `dumps/matmul/sm90_num_ctas1`
- `dumps/matmul/sm100_num_ctas1`
- `dumps/matmul/sm90_num_ctas2`
- `dumps/matmul/sm100_num_ctas2`

What the canonical matmul dumps show:

| chip | TTGIR `#mma` layout            | PTX tensor instr                         |
|------|--------------------------------|------------------------------------------|
| sm_86 Ampere | `nvidia_mma v2, instrShape [16,8]` | `mma.sync.m16n8k16`            |
| sm_90 Hopper | `nvidia_mma v3, instrShape [16,64,16]` | `wgmma.mma_async.m64n64k16` + fences |
| sm_100 Blackwell | Blackwell-specific MMA/TMEM-oriented layouts | Blackwell-specific tensor-core lowering |

(NOTE: tools/compile.py's `--target` is buggy — it passes arch as a string and
crashes on `arch >= 100`. Use `tools/compile_driver.py`, which builds `GPUTarget("cuda", <int>, 32)`.)

For Blackwell dumps, use `TRITON_ARCH=100` with `compile_and_dump.sh`. PTX/cubin generation
may require `ptxas-blackwell`; if that tool is unavailable, the earlier MLIR dumps are still
useful for comparing compiler IR before final assembly.

### 2. Dump all passes at each stage
`MLIR_ENABLE_DUMP=1` already prints every pass of every stage (ttir/ttgir/llir pass
managers) into `mlir-pass-dump.log`; `tools/split_pass_dump.py` makes it one file per pass.

### 3. Mark which passes actually take effect
Most passes are no-ops for a given kernel. To see which ones changed the IR:

    python tools/mark_effective_passes.py dumps/vecadd/mlir-pass-dump.log

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

    MLIR_ENABLE_DUMP=1 MLIR_DUMP_AFTER_PASS=1 python tools/compile_driver.py ...

Now every pass emits an `IR Dump Before X` AND an `IR Dump After X`, so you can read a
single pass's effect in place (and `tools/mark_effective_passes.py` auto-switches to its exact
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
