# Triton 学习总览

## 1. 目标和范围

这篇文档服务于“学习 Triton 的 device-side compilation”。

- 目标是理解一个 `@triton.jit` kernel 如何沿着 TTIR、TTGIR、LLVM IR、PTX 一路下降。
- 重点是编译器决策、IR 变化、硬件动机和下游 contract。
- 不是 Python API 教程，也不是 runtime / autotune / benchmark 指南。

如果你现在只想学 TTGIR，直接读
[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)。

## 2. 学习对象：Triton 后端的 5 个 stage

NVIDIA backend 的 stage wiring 在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:579)：

```text
Python kernel
  -> TTIR
  -> TTGIR
  -> LLVM IR
  -> PTX
  -> cubin / SASS
```

对应阶段入口是：

- `make_ttir`
- `make_ttgir`
- `make_llir`
- `make_ptx`
- `make_cubin`

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:579)。

学习时更实用的分层是：

| 阶段 | 你主要在学什么 |
|---|---|
| TTIR | Triton tensor program 的逻辑语义，还没有 GPU 执行分发 contract |
| TTGIR | distributed execution mapping（执行层级上的分工映射）、layout / movement organization、target-driven scheduling |
| LLVM IR | Triton contract 如何被拆成每个 lane 的地址计算、控制流和目标相关 intrinsic |
| PTX / cubin | 后端最终选择了哪些 ISA 级指令和协议 |

## 3. 当前学习材料和 target

当前学习路径主要围绕 NVIDIA backend。

- 本机 GPU：RTX 3080，`sm_86`
- Hopper / Blackwell：主要通过 AOT dump 学习，不依赖本机有对应 GPU

现成 dump 目录在：

- [learn_triton/dumps/vecadd](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd)
- [learn_triton/dumps/matmul](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul)

`matmul` 当前常用比较对象包括：

- `sm86_num_ctas1`
- `sm90_num_ctas1`
- `sm100_num_ctas1`
- `sm90_num_ctas2`
- `sm100_num_ctas2`

## 4. 推荐学习顺序

### 4.1 先把整个 pipeline 看清

不要一上来扎进某个优化 pass。先建立：

```text
前一层 IR 提供什么语义
  -> 当前层做了什么决定
  -> 下一层消耗了什么 contract
```

这是后面读任何 dump 的公共框架。

### 4.2 再把 TTGIR 学扎实

TTGIR 是整个学习路径里信息密度最高的一层，因为大部分“真正有语义分量的编译决策”都在这里显式化。

建议按这个顺序读：

1. [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)
2. [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md)
3. [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)
4. [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)

### 4.3 然后再往下追 LLVM / PTX

只有当你已经能稳定解释：

- 为什么一个 tensor 带某个 encoding
- 为什么会出现某个 `convert_layout` / `local_alloc`
- 为什么某段 loop 需要某个 schedule / barrier protocol

再往 LLVM / PTX 走才不会变成单纯“认指令名”。

## 5. 各阶段最值得先学的 pass

这里不追求完整 pass 清单，只列最值得先学的骨架。

### 5.1 TTIR

- `Inliner`
- `Canonicalizer` / `CSE` / `SymbolDCE`

TTIR 阶段先看语义成形和代码清理，不用在这里停太久。

### 5.2 TTIR -> TTGIR

最关键的是这些 pass：

- `ConvertTritonToTritonGPU`
  见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)
- `TritonGPUCoalesce`
  见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:235)
- `PlanCTA`
  见 [TritonNvidiaGPU Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:27)
- `AccelerateMatmul`
  见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:203)
- `RemoveLayoutConversions`
  见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)
- `AssignLatencies` / `ScheduleLoops` / `Pipeline`
  见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:29)
  和 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:43)
  以及 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:6)

在 NVIDIA backend 上，这一段主干顺序定义在
[compiler.py:261](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:261)，
其中 target 分叉分别在
[compiler.py:282](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:282)
和
[compiler.py:292](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:292)。

### 5.3 TTGIR -> LLVM IR

先看：

- `AllocateSharedMemory`
  见 [TritonGPUToLLVM Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonGPUToLLVM/Passes.td:6)
- `make_llir` 这整段 lowering 主干
  见 [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:366)

这一层要回答的问题是：TTGIR 里建立的 contract，最后如何变成 lane-local address computation、shared memory access、NVVM intrinsic 和控制流。

## 6. Dump 工作流

### 6.1 先生成单 kernel dump

常用入口是：

```bash
./learn_triton/scripts/compile_and_dump.sh \
  learn_triton/kernels/vec_add.py \
  add_kernel \
  "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
  "1024,1,1" \
  vecadd
```

脚本位置见
[compile_and_dump.sh](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/scripts/compile_and_dump.sh:6)。

### 6.2 再做跨架构对比

对 `matmul` 这类 kernel，常用入口是：

```bash
./learn_triton/scripts/dump_multi_chip.sh \
  learn_triton/kernels/matmul.py \
  matmul_kernel \
  "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
  unused \
  matmul \
  3
```

脚本位置见
[dump_multi_chip.sh](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/scripts/dump_multi_chip.sh:5)。

跨架构对比最适合回答这类问题：

- 同一个 `tt.dot` 在 `sm86`、`sm90`、`sm100` 上为什么会进入不同 path
- 哪些差异发生在 TTGIR
- 哪些差异要等到 LLVM / PTX 才出现

### 6.3 用 pass dump 读“是谁改了 IR”

最常用的三个材料是：

- `stage_dump/<hash>/*.ttir|ttgir|llir|ptx`
- `mlir-pass-dump.log`
- `mlir-pass-dump.split/NNN_*.mlir`

`split_pass_dump.py` 会把整份 log 按 pass 切开，见
[split_pass_dump.py](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/tools/split_pass_dump.py:4)。

如何系统地做邻接 diff，见
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md)。

### 6.4 先筛“真正起作用”的 pass

`mark_effective_passes.py` 会 diff 邻接 snapshot，告诉你哪些 pass 真正改了 IR，见
[mark_effective_passes.py](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/tools/mark_effective_passes.py:19)。

常用形式：

```bash
python learn_triton/tools/mark_effective_passes.py \
  learn_triton/dumps/vecadd/sm86/mlir-pass-dump.log
```

这一步的价值不是替代人工阅读，而是先把“值得看”的 pass 缩成一个短名单。

### 6.5 如果需要精确 before/after，同步打开 `MLIR_DUMP_AFTER_PASS`

本地仓库已经把 `MLIR_DUMP_AFTER_PASS` 接到了 pass manager debug printing：

- [python/src/ir.cc](/LocalRun/jiangzhe.zhao/my_repo/triton/python/src/ir.cc:1934)
- [GetEnv.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Tools/Sys/GetEnv.h:30)

这适合在你想把某个 pass 的 effect 和前后 pass 严格分开时使用。

## 7. 最小练习集

建议至少做这三组练习：

1. `vecadd/sm86`
   对比 `ConvertTritonToTritonGPU` 前后，再看 `Coalesce` 前后，先把 TTGIR 的第一跳读明白。
2. `matmul/sm86_num_ctas1` vs `sm90_num_ctas1` vs `sm100_num_ctas1`
   看同一个 `tt.dot` 为什么进入不同的 tensor-core / schedule path。
3. 选一个具体 value
   从 TTIR 一路追到 TTGIR、LLVM IR、PTX，确认每一层到底加了什么信息，而不是只记“最后变成了哪条指令”。

## 8. 这套文档各自负责什么

- [GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)
  负责整体学习路径和工具链
- [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)
  负责只面向 TTGIR 的心智模型、边界、对象和最小学习路径
- 三篇 TTGIR 专题文档
  负责把 TTGIR 拆成“执行层级分工映射”、layout / movement、scheduling 三个主题分别讲清
- [IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md)
  负责具体的 pass-diff 方法
