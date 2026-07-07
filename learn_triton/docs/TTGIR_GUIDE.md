# Triton TTGIR 学习指南

## 1. 范围和边界

这篇文档只讨论 TTGIR。

- 关注 `TTIR -> TTGIR -> LLVM IR` 之间，编译器建立了哪些 GPU-level contract。
- 关注 `make_ttgir` 里的 pass 分工，以及它们和 `make_llir` 的边界。
- 关注如何从 TTGIR dump / pass dump 读出 `mapping / organization / scheduling / legality / cleanup`。

不系统展开 Python API、autotune、runtime、LLVM IR、PTX、SASS。这些放到
[GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)。

NVIDIA backend 的 stage wiring 在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:579)：

```text
TTIR
  -> make_ttgir
  -> TTGIR / NVIDIA-specific GPU IR
  -> make_llir
  -> LLVM IR
```

三层边界：

- TTIR：logical tensor program，还没有 CTA / warp / thread 级执行分发 encoding。
- TTGIR：开始显式携带 GPU-level contract，例如 `#ttg.blocked`、`#ttg.slice`、`#ttg.dot_op`、`ttg.convert_layout`、`ttg.local_alloc`、pipeline metadata、`ttng` target protocol。
- LLVM IR：不再讨论 Triton tensor contract，而是讨论 lane-level 地址计算、控制流和目标指令选择。

另一个要分清的边界是 `ttg` 和 `ttng`：

- `ttg` = TritonGPU dialect，负责 generic GPU tensor execution contract。
- `ttng` = TritonNvidiaGPU dialect，负责 NVIDIA-specific transport / barrier / fence / TMEM / cluster protocol。

相关定义和 pass 入口：

- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:52)
- [TritonNvidiaGPU Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:109)

## 2. 五类问题总图

TTGIR 里最稳定的阅读框架就是这五类：

| 类别 | 核心问题 | 典型输出 |
|---|---|---|
| distributed execution mapping | 谁算，谁拥有这些 elements | execution distribution encoding |
| layout / data-movement organization | 值以什么 form / carrier 在 producer 和 consumer 之间流动 | `convert_layout`、local memory、descriptor、TMA、TMEM |
| target-driven scheduling | 这些工作何时发生，如何 overlap，靠什么 protocol 交接 | `loop.stage`、async op、wait / barrier |
| legality repair | 当前 IR 还缺什么约束才能继续 lower | fence / barrier / proxy / TMEM ordering |
| cleanup | 哪些中间表示噪音可以删除 | 冗余 convert、死代码、重复链、canonical form |

读任何 TTGIR dump，按这个顺序看：

```text
mapping
  -> organization
  -> scheduling
  -> legality
  -> cleanup
```

## 2.1 阅读 TTGIR Pass 的固定模板

这节只给阅读顺序。

如果你要把一个 pass 真正写成完整分析，包含 `Class / Boundary / Problem / Decision / Contract / Invariant / Deferred work / If absent` 这些栏目，展开版模板在
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md)。

看任意一个 pass，固定按下面顺序读：

```text
先定类
  -> 再定边界
  -> 再问为什么需要它存在
  -> 再看相关 IR diff
  -> 再回答这个样本里为什么这样改
  -> 最后看 after IR 被谁消费
```

### 2.1.1 定类

先判断它主要属于哪一类：

- distributed execution mapping
- layout / data-movement organization
- target-driven scheduling
- legality repair
- cleanup

不要一上来就看 diff。先定类，后面的信号才不会看乱。

### 2.1.2 抽象边界

对每个 pass，先强行回答这三句：

1. 输入时，IR 里应该已经有什么 contract？
2. 输出后，新建立了什么 contract？
3. 哪些东西不是它的责任？

也就是把 pass 先读成：

```text
input contract
  -> pass responsibility
  -> output contract
  -> non-goal
```

### 2.1.3 两个“为什么”

一定要把这两个问题分开：

1. 为什么有这个 pass？
2. 为什么这个 pass 在这个样本里这样改？

前者是设计问题，后者是实例问题。

更准确地说：

- `为什么有这个 pass`：
  上游 IR + 下游 consumer / lowering / hardware constraint
  为什么逼出了这一层独立 pass
- `为什么这个样本里这样改`：
  当前 IR pattern + pass analysis / heuristic / matching rule
  为什么产生了这次具体 diff

很多“我看懂 diff 了，但还没看懂这个 pass 为啥存在”的情况，就是只回答了第二个，没有回答第一个。

读具体样本时，先看和该类直接相关的 diff 信号，再回答第二个“为什么”。

### 2.1.4 看相关 IR diff

这一步不要泛看全部变化，而是只看和它那一类问题相关的信号。

| 类别 | 看 diff 时先盯什么 |
|---|---|
| mapping | encoding、ownership、`#ttg.blocked`、`CGAEncodingAttr`、`#ttg.slice` |
| organization | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMA、TMEM |
| scheduling | `loop.stage`、async op、wait、barrier protocol |
| legality | fence、proxy ordering、TMEM reuse barrier |
| cleanup | 冗余链、死代码、token、canonical form |

如果没先定类和边界，就很容易：

- 把 `RemoveLayoutConversions` 误读成 organization
- 把 `FenceInsertion` 误读成 scheduling
- 把穿插在主链里的 cleanup / legality 位置关系误读成分类关系

### 2.1.5 看 downstream consumer

每次都要补问一句：

```text
after IR 建立的这份 contract，后面到底是谁来消费？
```

常见答案有：

- 后续 layout / dot / descriptor pass
- `Pipeline` / `TMALowering`
- `FenceInsertion` / `ProxyFenceInsertion` / `TMemBarrierInsertion`
- `RemoveLayoutConversions` / `LoopAwareCSE` / `SymbolDCE`
- `LowerMMA`
- `to_llvmir`

不补这一问，很容易把 pass 读成“只是改了一下 IR 形状”。

### 2.1.6 最短版 checklist

真正常用时，可以压成这 6 问：

1. 它主要属于五类里的哪一类？
2. 它的 input / output / non-goal 是什么？
3. 为什么 pipeline 里需要它单独存在？
4. 和这类问题直接相关的 IR diff 信号是什么？
5. 这个样本里为什么做了这次具体改写？
6. after IR 会被谁继续消费？

## 3. Distributed Execution Mapping

这一类回答：谁拥有哪些 logical tensor elements。

最常见的载体：

- module attrs：`ttg.num-warps`、`ttg.num-ctas`、`ttg.threads-per-warp`、`ttg.target`
  定义见 [Dialect.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h:50)
- distributed hierarchy：`CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread`
  定义见 [TritonGPUAttrInterfaces.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td:46)
- `#ttg.blocked`
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738)
- `CGAEncodingAttr`
  定义见 [CGAEncodingAttr.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td:14)
- `#ttg.slice`
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1381)
- `#ttg.dot_op`
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)

代表 pass：

- `ConvertTritonToTritonGPU`
- `PlanCTA`

这一类 pass 的输出是 ownership contract，不是 memory layout 的局部改写。看到

```text
tensor<..., #ttg.blocked<...>>
```

更准确的读法是：这批 logical elements 已经被分配给特定的 CTA / warp / thread / per-thread values。

展开见
[DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md)。

## 4. Layout / Data-Movement Organization

这一类回答：值以什么 form 存在，通过什么 carrier 在 producer 和 consumer 之间交接。

最常见的载体：

- `ttg.convert_layout`
- `ttg.local_alloc`
- `ttg.local_load`
- `ttg.local_store`
- `memdesc_subslice` / `memdesc_trans` / `memdesc_reshape` / `memdesc_reinterpret`
- descriptor / TMA 路径
- TMEM carrier

代表 pass 可以再分成三组：

- memory-facing organization：`Coalesce`、`CoalesceAsyncCopy`
- locality-oriented organization：`OptimizeThreadLocality`
- compute / consumer-facing organization：`AccelerateMatmul`、`OptimizeDotOperands`、`OptimizeDescriptorEncoding`、`PromoteLHSToTMem`、`OptimizeTMemLayouts`
- loop 内 data-movement organization：`Prefetch`

这一类 pass 常见现象是都可能改 `ttg.convert_layout`，但原因不同：

- `Coalesce`：为了 memory path。
- `OptimizeDotOperands`：为了 dot / MMA consumer。
- `OptimizeDescriptorEncoding`：为了 descriptor consumer。
- `Prefetch`：为了把 loop 内下一轮 operand 构造和当前轮执行衔接起来。

看这类 pass，不要先问“语法上改了几个 convert”，要先问“哪个 consumer 想要这个 form / carrier”。

展开见
[LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)。

## 5. Target-Driven Scheduling

这一类回答：这些工作何时发生，如何 overlap，靠什么 protocol 交接。

最常见的载体：

- `loop.stage`
- `loop.cluster`
- async load / async MMA / async tcgen05
- `wait` / `barrier`
- warp specialization partition region

代表 pass：

- schedule decision：`FuseNestedLoops`、`AssignLatencies`、`ScheduleLoops`、`AutomaticWarpSpecialization`、`OptimizePartitionWarps`
- protocol materialization：`Pipeline`、`TMALowering`

这一类 pass 的输出不是新的 ownership，而是时序 contract：

- 哪些操作可以提前
- 哪些 producer / consumer 可以 overlap
- 何时切 stage
- 何时 wait
- 何时进入 target-specific async protocol

看 `Pipeline` 时，重点不是“它插了多少 async op”，而是“前面的 schedule decision 在这里怎样被展开成实际协议”。

展开见
[TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)。

## 6. Legality Repair

这一类回答：核心 contract 已经基本确定后，还缺哪些合法性条件才能继续 lower。

常见信号：

- `fence`
- `barrier`
- proxy ordering
- TMEM reuse ordering
- target-specific lowering hazard

要分清 TTGIR 内和 lowering 边界上的 legality：

- `FenceInsertion` 在 `make_ttgir`，见
  [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:325)
- `ProxyFenceInsertion` 和 `TMemBarrierInsertion` 在 `make_llir`，见
  [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:390)
  和
  [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:391)

后两者执行在 `allocate_shared_memory_nv` / `allocate_tensor_memory` 之后，所以它们属于 lowering-side legality repair，不属于 TTGIR pass 主链本身。

看这类 pass，重点不是“它是不是也影响时序”，而是“如果没有这一步，lowering 或 target protocol 会在哪个约束上变得不合法”。

展开见
[LEGALITY_REPAIR.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LEGALITY_REPAIR.md)。

具体 barrier / fence / async protocol 细节再看
[2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md)。

## 7. Cleanup

这一类回答：在不改变核心 ownership / carrier / schedule contract 的前提下，哪些表示噪音可以清掉。

代表 pass：

- representation cleanup：`RemoveLayoutConversions`、`ReduceDataDuplication`
- target-specific cleanup：`InterleaveTMem`、`RemoveTMEMTokens`
- IR cleanup：`ReorderInstructions`、`LoopAwareCSE`、`Canonicalizer`、`CSE`、`SCCP`、`SymbolDCE`

常见信号：

- 冗余 `ttg.convert_layout`
- 重复 producer chain
- 临时 token / 临时依赖边
- 死代码
- 只为 canonical form 服务的 SSA 重排

看这类 pass，重点不是“它改没改语义上重要的对象”，而是“它有没有重新定义 ownership / carrier / schedule”。如果没有，那首先按 cleanup 读。

展开见
[CLEANUP.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/CLEANUP.md)。

## 8. Pass 主链

`make_ttgir` 主干在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:261)，
target 分叉在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:282)
和
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:292)。

按五类去看，主链可以压成：

```text
mapping
  ConvertTritonToTritonGPU
  -> PlanCTA

organization
  -> Coalesce / CoalesceAsyncCopy
  -> OptimizeThreadLocality
  -> AccelerateMatmul / OptimizeDotOperands / OptimizeDescriptorEncoding
  -> PromoteLHSToTMem / OptimizeTMemLayouts / Prefetch

scheduling
  -> FuseNestedLoops / AssignLatencies / ScheduleLoops
  -> AutomaticWarpSpecialization / Pipeline / OptimizePartitionWarps / TMALowering

legality
  -> FenceInsertion
  -> make_llir 中的 ProxyFenceInsertion / TMemBarrierInsertion

cleanup
  -> RemoveLayoutConversions / ReduceDataDuplication / InterleaveTMem / RemoveTMEMTokens
  -> ReorderInstructions / LoopAwareCSE / Canonicalizer / CSE / SCCP / SymbolDCE
```

真实 pipeline 不是先把一类完全做完再进下一类，但读 TTGIR 时按这五类归位，比背一串线性 pass 名字更稳。

## 9. Dump 学习路径

先看 `vecadd`：

- [018_Before_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/018_Before_ConvertTritonToTritonGPU.mlir:12)
  看 TTIR 还没有 execution mapping。
- [019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:2)
  看 `#ttg.blocked` 怎样把 ownership contract 传播到 tensor type。
- [021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:3)
  看 memory-facing organization 怎样局部改 form。

再看 `matmul`：

- [061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:69)
  看 Hopper 路径上的 organization + scheduling。
- [059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)
  看 Blackwell 路径上的 coarse schedule contract。
- [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)
  看 barrier slot、async `tc_gen5_mma`、`wait_barrier`、phase rotation 这些 protocol materialization。

只读 `.ttgir` 不够。TTGIR 学习最有价值的材料通常是：

- `stage_dump/.../*.ttgir`
- `mlir-pass-dump.log`
- `mlir-pass-dump.split/NNN_*.mlir`

具体怎么做邻接 diff，见
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md)。

## 10. 自检

至少要能稳定回答这五个问题：

1. 这个 tensor 的 execution mapping 是怎么确定的。
2. 这次 `convert_layout` / `local_alloc` / descriptor / TMEM 切换，是哪个 consumer 或 carrier 逼出来的。
3. 这段 `loop.stage` / async op / wait / barrier，是怎样的 scheduling / protocol contract。
4. 这一步是在补 legality，还是只在做 cleanup。
5. 这件事发生在 `ttg`、`ttng`，还是 `make_llir` 边界。

如果这五个问题还答不稳，就先回到对应的 pass dump。
