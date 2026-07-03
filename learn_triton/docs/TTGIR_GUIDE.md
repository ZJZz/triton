# Triton TTGIR 学习指南

## 1. 目标和边界

这篇文档只服务于“学习 TTGIR”。

- 关注 `TTIR -> TTGIR -> LLVM IR` 之间，Triton 如何把 logical tensor program 变成 GPU-distributed tensor program。
- 关注 TTGIR 的对象、契约、pass 因果链，以及如何读 TTGIR dump。
- Python API、autotune、runtime、LLIR/PTX/SASS 的系统性学习，放到 [GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)。

NVIDIA backend 的 stage wiring 在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:579)。
这里把 `ttgir` 绑定到 `make_ttgir`，把 `llir` 绑定到 `make_llir`，所以 TTGIR 的职责边界就是：

```text
TTIR
  -> make_ttgir
  -> TTGIR / NVIDIA-specific GPU IR
  -> make_llir
  -> LLVM IR
```

## 2. TTGIR 到底在解决什么

最有用的心智模型是：

```text
TTGIR
  = distributed execution mapping
  + layout / data-movement organization
  + target-driven scheduling
```

- `distributed execution mapping`：决定哪些 CTA / warp / thread / per-thread values 拥有哪些 logical tensor elements。
- `layout / data-movement organization`：决定这些值在 producer 和 consumer 之间以什么 form 存在、通过什么 carrier 流动。
- `target-driven scheduling`：决定这些工作何时发生、如何 overlap、何时需要 wait / barrier / fence。

`ConvertTritonToTritonGPU` 的 pass 描述直接说明，TTIR tensor type 会被增强为带 layout encoding 的 GPU tensor type，而 encoding 一般包含 `numWarps`、`threadsPerWarp`、`numCTAs`，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)。

所以看到：

```text
tensor<..., #ttg.blocked<...>>
```

不要只把它读成“内存布局变了”，更准确的读法是：

```text
编译器已经为这批 logical elements 选择了 GPU-distributed ownership，
并把这份 contract 编码进 tensor type
```

## 3. 学 TTGIR 时必须分清的两个边界

### 3.1 TTIR / TTGIR / LLVM IR 的边界

- TTIR 主要保留 logical tensor 语义，还没有 GPU-distributed encoding。
- TTGIR 开始引入 `#blocked`、`#slice`、`#dot_op`、`ttg.convert_layout`、`ttg.local_alloc`、pipeline metadata 等 GPU-level contract。
- LLVM IR 阶段不再讨论 Triton tensor contract，而是讨论每个 lane 的地址计算、控制流和目标指令选择。

所以学习 TTGIR 时，重点不是“某个 Triton op 最后变成了哪条 PTX”，而是“LLVM lowering 之前，编译器先建立了哪些 GPU-level contract”。

### 3.2 `ttg` 和 `ttng` 的边界

学习 NVIDIA backend 时，另一个容易混淆的边界是：

- `ttg` = TritonGPU dialect，负责 generic GPU-distributed tensor 语义
- `ttng` = TritonNvidiaGPU dialect，负责 NVIDIA-specific transport / barrier / fence / TMEM / cluster 协议

相关定义和 pass 入口分别在：

- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:52)
- [TritonNvidiaGPU Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:109)

因此在 NVIDIA backend 上，“学习 TTGIR”通常必须同时读 `ttg` 和还未 lower 掉的 `ttng`，但不要把两层责任混在一起：

- `ttg` 先建立 generic ownership / layout / schedule contract
- `ttng` 再把 target-specific transport / TMEM / barrier / fence 协议显式化

## 4. 读 TTGIR 时固定问的三个问题

读任何 TTGIR dump，先不要从“这个 pass 改了什么 op”开始，先问：

| 问题 | 核心含义 | 展开文档 |
|---|---|---|
| 谁算、谁拥有这些元素 | distributed execution mapping | [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md) |
| 这些值在 producer / consumer 之间以什么 form 流动 | layout / data-movement organization | [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md) |
| 这些工作何时发生、如何 overlap、何时需要同步 | target-driven scheduling | [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md) |

这三个问题的顺序也很重要：

```text
先决定谁工作
  -> 再决定值以什么 form / carrier 存在
  -> 最后才决定这些工作何时发生、如何交接
```

很多人读 TTGIR 会误以为“主要就是在改 layout”。更准确的说法是：

```text
layout attr 和 convert_layout
往往只是更深一层编译决策的可见载体
```

## 5. 先认对象，再看 pass

如果这些对象还不熟，直接看 pass 很容易只看到“重写前后长得不一样”，看不到 contract。

### 5.1 execution mapping 载体

最先要认的是 distributed encoding 家族。

- module attrs：`ttg.num-warps`、`ttg.num-ctas`、`ttg.threads-per-warp`、`ttg.target`
  定义见 [Dialect.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h:50)
- distributed hierarchy：`CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread`
  定义见 [TritonGPUAttrInterfaces.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td:41)
- `#ttg.blocked`：最常见的 ownership contract
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738)
- `CGAEncodingAttr`：CTA-level split
  定义见 [CGAEncodingAttr.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td:14)
- `#ttg.slice`：对 parent layout 的投影
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1381)
- `#ttg.dot_op`：依附于 dot result parent 的 compute-facing distributed form
  定义见 [TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)

### 5.2 value organization / movement 载体

再看同一批 values 在不同 consumer 之间如何换形态。

- `ttg.convert_layout`
- `ttg.local_alloc`
- `ttg.local_load`
- `ttg.local_store`
- `memdesc_subslice` / `memdesc_trans` / `memdesc_reshape` / `memdesc_reinterpret`
- descriptor / TMA 路径
- Blackwell 的 TMEM carrier

这些对象分别在
[LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)
里展开。

### 5.3 schedule / protocol 载体

最后再看调度和交接协议。

- `loop.stage`
- `loop.cluster`
- async load / async MMA / async tcgen05
- `wait` / `barrier` / `fence`
- warp specialization partition region

这些对象分别在
[TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)
里展开。

## 6. 再看 TTGIR pass 因果链

对 NVIDIA backend，最应该背下来的不是平铺直叙的 pass 名单，而是这条因果链：

```text
ConvertTritonToTritonGPU
  -> Coalesce
  -> PlanCTA
  -> RemoveLayoutConversions / AccelerateMatmul / OptimizeDotOperands
  -> AssignLatencies -> ScheduleLoops -> (AutomaticWarpSpecialization) -> Pipeline
  -> NVIDIA-specific descriptor / TMA / TMEM / fence / barrier passes
  -> make_llir
```

主干顺序在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:261)，
target 分叉在
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:282)
和
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:292)。

理解这条链时，重点不是“哪个 pass 更重要”，而是“它消耗前一个 contract、建立下一个 contract”：

- `ConvertTritonToTritonGPU`
  把 logical tensor 变成 distributed tensor
- `Coalesce`
  在不改 tensor meaning 的前提下，重写 memory-facing form
- `PlanCTA`
  当 `num_ctas > 1` 时重写 CTA-level ownership
- `AccelerateMatmul` / `OptimizeDotOperands`
  为 dot / mma consumer 准备 compute-facing form
- `AssignLatencies` / `ScheduleLoops` / `Pipeline`
  把“值得 overlap 的点”变成真实的 loop schedule 和 async protocol
- `AutomaticWarpSpecialization`
  在 Blackwell 路径上先建立 partition schedule，再进入后续 materialization
- NVIDIA-specific barrier / fence / TMEM passes
  把 target-specific transport、同步和 hazard repair 显式化

如果你一眼说不清某个 `ttg.convert_layout`、`ttng.wait_barrier` 或 `#ttg.dot_op` 是被哪一层 contract 需要的，就说明还没读到 pass 因果链。

## 7. TTGIR 的最小 dump 学习路径

### 7.1 先看 `vecadd`：TTIR 到 TTGIR 的第一跳

先看
[018_Before_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/018_Before_ConvertTritonToTritonGPU.mlir:12)，
这里还没有 distributed encoding。

再看
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:2)，
这里开始出现 `#ttg.blocked`，同一份 ownership contract 被传播到 `tt.make_range`、`tt.load`、`arith.addf`、`tt.store` 的 tensor type 上。

最后看
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:3)，
观察 `Coalesce` 如何只在 memory-facing 路径局部改写 form。

### 7.2 再看 `matmul`：TTGIR 的三层 contract 如何叠起来

Hopper 路径先看
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:69)，
这里已经能看到 `ttg.local_alloc -> ttng.warp_group_dot {isAsync = true} -> ttng.warp_group_dot_wait` 的结构。

Blackwell 路径先看
[059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)，
确认 `loop.stage` / `loop.cluster` 这类 coarse schedule contract。

然后看
[085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)，
这里已经能看到 barrier slot init、async `tc_gen5_mma`、`wait_barrier`、phase rotation 等完整协议。

### 7.3 只读 `.ttgir` 不够，要读 pass dump

对 TTGIR 学习，最有价值的材料通常是：

- `stage_dump/.../*.ttgir`
- `mlir-pass-dump.log`
- `mlir-pass-dump.split/NNN_*.mlir`

具体如何做邻接 diff、怎样判断一个 pass 真正改了什么，见
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md)。

## 8. 判断自己有没有真正读懂 TTGIR

至少要能稳定回答这四个问题：

1. 这个 tensor 为什么带这个 encoding，而不是别的 encoding。
2. 这次 `convert_layout` / `local_alloc` / descriptor / TMEM 切换，是哪个 consumer 逼出来的。
3. 这段 `loop.stage` / async op / wait / barrier，是 schedule decision、protocol materialization，还是 hazard repair。
4. 这个 contract 是在 `ttg` 层建立的，还是在 `ttng` 层补出来的。

如果这四个问题还答不稳，不要急着往 PTX/SASS 走，先回到对应的 TTGIR pass dump。
