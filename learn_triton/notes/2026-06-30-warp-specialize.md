# 2026-06-30 学习笔记：WarpSpecialize

## Part A. Orientation

### 1. Pass Thesis

这一轮学习 `WarpSpecialize`，但这里先澄清一个容易混淆的点：

```text
“WarpSpecialize” 在 Triton 里不是单一源码文件里的单个变换。
用户侧看到的是一个 compiler feature；
实现侧是一条 pass 链，把 loop 从“可做 warp specialization”
逐步变成显式的 `ttg.warp_specialize` 区域，再交给后续
ScheduleLoops / Pipeline / ConvertWarpSpecializeToLLVM 消费。
```

最核心的主线是：

```text
loop has {tt.warp_specialize}
  -> TritonGPUAutomaticWarpSpecialization
       internally runs:
         PartitionScheduling
         NVWS helper passes
         PartitionLoops
         NVWSLowerWarpGroup
         ScheduleLoops
  -> optional TritonGPUOptimizePartitionWarps
  -> Pipeline
  -> ConvertWarpSpecializeToLLVM
```

一句话：

```text
Warp specialization 的 compiler decision
不是“要不要 async copy”这种局部问题，
而是“把一个 loop 里的 load / mma / compute / epilogue
拆成哪些并发 warp-group partitions，各分区之间如何传值、各用多少 warps，
最后怎样变成可执行的异步并行结构”。
```

相关定义与入口：

- Pass declarations: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:99)
- Automatic driver: [AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95)
- Partition scheduler: [PartitionScheduling.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:14)
- Loop splitter: [PartitionLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp:533)
- Partition attr verification / dataflow helpers: [Partition.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp:47)
- Warp count optimizer: [OptimizePartitionWarps.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:153)
- Warp-specialize IR ops: [TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:600)
- Public loop flag in frontend: [core.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/language/core.py:3709)
- Explicit IR builder path in Gluon: [_semantic.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/experimental/gluon/language/_semantic.py:587)

### 2. First Clarification: Two Different Meanings of "warp specialize"

Triton 里至少有两层“warp specialization”：

```text
1. Automatic warp specialization on scf.for
   由 `tl.range(..., warp_specialize=True)` 或 loop attr `tt.warp_specialize`
   触发，编译器自动决定 partitions。

2. Explicit ttg.warp_specialize IR op
   这是更低层、更显式的中间表示。
   既可以由自动 pass 生成，也可以由 Gluon / 手写 TTGIR 直接构造。
```

证据：

- `tl.range` 文档直接暴露 `warp_specialize` 选项，并注明“目前只支持 Blackwell，且主要针对 simple matmul loops”，见 [core.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/language/core.py:3709)
- Gluon 语义层直接创建 `create_warp_specialize` / `create_warp_specialize_partitions` / `create_warp_return`，见 [_semantic.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/experimental/gluon/language/_semantic.py:618)

所以学习时要分开：

```text
AutomaticWarpSpecialization:
  loop-level transformation pipeline

ttg.warp_specialize:
  transformation 的输出 IR contract
```

### 3. IR Op Semantics

`ttg.warp_specialize` 的语义定义非常重要，直接决定后面怎么理解 partitions。

IR 定义见 [TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:600)：

- `ttg.warp_specialize`：表示多个 warp groups “同时”执行不同代码，最后 join
- `default` region：当前执行 warp group 的默认路径，允许 implicit capture
- `ttg.warp_specialize.partitions`：真正 `IsolatedFromAbove` 的 partition 容器
- `partitionN ... num_warps(k)`：第 N 个分区使用多少 warps
- `ttg.warp_yield`：default region 的返回
- `ttg.warp_return`：partition region 的 terminator

一句话：

```text
`ttg.warp_specialize` 不是普通 if/else。
它表达的是多个 warp groups 的并发执行域，每个分区有自己的 warp 数和隔离的 SSA/layout 域。
```

## Part A1. Guide-Aligned Pass Card

### 3.1 Why This Note Needs A Feature-Level Template

更新后的 `IR_PASS_DIFF_LEARNING_GUIDE.md` 默认按“一个 pass 一节”组织。

`WarpSpecialize` 这里需要先做一个适配：

```text
学习目标仍然是 compiler decision + contract；
但分析单位不是单个源码文件，而是
AutomaticWarpSpecialization orchestration pipeline
+ OptimizePartitionWarps
+ 后续消费它的 Pipeline / ConvertWarpSpecializeToLLVM。
```

所以这篇笔记的做法是：

```text
先给整个 feature 一张 pass card，
再分别拆 PartitionScheduling / NVWS helper passes / PartitionLoops /
OptimizePartitionWarps 这些子决策。
```

### 3.2 Files

这一篇实际用了两类证据：

- feature-active lit tests：
  [automatic-warp-specialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:1),
  [partition-scheduling.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:1),
  [partition-loops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-loops.mlir:1),
  [optimize-partition-warps.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:1)
- canonical dump 中的真实 pass snapshot：
  Before: [060_Before_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:1)
  After: [083_After_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:1)
- orchestration source:
  [AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95)

这里要明确一点：

```text
本篇关于“feature 生效后的形态”主要靠 lit tests；
canonical matmul dump 提供的是“真实 pipeline 中这个 pass 可能 no-op”的反例证据。
```

### 3.3 Architecture Matrix

| Arch | Before | After | Changed? | Main before feature | Main after feature |
|---|---|---|---|---|---|
| Ampere `sm80/sm86` | 本篇没有可复用的 automatic dump | pending | pending | automatic path 不具现实意义 | 同左 |
| Hopper `sm90` | 本篇没有主打 automatic dump | pending | pending | 更常见是显式 `ttg.warp_specialize` lowering 相关路径 | automatic path 不是本篇主样本 |
| Blackwell `sm100` | [060_Before_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:1) | [083_After_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:1) | partially effective / no partition formed | 普通 scheduled `scf.for` + TMEM MMA loop | 仍未形成 `ttg.warp_specialize`，但 loop-carried accumulator init 被改写 |

补充说明：

```text
Blackwell 上“feature active”的主证据不是上面这份 canonical dump，
而是 automatic-warp-specialization / partition-* / optimize-partition-warps
这些专门 lit tests。
```

### 3.4 Cross-Architecture Before Comparison

- Ampere before：
  自动路径在当前仓库学习路径里不构成主样本；更合理的结论是“不值得强行做 automatic before/after 对比”。
- Hopper before：
  进入 warp-specialize 相关 lowering 的 IR 更常见，但 automatic loop partitioning 不是本篇主证据。
- Blackwell before：
  如果 loop 没有 `tt.warp_specialize` 或不满足 simple matmul-like 结构，即使 pipeline 里跑到了 `TritonGPUAutomaticWarpSpecialization`，它也可能不形成 `ttg.warp_specialize`；上面的 `sm100_num_ctas1` dump 就是这个例子，而且它还展示了“内部 helper pass 仍可能改写 IR”。

结论：

```text
WarpSpecialize 这一题最重要的 cross-architecture 结论
不是“三代 after 长什么样”，
而是“automatic path 本身就主要是 Blackwell-centric，
因此三代并排 before/after 并不是最有信息量的学习组织”。
```

### 3.5 One-line Summary

这个 feature 在本例中主要做了：

```text
把一个原本所有 warps 大体对称执行的 loop，
改写成多个 warp-groups 分工协作的并发程序，
并建立跨 partition 通信、warp 预算、以及后续 pipeline/lowering 可依赖的 IR contract。
```

### 3.6 Goal / Constraint / Design Intent

- Goal:
  让 Blackwell matmul-like hot loop 不只做时间维的软件流水，还能做空间维的 warp-group 分工。
- Constraint:
  普通 SSA 不能直接穿过 partition 边界；TMEM/TMA 路径还有 ownership、token、barrier、stage 的额外约束。
- Design intent:
  先用内部 attrs 和 NVWS 中间语义把“谁生产、谁消费、怎么交接”表达清楚，再落成 `ttg.warp_specialize` 这个可长期保留的 IR contract。

### 3.7 Compiler Decision

- Compiler question:
  一个 loop 是否值得拆成多个并发 warp-group partitions；如果值得，谁归哪个 partition、怎么通信、各分区分多少 warps。
- Decision made here:
  `PartitionScheduling` 决定角色划分，NVWS passes 决定交接协议，`PartitionLoops` 决定结构拆分，`OptimizePartitionWarps` 决定每个 partition 的 warp 预算。
- Why here in the pipeline:
  这一步必须发生在 matmul/TMEM/TMA 相关结构已经显式化之后、但 LLVM lowering 之前，因为它消费的是高层 TTGIR/NVWS 语义，输出的是后续 Pipeline / ConvertWarpSpecializeToLLVM 要吃的执行组织 contract。

### 3.8 Compiler Contract

- Input contract:
  loop 已带 `tt.warp_specialize` 或等价触发条件；更早的 layout / matmul / schedule 信息已经建立。
- Output contract:
  durable contract 是 `ttg.warp_specialize`、每个 partition 的 `num_warps`、以及合法的 cross-partition communication 结构。
- Next pass relies on:
  `ScheduleLoops`、`Pipeline`、`ConvertWarpSpecializeToLLVM`、以及可选的 `OptimizePartitionWarps`。
- Deferred work:
  它不负责最终 LLVM/NVVM lowering，不负责最终寄存器分配，也不负责所有 barrier/resource materialization；这些留给后续 passes。

### 3.9 Invariant

- Tensor shape:
  不因 warp specialization 而改变。
- Element type:
  不以 partitioning 为目的改变。
- Program semantics:
  数学结果、循环迭代语义、producer/consumer 依赖语义不变。
- Changed only:
  执行角色划分、跨 partition 交接方式、控制流结构、warp 数预算、以及后续 pipeline 可见的同步/并发组织。

### 3.10 Effective Or No-op

- Result:
  在 [060_Before_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:1)
  到
  [083_After_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:1)
  这份 canonical `sm100` matmul dump 里，更准确的判定是 `partially effective / no partition formed`，不是严格 no-op。
- Evidence:
  before/after 都仍保留同一个普通 `scf.for` TMEM MMA loop，没有出现 `ttg.warp_specialize`、`nvws.warp_group`、或 partition attrs。
  但 loop-carried accumulator iter-args 确实变了：
  Before 是 [060_Before_TritonGPUAutomaticWarpSpecialization.mlir:74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:74)
  的 `iter_args(..., %acc_52 = %acc, %acc_53 = %acc_38)`，
  After 是 [083_After_TritonGPUAutomaticWarpSpecialization.mlir:74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:74)
  的 `iter_args(..., %acc_51 = %false, %acc_52 = %acc_37)`。
- Interpretation:
  这说明两件事：
  1. 这次没有形成 partition / `ttg.warp_specialize` 结构；
  2. orchestration pass 即使没有产出目标结构，内部 helper pass 仍可能实质改写 IR。
  这里的改写与前面 `NVWSHoistTmemStore` 讨论的 accumulator initialization / `tmem_store` fold 语义是一致的。

## Part B. Where It Sits In The Pipeline

### 4. Pass Chain Inside `AutomaticWarpSpecialization`

`AutomaticWarpSpecialization::runOnOperation()` 不是自己直接完成所有工作，
而是组装了一条内部 pipeline，见 [AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95)：

```text
PartitionScheduling
-> NVWSHoistTmemStore
-> NVWSInsertAref
-> NVWSInsertTmemAref
-> SCCP
-> CSE
-> NVWSLowerAref(numStages)
-> PartitionLoops
-> NVWSLowerWarpGroup
-> ScheduleLoops
-> multiBufferTMADescriptors
-> clearInternalWarpSpecializationAttrs
```

注意：

```text
`TritonGPUOptimizePartitionWarps` 不在这条内部 pipeline 里。
它是后续可单独追加的优化 pass，
通常在 `ttg.warp_specialize` 已经形成后再运行。
```

这条链说明了 feature 的真实结构：

```text
AutomaticWarpSpecialization
  不是“一个 heuristic pass 直接吐出最终 IR”
  而是一个 orchestration pass。
它先决定 partition assignment，
再补 reference-semantic / aref / tmem 相关结构，
再真正 split loop，
最后重新进入 ScheduleLoops，让 partitioned loop 拥有新的 schedule contract。
```

两个很重要的细节：

1. `VerifyWarpSpecializationPartitions` 不是加在所有内部 pass 后面，而是只包在通过 `addPassWithPartitionVerifier(...)` 加进去的那几步：
   `PartitionScheduling`、`NVWSHoistTmemStore`、`NVWSInsertAref`、`NVWSInsertTmemAref`、`SCCP`、`CSE`、`NVWSLowerAref`，见 [AutomaticWarpSpecialization.cpp:95](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95)
   `PartitionLoops`、`NVWSLowerWarpGroup`、`ScheduleLoops` 是直接 `pm.addPass(...)`，后面不自动跟 verifier。
2. 结束时会清掉内部 attrs：`ttg.partition` / `ttg.partition.outputs` / `ttg.partition.stages` / `ttg.warp_specialize.tag`，见 [AutomaticWarpSpecialization.cpp:80](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:80)

这意味着：

```text
partition attrs 是 AutomaticWarpSpecialization 内部的中间 contract，
不是打算长期保留到最终 TTGIR 的 user-facing contract。
长期留下来的结构是 `ttg.warp_specialize` 及其后续 lowered form。
```

### 4.1 Why The NVWS Helper Passes Matter

我前一版笔记对这几个 pass 只写了名字，没有展开，这是不够的。

原因不是它们“不重要”，而是我当时把重点放在 Triton 自己的两段核心决策上：

```text
1. PartitionScheduling:
   决定 op 属于哪个 partition

2. PartitionLoops:
   真正把 loop clone/split 成多 partition 结构
```

但这样写会把 `AutomaticWarpSpecialization` 中间这段桥接层压扁：

```text
PartitionScheduling
  -> [NVWS passes: 把“跨 partition 依赖”改写成可同步、可 multibuffer、可 lowered 的中间语义]
  -> PartitionLoops
  -> NVWSLowerWarpGroup
```

严格说，这几个 NVWS pass 不是边角料，而是在回答下面这个问题：

```text
当 producer 和 consumer 被分到不同 warp-group partitions 之后，
原来的直接 SSA use 还能不能成立？

通常不能。
必须先把“跨 partition 传值”改写成显式的 buffer ownership /
async reference / barrier 协议，后面才能安全 split loop。
```

所以更准确的理解是：

```text
PartitionScheduling 决定“谁生产、谁消费”
NVWS helper passes 决定“怎么交接”
PartitionLoops 决定“把结构真正拆开”
NVWSLowerWarpGroup 决定“把 NVWS 容器落成 TritonGPU 的正式 IR contract”
```

### 4.2 What Each NVWS Pass Actually Does

下面把 `AutomaticWarpSpecialization` 里那五个 NVWS pass 连起来看。

#### `NVWSHoistTmemStore`

源码：

- pass 定义: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:134)
- 实现: [HoistTmemStore.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp:1)

它做的不是 generic CSE / LICM，而是一个很具体的 TMEM canonicalization：

```text
如果 nested loop 里的 tmem_alloc / tmem_store 其实可以提升到更外层，
就把它 hoist 出去，并把 async token 正确 thread 过 loop nest。
```

这个 pass 重要的原因有两个：

1. 它会把 `tmem_store` 尽量 fold 进 `tmem_alloc(src)`，见 [HoistTmemStore.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp:91)
2. 它会把 TMEM alloc/store 的 owner 收敛成更稳定的单一 partition 入口，给后面的 `NVWSInsertTmemAref` 创造条件；源码里甚至直接注释了“`aref-tmem-insert requires a single owner`”，见 [HoistTmemStore.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp:103)

一句话：

```text
这是在先整理 TMEM ownership 形态，
否则后面很难可靠地表达“哪个 partition 拥有这块 tensor memory”。
```

#### `NVWSInsertAref`

源码：

- pass 定义: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:90)
- 实现: [InsertAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp:1)
- Aref IR 定义: [NVWSOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td:44)

这个 pass 处理的是一般性的跨 partition producer/consumer 通信，覆盖 tensor、scalar、SMEM producer/consumer。

它引入的核心中间语义是：

```text
nvws.aref.create
nvws.aref.put.enter / nvws.aref.put.exit
nvws.aref.get.enter / nvws.aref.get.exit
```

直观理解：

```text
producer 不再“直接把 SSA value 给 consumer”
而是“进入 put 区间，把值写到 aref 管理的 buffer”

consumer 也不再“直接吃 producer 的 SSA result”
而是“进入 get 区间，从 aref 暴露的 buffer 读”
```

几个关键点：

1. `createAref()` 会把 produced value 包装成 `nvws.aref.create` 管理的 buffer，见 [InsertAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp:98)
2. producer 侧会插 `ArefPutEnterOp` / `ArefPutExitOp`，见 [InsertAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp:214)
3. consumer 侧会插 `ArefGetEnterOp` / `ArefGetExitOp`，并把 direct use 改成对 aref buffer 的间接访问，见 [InsertAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp:395)
4. 对 descriptor load，这个 pass 还会改成 NVWS 自己的 descriptor op，让 load 结果直接写入 aref-owned SMEM buffer，见 [InsertAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAref.cpp:166)

一句话：

```text
`NVWSInsertAref` 把“跨 partition 的直接 SSA 边”
改成“通过 aref + buffer + enter/exit 协议交接”。
```

#### `NVWSInsertTmemAref`

源码：

- pass 定义: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:115)
- 实现: [InsertTmemAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp:1)

它和 `NVWSInsertAref` 的差别是：这里处理的不是一般 shared-memory style value handoff，而是 TMEM ownership transfer。

源码说明很直接：

```text
Insert arefs when TMEM partition ownership changes.
uses ArefPut/ArefGet as ping-pong ownership transfer between two groups.
```

技术上它做的事情更像是：

```text
沿着 TMEM async token/use 链建一个 access DAG，
识别一块 TMEM 在不同 partition / loop / if 分支之间的 ownership 流转，
然后在 ownership change 的边上插 aref put/get。
```

证据见 [InsertTmemAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemAref.cpp:62) 开始的 `TmemAccessDag`。

这一点非常关键，因为 TMEM 不是普通 SSA tensor：

```text
它带 token，带 ownership，带异步完成语义。
partition 之间要转交的不是“一个值”，而是“对一块 tensor memory 的使用权”。
```

另外一个实现约束也值得记：

```text
当前 pass 假设特定 TMEM buffer 最多在两个 groups 之间 ping-pong 转交。
```

这不是抽象结论，而是 pass 描述里写死的当前限制，见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:121)

#### `NVWSLowerAref(numStages)`

源码：

- pass 定义: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)
- 实现: [LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:913)

`NVWSInsertAref` / `NVWSInsertTmemAref` 只是先把跨 partition 依赖抬升成高层 `aref` 语义。
`NVWSLowerAref` 才把这种高层语义继续落成更接近可执行 pipeline 的形式。

从实现看，它至少做了四类事：

1. 给 aref 建立实际的 multi-buffer / barrier 资源，例如 `createAndInitMbar()`，见 [LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:244)
2. 给 aref 相关 op 分配 stage/cluster 信息，并把 partition / ws tag 继续往下传，见 `assignStageCluster()`，[LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:87)
3. 把 `MMAv5` 的 async 性质和 aref/pipelineability 绑定起来，见 `setIsAsync()`，[LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:109)
4. 在 pass 结尾做 `combineArefs`、`multiBufferAref` 和 `LowerArefCreate` rewrite，见 [LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:929)

一句话：

```text
`NVWSInsert*` 负责“把跨 partition 依赖表示成 aref”
`NVWSLowerAref` 负责“把 aref 继续降成具体的 multibuffer + barrier + stage 机制”
```

它之所以在 `PartitionLoops` 之前，是因为：

```text
这一步还需要依赖 loop 内原始的数据流、partition attrs、stage 信息，
先把通信协议和 pipeline 资源准备好，
后面再真正 split 成多个 partition regions。
```

#### `NVWSLowerWarpGroup`

源码：

- pass 定义: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:27)
- IR 定义: [NVWSOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td:171)
- 实现: [LowerWarpGroup.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerWarpGroup.cpp:1)

这个 pass 要和 `PartitionLoops` 配套看。

`PartitionLoops` 并不是直接生成 `ttg.warp_specialize`，而是先生成一个 NVWS 容器：

```text
nvws.warp_group
  region #0: default partition / first group
  region #1..N: other partitions
```

证据见 [PartitionLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp:423)。

然后 `NVWSLowerWarpGroup` 再把这个较高层容器转换成正式的 TritonGPU IR：

```text
nvws.warp_group
  -> ttg.warp_specialize
     + ttg.warp_specialize.partitions
     + ttg.warp_yield / ttg.warp_return
```

这个 lowering 还有几个容易忽略的技术点：

1. 如果 `warp_group` 的第一个 region 使用的是 `globalNumWarps`，它会被当成 `ttg.warp_specialize` 的 default region；否则就不能承接有结果的 default path，见 [LowerWarpGroup.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerWarpGroup.cpp:207)
2. 对 partition region 捕获的外部值，这个 pass 会做 rematerialize 或 shared-memory indirection；tensor capture 不能简单裸捕获，见 [LowerWarpGroup.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerWarpGroup.cpp:117)
3. 它最后创建的是 TritonGPU 的正式 user-facing contract：`WarpSpecializeOp` / `WarpYieldOp` / `WarpReturnOp`，见 [LowerWarpGroup.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerWarpGroup.cpp:158)

一句话：

```text
`PartitionLoops` 先用 NVWS 的分析容器把 loop 拆出来，
`NVWSLowerWarpGroup` 再把这个临时容器落成 TritonGPU 正式 IR。
```

### 4.3 The Missing Middle, Reframed

所以这条链如果只写成：

```text
PartitionScheduling -> PartitionLoops -> ScheduleLoops
```

会漏掉最本质的中间步骤。

更完整的心智模型应该是：

```text
1. PartitionScheduling
   给 loop body 打 partition 决策

2. NVWSHoistTmemStore
   先把 TMEM ownership 形态整理好

3. NVWSInsertAref / NVWSInsertTmemAref
   把跨 partition 依赖改写成显式的 aref / ownership transfer 协议

4. SCCP / CSE / NVWSLowerAref
   清理改写后产生的算术与中间结构，
   并把 aref 进一步落到 multibuffer + barrier + stage 资源

5. PartitionLoops
   真正克隆/拆分 loop，生成 `nvws.warp_group`

6. NVWSLowerWarpGroup
   把 `nvws.warp_group` 变成 `ttg.warp_specialize`

7. ScheduleLoops
   在已经 partitioned 的 loop / warp-specialize 结构上重新建立 schedule contract
```

这才是 `AutomaticWarpSpecialization` 真正完整的中段。

### 5. Relationship To AssignLatencies / ScheduleLoops / Pipeline

结合前面的学习笔记，这条链最好这样记：

```text
AssignLatencies:
  哪些 op 值得跨 stage overlap？

ScheduleLoops:
  这些 op 的 coarse stage / cluster 是什么？

WarpSpecialize:
  哪些 op 应该由不同 warp-groups 并发执行？
  分区之间的 SSA / buffer / loop-carried dependencies 怎么拆？

Pipeline:
  在已有 stage schedule 和 partitions 基础上，
  真正展开 async copy / wait / barrier / prologue / epilogue。
```

所以 `WarpSpecialize` 不是 `Pipeline` 的替代品，而是更高一层的执行组织：

```text
Pipeline 解决时间上的 overlap。
WarpSpecialize 解决“由谁做这件事”的空间分工。
两者最终会叠加。
```

## Part C. Compiler Decision

### 6. The Actual Compiler Question

这个 feature 回答的是下面这个问题：

```text
对于一个 hot loop，
是否值得把 memory / mma / epilogue / scalar/vector compute
拆给不同 warp groups 并发执行？

如果值得：
  哪些 ops 属于哪个 partition？
  分区之间怎样传递 values？
  每个 partition 应该分配多少 warps？
  后续 software pipeline 应该如何在 partitioned loop 上继续工作？
```

这比普通 pipelining 多出三类新决策：

```text
1. Partition assignment
   op -> partition id set

2. Cross-partition communication
   SSA dependencies 改写成 shared memory / aref / multibuffer / loop-carried values

3. Warp budget allocation
   每个 partition 的 num_warps 不是固定等于整个 kernel 的 num_warps
```

### 6.1 Decision Tree

把整条 feature 链压成决策树，大致是：

```text
if loop 没有 tt.warp_specialize / 不满足 automatic path 前提:
  no-op
else:
  run PartitionScheduling
  if 没有得到有价值的 partitions:
    no-op or clear temporary attrs
  else:
    run NVWS helper passes
      - normalize TMEM ownership
      - rewrite cross-partition SSA/value ownership into aref/TMEM protocol
    run PartitionLoops
      - structurally split loop into per-partition regions
    run NVWSLowerWarpGroup
      - lower NVWS container into ttg.warp_specialize
    run ScheduleLoops again
      - rebuild schedule contract on partitioned loop
    optionally run OptimizePartitionWarps
      - shrink num_warps where register/TMEM/TMA constraints still fit
```

### 6.2 Alternative Design

- Alternative:
  完全不做 warp specialization，只保留对称 loop，再靠普通 software pipelining 做时间维 overlap。
- Why not here:
  这种设计无法稳定表达 producer warp-group、MMA warp-group、consumer/epilogue warp-group 的空间分工，也无法把 TMEM/TMA ownership transfer 编进 IR contract。
- Cost of the alternative:
  Blackwell matmul 路径只能吃到时间 overlap，吃不到“空间分工 + 时间 overlap”的组合收益。

## Part D. Internal Contracts

### 7. Internal Attributes

`PartitionAttrs.h` 定义了内部 contract，见 [PartitionAttrs.h](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h:16)：

- `ttg.partition`
- `ttg.partition.outputs`
- `ttg.partition.stages`
- `ttg.warp_specialize.tag`

它们分别表示：

```text
ttg.partition:
  一个 op 属于哪些 partition ids

ttg.partition.outputs:
  多结果 op / loop yield 的每个 output 属于哪些 partitions

ttg.partition.stages:
  partition 级别的 stage 信息

ttg.warp_specialize.tag:
  用来把同一轮 warp specialization 的相关结构关联起来
```

这些 attrs 的语义约束由 `verifyPartitionAttrs()` 检查，见 [Partition.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp:47)。

最关键的 invariant：

```text
如果 loop 带 `tt.warp_specialize`，
它的所有相关 child ops 都必须有 partition attrs；
如果 op 有 children / results，那么 parent / outputs 的 partition 集必须覆盖 child / result 的 partition 集。
```

这其实就是在保证：

```text
partition assignment 必须是一个完整、自洽、可继续下游改写的数据流标注。
```

## Part E. Step 1: PartitionScheduling

### 8. What PartitionScheduling Does

`PartitionScheduling` 是自动 warp specialization 的第一步核心决策，源码总述就在文件头，见 [PartitionScheduling.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:14)：

```text
1. 先把 op 划分成 data ops 和 non-data ops
2. 建 dataflow graph
3. 初始时每个 data op 单独一个 partition
4. 用 heuristics 合并跨 partition edges
5. 再把 non-data ops 传播到需要它们的 partitions
6. 最后把结果 serialize 成 partition attrs
```

几个关键入口：

- `initialDataValues()`：识别初始 data values，重点是 descriptor load、TMEM load、TCGen5 MMA、手动 `data` 标记，见 [PartitionScheduling.cpp:215](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:215)
- `propagateDataValues()`：沿 use-def 传播 data-ness，见 [PartitionScheduling.cpp:247](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:247)
- `initialPartitionAssignment()`：每个 data node 初始独立 partition，见 [PartitionScheduling.cpp:273](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:273)
- `deserializeManualPartitions()`：支持手工分区标记，见 [PartitionScheduling.cpp:305](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:305)

### 9. IR Evidence From Lit Test

最直接的最小证据来自 [partition-scheduling.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:1)。

#### 9.1 `attention_forward`: memory / mma / epilogue 分到不同 partitions

看 [partition-scheduling.mlir:15](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:15) 的 `@attention_forward`：

- `tt.descriptor_load %K_desc` / `ttg.local_alloc %K` 被标到 `ttg.partition = array<i32: 3>`
- `ttg.memdesc_trans` / `ttng.tc_gen5_mma` 被标到 `ttg.partition = array<i32: 2>`
- `ttng.tmem_load %QK_tmem` / `math.exp2 %QK_adj` / `tt.reduce` 被标到 `ttg.partition = array<i32: 0>`
- 一些共享中间结果同时属于多个 partition，例如 `arith.subf` / `arith.mulf`

这说明：

```text
PartitionScheduling 不是只按 op kind 粗暴一刀切。
它允许一个程序天然分裂成 producer / mma / consumer / epilogue 多个 partition，
也允许某些 scalar/vector ops 被复制或共享到多个 partitions。
```

#### 9.2 `mma_operand_view`: 一个共享 memdesc 的不同视图落到不同 partitions

看 [partition-scheduling.mlir:108](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:108) 的 `@mma_operand_view`：

- 同一个 `%K_shared` 会派生出给 MMA 用的 `memdesc_trans` / `memdesc_subslice`
- 也会派生出给 user path 用的 `local_load`
- FileCheck 明确要求前者是 partition 1，后者是 partition 0

这很关键：

```text
partition assignment 的单位不是“整个源 tensor”而是具体 IR use-path。
同一底层 buffer 的不同 consumer path 可以分属不同 partitions。
```

#### 9.3 No-op Cases

同一份 test 里还有两个很重要的反例：

- `@no_partitions`，见 [partition-scheduling.mlir:185](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:185)
- `@mma_no_memory_ops`，见 [partition-scheduling.mlir:198](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:198)

两者都要求：

```text
CHECK-NOT: ttg.partition
CHECK-NOT: ttg.warp_specialize.tag
```

结论：

```text
不是所有带 `tt.warp_specialize` 的 loop 最终都会真的得到 partitions。
如果没有足够的 memory / mma / consumer structure，pass 会选择 no-op。
```

## Part F. Step 2: PartitionLoops

### 10. What PartitionLoops Does

当 `PartitionScheduling` 已经把 loop 里的 op 标好 partition attrs 之后，
`PartitionLoops` 才真正把一个 loop 拆成显式分区结构。

入口很简单，见 [PartitionLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp:533)：

```text
collect scf.for with ttg.partition.stages
  -> partitionLoop(loop)
```

真正的主逻辑是 `partitionLoop(scf::ForOp loop)`，见 [PartitionLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp:354)。

`WarpSpecialization.h` 对它的职责总结得很准确，见 [WarpSpecialization.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h:11)：

```text
rewritePartitionDependencies:
  把分区间 SSA dependencies 改写成 shared memory / ref-semantic / multibuffer 形式

partitionLoop:
  在 dependencies 已经可跨 partition 传递后，
  复制 loop 到各个 partition，并在 root partition 里按需 rematerialize
```

### 11. Why Dependency Rewriting Is Necessary

这是整个 feature 最容易低估的点。

如果只是“把不同 op 复制进不同 regions”，那么跨 partition 的 SSA value 会直接失效：

```text
partition A 定义的 value
  不能继续用普通 SSA 直接喂给 partition B
因为 B 是 isolated-from-above region，
而且它代表的是另一个 warp-group / layout domain。
```

`Partition.cpp` 里的 `iterateInputs()` / `iterateOutputs()` 正是在枚举这种依赖，见：

- [Partition.cpp:171](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp:171)
- [Partition.cpp:201](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp:201)

它区分两类跨分区边：

```text
1. same iteration, different partition
2. subsequent iteration, via scf.yield / iter args
```

这直接对应 warp specialization 的两个核心难点：

```text
同一轮并发分区间通信
循环迭代之间的跨分区循环携带值
```

### 12. IR Evidence From `partition-loops.mlir`

#### 12.1 `multiple_partitions`: 一个 loop 被复制到多个 partition regions

看 [partition-loops.mlir:53](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-loops.mlir:53) 的 `@multiple_partitions`。

原始 loop 里：

- partition 0 使用 `%i`
- partition 1 使用 `%a = addi %i, %i`
- partition 2 使用 `%b = addi %i, %a`

FileCheck 要求 after 里：

- `partition0` 内有自己的 `scf.for`，只 rematerialize 它需要的 `%i`
- `partition1` 内独立重建 `%a`
- `partition2` 内独立重建 `%a` 和 `%b`

这揭示了 `partitionLoop` 的关键策略：

```text
不是把整段 loop body 原封不动拷贝三份。
而是每个 partition 只保留自身需要的子图，并在本分区内重 materialize 必要的 index/scalar ops。
```

#### 12.2 `split_block_arguments`: loop-carried values 按 partition 拆分

看 [partition-loops.mlir:216](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-loops.mlir:216) 的 `@split_block_arguments`：

- 原 loop 有两个 iter args：`%a` 给 partition 0，用于 `op_a`
- `%b` 给 partition 1，用于 `op_b`

after 期望是：

- partition0 内的 `scf.for` 只保留 `%a`
- partition1 内的 `scf.for` 只保留 `%b`

这说明：

```text
PartitionLoops 会把 loop-carried state 做 partition-sensitive 拆分，
避免每个 partition 背无关的 iter args。
```

#### 12.3 `partition_outputs`: 分区输出需要显式 buffer / join

看 [partition-loops.mlir:239](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-loops.mlir:239) 的 `@partition_outputs`：

- test 期待 `ttg.local_alloc`
- 期待 `nvws.warp_group`
- 各 partition 分别执行自己的 loop，然后由 root/default 侧组合输出

这说明：

```text
当 partition 结果需要在 join 点汇合时，
普通 SSA return 不够，需要显式通信/storage 结构。
```

## Part G. Step 3: OptimizePartitionWarps

### 13. What It Decides

`OptimizePartitionWarps` 的问题不是“是否继续分区”，而是：

```text
已经分好的 partitions，
每个分区到底要给多少 warps 才合适？
能不能缩小某些分区的 warp 数，提高总体寄存器预算？
缩完之后 layout 是否也要随之重算？
```

关键入口：

- `optimizePartitionNumWarps()`，见 [OptimizePartitionWarps.cpp:153](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:153)
- `relayoutWarps()`，见 [OptimizePartitionWarps.cpp:86](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:86)

### 14. Core Heuristic

源码的思想很直接：

```text
1. 估算每个 partition 最大 tensor 大致需要多少 i32 registers
2. 假设 PTXAS 会把总寄存器池平均分给所有 warps
3. 尝试把某个 partition 的 numWarps 减半
4. 如果减半后该 partition 仍能容纳它的 tensor register 需求，就接受 shrink
5. 重复直到到达 fixed point
```

其中还有三个硬约束 / 下限：

- 没有 tensor computation 的 partition，尽量缩到 1 warp
- `TMALoadLikeOpInterface` 所在 partition 至少保留 2 warps，见 [OptimizePartitionWarps.cpp:205](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:205)
- TMEM 相关 op 至少保留 4 warps，见 [OptimizePartitionWarps.cpp:208](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:208)

### 15. Relayout After Shrink

这一步很容易忽略，但非常关键。

`relayoutWarps()` 的做法是：

```text
把 partition body 抽进临时 tt.func
  -> 清掉 tensor encodings
  -> 以新的 numWarps 重新跑 ConvertTritonToTritonGPU / Relayout / Coalesce / AccelerateMatmul 等
  -> 再把 body 塞回 partition
```

见 [OptimizePartitionWarps.cpp:23](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:23) 和 [OptimizePartitionWarps.cpp:86](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:86)。

这背后的 compiler contract 是：

```text
partition 的 num_warps 变了，layout contract 也必须跟着重建。
否则原先按 8 warps 合法/高效的 blocked/shared/mma encoding，
放到 4 warps 或 1 warp 的 partition 中就可能不再成立。
```

### 16. IR Evidence From `optimize-partition-warps.mlir`

几个最有代表性的 test：

- `@no_tensor_computations`，见 [optimize-partition-warps.mlir:22](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:22)
  `num_warps(8)` / `num_warps(4)` 都被缩成 `1`
- `@small_tensor_computation`，见 [optimize-partition-warps.mlir:41](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:41)
  小 tensor 计算也能缩到 `1`
- `@fits_after_shrink`，见 [optimize-partition-warps.mlir:136](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:136)
  展示“缩小后仍 fit registers，所以 shrink 成功”
- `@register_use_heuristic`，见 [optimize-partition-warps.mlir:159](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:159)
  直接检查 `requestedRegisters = array<i32: 24, 88>`
- `@tmem_min_4_warps`，见 [optimize-partition-warps.mlir:176](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:176)
  验证 TMEM 相关 partitions 最低保留 `4` warps

这里再补一个实现细节：

```text
`requestedRegisters = array<i32: 24, 88>` 不是实测寄存器分配结果，
而是 pass 内部写回到 `ttg.warp_specialize` 上的 heuristic estimate。
源码是 `tensorRegs ? 88 : 24`，见 [OptimizePartitionWarps.cpp:261](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:261)。
```

## Part H. What The Automatic Lit Test Proves

### 17. `automatic-warp-specialization.mlir` Is The Best End-to-End Evidence

如果只看一份 test，我认为最该看 [automatic-warp-specialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:1)。

它直接把整条链串了起来：

```text
hoist-tmem-alloc
-> assign-latencies
-> schedule-loops
-> automatic-warp-specialization
-> optional pipeline
-> optional optimize-partition-warps
```

#### 17.1 `matmul_change_desc_in_prologue`

看 [automatic-warp-specialization.mlir:21](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:21)：

- 输入 loop 带 `{tt.warp_specialize, tt.num_stages = 2}`
- after 出现 `ttg.warp_specialize`
- `partition0` 最终被优化成 `num_warps(1)`
- `partition1` 最终被优化成 `num_warps(2)`
- pipeline 版本里，partition 中出现 `async_tma_copy_global_to_local` 和 `tc_gen5_mma`

这说明：

```text
自动 warp specialization 的目标并不是只把 loop 平铺复制。
它在为后续 pipeline 预先组织出 producer / consumer / mma 角色分工。
```

#### 17.2 `attention_forward`

看 [automatic-warp-specialization.mlir:203](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:203)：

- FileCheck 期待 `ttg.warp_specialize`
- pipeline 版本要求 `partition1` 中有多次 `tc_gen5_mma`
- `partition2` 中有多次 `ttng.async_tma_copy_global_to_local`

这几乎就是 warp specialization 的最典型角色拆分：

```text
一组 partitions 专注发起内存搬运
另一组 partitions 专注 tensor core compute
default/root partition 处理 consumer / reduction / epilogue
```

## Part I. Cross-Architecture View

### 18. Why This Feature Is Blackwell-Centric Today

前端文档已经明确写了：

```text
warp specialization is only supported on Blackwell GPUs
and only works on simple matmul loops
```

见 [core.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/language/core.py:3717)。

这和目前测试覆盖是吻合的：

- 自动 pass 的主力 lit tests 都是 `ttg.target = "cuda:100"`，见 [automatic-warp-specialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:1)
- runtime 单测允许 Hopper/Blackwell 上测试显式 `ttg.warp_specialize` IR，但自动 loop-level 特性更偏 Blackwell，见 [test_warp_specialization.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:24)

所以当前阶段更准确的表述是：

```text
显式 `ttg.warp_specialize` IR 和其 lowering 不是绝对只属于 Blackwell；
但 AutomaticWarpSpecialization 这条 loop-level 自动化路径，
当前明显是围绕 Blackwell TMEM / TMA / TCGen5 matmul 模式构建的。
```

这和我们前面在 `AssignLatencies` / `ScheduleLoops` / `Pipeline` 里看到的现象一致：

```text
真正成熟的 warp-specialized pipeline contract，
现在主要出现在 Blackwell MMAv5/TMEM 路径上。
```

所以从 GUIDE 的“三代 before/after 横向对比”角度看，这个 pass 有一个特殊结论：

```text
AutomaticWarpSpecialization 当前并不适合做 Ampere / Hopper / Blackwell
三代并排的 before/after 学习样本。

原因不是资料不全，而是 feature 本身就是明显的 Blackwell-centric 自动路径：
Ampere 上 automatic path 不具现实意义，
Hopper 上更常见的是显式 `ttg.warp_specialize` / lowering 相关能力，
真正完整的 automatic loop partitioning 主要落在 Blackwell。
```

## Part J. Contracts Summary

### 19. Input Contract

对 automatic warp specialization 来说，进入 pass 前通常需要：

- loop 已经被标记 `tt.warp_specialize`
- loop 足够像“可分解的 matmul / memory-mma-consumer”结构
- 更早的 layout / matmul lowering 已经建立好基本 GPU contract
- 如果要继续和 SWP 叠加，最好已有 `AssignLatencies` / `ScheduleLoops` 产出的 schedule 信息

### 20. Output Contract

执行后建立的 contract 分两层：

#### 20.1 Internal temporary contract

- `ttg.partition`
- `ttg.partition.outputs`
- `ttg.partition.stages`
- `ttg.warp_specialize.tag`

这是 `PartitionScheduling` 到 `PartitionLoops` 之间使用的内部标注。

#### 20.2 Durable IR contract

- `ttg.warp_specialize`
- `ttg.warp_specialize.partitions`
- 每个 `partitionN ... num_warps(k)`
- 经重写后的 shared-memory / aref / multibuffer / loop-carried communication

后续 pass 可以依赖这些事实：

```text
不同 partitions 已经是隔离的 warp-group 执行域
cross-partition dependencies 已经被合法重写
每个 partition 的 num_warps 已经定下或进一步优化过
```

#### 20.3 Deferred Work

这一步刻意不解决的事也要单独记：

- 最终 async 指令序列、prologue/epilogue 展开：
  主要留给 `Pipeline`
- `ttg.warp_specialize` 到 LLVM/NVVM-level control-flow / barrier / capture frame 的 lowering：
  留给 `ConvertWarpSpecializeToLLVM`
- 最终机器级寄存器分配：
  `OptimizePartitionWarps` 只能做 heuristic estimate，不是 PTXAS 实测分配
- 更晚阶段的 resource/barrier materialization：
  留给后面的 TMEM/barrier/allocation 相关 passes

### 21. Invariants

无论怎么 partition，下面这些语义不能变：

- 数学结果不变
- loop iteration 语义不变
- producer/consumer 依赖语义不变
- 只改变执行组织方式，不改变程序的逻辑含义

改变的内容是：

- 谁执行哪段代码
- 哪些 value 通过 shared/ref-semantic/buffer 传递
- 每个 partition 的 warp 数
- 后续 pipeline / lowering 看到的控制流和同步结构

### 21.1 If This Pass Did Not Exist

如果没有 automatic warp specialization，这类 loop 会退化成更传统的执行组织：

```text
所有 warps 更对称地执行同一套 loop body，
编译器最多只能依赖普通 software pipelining 在时间维做 overlap。
```

缺掉的能力主要是：

```text
1. 不能把 TMA / descriptor / producer 工作稳定拆给专门 warp-groups
2. 不能把 TCGen5 MMA / TMEM 相关路径与 consumer / epilogue 明确解耦
3. 无法在 Blackwell matmul 路径上同时利用“空间分工 + 时间 overlap”
4. 后续 Pipeline / lowering 也就拿不到 `ttg.warp_specialize` 这一层执行组织 contract
```

## Part K. Runtime / Explicit IR Tests

### 22. What The Python Tests Add

[test_warp_specialization.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:24) 补了另外一类证据：

- `test_warp_specialize_basic_ir`：最小显式 `ttg.warp_specialize` IR 可以编译并执行，见 [test_warp_specialization.py:24](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:24)
- `test_warp_specialize_tmem_ir`：显式 TMEM + warp specialize 路径可运行，见 [test_warp_specialization.py:59](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:59)
- `test_warpgroup_reduction`：不同 partitions 允许不同 `num_warps`，见 [test_warp_specialization.py:127](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:127)
- `matmul_tma_ws_kernel`：真实 kernel 级用法，见 [test_warp_specialization.py:214](/LocalRun/jiangzhe.zhao/my_repo/triton/python/test/unit/language/test_warp_specialization.py:214)

这些测试说明：

```text
`ttg.warp_specialize` 不只是 analysis artifact。
它是一个真正会进入 codegen、并对运行结果负责的 IR abstraction。
```

## Part L. My Current Mental Model

### 23. The Best Short Model

目前我认为最稳的记忆方式是：

```text
AutomaticWarpSpecialization
  = 把一个 loop 变成“多 warp-group 协作程序”的 pass pipeline

PartitionScheduling
  = 先决定每个 op 属于哪个 worker role

PartitionLoops
  = 再把这些 roles 实体化成 `ttg.warp_specialize` regions

OptimizePartitionWarps
  = 最后压缩各 worker role 的 warp 预算，并重建对应 layout
```

换成更贴近硬件的话：

```text
普通 loop:
  所有 warps 做相似工作

warp-specialized loop:
  一部分 warps 偏 memory producer
  一部分 warps 偏 tensor-core consumer/compute
  一部分 warps 偏 epilogue / scalar work
  编译器负责把它们的并发执行、通信和资源预算安排清楚
```

### 23.1 Knowledge Card

```text
Pass / Feature:
  AutomaticWarpSpecialization (+ optional OptimizePartitionWarps)

Purpose:
  把 loop 变成多 warp-group 协作程序

Compiler decision:
  谁归哪个 partition、怎么交接、每个 partition 分多少 warps

Main IR attribute/op:
  temporary: ttg.partition / ttg.partition.outputs / ttg.partition.stages / ttg.warp_specialize.tag
  durable:   ttg.warp_specialize

Input contract:
  loop 已有 warp_specialize trigger，且更早的 layout/matmul/schedule 信息已就位

Output contract:
  partitioned warp-group execution domains + legal cross-partition communication + per-partition num_warps

Invariant:
  数学语义、迭代语义、tensor shape/type 不因 partitioning 而改变

Hardware reason:
  为 Blackwell-centric TMA / TMEM / TCGen5 matmul 路径建立空间分工 contract

Next dependencies:
  ScheduleLoops / Pipeline / OptimizePartitionWarps / ConvertWarpSpecializeToLLVM
```

## Part M. Open Questions

### 24. Still Worth Following Up

这轮已经能把主线讲清，但还有几件事值得后续单独深挖：

1. `PartitionScheduling` 里“跨 partition edge 合并 heuristics”具体每条规则是什么。
   这一轮只把总体图算法和 test 证据理清了，还没逐条走 merge heuristic。

2. `rewritePartitionDependencies()` 的 shared memory / aref / multibuffer 具体 IR 改写细节。
   这是 warp specialization 最难的一段，值得单开一篇。

3. `ConvertWarpSpecializeToLLVM` 如何把 `ttg.warp_specialize` 进一步 lower 成实际 warpgroup control flow、barrier、capture frame、register reallocation。
   这会更靠近最终 codegen。

4. Hopper 和 Blackwell 在“显式 `ttg.warp_specialize` lowering”上的差异。
   当前自动化路径明显偏 Blackwell，但 explicit IR lowering 还有跨架构细节可以继续拆。
