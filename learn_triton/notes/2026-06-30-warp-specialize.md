# 2026-06-30 学习笔记：WarpSpecialize

这篇笔记按新版
[IR_PASS_DIFF_LEARNING_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/IR_PASS_DIFF_LEARNING_GUIDE.md:1)
重写。

但先说明一个边界：

```text
WarpSpecialize 不是单个源码文件里的单一 rewrite pass。
用户看到的是一个 compiler feature；
实现上是 AutomaticWarpSpecialization orchestration pass
+ 可选的 OptimizePartitionWarps
+ 后续 ScheduleLoops / Pipeline / LLVM lowering 对其结果的消费。
```

所以这篇笔记采用：

```text
feature-level 主卡片
  + 关键子 pass 的 decision / contract
  + 真实 dump 反例
```

而不是机械地把 “WarpSpecialize” 当成一个独立源码文件来写。

## 1. Pass 基本信息

### 1.1 Feature 入口

- 前端 loop 入口：
  [python/triton/language/core.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/language/core.py:3709)
- 显式 IR builder 入口：
  [python/triton/experimental/gluon/language/_semantic.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/experimental/gluon/language/_semantic.py:587)
- orchestration pass：
  [lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:1)
- 相关 pass 声明：
  [include/triton/Dialect/TritonGPU/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:83)
- `ttg.warp_specialize` IR 定义：
  [include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:600)

### 1.2 主要证据

- feature-active lit tests：
  [test/TritonGPU/automatic-warp-specialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:1)
  [test/TritonGPU/partition-scheduling.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-scheduling.mlir:1)
  [test/TritonGPU/partition-loops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/partition-loops.mlir:1)
  [test/TritonGPU/optimize-partition-warps.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/optimize-partition-warps.mlir:1)
- canonical dump 反例：
  [060_Before_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:1)
  [083_After_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:1)

### 1.3 当前 pipeline 中的真实 pass 链

`AutomaticWarpSpecialization` 内部不是一个 pass，而是一个 pass manager：

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

证据：
[AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:86)

补充：

```text
OptimizePartitionWarps 不在这条内部链里。
它是 AutomaticWarpSpecialization 之后可单独追加的后处理 pass。
```

证据：
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:118)

## 2. Cross-Architecture Framing

### 2.1 架构矩阵

| Arch | Automatic path status | 本篇证据价值 |
|---|---|---|
| Ampere `sm80/sm86` | 当前前端入口不支持自动 path | 不是主样本 |
| Hopper `sm90` | 更常见是显式 `ttg.warp_specialize` 及其后续 lowering | 适合学 IR 语义，不适合作为 automatic 主样本 |
| Blackwell `sm100` | 前端明确支持 automatic warp specialization | 本篇主样本 |

最直接的证据来自前端文档：

```text
warp specialization is only supported on Blackwell GPUs
and only works on simple matmul loops
```

见
[python/triton/language/core.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/language/core.py:3717)

### 2.2 Cross-architecture before comparison

这题最重要的 cross-arch 结论不是“三代 dump 并排 diff”。

更准确的结论是：

```text
Automatic warp specialization 本身就是 Blackwell-centric feature。
所以这篇笔记的 cross-arch 边界主要体现为：
  Ampere: automatic path 不成立
  Hopper: explicit IR / lowering 更重要
  Blackwell: automatic partitioning 才是主学习对象
```

这和一般 pass 不同。这里“输入 IR 已经在架构能力边界上分叉”是先验事实，不是当前 pass 才造成的。

## 3. Problem / Goal / Constraint / Design Intent / Decision

### 3.1 Problem

普通 software pipelining 主要解决：

```text
同一组 warps 在时间维上怎样 overlap load 和 compute
```

但 Blackwell matmul-like loop 又多出一个更强的机会：

```text
不同 warp groups 是否可以空间分工：
  一组负责 memory / TMA / descriptor update
  一组负责 MMA / TMEM compute
  另一组负责别的 producer / consumer 子任务
```

如果仍保持“所有 warps 对称地跑同一段 loop body”，编译器就无法显式表达：

- 哪些 op 属于哪个 warp-group
- 分区之间如何传值
- 哪块 TMEM / SMEM 的 ownership 怎么交接
- 后续 Pipeline / lowering 该以什么并发结构为前提

### 3.2 Goal

建立下面这个执行组织：

```text
一个原本单体的 loop
  -> 多个并发 warp-group partitions
  -> 每个 partition 有自己的 warp 数、局部控制流、layout 域
  -> 分区之间通过显式协议通信
```

### 3.3 Constraint

`Constraint.source`:

- `ttg.warp_specialize` 的 partition region 必须 `IsolatedFromAbove`
- TMEM/TMA 路径不是普通 SSA 值，它带 token、ownership 和异步完成语义
- default region 和 partition regions 的 warp 数可以不同，因此 layout 域也可能不同

`Constraint.manifestation`:

- 不能直接保留 “producer in partition A -> consumer in partition B” 的裸 SSA edge
- 某些 tensor result 如果只在非 default partition 里算出，不能直接作为 default result 返回
- `PartitionLoops` 真正 split 之前，必须先把跨 partition 依赖改写成 buffer / aref / ownership transfer

### 3.4 Design intent

Triton 这里没有直接把 loop 一刀切成 `ttg.warp_specialize`。

它采用的是分阶段设计：

```text
先决定谁属于哪个 partition
-> 再把跨 partition 依赖改写成显式协议
-> 再 clone/split loop
-> 再把临时 NVWS 容器 lowering 成 TritonGPU 正式 IR
```

这个设计比“直接 split loop”更合理，因为 split 之前还需要保留原始 loop 的数据流信息来构造 aref、multibuffer、barrier 和 ownership transfer。

### 3.5 Compiler decision

这个 feature 回答的 compiler question 是：

```text
这个 hot loop 是否值得拆成多个并发 warp-group partitions？

如果值得：
  1. 哪些 op 属于哪个 partition？
  2. 跨 partition 的 value / TMEM ownership 如何交接？
  3. 哪些 loop-carried values 要保留、删除、或改走 SMEM？
  4. 每个 partition 最终给多少 warps？
```

## 4. Before / After IR 变化

### 4.1 Feature-active 样本：普通 `scf.for` -> `ttg.warp_specialize`

在
[automatic-warp-specialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/automatic-warp-specialization.mlir:1)
里，输入是带 `{tt.warp_specialize}` 的 `scf.for`。

After IR 的关键 contract 是：

- 出现 `ttg.warp_specialize`
- 有 `default` region
- 有若干 `partitionN ... num_warps(k)` regions
- 清掉内部属性：`ttg.partition` / `ttg.warp_specialize.tag`

这说明 durable after-IR 不是内部 attrs，而是显式的 warp-specialize op。

### 4.2 No-op 样本：loop 被标记，但不满足 partition 条件

同一个测试文件里还有反例：

- `@no_eligible_memory_ops` 仍保留普通 `scf.for`
- after IR 不出现 `ttg.warp_specialize`
- 也不留下 `ttg.partition` / `ttg.warp_specialize.tag`

这说明：

```text
有 loop attr != 一定形成 warp-specialize 结构
```

对应源码逻辑在 `PartitionScheduling::analyze()`：

```text
if (!hasEligibleMemoryOps(graph.get()))
  return;
serialize(idx, op, graph.get());
```

证据：
[PartitionScheduling.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:1519)

### 4.3 Canonical dump 反例：pass 部分生效，但没有形成 partition

在
[060_Before_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUAutomaticWarpSpecialization.mlir:73)
和
[083_After_TritonGPUAutomaticWarpSpecialization.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/083_After_TritonGPUAutomaticWarpSpecialization.mlir:73)
之间，没有出现 `ttg.warp_specialize`，但 loop 确实改了。

最显眼的变化：

```text
Before:
  iter_args(..., %acc_52 = %acc, %acc_53 = %acc_38)

After:
  iter_args(..., %acc_51 = %false, %acc_52 = %acc_37)
```

以及对应 `ttng.tc_gen5_mma` 的 use-D flag / token threading 都随之调整。

这说明：

```text
AutomaticWarpSpecialization 在真实 pipeline 里可能是
“partially effective but no partition formed”
而不是简单的 no-op。
```

这个反例很重要，因为它提醒我：

- automatic feature 的最终目标是 `ttg.warp_specialize`
- 但内部 NVWS / TMEM helper pass 即使没产出最终结构，也可能先做局部规范化

## 5. Per-Pass Cards

这一节只做一件事：把这条 feature 链上的每个 pass 单独写成
`Problem / Goal / Constraint / Design Intent / Decision`。

### 5.1 `AutomaticWarpSpecialization`

证据：
[AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:86)

- Problem:
  `warp specialization` 不是单一步骤能完成的 rewrite。partition assignment、跨 partition 通信、loop split、正式 IR materialization、重新调度，都依赖不同中间 contract。
- Goal:
  把一个被标记为 `tt.warp_specialize` 的 loop，沿着一条受控 pipeline 变成后续 pass 可消费的 warp-specialized IR。
- Constraint:
  各步骤之间有严格前后依赖；部分阶段需要验证 partition invariants；结束后还要清掉内部 attrs，避免中间 contract 泄漏到最终 IR。
- Design intent:
  用 orchestration pass 而不是单个大 rewrite，把复杂问题拆成一串小 decision，并在关键点插 verifier。
- Decision:
  这个 pass 不直接决定具体 partition 内容；它决定“按什么顺序运行哪些子 pass，以及何时验证和清理中间状态”。

### 5.2 `PartitionScheduling`

证据：
[PartitionScheduling.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp:1443)

- Problem:
  before IR 只有一个带 `tt.warp_specialize` 的 loop，但还没有回答“哪些 op 应该由哪个 warp-group 执行”。
- Goal:
  给 loop body 中的 op 建立 partition assignment，并把 partition 信息序列化回 IR。
- Constraint:
  partition 不能只按语法块划分，而要看 dataflow、memory op、MMA、view-like op、以及多 partition data op 的合法性；如果根本没有 eligible memory ops，就不值得 serialize。
- Design intent:
  先 build graph，再做初始分配、merge、propagate、assign partition ids，最后把决定写成 attrs，而不是立刻 clone/split IR。
- Decision:
  这个 pass 回答“谁属于哪个 partition、每个 partition 的 stage/output 是什么、当前 loop 是否值得继续进入 warp-specialize 流水线”。

### 5.3 `NVWSHoistTmemStore`

证据：
[third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:134)

- Problem:
  嵌套 loop 中的 `ttng.tmem_alloc/store` 形态可能把 accumulator clear 固定在内层，导致之后的 TMEM ownership transfer 需要额外同步。
- Goal:
  尽可能把 TMEM alloc/store hoist 到更外层，并把 token 正确 thread 过 loop nest。
- Constraint:
  只有在 hoist 后仍保持语义正确时才成立；特别是内层 loop 执行次数必须可证明至少一次，或不影响 use-D / zero-clear 语义。
- Design intent:
  先把 TMEM ownership 形态整理成更稳定的“单一 owner + token threading”模式，再交给后面的 TMEM aref pass。
- Decision:
  这个 pass 回答“当前这组 TMEM alloc/store 是否应该提升到更外层，以消除不必要的 partition 间同步并稳定 ownership 入口”。

### 5.4 `NVWSInsertAref`

证据：
[third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:90)

- Problem:
  一般性的 tensor/scalar/SMEM producer 和 consumer 一旦被分到不同 partition，裸 SSA edge 不能再直接表达同步和 buffer ownership。
- Goal:
  在 producer/consumer partitions 之间插入 `aref` 协议，把直接依赖改写成显式的 put/get 通信。
- Constraint:
  partition region 最终要 `IsolatedFromAbove`；descriptor load 这类 op 还需要改写成能直接写入 aref-owned buffer 的形式。
- Design intent:
  用 `ArefPutEnter/Exit` 和 `ArefGetEnter/Exit` 把跨 partition 依赖变成显式协议，而不是等 split 后再补救。
- Decision:
  这个 pass 回答“哪些跨 partition 依赖应该用一般 aref 机制承载，以及 producer/consumer 两端具体在哪插 put/get 边界”。

### 5.5 `NVWSInsertTmemAref`

证据：
[third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:115)

- Problem:
  TMEM 不是普通 value handoff；partition 之间转交的是一块 tensor memory 的 ownership 和 token 链。
- Goal:
  在 TMEM ownership 改变的位置插入 aref，把 ownership transfer 显式化。
- Constraint:
  当前实现只支持某个 TMEM buffer 在不超过两个 groups 之间 ping-pong 转交；还要沿着 token/use 链识别真实 ownership 变化边。
- Design intent:
  把“TMEM 被谁拥有、什么时候交给谁”单独建模，而不是把它塞进一般 value aref 逻辑里。
- Decision:
  这个 pass 回答“哪些 TMEM access 边界构成 ownership change，因此必须插入 TMEM-specific aref transfer”。

### 5.6 `SCCP`

- Problem:
  前面的 partition attrs、aref insertion、TMEM canonicalization 会制造额外条件、flag threading 和死路径。
- Goal:
  折叠常量与死分支，简化后续 aref lowering 和 loop split 的输入。
- Constraint:
  它不是 warp-specialize 专属 pass，只能利用当前 IR 已暴露出的常量信息，不能发明新的 partition semantics。
- Design intent:
  在真正 split loop 前先清掉显然可折叠的算术/控制流，减少后续 pass 处理噪音。
- Decision:
  这个 pass 回答“哪些新引入的控制/数据流已经退化成 compile-time constant，可直接消掉”。

### 5.7 `CSE`

- Problem:
  前几步会引入重复算术、重复 helper op、以及局部等价表达式，放任不管会放大后续 split 和 lowering 的复杂度。
- Goal:
  删除冗余公共子表达式，让后续 pass 在更干净的 IR 上工作。
- Constraint:
  只能消除语义等价且安全合并的表达式；它不拥有 partition decision，也不应改变同步协议。
- Design intent:
  把 CSE 放在 `NVWSLowerAref` 前，是为了先收缩中间 IR，再把剩余依赖物化为 barrier/resource 结构。
- Decision:
  这个 pass 回答“哪些前面引入的中间值已经重复到可以合并，从而降低后续 pass 的工作量”。

### 5.8 `NVWSLowerAref`

证据：
[third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)

- Problem:
  高层 `nvws.aref.*` 只是“需要通信”的逻辑语义，还不是后续 pipeline 能直接消费的 barrier / stage / multibuffer 结构。
- Goal:
  把 aref lowering 成匹配 value/barrier 集合，并决定 appropriate waits/signals、buffer stages、phase 等资源。
- Constraint:
  lowering 必须同时保持 producer/consumer 的 use-def 关系、empty/full 状态语义，以及 pipeline 的 `numStages` 约束。
- Design intent:
  先用 aref 把跨 partition 依赖抽象出来，再在这里统一 materialize 成 barrier-centric contract，而不是把 barrier 插入逻辑散落在多个 pass 里。
- Decision:
  这个 pass 回答“每条 aref communication 需要哪些 barrier/multibuffer resources，以及怎样安排 wait/signal/stage 才能合法执行”。

### 5.9 `PartitionLoops`

证据：
[PartitionLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp:423)

- Problem:
  到这一步为止只有 partition attrs 和 communication protocol，还没有真正把 loop 变成多个隔离 region。
- Goal:
  clone/split 原始 loop，并把结果先组织成 `nvws.warp_group`。
- Constraint:
  partition 之间不能再保留非法 direct SSA dependency；非 default partition 计算出的 tensor result 若需要返回，必须通过 SMEM 间接回传。
- Design intent:
  先在 NVWS 容器里完成结构拆分和结果重组，再交给下一个 pass 统一 materialize 成 TritonGPU 正式 op。
- Decision:
  这个 pass 回答“原始 loop 的哪些 iter_args/results 在每个 partition 中保留、删除、还是改走 SMEM，以及如何把单体 loop 结构拆成 partitioned region graph”。

### 5.10 `NVWSLowerWarpGroup`

证据：
[third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:27)

- Problem:
  `nvws.warp_group` 仍是 NVWS 中间容器，不是后续 TritonGPU pipeline 要长期依赖的正式 IR contract。
- Goal:
  把 `nvws.warp_group` 变成 `ttg.warp_specialize`、`ttg.warp_specialize.partitions`、`ttg.warp_yield`、`ttg.warp_return`。
- Constraint:
  default region 只有在第一个 warp group 匹配全局 `ttg.num-warps` 时才能自然承接；否则只能留下空 default 或纯 isolated partitions。
- Design intent:
  把中间容器和用户可见 contract 分离，避免 NVWS 特定结构泄漏到 TritonGPU 的稳定 IR 边界。
- Decision:
  这个 pass 回答“当前 `nvws.warp_group` 应如何映射成合法的 `ttg.warp_specialize` 结构，以及谁成为 default region”。

### 5.11 `ScheduleLoops`

证据：
[include/triton/Dialect/TritonGPU/Transforms/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:43)
[lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:398)

- Problem:
  loop 在 split/partition 之后，原先的 coarse schedule 已经不再是最终执行结构的准确描述。
- Goal:
  基于当前 latency annotations 和新形成的 partitioned loop，重新序列化 coarse schedule。
- Constraint:
  它只能消费已经存在的 latency/schedule 信息，不能替代 partition assignment 或 communication lowering；如果前面 contract 没建好，它也无从调度。
- Design intent:
  把“谁做什么”先定下来，再重新回答“这些事按什么 stage/cluster 顺序发生”。
- Decision:
  这个 pass 回答“在已经 warp-specialized 的 loop 上，哪些 op 属于哪个 coarse stage，以及新的执行顺序应如何写回 IR”。

### 5.12 `OptimizePartitionWarps`

证据：
[OptimizePartitionWarps.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp:138)

- Problem:
  `ttg.warp_specialize` 已经形成后，各 partition 默认沿用较大的 warp 预算，可能造成不必要的寄存器竞争和资源浪费。
- Goal:
  在保持 partition 语义不变的前提下，尽量缩减各 partition 的 `num_warps`。
- Constraint:
  受寄存器压力估计、总寄存器池、以及硬件下界约束影响；例如 TMA-like partition 至少 2 warps，TMEM load/store/alloc 至少 4 warps。
- Design intent:
  先形成正确的 warp-specialize 结构，再做资源层的后处理，而不是把 warp 数优化混进前面的结构化 passes。
- Decision:
  这个 pass 回答“每个 partition 是否可以安全地把 `num_warps` 降到一半，以及降到什么下界仍满足 register/hardware 约束”。

### 5.13 两个不是 pass 的收尾步骤

- `multiBufferTMADescriptors`:
  回答的是“nested loops 里的 descriptor update 需要怎样的额外 multibuffer 才安全”。
- `clearInternalWarpSpecializationAttrs`:
  回答的是“哪些内部 attrs 只是中间 contract，必须在 feature 结束时清掉”。

它们重要，但不是独立 MLIR pass。

## 6. Compiler Contract

### 6.1 Input contract

- loop 已被标记为 `tt.warp_specialize`，或 IR 中已经显式存在 `ttg.warp_specialize`
- 更早的 matmul / TMEM / TMA / scheduling 结构已经显式化到足以分析
- 对 automatic path 而言，当前 loop 至少要有可参与 partition 的 memory/computation pattern

### 6.2 Output contract

对 automatic path 成功样本，durable after-IR 是：

- `ttg.warp_specialize`
- `partitionN ... num_warps(k)`
- default / partition region 边界
- 合法的 capture / return / yield 结构

对 automatic path 的内部中间阶段，temporary contract 是：

- `ttg.partition`
- `ttg.partition.outputs`
- `ttg.partition.stages`
- `ttg.warp_specialize.tag`
- NVWS aref / warp_group 等中间语义

### 6.3 Next pass relies on

- `ScheduleLoops` 重新建立 partitioned 结构上的 schedule contract
- `Pipeline` 再在已有 partition + stage 基础上做时间维 overlap
- `ConvertWarpSpecializeToLLVM` / 更后面的 lowering 消费 `ttg.warp_specialize`

### 6.4 Deferred work

这个 feature 不负责：

- 最终 LLVM/NVVM lowering
- 最终寄存器分配
- PTXAS 真实 `nreg` 结果
- 全部 barrier/fence 的最终硬件化细节

这些都留给后续 passes 或后端工具链。

## 7. Invariant

- Tensor logical shape 不因 warp specialization 改变
- Element type 不以 partitioning 为目的改变
- 程序数学语义不变
- 改变的是：
  layout domain、执行角色划分、跨 partition 交接方式、loop-carried state 组织、warp 数预算

## 8. Decision Tree

把整条 feature 压缩成决策树，可以记成：

```text
if loop has no `tt.warp_specialize`:
  ignore

else:
  PartitionScheduling
    if no eligible memory ops:
      do not serialize partitions
      remain ordinary loop
    else:
      assign partitions / stages / outputs
      duplicate multi-partition data ops when needed

  NVWS helper passes
    if cross-partition direct SSA / TMEM ownership would be illegal:
      rewrite through aref / ownership transfer / multibuffer / barriers

  PartitionLoops
    if partition count <= 1:
      keep loop
    else:
      split into `nvws.warp_group`
      move non-default tensor results through SMEM if needed

  NVWSLowerWarpGroup
    lower to `ttg.warp_specialize`

  ScheduleLoops
    reschedule on partitioned structure

  optional OptimizePartitionWarps
    shrink `num_warps` if register + hardware limits allow
```

## 9. Triton Mechanism vs Hardware Reason

### 9.1 Triton mechanism

在 Triton IR 层，这个 feature 的职责是：

```text
把 loop 从“统一执行体”
改写成“多个异步 warp-group region”
并建立合法的 IR-level communication / ownership / result contract
```

### 9.2 Hardware / execution reason

这不是泛泛的“CUDA 优化”。
它背后的硬件动机更具体：

- Blackwell 允许把 warp-group / TMEM / TMA 路径更显式地编排
- 某些 producer 和 consumer 天然适合空间分工，而不只是时间重排
- 不同 partition 可以需要不同数量的 warps
- TMA/TMEM 对 ownership、token、参与 warp 数都有硬约束

所以这题的本质不是单纯 layout，也不是单纯 pipelining，而是：

```text
execution ownership + communication protocol + resource partitioning
```

## 10. Alternative Design

一个自然问题是：

```text
为什么不只靠 Pipeline？
```

因为 Pipeline 主要回答：

```text
同一批 warps 在时间维上怎样 overlap
```

而 WarpSpecialize 回答的是：

```text
哪些 warps 根本不该做同样的事
```

只靠 Pipeline，编译器仍难以表达：

- 独立 warp-group 的代码区域
- 不同 partition 的 warp 数
- partition 间显式 ownership transfer

所以两者不是替代关系，而是：

```text
WarpSpecialize: 空间分工
Pipeline: 时间重排
```

## 11. 如果没有这个 feature，会怎样

如果没有 automatic / explicit warp specialization：

- loop 仍可做普通 scheduling / pipelining
- 但无法把 memory / mma / epilogue 明确拆成并发 warp-group roles
- 不能显式建立跨 partition 的 buffer/ownership contract
- 后续 lowering 也拿不到 `ttg.warp_specialize` 这类强执行结构

在 Blackwell matmul-like hot loop 上，这通常意味着：

```text
只能做“同构 warps + 时间 overlap”
做不到“异构 warp-groups + 时间 overlap”的叠加
```

## 12. 最后结论

我现在认为，这题最稳的心智模型是：

```text
WarpSpecialize 不是一个“把 loop 变快”的单点 pass。

它是一个分阶段 feature：
  1. 识别哪些工作应该由不同 warp-groups 执行
  2. 把跨 partition 依赖改写成显式协议
  3. 把 loop 结构拆成并发 regions
  4. 把该结构变成后续 pipeline / lowering 可长期依赖的 IR contract
  5. 必要时再重新分配每个 partition 的 warp 预算
```

因此学习它时，最应该记住的不是某个单独 rewrite，而是下面这条因果链：

```text
Blackwell execution opportunity
  -> symmetric-loop model is too weak
  -> compiler must choose warp-group roles
  -> direct SSA edges become insufficient
  -> explicit communication / ownership protocol is introduced
  -> loop is split into warp-specialized regions
  -> later passes consume that contract
```
