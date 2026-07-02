# Triton TTGIR 中的 Target-Driven Scheduling

目标：把
[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:60)
里那句

```text
TTGIR = distributed execution mapping
      + layout / data-movement organization
      + target-driven scheduling
```

真正落成一条可验证的源码阅读路径。

这篇文档回答的问题不是“有哪些 barrier / fence”，而是：

```text
同一个 kernel 的主循环，
为什么在不同 target 上会形成不同的 load / compute / overlap / synchronize 结构？

这些结构分别由哪些 pass 决定，
又由哪些更后面的 pass 具体物化成 barrier / wait / mbarrier / fence？
```

同步原语本身的种类、scope、可见性、proxy 语义，请看
[2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md:1)。
那篇是“同步契约手册”；本文是“调度决策与契约落地总览”。

---

## 1. 一句话结论

```text
target-driven scheduling
  = 编译器根据目标架构的执行单元、异步引擎、可见同步原语和内存路径，
    决定主循环里哪类工作应该提前、分区、重叠，以及最终通过哪套协议交接。
```

在 Triton 里，这件事不是单个 pass 完成的，而是三层职责分工：

1. **schedule decision**
   决定哪些 op 值得跨 stage overlap，哪些 op 属于哪个 stage / partition。
2. **protocol abstraction / materialization**
   把这些决策变成可执行的 async copy、wait、mbarrier、warp-specialize region、
   multi-buffer state、loop prologue / steady-state / epilogue。
3. **hazard repair / target-specific synchronization completion**
   在更后面补齐 generic-vs-async proxy、TMEM reuse、shared memory RAW/WAR/WAW
   等约束，确保前面的调度真的合法。

如果把第 3 层误认为第 1 层，就会把 “为什么这样调度” 错看成 “后面补了哪些 barrier”。

---

## 2. 它不是什么

### 2.1 不是“barrier/fence 的别名”

[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:225)
把 TTGIR 的第三问定义为：

```text
When should the work happen, and what ordering is required?
```

这里的核心先是 **when**，然后才是 **ordering**。

所以：

- `ttng.wait_barrier`
- `ttng.fence_async_shared`
- `ttng.tc_gen5_commit`
- `ttng.warp_group_dot_wait`

都只是调度决策已经成立之后的协议部件，不等于调度决策本身。

### 2.2 不是“只要看到 async op 就叫 scheduling”

`ttng.warp_group_dot {isAsync = true}`、`ttng.tc_gen5_mma {is_async}`、
`ttg.async_copy_global_to_local` 这些 op 很显眼，但它们只是“调度已经被显式化”的结果。

真正更早的问题是：

- 哪些 load 值得提前
- 哪些 compute 值得异步化
- 需要几个 stage
- 是否需要 producer / consumer 分区
- descriptor / buffer 是否要 multi-buffer

### 2.3 不是“所有架构都走同一套 pass 逻辑”

当前 NVIDIA backend 在 `make_ttgir` 阶段就按 capability 分叉：

- `sm80/sm90` 的**调度主干**是 `AssignLatencies -> ScheduleLoops -> Pipeline`
- `sm100+` 的**调度主干**是 `AssignLatencies -> ScheduleLoops -> WarpSpecialize -> Pipeline -> OptimizePartitionWarps`

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:282),
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:292)。

这说明 target-driven scheduling 从 pass pipeline 组织层面就已经是 target-sensitive 的。

---

## 3. 先抓目标，再抓约束

target-driven scheduling 想解决的统一目标很简单：

```text
在不破坏语义的前提下，
让 memory / tensor-core / async engine / TMEM 的长延迟
尽量被别的有用工作覆盖。
```

但不同 target 的约束不同，所以“同一个目标”会长成不同协议。

### 3.1 Ampere `sm80/sm86`

主约束：

- tensor core 还是 `mma.sync` world
- 异步 copy 主要是 `cp.async` group 模型
- 没有 Hopper TMA / cluster 路径
- 没有 Blackwell TMEM / tcgen05

因此更自然的调度结构是：

```text
普通 load / local staging / dot
  + 可能的 cp.async-style overlap
  + 最终仍以 CTA 内同步为主
```

canonical matmul 的 after pipeline 仍然是普通
`tt.load -> ttg.convert_layout -> tt.dot` 主循环，见
[063_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/063_After_TritonGPUPipeline.mlir:66)。

### 3.2 Hopper `sm90`

主约束：

- compute 核心已经是 warp-group 级 `wgmma`
- shared memory 既被 generic proxy 访问，也被 async proxy 消费
- TMA / mbarrier / named barrier / cluster 特性可用

因此更自然的调度结构是：

```text
shared staging
  + warp-group async MMA
  + wait-group / proxy fence
  + 必要时用 warp specialization 做 sub-CTA producer/consumer 分工
```

canonical matmul 的 after pipeline 已经变成
`ttg.local_alloc -> ttng.warp_group_dot {isAsync = true} -> ttng.warp_group_dot_wait`，
见
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:69)。

### 3.3 Blackwell `sm100`

主约束：

- compute 核心变成 `tcgen05` / TMEM world
- completion 不再像 WGMMA 那样只靠 wait-group，而要借助 mbarrier
- TMEM reuse 有自己独立的 hazard
- automatic warp specialization 成为重要组织手段

因此更自然的调度结构是：

```text
TMEM accumulator
  + tc_gen5_mma async issue
  + per-stage barrier object
  + phase rotation / double buffering
  + 必要时 producer/consumer partition
```

在 `sm100_num_ctas2` 的 after pipeline 里，已经能直接看到这条链：

- barrier slot init:
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)
- async `tc_gen5_mma`:
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:100)
- steady-state `wait_barrier`:
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:106)
- phase / slot rotation:
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:115)

---

## 4. Triton 里“谁做什么决定”

把源码职责压成一张表：

| 层 | 主要问题 | 典型载体 | 主要实现 |
|---|---|---|---|
| schedule decision | 哪些 op 值得 overlap；哪个 stage / partition 执行什么 | `tt.latency`、`loop.stage`、`loop.cluster`、partition attrs | `AssignLatencies`、`ScheduleLoops`、`PartitionScheduling` |
| protocol abstraction / materialization | 如何把调度计划变成真实 async pipeline | async copy、`ttng.tc_gen5_mma {is_async}`、ARef、`ttg.warp_specialize`、prologue/kernel/epilogue | `Pipeline`、`AutomaticWarpSpecialization`、`NVWSInsertAref`、`NVWSLowerAref`、`PartitionLoops` |
| hazard repair / sync completion | 哪些地方还需要补合法性同步 | `ttg.barrier`、`ttng.fence_async_shared`、`ttng.wait_barrier`、TMEM barrier | `MembarAnalysis`、`FenceInsertion`、`ProxyFenceInsertion`、`TMemBarrierInsertion` |

### 4.1 `AssignLatencies`: 先挑“值得重叠”的点

这个 pass 的定义就写得很直接：
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:29)。

它不改 loop 结构，不生成 async op，只给“interesting ops”写 latency 属性。

因此它建立的 contract 不是“已经 pipeline”，而是：

```text
这些 op 是后续调度可以锚定的 latency source
```

### 4.2 `ScheduleLoops`: 把 latency anchor 变成 coarse schedule

这个 pass 的定义是 “software pipeline loop scheduling”，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:43)。

它的 durable after-IR contract 不是 async op，而是：

- `loop.stage`
- `loop.cluster`
- `tt.scheduled_max_stage`

在 `sm100_num_ctas1` 的 canonical matmul 里就能直接看到这一层 contract：
[059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)。

注意这里 `tt.scheduled_max_stage = 0`，说明它不是一个强 overlap 样本；只是 schedule
contract 已经出现了。

### 4.3 `Pipeline`: 把 coarse schedule 展成可执行 loop

`Pipeline` pass 的描述是：

```text
Applies software pipelining to loops in the module based on number of stages.
This may convert some load into asynchronous loads, and multi-buffer the data.
```

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:6)。

这里开始，调度不再只是 metadata，而是变成真实程序结构：

- async load / async MMA / wait
- loop-carried token
- multi-buffer state
- prologue / steady-state / epilogue

Blackwell 两个 canonical 样本正好展示了“弱 contract”和“强 materialization”的区别：

- `sm100_num_ctas1` after schedule 只有 stage 0 contract：
  [059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)
- `sm100_num_ctas2` after pipeline 已经出现 per-stage barrier、wait、phase toggling：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)

### 4.4 `AutomaticWarpSpecialization`: 不是单 pass，而是一条子流水线

这是理解 Blackwell target-driven scheduling 时最容易漏掉的点。

从定义上看，它的目标是：

```text
analyze the loops in the kernel and attempt to create a partition schedule,
which if successful lowers the loop by duplicating it into ttg.warp_specialize partition regions
```

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:99)。

但实现上它不是单个 rewrite，而是内部 pass manager：

- `PartitionScheduling`
- `NVWSHoistTmemStore`
- `NVWSInsertAref`
- `NVWSInsertTmemAref`
- `SCCP`
- `CSE`
- `NVWSLowerAref`
- `PartitionLoops`
- `NVWSLowerWarpGroup`
- `ScheduleLoops`

见
[AutomaticWarpSpecialization.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp:95)。

这件事的重要性在于：

```text
target-driven scheduling 并不是
"先跑完普通 schedule，再额外套一个 warp specialization"。

对 Blackwell 来说，
partition schedule 本身就是 scheduling contract 的一部分。
```

### 4.5 `NVWSLowerAref`: 把“跨 partition 协议”降成 barrier 集合

`NVWSLowerAref` 的 pass 描述已经把设计意图写出来了：

```text
Convert nvws.aref.* to ttng.*barrier* ops.
```

见
[NVWS Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)。

更关键的是它内部还会先跑 `NVWSAssignStagePhase`：
[LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:942)。

所以更准确的责任链是：

```text
ARef = 调度层的 producer/consumer 抽象
AssignStagePhase = 给这个抽象分配 stage / phase
LowerAref = 把抽象展开成 barrier / wait / buffer views
```

这解释了为什么
[2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md:588)
会把 ARef 放在“同步抽象”而不是“硬件原语”那一层。

---

## 5. 一个更准确的 Blackwell 心智模型

对 `sm100` 路径，最稳的读法不是“tcgen05 比 wgmma 多了几个 barrier”，而是下面这条链：

```text
TMEM accumulator
  -> async tc_gen5_mma issue
  -> completion observable through mbarrier-like object
  -> consumer waits by stage/phase
  -> buffers and barrier slots are rotated across iterations
```

在 `sm100_num_ctas2` 的 after pipeline 里，这条链是可直接看到的：

1. barrier storage 先被分配并初始化：
   [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)
2. 第一次 `tc_gen5_mma` 已经带 `{is_async}`，并绑定 barrier slot：
   [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:100)
3. steady-state loop 先 `wait_barrier`，再发下一次 async MMA：
   [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:105)
4. `xori` + `select` 做 phase / slot 轮转：
   [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:115)
5. loop 结束后 barrier 被 `inval`，TMEM 结果才被 load 回寄存器：
   [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:129)

这就是 target-driven scheduling 的本质例子：

```text
不是“因为有 barrier 所以叫 scheduling”，
而是“因为 target 的 compute/completion model 变了，
所以调度必须显式持有 barrier slot、phase、token 和多 buffer 状态”。
```

---

## 6. 为什么后面还要再补同步

即使 schedule 已经成形，后面还是必须有 target-specific sync repair。

这不是重复工作，而是另一层问题。

### 6.1 `MembarAnalysis`: CTA 级 shared-memory hazard

它只负责 shared memory RAW/WAR/WAW，并且只会插 `ttg.barrier local`：
[Membar.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Analysis/Membar.cpp:241)。

更关键的是，它把下面这些 op 当作“已经建立了 local sync point”：

- `gpu::BarrierOp`
- `ClusterBarrierOp`
- `ClusterWaitOp`
- `WarpSpecializePartitionsOp`
- `ArriveBarrierOp`
- `BarrierExpectOp`
- `TCGen5CommitOp`

见
[Membar.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Analysis/Membar.cpp:247)。

这说明：

```text
MembarAnalysis 不是在决定调度；
它是在识别“前面的调度 / lowering 已经建立了哪些同步点”，
然后避免重复补 barrier。
```

### 6.2 `ProxyFenceInsertion`: generic vs async proxy 不是同一回事

这个 pass 的文件头直接写明：

```text
On Hopper+, async proxy is separate from generic proxy
```

见
[ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:9)。

所以它解决的是：

```text
前面的 schedule 决定了谁先生产、谁后消费，
但如果两边跨 proxy，仍要再补 fence 才真的合法。
```

这和“调度决定 producer/consumer 关系”是两层不同问题。

### 6.3 `TMemBarrierInsertion`: TMEM reuse 也不是 Pipeline 自己兜底

它专门处理 TMEM reuse 的 ordering，核心规则写得很清楚：

- `load->mma` 和 `store->mma` 需要 barrier
- `mma->load/store` 不需要额外 barrier，因为后面的 `mbarrier wait` 会保证完成

见
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:66)。

这再次说明：

```text
Pipeline 决定“做异步 tc_gen5 pipeline”
!=
所有 TMEM legality 都在 Pipeline 内部一次解决完
```

---

## 7. 读 TTGIR 时应该怎么问

看一个 scheduling 相关 pass，最稳的是按下面三问：

### 7.1 这一步是在决定“谁先谁后”，还是在把决定物化？

- `AssignLatencies` / `ScheduleLoops` 更偏决定
- `Pipeline` / `NVWSLowerAref` / `PartitionLoops` 更偏物化
- `Membar` / `ProxyFenceInsertion` / `TMemBarrierInsertion` 更偏合法性修补

### 7.2 这个 target 的执行单元是什么？

- Ampere：warp / CTA，`mma.sync`
- Hopper：warp-group，`wgmma`
- Blackwell：TMEM + `tcgen05`, completion 经 barrier object 观察

这个问题先搞清，再看同步协议才不会混。

### 7.3 现在看到的 barrier / wait，是 schedule 的输入、输出，还是后补？

这是最重要的层次区分：

- `loop.stage` 是 schedule 输出 contract
- `ttng.wait_barrier` 往往是 pipeline / NVWS lowering 的输出
- `ttng.fence_async_shared` 往往是后补的 proxy legality

---

## 8. 和现有文档怎么配合读

建议阅读顺序：

1. 先看
   [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:211)
   的三问框架。
2. 再看本文，建立 “decision -> materialization -> hazard repair” 的职责链。
3. 然后按需要深挖单篇 note：
   - `AssignLatencies`：
     [2026-06-29-assign-latencies.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-29-assign-latencies.md:1)
   - `ScheduleLoops`：
     [2026-06-29-schedule-loops.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-29-schedule-loops.md:1)
   - `Pipeline`：
     [2026-06-29-pipeline.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-29-pipeline.md:1)
   - `WarpSpecialize`：
     [2026-06-30-warp-specialize.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-30-warp-specialize.md:1)
   - barrier / fence 手册：
     [2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md:1)

---

## 9. 最后压成一个统一模型

```text
target-driven scheduling
  不是“看到 barrier”
  也不是“看到 async op”

它是：
  target capability
    -> 决定哪类 overlap 值得做
    -> 决定执行单元如何分工
    -> 决定 completion 通过什么对象观察
    -> 决定 loop 需要持有哪些 stage / phase / token / buffer state
    -> 最终再由后续 pass 补齐 proxy / TMEM / shared-memory legality
```

所以：

- Ampere 看起来更像“普通 load/dot loop + 少量 pipeline 痕迹”
- Hopper 更像“warp-group async MMA + wait-group / proxy fence”
- Blackwell 更像“TMEM async pipeline + barrier slot + phase rotation + optional warp specialization”

这三者不是三套互不相关的技巧，而是同一个目标在三种 target 约束下的不同实现。
