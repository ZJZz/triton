# Triton TTGIR：Target-Driven Scheduling

本文只回答一个问题：同一个 kernel 的主循环，为什么会在不同 target 上形成不同的 load / compute / overlap / synchronize 结构。这里关注的是 `when should the work happen, and what ordering is required`，不是同步原语清单本身。

## 1. 核心定义

```text
target-driven scheduling
  = 编译器根据目标架构的执行单元、异步引擎、可见同步原语和内存路径，
    决定主循环里哪类工作应该提前、分区、重叠，
    以及最终通过哪套协议交接。
```

在 Triton 里，这件事不是单个 pass 完成的，而是三层职责链：

```text
schedule decision
  -> 协议显化
  -> hazard repair
```

- `schedule decision`：决定哪些 op 值得 overlap，哪些 op 属于哪个 stage / partition。
- `协议显化`：把这些决策变成 async copy、async MMA、multi-buffer、warp-specialize region、barrier protocol。
- `hazard repair`：补齐 generic-vs-async proxy、TMEM reuse、shared memory RAW/WAR/WAW 等合法性约束。

如果把 TTGIR 三层压成一句短记忆：

```text
mapping = 分工
organization = 衔接
scheduling = 时序
```

那么本文只负责第三层：`scheduling = 时序`。

## 2. 和另外两层的边界

| 主题 | 核心问题 | 典型载体 |
|---|---|---|
| distributed execution mapping | 谁拥有这些元素 | `#blocked`、`CGAEncodingAttr`、`#ttg.slice` |
| layout / data-movement organization | 这些值以什么 form / carrier 流动 | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMEM |
| target-driven scheduling | 这些工作何时发生、如何 overlap、何时同步 | `loop.stage`、`loop.cluster`、async op、wait、barrier |

有三个容易混淆的点：

- 它不是 barrier / fence 的别名。barrier、wait、fence 往往是 scheduling 之后的协议部件。
- 它不是“看到 async op 就算理解 scheduling”。async op 只是调度已经显式化之后的结果。
- 它从 pass pipeline 组织层面就是 target-sensitive，不是所有架构都走同一套调度主干。

`TTGIR_GUIDE.md` 把第三问定义为 `这些工作何时发生、如何 overlap、何时需要同步`，见
[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)。
同步原语本身的种类、scope、visibility、proxy 语义，请看
[2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md:1)。

## 3. target 约束决定调度形态

统一目标很简单：

```text
在不破坏语义的前提下，
让 memory / tensor-core / async engine / TMEM 的长延迟
尽量被别的有用工作覆盖
```

但不同 target 的执行单元和 completion model 不同，所以主循环会长成不同协议。

### 3.1 Ampere `sm80/sm86`

主约束：

- tensor core 还是 `mma.sync`
- 异步 copy 主要是 `cp.async` group 模型
- 没有 Hopper TMA / cluster 路径
- 没有 Blackwell TMEM / `tcgen05`

因此更自然的调度结构是：

```text
普通 load / local staging / dot
  + 可能的 cp.async-style overlap
  + 最终仍以 CTA 内同步为主
```

canonical matmul 的 after pipeline 仍是普通
`tt.load -> ttg.convert_layout -> tt.dot` 主循环，见
[063_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/063_After_TritonGPUPipeline.mlir:66)。

### 3.2 Hopper `sm90`

主约束：

- compute 核心已经是 warp-group 级 `wgmma`
- shared memory 同时被 generic proxy 和 async proxy 访问
- TMA / mbarrier / named barrier / cluster 特性可用

因此更自然的调度结构是：

```text
shared staging
  + warp-group async MMA
  + wait-group / proxy fence
  + 必要时用 warp specialization 做 sub-CTA producer / consumer 分工
```

canonical matmul 的 after pipeline 已经变成
`ttg.local_alloc -> ttng.warp_group_dot {isAsync = true} -> ttng.warp_group_dot_wait`，
见
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:69)。

### 3.3 Blackwell `sm100`

主约束：

- compute 核心变成 `tcgen05` / TMEM world
- completion 不再像 WGMMA 那样只靠 wait-group，而要借助 barrier object
- TMEM reuse 有独立 hazard
- automatic warp specialization 成为重要组织手段

因此更自然的调度结构是：

```text
TMEM accumulator
  + async tc_gen5_mma issue
  + per-stage barrier object
  + phase rotation / double buffering
  + 必要时 producer / consumer partition
```

在 `sm100_num_ctas2` 的 after pipeline 里已经能直接看到这条链：

- barrier slot init：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)
- async `tc_gen5_mma`：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:100)
- steady-state `wait_barrier`：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:106)
- phase / slot rotation：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:115)

## 4. 调度职责链由哪些 pass 实现

当前 NVIDIA backend 在 `make_ttgir` 阶段就按 capability 分叉：

- `sm80/sm90`：`AssignLatencies -> ScheduleLoops -> Pipeline`
- `sm100+`：`AssignLatencies -> ScheduleLoops -> WarpSpecialize -> Pipeline -> OptimizePartitionWarps`

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:282)，
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:292)。

### 4.1 `AssignLatencies`：先挑“值得重叠”的点

这个 pass 的定义直接说明，它先给 interesting ops 写 latency 属性，供后续调度锚定，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:29)。

它建立的 contract 不是“已经 pipeline”，而是：

```text
这些 op 是后续调度可以锚定的 latency source
```

### 4.2 `ScheduleLoops`：把 latency anchor 变成 coarse schedule

`ScheduleLoops` 的定义是 `software pipeline loop scheduling`，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:43)。

它的 durable after-IR contract 主要是：

- `loop.stage`
- `loop.cluster`
- `tt.scheduled_max_stage`

在 `sm100_num_ctas1` 的 canonical matmul 里可以直接看到这层 contract，见
[059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)。

注意这里 `tt.scheduled_max_stage = 0`，说明 schedule contract 已经出现，但还不是强 overlap 样本。

### 4.3 `Pipeline`：把 coarse schedule 展成可执行 loop

`Pipeline` 的描述是：

```text
Applies software pipelining to loops in the module based on number of stages.
This may convert some load into asynchronous loads, and multi-buffer the data.
```

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:6)。

从这里开始，调度不再只是 metadata，而是变成真实程序结构：

- async load / async MMA / wait
- loop-carried token
- multi-buffer state
- prologue / steady-state / epilogue

`sm100_num_ctas1` 和 `sm100_num_ctas2` 正好展示了“弱 contract”和“强显化”的区别：

- `sm100_num_ctas1` after schedule 只有 stage 0 contract：
  [059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)
- `sm100_num_ctas2` after pipeline 已经出现 per-stage barrier、wait、phase toggling：
  [085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)

### 4.4 `AutomaticWarpSpecialization`：Blackwell 上的调度子流水线

`AutomaticWarpSpecialization` 的描述是：

```text
analyze the loops in the kernel and attempt to create a partition schedule,
which if successful lowers the loop by duplicating it into ttg.warp_specialize partition regions
```

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:99)。

实现上它不是单个 rewrite，而是内部 pass manager，顺序包括：

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

这说明对 Blackwell 来说，partition schedule 本身就是 scheduling contract 的一部分，而不是“普通 schedule 之后再附加一个 warp specialization”。

### 4.5 `NVWSLowerAref`：把 partition 协议降成 barrier 集合

`NVWSLowerAref` 的 pass 描述直接写着：

```text
Convert nvws.aref.* to ttng.*barrier* ops.
```

见
[NVWS Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)。

更关键的是它内部会先跑 `NVWSAssignStagePhase`，见
[LowerAref.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp:942)。

更准确的责任链是：

```text
ARef = 调度层的 producer / consumer 抽象
AssignStagePhase = 给这个抽象分配 stage / phase
LowerAref = 把抽象展开成 barrier / wait / buffer views
```

## 5. `wait / barrier / fence` 在三层职责链里的位置

这三个词本身不能只按名字归类，要看它们是在“实现协议”，还是在“补 legality”。

| IR 载体 | 更常见归类 | 什么时候会落在这一层 |
|---|---|---|
| `wait` | `协议显化` | 调度已经决定要 overlap，`wait` 作为 async pipeline / async MMA / barrier protocol 的组成部分被显式生成 |
| `barrier` | 多数是 `协议显化`，有时是 `hazard repair` | 如果它是 stage / partition / barrier object 协议的一部分，更偏显化；如果它是后面为 shared-memory legality 补插的 `ttg.barrier local`，更偏 repair |
| `fence` | 多数是 `hazard repair` | 前面的 producer / consumer 关系已经成立，但 proxy / visibility / memory-order legality 还不完整，需要再补 fence |

可以用这三问快速判断：

- 它是在决定时序吗：那是 `schedule decision`，通常还不是 `wait / barrier / fence` 本身。
- 它是在把已有调度决定展开成 async / token / barrier protocol 吗：更偏 `协议显化`。
- 它是在补 shared memory、proxy、TMEM reuse 之类的合法性缺口吗：更偏 `hazard repair`。

## 6. 为什么调度之后还要再补同步

即使 schedule 已经成形，后面仍然需要 target-specific sync repair。这不是重复工作，而是另一层合法性问题。

### 6.1 `MembarAnalysis`：CTA 级 shared-memory hazard

它只负责 shared memory RAW/WAR/WAW，并且只会插 `ttg.barrier local`，见
[Membar.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Analysis/Membar.cpp:241)。

更关键的是，它把这些 op 当作已经建立了 local sync point：

- `gpu::BarrierOp`
- `ClusterBarrierOp`
- `ClusterWaitOp`
- `WarpSpecializePartitionsOp`
- `ArriveBarrierOp`
- `BarrierExpectOp`
- `TCGen5CommitOp`

见
[Membar.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Analysis/Membar.cpp:247)。

这说明 `MembarAnalysis` 不是在决定调度，而是在识别前面的调度 / lowering 已经建立了哪些同步点。

### 6.2 `ProxyFenceInsertion`：generic proxy 和 async proxy 不是一回事

这个 pass 的文件头直接写明：

```text
On Hopper+, async proxy is separate from generic proxy
```

见
[ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:9)。

所以它解决的是：前面的 schedule 决定了谁先生产、谁后消费，但如果两边跨 proxy，仍要再补 fence 才真的合法。

### 6.3 `TMemBarrierInsertion`：TMEM reuse 也需要单独 repair

它专门处理 TMEM reuse 的 ordering，核心规则是：

- `load->mma` 和 `store->mma` 需要 barrier
- `mma->load/store` 不需要额外 barrier，因为后面的 `mbarrier wait` 会保证完成

见
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:66)。

这再次说明：`Pipeline` 决定“做异步 tc_gen5 pipeline”，不等于所有 TMEM legality 都在 `Pipeline` 内部一次解决完。

## 7. 最小 IR 证据

### 7.1 `sm86`：after pipeline 仍是普通 `load -> convert -> dot`

Ampere canonical matmul 的主循环仍然是
`tt.load -> ttg.convert_layout -> tt.dot`，见
[063_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/063_After_TritonGPUPipeline.mlir:66)。

### 7.2 `sm90`：after pipeline 已经是 `local_alloc -> warp_group_dot async -> wait`

Hopper canonical matmul 里可以直接看到：

- `ttg.local_alloc`
- `ttng.warp_group_dot {isAsync = true}`
- `ttng.warp_group_dot_wait`

见
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:69)。

### 7.3 `sm100_num_ctas1`：`ScheduleLoops` 已经写出 schedule contract

在
[059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:74)
里，已经能看到 `loop.stage`、`loop.cluster` 和 `tt.scheduled_max_stage = 0`。

这说明 schedule contract 先出现，协议显化可以更强也可以更弱。

### 7.4 `sm100_num_ctas2`：after pipeline 出现完整 async barrier protocol

在
[085_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas2/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir:90)
之后，可以顺序看到：

- `init_barrier`
- `tc_gen5_mma {is_async}`
- `wait_barrier`
- `xori` + `select` 的 phase rotation
- `inval_barrier`

这组 IR 直接展示了 Blackwell 的 target-driven scheduling contract。

## 8. 读这类 TTGIR 时的检查顺序

1. 先分清这是在决定 `when / overlap`，还是在物化协议，还是在补合法性同步。
2. 先问这个 target 的执行单元和 completion model 是什么，再看具体 barrier / wait。
3. 看到 `loop.stage`、`loop.cluster` 时，把它们当 schedule contract，不要直接当 runtime protocol。
4. 看到 `wait_barrier`、`warp_group_dot_wait`、`fence_async_shared` 时，先判断它们属于协议显化还是 repair。
5. 只有把 `decision -> 协议显化 -> repair` 三层拆开，barrier/fence 才不会和 scheduling 本身混在一起。
