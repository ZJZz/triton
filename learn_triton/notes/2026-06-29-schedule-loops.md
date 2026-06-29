# 2026-06-29 学习笔记：ScheduleLoops

## 1. Pass 基本信息

本轮学习 `TritonGPUScheduleLoops`。它位于软件流水相关 pass 的中间：

```text
... -> AssignLatencies
    -> ScheduleLoops
    -> Pipeline
    -> ...
```

一句话：

```text
ScheduleLoops 不复制 loop，也不生成 prologue / epilogue / async wait。
它把 AssignLatencies 留下的 tt.latency 读成 stage-distance 信息，
再给 loop body op 写入 loop.stage / loop.cluster，
把“这个 op 属于第几个 pipeline stage、同 stage 内相对顺序如何”这个 contract
交给后续 Pipeline / LowerLoops 消费。
```

相关文件：

- Source: [ScheduleLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:35)
- Schedule helper: [Schedule.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/Schedule.cpp:254)
- Attr names: [PipeliningUtility.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h:20)
- Latency deserialization: [PipeliningUtility.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:337)
- Pass td: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:43)
- Focused lit coverage: [pipeline-schedule-loop.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-schedule-loop.mlir:1)

Canonical matmul dumps used here:

- Ampere before: [060_Before_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUScheduleLoops.mlir:1)
- Ampere after: [061_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUScheduleLoops.mlir:1)
- Hopper before: [058_Before_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:1)
- Hopper after: [059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:1)
- Blackwell before: [058_Before_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:1)
- Blackwell after: [059_After_TritonGPUScheduleLoops.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:1)

## 2. Architecture Matrix

| Arch | Before feature | After feature | Changed? |
| --- | --- | --- | --- |
| Ampere `sm86` | loop body has `tt.load`, `ttg.convert_layout`, `tt.dot`, pointer increments; no visible `tt.latency` | IR text is effectively unchanged | no visible change |
| Hopper `sm90` | loop body has `tt.load`, `ttg.local_alloc`, `ttng.warp_group_dot`; no visible `tt.latency` | IR text is effectively unchanged | no visible change |
| Blackwell `sm100` | `ttng.tc_gen5_mma` has `{tt.latency = 1, tt.self_latency = 1}` | `tt.latency` is removed; loop body ops get `{loop.stage = 0, loop.cluster = 0}`; loop gets `{tt.scheduled_max_stage = 0}` | yes |

Important caveat:

```text
canonical sm86/sm90 matmul 这里没有 visible schedule，不代表 ScheduleLoops 只能处理 Blackwell。
原因是本 dump 里 AssignLatencies 没有给 sm86/sm90 的 loads 留下 tt.latency anchor。
test/TritonGPU/pipeline-schedule-loop.mlir 里有典型多 stage 例子：
tt.load {tt.latency = 2} -> stage 0，
tt.addptr -> stage 1，
ttg.convert_layout / tt.dot -> stage 2。
```

## 3. Cross-Architecture Before Comparison

进入 `ScheduleLoops` 前，三代 before 已经分叉：

- Ampere `sm86`: loop body 是 register-side `tt.load -> ttg.convert_layout -> tt.dot`，见 [060_Before_TritonGPUScheduleLoops.mlir:66](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUScheduleLoops.mlir:66)
- Hopper `sm90`: loop body 是 `tt.load -> ttg.local_alloc -> ttng.warp_group_dot`，见 [058_Before_TritonGPUScheduleLoops.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:69)
- Blackwell `sm100`: loop 外有 TMEM alloc/store，loop body 是 `ttng.tc_gen5_mma`，见 [058_Before_TritonGPUScheduleLoops.mlir:72](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:72)

Conclusion:

```text
ScheduleLoops 本身不是最早造成三代 IR 分叉的 pass。
它消费的是当前 loop 内已有 latency anchor 或已有 warp-specialized schedule。
在本 canonical matmul 中，只有 Blackwell path 还有可见 tt.latency，
所以只有 Blackwell after 出现可见 scheduling attr。
```

## 4. IR Changes

### 4.1 Ampere `sm86`: no visible change in canonical matmul

Before loop core:

- `tt.load` A/B 见 [060_Before_TritonGPUScheduleLoops.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUScheduleLoops.mlir:67)
- `ttg.convert_layout` A/B 见 [060_Before_TritonGPUScheduleLoops.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUScheduleLoops.mlir:69)
- `tt.dot` 见 [060_Before_TritonGPUScheduleLoops.mlir:71](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/060_Before_TritonGPUScheduleLoops.mlir:71)

After loop core:

- same `tt.load` / `ttg.convert_layout` / `tt.dot` sequence，见 [061_After_TritonGPUScheduleLoops.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUScheduleLoops.mlir:67)

IR-level meaning:

```text
没有 op 获得 loop.stage / loop.cluster。
ScheduleLoops 没有找到 latency anchor，也没有已有 schedule 可恢复，因此返回空 schedule。
```

### 4.2 Hopper `sm90`: no visible change in canonical matmul

Before loop core:

- A/B `tt.load` + `ttg.local_alloc` 见 [058_Before_TritonGPUScheduleLoops.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:70)
- `ttng.warp_group_dot` 见 [058_Before_TritonGPUScheduleLoops.mlir:74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:74)

After loop core:

- same `tt.load -> ttg.local_alloc -> ttng.warp_group_dot` sequence，见 [059_After_TritonGPUScheduleLoops.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:70)

IR-level meaning:

```text
当前 sm90 canonical matmul 没有可见 schedule attr。
这和 AssignLatencies 的结论一致：A/B loads 被 small-load filter drop，
没有留下 tt.latency 给 ScheduleLoops 消费。
```

### 4.3 Blackwell `sm100`: consumes `tt.latency`, serializes schedule

Before:

- `ttng.tc_gen5_mma` 带 `{tt.latency = 1 : i32, tt.self_latency = 1 : i32}`，见 [058_Before_TritonGPUScheduleLoops.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:79)

```text
%acc_56 = ttng.tc_gen5_mma ...
  {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
```

After:

- A/B `tt.load` 得到 `{loop.cluster = 0, loop.stage = 0}`，见 [059_After_TritonGPUScheduleLoops.mlir:75](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:75)
- A/B `ttg.local_alloc` 得到 `{loop.cluster = 0, loop.stage = 0}`，见 [059_After_TritonGPUScheduleLoops.mlir:76](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:76)
- `ttng.tc_gen5_mma` 得到 `{loop.cluster = 0, loop.stage = 0, tt.self_latency = 1}`，但 `tt.latency` 被删除，见 [059_After_TritonGPUScheduleLoops.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:79)
- pointer increments 也得到 `{loop.cluster = 0, loop.stage = 0}`，见 [059_After_TritonGPUScheduleLoops.mlir:80](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:80)
- `scf.for` 得到 `{tt.scheduled_max_stage = 0}`，见 [059_After_TritonGPUScheduleLoops.mlir:83](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:83)

IR-level meaning:

```text
tt.latency 是 ScheduleLoops 的输入 contract，不是长期 IR contract。
它被 deserializeLatencies 读走并删除。

loop.stage / loop.cluster / tt.scheduled_max_stage 是 ScheduleLoops 的输出 contract。
后续 Pipeline 会根据这些 attr 判断哪些 op 要被提前、复制、分配到 prologue/main/epilogue。
```

为什么这里 `scheduled_max_stage = 0`？

```text
canonical sm100 matmul 里只有一个 latency anchor：tc_gen5_mma {tt.latency = 1}。
computeDistance(tc_gen5_mma) 时，它没有普通 in-loop user：
  accumulator/token 通过 scf.yield 形成 loop-carried distance-1 dependency，
  但 computeDistance 显式跳过 terminator。

所以:
  maxDist = -1
  lat = 1
  d = lat + 0 = 1
  maxDistance = 1
  stage = maxDistance - d = 0

因此 scheduleKeyOps 计算出的可见 coarse schedule 只有一个 stage。
所以 pass 仍然生效，但不是一个多 stage overlap 的展示样本。
```

## 5. Source Mapping

### 5.1 Top-level dispatch: getInitialSchedule

Code:

- `scheduleLoop` 先调用 `getInitialSchedule(forOp, opLatency)`，见 [ScheduleLoops.cpp:351](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:351)
- `getInitialSchedule` 是三路分派入口：safety gate、latency branch、warp-specialize existing-schedule branch、否则空 schedule，见 [ScheduleLoops.cpp:251](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:251)
- `isSafeToPipeline` 拒绝 distance > 1 的 loop、outer loop、带 barrier/assert/print 的 loop，见 [ScheduleLoops.cpp:35](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:35)

Logic:

```text
ScheduleLoops 只给后续 Pipeline 能正确展开的软件流水 loop 写 schedule。
如果 loop-carried dependency 复杂到 distance > 1，或者是 outer loop，或者已有显式 barrier/assert/print，
这里不冒险生成 pipeline schedule。

如果 loop 内有 tt.latency anchor，就进入 scheduleKeyOps。
如果 loop 带 warp-specialize attr 且已有 serialized schedule，就恢复/规整已有 schedule。
否则返回空 schedule，当前 loop 不写 loop.stage / loop.cluster。
```

IR evidence:

- canonical matmul 的 inner `scf.for` 满足 safety gate；sm100 after 能看到 schedule attr。

### 5.2 Read and remove latency attrs

Code:

- `scheduleLoops` 先调用 `deserializeLatencies(moduleOp)`，见 [ScheduleLoops.cpp:389](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:389)
- `deserializeLatencies` 收集 `tt.latency` 后立即 remove，见 [PipeliningUtility.cpp:337](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:337)

IR evidence:

- sm100 before 的 `ttng.tc_gen5_mma` 有 `tt.latency = 1`，见 [058_Before_TritonGPUScheduleLoops.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUScheduleLoops.mlir:79)
- sm100 after 同一个 op 没有 `tt.latency`，但保留 `tt.self_latency = 1`，见 [059_After_TritonGPUScheduleLoops.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:79)

### 5.3 Build initial schedule from latency ops

Code:

- `hasLatenciesAssigned` 判断 loop 内是否有 latency op，见 [ScheduleLoops.cpp:143](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:143)
- `scheduleKeyOps` 收集 latency ops，并沿 op users 计算 latency 加权最长路径，见 [ScheduleLoops.cpp:152](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:152)
- `computeDistance` 只考虑同一 loop body 内、且不是 terminator 的 user，见 [ScheduleLoops.cpp:175](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:175)
- stage 计算是 `stage = maxDistance - dist`，见 [ScheduleLoops.cpp:209](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:209)

Logic:

```text
latency attr 描述“这个 op 和后续 consumer 应该隔多远”。
ScheduleLoops 把 latency op 看成 anchor，计算 latency 加权最长路径：

  distance(op) = latency(op) + max(distance(user))

其中 user 必须能映射回当前 loop body 的 top-level op，且不能是 scf.yield terminator。
如果没有 in-loop user，maxDist = -1，被当成 0。

注意这个 distance 不是 def-use hop count。
tt.latency = 2 比 tt.latency = 1 拉开更多 stage，
是因为 latency 值作为路径权重直接累加。

然后 ScheduleLoops 用：

  stage(op) = maxDistance - distance(op)

再把较早要发起的 op 放到较小 stage，把较晚消费的 op 放到较大 stage。
```

Focused lit evidence:

- `tt.load {tt.latency = 2}` 被排到 `loop.stage = 0`，见 [pipeline-schedule-loop.mlir:187](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-schedule-loop.mlir:187)
- `ttg.convert_layout` 和 `tt.dot` 被排到 `loop.stage = 2`，见 [pipeline-schedule-loop.mlir:189](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-schedule-loop.mlir:189)
- loop-carried pointer increments 被排到 `loop.stage = 1`，见 [pipeline-schedule-loop.mlir:198](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-schedule-loop.mlir:198)

Cluster detail:

```text
初始 cluster 不是 per-stage 私有顺序，而是 CoarseSchedule::ClusterList 上的一条全局顺序链。
scheduleKeyOps 先为每个 stage 建一个 cluster，
再用 clusters[maxStage - stage] 插入 op。
也就是说，高 stage 初始会拿到更靠前的 cluster，
源码注释说这是为了得到 roughly reverse program order。
```

对应代码见 [ScheduleLoops.cpp:212](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:212)。

### 5.4 Warp-specialize existing-schedule branch

Code:

- 如果 loop 带 `kWarpSpecializeAttrName` 且 `CoarseSchedule::deSerialize(forOp)` 成功，`getInitialSchedule` 会恢复已有 schedule，见 [ScheduleLoops.cpp:261](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:261)
- `isLatencyOp` 白名单包括 `LoadOp`、descriptor load-like、local load/store、TMEM load/store、TMA interface、MMAv5、barrier wait/arrive，见 [ScheduleLoops.cpp:268](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:268)
- 如果 latency-like ops 只占 0 或 1 个 stage，就把所有 ops normalize 到单 stage，见 [ScheduleLoops.cpp:277](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:277)
- 否则 `shrinkToFit()` 后复用已有 schedule，见 [ScheduleLoops.cpp:295](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:295)

Logic:

```text
这是 ScheduleLoops 的第二条完整执行路径。
它不是从 tt.latency 重新推导 schedule，
而是恢复 warp-specialization 之前/之中留下的 serialized schedule。

如果恢复后发现真正 latency-like 的 ops 没有跨多个 stage，
pass 会把 loop 规整成单 stage，避免后续 Pipeline 做无意义展开。
```

IR evidence:

- 本 canonical matmul 三架构样本没有触发这条路径。
- 这个分支的 arch coupling 主要体现在 `isLatencyOp` 白名单包含 TMEM/TMA/MMAv5/barrier 等 Hopper/Blackwell 相关 op；pass 主算法仍没有按 `cuda:86/90/100` 写显式 target 分支。

### 5.5 Add prologue/epilogue `scf.if`

Code:

- `scheduleKeyOps` 会把当前 latency-forward slice 中可安全后移的 `scf.if` 放到 epilogue cluster，见 [ScheduleLoops.cpp:225](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:225)
- `schedulePrologueAndEpilogue` 用 scheduled ops 的 backward slice 找 prologue `scf.if`，并放到最前面的 cluster，见 [ScheduleLoops.cpp:305](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:305)
- 其他 `scf.if` 被放到最后一个 stage 的 epilogue cluster，见 [ScheduleLoops.cpp:338](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:338)

Logic:

```text
条件控制流本身不是 memory latency anchor，
但它可能守护地址计算、mask、prefetch extract 或尾部逻辑。
ScheduleLoops 尽量把依赖 scheduled ops 的 if 拉到前面，
把无关或尾部 if 推到后面，减少它们阻塞主流水 schedule。
```

IR evidence:

- 本 canonical matmul loop 没有 `scf.if`，所以这条源码路径未触发。

### 5.6 Add dependencies and loop-carried distance-1 deps

Code:

- `scheduleDependencies` 把 anchor op 的普通 operands recursively 放到同 stage / cluster，见 [Schedule.cpp:410](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/Schedule.cpp:410)
- `scheduleDistanceOneDependencies` 识别由 loop block argument 回到 yield operand 的 distance-1 dependency，见 [ScheduleLoops.cpp:50](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:50)
- 对 distance-1 的 `tt.load` 有例外：跟当前 op 放同 stage / cluster；其他 op 放到下一 stage 且在当前 cluster 前，见 [ScheduleLoops.cpp:72](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:72)

Logic:

```text
ScheduleLoops 不是只给 latency op 自己打标。
它必须把 producer、地址计算、layout conversion、loop-carried pointer update 一起纳入 schedule，
否则后续 Pipeline 复制/移动 op 时会破坏 SSA dependency。
```

### 5.7 Schedule remaining ops and serialize

Code:

- 未排到 schedule 的 op 默认放到最后一个 stage，且用 cluster 调整避免 use 被排到 def 前面，见 [ScheduleLoops.cpp:100](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:100)
- `scheduleLoop` 的最终步骤是 `schedule.serialize(forOp)`，见 [ScheduleLoops.cpp:384](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:384)
- `CoarseSchedule::serialize` 写 `loop.stage` / `loop.cluster`，并给 for op 写 `tt.scheduled_max_stage`，见 [Schedule.cpp:254](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/Schedule.cpp:254)

IR evidence:

- sm100 after 中 loop body op 都有 `loop.stage = 0, loop.cluster = 0`，见 [059_After_TritonGPUScheduleLoops.mlir:75](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:75)
- sm100 after 中 `scf.for` 有 `tt.scheduled_max_stage = 0`，见 [059_After_TritonGPUScheduleLoops.mlir:83](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:83)

## 5.8 Effective Or No-op

| Arch | Result | Evidence |
| --- | --- | --- |
| Ampere `sm86` | no visible change | before/after loop body remains `tt.load -> ttg.convert_layout -> tt.dot` with no `loop.stage` / `loop.cluster`，见 [061_After_TritonGPUScheduleLoops.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUScheduleLoops.mlir:67) |
| Hopper `sm90` | no visible change | before/after loop body remains `tt.load -> ttg.local_alloc -> ttng.warp_group_dot` with no schedule attrs，见 [059_After_TritonGPUScheduleLoops.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:70) |
| Blackwell `sm100` | changed | `tt.latency` is removed; loop body ops receive `loop.stage` / `loop.cluster`; `scf.for` receives `tt.scheduled_max_stage = 0`，见 [059_After_TritonGPUScheduleLoops.mlir:75](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUScheduleLoops.mlir:75) |

## 5.9 Deferred Work

`ScheduleLoops` deliberately does not:

- rewrite the loop into prologue / steady-state / epilogue;
- allocate multibuffered shared memory or TMEM buffers;
- insert async waits, barriers, or fences;
- lower `tt.load`, TMA, WGMMA, MMAv5, or TMEM ops.

这些工作留给后续 `TritonGPUPipeline` / `LowerLoops` / target-specific lowering。当前 pass 只建立 scheduling metadata contract。

## 6. Compiler Decision

Compiler question:

```text
在一个可 pipeline 的 scf.for 内，
哪些 op 应该属于 stage 0 / stage 1 / ...，
同一个 stage 内哪些 op 必须先后成组，
才能让后续 Pipeline 正确构造 prologue / steady-state / epilogue？
```

Decision made here:

- 如果 loop 没有 latency anchor，也没有可恢复的 warp-specialized schedule：不写 schedule。
- 如果有 `tt.latency` anchor：从 latency op 出发计算 stage-distance，并把 dependencies 补到对应 stage / cluster。
- 如果来自 warp specialization 且已有 schedule：恢复旧 schedule，必要时 shrink 或 normalize。

Why here in the pipeline:

```text
AssignLatencies 已经识别哪些 op 值得跨 stage overlap；
Pipeline 还没有展开 loop。
ScheduleLoops 正好处在两者之间，可以用轻量 attr 把“调度计划”写到 TTGIR，
让后续 pass 不必重新理解 latency heuristic。
```

## 7. Compiler Contract

Input contract:

- Loop 是可软件流水化的 inner `scf.for`。
- `AssignLatencies` 可能已经在 loop body op 上写了 `tt.latency`。
- 或者 warp-specialization flow 已经给 loop/op 写过 `loop.stage` / `loop.cluster` / `tt.scheduled_max_stage`。
- IR 仍保持 SSA def-use，关键 load/dot/local_alloc/TMEM op 还在 TTGIR/NvidiaGPU dialect 层。

Output contract:

- 被 schedule 的 op 有 `loop.stage = <i32>`。
- 被 schedule 的 op 有 `loop.cluster = <i32>`，表示全局 cluster ordering 链上的位置；同 stage 和跨 stage 的相对顺序都会受它影响。
- `scf.for` 有 `tt.scheduled_max_stage = <i32>`。
- `tt.latency` 被消费并删除；`tt.self_latency` 不属于这个 pass 的 stage-distance 输入，仍可保留给后续 pass。

Next pass relies on:

- `TritonGPUPipeline` / `LowerLoops` 用 `loop.stage` 决定哪些 op 进入 prologue、steady-state、epilogue。
- async load / local_alloc / MMA / TMEM lowering 用 schedule 顺序决定 buffer rotation、wait、barrier、token 的位置。
- 后续 canonicalize/CSE 可以清理展开后的冗余 IR。

## 8. Invariant

- Tensor shape: unchanged。
- Element type: unchanged。
- Memory address logical meaning: unchanged。
- Program semantics: unchanged。
- Changed only: scheduling metadata attrs。

这个 pass 不应该改变数学计算，也不应该移动真实 op 的 textual order；它只写 metadata。真正的 loop restructuring 发生在后续 `Pipeline`。

## 9. Triton Mechanism

`ScheduleLoops` 是 Triton TTGIR 层的软件流水调度计划生成器。

它的核心内部数据结构是 `CoarseSchedule`：

```text
Operation* -> (stage, cluster)
```

其中：

- `stage` 表示跨 iteration pipeline 的时间层级。
- `cluster` 表示 `CoarseSchedule::ClusterList` 全局 ordering 链上的粗粒度顺序组，不是 per-stage 私有序。它避免 producer/user、prologue/epilogue `scf.if`、loop-carried deps 在最终 linearized schedule 中乱序。

一个关键细节：

```text
scheduleKeyOps 初始建 cluster 时使用 clusters[maxStage - stage]。
所以 higher stage 初始会落到更靠前的 global cluster。
stage 决定跨 iteration 的时间层级，cluster 决定全局 linearized order。
两者共同决定后续 Pipeline 展开后的实际 op 顺序。
```

这仍然是 Triton IR mechanism，不是 CUDA asm lowering。它还没有生成 `cp.async`、`wgmma.wait_group`、TMEM wait 或真实 shared-memory multibuffer。

## 10. GPU / Hardware / Optimization Reason

Hardware reason:

- GPU 可以用多 stage 软件流水把 memory latency、shared memory staging、MMA/TMEM async work 和 compute overlap。
- 但硬件不会替 compiler 自动重排 Triton loop；compiler 必须显式表达每个 op 属于哪个 pipeline stage。

Instruction reason:

- Ampere path 最终要服务于 `mma.sync` 前的数据准备。
- Hopper path 最终要服务于 WGMMA / shared-memory operand readiness。
- Blackwell path 还要服务于 TMEM token、MMAv5 async issue、TMEM load/store wait 的 placement。

Execution reason:

- `loop.stage` 决定跨 iteration 的相对时间。
- `loop.cluster` 决定全局 linearized schedule 中不能随便交换的 ordering boundary。

Memory reason:

- 对 async global-to-shared、TMA、local_alloc、TMEM 这类资源，stage schedule 是后续 multibuffering / wait placement 的基础。

## 11. Decision Tree

```text
for each scf.for:
  if not isSafeToPipeline(forOp):
    no schedule

  opLatency = deserializeLatencies(module)

  if loop has op in opLatency:
    collect latency ops
    compute latency-weighted longest user path inside loop:
      distance(op) = latency(op) + max(distance(user))
      skip users outside loop body and scf.yield terminator
    assign stage = maxDistance - distance
    create initial CoarseSchedule

  else if loop has warp-specialize attr and existing serialized schedule:
    deserialize existing schedule
    if latency-like ops occupy <= 1 stage:
      normalize all ops into one stage
    else:
      shrink stages and reuse schedule

  else:
    no schedule

  schedule prologue / epilogue scf.if
  schedule normal dependencies into same stage / cluster
  schedule loop-carried distance-1 dependencies
  schedule remaining ops to last stage
  serialize CoarseSchedule:
    op -> loop.stage / loop.cluster
    forOp -> tt.scheduled_max_stage
```

## 12. Alternative Design

Alternative:

```text
让 Pipeline pass 自己读取 tt.latency、分析 dependencies、决定 stage，并直接展开 loop。
```

Why not here:

- 分离后，`AssignLatencies` 负责识别 latency anchor，`ScheduleLoops` 负责生成 coarse schedule，`Pipeline` 负责结构化展开 loop。
- 这样 debug 更清楚：在 `ScheduleLoops` after dump 里可以直接检查 schedule metadata 是否符合预期。
- lit test 可以只测 scheduling，不必同时验证复杂的 prologue/epilogue lowering。

Cost:

- `tt.latency` 是临时 attr，`ScheduleLoops` 会删除它；如果调试只看更后面的 dump，需要知道 latency 已被消费。
- `loop.stage` / `loop.cluster` 也是中间 contract，必须和后续 Pipeline 对齐。

## 13. Architecture Evolution

Ampere:

- 典型目标是把 global load / layout conversion / `tt.dot` 安排成多 stage，最终服务于 `mma.sync` path。
- 本 canonical sm86 matmul 因 small-load filter 没有留下 latency anchor，所以 ScheduleLoops no-op。

Hopper:

- WGMMA path 引入 shared-memory operand staging，`tt.load -> ttg.local_alloc -> ttng.warp_group_dot` 更依赖正确 wait / pipeline。
- 本 canonical sm90 matmul 同样没有留下 visible latency anchor，所以 ScheduleLoops no-op。

Blackwell:

- MMAv5/TMEM path 增加 TMEM token、TMEM load/store、self latency。
- 本 canonical sm100 matmul 中 `ttng.tc_gen5_mma` 的 `tt.latency = 1` 被 ScheduleLoops 消费，产生 schedule attr。

Compiler implication:

```text
ScheduleLoops 的核心算法基本是 architecture-independent：
它看 latency attrs、existing schedule、def-use 和 loop-carried deps。
架构差异主要通过更早 pass 产生的 op 形态和 AssignLatencies 留下的 anchors 进入这个 pass。
pass 内部少数 arch-coupled 点是 warp-specialize 分支的 isLatencyOp 白名单：
它显式列了 TMEMLoadOp/TMEMStoreOp、TMAOpInterface、MMAv5OpInterface、barrier wait/arrive 等 Hopper/Blackwell 相关 op。
```

## 14. If This Pass Did Not Exist

Correctness:

- 对完全不需要 pipelining 的 loop，可能仍能 lower。
- 对需要 Pipeline 展开的 loop，后续 pass 缺少 `loop.stage` / `loop.cluster`，无法可靠知道哪些 op 应该提前、复制或放入 epilogue。

Performance:

- Global load、TMA、shared-memory staging、MMA/TMEM async work 很难和 compute overlap。
- 即使 `AssignLatencies` 识别了 latency anchor，也没有 pass 把 hint 转成可执行的 stage plan。

Compiler pipeline:

- `AssignLatencies` 的 `tt.latency` contract 没有 consumer。
- `Pipeline` 需要重新承担 scheduling 分析，pass 边界变混乱，debug 和 lit 覆盖都会变差。

## 15. Knowledge Card

```text
Pass: TritonGPUScheduleLoops
Purpose: build a coarse loop-pipelining schedule from latency hints or existing WS schedule
Compiler decision: assign loop body ops to pipeline stages and ordering clusters
Main IR attribute/op:
  input: tt.latency
  output: loop.stage, loop.cluster, tt.scheduled_max_stage
Input contract:
  safe inner scf.for; latency attrs from AssignLatencies or existing serialized schedule
Output contract:
  every scheduled op has stage/cluster metadata; forOp records max scheduled stage
Invariant:
  shapes, element types, addresses, and semantics unchanged
Hardware reason:
  provide the stage plan needed for latency hiding, async wait placement, and multibuffering
Next dependencies:
  TritonGPUPipeline, LowerLoops, async load/TMA/WGMMA/MMAv5/TMEM lowering
```

## 16. Open Questions

- canonical sm86/sm90 matmul 的 A/B loads 因 width=16 被 AssignLatencies drop；后续可以构造一个 contiguous/vectorized load dump，让 ScheduleLoops 在 Ampere/Hopper 上产生真正多 stage schedule。
- Blackwell canonical sm100 这里只生成 `scheduled_max_stage = 0`；更复杂的 `pipeline-schedule-loop.mlir` 和 `loop-pipeline-blackwell.mlir` 能展示 `tc_gen5_mma` 与 `tmem_load/store` 跨 stage 的完整行为。
- 当前笔记没有覆盖 `kWarpSpecializeAttrName` + existing serialized schedule 分支；后续需要构造或定位一个带 warp-specialize attr 的 dump，观察 normalize/shrink 行为。
- 当前 canonical matmul 没有 `scf.if`；后续可以用带 mask/conditional prefetch 的 loop 补一例，观察 prologue/epilogue cluster 处理。
- `loop.cluster` 的精确定义最好结合 `Pipeline` / `LowerLoops` 再读一次，因为它的最终语义体现在 loop 展开后的 linearized schedule。
