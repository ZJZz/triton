# 2026-06-29 学习笔记：AssignLatencies

## 1. Pass 基本信息

本轮学习 `TritonGPUAssignLatencies`。它位于软件流水相关 pass 的前半段：

```text
... -> FuseNestedLoops / canonicalize
    -> AssignLatencies
    -> ScheduleLoops
    -> Pipeline
    -> ...
```

一句话：

```text
AssignLatencies 不重排 loop，也不直接生成 async copy。
它只给适合 pipelining 的 op 写入 tt.latency / tt.self_latency 属性，
把“哪些 op 应该跨 stage overlap”这个决策交给后续 ScheduleLoops / Pipeline 消费。
```

相关文件：

- Source: [AssignLatencies.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:252)
- Pass td: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:29)
- Attribute serialization: [PipeliningUtility.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:322)
- Next consumer: [ScheduleLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:251)
- Unit-style coverage: [pipeline-assign-latencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-assign-latencies.mlir:1)

Canonical matmul dumps used here:

- Ampere before: [058_Before_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:1)
- Ampere after: [059_After_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:1)
- Hopper before: [056_Before_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:1)
- Hopper after: [057_After_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:1)
- Blackwell before: [056_Before_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:1)
- Blackwell after: [057_After_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:1)

## 2. Architecture Matrix

| Arch | Before feature | After feature | Changed? |
| --- | --- | --- | --- |
| Ampere `sm86` | loop body has `tt.load`, `tt.dot`, pointer increments | no visible `tt.latency`; IR text is effectively unchanged in this dump | no visible change |
| Hopper `sm90` | loop body has `tt.load`, `ttg.local_alloc`, `ttng.warp_group_dot` | no visible `tt.latency`; IR text is effectively unchanged in this dump | no visible change |
| Blackwell `sm100` | loop body has `tt.load`, `ttg.local_alloc`, `ttng.tc_gen5_mma` with TMEM token | `ttng.tc_gen5_mma` gets `{tt.latency = 1, tt.self_latency = 1}` | yes |

Important caveat:

```text
canonical matmul sm86/sm90 这里没有 visible latency attr，
不是说 AssignLatencies 不会给 Ampere/Hopper load 标 latency。
test/TritonGPU/pipeline-assign-latencies.mlir 明确覆盖了 tt.load {tt.latency = 2}
和 small-load 不标记的行为。
```

Debug verification:

用下面的环境变量重跑 AOT compile，可以打开 pass 内部的 `LDBG`：

```bash
TRITON_CACHE_DIR=/tmp/triton-cache-assign-latencies \
TRITON_ENABLE_LLVM_DEBUG=1 \
TRITON_LLVM_DEBUG_ONLY=triton-loop-pipeline \
python learn_triton/tools/compile_driver.py \
  learn_triton/kernels/matmul.py matmul_kernel \
  "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
  86 4 3 1
```

`TRITON_CACHE_DIR` 指到 `/tmp` 是为了避免 sandbox 里写 `/home/.../.triton/cache`。

三代的关键 debug 输出一致：

```text
sm86:
Load %54 = tt.load %arg13 ... has width 16
Load %54 = tt.load %arg13 ... is too small for pipelining
Load %55 = tt.load %arg14 ... has width 16
Load %55 = tt.load %arg14 ... is too small for pipelining

sm90:
Load %54 = tt.load %arg13 ... has width 16
Load %54 = tt.load %arg13 ... is too small for pipelining
Load %56 = tt.load %arg14 ... has width 16
Load %56 = tt.load %arg14 ... is too small for pipelining

sm100:
Load %55 = tt.load %arg13 ... has width 16
Load %55 = tt.load %arg13 ... is too small for pipelining
Load %57 = tt.load %arg14 ... has width 16
Load %57 = tt.load %arg14 ... is too small for pipelining
```

所以 canonical matmul 的 A/B loads 没有 `tt.latency` 的根因已经坐实：

```text
canBeConvertedToAsyncLoad computes width = 16 bits,
but requires width >= 32 bits.
因此 AssignLoadLatencies drop 了这两个 loads。
```

## 3. Cross-Architecture Before Comparison

进入 `AssignLatencies` 前，三代 before 已经明显不同。这些差异不是
`AssignLatencies` 造成的，而是更早的 `AccelerateMatmul`、TMEM/warp-specialization
相关 pass 已经把 dot path 分叉了。

Ampere before:

- module target 是 `cuda:86`，见 [058_Before_TritonGPUAssignLatencies.mlir:18](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:18)
- loop body 是 register-side load + layout convert + `tt.dot`，见 [058_Before_TritonGPUAssignLatencies.mlir:66](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:66)

```text
%a = tt.load ...
%b = tt.load ...
%a_47 = ttg.convert_layout ...
%b_48 = ttg.convert_layout ...
%acc_49 = tt.dot ...
```

Hopper before:

- module target 是 `cuda:90`，见 [056_Before_TritonGPUAssignLatencies.mlir:21](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:21)
- loop body 已经是 `tt.load -> ttg.local_alloc -> ttng.warp_group_dot`，见 [056_Before_TritonGPUAssignLatencies.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:69)

```text
%a = tt.load ...
%a_47 = ttg.local_alloc ...
%b = tt.load ...
%b_48 = ttg.local_alloc ...
%acc_49 = ttng.warp_group_dot ...
```

Blackwell before:

- module target 是 `cuda:100`，见 [056_Before_TritonGPUAssignLatencies.mlir:22](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:22)
- loop 外有 TMEM allocation/store，loop body 是 `ttng.tc_gen5_mma`，见 [056_Before_TritonGPUAssignLatencies.mlir:72](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:72)

```text
%acc_36, %acc_37 = ttng.tmem_alloc ...
%acc_38 = ttng.tmem_store ...
%acc_39:4 = scf.for ...
  %acc_56 = ttng.tc_gen5_mma ...
```

Conclusion:

```text
Before IR 已经分叉。
AssignLatencies 在各自输入基础上标记 latency；
它本身没有根据 arch target 做显式 cuda:86/cuda:90/cuda:100 分支，
但它会识别 Blackwell path 里的 MMAv5OpInterface。
```

## 4. IR Changes

### 4.1 Ampere `sm86`: no visible change in canonical matmul

Before loop core:

- `scf.for` 见 [058_Before_TritonGPUAssignLatencies.mlir:66](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:66)
- `tt.load` A/B 见 [058_Before_TritonGPUAssignLatencies.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:67)
- `tt.dot` 见 [058_Before_TritonGPUAssignLatencies.mlir:71](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:71)

After loop core:

- same `scf.for` 见 [059_After_TritonGPUAssignLatencies.mlir:66](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:66)
- same `tt.load` A/B 见 [059_After_TritonGPUAssignLatencies.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:67)
- same `tt.dot` 见 [059_After_TritonGPUAssignLatencies.mlir:71](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:71)

IR-level meaning:

```text
当前 canonical sm86 matmul 中，pass 没有留下可见 latency attr。
原因不是 loop trip count，也不是 numStages。
debug log 显示 A/B load 的 async width 都是 16，小于 width >= 32 的门槛，
所以 AssignLoadLatencies 把它们 drop 了。
```

### 4.2 Hopper `sm90`: no visible change in canonical matmul

Before loop core:

- `scf.for` 见 [056_Before_TritonGPUAssignLatencies.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:69)
- A/B `tt.load` + `ttg.local_alloc` 见 [056_Before_TritonGPUAssignLatencies.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:70)
- `ttng.warp_group_dot` 见 [056_Before_TritonGPUAssignLatencies.mlir:74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:74)

After loop core:

- `scf.for` 见 [057_After_TritonGPUAssignLatencies.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:69)
- A/B `tt.load` + `ttg.local_alloc` 见 [057_After_TritonGPUAssignLatencies.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:70)
- `ttng.warp_group_dot` 见 [057_After_TritonGPUAssignLatencies.mlir:74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:74)

IR-level meaning:

```text
当前 canonical sm90 matmul 中，A/B loads 虽然 feed local_alloc + WGMMA，
但 after dump 没有可见 tt.latency attr。
debug log 显示它们同样是 width = 16，被 small-load filter drop。
这不是 shared encoding incompatible 的问题；LDBG 没有打印 cannot have shared encoding。
```

### 4.3 Blackwell `sm100`: MMAv5 gets latency and self_latency

Before:

- `ttng.tc_gen5_mma` 没有 latency attributes，见 [056_Before_TritonGPUAssignLatencies.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:79)

```text
%acc_56 = ttng.tc_gen5_mma %a_54, %b_55, %acc_36[%acc_53], %acc_52, %true
```

After:

- 同一个 `ttng.tc_gen5_mma` 得到 `{tt.latency = 1 : i32, tt.self_latency = 1 : i32}`，见 [057_After_TritonGPUAssignLatencies.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:79)

```text
%acc_56 = ttng.tc_gen5_mma ...
  {tt.latency = 1 : i32, tt.self_latency = 1 : i32}
```

IR-level meaning:

```text
tt.latency = 1:
  MMA 到其 users 的 pipeline stage distance hint 是 1。

tt.self_latency = 1:
  MMA 自身相关的 async wait / token 关系可以推迟 1 个 pipeline stage。
```

注意一个细节：debug log 在 `ScheduleLoops` 阶段打印的 scheduled IR 里只看到
`tt.self_latency = 1`，看不到 `tt.latency = 1`。这是因为 `ScheduleLoops` 会通过
`deserializeLatencies` 读走并删除 `tt.latency`；而 `tt.self_latency` 仍留在 IR 上给后续
MMAv5 lowering 使用。

## 4.4 Effective Or No-op

| Arch | Effective? | Evidence |
| --- | --- | --- |
| `sm86` | no visible change | before/after loop body 行号相同形态，A/B loads 没有 `tt.latency`，见 [059_After_TritonGPUAssignLatencies.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:67) |
| `sm90` | no visible change | before/after loop body 行号相同形态，A/B loads 没有 `tt.latency`，见 [057_After_TritonGPUAssignLatencies.mlir:70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:70) |
| `sm100` | yes | `ttng.tc_gen5_mma` 得到 `{tt.latency = 1 : i32, tt.self_latency = 1 : i32}`，见 [057_After_TritonGPUAssignLatencies.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:79) |

## 5. Source Mapping

### 5.1 Pass 入口：只处理可 pipeline 的 inner loop

源码：

- `preCondition` 见 [AssignLatencies.cpp:29](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:29)
- pass 主逻辑见 [AssignLatencies.cpp:252](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:252)

逻辑：

```text
for each scf.for:
  skip if loop distance > 1
  skip if outer loop
  separately skip if numStages <= 1
  if user already provided tt.latency:
    preserve / collect user latencies
  else:
    AssignLoadLatencies
    AssignMMALatencies
serialize latency attrs to IR
```

精确地说，`preCondition` 本身只检查 `loopHasDistGreaterThanOne` 和 `isOuterLoop`。
`numStages <= 1` 是 `assignLatencies` 里和 `preCondition` 并列的检查。

这解释了为什么 vecadd dump 是 no-op：vecadd 没有 `scf.for` loop，after 只保留普通
`tt.load`，没有 latency attr，见 [055_After_TritonGPUAssignLatencies.mlir:20](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/055_After_TritonGPUAssignLatencies.mlir:20)。

对 canonical matmul，`K` 是运行时参数，loop 形态是 `scf.for %k = 0 to %K step 32`。
即使某次运行时 `K=32` 只执行一次迭代，这个 pass 也不会用 runtime trip count 决定是否标记；
它只看 IR 里的 loop、`numStages` 和 legality/benefit 条件。

### 5.2 Load latency：从 dot / TMEMStore 反向 DFS 找 load chain

源码：

- `AssignLoadLatencies::run` 见 [AssignLatencies.cpp:65](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:65)
- `loadOpsToIndirectionLevel` 见 [AssignLatencies.cpp:282](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:282)
- small load / async load width filter 见 [PipeliningUtility.cpp:295](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:295)
- dot accumulator operand skip 见 [AssignLatencies.cpp:319](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:319)
- multi-distance exclusion 见 [AssignLatencies.cpp:298](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:298)
- `pipelineWithoutDot` extra DFS 见 [AssignLatencies.cpp:341](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:341)

核心规则：

```text
start from DotOpInterface or TMEMStoreOp
walk backward through operands
if current op is dot:
  skip operand 2, the accumulator
when seeing tt.load / descriptor load:
  check if it is beneficial and legal to pipeline
  record indirection distance
  if same load appears at different distances:
    erase it and exclude it
if loop has explicit num_stages attr:
  also consider loads not directly feeding dot
drop loads whose distance >= numStages - 1
loadLatency = (numStages - 1) / (maxIndirectionLevel + 1)
```

对默认 `numStages = 3`：

- direct load feeding dot: `maxIndirectionLevel = 0`，所以 `loadLatency = 2`
- indirect load chain with two load levels: `maxIndirectionLevel = 1`，所以每层 `loadLatency = 1`

测试证据：

- direct loads 得到 `{tt.latency = 2}`，见 [pipeline-assign-latencies.mlir:35](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-assign-latencies.mlir:35)
- indirect loads 得到 `{tt.latency = 1}`，见 [pipeline-assign-latencies.mlir:166](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-assign-latencies.mlir:166)
- small load 不标记，见 [pipeline-assign-latencies.mlir:40](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonGPU/pipeline-assign-latencies.mlir:40)

Canonical matmul 的具体根因：

```text
A/B loads are direct loads feeding dot-like compute, so the formula would be latency = 2.
But before assigning latency, isPipeliningBeneficial calls canBeConvertedToAsyncLoad.
For these loads, debug log reports width = 16.
PipeliningUtility.cpp requires width >= 32.
Therefore the loads are filtered out before opLatency[load] is written.
```

为什么 width 只有 16：

```text
canBeConvertedToAsyncLoad:
  width = axisInfoAnalysis.getContiguity(ptr) * pointeeElementBitWidth

当前 pointee type 是 f16，所以 elementBitWidth = 16。
debug log 里的 width = 16 反推出 contiguity(ptr) = 1。
```

这个 `contiguity = 1` 不是因为 `64x32` / `32x64` 的 logical shape 小，而是因为
canonical matmul 的 pointer expression 依赖 runtime stride。before IR 里 A 指针使用
`%stride_am` / `%stride_ak`，例如 [058_Before_TritonGPUAssignLatencies.mlir:39](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:39)
到 [49](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:49)；B 指针同样依赖
`%stride_bk` / `%stride_bn`，见 [058_Before_TritonGPUAssignLatencies.mlir:52](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:52)
到 [60](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:60)。

AxisInfo 不能静态证明这些 runtime stride 形成连续访问，所以保守给出
`contiguity = 1`。因此：

```text
contiguity 1 * f16 16 bits = width 16 bits
width 16 < 32
load latency is not assigned
```

一个有用推论：这个 pass 真正需要的是最终 pointer SSA value 上可被
AxisInfo 读到的 contiguity 信息。在本实验中，`tl.max_contiguous` /
`tl.multiple_of` 直接标注最终 pointer 后，`contiguity * elementBitWidth >= 32`
才成立；单纯把内层 stride 写成 compile-time contiguous 不足以生效。通过
small-load filter 后，这些 loads 才有机会进入 `opLatency[load] = 2` 的
direct-load path。这里仍需满足 shared encoding compatibility 和 multi-distance
exclusion 等其他条件，所以这不是无条件保证。

这也解释了为什么 sm86/sm90/sm100 的 A/B loads 都没有 `tt.latency`。三代的 dot/MMA
形态不同，但 load vector width 这一关相同地失败了。

验证这个推论的隔离 case：

- kernel: [matmul_contiguous.py](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/kernels/matmul_contiguous.py:1)
- Ampere before/after:
  [058_Before_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/058_Before_TritonGPUAssignLatencies.mlir:65),
  [059_After_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/059_After_TritonGPUAssignLatencies.mlir:65)
- Hopper before/after:
  [056_Before_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm90_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:68),
  [057_After_TritonGPUAssignLatencies.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm90_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:68)

这个 case 做两件事：

```python
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :]
a_ptrs = tl.max_contiguous(tl.multiple_of(a_ptrs, (1, BLOCK_K)), (1, BLOCK_K))
b_ptrs = tl.max_contiguous(tl.multiple_of(b_ptrs, (1, BLOCK_N)), (1, BLOCK_N))
```

第一步把 A 的 K 维、B 的 N 维写成 compile-time contiguous；第二步更关键，把最终
pointer SSA value 显式标上 `tt.divisibility` / `tt.contiguity`。只写死内层 stride
在本实验中还不够：before IR 中最终 `%a_ptrs` / `%b_ptrs` 没有保留可供
`ModuleAxisInfoAnalysis` 使用的 `tt.contiguity` attr，debug log 仍然是 `width = 16`。
加上 pointer-level hints 后，debug log 变为：

```text
sm86:
Load %52 = tt.load %arg12 ... has width 128
Load %52 = tt.load %arg12 ... considered for pipelining with distance 0
Load %53 = tt.load %arg13 ... has width 128
Load %53 = tt.load %arg13 ... considered for pipelining with distance 0

sm90:
Load %52 = tt.load %arg12 ... has width 128
Load %52 = tt.load %arg12 ... considered for pipelining with distance 0
Load %54 = tt.load %arg13 ... has width 128
Load %54 = tt.load %arg13 ... considered for pipelining with distance 0
```

after IR 也闭合：

```text
sm86:
%a = tt.load %a_ptrs_44 {tt.latency = 2 : i32}
%b = tt.load %b_ptrs_45 {tt.latency = 2 : i32}

sm90:
%a = tt.load %a_ptrs_44 {tt.latency = 2 : i32}
%b = tt.load %b_ptrs_45 {tt.latency = 2 : i32}
```

这里没有看到 `too small for pipelining` 或 `cannot have shared encoding`，说明 small-load
filter 和后续 shared/local_alloc gate 都通过了。Ampere 的路径是
`tt.load -> convert_layout -> tt.dot`，Hopper 的路径是
`tt.load -> ttg.local_alloc -> ttng.warp_group_dot`；两者在这个 case 中都能拿到
`tt.latency = 2`。

### 5.3 MMA latency：只处理 MMAv5 interface

源码：

- `AssignMMALatencies::run` 见 [AssignLatencies.cpp:156](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:156)
- MMAv5 type check 见 [AssignLatencies.cpp:165](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:165)
- `tt.latency = 1` 写入逻辑见 [AssignLatencies.cpp:183](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:183)
- `tt.self_latency` serialization 见 [AssignLatencies.cpp:218](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp:218)

关键代码路径：

```text
if op implements ttng::MMAv5OpInterface:
  skip if loop still contains sync tt.dot
  use MMAv5PipelineableOperandsHelper
  if pipelineable or operands state determined and no loads after MMA:
    mmaSelfLatency[mma] = 1
    if accumulator buffering is not required or possible:
      opLatency[mma] = 1
```

这里有两个 gate：

- `tt.self_latency = 1` 的 gate 是 MMAv5 op 可 pipeline，或者 operand state 已确定且 MMA 后面没有 loads。
- `tt.latency = 1` 的 gate 更严格，还要求 accumulator 不需要 multibuffering，或者 multibuffering 可行且没有被禁用。

这直接解释 Blackwell canonical dump：

- before `ttng.tc_gen5_mma` 是 MMAv5 op，见 [056_Before_TritonGPUAssignLatencies.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/056_Before_TritonGPUAssignLatencies.mlir:79)
- after 得到 `tt.latency = 1` 和 `tt.self_latency = 1`，见 [057_After_TritonGPUAssignLatencies.mlir:79](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/057_After_TritonGPUAssignLatencies.mlir:79)

它也解释 Ampere/Hopper canonical dump：

```text
Ampere uses tt.dot, not MMAv5OpInterface.
Hopper uses ttng.warp_group_dot, not the Blackwell MMAv5 op in this path.
所以 MMA-latency 分支不会给它们的 dot op 加 tt.self_latency。
```

`hasSyncDots` 这个 gate 查的是 `mlir::triton::DotOp`，不是 `ttng.warp_group_dot`。
所以 Hopper 的 WGMMA op 不会因为 `hasSyncDots` 被当成 sync dot；它只是没有实现这里匹配的
`ttng::MMAv5OpInterface`。

### 5.4 Attr 是临时 contract，ScheduleLoops 会读走并删除

源码：

- 写入 `tt.latency` 见 [PipeliningUtility.cpp:322](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:322)
- 写入 `tt.self_latency` 见 [PipeliningUtility.cpp:331](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:331)
- `ScheduleLoops` 读走并移除 latency attr，见 [PipeliningUtility.cpp:340](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:340)
- `ScheduleLoops` 用 latency op 构造 initial schedule，见 [ScheduleLoops.cpp:152](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp:152)

这说明 `tt.latency` 不是最终 IR 语义，而是两个 pass 之间的临时调度 contract。

### 5.5 latency 数值描述的是 stage distance，不是 cycle

核心概念：

```text
numStages:
  software pipeline 可用的总 stage 槽数，也可以粗略理解成总逻辑节拍数。

tt.latency = N:
  producer 到 downstream users 的 pipeline stage distance hint。

tt.self_latency = N:
  op 到自身相关 async wait / token / next producer 关系的 pipeline stage distance hint。
```

这里的“节拍”是 compiler software pipeline 的逻辑 stage，不是 GPU clock cycle。
`tt.latency = 2` 不表示 load 延迟 2 cycles，而是告诉 `ScheduleLoops`：

```text
这个 load 是一个 latency anchor；
请把它和它的 consumer chain 尽量拉开 2 个 pipeline stages。
```

例如 `numStages = 3`、direct load 时：

```text
stage 0: load for a future iteration
stage 1: copy / layout / preparation
stage 2: dot / MMA for the current iteration
```

所以三者的关系是：

```text
numStages = pipeline stage budget
latency / self_latency = 在这个 budget 内给某条依赖链的调度距离提示
```

这个数值也不是 live range。`tt.latency` 会间接影响 live range，因为 load
被提前后，它的结果或 shared buffer 可能需要存活更久；但 live range 是后续调度的结果，
不是这个 attr 本身描述的对象。更准确的叫法是：

```text
pipeline stage dependency distance hint
```

它也不保证最优。`AssignLatencies` 没有精确建模 global memory latency、MMA issue rate、
occupancy、register pressure、shared memory conflict 等硬件细节。这个数值的意义是：

- 标记哪些 op 值得成为 pipeline anchor。
- 在 `numStages` 预算内给 producer-consumer chain 一个可执行的相对 stage 距离。
- 把 `AssignLatencies` 的识别结果传给 `ScheduleLoops`，让后者能生成 `loop.stage` /
  `loop.cluster`，而不是只看到普通 SSA def-use。

因此它是一个保守、可解释、可 schedule 的 heuristic contract。最终是否接近最优，
仍然依赖 `num_stages` / block size / num warps 的 heuristic 或 autotune，以及 benchmark。

### 5.6 心智模型：layout 空间的 anchor vs stage 空间的 anchor

可以把 `AssignLatencies` 和 `RemoveLayoutConversions` 放在同一个抽象框架里理解：

```text
先找 anchor op，
再围绕 anchor 给上下游建立某种 compiler contract。
```

`RemoveLayoutConversions` 的 anchor 更像是在 layout 空间里说：

```text
我必须要这个 layout。
能一起用这个 layout 的 value 就一起用；
不能一起用的边界，就保留或插入 convert_layout。
```

所以它最后影响的是 tensor encoding / `ttg.convert_layout` 边界。它服务的是
layout 一致性、tensor-core operand layout、memory layout 等空间上的约束。

`AssignLatencies` 的 anchor 更像是在 pipeline stage 空间里说：

```text
我是长延迟或 async 相关的起点。
依赖我的 downstream use chain，调度时最好和我隔 N 个 pipeline stages。
```

这里的 downstream 不是源码文本顺序里的“下面”，而是 SSA def-use 上的 users。
例如：

```text
tt.load -> ttg.local_alloc / ttg.convert_layout -> tt.dot / MMA
```

如果 `tt.load` 被选成 latency anchor，并获得 `tt.latency = 2`，它表达的是：

```text
load 这条链值得提前；
后续 consumer chain 最好被 ScheduleLoops 排到相对更晚的 stage。
```

这个类比的边界也很重要：

- `RemoveLayoutConversions` 会真实改写 layout / convert 边界。
- `AssignLatencies` 本 pass 只写 `tt.latency` / `tt.self_latency`，不直接重排 loop。
- 真正把 op 放到 `loop.stage` / `loop.cluster` 的是后续 `ScheduleLoops` / `Pipeline`。

一句话记忆：

```text
RemoveLayoutConversions 是 layout 空间上的 anchor negotiation。
AssignLatencies 是 pipeline 时间/stage 空间上的 anchor scheduling hint。
```

## 6. Compiler Decision

Compiler question:

```text
在一个 loop 里，哪些 op 足够昂贵或异步化收益足够大，应该被后续 pipeline
安排到更早 stage，从而和后续 dot/MMA/compute overlap？
```

Decision made here:

- 为 direct/indirect load chain 计算 stage distance。
- 为 Blackwell MMAv5 async MMA 标出 MMA 自身 latency 和 user latency。
- 对 small load、outer loop、distance > 1 loop、`numStages <= 1` loop 不标记。
- 对 dot accumulator operand 不做 load pipelining。

Why here:

```text
这个 pass 位于 layout / matmul acceleration 之后，所以能看到 load 是否 feed dot、
Hopper 是否 local_alloc + WGMMA、Blackwell 是否 TCGen5/TMEM。
它位于 ScheduleLoops / Pipeline 之前，所以还可以用 lightweight attr
把 scheduling hint 传给后续 pass。
```

## 7. Compiler Contract

Input contract:

- TTGIR loop 已经形成 `scf.for`。
- matmul path 已经被更早 pass 改写成目标架构形态。
- `numStages` 已知，来自 pass option 或 loop attribute。
- load/dot/MMA producer-consumer chain 仍然在同一个 loop block 内，可做 backward DFS。

Output contract:

- `tt.latency = N` 表示这个 op 到其 downstream users 的 pipeline stage distance hint 是 N。
- `tt.self_latency = N` 表示这个 MMA-like op 自身相关 async wait / token 关系的 stage distance hint 是 N。
- 未被标记的 op 不作为 latency-starting op。

Next pass relies on:

- `ScheduleLoops` 用 `deserializeLatencies` 读取并删除 `tt.latency`，再计算 stage schedule。
- `Pipeline` 根据 schedule 真正重排 loop、生成 prologue/epilogue、多 buffer 和 async behavior。
- Blackwell lowering 会通过 `self_latency` 影响 MMAv5 wait / token placement。

## 8. Invariant

- Tensor shape: unchanged。
- Element type: unchanged。
- Memory address logical meaning: unchanged。
- Program semantics: unchanged。
- Control flow: unchanged in this pass。
- Changed only: op attributes that encode scheduling latency hints。

## 9. Triton Mechanism

`AssignLatencies` 是 Triton 软件流水线前的标注 pass。

它不做这些事：

- 不把 `tt.load` 改成 async load。
- 不插入 shared memory allocation。
- 不重排 loop body。
- 不生成 prologue / epilogue。
- 不选择 MMA/WGMMA/TCGen5 指令形态。

它做这些事：

- 找 loop 中值得 pipelining 的 load chain。
- 找 Blackwell MMAv5 async MMA。
- 用 `tt.latency` / `tt.self_latency` 把 stage distance 写回 IR。

## 10. GPU / Hardware / Optimization Reason

Hardware reason:

- GPU global memory load latency 很高，software pipeline 需要提前发起 load。
- Blackwell TCGen5 MMA / TMEM token path 有 async execution / wait placement 问题。

Instruction reason:

- `cp.async` 类路径对 copy size 有限制；小 load 不一定值得异步化。
- MMAv5 op 可以和自身或后续 users overlap，但需要 accumulator buffering / token 规则满足。

Execution reason:

- `numStages = 3` 代表 loop 可以跨多个 stage 组织当前迭代 compute 和未来迭代 load。
- `latency` attr 是后续 coarse schedule 的 anchor。

Memory reason:

- load 只有在可转换成 shared-memory async path、shared encoding 兼容、vector width 足够时才标记。
- 这避免为了 pipeline 小 load 而增加 register pressure 或生成不合法 cp.async。

## 11. Decision Tree

```text
for each scf.for:
  if loopHasDistGreaterThanOne:
    skip
  if isOuterLoop:
    skip
  if numStages <= 1:
    skip

  if any op already has tt.latency:
    collect user-provided latencies
    continue

  AssignLoadLatencies:
    pipelineWithoutDot = loop has explicit num_stages attr
    start DFS from DotOpInterface or TMEMStoreOp
    when visiting a dot:
      skip operand 2 accumulator
    for each tt.load / descriptor load found:
      require pipelining beneficial
      require compatible shared encoding
      require not too small for async copy when filter applies
      if same load was seen at a different distance:
        erase it and exclude it
      record indirection distance
    if pipelineWithoutDot:
      also DFS from other loop ops to catch non-dot loads
    drop loads with distance >= numStages - 1
    assign loadLatency = (numStages - 1) / (maxIndirectionLevel + 1)

  AssignMMALatencies:
    for each op implementing MMAv5OpInterface:
      if loop has sync tt.dot:
        skip MMA latency
      if operands are pipelineable or state is known and no loads after MMA:
        set self_latency = 1
        if accumulator buffering is okay:
          set latency = 1
      if warp specialized:
        adjust latency/self_latency for WS constraints

serialize tt.latency and tt.self_latency attrs
```

## 12. Alternative Design

Alternative:

```text
让 ScheduleLoops 自己分析 load/dot/MMA chain，不单独保留 AssignLatencies pass。
```

Why not here:

- `AssignLatencies` 把“识别 latency op”和“根据 latency op 排 stage”分开，职责更清楚。
- 后续 `ScheduleLoops` 可以只消费 `opLatency` map，不需要重复理解 async load legality、small load filter、MMAv5 accumulator buffering。
- 用户手写 `tt.latency` 时，pass 也能把手工 latency 和自动 latency 放在同一个 pipeline contract 里。

Cost:

- latency attr 是临时 IR 状态；如果调试只看后续 dump，可能看不到它，因为 `ScheduleLoops` 会读取并删除。
- pass 是否 visibly changed 取决于具体 kernel，容易误判为 no-op。

## 13. Architecture Evolution

Ampere:

- 主要目标是让 load feeding `tt.dot` 有机会进入 software pipeline / async copy。
- canonical matmul dump 中没有 visible latency attr；debug log 证明原因是 A/B load width = 16，小于 async load 门槛。
- unit tests 覆盖了 Ampere-style `tt.dot` load latency。

Hopper:

- before 已经是 shared-memory operand + `ttng.warp_group_dot`。
- load latency path 仍然重要，但 canonical dump 的 A/B loads 同样因为 width = 16 被 drop。
- WGMMA 自身没有走 `MMAv5OpInterface` 分支，因此没有 `tt.self_latency`。

Blackwell:

- before 已经包含 TMEM allocation 和 `ttng.tc_gen5_mma`。
- `AssignLatencies` 明确给 MMAv5 op 标 `tt.latency = 1` 与 `tt.self_latency = 1`。
- A/B loads 仍然因为 width = 16 被 small-load filter drop。
- 这反映 Blackwell compiler pipeline 需要表达 async MMA / TMEM token 的 schedule contract。

Compiler implication:

```text
Ampere/Hopper 更关注 load-to-dot latency；
Blackwell 额外需要 MMA self latency。
AssignLatencies 因此成为 Blackwell TCGen5/TMEM path 的重要 contract pass。
```

## 14. If This Pass Did Not Exist

Correctness:

- 对普通同步路径，缺少它通常不是立刻语义错误，因为 loop 仍能按原顺序执行。
- 对需要 async MMA wait placement 的 Blackwell path，缺少 `self_latency` 会让后续 lowering 少一个明确 scheduling contract。

Performance:

- `ScheduleLoops` 没有 latency anchors，可能无法把 load 提前到合适 stage。
- global load 与 dot/MMA overlap 会变差。
- Blackwell MMAv5/TMEM path 可能无法正确表达 MMA 自身跨迭代 overlap。

Compiler pipeline:

- `ScheduleLoops` 会走空 schedule 或 fallback schedule。
- `Pipeline` 缺少依据来生成高质量 prologue/steady-state/epilogue。

## 15. Knowledge Card

```text
Pass: TritonGPUAssignLatencies
Purpose: mark latency-carrying ops before software pipeline scheduling
Compiler decision: which loads/MMAv5 ops should anchor pipeline stages
Main IR attribute/op: tt.latency, tt.self_latency
Input contract: inner scf.for exists; numStages > 1; matmul/load chain is visible
Output contract: latency attrs encode stage-distance hints for ScheduleLoops
Invariant: shapes, types, memory semantics, and control flow unchanged
Hardware reason: overlap global loads / async MMA with compute, avoid bad small-load pipelining
Next dependencies: ScheduleLoops, Pipeline, Blackwell MMAv5 lowering/wait placement
```

## 16. Five-Sentence Summary

1. Before -> After 主要 IR 变化是：canonical Blackwell matmul 的 `ttng.tc_gen5_mma` 获得 `tt.latency = 1` 和 `tt.self_latency = 1`；canonical Ampere/Hopper matmul 没有 visible attr。
2. Ampere/Hopper 的 canonical A/B loads 没有 `tt.latency` 的根因已经由 debug log 坐实：runtime stride 让 AxisInfo 保守得到 `contiguity = 1`，所以 f16 load width = 16，小于 `canBeConvertedToAsyncLoad` 的 `width >= 32` 门槛。
3. 这段逻辑在 Triton 机制上是在给后续 software pipeline pass 写调度提示，而不是直接重排 IR。
4. 它服务的 GPU / hardware / optimization 目标是隐藏 global load latency，并在 Blackwell MMAv5/TMEM path 中表达 async MMA 的跨 stage / 跨迭代 latency。
5. 如果没有这个 pass，`ScheduleLoops` 缺少 latency anchors，后续 pipeline 质量下降，Blackwell MMAv5 wait/token scheduling 也少了关键 contract。

## 17. Open Questions

- 已验证：用 `tl.max_contiguous` / `tl.multiple_of` 标注最终 pointer SSA value 后，
  Ampere/Hopper A/B loads 的 width 从 16 提高到 128，并触发 `tt.latency = 2`。
  后续可以继续拆分实验：只用 constexpr stride、不加 pointer-level hint 时，
  哪个 frontend/lowering 阶段丢失了可用于 AxisInfo 的 contiguity 信息？
- `tt.latency = 1` 在 Blackwell after dump 里出现，但 `ScheduleLoops` debug IR 里被删除；
  后续可以单独学习 `ScheduleLoops` 如何把它转成 `loop.stage` / `loop.cluster`。
