# IR Pass Diff Learning Guide

这份文档定义我们后面学习 Triton compiler pass 的固定方法。

目标不是简单看 `diff`，而是通过一个 pass 的 `Before IR` 和 `After IR` 回答这些问题：

- 这个 pass 在回答什么 compiler decision？
- 这个 pass 改了哪些 IR？
- 每一类变化来自 pass 源码里的哪段逻辑？
- 这段逻辑主要在做什么？
- 为什么要这样改？
- after IR 建立了什么 compiler contract？
- 哪些 invariant 必须保持不变？
- 这是 Triton 编译器自己的机制，还是为了匹配 GPU 硬件/指令/执行模型？
- 如果没有这个 pass，后续 lowering 或性能会受到什么影响？

当前阶段的架构对比范围固定为 NVIDIA 三代：

| Architecture | Dump target | 重点 |
|---|---:|---|
| Ampere | `sm80` / `sm86` | `mma.sync`、Ampere tensor core baseline |
| Hopper | `sm90` | `wgmma`、TMA、warp specialization、Hopper TMEM path |
| Blackwell | `sm100` | Blackwell-specific MMA/TMEM/cluster path；必要时补看 `sm120` consumer Blackwell |

所以学习一个 pass 时，不只看单架构的 `Before -> After`，还要看三代架构之间的同一 pass 对比：

```text
Ampere:    Before -> After
Hopper:    Before -> After
Blackwell: Before -> After
```

如果三个架构的 `Before IR` 已经明显不同，说明当前 pass 接收到的输入目标本来就不同；这通常是更早的 arch-specific lowering 或 layout decision 已经生效。此时分析当前 pass 时，要先说明“输入 IR 已经分叉”，再分析每个架构自己的 after 目标。

## 核心视角：Decision + Contract

不要只把一个 pass 记成“它修改了什么 IR”。更好的记法是：

```text
Input IR
  -> compiler decision
  -> output IR
  -> hardware / execution motivation
  -> next-pass contract
```

也就是说，一个 pass 往往是在回答一个 compiler question：

| Compiler question | Typical pass / mechanism |
|---|---|
| logical tensor 如何映射到 GPU thread/warp/CTA？ | `ConvertTritonToTritonGPU` / layout encoding |
| 哪些 thread 访问相邻 memory address？ | `TritonGPUCoalesce` |
| 多个 CTA 如何分摊一个 tile？ | `PlanCTA` / `CGALayout` |
| dot operand 必须满足什么 MMA/WGMMA layout？ | `AccelerateMatmul` / `OptimizeDotOperands` |
| value 应该留在 register、shared memory 还是 TMEM？ | shared/TMEM allocation and layout passes |
| load 应该什么时候发起，才能和 compute overlap？ | `AssignLatencies` / `ScheduleLoops` / `Pipeline` |
| 哪些同步必须显式插入？ | fence / barrier passes |

所以每个 pass 最终至少要回答：

```text
Decision:
  Compiler 在这里必须做什么决定？

Input contract:
  进入这个 pass 前，IR 必须满足什么条件？

Output contract:
  pass 结束后，后续 pass 可以依赖什么事实？

Invariant:
  pass 不能改变哪些语义？

Deferred work:
  这个 pass 刻意没有解决什么，留给了后续哪个 pass？
```

## 一、固定学习流程

每看一个 pass，都按下面顺序记录。

### 1. Pass 基本信息

记录：

- pass 名字
- before dump 文件
- after dump 文件
- 对应源码文件
- 当前 kernel / arch / compile option
- pass 是否真的生效

例子：

```text
Pass: TritonGPUPlanCTAPass
Before: learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir
After:  learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir
Source: lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp
Kernel: matmul_kernel
Arch: sm90
Option: ttg.num-ctas = 2
Effective: yes
```

如果做架构对比，记录三份：

```text
Ampere:
  Before: learn_triton/dumps/<kernel>/sm86/...
  After:  learn_triton/dumps/<kernel>/sm86/...

Hopper:
  Before: learn_triton/dumps/<kernel>/sm90/...
  After:  learn_triton/dumps/<kernel>/sm90/...

Blackwell:
  Before: learn_triton/dumps/<kernel>/sm100/...
  After:  learn_triton/dumps/<kernel>/sm100/...
```

如果当前还没生成某个架构的 dump，就明确写：

```text
Blackwell: pending, need dump with TRITON_ARCH=100
```

### 2. 先只看 IR 变化，不急着解释

先做一张变化清单。只描述“看到了什么”，先不要解释“为什么”。

建议分类：

- module attribute 是否变了
- layout / encoding 定义是否变了
- tensor type 是否变了
- `ttg.convert_layout` 是否增加、减少、移动、换 source/target
- 关键 op 是否变了，比如 `tt.dot`、`tt.load`、`tt.store`、`scf.for`
- control flow 是否变了
- memory op 是否变了
- 后续 pass 可能依赖的 contract 是否变了

每条都要带文件和行号。

格式：

```text
IR change:
- Before line X: ...
- After line Y: ...
- Meaning at IR level: ...
```

注意：这一节只讲 IR 事实，不讲 pass 意图。

### 2.5. 横向比较三个架构的 Before IR

在解释当前 pass 之前，先比较：

```text
Ampere Before vs Hopper Before vs Blackwell Before
```

这一步回答：

- 进入这个 pass 前，三个架构的 IR 是否已经不同？
- 如果不同，差异主要在哪些层面？
  - `ttg.target`
  - module attributes
  - layout / encoding，比如 `#blocked`、`#mma`、`#ttg.dot_op`
  - `tt.dot` / `tt.load` / `tt.store` 的 tensor type
  - 是否已经出现 Hopper/Blackwell-specific op、layout 或 barrier
  - 是否已经有不同的 pipeline / warp specialization / TMEM 结构
- 当前 pass 的 after 差异，是当前 pass 造成的，还是继承了 before 差异？

判断原则：

```text
如果 Before 就不同：
  先把差异归因到更早的 pass 或 arch-specific pipeline。
  当前 pass 的任务是“在各自输入基础上继续变换”。

如果 Before 基本相同，但 After 不同：
  重点看当前 pass 内部是否有 arch/capability 分支。
  这通常说明当前 pass 本身在做 architecture-specific decision。

如果 Before 和 After 都类似：
  当前 pass 可能是 architecture-independent mechanism。
```

记录格式：

```text
Cross-architecture before comparison:
- Ampere before: ...
- Hopper before: ...
- Blackwell before: ...
- Conclusion: same input / already diverged before this pass
- If already diverged: likely caused by ...
```

### 3. 再把 IR 变化映射到 pass 源码

对每类 IR 变化，找对应源码逻辑。

记录格式：

```text
Code path:
- Source line X: ...
- Source line Y: ...
- This code creates/replaces/removes ...

IR evidence:
- Before line ...
- After line ...
```

不要只写“这是 PlanCTA 做的”。要写到函数/分支/关键变量，例如：

- `processDot`
- `getCTATiling`
- `insertCasts`
- `propagateBackward`
- `propagateForward`
- `processLoadStore`
- `processElementwise`

### 4. 写出 Compiler Decision

这一节回答“编译器在这里必须做什么决定”。

例子：

```text
ConvertTritonToTritonGPU:
  logical tensor 的元素如何映射到 GPU threads/warps/CTA？

PlanCTA:
  多个 CTA 如何分摊一个 output tile？

Pipeline:
  load 和 compute 应该怎样重排，才能隐藏 latency？
```

这一步很重要：IR 里的 `#blocked`、`CGALayout`、`ttg.convert_layout` 只是答案的表示形式，不是问题本身。

### 5. 写出 Compiler Contract

每个重要 pass 都要写 contract。

```text
Input contract:
- 这个 pass 假设 before IR 已经满足什么？

Output contract:
- after IR 建立了什么新事实？
- 后续 pass 可以依赖什么？

Next pass relies on:
- 哪些后续 pass 会消费这个 contract？
```

例子：

```text
ConvertTritonToTritonGPU
Input contract:
  TTIR tensors can be hardware-agnostic.
Output contract:
  Tensor values in TTGIR have GPU layout/encoding.
Next pass relies on:
  Coalesce and later layout passes can reason about thread/warp/CTA mapping.
```

```text
PlanCTA
Input contract:
  Dot/reduce/store-like ops already have TTGIR layouts and num_ctas is known.
Output contract:
  CTA-sensitive ops have a CGA layout that encodes CTA ownership/splitting.
Next pass relies on:
  Layout propagation, RemoveLayoutConversions, MMA/store lowering.
```

### 6. 写出 Invariant

这一节回答“pass 改了很多 IR，但哪些东西绝不能变”。

常见 invariant：

- tensor logical shape 不变，除非 pass 明确做 tiling/packing reshape
- element type 不变，除非 pass 明确做 type conversion
- program semantics 不变
- memory address 的 logical meaning 不变
- `tt.dot` / `tt.load` / `tt.store` 的数学语义不变

记录格式：

```text
Invariant:
- Tensor shape: unchanged
- Element type: unchanged
- Semantics: unchanged
- Changed only: layout / encoding / ownership / scheduling
```

### 7. 解释这段逻辑在做什么

这一节回答“pass 自己在 Triton IR 层面承担什么职责”。

建议写成：

```text
Triton mechanism:
这个 pass 在 TTGIR 层面决定/重写/传播/删除 ...
它操作的对象是 ...
它输出给后续 pass 的 contract 是 ...
```

这部分要避免直接说“CUDA 优化”。很多 pass 的第一身份是 Triton IR mechanism。

### 8. 解释为什么会产生这些变化

这一节回答“为什么 pass 认为这个 IR 应该变成这样”。

常见原因：

- 原 IR 还没有 GPU layout 信息
- 原 layout 不适合 coalesced memory access
- 原 layout 不满足 dot / MMA / WGMMA operand contract
- 原 IR 里有过渡性的 layout conversion
- 原 IR 没有表达 CTA / warp / thread 级别的数据分配
- 原 IR 没有表达 async / pipeline / barrier 依赖
- 原 IR 还不能 lower 到 LLVM / NVVM

要把原因和具体 IR 对上，不要停留在抽象描述。

### 9. 判断它属于哪一层机制

每个 pass 都分两列记：

```text
Triton mechanism:
- 在 Triton IR 里具体改了什么

Optimization / hardware reason:
- 这个变化服务于什么 GPU 执行目标
```

常见分类：

- `Triton IR lowering mechanism`
- `Triton layout management`
- `GPU memory access optimization`
- `Tensor Core / MMA / WGMMA preparation`
- `CTA / warp / thread work partitioning`
- `Latency hiding / software pipeline`
- `Hardware resource allocation`
- `Synchronization correctness`
- `Cleanup / canonicalization`

一个 pass 可能同时属于多类，但记录时要分清主次。

### 10. 细分 GPU / Hardware Reason

不要只写“为了 GPU 优化”。尽量拆成：

```text
Hardware reason:
  来自硬件执行模型的原因，比如 CTA/warp/warp-group/TMEM。

Instruction reason:
  来自具体指令 contract 的原因，比如 mma.sync / wgmma operand layout。

Execution reason:
  来自执行组织的原因，比如多 CTA 协作、warp specialization、latency hiding。

Memory reason:
  来自 memory hierarchy / coalescing / reuse / shared memory / TMA 的原因。
```

不是每个 pass 都有四类原因；没有就写 `not central in this pass`。

### 11. 写出 Decision Tree

把源码逻辑压缩成决策树，而不是复制源码。

例子：

```text
if num_ctas == 1:
  no-op
else if has tt.dot:
  compute M/N/K
  choose splitM/splitN
  create CGALayout
  insert casts
  propagate layout
else if has tt.reduce:
  choose reduce-aware CTA layout
else:
  use store-like op as fallback anchor
```

这能帮助区分“源码细节”和“compiler decision”。

### 12. 写出 Alternative Design

如果这个 pass 做了选择，就问：

```text
Could compiler choose another design?
Why did it not choose that here?
What cost would the alternative have?
```

例子：

- `PlanCTA` 可以 split M，也可以 split N，也可以 split M/N。
- `Coalesce` 可以保留旧 layout，也可以换 layout 并插入 conversion。
- `Pipeline` 可以不提前 load，也可以增加 stage 数。

这一节不需要每次都很长，但至少记录一个有价值的 alternative。

### 13. 写出 Architecture Evolution

对 Ampere / Hopper / Blackwell 对比，不只记录“IR 不同”，还要问：

```text
Hardware changed:
  Ampere -> Hopper -> Blackwell 哪个硬件能力变了？

Compiler changed:
  编译器因此需要表达什么新概念？

Pass changed:
  当前 pass 的 decision / contract 是否因此变化？
```

例子：

```text
Ampere:
  mma.sync path

Hopper:
  wgmma + warp-group + TMA path

Compiler implication:
  IR must express warp-group/tensor-memory/synchronization contracts.

Pass implication:
  Later pipeline, layout, TMEM, and barrier passes become more important.
```

### 14. 回答“没有这个 pass 会怎样”

这是判断 pass 目的最有效的问题。

可以从三个层面回答：

```text
Correctness:
- 没有它会不会无法 lower？
- 会不会缺少同步、layout contract 或硬件资源分配？

Performance:
- 没有它会不会产生更多 layout conversion？
- 会不会访存不 coalesced？
- 会不会不能使用 tensor core / WGMMA？
- 会不会 pipeline/overlap 失败？

Compiler pipeline:
- 后续哪个 pass 会吃到更差或不合法的 IR？
```

## 二、推荐记录模板

复制下面模板，为每个 pass 建一节。

```markdown
## Pass: <PassName>

### Files

- Before: [file](path:line)
- After: [file](path:line)
- Source: [file](path:line)

### Architecture Matrix

| Arch | Before | After | Changed? | Main before feature | Main after feature |
|---|---|---|---|---|---|
| Ampere `sm80/sm86` | ... | ... | yes/no | ... | ... |
| Hopper `sm90` | ... | ... | yes/no | ... | ... |
| Blackwell `sm100` | ... | ... | yes/no/pending | ... | ... |

### Cross-Architecture Before Comparison

- Are the three before IRs already different?
- If yes, what earlier arch-specific mechanism likely caused the divergence?
- If no, does this pass create the first arch-specific divergence?

### One-line Summary

这个 pass 在本例中主要做了：...

### Compiler Decision

- Compiler question: ...
- Decision made here: ...
- Why here in the pipeline: ...

### Compiler Contract

- Input contract: ...
- Output contract: ...
- Next pass relies on: ...

### Invariant

- Tensor shape: ...
- Element type: ...
- Program semantics: ...
- Changed only: ...

### Effective Or No-op

- Result: changed / no-op
- Evidence:
  - Before ...
  - After ...

### IR Changes

#### Cross-Architecture Changes

1. Ampere:
   - Before: ...
   - After: ...
   - Meaning: ...

2. Hopper:
   - Before: ...
   - After: ...
   - Meaning: ...

3. Blackwell:
   - Before: ...
   - After: ...
   - Meaning: ...

#### Single-Architecture Detailed Changes

1. <change category>
   - Before: ...
   - After: ...
   - IR-level meaning: ...

2. <change category>
   - Before: ...
   - After: ...
   - IR-level meaning: ...

### Source Mapping

1. <IR change category>
   - Code: ...
   - Logic: ...
   - Why this creates the observed IR change: ...

2. <architecture-specific branch if any>
   - Code: ...
   - Arch condition: ...
   - Ampere effect: ...
   - Hopper effect: ...
   - Blackwell effect: ...

### Triton Mechanism

...

### GPU / Hardware / Optimization Reason

- Hardware reason: ...
- Instruction reason: ...
- Execution reason: ...
- Memory reason: ...

### Decision Tree

```text
...
```

### Alternative Design

- Alternative: ...
- Why not here: ...

### Architecture Evolution

- Ampere: ...
- Hopper: ...
- Blackwell: ...
- Compiler implication: ...

### Knowledge Card

```text
Pass:
Purpose:
Compiler decision:
Main IR attribute/op:
Input contract:
Output contract:
Invariant:
Hardware reason:
Next dependencies:
```

...

### If This Pass Did Not Exist

...

### Open Questions

- ...
```

## 三、示例：PlanCTA on matmul sm90 num_ctas=2

### Files

- Before: [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:1)
- After: [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:1)
- Source: [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)

### Architecture Matrix

当前 `num_ctas=2` 的完整 before/after 示例只有 Hopper `sm90` dump：

| Arch | Before | After | Changed? | Main before feature | Main after feature |
|---|---|---|---|---|---|
| Ampere `sm80/sm86` | pending, need `TRITON_ARCH=80/86 TRITON_NUM_CTAS=2` | pending | pending | unknown | unknown |
| Hopper `sm90` | [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:1) | [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:1) | yes | dot parent `CGALayout = [[1, 0]]` | dot parent `CGALayout = [[0, 1]]` |
| Blackwell `sm100` | pending, need `TRITON_ARCH=100 TRITON_NUM_CTAS=2` | pending | pending | unknown | unknown |

### Cross-Architecture Before Comparison

这个例子目前还不能完整回答 Ampere/Hopper/Blackwell 三架构之间的 PlanCTA 差异，因为只有 `sm90_num_ctas2` dump。

后续补齐 dump 后，先比较三份 `Before_TritonGPUPlanCTAPass.mlir`：

- 如果三份 before 的 `tt.dot`、`#ttg.dot_op` parent、`#blocked` / `CGALayout` 已经不同，说明 PlanCTA 的输入已经被更早的 arch-specific pass 或 target-dependent layout decision 分叉。
- 如果三份 before 基本相同，而 after 不同，再回到 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212) 查当前 pass 是否有 arch-specific branch。
- 对当前源码片段来说，`processDot` 的 tiling 选择主要依赖 `M/N/K/numCTAs`，没有显式 Ampere/Hopper/Blackwell 分支；所以如果后续看到不同架构 after 不同，优先检查 before 是否已经不同，以及 layout propagation 是否受输入 layout 影响。

### One-line Summary

`PlanCTA` 在这个例子里为 `tt.dot` 选择 CTA 级别的输出切分：`splitM=1, splitN=2`，并把 dot 周围的 layout 传播成对应的 CGA layout。

### Compiler Decision

- Compiler question: 多个 CTA 如何分摊一个 logical output tile？
- Decision made here: 对 `C[M, N]` 选择 `splitM=1, splitN=2`，也就是不切 `M`，沿 `N` 维切成两个 CTA 分片。
- Why here in the pipeline: 这个 pass 位于 TTGIR layout 已经存在、但后续 MMA/store lowering 还没有发生的位置；此时既能读到 `tt.dot` 的 tensor shape 和 layout，又还能通过 layout propagation 改写 producer/consumer 链。

### Compiler Contract

- Input contract:
  - module 上 `ttg.num-ctas` 已知，当前例子是 `2`
  - `tt.dot` 的 A/B operand 已经是 `#ttg.dot_op` encoding
  - dot accumulator/result 已经是 `#ttg.blocked` encoding
- Output contract:
  - CTA-sensitive dot result 有明确的 `CGAEncodingAttr`
  - `tt.dot` 的 A/B operand parent layout 和 accumulator/result layout 对齐到新的 CTA tiling
  - producer/consumer 链中尽量传播同一个 CTA ownership model
- Next pass relies on:
  - `RemoveLayoutConversions` 可以基于更一致的 CTA layout 删除过渡 conversion
  - MMA/WGMMA lowering 可以看到 dot operand/result 的 CTA ownership
  - store lowering 可以知道输出 tile 中每个 CTA 负责哪一部分

### Invariant

- Tensor shape: unchanged，仍然是 A `64x32`、B `32x64`、C `64x64`
- Element type: unchanged，A/B 仍是 `f16`，accumulator 仍是 `f32`
- Program semantics: unchanged，仍然计算同一个 matmul tile
- Changed only: layout / encoding / CTA ownership / layout conversions around dot

### Effective Or No-op

这个 pass 生效了。

证据：

- Before module 已经有 `"ttg.num-ctas" = 2`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:24)
- After 仍是 `"ttg.num-ctas" = 2`，但 layout 和 dot operand/result 已改变，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:20)
- Before 有 `#blocked` 到 `#blocked8` 九个主要 blocked layout，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:2)
- After 只剩 `#blocked` 到 `#blocked4` 五个主要 blocked layout，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)

### IR Changes

#### Cross-Architecture Changes

当前已知：

1. Hopper `sm90, num_ctas=2`
   - Before: dot parent 是 `#blocked8`，`CGALayout = [[1, 0]]`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:10)
   - After: dot parent 是 `#blocked`，`CGALayout = [[0, 1]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)
   - Meaning: PlanCTA 在这个输入上选择 `splitM=1, splitN=2`

2. Ampere `sm80/sm86, num_ctas=2`
   - Pending: 需要生成同样 compile option 的 dump 后补行号。

3. Blackwell `sm100, num_ctas=2`
   - Pending: 需要生成同样 compile option 的 dump 后补行号。

#### Single-Architecture Detailed Changes

#### 1. Dot parent layout 从 M split 变成 N split

Before:

- `#blocked8` 是 dot parent layout，`CGALayout = [[1, 0]]`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:10)
- `tt.dot` 的两个 operand 都使用 `parent = #blocked8`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:79)
- `tt.dot` 结果也是 `tensor<64x64xf32, #blocked8>`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:82)

After:

- 新的主 dot parent layout 是 `#blocked`，`CGALayout = [[0, 1]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)
- `tt.dot` 的两个 operand 都使用 `parent = #blocked`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:66)
- `tt.dot` 结果直接是 `tensor<64x64xf32, #blocked>`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:68)

IR-level meaning:

- 输出 accumulator `C[M, N]` 的 CTA split 被规划成沿 `N` 维，而不是沿 `M` 维。
- 这里 rank-2 tensor 的 `dim0 = M`，`dim1 = N`。

#### 2. Dot 前后的过渡 conversion 减少

Before:

- accumulator 先从 `#blocked` 转成 `#blocked8`，再 dot，最后从 `#blocked8` 转回 `#blocked`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:81)

After:

- loop-carried accumulator 已经是新的 `#blocked`，`tt.dot` 直接返回 `#blocked`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:62)
- `tt.dot` 后没有再转回旧 accumulator layout，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:68)

IR-level meaning:

- `PlanCTA` 不只是改 dot 本身，还把新 CTA tiling 沿 producer/consumer 链传播，减少 dot 周围的 layout mismatch。

#### 3. A load 出现 `CGALayout = [[0, 0]]`

After:

- `#blocked4` 是 `CGALayout = [[0, 0]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:6)
- A pointer 在 load 前从 `#blocked2` 转成 `#blocked4`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:63)

IR-level meaning:

- 当前输出沿 `N` split。
- A operand 是 `A[M, K]`，不含输出 `N` 维。
- 因此不同 CTA 在 `N` split 下需要看到同一块 A 数据，A 的实际 load 可以是 CTA-broadcast/replicated 语义，即 `[[0, 0]]`。

### Source Mapping

#### 1. 为什么选出 `splitM=1, splitN=2`

源码：

- `processDot` 入口见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)
- `getCTATiling` 逻辑见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:214)
- 从 dot type 读取 `M/N/K` 见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:245)
- 调用 `getCTATiling` 见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:249)

当前 IR 的 dot shape：

```text
A: tensor<64x32xf16>
B: tensor<32x64xf16>
D/C: tensor<64x64xf32>
numCTAs = 2
```

代入源码：

```text
M = 64
N = 64
K = 32
numCTAs = 2
```

`getCTATiling` 当前实现：

```text
chunk_m = 128
splitM = clamp(M / chunk_m, 1, numCTAs)
splitN = numCTAs / splitM
要求 N / splitN >= 64
```

第一次：

```text
chunk_m = 128
splitM = clamp(64 / 128, 1, 2) = 1
splitN = 2
N / splitN = 32
```

`32 < 64`，继续。

第二次：

```text
chunk_m = 64
splitM = clamp(64 / 64, 1, 2) = 1
splitN = 2
N / splitN = 32
```

仍不满足。下一轮 `chunk_m = 32` 不合法，循环退出，保留最后一次结果：

```text
splitM = 1
splitN = 2
```

注意：`K` 在当前代码里被读出，但没有参与这个 tiling 选择。

#### 2. 为什么 IR 里出现 `CGALayout = [[0, 1]]`

源码：

- 新 CGA layout 构造见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:258)

代码等价于：

```cpp
fromSplitParams(ctx, {splitM, splitN}, {splitM, splitN}, {1, 0})
```

代入 `splitM=1, splitN=2`：

```text
CTAsPerCGA = [1, 2]
CTASplitNum = [1, 2]
CTAOrder = [1, 0]
```

`CGAEncodingAttr` 的定义说明它描述 block/CTA 如何映射到 tensor 逻辑维度，见 [CGAEncodingAttr.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td:17)。

rank-2 matmul 中：

```text
dim0 = M
dim1 = N
```

所以 `[[0, 1]]` 表示唯一的 CTA split basis 落在 `dim1`，也就是沿 `N` 维切。

#### 3. 为什么 dot operand/result 被改成新 layout

源码：

- new D layout 构造见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:260)
- new A/B dot operand layout 构造见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:263)
- `insertCasts` 把新 layout 插入 dot 周围，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:268)

这直接解释了：

- Before dot parent 是 `#blocked8`
- After dot parent 是新的 `#blocked`
- After dot result 直接返回 `#blocked`

### Triton Mechanism

`PlanCTA` 是 TTGIR 层的 CTA layout planning pass。

它不是把 Python grid 改成多个 block，也不是 CUDA runtime 层面的 launch policy。它做的是：

- 在 TTGIR 的 tensor layout system 里表达多个 CTA 如何协作处理一个 tile
- 为 `tt.dot`、`tt.reduce` 或 store-like op 选择一个 CTA tiling
- 把这个 tiling 变成 `CGAEncodingAttr`
- 通过 layout cast 和 propagation，把 dot 周围 producer/consumer 尽量统一到同一个 CTA layout

### GPU / Hardware / Optimization Reason

这个 pass 服务的是 GPU block/CTA 级别的工作组织。

- Hardware reason: GPU work is organized by CTAs; when `num_ctas > 1`, TTGIR must represent how CTAs in a CGA map to tensor dimensions.
- Instruction reason: not directly selecting `mma.sync` or `wgmma`; it prepares the dot operand/result ownership contract that later MMA/WGMMA lowering consumes.
- Execution reason: multiple CTAs cooperate on one logical tile, so each CTA needs a stable ownership rule for output and related operands.
- Memory reason: A/B/C pointer and load/store layouts need to remain compatible with the chosen CTA split; in this example A becomes CTA-broadcast-like for the `N` split.

在 `num_ctas > 1` 时，Triton 需要在 IR 里明确多个 CTA 如何分摊一个 logical tile，否则后续 lowering 不知道：

- 哪个 CTA 负责输出 tile 的哪一部分
- A/B/C 这些 tensor 的 layout 如何对应这个 CTA split
- dot operand 的 layout contract 应该怎样传播

它是 Triton 编译器自己的 layout planning mechanism，但动机来自 GPU 执行模型中的 CTA/CGA 协作和 tensor-core matmul 的数据组织需求。

### Decision Tree

PlanCTA 的高层决策树可以这样读：

```text
if module num_ctas == 1:
  skip pass
else:
  processDot(func)
    if tt.dot exists:
      read M/N/K from dot types
      choose splitM/splitN by getCTATiling
      create CGALayout from split params
      create new accumulator layout
      create new dot operand layouts
      insert casts around dot
      propagate layout through producer/consumer chain

  processReduce(func)
    if no dot anchored the tiling and reduce exists:
      choose reduce-aware CTA tiling

  if still no tiling anchor:
    processStoreLikeOps(func)
      use store-like op as fallback anchor
```

当前例子走的是 `tt.dot` 分支。

### Alternative Design

- Alternative: `splitM=2, splitN=1`，两个 CTA 沿输出 `M` 维分摊。
- Why not here: 当前 `getCTATiling` 的启发式从较大的 `chunk_m` 开始，先计算 `splitM`，再用剩余 CTA 给 `splitN`；对 `M=N=64, numCTAs=2`，最后得到的是 `splitM=1, splitN=2`。
- Cost/tradeoff: 不同 split 会改变 A/B/C 的 CTA ownership 和 operand reuse 形态。沿 `N` 切时 A 对 CTA split 更像 broadcast，B/C 跟随 `N` 分片；沿 `M` 切则相反。

### Architecture Evolution

当前 `PlanCTA` 的 `processDot` 片段没有显式 Ampere/Hopper/Blackwell 分支；它主要依赖 `M/N/K/numCTAs` 和输入 layout。

- Ampere: 后续通常走 `mma.sync` 路径，PlanCTA 的输出主要服务于 CTA ownership 和 MMA operand/result layout consistency。
- Hopper: 后续可能进入 `wgmma`、warp-group、TMA、TMEM/barrier 更复杂路径，PlanCTA 输出的 CTA ownership 会成为这些后续结构的前置 contract。
- Blackwell: 预计更依赖 Blackwell-specific MMA/TMEM/cluster lowering；需要 `sm100` dump 补齐后判断当前 pass 的 before 是否已由更早架构分支分叉。
- Compiler implication: 如果三个架构的 PlanCTA before 已经不同，先找更早 pass 的 arch-specific decision；如果 before 相同但 after 不同，再检查当前 pass 或 layout propagation 是否有 target-dependent behavior。

### Knowledge Card

```text
Pass: TritonGPUPlanCTAPass
Purpose: assign CTA ownership / CGA layout for CTA-sensitive ops
Compiler decision: choose how multiple CTAs split a logical tile
Main IR attribute/op: CGALayout, #ttg.blocked, #ttg.dot_op parent layout
Input contract: TTGIR layouts exist; num_ctas is known; dot/reduce/store-like anchors are present
Output contract: CTA-sensitive ops carry a consistent CGA layout
Invariant: logical shape, element type, and mathematical semantics unchanged
Hardware reason: CTA/CGA execution and later tensor-core/store lowering need stable ownership
Next dependencies: RemoveLayoutConversions, MMA/WGMMA lowering, store lowering
```

### If This Pass Did Not Exist

在这个例子里，如果没有 `PlanCTA`：

- `num_ctas=2` 的信息不会被系统性地落实到 dot 的 accumulator layout 上
- dot 可能继续使用前面 pass 留下的临时 CGA layout，比如 before 里的 `#blocked8 CGALayout = [[1, 0]]`
- dot 前后会保留更多 layout mismatch 和 `ttg.convert_layout`
- 后续 `RemoveLayoutConversions`、MMA lowering、store lowering 会面对更差的 layout contract
- 多 CTA 协作的语义会更难稳定传递到后续 lowering

最重要的是：我们无法从 IR 中清楚读出“这两个 CTA 到底沿 M 还是 N 分摊输出 tile”。

## 四、后续每个 pass 都要沉淀的内容

后面继续学习时，每个 pass 至少记录这五句话：

```text
1. Before -> After 主要 IR 变化是 ...
2. 这些变化由源码里的 ... 逻辑产生。
3. 这段逻辑在 Triton 机制上是在 ...
4. 它服务的 GPU / hardware / optimization 目标是 ...
5. 如果没有这个 pass，后续会 ...
```

这样看 dump 时不会只停留在“IR 变了”，而是能把 IR、pass 源码、Triton mechanism 和 GPU reason 串起来。
