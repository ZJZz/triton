# IR Pass Diff Learning Guide

## 1. 角色和边界

这是一份方法论文档。

它只回答一个问题：

```text
如何通过 before IR / after IR / pass 源码，
读出一个 compiler pass 在这里到底做了什么决定，
建立了什么 contract，
以及为什么必须这样做。
```

它不承担主题总览职责：

- 学 Triton backend 全链路，看 [GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)
- 学 TTGIR 的对象、边界和三问框架，看 [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)
- 学 TTGIR 三个专题，看：
  - [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md)
  - [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)
  - [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)

## 2. 最终要产出什么

不要只写“这个 pass 改了哪些 IR”。更有用的产物是：

```text
Input IR
  -> compiler question
  -> compiler decision
  -> output IR
  -> hardware / execution motivation
  -> next-pass contract
```

对每个 pass，至少要回答这六件事：

```text
Problem:
  进入这个 pass 时，IR 还缺什么事实，或者还违反什么约束？

Decision:
  编译器在这里必须做什么决定？

Contract:
  after IR 建立了什么新事实，后续 pass 可以依赖什么？

Invariant:
  这个 pass 改了很多 IR，但哪些语义绝不能变？

Deferred work:
  这个 pass 刻意没有解决什么，留给了后续哪个 pass？

If absent:
  没有这个 pass，会出现 correctness、performance 还是 pipeline 问题？
```

## 3. 先决定分析范围

默认先做单架构分析，不要一上来强行铺 Ampere / Hopper / Blackwell 三张表。

### 3.1 单架构是默认

这些情况，单架构通常就够：

- pass 没有显式 target / capability 分支
- before IR 没有明显 target-specific 分叉
- 当前只是想先搞清一个 pass 的基本职责
- pass 本身更像 cleanup / canonicalization / local rewrite

### 3.2 三架构对比是按需加的

这些情况，再补 cross-architecture analysis：

- pass 所在 pipeline 本身按 capability 分叉
- before IR 在不同架构上已经明显不同
- after IR 在不同架构上出现不同结构
- pass 的职责明显和 `mma.sync` / `wgmma` / `tcgen05`、TMA、TMEM、warp specialization、barrier protocol 绑定
- 你要回答的问题本来就是“同一个 pass 在三代架构上为什么不同”

判断原则：

```text
Before 就不同：
  先归因到更早的 target-specific decision。

Before 基本相同，After 不同：
  再重点查当前 pass 是否自己做了 target-specific decision。

Before / After 都近似：
  当前 pass 更可能是 target-generic mechanism。
```

## 4. 固定流程

### 4.1 先建最小事实表

先记录：

- pass 名字
- before dump
- after dump
- 源码文件
- 当前 kernel / arch / compile option
- `changed` 还是 `no-op`

例子：

```text
Pass: TritonGPUPlanCTAPass
Before: learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir
After:  learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir
Source: lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp
Kernel: matmul_kernel
Arch: sm90
Option: ttg.num-ctas = 2
Effective: changed
```

### 4.2 先写 IR 事实，不急着解释

先只描述“看到了什么变化”，不要抢先写动机。

建议分类：

- module attrs
- layout / encoding 定义
- tensor type
- `ttg.convert_layout`
- 关键 op：`tt.dot`、`tt.load`、`tt.store`、`scf.for`
- control flow
- sync / async op
- 后续 pass 会消费的 contract carrier

格式：

```text
IR change:
- Before line X: ...
- After line Y: ...
- IR-level meaning: ...
```

### 4.3 再写 compiler question

这是最关键的一步。

不要把 `#blocked`、`CGALayout`、`ttg.convert_layout` 当成问题本身。它们只是答案的载体。

对 TTGIR pass，优先把问题归到这三类之一：

- 谁拥有这些元素
- 这些值以什么 form / carrier 流动
- 这些工作何时发生、如何 overlap、何时同步

这三类问题的展开，分别见：

- [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md)
- [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)
- [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)

例子：

```text
ConvertTritonToTritonGPU:
  logical tensor 如何映射到 GPU threads / warps / CTAs？

Coalesce:
  已经知道谁来访问后，哪些 memory-facing layout 更适合 coalesced access？

PlanCTA:
  多个 CTA 如何分摊一个 logical output tile？

Pipeline:
  load / compute 应该怎样重排，才能把 latency 隐藏掉？
```

### 4.4 再写 Problem / Goal / Constraint / Design Intent / Decision

推荐固定顺序：

```text
Problem:
  当前 IR 缺了什么事实，或者为什么当前 contract 还不够？

Goal:
  这个 pass 想建立什么 compiler / hardware / performance 目标？

Constraint:
  哪些硬件、指令、layout legality、ownership、scheduling 约束限制了可选方案？

Design intent:
  为什么当前机制适合这个问题，而不是别的机制？

Decision:
  当前 pass 在这里实际做了什么决定？
```

`Problem` 和 `Constraint` 经常引用同一个硬件事实，但视角不同：

- `Problem`：因此 before IR 还缺了什么
- `Constraint`：因此当前 pass 不能随便选一个更简单方案

### 4.5 把 IR 变化映射到源码逻辑

对每类 IR 变化，找到对应的函数、分支、关键变量，而不是只写“这是某 pass 做的”。

格式：

```text
Code path:
- Source line X: ...
- Function / branch: ...
- This code creates / replaces / removes ...

IR evidence:
- Before line ...
- After line ...
```

### 4.6 写 contract 和 invariant

每个重要 pass 都要明确这两组内容。

```text
Input contract:
  这个 pass 假设 before IR 已经满足什么？

Output contract:
  after IR 新建立了什么事实？

Next pass relies on:
  哪些后续 pass 会消费这个 contract？

Invariant:
  shape / element type / semantics / ownership meaning / ordering meaning 中哪些不能变？
```

### 4.7 最后再问“没有它会怎样”

这是判断 pass 目的最稳的问题。

从三层看：

```text
Correctness:
  没有它会不会无法 lower，或缺少 legality / sync / resource contract？

Performance:
  没有它会不会多出 layout conversion、访存不 coalesced、用不上 tensor core、
  或 pipeline overlap 失败？

Compiler pipeline:
  后续哪个 pass 会吃到更差或不合法的 IR？
```

## 5. 最小模板

下面模板是默认版本。

- `Cross-Architecture`、`Decision Tree`、`Alternative Design` 都是按需加，不是每次强制写。
- 对 no-op pass，重点放在“为什么 no-op 也是合理结果”。

~~~markdown
## Pass: <PassName>

### Files

- Before: [file](path:line)
- After: [file](path:line)
- Source: [file](path:line)

### Scope

- Kernel / arch / option: ...
- Result: changed / no-op

### One-line Summary

这个 pass 在本例中主要做了：...

### Compiler Question

- ...

### Problem / Goal / Constraint / Design Intent / Decision

- Problem: ...
- Goal: ...
- Constraint: ...
- Design intent: ...
- Decision: ...

### IR Changes

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
   - Why this creates the observed change: ...

### Contract

- Input contract: ...
- Output contract: ...
- Next pass relies on: ...

### Invariant

- ...

### If This Pass Did Not Exist

- Correctness: ...
- Performance: ...
- Compiler pipeline: ...

### Cross-Architecture (Optional)

- Why needed: ...
- Ampere before/after: ...
- Hopper before/after: ...
- Blackwell before/after: ...
- Conclusion: ...

### Decision Tree (Optional)

```text
...
```

### Alternative Design (Optional)

- Alternative: ...
- Why not here: ...
~~~

## 6. 短示例：PlanCTA on `sm90_num_ctas2`

这个例子只演示“怎么写方法”，不试图替代单独的专题文档。

### Files

- Before: [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:1)
- After: [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:1)
- Source: [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)

### Scope

- Kernel / arch / option: `matmul_kernel`, `sm90`, `ttg.num-ctas = 2`
- Result: `changed`

### One-line Summary

`PlanCTA` 在这个样本里为 `tt.dot` 选择 `splitM=1, splitN=2`，并把新的 CTA ownership 传播到 dot 周围的 layout。

### Compiler Question

- 多个 CTA 如何分摊一个 logical output tile？

### Problem / Goal / Constraint / Design Intent / Decision

- Problem:
  当前 module 已知 `ttg.num-ctas = 2`，但 dot 周围的 layout 还没有稳定表达“这两个 CTA 到底沿哪一维切 output tile”。
- Goal:
  为 CTA-sensitive op 建立明确的 CGA / CTA ownership contract。
- Constraint:
  dot operand、accumulator、load/store layout 必须和同一份 CTA split 兼容，后续 MMA/WGMMA lowering 也要能消费它。
- Design intent:
  在 TTGIR layout system 内完成 CTA split 选择和传播，而不是等到更晚的 lowering 才临时决定。
- Decision:
  选择 `splitM=1, splitN=2`，把 output tile 沿 `N` 维拆给两个 CTA。

### IR Changes

1. `tt.dot` parent layout 改变
   - Before: `#blocked8` 带 `CGALayout = [[1, 0]]`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:10)
   - After: `#blocked` 带 `CGALayout = [[0, 1]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)
   - IR-level meaning: output accumulator 的 CTA split 从沿 `M` 维读，变成沿 `N` 维读。

2. dot 前后的 layout mismatch 减少
   - Before: accumulator 先转到 `#blocked8`，dot 后再转回 `#blocked`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:81)
   - After: loop-carried accumulator 直接使用新的 `#blocked`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:62)
   - IR-level meaning: 新 CTA tiling 被沿 producer / consumer 链传播，不只是改了一个 `tt.dot`。

3. A load 变成 CTA-broadcast-like form
   - After: `#blocked4` 带 `CGALayout = [[0, 0]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:6)
   - After: A pointer 在 load 前转成 `#blocked4`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:63)
   - IR-level meaning: 输出沿 `N` split 后，A operand 不含 `N` 维，所以不同 CTA 可以共享同一块 A 数据视图。

### Source Mapping

1. `splitM=1, splitN=2` 的选择
   - Code: `processDot` 和 `getCTATiling`，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)
   - Logic: 从 dot type 读取 `M/N/K`，结合 `numCTAs` 选择 split 参数，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:245)
   - Why: 这是当前 pass 对“output tile 该怎么分 CTA”给出的核心 decision。

2. 新 CGA layout 和 dot layout 的物化
   - Code: `fromSplitParams` 构造新 CGA layout，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:258)
   - Code: 新 D / A / B layout 构造和 `insertCasts`，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:260)
   - Why: 这一步把“CTA split decision”变成 after IR 真正携带的 contract。

### Contract

- Input contract:
  `ttg.num-ctas` 已知，dot operand / result 已经是 TTGIR layout。
- Output contract:
  CTA-sensitive dot result 和相关 operand 挂上了相互一致的 CGA / CTA ownership。
- Next pass relies on:
  `RemoveLayoutConversions`、MMA/WGMMA lowering、store lowering 都会消费这份 contract。

### Invariant

- Tensor shape 不变
- Element type 不变
- Matmul 数学语义不变
- 改变的是 layout / encoding / CTA ownership 及其传播

### If This Pass Did Not Exist

- Correctness:
  多 CTA 如何分摊 output tile 仍然不够显式，后续 lowering 更难稳定理解 CTA ownership。
- Performance:
  dot 周围会保留更多 layout mismatch 和 `ttg.convert_layout`。
- Compiler pipeline:
  后续 layout cleanup、MMA/WGMMA lowering、store lowering 会吃到更差的 CTA contract。

### Cross-Architecture (Optional)

这个样本不默认展开三架构矩阵，因为这里的目的只是说明方法，不是比较 target divergence。

如果后续要回答“PlanCTA 在 Ampere / Hopper / Blackwell 上有没有 target-specific 行为”，再补做：

- `sm86_num_ctas2` before / after
- `sm90_num_ctas2` before / after
- `sm100_num_ctas2` before / after

然后先比较三份 before 是否已经分叉，再判断差异是不是由当前 pass 产生。

## 7. 什么时候需要额外加码

不是每个 pass 都值得写满所有栏目。

### 7.1 这些情况再加 `Decision Tree`

- pass 内部有明显分支逻辑
- 你需要压缩源码为一张决策图
- 同一 pass 会对 dot / reduce / store-like 走不同路径

### 7.2 这些情况再加 `Alternative Design`

- pass 在多个合法方案里做了选择
- 你真正关心“为什么不是另一种 split / layout / stage count”

### 7.3 这些情况一定要做 cross-arch

- pass 名字、pipeline 位置或 after IR 明显带 target-sensitive 痕迹
- 你已经看到 Hopper / Blackwell-specific op、layout、barrier、TMEM、warp specialization 结构
- 用户问题本身就是“同一个 pass 在不同架构上为什么不同”

## 8. 最后检查

写完一个 pass，至少要能用五句话收尾：

```text
1. Before -> After 最主要的 IR 变化是 ...
2. 这些变化由源码里的 ... 逻辑产生。
3. 这段逻辑回答的 compiler question 是 ...
4. 它建立的 after-IR contract 是 ...
5. 如果没有这个 pass，后续会 ...
```

如果这五句话还说不顺，就说明还停留在“看到了 diff”，还没真正读懂这个 pass。
