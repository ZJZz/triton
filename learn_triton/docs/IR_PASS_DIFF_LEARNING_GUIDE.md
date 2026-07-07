# IR Pass Diff Learning Guide

## 1. 角色和边界

这是一份操作手册。

它只回答一个问题：

```text
拿到 before IR / after IR / pass 源码后，
怎样系统读出一个 pass 在这里到底做了什么决定，
建立了什么 contract，
为什么 pipeline 里需要它，
以及为什么这个样本里会产生这次具体 diff。
```

它不承担 TTGIR 总地图职责：

- 学 Triton backend 全链路，看 [GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)
- 学 TTGIR 的对象、五类框架和六步模板，看 [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md)
- 学五类专题，看：
  - [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md)
  - [LAYOUT_DATA_MOVEMENT_ORGANIZATION.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LAYOUT_DATA_MOVEMENT_ORGANIZATION.md)
  - [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md)
  - [LEGALITY_REPAIR.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/LEGALITY_REPAIR.md)
  - [CLEANUP.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/CLEANUP.md)

## 2. 和 `TTGIR_GUIDE.md` 的分工

两份文档不是重复关系，而是上下位关系：

- `TTGIR_GUIDE`：先给总地图，回答 TTGIR 里有哪些问题域，pass 先怎么归位
- 本文：再给操作步骤，回答具体该怎样读一个 pass 的 before / after / source

可以把分工压成：

```text
TTGIR_GUIDE
  = 先知道“这是什么问题”

IR_PASS_DIFF_LEARNING_GUIDE
  = 再知道“这个问题在当前样本里是怎么被 pass 处理的”
```

如果没有前者，后者很容易变成“只会看 diff”；如果没有后者，前者又容易停留在分类层。

## 3. 最终要产出什么

不要只写“这个 pass 改了哪些 IR”。更有用的产物是：

```text
Input IR
  -> compiler question
  -> compiler decision
  -> output IR
  -> next-pass contract
  -> if absent, what breaks
```

对每个 pass，至少要回答这八件事：

```text
Class:
  它主要属于五类里的哪一类？

Boundary:
  input / output / non-goal 是什么？

Problem:
  进入这个 pass 时，IR 还缺什么事实，或者还违反什么约束？

Decision:
  编译器在这里实际做了什么决定？

Contract:
  after IR 建立了什么新事实，后续 pass 可以依赖什么？

Invariant:
  它改了很多 IR，但哪些语义绝不能变？

Deferred work:
  它刻意没有解决什么，留给了后续哪个 pass？

If absent:
  没有它，会出现 correctness、performance 还是 pipeline 问题？
```

## 4. 先决定分析范围

默认先做单架构分析，不要一上来强行铺 Ampere / Hopper / Blackwell 三张表。

### 4.1 单架构是默认

这些情况，单架构通常就够：

- pass 没有显式 target / capability 分支
- before IR 没有明显 target-specific 分叉
- 当前只是想先搞清一个 pass 的基本职责
- pass 本身更像 cleanup / canonicalization / local rewrite

### 4.2 三架构对比是按需加的

这些情况，再补 cross-architecture analysis：

- pass 所在 pipeline 本身按 capability 分叉
- before IR 在不同架构上已经明显不同
- after IR 在不同架构上出现不同结构
- pass 的职责明显和 `wgmma`、`tcgen05`、TMA、TMEM、warp specialization、barrier protocol 绑定
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

## 5. 执行流程

本文展开的是 [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md) 里的六步模板：

```text
先定类
  -> 再定边界
  -> 再问为什么需要它存在
  -> 再看相关 IR diff
  -> 再回答这个样本里为什么这样改
  -> 最后看 after IR 被谁消费
```

### 5.1 进入六步前：先建最小事实表

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

这个表的作用只是防止后面分析脱离样本上下文。

### 5.2 第一步：定类

先判断它主要属于五类里的哪一类：

- distributed execution mapping
- layout / data-movement organization
- target-driven scheduling
- legality repair
- cleanup

判断时先问它主要在回答哪种 compiler question：

| 类别 | 核心问题 |
|---|---|
| mapping | 谁拥有这些 elements |
| organization | 值以什么 form / carrier 流动 |
| scheduling | 这些工作何时发生、如何 overlap |
| legality | 还缺什么约束才能继续 lower |
| cleanup | 哪些中间表示噪音可以删除 |

注意两点：

- 先选主类，再承认它可能有次要效应
- 不要按“改动看起来像什么”分，要按“它主要解决什么问题”分

最常见的误读就是：

- 把 `RemoveLayoutConversions` 误读成 organization
- 把 `FenceInsertion` 误读成 scheduling
- 把主链里的位置关系误读成分类关系

### 5.3 第二步：定抽象边界

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

这里最好顺手再补一句：

```text
deferred work
  = 它刻意留给后续 pass 的事
```

因为很多 pass 不是真的“做不完”，而是有意把职责压在某个抽象边界之内。

### 5.4 第三步：为什么 pipeline 里需要它存在

这一步回答的是设计问题，不是样本问题。

固定从这五个角度写：

```text
Problem:
  当前 IR 缺了什么事实，或者为什么当前 contract 还不够？

Goal:
  这个 pass 想建立什么 compiler / hardware / performance 目标？

Constraint:
  哪些硬件、指令、layout legality、ownership、scheduling 约束限制了可选方案？

Design intent:
  为什么当前机制适合这个问题，而不是别的机制？

If absent:
  没有它，会出现 correctness、performance 还是 pipeline 问题？
```

这里要特别注意：

- 不要把 `#blocked`、`CGAEncodingAttr`、`ttg.convert_layout`、`loop.stage` 当成问题本身
- 它们只是编译器答案的载体

`Problem` 和 `Constraint` 经常引用同一个硬件事实，但视角不同：

- `Problem`：因此 before IR 还缺了什么
- `Constraint`：因此当前 pass 不能随便选一个更简单方案

### 5.5 第四步：看和该类直接相关的 IR diff

这一步不要泛看全部变化，而是只看和它那一类问题直接相关的信号。

| 类别 | 看 diff 时先盯什么 |
|---|---|
| mapping | encoding、ownership、`#ttg.blocked`、`CGAEncodingAttr`、`#ttg.slice` |
| organization | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMA、TMEM |
| scheduling | `loop.stage`、async op、wait、barrier protocol |
| legality | fence、proxy ordering、TMEM reuse barrier |
| cleanup | 冗余链、死代码、token、canonical form |

记录 IR 事实时，建议先按这些栏目扫一遍：

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

这一步先写事实，不急着写动机。

### 5.6 第五步：为什么这个样本里这样改

这一步回答的是实例问题。

更准确地说，它是在回答：

```text
当前 IR pattern
  + pass analysis / heuristic / matching rule
  -> 为什么产生了这次具体 diff
```

这里要把 `IR change` 映射到具体源码路径，而不是只写“这是某 pass 做的”。

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

重点找这些东西：

- pass 内部用了哪条匹配路径
- 哪个 branch / heuristic / gate 被触发
- 哪些变量决定了这次具体选择
- 为什么这个样本是 `changed`，或者为什么 `no-op` 也是合理结果

很多“我看懂 diff 了，但还没看懂这个 pass 为啥存在”的情况，就是把这一步误当成了上一步。

### 5.7 第六步：看 downstream consumer

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

这一步最好写成：

```text
Output contract:
  after IR 新建立了什么事实？

Next pass relies on:
  哪些后续 pass 会消费这个 contract？
```

不补这一问，很容易把 pass 读成“只是改了一下 IR 形状”。

## 6. 最小模板

下面模板是默认版本。

- `Cross-Architecture`、`Decision Tree`、`Alternative Design` 都是按需加，不是每次强制写。
- 对 `no-op` pass，重点放在“为什么 no-op 也是合理结果”。

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

### Class

- Primary class: mapping / organization / scheduling / legality / cleanup
- Why this class: ...

### Boundary

- Input contract: ...
- Output contract: ...
- Non-goal: ...
- Deferred work: ...

### Why This Pass Exists

- Problem: ...
- Goal: ...
- Constraint: ...
- Design intent: ...
- If absent: ...

### Relevant IR Diff

1. <change category>
   - Before: ...
   - After: ...
   - IR-level meaning: ...

2. <change category>
   - Before: ...
   - After: ...
   - IR-level meaning: ...

### Why This Sample Changed

1. <IR change category>
   - Code: ...
   - Function / branch: ...
   - Logic: ...
   - Why this creates the observed change: ...

### Downstream Consumer

- Next pass relies on: ...
- Why this contract matters downstream: ...

### Invariant

- ...

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

## 7. 短示例：PlanCTA on `sm90_num_ctas2`

这个例子只演示“怎么写方法”，不试图替代专题文档。

### Files

- Before: [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:1)
- After: [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:1)
- Source: [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)

### Scope

- Kernel / arch / option: `matmul_kernel`, `sm90`, `ttg.num-ctas = 2`
- Result: `changed`

### One-line Summary

`PlanCTA` 在这个样本里为 `tt.dot` 选择 `splitM=1, splitN=2`，并把新的 CTA ownership 传播到 dot 周围的 layout。

### Class

- Primary class: `distributed execution mapping`
- Why this class:
  它主要在回答“多个 CTA 如何分摊一个 logical output tile”，不是先回答 memory form 或 scheduling。

### Boundary

- Input contract:
  `ttg.num-ctas` 已知，dot operand / result 已经是 TTGIR distributed layout。
- Output contract:
  CTA-sensitive dot result 和相关 operand 挂上了相互一致的 CGA / CTA ownership。
- Non-goal:
  不负责最终删除所有 layout mismatch，也不负责 scheduling / async protocol。
- Deferred work:
  新 ownership 传播后的冗余 convert 交给 `RemoveLayoutConversions`，更晚的 compute lowering 交给 MMA/WGMMA lowering。

### Why This Pass Exists

- Problem:
  当前 module 已知 `ttg.num-ctas = 2`，但 dot 周围的 layout 还没有稳定表达“这两个 CTA 到底沿哪一维切 output tile”。
- Goal:
  为 CTA-sensitive op 建立明确的 CGA / CTA ownership contract。
- Constraint:
  dot operand、accumulator、load/store layout 必须和同一份 CTA split 兼容，后续 MMA/WGMMA lowering 也要能消费它。
- Design intent:
  在 TTGIR layout system 内完成 CTA split 选择和传播，而不是等到更晚的 lowering 才临时决定。
- If absent:
  多 CTA ownership 不够显式，后续 cleanup、MMA/WGMMA lowering、store lowering 会吃到更差的 CTA contract。

### Relevant IR Diff

1. `tt.dot` parent layout 改变
   - Before: `#blocked8` 带 `CGALayout = [[1, 0]]`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:10)
   - After: `#blocked` 带 `CGALayout = [[0, 1]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)
   - IR-level meaning:
     output accumulator 的 CTA split 从沿 `M` 维读，变成沿 `N` 维读。

2. loop-carried accumulator 的 ownership 被统一
   - Before: accumulator 先转到 `#blocked8`，dot 后再转回 `#blocked`，见 [026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:81)
   - After: loop-carried accumulator 直接使用新的 `#blocked`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:62)
   - IR-level meaning:
     新 CTA tiling 被沿 producer / consumer 链传播，不只是改了一个 `tt.dot`。

3. A load 变成 CTA-broadcast-like form
   - After: `#blocked4` 带 `CGALayout = [[0, 0]]`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:6)
   - After: A pointer 在 load 前转成 `#blocked4`，见 [027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:63)
   - IR-level meaning:
     输出沿 `N` split 后，A operand 不含 `N` 维，所以不同 CTA 可以共享同一块 A 数据视图。

### Why This Sample Changed

1. `splitM=1, splitN=2` 的选择
   - Code: `processDot` 和 `getCTATiling`，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)
   - Function / branch:
     从 dot type 读取 `M/N/K`，结合 `numCTAs` 选择 split 参数，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:245)
   - Logic:
     当前样本里两个 CTA 更适合沿 `N` 维分摊 output tile。
   - Why this creates the observed change:
     这是 after IR 中 `CGALayout = [[0, 1]]` 的直接来源。

2. 新 CGA layout 和 dot layout 的物化
   - Code: `fromSplitParams` 构造新 CGA layout，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:258)
   - Code: 新 D / A / B layout 构造和 `insertCasts`，见 [PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:260)
   - Logic:
     把“CTA split decision”传播到 dot result、operand 和周围 producer / consumer。
   - Why this creates the observed change:
     这解释了 after IR 为什么既改了 `tt.dot` parent，也改了 accumulator 和 A load form。

### Downstream Consumer

- Next pass relies on:
  `RemoveLayoutConversions`、MMA/WGMMA lowering、store lowering 都会消费这份 contract。
- Why this contract matters downstream:
  如果没有这份稳定的 CTA ownership，后续 pass 只能在更混乱的 layout 关系上继续工作。

### Invariant

- Tensor shape 不变
- Element type 不变
- Matmul 数学语义不变
- 改变的是 layout / encoding / CTA ownership 及其传播

### Cross-Architecture (Optional)

这个例子不默认展开三架构矩阵，因为这里的目的只是说明方法，不是比较 target divergence。

如果后续要回答“PlanCTA 在 Ampere / Hopper / Blackwell 上有没有 target-specific 行为”，再补做：

- `sm86_num_ctas2` before / after
- `sm90_num_ctas2` before / after
- `sm100_num_ctas2` before / after

然后先比较三份 before 是否已经分叉，再判断差异是不是由当前 pass 产生。

## 8. 什么时候需要额外加码

不是每个 pass 都值得写满所有栏目。

### 8.1 这些情况再加 `Decision Tree`

- pass 内部有明显分支逻辑
- 你需要压缩源码为一张决策图
- 同一 pass 会对 dot / reduce / store-like 走不同路径

### 8.2 这些情况再加 `Alternative Design`

- pass 在多个合法方案里做了选择
- 你真正关心“为什么不是另一种 split / layout / stage count”

### 8.3 这些情况一定要做 cross-arch

- pass 名字、pipeline 位置或 after IR 明显带 target-sensitive 痕迹
- 你已经看到 Hopper / Blackwell-specific op、layout、barrier、TMEM、warp specialization 结构
- 用户问题本身就是“同一个 pass 在不同架构上为什么不同”

## 9. 最后检查

写完一个 pass，至少要能用这六句话收尾：

```text
1. 这个 pass 主要属于五类里的 ...
2. 它的 input / output / non-goal 是 ...
3. pipeline 里需要它，是因为 ...
4. 当前样本里最重要的 diff 信号是 ...
5. 这些 diff 由源码里的 ... 逻辑产生。
6. after IR 建立的 contract 会被 ... 继续消费。
```

如果这六句话还说不顺，就说明还停留在“看到了 diff”，还没真正读懂这个 pass。
