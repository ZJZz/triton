# Triton TTGIR：Cleanup

本文只回答一个问题：在不重新定义核心 contract 的前提下，哪些中间表示噪音可以删、并、重排。

## 1. 核心定义

```text
cleanup
  = 不重新定义 ownership / carrier / schedule / legality contract，
    而是压缩表示链、消除重复、删掉临时状态、整理 IR 结构
```

这里的重点不是“它改动小不小”，而是：

- 它有没有建立新的 `who computes` contract
- 它有没有建立新的 `what form / what path` contract
- 它有没有建立新的 `when / what ordering` contract
- 它有没有补新的 legality constraint

如果都没有，更可能属于 cleanup。

## 2. 和其他四类的边界

| 类别 | 核心问题 | 典型载体 |
|---|---|---|
| distributed execution mapping | 谁拥有这些 elements | `#ttg.blocked`、`CGAEncodingAttr` |
| layout / data-movement organization | 值以什么 form / carrier 流动 | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMEM |
| target-driven scheduling | 这些工作何时发生、如何 overlap | `loop.stage`、async op、wait、barrier protocol |
| legality repair | 还缺什么约束才能继续 lower | fence、proxy ordering、TMEM reuse barrier |
| cleanup | 哪些中间表示噪音可以删除 | 冗余 convert、重复链、临时 token、死代码 |

最容易混淆的是：

- `cleanup` 不是“只做 trivial DCE”。有些 cleanup pass 会重写大段 IR。
- `cleanup` 也不等于“完全不影响性能”。很多 cleanup 直接影响 register pressure 和后续 codegen 质量。
- 分类时要看 durable effect，不要只看 rewrite 规模。

## 3. `cleanup` 不等于“小修小补”

`RemoveLayoutConversions` 是最典型的边界样本。

它会大幅改写 producer / consumer 链，甚至把 loop-carried tensor type 一起改掉；但它没有发明新的 consumer contract。真正的 contract 还是前面这些 pass 定的：

- `Coalesce` 选 memory-facing form
- `OptimizeDotOperands` 选 dot / MMA operand form
- `OptimizeDescriptorEncoding` 选 descriptor/TMA form

`RemoveLayoutConversions` 做的是：

```text
围绕这些既有 anchor
  压缩中间 representation 链
  只在真实 layout boundary 保留 convert
```

所以在这个五类框架里，把它归到 cleanup 更稳。

## 4. Cleanup 可以分成三组

### 4.1 representation cleanup

这类 pass 主要清理“值已经决定怎么流动了，但中间表示链太吵”。

代表 pass：

- `RemoveLayoutConversions`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)
  - 目标是减少 `ConvertLayoutOp`，并偏向更 favorable 的 load/store 与 tensor-op layout
- `ReduceDataDuplication`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:301)
  - 通过把 `distributed -> dotOperand` 改写成可共享的路径，降低 register duplication

这类 cleanup 的判断标准是：

```text
它在压缩已有 representation chain
不是在决定新的 carrier
```

`RemoveLayoutConversions` 的深入分析见
[2026-06-29-remove-layout-conversions.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-29-remove-layout-conversions.md)。

### 4.2 target-specific state cleanup

这类 pass 主要清掉 target-specific 中间状态，或者把它整理成更适合后续 codegen 的形状。

代表 pass：

- `InterleaveTMem`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:182)
  - 会 sink TMEM loads、hoist TMEM stores，以降低 register pressure
- `RemoveTMEMTokens`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:192)
  - 删除 TMEM dependency token，因为这些 token 在后续已经不再需要

`RemoveTMEMTokens` 的实现也很直接：把 token result/operand 去掉，必要时塞一个会被后续 DCE 掉的 dummy async token，见
[RemoveTMEMTokens.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/RemoveTMEMTokens.cpp:65)。

### 4.3 generic IR cleanup

这类 pass 不关心某个特定 tensor form，而是清理 SSA、循环和通用 IR 冗余。

代表 pass：

- `ReorderInstructions`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:290)
  - 目标是降低 register pressure，并让 LLVM/PTX 指令序更友好
- `LoopAwareCSE`
  - 定义见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/Transforms/Passes.td:71)
  - 会递归消除 loop body 内的公共子表达式，尤其是等价的 iter args
- `Canonicalizer`、`CSE`、`SCCP`、`SymbolDCE`
  - 负责 canonical form、常量传播、死代码与符号清理

`LoopAwareCSE` 的实现不是普通单次 CSE。它会：

- 先跑一次 CSE
- 再做 loop iter args 的递归去重
- 再跑一次 CSE
- 最后跑 `scf.for` canonicalizer

见
[LoopAwareCSE.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/Triton/Transforms/LoopAwareCSE.cpp:132)。

## 5. Cleanup 在主链里是穿插出现的

cleanup 不是“最后一波扫地”。

按当前 `make_ttgir`，它至少分布在这些位置：

```text
PlanCTA
  -> RemoveLayoutConversions
  -> ...
  -> AccelerateMatmul
  -> RemoveLayoutConversions
  -> ...
  -> Blackwell: RemoveTMEMTokens
  -> ...
  -> RemoveLayoutConversions
  -> InterleaveTMem
  -> ReduceDataDuplication
  -> ReorderInstructions
  -> LoopAwareCSE
  -> SymbolDCE
  -> ...
  -> SCCP / CSE / Canonicalizer
```

主链位置见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:269)。

这里列的是 cleanup 自己的落点，不是说中间只会经过 cleanup pass。实际主链里还会穿插
`FenceInsertion` 这类 legality pass；位置关系不能直接当成分类关系。

这说明 cleanup 的职责不是“最后才发生”，而是：

- 在每次大改写后压缩表示链
- 在 target-specific protocol 出现后清掉临时状态
- 在 lowering 前把 IR 整理到更稳定的 canonical form

## 6. 读一个 pass 时怎么判断它是不是 cleanup

先问四件事：

1. 它有没有建立新的 ownership contract？
2. 它有没有决定新的 form / carrier？
3. 它有没有决定新的 stage / overlap / protocol？
4. 它有没有补一个“没有就不合法”的约束？

如果四个答案都偏向“没有”，但它明显在：

- 删冗余 convert
- 合并重复 producer chain
- 去掉临时 token / 无用 iter arg
- 重排指令以降低 register pressure
- 把 IR 推向 canonical form

那就优先按 cleanup 读。

对 TTGIR 来说，`cleanup` 的本质不是“低价值附属 pass”，而是把前面已经选好的 contract 压成更短、更稳、更容易 lower 的 IR。
