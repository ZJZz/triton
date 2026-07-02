# Triton TTGIR 中的 Distributed Execution Mapping

目标：把
[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:60)
里这句

```text
TTGIR = distributed execution mapping
      + layout / data-movement organization
      + target-driven scheduling
```

真正落成一条可验证的源码阅读路径。

这篇文档回答的问题不是“有哪些 layout 名字”，而是：

```text
一个 logical tensor，
到底是如何被切给 CTA / warp / thread / per-thread values 的？

这些 ownership 决策由哪些 pass 建立，
又通过哪些 type encoding 留在 TTGIR 里，供后续 pass 消费？
```

如果你想看 “when should the work happen”，请看
[TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md:1)。
本文只看 “who computes / owns these tensor elements”。

---

## 1. 一句话结论

```text
distributed execution mapping
  = 编译器把 logical tensor 切分到 GPU 的 CTA / warp / thread / value 层级上，
    并把这种 ownership 编码进 tensor type，
    让后续的 coalescing、dot lowering、multi-CTA planning、LLVM lowering
    都能基于同一份分工契约继续工作。
```

这里最重要的点不是“有个 layout attr”，而是：

```text
layout attr 在 TTGIR 里经常就是 execution ownership contract 的承载体
```

---

## 2. 它不是什么

### 2.1 不是“memory layout 的别名”

`#blocked` 当然会影响 memory access pattern，但它先回答的是：

```text
哪些元素归哪个 thread / warp / CTA
```

源码里 `DistributedEncodingTrait` 直接把这个层级写死为：

```text
CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread
```

见
[TritonGPUAttrInterfaces.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td:41)。

### 2.2 不是“target-driven scheduling”

distributed execution mapping 回答的是 **who**；
target-driven scheduling 回答的是 **when**。

两者当然会相互影响，但职责不同：

- mapping 决定谁持有数据、谁发起 load/store、谁参与 dot
- scheduling 决定这些工作在循环里何时发生、是否 overlap、是否需要 wait/barrier

### 2.3 不是“LLVM lowering 之后才决定”

TTGIR 层已经把 ownership 写进 type 了。`ConvertTritonToTritonGPU` 的 pass 定义就说得很清楚：
它会给 tensor type 增加 layout encoding，而这些 encoding 一般包含
`numWarps`、`threadsPerWarp` 和 `numCTAs`，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)。

---

## 3. 为什么 TTGIR 必须先回答这个问题

TTIR 里的 tensor 仍然是 logical tensor。
例如一个 `tensor<1024xf32>`，在 TTIR 里你只知道它有 1024 个元素，
并不知道：

- 每个 thread 拿几个元素
- 一个 warp 沿哪一维展开
- 一个 CTA 覆盖多大 tile
- 多个 CTA 是否在协作切同一个 output tile

但 GPU 后端必须尽早知道这些事，因为后续很多决策都依赖它：

- global load/store 是否 coalesced
- `tt.make_range` / `tt.expand_dims` 应该生成什么 ownership
- `tt.dot` 的 operand/result 应该对齐到什么 parent layout
- `num_ctas > 1` 时 output tile 沿哪个维度切
- LLVM lowering 时每个 thread 该算哪些地址

所以 TTGIR 不是“把 tensor 换个长相”，而是把 distributed execution model 显式化。

---

## 4. distributed execution mapping 的主要承载体

### 4.1 module attrs: execution envelope

TTGIR module 顶层会挂这些 attr：

- `ttg.num-warps`
- `ttg.num-ctas`
- `ttg.threads-per-warp`
- `ttg.target`

定义见
[Dialect.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h:49)。

这些 attr 还不是每个 tensor 的 ownership，但它们给出了所有 distributed layout 的全局边界条件。

### 4.2 `#blocked`: 默认、最常见、也是最重要的 ownership contract

`BlockedEncodingAttr` 的官方描述很直接：

- 每个 warp 拥有目标 tensor 的一个 contiguous portion
- 它由 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA` 三组量描述
- 这三组量分别对应 thread、warp、CTA 级别持有的元素组织

见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738)。

对阅读 TTGIR 更实用的心智模型是：

```text
#blocked
  = values per thread
  + threads per warp
  + warps per CTA
  + order
  + optional CGA layout
```

它不是“普通 tensor + 一个 tag”，而是 ownership 的压缩表示。

### 4.3 `CGAEncodingAttr`: CTA 级切分单独编码

多 CTA 情况下，`CGAEncodingAttr` 单独负责：

```text
blocks (CTAs) in a cooperative thread array
如何映射到 logical tensor dimensions
```

定义见
[CGAEncodingAttr.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td:14)。

所以看到：

```text
CGALayout = [[0, 1]]
```

不要只把它当成一个附属字段。它通常就是“output tile 沿哪个 logical dim 被多个 CTA 分摊”的答案。

### 4.4 `#ttg.slice`: parent layout 的投影，不是新的独立分工

dump 里经常看到：

```text
#ttg.slice<{dim = 1, parent = #blocked}>
```

这类 encoding 很容易误读成“又来了一种 layout”。
更准确的理解是：

```text
对 parent distributed layout 做降维投影，
方便 `tt.make_range` / `tt.expand_dims` 一类 1D helper 值继续保持同一套 ownership 语义
```

`SliceEncodingAttr` 的描述也明确说它常用于构造 `expand_dims` 的逆布局，见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1408)。

### 4.5 `#ttg.dot_op`: 不是独立世界，而是挂在 result parent 上的 compute view

`DotOperandEncodingAttr` 明确写着：

- 它记录 `opIdx` / `parent` / `kWidth`
- `parent` 字段就是 dot result 的 layout

见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)。

这件事很关键，因为它说明：

```text
dot operand 的分布不是孤立决定的，
而是依附于 accumulator / result ownership contract
```

---

## 5. 源码里这份 mapping 是怎么造出来的

先看 NVIDIA backend 的 TTGIR 主干顺序：

```text
ConvertTritonToTritonGPU
  -> Coalesce
  -> PlanCTA
  -> ...
```

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:269)。

### 5.1 `ConvertTritonToTritonGPU`: 先把“没有 ownership”的 tensor 变成 distributed tensor

`TritonGPUTypeConverter` 对没有 encoding 的 tensor type 直接调用
`getDefaultBlockedEncoding(...)`，然后 `cloneWithEncoding`，见
[TritonGPUConversion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:19)。

这一步建立的 contract 很简单但非常关键：

```text
从这里开始，tensor 不再只是 logical shape，
而是 logical shape + distributed ownership
```

同一个文件里，conversion target 还要求：

- `tt.dot` 的 A/B operand 必须已经带 `DotOperandEncodingAttr`
- function 的 tensor argument/result 也必须有 encoding

见
[TritonGPUConversion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:80)。

这说明 distributed execution mapping 不是“优化建议”，而是 TTGIR legality contract。

### 5.2 `BlockedEncodingAttr` builder: 先定 CTA split，再定 warp/thread decomposition

`BlockedEncodingAttr` 的 builder 不是简单地把参数打印出来。
它内部会：

1. 根据 `shape / sizePerThread / order / numCTAs` 推出 `CTAsPerCGA` 和 `CTASplitNum`
2. 由此得到 `CGALayout`
3. 再基于 `shapePerCTA` 计算 `threadsPerWarp` 和 `warpsPerCTA`

实现见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:831)。

所以更贴近源码的思路是：

```text
shape
  + sizePerThread
  + order
  + numWarps / threadsPerWarp / numCTAs
    -> CTA-level split
    -> shape per CTA
    -> warp/thread decomposition
    -> final #blocked ownership
```

### 5.3 `Coalesce`: 不改 tensor 含义，但会改 memory-facing ownership

`TritonGPUCoalesce` 的 pass 定义说得很清楚：

- 它分析 `tensor<tt.ptr<...>>` 的 load/store
- 为这些 op 选 coalesced layout
- 在前后插 `ttg.convert_layout` 保持程序其余部分不变

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:235)。

源码里它会对每个 memory op：

- 取 pointer tensor 的现有 `CGAEncoding`
- 算 `shapePerCTA`
- 调 `buildCoalescedEncoding(...)`

见
[Coalesce.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:77)。

`buildCoalescedEncoding(...)` 则根据：

- pointer 的 contiguity
- 同 shape / 同 order memory slice 的共同需求
- `shapePerCTA`
- 每线程最多处理多少元素

来生成新的 `BlockedEncodingAttr`，见
[CoalesceUtils.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:16)。

所以 Coalesce 的本质不是“插两个 convert_layout”，而是：

```text
在不改逻辑 tensor 含义的前提下，
把 memory access 那一段的 ownership 调整成更适合 cache / vectorization 的版本
```

### 5.4 `PlanCTA`: 当 `num_ctas > 1` 时，重新规划 CTA ownership

`PlanCTA` 是 TTGIR 里最典型的 “distributed execution mapping still evolves after conversion” 例子。

它做的不是同步或 pipelining，而是：

- 看 dot/reduce/store-like pattern
- 选 `splitM / splitN`
- 重新构造 `CGAEncodingAttr`
- 把这个新的 CTA ownership 沿 producer/consumer 链传播

`processDot` 的核心逻辑见
[PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)。

特别要注意两点：

- 它显式要求 dot operand/result 现在已经是 TTGIR distributed layout
- 它不是只改 dot result，而是会通过 `insertCasts` 把新 layout 传播到 operand/result 周围

这就是为什么 multi-CTA kernel 的 mapping 不能只看一个 `tt.dot`，要看整条 around-dot 链。

---

## 6. 三个最小 IR 证据

### 6.1 vecadd: TTIR 还没有 ownership，TTGIR 才开始有

在
[018_Before_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/018_Before_ConvertTritonToTritonGPU.mlir:12)
里，`%offsets`、`%x_3`、`%output` 都还只是：

```text
tensor<1024xi32>
tensor<1024x!tt.ptr<f32>>
tensor<1024xf32>
```

没有任何 distributed encoding。

到了
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:2)
就出现了：

```text
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

而且 `tt.make_range`、`tt.load`、`arith.addf`、`tt.store` 的 tensor type 全部带上了同一个 `#blocked`，
见
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:13)。

这就是 distributed execution mapping 的最小起点。

### 6.2 vecadd: Coalesce 不是“只做优化”，它真的改了 memory-facing ownership

在
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:3)
里，新增了：

```text
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

然后 load/store 周围都被改成：

```text
#blocked -> #blocked1 -> load/store -> #blocked
```

见
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:21)。

这说明 Coalesce 不是在“附加一个内存 hint”，而是在局部重写：

```text
哪个 thread 一次拿几个连续元素
```

### 6.3 matmul `sm90_num_ctas2`: PlanCTA 真正在改 CTA 级 ownership

PlanCTA 前，主 dot parent 是：

```text
#blocked8 ... CGALayout = [[1, 0]]
```

见
[026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:10)。
对应的 dot operand/result 也都挂在这个 parent 上，见
[026_Before_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/026_Before_TritonGPUPlanCTAPass.mlir:79)。

PlanCTA 后，主 dot parent 变成：

```text
#blocked ... CGALayout = [[0, 1]]
```

见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:2)。
新的 dot 直接返回这个 layout，见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:68)。

同时还出现了一个很有代表性的 A load layout：

```text
#blocked4 ... CGALayout = [[0, 0]]
```

见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:6)。
它在 A operand load 前被使用，见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:63)。

这组证据合起来，表达的是：

- output accumulator 沿 `N` 维被两个 CTA 分摊
- A operand 对这次 `N` 维 split 来说是 CTA-replicated / broadcasted
- 也就是说，PlanCTA 不只是“把结果 layout 改了一下”，而是在重写整个 dot neighborhood 的 CTA ownership model

---

## 7. 一张职责表

| 层 | 回答的问题 | 典型载体 | 代表 pass |
|---|---|---|---|
| default ownership establishment | 这个 logical tensor 首次落到 GPU 层时，默认由谁持有？ | module attrs, `#blocked` | `ConvertTritonToTritonGPU` |
| memory-facing ownership refinement | 对 load/store，谁应该一次拿多少连续元素更合适？ | 新的 `#blocked`, `ttg.convert_layout` | `Coalesce` |
| CTA-level collaboration planning | 多个 CTA 如何分摊一个 output tile？哪些输入需要复制或广播？ | `CGAEncodingAttr`, `#slice`, around-dot layout propagation | `PlanCTA` |
| compute-facing specialization | dot operand 如何对齐到 tensor-core / wgmma / tcgen05 所需分布？ | `#ttg.dot_op`, MMA parent layout | `AccelerateMatmul`, `OptimizeDotOperands`, later target-specific passes |
| execution ordering | 这些已分配好的 ownership 什么时候执行？ | stage/partition/wait/barrier attrs and ops | scheduling / pipeline passes |

---

## 8. 读 TTGIR mapping 时最有效的检查顺序

1. 先看 module attrs 里的 `ttg.num-warps`、`ttg.num-ctas`、`ttg.threads-per-warp`。
2. 再看文件顶部定义了哪些主 `#blocked`，尤其是 `sizePerThread`、`warpsPerCTA`、`CGALayout`。
3. 如果出现 `#ttg.slice`，不要把它当新 layout；先回到它的 `parent`。
4. 如果出现 `#ttg.dot_op`，先找它的 `parent` 是哪个 accumulator/result layout。
5. 如果 `num_ctas > 1`，重点看 `CGALayout` 是 `[[1, 0]]`、`[[0, 1]]` 还是 `[[0, 0]]` 这类模式。
6. 最后才去看 `ttg.convert_layout`。它通常是“ownership 在边界上切换”的显式症状，不是根因。

---

## 9. 和另外两篇文档的边界

- [TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:216) 给的是总框架。
- [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md:1) 讲 `when`.
- 本文讲 `who`.

压成一句话就是：

```text
TTGIR
  先决定谁持有 / 谁计算
  再决定值在不同阶段之间以什么组织形式流动
  必要时再决定这些工作何时执行、如何重叠
```

如果你把这三件事混在一起，`ttg.convert_layout`、`ttg.local_alloc`、`wait_barrier`、`dot_op`
这些表面现象就会很难读；拆开之后，TTGIR 的职责边界会清楚很多。
