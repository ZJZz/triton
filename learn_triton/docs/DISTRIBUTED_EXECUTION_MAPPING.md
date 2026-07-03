# Triton TTGIR：Distributed Execution Mapping

本文只回答一个问题：一个 logical tensor 在 TTGIR 里如何被切给 CTA、warp、thread 和 per-thread values。TTGIR 里的 layout encoding 不只是“内存布局”，很多时候就是这份 ownership contract 的承载体。

## 1. 核心定义

```text
distributed execution mapping
  = 把 logical tensor 分解到 CTA / warp / thread / value 层级，
    并把这份 ownership contract 编码进 tensor type，
    让后续的 Coalesce、PlanCTA、dot lowering 和 LLVM lowering
    都基于同一份分工继续工作。
```

源码把这套层级直接写成：

```text
CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread
```

见
[TritonGPUAttrInterfaces.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.td:41)。

这里回答的是 `who owns / who computes these elements`，不是 `what form`，也不是 `when`.

## 2. 和另外两层的边界

| 主题 | 核心问题 | 典型载体 |
|---|---|---|
| distributed execution mapping | 谁拥有这些元素 | `#blocked`、`CGAEncodingAttr`、`#ttg.slice`、`#ttg.dot_op` 的 parent |
| layout / data-movement organization | 这些值以什么 form / carrier 流动 | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMEM |
| target-driven scheduling | 这些工作何时发生、如何 overlap、何时同步 | `loop.stage`、`loop.cluster`、async op、wait、barrier |

有三个容易混淆的点：

- 它不是 `memory layout` 的别名。`#blocked` 当然影响 memory access，但先回答的是 `哪些元素归哪个 CTA / warp / thread`。
- 它不是 LLVM lowering 之后才决定。`ConvertTritonToTritonGPU` 已经把 encoding 写进 tensor type。
- 它和 scheduling 有依赖关系，但职责不同。mapping 先决定谁工作，scheduling 再决定这些工作何时发生。

## 3. 关键载体

### 3.1 module attrs：execution envelope

TTGIR module 顶层会挂这些 attr：

- `ttg.num-warps`
- `ttg.num-ctas`
- `ttg.threads-per-warp`
- `ttg.target`

定义见
[Dialect.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/Dialect.h:49)。

这些 attr 还不是每个 tensor 的具体 ownership，但它们给出了所有 distributed layout 的全局边界条件。

### 3.2 `#blocked`：默认、最常见的 ownership contract

`BlockedEncodingAttr` 的描述很直接：

- 每个 warp 拥有目标 tensor 的一个 contiguous portion
- 它由 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA` 三组量描述
- 这三组量分别对应 thread、warp、CTA 级别的元素组织

见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738)。

读 TTGIR 时，更实用的心智模型是：

```text
#blocked
  = valuesPerThread
  + threadsPerWarp
  + warpsPerCTA
  + order
  + optional CGALayout
```

它不是“普通 tensor + 一个 tag”，而是 ownership 的压缩表示。

### 3.3 `CGAEncodingAttr`：CTA 级切分单独编码

多 CTA 情况下，`CGAEncodingAttr` 单独负责：

```text
blocks (CTAs) in a cooperative thread array
如何映射到 logical tensor dimensions
```

定义见
[CGAEncodingAttr.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/CGAEncodingAttr.td:14)。

所以看到 `CGALayout = [[0, 1]]` 一类字段时，不要把它当附属信息。它通常就是“多个 CTA 沿哪个逻辑维度分摊 output tile”的答案。

### 3.4 `#ttg.slice`：parent layout 的投影

dump 里常见：

```text
#ttg.slice<{dim = 1, parent = #blocked}>
```

这不是新的独立分工，更准确的理解是：

```text
对 parent distributed layout 做降维投影，
让 `tt.make_range` / `tt.expand_dims` 一类 helper 值
继续共享同一套 ownership 语义
```

`SliceEncodingAttr` 的描述也明确说它常用于构造 `expand_dims` 的逆布局，见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1408)。

### 3.5 `#ttg.dot_op`：compute view 依附于 result parent

`DotOperandEncodingAttr` 记录 `opIdx`、`parent` 和 `kWidth`，其中 `parent` 就是 dot result 的 layout，见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)。

这说明：

```text
dot operand 的分布不是孤立决定的，
而是依附于 accumulator / result ownership contract
```

## 4. 这份 mapping 是怎么建立和演化的

NVIDIA backend 的 TTGIR 主干顺序是：

```text
ConvertTritonToTritonGPU
  -> Coalesce
  -> PlanCTA
  -> ...
```

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:269)。

### 4.1 `ConvertTritonToTritonGPU`：把 logical tensor 变成 distributed tensor

`ConvertTritonToTritonGPU` 的 pass 定义明确说，它会给 tensor type 增加 layout encoding，而这些 encoding 一般包含 `numWarps`、`threadsPerWarp` 和 `numCTAs`，见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)。

实现上，`TritonGPUTypeConverter` 会对没有 encoding 的 tensor type 调
`getDefaultBlockedEncoding(...)`，再 `cloneWithEncoding(...)`，见
[TritonGPUConversion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:19)。

同一个文件里，conversion target 还要求：

- `tt.dot` 的 A/B operand 必须已经带 `DotOperandEncodingAttr`
- function 的 tensor argument/result 也必须有 encoding

见
[TritonGPUConversion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp:80)。

这一步建立的 contract 是：

```text
从这里开始，tensor 不再只是 logical shape，
而是 logical shape + distributed ownership
```

### 4.2 `BlockedEncodingAttr` builder：先定 CTA split，再定 warp/thread decomposition

`BlockedEncodingAttr` 的 builder 会：

1. 根据 `shape / sizePerThread / order / numCTAs` 推出 `CTAsPerCGA` 和 `CTASplitNum`
2. 由此得到 `CGALayout`
3. 再基于 `shapePerCTA` 计算 `threadsPerWarp` 和 `warpsPerCTA`

实现见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:831)。

所以更贴近源码的读法是：

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

### 4.3 `Coalesce`：局部重写 memory-facing ownership

`TritonGPUCoalesce` 的 pass 定义说得很清楚：

- 它分析 `tensor<tt.ptr<...>>` 的 load/store
- 为这些 op 选择 coalesced layout
- 在前后插 `ttg.convert_layout` 保持程序其余部分不变

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:235)。

源码里它会取 pointer tensor 的 `CGAEncoding`、计算 `shapePerCTA`，再调
`buildCoalescedEncoding(...)`，见
[Coalesce.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:77)。

`buildCoalescedEncoding(...)` 又会结合：

- pointer 的 contiguity
- 同 shape / 同 order memory slice 的共同需求
- `shapePerCTA`
- 每线程最多处理多少元素

来生成新的 `BlockedEncodingAttr`，见
[CoalesceUtils.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:16)。

所以这一步回答的不是“谁来 load”，而是：

```text
既然已经知道谁来 load，
那让这些线程以什么 ownership 粒度访问更适合 memory system
```

### 4.4 `PlanCTA`：当 `num_ctas > 1` 时重写 CTA 级 ownership

`PlanCTA` 是 “mapping 在 conversion 之后还会继续演化” 的典型例子。

它会：

- 分析 dot/reduce/store-like pattern
- 选择 `splitM / splitN`
- 重建 `CGAEncodingAttr`
- 用 `insertCasts` 把新的 CTA ownership 沿 producer/consumer 链传播

核心逻辑见
[PlanCTA.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp:212)。

因此 multi-CTA kernel 的 mapping 不能只看一个 `tt.dot`，要看整个 around-dot neighborhood 的 layout propagation。

## 5. 最小 IR 证据

### 5.1 `vecadd`：TTIR 还没有 ownership，TTGIR 才开始有

在
[018_Before_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/018_Before_ConvertTritonToTritonGPU.mlir:12)
里，`%offsets`、`%x_3`、`%output` 还只是普通 tensor / pointer tensor，没有 distributed encoding。

到了
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:2)
就出现了：

```text
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

而且 `tt.make_range`、`tt.load`、`arith.addf`、`tt.store` 的 tensor type 都带上了同一个 `#blocked`，见
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:13)。

### 5.2 `vecadd`：`Coalesce` 会局部改写 memory-facing ownership

在
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:3)
里，新增了：

```text
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
```

load/store 周围变成：

```text
#blocked -> #blocked1 -> load/store -> #blocked
```

见
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:21)。

这说明 `Coalesce` 不是附加一个 memory hint，而是在局部重写“每个 thread 一次拿几个连续元素”。

### 5.3 `matmul` `sm90_num_ctas2`：`PlanCTA` 真正在改 CTA 级 ownership

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

同时还出现了：

```text
#blocked4 ... CGALayout = [[0, 0]]
```

见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:6)。
它在 A operand load 前被使用，见
[027_After_TritonGPUPlanCTAPass.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas2/mlir-pass-dump.split/027_After_TritonGPUPlanCTAPass.mlir:63)。

这组证据表达的是：

- output accumulator 沿 `N` 维被两个 CTA 分摊
- A operand 对这次 `N` 维 split 来说是 CTA-replicated / broadcasted
- `PlanCTA` 改的不是一个 result type，而是整个 dot neighborhood 的 CTA ownership model

## 6. 读这类 TTGIR 时的检查顺序

1. 先看 module attrs：`ttg.num-warps`、`ttg.num-ctas`、`ttg.threads-per-warp`。
2. 再看文件顶部定义的主 `#blocked`，尤其是 `sizePerThread`、`warpsPerCTA`、`CGALayout`。
3. 遇到 `#ttg.slice` 时先回到它的 `parent`，不要把它当独立 layout。
4. 遇到 `#ttg.dot_op` 时先找它的 `parent`，再判断 dot neighborhood 的 ownership。
5. 如果 `num_ctas > 1`，重点看 `CGALayout` 如何变化。
6. `ttg.convert_layout` 通常是 ownership 切换的症状，不是分析起点。
