# Triton TTGIR 中的 Layout / Data-Movement Organization

目标：把
[TTGIR_GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TTGIR_GUIDE.md:60)
里这句

```text
TTGIR = distributed execution mapping
      + layout / data-movement organization
      + target-driven scheduling
```

真正落成一条可验证的源码阅读路径。

这篇文档回答的问题不是“谁算这些元素”，也不是“这些工作何时发生”，而是：

```text
已经分配好 ownership 的这些值，
在 producer 和 consumer 之间，
应该以什么组织形式存在？

它们应该保持 distributed tensor、
转成 coalesced load/store 友好的 form、
落到 shared memory tile、
挂到 descriptor/TMA 路径、
还是暂时住进 TMEM？
```

如果你想看 `who`，请看
[DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md:1)。
如果你想看 `when`，请看
[TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md:1)。
本文只看 “in what form should these values exist between uses?”。

---

## 1. 一句话结论

```text
layout / data-movement organization
  = 编译器为“同一批 logical values”选择一条表示与搬运路径：
    distributed tensor
    -> coalesced memory-facing form
    -> shared-memory memdesc view
    -> compute-specific operand form
    -> descriptor/TMA/TMEM 等 target-specific carrier
```

它关心的不是“谁拥有这些值”，而是：

```text
这些值接下来要被谁消费，
因此它们现在该长成什么样、放在哪里、通过什么介质流动
```

---

## 2. 它不是什么

### 2.1 不是 distributed execution mapping

distributed execution mapping 回答的是：

```text
谁拥有 / 谁计算这些元素
```

layout / data-movement organization 回答的是：

```text
同样这批元素，
在 load/store/dot/TMA/TMEM 等边界上，
应该以什么表示被消费
```

同一个 ownership contract 下面，可以出现多种 value form。

例如同一个 tile 可以先是 `#blocked` distributed tensor，
再转成 coalesced load/store form，
再落成 shared `memdesc`，
再通过 `#ttg.dot_op` 或 `#mma` 喂给 tensor-core consumer。

### 2.2 不是 target-driven scheduling

layout / movement 回答的是 **what form / what path**；
target-driven scheduling 回答的是 **when / what ordering**。

例如：

- `ttg.local_alloc`
- `ttg.local_load`
- `ttg.memdesc_subslice`
- `ttng.tmem_alloc`
- `ttg.convert_layout`

这些大多是在改值的承载体或表示形式。

而：

- `loop.stage`
- `ttg.async_wait`
- `ttng.wait_barrier`
- `ttng.warp_group_dot_wait`

这些主要是在控制时序和交接。

### 2.3 不是“只要看到 layout attr 就是一回事”

TTGIR 里 `layout` 这个载体本身有点超载：

- 有些 layout attr 主要承载 ownership
- 有些 layout attr 主要承载 memory-facing / compute-facing organization
- 有些 op 则把“值不再是普通 distributed tensor”这件事显式化

所以不能把所有变化都压扁成一句“又改 layout 了”。

---

## 3. 为什么 TTGIR 必须单独回答这个问题

GPU 后端面对的不是一个抽象 tensor，而是一条真实的消费链。

同一个逻辑 tile，后面可能会遇到这些 consumer / transport 约束：

- global load/store 想要更 coalesced 的访问模式
- dot / mma / wgmma / tcgen05 想要特定 operand form
- shared memory staging 想要 `memdesc` 视图和子切片
- Hopper TMA 想要 descriptor-backed transport
- Blackwell TCGen5 想要 TMEM accumulator

因此 TTGIR 不能只知道 “这个 tile 归谁算”，还必须知道：

```text
它现在该以什么 representation 继续存在，
才能让后面的 memory system / tensor-core path / target-specific transport
都合法且高效
```

---

## 4. 主要承载体

### 4.1 `ttg.convert_layout`: 同值、异形态 的显式边界

`ttg.convert_layout` 的 op 定义很克制：
它是 pure op，保持 shape 和 element type 不变，见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32)。

所以它最应该被读成：

```text
logical value 没变，
但 consumer 希望它换一种 distributed / memory-facing / compute-facing form
```

更关键的是，allocation analysis 里直接把：

- `ttg.local_alloc` 记为 explicit buffer
- `ttg.convert_layout` 记为 scratch buffer

见
[Allocation.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Analysis/Allocation.h:167)。

这说明 `convert_layout` 在编译器眼里并不只是“类型标签改写”；
它经常意味着真实的数据重组织，甚至可能需要 scratch/shared storage。

### 4.2 shared `memdesc`: 值暂时不再是 distributed tensor

`ttg.local_alloc` 的定义明确说：它在 shared memory 中分配 buffer，
返回的是一个 descriptor，而不是 tensor 本身，见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:152)。

对应地：

- `ttg.local_load`：从 local `memdesc` 读回 distributed tensor
- `ttg.local_store`：把 distributed tensor 写进 local `memdesc`

定义见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:359)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:382)。

这条边界非常重要，因为它表达的是：

```text
值暂时退出“distributed tensor world”，
改以 shared-memory descriptor 的方式存在
```

而 `memdesc_subslice` / `memdesc_trans` / `memdesc_reshape` /
`memdesc_reinterpret` 则是在同一块 shared buffer 上重放 view-level 组织，
见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:242)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:273)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:297)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:336)。

### 4.3 compute-facing form: `#ttg.dot_op` 不是普通 load layout

`DotOperandEncodingAttr` 的定义写得很直白：

- pre-Hopper 的 `tt.dot` operand 必须是 `DotOperandEncodingAttr`
- 它的 `parent` 是 dot result 的 layout

见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)。

这意味着：

```text
dot operand form 不是“顺手改一下 layout”，
而是为特定 compute consumer 准备的专用表示
```

`RemoveLayoutConversions` 的 pass 描述也直接说明了它的偏好：

- 对 expensive load/store，更偏好 `BlockedEncodingAttr`
- 对 tensor ops，更偏好 `NvidiaMmaEncodingAttr`

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)。

所以 compute-facing organization 的核心不是“再插一个 convert”，而是：

```text
consumer 变了，
值的最佳 form 也跟着变
```

### 4.4 descriptor-backed transport: Hopper+ 上值可以走 descriptor/TMA 路径

这部分和普通 `tt.load` / `tt.store` 不同。

在 Triton dialect 里：

- `tt.make_tensor_descriptor` 创建带 parent meta-info 和 block size 的 descriptor
- `tt.descriptor_load` / `tt.descriptor_store` 在支持的 target 上会 lower 成 NVIDIA TMA

见
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:983)，
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1226)，
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1254)。

而 backend 从 pipeline 组织层面就把 pre-Hopper 和 Hopper+ 分开了：

- `capability // 10 < 9` 时，在 `make_ttir` 里先把 tensor descriptor 改写回 pointer 语义
- Hopper+ 则保留 descriptor 路径，留给后面的 TMA 相关 pass

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:250)，
[Triton Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/Transforms/Passes.td:39)。

descriptor 自身的 shared-memory encoding 也不是随便挂的。
`AssignDescriptorMemoryLayouts` 抽象负责给 descriptor 选 memory layout，
而 NVIDIA backend 的 `OptimizeDescriptorEncoding` 会挑兼容 TMA 的 shared encoding，
见
[DescriptorMemoryLayouts.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h:25)，
[OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:13)，
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:157)。

`TMALowering` 则把这条 descriptor 路径真正物化成 transport protocol：

- load 侧：`LocalAllocOp` + barrier alloc + `AsyncTMACopyGlobalToLocal` + `WaitBarrierOp`
- store 侧：`LocalAllocOp %src` + `FenceAsyncSharedOp` + async TMA store + `TMAStoreWaitOp`

见
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:27)，
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:100)，
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:166)。

这条链说明：

```text
descriptor-backed transport
  不是“又一种 load op 名字”，
  而是值在 target-specific movement path 上的另一种 carrier
```

说明：当前 `learn_triton/dumps/` 这组 canonical dump 里没有单独的 descriptor kernel，
这一小节主要依赖源码链而不是现成 dump。

### 4.5 TMEM / linear accumulator: Blackwell 上 accumulator 也会换 carrier

Blackwell 上，值的组织形式还会再多一层：

- accumulator 可以先从 distributed `#blocked` tensor
- 转成 `#linear`
- 再住进 `#tmem`
- 读回时再从 `#linear` 转回 distributed tensor

相关 pass 入口见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:121)，
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:169)。

这类变化很明显不是 ownership 变化，而是：

```text
为了 TCGen5 / TMEM 这条 consumer path，
accumulator 暂时改用另一种 residence + layout contract
```

---

## 5. 源码里“谁负责哪类 organization 决策”

先看 NVIDIA backend 的主干顺序：

```text
ConvertTritonToTritonGPU
  -> Coalesce
  -> PlanCTA
  -> RemoveLayoutConversions
  -> OptimizeThreadLocality
  -> AccelerateMatmul
  -> RemoveLayoutConversions
  -> OptimizeDotOperands
  -> OptimizeDescriptorEncoding
  -> ...
  -> CoalesceAsyncCopy
  -> OptimizeTMemLayouts
  -> TMALowering
  -> RemoveLayoutConversions
```

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:269)。

可以把这条链压成一张职责表：

| 层 | 主要问题 | 典型载体 | 主要 pass |
|---|---|---|---|
| memory-facing organization | 对 global load/store，值先以什么 form 访问更合适？ | 新的 `#blocked`、`ttg.convert_layout` | `Coalesce` |
| convert-chain cleanup / preference | 现有 representation 链里哪些 form 是多余的，哪些更适合 load/store 或 tensor op？ | 更少的 `ttg.convert_layout`、更直接的 dot/store 入口 | `RemoveLayoutConversions` |
| cross-thread / local-consumer refinement | 某些 reduction/gather 是否该改成更 thread-local 或 warp-synchronous 的 form？ | layout tweaks around reduction/gather | `OptimizeThreadLocality` |
| compute-facing specialization | dot operand 应该以什么 consumer-specific form 喂给 tensor-core path？ | `#ttg.dot_op`、shared `memdesc`、mma-friendly shared encoding | `AccelerateMatmul`、`OptimizeDotOperands` |
| register/shared traffic refinement | distributed -> dotOperand 是否该分解成 shared-backed path，以减少 duplication？ | `local_alloc`、`local_load`、shared reuse | `ReduceDataDuplication` |
| async-copy-facing refinement | async global-to-local 的 coalesced form 是否还需要修正？ | clipped `sizePerThread`, async-copy-friendly form | `CoalesceAsyncCopy` |
| descriptor/TMA organization | descriptor 应该挂什么 shared encoding；desc load/store 如何变成 TMA path？ | tensor descriptor、shared encoding、TMA ops | `OptimizeDescriptorEncoding`、`TMALowering` |
| TMEM organization | 哪些值该住进 TMEM；TMEM layout 怎么选？ | `#linear`、`#tmem`、`ttng.tmem_*` | `PromoteLHSToTMem`、`OptimizeTMemLayouts` |

### 5.1 `Coalesce`: 先回答 memory-facing form

`Coalesce` 的 pass 描述是：

```text
分析 tensor<tt.ptr<...>> load/store，
把这些 op 的 layout 换成 coalesced layout，
并在前后插 layout conversion 保持程序其余部分不变
```

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:235)。

实现上，`buildCoalescedEncoding(...)` 主要看：

- pointer contiguity 推出来的 `order`
- `shapePerCTA`
- `numWarps * threadsPerWarp`
- 每线程该拿多少元素 `perThread`

见
[CoalesceUtils.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:17)。

所以它回答的不是“谁来 load”，而是：

```text
既然已经知道谁来 load，
那让这些线程以什么 memory-facing form 访问更合适
```

### 5.2 `RemoveLayoutConversions`: 把 representation 链压到更合理的形状

`RemoveLayoutConversions` 的目标不是盲目删 op。
它会减少 `ConvertLayoutOp` 的数量，并偏向：

- load/store 用更 favorable 的 `BlockedEncodingAttr`
- tensor op 用更 favorable 的 `NvidiaMmaEncodingAttr`

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)。

这其实是在做 representation-chain normalization：

```text
值要换 form 是真的，
但不需要每经过一个小边界都插一层多余的中间 form
```

### 5.3 `OptimizeDotOperands`: 把 consumer 需求反推回 shared / memdesc 视图

`OptimizeDotOperands.cpp` 里几类 pattern 很能说明问题：

- `SwizzleShmemConvert`
- `FuseTransMMAV3Plus`
- `ReshapeMemDesc`
- `RewriteMmaOperandViewsToMemDescForDotOp`

见
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:24)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:91)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:141)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:191)。

它们的共同点是：

```text
不是先把 tensor 变完再喂给 dot，
而是从 dot-like consumer 反推：
shared encoding / transpose / reshape / memdesc view
应该长成什么样，才能更适合这个 consumer
```

所以这类 pass 处理的不是 ownership，而是 consumer-driven organization。

---

## 6. dump 证据

### 6.1 `vecadd` `sm86`: Coalesce 在 load/store 周围插入 organization boundary

`After ConvertTritonToTritonGPU` 时，load/store 还直接吃默认 `#blocked`：

- `tt.load %x_3, %mask_2`
- `tt.load %y_5, %mask_2`
- `tt.store %1, %output, %mask_2`

见
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:20)。

`After TritonGPUCoalesce` 之后，前后都被 `ttg.convert_layout` 包起来：

- pointer / mask 先转到 `#blocked1`
- `tt.load` 在 `#blocked1` 上发生
- 结果再转回原先 form
- store 也同样包起来

见
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:21)。

这组对比最能说明：

```text
Coalesce 没改程序语义，也没改 ownership；
它改的是值经过 memory op 时的组织形式
```

### 6.2 `matmul_contiguous` `sm86`: RemoveLayoutConversions 压缩中间表示链

`Before TritonGPURemoveLayoutConversions` 时，dot 之前的表示链很长：

- pointer 先转成临时 blocked form
- load 之后再转回另一种 blocked form
- 再转成 `#ttg.dot_op`
- accumulator 也要先转 parent form，dot 后结果再转回来

见
[030_Before_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/030_Before_TritonGPURemoveLayoutConversions.mlir:68)。

`After TritonGPURemoveLayoutConversions` 后，这条链明显变短：

- load 直接产出更合适的 blocked form
- 只在真正的 dot consumer 边界上转到 `#ttg.dot_op`
- store 前保留必要的一次 convert

见
[031_After_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/031_After_TritonGPURemoveLayoutConversions.mlir:63)。

这说明 `RemoveLayoutConversions` 的核心不是“删 op 数量”本身，而是：

```text
把值的表示链压到更接近真实 consumer 边界的形状
```

### 6.3 `matmul` `sm90_num_ctas1`: shared `memdesc` 成为 WGMMA consumer 的直接输入

Hopper 这份 dump 很直接：

- A/B 先 `tt.load`
- 立刻 `ttg.local_alloc` 成 shared `memdesc`
- `ttng.warp_group_dot` 直接吃 shared `memdesc`
- 结果出来后再 `ttg.convert_layout` 回 store-facing form

见
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:71)，
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:74)，
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:90)。

这不是 mapping 的变化，而是：

```text
同一个 tile，
为了 WGMMA consumer，
改成 shared descriptor-backed tile 的形式存在
```

### 6.4 `matmul` `sm100_num_ctas1`: accumulator 走 `blocked -> linear -> TMEM -> linear -> blocked`

Blackwell 这份 dump 更能看出 “carrier” 的变化：

- A/B 先落成 shared `memdesc`
- accumulator 先 `ttg.convert_layout` 到 `#linear`
- 再 `ttng.tmem_alloc` 进 `#tmem`
- `ttng.tc_gen5_mma` 在 TMEM accumulator 上执行
- `ttng.tmem_load` 读回 `#linear`
- 再 `ttg.convert_layout` 回 `#blocked`

见
[033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:70)，
[033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:73)，
[033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:74)，
[033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:76)。

这组证据很关键，因为它证明：

```text
layout / movement organization
  不只是“换个 blocked 参数”
  还包括值短暂离开 distributed tensor world，
  住进 target-specific memory carrier
```

---

## 7. 读 TTGIR 的组织变化时，最有效的检查顺序

1. 先问自己：这里改的是 ownership，还是 value form。
2. 看到 `ttg.convert_layout` 时，不要先问“为什么又有 convert”；先问“前后两个 consumer 分别需要什么 form”。
3. 看到 `ttg.local_alloc` / `local_load` / `local_store` 时，要意识到值已经进出 shared `memdesc` world 了。
4. 看到 `memdesc_subslice` / `trans` / `reshape` / `reinterpret` 时，先把它当成“同一 buffer 的 view 重组”，不要急着当成新 copy。
5. 看到 `#ttg.dot_op`、`#mma`、`#linear`、`#tmem` 时，优先问“这是哪个 compute/transport consumer 的专用 form”。
6. 只有最后才问“这一步对时序意味着什么”；那通常属于 scheduling 文档的边界。

---

## 8. 和另外两篇文档的边界

- [DISTRIBUTED_EXECUTION_MAPPING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/DISTRIBUTED_EXECUTION_MAPPING.md:433) 讲 `who`.
- 本文讲 `what form / what movement path`.
- [TARGET_DRIVEN_SCHEDULING.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md:1) 讲 `when`.

压成一句话就是：

```text
TTGIR
  先决定谁持有 / 谁计算
  再决定这些值以什么组织形式在不同 consumer 之间流动
  必要时再决定这些工作何时发生、如何重叠、如何同步
```

一旦把这三层拆开，`ttg.convert_layout`、`ttg.local_alloc`、`tt.make_tensor_descriptor`、
`ttng.tmem_alloc` 这些表面现象就不会再全都挤成一句“都是调整 layout”了。
