# Triton TTGIR：Layout / Data-Movement Organization

本文只回答一个问题：同一批 logical values 在 producer 和 consumer 之间，应该以什么表示存在，通过什么 carrier 流动。这里关注的是 `what form / what path`，不是 `who`，也不是 `when`。

## 1. 核心定义

```text
layout / data-movement organization
  = 为同一批 values 选择一条表示与搬运路径：
    带执行分发信息的 tensor
    -> memory-facing form
    -> shared memdesc view
    -> compute-specific operand form
    -> descriptor / TMA / TMEM 等 target-specific carrier
```

它回答的是：

```text
这些值接下来要被谁消费，
因此它们现在该长成什么样、放在哪里、通过什么介质流动
```

如果把 TTGIR 三层压成一句短记忆：

```text
mapping = 分工
organization = 衔接
scheduling = 时序
```

那么本文只负责第二层：`organization = 衔接`。更准确地说，它桥接的是 producer form 和 consumer-required form。

## 2. 和另外两层的边界

| 主题 | 核心问题 | 典型载体 |
|---|---|---|
| distributed execution mapping（执行层级上的分工映射） | 谁拥有这些元素 | `#blocked`、`CGAEncodingAttr`、`#ttg.slice` |
| layout / data-movement organization | 这些值以什么 form / carrier 流动 | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMEM |
| target-driven scheduling | 这些工作何时发生、如何 overlap、何时同步 | `loop.stage`、async op、wait、barrier |

有三个容易混淆的点：

- 它不是 distributed execution mapping。这里的重点是 CTA / warp / thread 执行层级上的分工；mapping 回答 `who computes`，本文回答 `same values in what form move between uses`。
- 它不是 scheduling。本文回答 `what form / what path`；scheduling 回答 `when / what ordering`。
- 它也不是“只要看到 layout attr 就是一回事”。有些 encoding 主要承载 ownership，有些主要承载 memory-facing / compute-facing organization，还有些 op 直接把 carrier 切换显式化。

## 3. 关键载体

### 3.1 `ttg.convert_layout`：同值、异形态 的显式边界

`ttg.convert_layout` 是 pure op，保持 shape 和 element type 不变，见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32)。

它最应该被读成：

```text
logical value 没变，
但下一个 consumer 希望它换一种执行分发表达 / memory-facing / compute-facing form
```

更关键的是，allocation analysis 里直接把：

- `ttg.local_alloc` 记为 explicit buffer
- `ttg.convert_layout` 记为 scratch buffer

见
[Allocation.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Analysis/Allocation.h:167)。

这说明 `convert_layout` 在编译器眼里往往意味着真实的数据重组织，而不只是标签改写。

### 3.2 shared `memdesc`：值暂时退出“带执行分发信息的 tensor”世界

`ttg.local_alloc` 会在 shared memory 中分配 buffer，返回的是 descriptor，而不是 tensor 本身，见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:152)。

可以把它近似理解成“这批 tensor values 在 shared memory 里的承载形式”，但更准确地说，它是这份数据在 shared memory 中的 descriptor / handle，不是 tensor 本体。

对应地：

- `ttg.local_load`：从 local `memdesc` 读回带执行分发信息的 tensor
- `ttg.local_store`：把带执行分发信息的 tensor 写进 local `memdesc`

定义见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:359)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:382)。

这条边界表达的是：

```text
值暂时不再以“带执行分发信息的 tensor”的形式存在，
而是以 shared-memory descriptor 的方式存在
```

同一块 shared buffer 上的 view-level 重组由这些 op 表达：

- `memdesc_subslice`
- `memdesc_trans`
- `memdesc_reshape`
- `memdesc_reinterpret`

定义见
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:242)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:273)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:297)，
[TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:336)。

### 3.3 compute-facing form：`#ttg.dot_op` 不是普通 load layout

`DotOperandEncodingAttr` 明确写着：

- pre-Hopper 的 `tt.dot` operand 必须是 `DotOperandEncodingAttr`
- 它的 `parent` 是 dot result 的 layout

见
[TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1431)。

这意味着：

```text
dot operand form 不是“顺手改一下 layout”，
而是为特定 compute consumer 准备的专用表示
```

`RemoveLayoutConversions` 的描述也说明了这种 consumer preference：

- 对 expensive load/store，更偏好 `BlockedEncodingAttr`
- 对 tensor ops，更偏好 `NvidiaMmaEncodingAttr`

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)。

### 3.4 descriptor / TMA path：值可以走 target-specific transport

在 Triton dialect 里：

- `tt.make_tensor_descriptor` 创建带 parent meta-info 和 block size 的 descriptor
- `tt.descriptor_load` / `tt.descriptor_store` 在支持的 target 上会 lower 成 NVIDIA TMA

见
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:983)，
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1226)，
[TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1254)。

backend 在 pipeline 组织层面就把 pre-Hopper 和 Hopper+ 分开了：

- `capability // 10 < 9` 时，在 `make_ttir` 里先把 tensor descriptor 改写回 pointer 语义
- Hopper+ 则保留 descriptor 路径，留给后面的 TMA pass

见
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:250)，
[Triton Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/Transforms/Passes.td:39)。

descriptor 自身的 shared-memory encoding 也不是随便挂的：

- `AssignDescriptorMemoryLayouts` 负责给 descriptor 选择 memory layout
- `OptimizeDescriptorEncoding` 负责挑兼容 TMA 的 shared encoding

见
[DescriptorMemoryLayouts.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h:25)，
[OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:13)，
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:157)。

`TMALowering` 再把这条 descriptor 路径物化成 transport protocol：

- load 侧：`LocalAllocOp` + barrier alloc + `AsyncTMACopyGlobalToLocal` + `WaitBarrierOp`
- store 侧：`LocalAllocOp %src` + `FenceAsyncSharedOp` + async TMA store + `TMAStoreWaitOp`

见
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:27)，
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:100)，
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:166)。

### 3.5 TMEM / linear accumulator：Blackwell 上 accumulator 也会换 carrier

Blackwell 上，accumulator 还会出现一层 target-specific residence 变化：

```text
#blocked
  -> #linear
  -> #tmem
  -> #linear
  -> #blocked
```

相关 pass 入口见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:121)，
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:169)。

这类变化不是 ownership 变化，而是为了 `tcgen05` / TMEM consumer path 选择另一种 carrier。

## 4. 这些 organization 决策由哪些 pass 负责

NVIDIA backend 的相关主干是：

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

可以压成四类职责：

| 层 | 主要问题 | 代表 pass |
|---|---|---|
| memory-facing organization | load/store 周围的值应以什么 form 访问更合适 | `Coalesce` |
| representation-chain cleanup | 哪些中间 form 是多余的，哪些 form 更适合当前 consumer | `RemoveLayoutConversions` |
| compute-facing specialization | dot / mma consumer 需要什么 operand / shared view | `AccelerateMatmul`、`OptimizeDotOperands` |
| target-specific carrier selection | descriptor/TMA/TMEM 该如何承载这些值 | `OptimizeDescriptorEncoding`、`TMALowering`、`PromoteLHSToTMem`、`OptimizeTMemLayouts` |

### 4.1 `Coalesce`：先回答 memory-facing form

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

### 4.2 `RemoveLayoutConversions`：压缩 representation 链

`RemoveLayoutConversions` 的目标不是盲目删 op。它会减少 `ConvertLayoutOp` 的数量，并偏向：

- load/store 用更 favorable 的 `BlockedEncodingAttr`
- tensor op 用更 favorable 的 `NvidiaMmaEncodingAttr`

见
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)。

这一步做的是 representation-chain normalization，而不是重新定义 ownership。

### 4.3 `OptimizeDotOperands`：从 consumer 反推 shared / memdesc 视图

`OptimizeDotOperands.cpp` 里的代表性 pattern 有：

- `SwizzleShmemConvert`
- `FuseTransMMAV3Plus`
- `ReshapeMemDesc`
- `RewriteMmaOperandViewsToMemDescForDotOp`

见
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:24)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:91)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:141)，
[OptimizeDotOperands.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp:191)。

它们共同说明：

```text
不是先把 tensor 变完再喂给 dot，
而是从 dot-like consumer 反推：
shared encoding / transpose / reshape / memdesc view
应该长成什么样
```

## 5. 最小 IR 证据

### 5.1 `vecadd` `sm86`：`Coalesce` 在 memory op 周围插入 organization boundary

`After ConvertTritonToTritonGPU` 时，load/store 还直接吃默认 `#blocked`，见
[019_After_ConvertTritonToTritonGPU.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/019_After_ConvertTritonToTritonGPU.mlir:20)。

`After TritonGPUCoalesce` 之后，pointer / mask 先转到 `#blocked1`，`tt.load` 在 `#blocked1` 上发生，结果再转回原先 form，store 侧也一样，见
[021_After_TritonGPUCoalesce.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/vecadd/sm86/mlir-pass-dump.split/021_After_TritonGPUCoalesce.mlir:21)。

这组对比表达的是：`Coalesce` 改的是值经过 memory op 时的组织形式。

### 5.2 `matmul_contiguous` `sm86`：`RemoveLayoutConversions` 压缩中间表示链

`Before TritonGPURemoveLayoutConversions` 时，dot 前的表示链很长：pointer 先转成临时 blocked form，load 后再转回另一种 blocked form，再转成 `#ttg.dot_op`，见
[030_Before_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/030_Before_TritonGPURemoveLayoutConversions.mlir:68)。

`After TritonGPURemoveLayoutConversions` 后，load 直接产出更合适的 blocked form，只在真正的 dot consumer 边界上转到 `#ttg.dot_op`，见
[031_After_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul_contiguous/sm86_num_ctas1/mlir-pass-dump.split/031_After_TritonGPURemoveLayoutConversions.mlir:63)。

### 5.3 `matmul` `sm90_num_ctas1`：shared `memdesc` 成为 WGMMA consumer 的直接输入

Hopper 这份 dump 里：

- A/B 先 `tt.load`
- 立刻 `ttg.local_alloc` 成 shared `memdesc`
- `ttng.warp_group_dot` 直接吃 shared `memdesc`
- 结果出来后再 `ttg.convert_layout` 回 store-facing form

见
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:71)，
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:74)，
[061_After_TritonGPUPipeline.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/061_After_TritonGPUPipeline.mlir:90)。

这表达的是：同一个 tile 为了 WGMMA consumer 改成 shared descriptor-backed tile 的形式存在。

### 5.4 `matmul` `sm100_num_ctas1`：accumulator 走 `blocked -> linear -> TMEM -> linear -> blocked`

Blackwell 这份 dump 里：

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

这里最关键的事实是：值短暂离开了普通“带执行分发信息的 tensor”世界，住进了 target-specific memory carrier。

说明：当前 `learn_triton/dumps/` 这组 canonical dump 里没有单独的 descriptor kernel，所以 descriptor / TMA 小节主要依赖源码链而不是现成 dump。

## 6. 读这类 TTGIR 时的检查顺序

1. 先分清这是 ownership 问题，还是 value form / carrier 问题。
2. 看到 `ttg.convert_layout` 时，先问前后两个 consumer 各自需要什么 form。
3. 看到 `ttg.local_alloc` / `local_load` / `local_store` 时，要意识到值已经进出 shared `memdesc` world。
4. 看到 `memdesc_subslice` / `trans` / `reshape` / `reinterpret` 时，先把它们当同一块 buffer 的 view 重组。
5. 看到 `#ttg.dot_op`、`#linear`、`#tmem`、descriptor 时，优先问“这是哪个 consumer / transport path 的专用 form”。
6. 只有最后才问时序问题；那属于 scheduling 的边界。
