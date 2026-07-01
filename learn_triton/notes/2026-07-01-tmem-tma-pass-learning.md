# 2026-07-01 学习笔记：Tensor Memory / TMA 相关 pass

## 1. 目标

这份笔记只看一条后端主线：

```text
tensor descriptor / async shared path
  + tensor memory path
  -> 在 NVIDIA backend 中分别由哪些 pass 建立 contract
  -> 这些 contract 为什么必须存在
  -> Hopper 和 Blackwell 在哪里开始分叉
```

这里把两类东西先分清：

- `TMA`:
  Tensor Memory Accelerator。核心问题是“如何把 global tensor tile 通过 descriptor 异步搬到 shared memory，或者从 shared memory 异步写回 global”。
- `TMEM`:
  Tensor Memory。核心问题是“如何给 Blackwell TCGen5 MMA 路径提供专用 tensor-memory 存储、布局、分配和同步”。

它们相关，但不是同一个东西：

- TMA 的数据落点通常是 shared memory。
- TMEM 是 Blackwell tensor-core pipeline 自己消费/产出的专用 memory space。

## 2. 先给结论

### 2.1 Hopper 主线

Hopper (`sm90`) 的重点是：

- 用 `OptimizeDescriptorEncoding` 先把 `tt.tensordesc` 绑定到 TMA 可接受的 shared layout。
- 用 `TMALowering` 把 `tt.descriptor_*` 改写成 `ttng.async_tma_*` + `mbarrier` + `local_alloc/local_load`。
- 用 `FenceInsertion` / `ProxyFenceInsertion` 保证 generic proxy 和 async proxy 间的可见性。

这一代没有 Blackwell 那条“accumulator 常驻 TMEM”的主路径。

### 2.2 Blackwell 主线

Blackwell (`sm100`) 在保留 TMA descriptor 路径的同时，多了一条更核心的 TMEM 路线：

- `PromoteLHSToTMem` 把满足条件的 MMA LHS 从 shared promotion 成 TMEM operand。
- `OptimizeTMemLayouts` 选择更适合 `tmem_load/tmem_store` 和 TMEM reduction 的 distributed layout。
- `InterleaveTMem` 把 `tmem_load` 尽量下沉到 use 附近，减少寄存器活跃范围。
- `TensorMemoryAllocation` 给 `ttng.tmem_alloc` 分配实际 row/col offset。
- `TMemBarrierInsertion` 在 TMEM RAW/WAR/WAW 或 MMA 相关依赖间插 `ttg.barrier`。

所以 Hopper 的关键词是：

- descriptor
- TMA async copy
- async/shared fence

Blackwell 的关键词是：

- TMEM allocation
- TCGen5 MMA
- TMEM layout
- TMEM barrier

## 3. Pipeline 位置

当前 NVIDIA backend 的关键位置在 [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:269)。

和这份笔记直接相关的 pass 顺序大致是：

```text
make_ttgir:
  ...
  optimize_descriptor_encoding
  ...
  if sm100+:
    hoist_tmem_alloc
    promote_lhs_to_tmem
    ...
    hoist_tmem_alloc
    remove_tmem_tokens
  ...
  optimize_tmem_layouts
  if sm90+:
    tma_lowering
  remove_layout_conversions
  interleave_tmem
  ...
  fence_insertion
  lower_mma

make_llir:
  ...
  allocate_shared_memory_nv
  allocate_tensor_memory
  ...
  proxy_fence_insertion
  tmem_barrier_insertion
  to_llvmir
```

这里最重要的分层是：

1. `make_ttgir` 先建立“descriptor / TMEM / async op”这些高层 GPU contract。
2. `make_llir` 再把 shared/TMEM 这些抽象资源分配到真实地址，并补齐更低层的 fence / barrier。

## 4. 统一心智模型

这条线本质上在回答四个问题：

1. descriptor 对应的 shared layout 究竟是什么？
2. descriptor 访问何时从抽象 `tt.descriptor_*` 变成真实的 TMA async op？
3. Blackwell 的 MMA operand / accumulator 何时切到 TMEM？
4. TMEM 的地址、生命周期和同步何时被实体化？

把这四个问题拆开看，比把所有 pass 记成名字清单更有用。

## 5. TMA 路径

## 5.1 `OptimizeDescriptorEncoding`

源码：
[OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:1)

### Goal

给 `tt.tensordesc` 绑定一个 NVIDIA TMA 真正能接受的 shared-memory encoding。

### Constraint

TMA descriptor 不接受任意 Triton shared layout。

当前实现里最关键的限制是：

- 兼容的 shared encoding 必须是非转置的 `NVMMASharedEncodingAttr`。

代码见：
[OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:26)

### Decision

如果 descriptor 当前 layout 已经兼容，就保留。

如果它来自 `SharedLinearEncodingAttr`，则尝试找一个线性等价、但 TMA 可编码的 `nvmma_shared`：

- 优先用 shape/order 推导出的 preferred candidate。
- 不行再枚举 swizzle `{0, 32, 64, 128}` 和 `fp4Padded`。

### Output contract

在这个 pass 之后，后续 TMA lowering 可以假设：

- descriptor 上已经有 shared layout；
- 这个 layout 能映射到合法的 TMA swizzle mode。

### 为什么它必须先于 `TMALowering`

因为 `TMALowering` 里的 `createTMADesc` 会直接从 descriptor 的 shared layout 读出：

- swizzle mode
- block shape
- element type enum

见：
[TMAUtilities.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:27)
[TMAUtilities.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:125)

如果这里还是抽象或不兼容 layout，后面的 tensormap create 就没有确定语义。

## 5.2 `TMALowering`

源码：
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:1)

测试：
[tma_lowering.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonNvidiaGPU/tma_lowering.mlir:1)

### Goal

把 TTIR 层的抽象 descriptor 访问，改成 NVIDIA dialect 里的真实 TMA 异步协议。

### 它改写什么

- `tt.descriptor_load`
  -> `ttng.async_tma_copy_global_to_local` + barrier + `ttg.local_load`
- `tt.descriptor_gather`
  -> `ttng.async_tma_gather` + barrier + `ttg.local_load`
- `tt.descriptor_store`
  -> `ttg.local_alloc` + `ttng.fence_async_shared` + `ttng.async_tma_copy_local_to_global`
- `tt.descriptor_reduce`
  -> `ttg.local_alloc` + `ttng.fence_async_shared` + `ttng.async_tma_reduce`
- `tt.descriptor_scatter`
  -> `ttg.local_alloc` + `ttng.fence_async_shared` + `ttng.async_tma_scatter`
- `tt.make_tensor_descriptor`
  -> `ttg.global_scratch_alloc` + `ttng.tensormap_create` + `ttng.tensormap_fenceproxy_acquire` + `ttng.reinterpret_tensor_descriptor`

### 关键机制

#### Load / Gather

`lowerTMALoad` 的固定模板是：

1. 分配 shared 落点 `ttg.local_alloc`
2. 分配 barrier memdesc
3. `ttng.init_barrier`
4. `ttng.barrier_expect`
5. 发起 `ttng.async_tma_*`
6. `ttng.wait_barrier`
7. `ttng.inval_barrier`
8. 再用 `ttg.local_load` 读回 tensor value

代码见：
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:22)

这说明 TMA load 在 Triton IR 里的 contract 不是“直接产出 tensor”，而是：

```text
async copy to shared
  -> barrier completion
  -> local_load to registers
```

#### Store / Reduce / Scatter

store 类路径先把 tensor materialize 到 shared，然后显式插：

- `ttng.fence_async_shared`
- `ttng.async_tma_copy_local_to_global` / `async_tma_reduce` / `async_tma_scatter`
- `ttng.async_tma_store_wait`

代码见：
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:95)

设计意图很明确：

- generic proxy 先把 shared 写好；
- 再用 fence 把 shared 对 async proxy 可见；
- 然后才允许 TMA store engine 读取。

#### `make_tensor_descriptor`

这里真正把抽象 tensordesc 变成设备侧 tensormap：

1. `ttg.global_scratch_alloc`
2. `ttng.tensormap_create`
3. `ttng.tensormap_fenceproxy_acquire`
4. `ttng.reinterpret_tensor_descriptor`

代码见：
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:167)

### Output contract

经过 `TMALowering` 后，可以认为：

- `tt.descriptor_*` 这层抽象已经消失；
- 剩下的是显式 async copy / barrier / fence 协议；
- 后续 pass 不再需要理解“抽象 descriptor 访存”，只需要理解 shared/TMA/mbarrier 依赖。

## 5.3 `FenceInsertion` 与 `ProxyFenceInsertion`

和 TMA 路径最相关的两个同步 pass 是：

- `fence_insertion`
  在 TTGIR 优化阶段插入较理想位置的 fence
- `proxy_fence_insertion`
  在 shared/TMEM 资源分配后，保守补齐 async proxy 和 generic proxy 的一致性

其中 `ProxyFenceInsertion` 源码在：
[ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:1)

### Goal

处理 Hopper+ 上 async proxy 和 generic proxy 分离带来的可见性问题。

源码注释已经把目标写得很直接：

- shared memory 对 async proxy 可见前，需要显式 fence
- pass 通过依赖分析保守插 fence，避免 proxy 间 race

### 关键判断

它把这些 op 视为 async proxy write：

- `TMALoadLikeOpInterface`

把这些 op 视为 async proxy read：

- `WarpGroupDotOp`
- `MMAv5OpInterface`
- `TMEMCopyOp`
- `TMAStoreLikeOpInterface`

也就是说，这个 pass 的重点不是普通 shared RAW/WAR，而是：

```text
generic shared producer
  -> async consumer

async producer
  -> generic shared consumer
```

### Output contract

这个 pass 之后，后续 LLVM lowering 可以假设：

- 需要跨 proxy 传播可见性的地方已经有 `ttng.fence_async_shared`；
- lowering 不需要自己重新推导高层 alias / proxy 依赖。

## 6. TMEM 路径

## 6.1 `PromoteLHSToTMem`

源码：
[PromoteLHSToTMem.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:1)

测试：
[test_promotion_to_tensor_memory.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir:1)

### Goal

把 Blackwell TCGen5 MMA 的 LHS operand 从 shared-memory path 提升到 TMEM operand path。

### Constraint

不是所有 LHS 都能提升。

当前源码显式要求：

- LHS 必须来自 `ttg.local_alloc`
- `local_alloc` 和 `tc_gen5_mma` 必须在同一个 region，限制 TMEM 生命周期
- element bit width 必须是 `8/16/32`
- 不能是 `fp4Padded`
- 原始 distributed layout 必须和 TMEM access 兼容；不兼容时才考虑插 `convert_layout`

### Decision

pass 实际上在回答：

```text
这个 tc_gen5_mma 的 A operand
是否值得且是否允许切到 TMEM？
```

如果兼容：

- 直接 `ttng.tmem_alloc` 出一个 A operand 的 TMEM memdesc

如果不兼容：

- 对来自 load / block-arg 的来源，默认更保守；若未放开环境变量，可能直接放弃提升
- 对其他来源，或者显式放开环境变量时，会先插 layout conversion 再分配 TMEM

### Output contract

这个 pass 之后，某些 `tc_gen5_mma` 已经不再从 shared 读 A，而是从 TMEM 读 A。

这会把后续问题全部改变：

- alias 分析对象变了
- 需要 TMEM allocation
- 需要 TMEM barrier
- 需要 TMEM-specific layout optimization

所以它不是普通 cleanup pass，而是 Blackwell MMA 数据路径的实质性分叉点。

## 6.2 `OptimizeTMemLayouts`

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:1)

### Goal

对 `tmem_load` / `tmem_store` 周围的 distributed layout 做局部重写，让后续 lowering 更贴近 TMEM 硬件约束和更便宜的访问模式。

### 它在做哪些 decision

#### 1. `split(reshape(trans(tmem_load)))` 改写成多个 `tmem_subslice + tmem_load`

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:60)

设计意图是：

- 与其先把一整块 TMEM load 出来，再做 reshape/trans/split，
- 不如直接对互不重叠的 N 子块分别 load。

这样建立的新 contract 是：

- 子块已经在 TMEM 边界被显式切开；
- 后续每个 use 只消费自己的 subslice。

#### 2. `join -> trans -> reshape -> tmem_store` 拆成多个 `tmem_store`

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:144)

这和上面的 split-load 是对偶关系：

- 不再先把上层 tensor join 成大块再一次性 store；
- 而是在 TMEM 边界上分别 store 到不同 N offset。

#### 3. 对 8 warps 且沿 N reduction 的 `tmem_load` 改用更适合按 M 分布的 layout

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:206)

这里的核心问题是：

- 多个 warpgroup 默认可能沿 N 分布；
- 但如果后面要沿 N reduction，这会引入更重的跨 warp 规约；
- 所以改成沿 M 分布更合适。

这就是典型的“根据 use pattern 回推 layout”。

#### 4. 对 shared -> tmem_store 选择更适合 local_load lowering 的 layout

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:258)

它会检查 backward slice，确认某些 local load 在替代 layout 下更容易矢量化，然后只改那类情况。

### Output contract

这个 pass 后，TMEM 相关 value 的 layout 更接近：

- TMEM 自身原子访问模式；
- 后续 local/shared lowering 的有利形状；
- 减少不必要的 cross-warp reduction / reshape cost。

## 6.3 `InterleaveTMem`

源码：
[InterleaveTMem.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp:1)

### Goal

把 `tmem_load` 尽量向它的 use 下沉，减少寄存器压力。

### Decision

这个 pass 不改变 TMEM 语义，不改 layout，不改 allocation。

它做的是调度层面的决定：

- 在不穿过 aliasing write/free/barrier 的前提下，
- 把 `tmem_load` 及其纯 use-chain 往后挪。

源码中的注释已经点明：

```text
Sink tmem_loads as close to their use as possible to reduce register pressure.
```

### Output contract

这个 pass 之后，可以假设：

- `tmem_load` 的活跃范围已经尽量短；
- 后续 LLVM lowering 更不容易把 TMEM read 提前太久，造成寄存器膨胀。

## 6.4 `TensorMemoryAllocation`

源码：
[TensorMemoryAllocation.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:1)

### Goal

给每个 `ttng.tmem_alloc` 分配真实的 tensor-memory row/column offset。

### Constraint

TMEM 不是无限线性空间。

当前实现显式建模了：

- 2 行 row-group
- 列方向不断扩展
- row allocation granularity = 64
- first-fit 分配
- 基于 liveness 的复用

### Decision

pass 在回答：

```text
这个 TMEM allocation
在物理 tensor memory 里放到哪一行哪一列？
哪些 alloc 可以共享同一片物理空间？
```

它通过：

- liveness interval
- subview / subslice / loop-carried / ws-capture 追踪
- coexisting chunk 冲突分析

来决定 offset。

### Output contract

这个 pass 后，`ttng.tmem_alloc` 上会带：

- `tensor_memory_col_offset`
- `tensor_memory_row_offset`

这一点可以直接从已有 dump 看到，例如：

- [sm100_num_ctas1 after allocation](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.log:26515)

这意味着从这个点开始，后续 pass 不再面对“抽象 TMEM object”，而是面对“已经放到物理 TMEM 地址空间里的对象”。

## 6.5 `TMemBarrierInsertion`

源码：
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:1)

### Goal

在 TMEM access 之间补齐正确的 CTA-level barrier。

### 它关心的依赖

pass 把 TMEM 访问分成三类：

- `Load`
- `Store`
- `MMA`

并把这些依赖视为必须 barrier：

- WAR
- RAW
- WAW
- `load -> mma`
- `store -> mma`

源码见：
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:45)

特别值得注意的一点：

- `mma -> load/store` 不要求这里再插 barrier，
- 因为设计上依赖的是后续 mbarrier wait 语义。

这说明它不是“凡是接触 TMEM 都 barrier”，而是在遵守 TCGen5/TMEM 完成语义。

### 为什么它必须在 allocation 之后

因为它的 slice 分析直接依赖：

- `tensor_memory_col_offset`
- `tensor_memory_row_offset`

只有知道物理 row/col 后，才能判断两个 TMEM access 是否真的 alias。

所以 `TensorMemoryAllocation` 是 `TMemBarrierInsertion` 的前置 contract。

## 7. Hopper 与 Blackwell 的分叉点

## 7.1 共同部分

两代都共享：

- `OptimizeDescriptorEncoding`
- `TMALowering`
- `FenceInsertion`
- `ProxyFenceInsertion`

也就是说，“descriptor -> TMA async shared path”是共享的。

## 7.2 Hopper 到此为止

在 `sm90` dump 里，主线仍是：

- `tt.descriptor_*`
  -> TMA load/store
  -> shared
  -> WGMMA / warp-group path

从现有 `sm90_num_ctas1` dump 看，能清楚看到：

- `OptimizeDescriptorEncoding`
- `TMALowering`
- `fence_async_shared`

但没有 `ttng.tmem_alloc` 作为主 accumulator 路径。

## 7.3 Blackwell 继续下沉到 TMEM

在 `sm100` dump 里，`tc_gen5_mma` 周围已经能看到：

- `ttng.tmem_alloc`
- `ttng.tmem_store`
- `ttng.tmem_load`
- `ttng.wait_barrier`
- 后续 allocation pass 写入 row/col offset

这说明 Blackwell 把一部分 tensor-core 数据流从“shared + warpgroup MMA”改成了“TMEM + TCGen5 MMA”。

所以从设计意图上看：

- Hopper 重点是 async shared transport。
- Blackwell 不只是 transport，还把 tensor-core compute state 本身搬进 TMEM。

## 8. 读 dump 时应重点观察什么

建议按下面顺序看 `sm90_num_ctas1` 和 `sm100_num_ctas1` / `sm100_num_ctas2`：

1. `OptimizeDescriptorEncoding` 前后：
   看 `!tt.tensordesc<...>` 是否开始带 `#ttg.nvmma_shared<...>`。
2. `TMALowering` 前后：
   看 `tt.descriptor_*` 是否消失，是否变成 `ttng.async_tma_*`、`ttg.local_alloc`、`ttng.wait_barrier`。
3. `PromoteLHSToTMem` 之后（Blackwell）：
   看 `tc_gen5_mma` 的 A operand 是否从 shared 变成 TMEM memdesc。
4. `OptimizeTMemLayouts` 前后：
   看 `tmem_load/tmem_store` 周围是否出现 `tmem_subslice`、新的 `convert_layout`，或 result encoding 变化。
5. `InterleaveTMem` 前后：
   看 `tmem_load` 是否被下沉到更靠近 use 的位置。
6. `TensorMemoryAllocation` 前后：
   看 `ttng.tmem_alloc` 是否新增 row/col offset attr。
7. `TMemBarrierInsertion` 前后：
   看 TMEM access 之间是否新增 `ttg.barrier`。

## 9. 最小总结

如果只记一条主线，可以记成：

```text
TMA path:
  OptimizeDescriptorEncoding
    -> TMALowering
    -> Fence / ProxyFence

TMEM path (Blackwell):
  PromoteLHSToTMem
    -> OptimizeTMemLayouts
    -> InterleaveTMem
    -> TensorMemoryAllocation
    -> TMemBarrierInsertion
```

更抽象一点说：

- TMA 这条线在解决“descriptor async transport contract”。
- TMEM 这条线在解决“tensor-core state placement, layout, allocation, synchronization contract”。

## 10. 后续建议

下一步最值得单独展开的两个主题是：

1. `TMALowering` 的 before/after diff
   因为它是“抽象 descriptor 语义”变成“真实 async protocol”的分界点。
2. `TensorMemoryAllocation` + `TMemBarrierInsertion`
   因为这是 Blackwell TMEM 路径里最直接体现“资源分配 + 正确性同步”的组合。
