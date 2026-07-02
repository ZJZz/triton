# 2026-07-01 学习笔记：TMEM 相关 pass

## 1. 这篇笔记只回答什么问题

这篇只看 Blackwell `TMEM` 主线：

```text
shared/local-side tensor
  -> 是否提升为 TMEM operand
  -> TMEM 边界 layout 如何重写
  -> tmem_load 何时下沉
  -> 抽象 TMEM object 何时分配物理 row/col
  -> TMEM 访问何时补 barrier
```

这里讨论的是 Blackwell tensor-core compute-state path，不是 `TMA` 的 global/shared transport path。

这篇真正要回答的是五件事：

1. 哪些 value 会被切到 TMEM？
2. 进入 TMEM 后，边界 layout 如何围绕实际 use pattern 重写？
3. `tmem_load` 何时 materialize 到寄存器更合适？
4. 抽象 `ttng.tmem_alloc` 何时映射到真实 row/col？
5. 知道真实物理地址后，哪些 TMEM 依赖要显式 barrier？

## 2. TMEM 的边界先说清楚

`TMEM` 是 Blackwell tensor-core 路径上的 tensor memory。当前主题是：

- `ttng.tmem_alloc`
- `ttng.tmem_load`
- `ttng.tmem_store`
- `tc_gen5_mma` 周围的 placement / layout / scheduling / allocation / barrier

它和 `TMA` 的关系是：

- `TMA` 主要负责 `global <-> shared`
- `TMEM` 主要负责 Blackwell tensor-core operand / accumulator 驻留在什么 memory space，以及如何布局、分配、同步
- TMEM path 往往从已经 materialize 成 shared/local-side object 的值继续下沉，而不是直接从 descriptor 进入 TMEM

## 3. Pipeline 位置

当前 NVIDIA backend 中，与 TMEM 直接相关的顺序在 [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:297)：

```text
make_ttgir:
  if sm100+:
    hoist_tmem_alloc
    promote_lhs_to_tmem
    ...
    hoist_tmem_alloc
    remove_tmem_tokens
  ...
  optimize_tmem_layouts
  ...
  interleave_tmem
  ...
  fence_insertion
  lower_mma

make_llir:
  allocate_tensor_memory
  ...
  proxy_fence_insertion
  tmem_barrier_insertion
  to_llvmir
```

对应位置：

- `promote_lhs_to_tmem`:
  [compiler.py:298](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:298)
- `optimize_tmem_layouts`:
  [compiler.py:316](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:316)
- `interleave_tmem`:
  [compiler.py:320](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:320)
- `allocate_tensor_memory`:
  [compiler.py:385](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:385)
- `proxy_fence_insertion`:
  [compiler.py:390](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:390)
- `tmem_barrier_insertion`:
  [compiler.py:391](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:391)

这条线的分层很清楚：

1. `make_ttgir` 先决定值要不要进 TMEM，以及进 TMEM 后的高层 layout / scheduling。
2. `make_llir` 再把抽象 TMEM object 分配到真实 row/col，并按真实 alias 关系补 barrier。

## 4. 一句话心智模型

TMEM 这条线可以压成五个 compiler question：

```text
placement
  -> layout
  -> scheduling
  -> allocation
  -> synchronization
```

它们回答的是 Blackwell tensor-core state 怎么驻留、怎么被访问、怎么保持顺序，而不是普通 shared path 的整理。

## 5. 各 pass 在做什么

## 5.1 `PromoteLHSToTMem`

源码：
[PromoteLHSToTMem.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:37)

测试：
[test_promotion_to_tensor_memory.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonNvidiaGPU/test_promotion_to_tensor_memory.mlir:1)

### Problem

进入这个 pass 时，`tc_gen5_mma` 的 LHS 还是 shared/local-side producer，IR 里还没有决定它是继续沿传统 shared operand path 前进，还是切到 Blackwell 专有的 TMEM-resident operand path。

这里之所以是问题边界，是因为从更上游看，`local_alloc` 还只是一般 shared/local materialization，编译器仍然可以不表态“这个值以后是不是 TMEM operand”；但从这里往下，后续 `OptimizeTMemLayouts`、`InterleaveTMem`、`TensorMemoryAllocation`、`TMemBarrierInsertion` 都只会处理已经进入 TMEM 的对象。如果不在这里做分叉，TMEM 路径根本不会出现，后面也无从优化、分配或同步。

### Goal

把满足条件的 Blackwell `TCGen5 MMA` LHS operand 从 shared/local path 提升到 TMEM path。

### Constraint

source:

- TMEM 不是通用 memory space，而是绑定到 Blackwell TCGen5 tensor-core 数据路径的专用资源
- 当前 pass 接到的输入不是一张白纸，而是已经 materialize 成 `ttg.local_alloc` 的 shared/local-side object
- A operand 的位宽、padding 和 distributed layout 还要满足 TMEM 访问路径的硬约束

manifestation:

- 不能把任意 A operand 都推进 TMEM
- 当前实现显式要求：
  - LHS 必须来自 `ttg.local_alloc`
  - `local_alloc` 和 `tc_gen5_mma` 必须在同一个 region
  - element bit width 必须属于 `{8, 16, 32}`
  - 不能是 `fp4Padded`
  - 原始 distributed layout 必须与 TMEM access 兼容；不兼容时才考虑插 `convert_layout`
- 对来自 load / block-arg 的来源更保守：只有满足条件时才允许 layout conversion

关键代码：

- `local_alloc` / region / 位宽 / fp4 限制：
  [PromoteLHSToTMem.cpp:37](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:37)
  [PromoteLHSToTMem.cpp:53](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:53)
- 布局兼容与保守 conversion：
  [PromoteLHSToTMem.cpp:70](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:70)
  [PromoteLHSToTMem.cpp:75](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:75)

### Design intent

把 TMEM promotion 做成一个尽量小的路径分叉点：满足条件时切到 TMEM，不满足时保持原有 shared/local path，必要时只插最小的 `convert_layout`。这样既建立 Blackwell 专有 operand path，又不把一般 shared path 的已有语义推翻。

### Decision

这个 pass 实际回答的是：

```text
这个 tc_gen5_mma 的 A operand
是否允许并值得切到 TMEM？
如果允许，需不需要先把布局改成 TMEM 可接受的形状？
```

### Output contract

这个 pass 之后，某些 `tc_gen5_mma` 的 A operand 不再从 shared 读取，而是改成 `ttng.tmem_alloc` 所代表的 TMEM object。后续 TMEM passes 可以只围绕这批已经进入 TMEM 的值继续工作。

## 5.2 `OptimizeTMemLayouts`

源码：
[OptimizeTMemLayouts.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:60)

### Problem

进入这个 pass 时，TMEM path 已经合法存在，但 surrounding layout 仍然大多反映一般 distributed tensor 组织，而不是 TMEM 边界最便宜的访问形状。也就是说，“能用 TMEM”这件事已经成立，但“进了 TMEM 以后该怎样组织边界布局”这件事还没有被回答。

这里之所以是问题边界，是因为 placement 已经在 `PromoteLHSToTMem` 确定了，编译器终于知道哪些值属于 TMEM path；但物理 row/col allocation 还没有发生，所以这里最适合围绕 use pattern 去重写高层 layout。如果拖到 allocation 之后再做，很多 shape/reshape/split/join 代价已经固化进更低层对象里；如果更早做，又还不知道哪些值真的会进入 TMEM。

### Goal

围绕 `tmem_load` / `tmem_store` 的使用模式，重写 distributed layout，让 TMEM 边界更贴近硬件访问和后续 lowering 的实际成本。

### Constraint

source:

- 上游 layout 首先服务一般 tensor/thread 映射，而不是专门为 TMEM 边界定制
- 不同 use pattern 对 TMEM 边界布局的偏好不同，不存在一条对所有场景都好的“全局最优布局”
- 这些偏好必须结合具体用户来判断，不能只看孤立的 `tmem_load` / `tmem_store`

manifestation:

- 不能用单一规则重排所有 TMEM 边界布局
- 当前实现按局部 use pattern 做四类重写：
  - split-load
  - store-join
  - reduction-aware tmem load layout
  - shared -> TMEM store layout

关键 pattern 入口：

- `TMemSplitLoadPattern`:
  [OptimizeTMemLayouts.cpp:60](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:60)
- `TMemStoreJoinPattern`:
  [OptimizeTMemLayouts.cpp:144](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:144)
- `TMemLoadReducePattern`:
  [OptimizeTMemLayouts.cpp:206](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:206)
- `TMemFromSharedMemPattern`:
  [OptimizeTMemLayouts.cpp:258](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:258)

### Design intent

不试图定义一个“TMEM 通用最优布局”，而是围绕几类代价最明确的 use pattern 做局部重写，把 split/join/trans/reshape 成本尽量压回 TMEM 边界本身更便宜的原子访问形状。

### Decision

这个 pass 实际回答的是：

```text
面对当前这个 TMEM use pattern
应不应该把边界 layout 改写成另一种更便宜的局部形状？
```

### 四类核心重写

#### 1. split-load

把 `split(reshape(trans(tmem_load)))` 改写成多个 `tmem_subslice + tmem_load`。

这意味着：

- 不先把大块 TMEM load 出来再切
- 而是在 TMEM 边界直接 materialize 独立 subslice

#### 2. store-join

把 `join -> trans -> reshape -> tmem_store` 改写成多个 `tmem_store`。

这和 split-load 是对偶关系：

- 不先 join 成大块再一次性 store
- 而是在 TMEM 边界分别 store 到不同偏移

#### 3. reduction-aware tmem load layout

对 8 warps 且沿 N reduction 的 `tmem_load`，改用更适合沿 M 分布的 layout。

注释见：
[OptimizeTMemLayouts.cpp:203](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp:203)

它解决的是：

- 默认分布可能让跨 warp 规约沿 N 方向更重
- 如果后面确实沿 N reduction，就提前把布局改成更利于沿 M 分布的形状

#### 4. shared -> TMEM store 布局

对 shared -> `tmem_store` 选择更适合 local-load lowering 的布局。

它会看 backward slice，只在替代 layout 真能改善后续 lowering 时才改。

### Output contract

这个 pass 之后，TMEM 相关 value 的 layout 更接近：

- TMEM 自身访问的原子形状
- 后续 local/shared lowering 更便宜的形状
- 更少的跨 warp reduction / reshape / join / split 成本

## 5.3 `InterleaveTMem`

源码：
[InterleaveTMem.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp:194)

### Problem

进入这个 pass 时，TMEM path 和周围 layout 已经基本稳定，但 `tmem_load` 的程序位置还未必反映“什么时候才真的需要把 TMEM value materialize 到寄存器”。换句话说，前面的 pass 解决了路径和布局，还没有解决 live range。

这里之所以是问题边界，是因为它正处在高层布局已经稳定、物理 allocation 尚未介入的位置：此时 pass 可以把问题专注为“调度上的 load 下沉”，不用再同时处理 ownership 或地址分配；如果更早做，layout 还没定，load 形状本身可能继续变化；如果更晚做，真实 alias / barrier 边界已经引入，会把一个本来是 live-range 优化的问题混成同步问题。

### Goal

把 `tmem_load` 尽量下沉到 use 附近，缩短寄存器 live range，降低寄存器压力。

### Constraint

source:

- `tmem_load` 的价值最终体现在寄存器消费者上，所以它天然受 live-range 成本约束
- 但 TMEM value 的可读性又受 aliasing write、free、barrier 等边界限制，load 不能任意漂移

manifestation:

- 这个 pass 只能把 load 后移到安全边界以内，不能穿过会改变值可读性的 op
- 为了保持职责单一，它也不能顺手去改 layout、allocation 或 ownership

### Design intent

把 `tmem_load` 视为一种应尽量延后的 value materialization，而不是越早发起越好。这个 pass 只做调度层面的 live-range 收缩。

### Decision

这个 pass 实际回答的是：

```text
这个 tmem_load
在不跨越别名写入、free 或 barrier 的前提下
能否更靠近它的真实 use？
```

源码注释原话就是：

```text
Sink tmem_loads as close to their use as possible to reduce register pressure.
```

### Output contract

这个 pass 之后，可以假设：

- `tmem_load` 的活跃范围已经尽量短
- 后续 lowering 更不容易因为过早读取 TMEM 而造成寄存器膨胀

## 5.4 `TensorMemoryAllocation`

源码：
[TensorMemoryAllocation.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:28)

### Problem

进入这个 pass 时，IR 已经知道哪些值属于 TMEM，也已经知道它们周围的大致 layout / use pattern，但还不知道这些抽象 `ttng.tmem_alloc` 在物理 TMEM 里到底落在哪一行哪一列。也就是说，前面的 TMEM path 还是“抽象 ownership 成立”，还不是“物理地址已定”。

这里之所以是问题边界，是因为只有当前面的 placement、layout 和 scheduling 都基本稳定后，编译器才能把 TMEM object 的生命周期和重叠关系看全，进而做统一物理分配；如果更早分配，后续布局改写和 subslice 关系可能改掉资源需求；如果更晚不分配，`TMemBarrierInsertion` 和更低层 lowering 都拿不到真实物理地址。

### Goal

给每个 `ttng.tmem_alloc` 分配真实的 tensor-memory row / column offset。

### Constraint

source:

- Blackwell TMEM 不是抽象无限空间，而是有 row/column 组织和固定分配粒度的物理资源
- 是否能复用同一物理区间，取决于 alloc 之间真实的生命周期冲突
- 更早 passes 已经把 subslice、loop-carried value、warp-specialize capture 等关系编码进 IR，当前 pass 必须理解这些关系

manifestation:

- 不能把 TMEM 当成普通线性地址空间顺序分配
- 当前实现显式建模了：
  - `allocGranularity = 64`
  - `kNumRows = 2`
  - first-fit 线性扫描
  - 基于 liveness 的复用
- 分配结果最终写回：
  - `tensor_memory_col_offset`
  - `tensor_memory_row_offset`

关键代码：

- 资源粒度与行数：
  [TensorMemoryAllocation.cpp:28](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:28)
  [TensorMemoryAllocation.cpp:115](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:115)
- first-fit：
  [TensorMemoryAllocation.cpp:337](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:337)
- 写回 offset：
  [TensorMemoryAllocation.cpp:377](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:377)
  [TensorMemoryAllocation.cpp:381](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp:381)

### Design intent

把 TMEM allocation 留到较晚阶段统一处理：前面的 passes 可以自由建立抽象 TMEM object 和 use pattern，等这些关系稳定后，再一次性根据真实生命周期和物理粒度做 row/col 分配与复用。

### Decision

这个 pass 实际回答的是：

```text
这个抽象 TMEM allocation
在物理 tensor memory 里放到哪一行哪一列？
哪些 alloc 可以共享同一片物理空间？
```

### Output contract

这个 pass 之后，`ttng.tmem_alloc` 不再只是抽象 TMEM object，而是已经带着真实物理位置的 TMEM object。后续 barrier 插入和 lowering 可以直接依赖 row / col offset。

## 5.5 `TMemBarrierInsertion`

源码：
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:59)

### Problem

进入这个 pass 时，TMEM access 已经有了真实 row/col，因此哪些 op 在物理 TMEM 上会撞到一起终于可见了；但“地址会重叠”还没有自动变成“IR 上顺序已经正确”。前面几步建立的是 placement、layout、live range 和物理地址，还没有最终补齐依赖顺序。

这里之所以是问题边界，是因为 barrier 的必要性依赖真实 alias 关系，而这个关系只有 `TensorMemoryAllocation` 之后才完整出现。更早做只能凭抽象 TMEM object 猜，既可能漏掉真正冲突，也可能过度同步；更晚再做，LLVM lowering 已经默认这些顺序义务在高层 IR 中被表达好了。

### Goal

在 TMEM access 之间补齐正确的 CTA-level barrier。

### Constraint

source:

- alias 事实只有 allocation 之后才完整可见
- TMEM consumer 不只有纯 load/store，还有 MMA 这种读写混合操作
- 一部分顺序义务已经由后续 `mbarrier wait` contract 承担，不能在这里重复承担一遍

manifestation:

- barrier 插入必须建立在真实 row/col alias 之上
- 当前实现把下面这些依赖视为必须 barrier：
  - WAR
  - RAW
  - WAW
  - `load -> mma`
  - `store -> mma`
- 但 `mma -> load/store` 不需要这里再插 barrier，因为设计上依赖后续 `mbarrier wait`

关键代码：

- 依赖分类与规则：
  [TMemBarrierInsertion.cpp:59](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:59)
  [TMemBarrierInsertion.cpp:66](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:66)
- 读取 row/col offset：
  [TMemBarrierInsertion.cpp:181](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:181)
  [TMemBarrierInsertion.cpp:183](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:183)

### Design intent

把 TMEM barrier 建立在“真实物理 alias + 访问类别”之上，而不是只按 op 名字一刀切。这样既补齐必要顺序，又避免把已经由 `mbarrier wait` 覆盖的方向重复同步。

### Decision

这个 pass 实际回答的是：

```text
这两个 TMEM access
在真实物理 row/col 上是否相交？
如果相交，它们的访问方向是否属于必须显式 barrier 的那一类？
```

### Output contract

这个 pass 之后，后续 lowering 可以假设：

- 对会在物理 TMEM 上冲突、且未被其他协议覆盖的依赖，IR 中已经有 CTA-level barrier
- LLVM/NVVM lowering 不需要重新推导 TMEM alias 顺序

## 6. Hopper 与 Blackwell 的分叉点

Hopper (`sm90`) 没有这条“accumulator / operand 常驻 TMEM”的主路径。

Blackwell (`sm100`) 才会稳定看到：

- `ttng.tmem_alloc`
- `ttng.tmem_store`
- `ttng.tmem_load`
- allocation 后写入的 row / col offset

所以更准确地说：

- Hopper 重点是 async shared transport 和 warp-group compute path
- Blackwell 还要进一步回答 tensor-core state 是否进 TMEM、怎么布局、怎么分配、怎么同步

## 7. 这条线为什么不是普通 cleanup

这些 pass 建立的是后续 lowering 必须消费的硬约束，而不是“换一种更好看的 IR”：

- 哪个 operand 驻留在 TMEM
- TMEM value 用什么 distributed layout
- `tmem_load` 活跃范围多长
- TMEM object 的物理 row / col 是多少
- 哪些 TMEM 依赖已经有 barrier

没有这些 contract，后面的 LLVM/NVVM lowering 不能只靠局部模式匹配恢复正确资源和同步语义。

## 8. 读 dump 时重点看什么

建议按下面顺序读 `sm100` dump：

1. `PromoteLHSToTMem` 之后：
   看 `tc_gen5_mma` 的 A operand 是否从 shared/local path 变成 `ttng.tmem_alloc`。
2. `OptimizeTMemLayouts` 前后：
   看 `tmem_load/tmem_store` 周围是否出现 `tmem_subslice`、新的 `convert_layout` 或 encoding 变化。
3. `InterleaveTMem` 前后：
   看 `tmem_load` 是否更靠近真实 use。
4. `TensorMemoryAllocation` 前后：
   看 `ttng.tmem_alloc` 是否新增 `tensor_memory_col_offset` / `tensor_memory_row_offset`。
5. `TMemBarrierInsertion` 前后：
   看 TMEM access 之间是否新增 `ttg.barrier`。

## 9. TMEM 和 TMA 在哪里相连

### 9.1 TMA 不是 TMEM 的同义词

- `TMA` 主要解决 `global <-> shared`
- `TMEM` 主要解决 `tensor-core state <-> tensor memory`

所以不要把 “用了 TMA” 理解成 “数据已经在 TMEM 里”。

### 9.2 TMEM path 通常从 shared/local 侧对象继续下沉

[PromoteLHSToTMem.cpp:37](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp:37)
第一条硬条件就是：LHS 必须来自 `ttg.local_alloc`。

这说明当前实现里，TMEM promotion 的输入通常已经是 shared/local-side materialized object，而不是直接从 descriptor 进入 TMEM。

常见链路更接近：

```text
global
  --(load 或 TMA)--> shared/local object
  --(PromoteLHSToTMem)--> TMEM operand
  --(TCGen5 MMA)--> TMEM resident state
```

### 9.3 它们在 proxy visibility 上会相遇

[ProxyFenceInsertion.cpp:48](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:48)
把 `MMAv5OpInterface`、`TMEMCopyOp`、`TMAStoreLikeOpInterface` 视为 async proxy read；而 [ProxyFenceInsertion.cpp:33](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:33) 把 `TMALoadLikeOpInterface` 视为 async proxy write。

这说明 TMA transport 和后续 TMEM / MMA 消费共享同一套 Hopper+/Blackwell proxy memory model。

## 10. 最小总结

如果只记一条主线，可以记成：

```text
TMEM path (Blackwell):
  PromoteLHSToTMem
    -> OptimizeTMemLayouts
    -> InterleaveTMem
    -> TensorMemoryAllocation
    -> TMemBarrierInsertion
```

更抽象一点：

```text
TMEM 解决的是
placement
  -> layout
  -> scheduling
  -> allocation
  -> synchronization
```
