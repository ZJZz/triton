# SM100 编译优化总览

## 1. 范围

这篇文档只讨论当前仓库里 NVIDIA backend 针对 `capability >= 100` 的
SM100/Blackwell 路径。

这里说的“编译优化”不只包含传统意义上的 performance tuning，也包含：

- 为 `tcgen05` / `TMEM` / `TMA` 建立 target-specific execution contract
- 为 warp specialization 建立 producer/consumer 分区和寄存器分配策略
- 为 async proxy / cluster / TMEM reuse 补合法性修复

如果只看结论，SM100 这条线的核心目标可以压成一句话：

```text
把 generic Triton matmul / descriptor / loop IR
  变成 tcgen05 + TMEM + TMA + warp specialization 可执行协议，
  同时压低寄存器压力，并补齐所有 async / cluster / reuse 同步约束。
```

主入口在：

- `third_party/nvidia/backend/compiler.py`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`

---

## 2. 先按源码看实际 pipeline

SM100 的关键不是某一个 pass，而是 `make_ttgir` 和 `make_llir` 两段
pipeline 一起成立。

### 2.1 `make_ttir`

旧架构会先把 tensor descriptor 改写成 pointer；Hopper/Blackwell 不会。

- `capability // 10 < 9` 才跑
  `rewrite_tensor_descriptor_to_pointer`
- 所以 `sm100` 会保留 descriptor，给后面的 `TMA lowering` 留出空间

这一步很重要，因为一旦过早退化成 raw pointer，后面就不可能再做
descriptor/TMA 路径优化。

### 2.2 `make_ttgir`

下面按 `third_party/nvidia/backend/compiler.py` 里 `capability >= 100`
的当前实际顺序展开；为了可读性，只省掉了 `capability == 8` 的专用分支和
`fpsan` instrumentation 分支。

```text
convert_to_ttgpuir
  -> coalesce
  -> f32_dot_tc
  -> plan_cta
  -> remove_layout_conversions
  -> optimize_thread_locality
  -> accelerate_matmul
  -> remove_layout_conversions
  -> optimize_dot_operands
  -> optimize_descriptor_encoding
  -> loop_aware_cse
  -> capability >= 100 专用分支：
       fuse_nested_loops
       canonicalizer
       triton_licm
       optimize_accumulator_init
       hoist_tmem_alloc(false)
       promote_lhs_to_tmem
       assign_latencies
       schedule_loops
       warp_specialize
       pipeline
       optimize_partition_warps
       combine_tensor_select_and_if
       hoist_tmem_alloc(true)
       remove_tmem_tokens
  -> canonicalizer
  -> loop_aware_cse
  -> optimize_dot_operands
  -> coalesce_async_copy
  -> optimize_tmem_layouts
  -> tma_lowering
  -> remove_layout_conversions
  -> interleave_tmem
  -> reduce_data_duplication / reorder_instructions
  -> loop_aware_cse / symbol_dce
  -> fence_insertion
  -> lower_mma
  -> sccp / cse / canonicalizer
```

这里的 `lower_mma` 就是 Python 绑定里的 `add_lower_mma`；它对应的实现文件是
`lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp`。所以本文后面写
`mma_lowering` 时，说的是同一个 pass，只是命名角度不同。

### 2.3 `make_llir`

LLVM lowering 前后还有一串对 SM100 很关键的资源分配和 legality repair：

```text
allocate_shared_memory_nv
  -> allocate_tensor_memory
  -> check_matmul_two_cta
  -> proxy_fence_insertion
  -> tmem_barrier_insertion
  -> convert-triton-gpu-to-llvm
       (内部还会做 cluster barrier insertion / cross-CTA mbarrier init sync)
  -> initialize_ws_cluster_barriers
  -> warp_specialize_to_llvm
  -> nvgpu_to_llvm
```

---

## 3. 可以把 SM100 优化分成五组

| 组 | 关键对象 | 主要 pass / lowering |
|---|---|---|
| matmul target 选择 | `tt.dot -> tcgen05` | `accelerate_matmul`, `plan_cta`, `check_matmul_two_cta` |
| descriptor / TMA | `tt.descriptor_*` | 保留 descriptor、`optimize_descriptor_encoding`, `tma_lowering` |
| TMEM compute-state | accumulator / operand residence | `optimize_accumulator_init`, `hoist_tmem_alloc`, `promote_lhs_to_tmem`, `optimize_tmem_layouts`, `interleave_tmem`, `tensor_memory_allocation`, `tmem_barrier_insertion`, `mma_lowering` |
| warp specialization | producer/consumer overlap | `assign_latencies`, `schedule_loops`, `warp_specialize`, `optimize_partition_warps`, `warp_specialize_to_llvm` |
| legality / sync repair | async proxy / cluster / reuse | `fence_insertion`, `proxy_fence_insertion`, cluster barrier insertion, `initialize_ws_cluster_barriers` |

### 3.1 如果改按 `TTGIR_GUIDE.md` 的五类来读

上面五组是“从 SM100 想建成什么协议”来分；如果改成 TTGIR 的 pass 职责语言，
同一条 pipeline 还可以再归位成：

| TTGIR 类别 | 在 SM100 主线里的代表 pass | 读法重点 |
|---|---|---|
| distributed execution mapping | `convert_to_ttgpuir`, `plan_cta` | 谁拥有 tile、几个 CTA 合作、CGA contract 何时固定 |
| layout / data-movement organization | `coalesce`, `optimize_thread_locality`, `accelerate_matmul`, `optimize_dot_operands`, `optimize_descriptor_encoding`, `promote_lhs_to_tmem`, `optimize_tmem_layouts` | 值以什么 layout / carrier / residence 交给 consumer |
| target-driven scheduling | `fuse_nested_loops`, `assign_latencies`, `schedule_loops`, `warp_specialize`, `pipeline`, `tma_lowering`, `mma_lowering`, `optimize_partition_warps` | 何时 overlap、何时 wait、何时进入 target-specific async protocol |
| legality repair | `fence_insertion`, `check_matmul_two_cta`, `proxy_fence_insertion`, `tmem_barrier_insertion`, cluster barrier insertion | 哪个 target hazard 或一致性条件还没满足 |
| cleanup | `remove_layout_conversions`, `loop_aware_cse`, `canonicalizer`, `hoist_tmem_alloc`, `combine_tensor_select_and_if`, `coalesce_async_copy`, `interleave_tmem`, `remove_tmem_tokens`, `reduce_data_duplication`, `reorder_instructions`, `symbol_dce` | 收敛表示噪音，不重新定义主 contract |

这两个视角不要混：

- 第 3 节的五组回答“SM100 最终要建哪些大协议”
- 这里的五类回答“每个 pass 在 TTGIR / make_llir 边界上主要负责什么”

---

## 4. 逐项看每类优化在做什么

下面每个小节尽量回答 4 个问题：

- 它主要属于 `mapping / organization / scheduling / legality / cleanup` 哪一类
- 输入时 IR 里已经有什么 contract
- 它新增或收紧了什么 contract
- 这份 contract 后面由谁继续消费

同一个 pass 可能跨两类；这里按“主责任”归类。这里说的 `下游 consumer`
也不一定是紧邻的下一个 pass，有时会直接指向后段 lowering。

### 4.1 让 matmul 真正走 Blackwell path

#### `accelerate_matmul`

关键文件：

- `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`

- `归类：` 以 compute consumer 为中心的 organization pass，但会顺带建立一部分
  MMAv5/TMEM protocol 脚手架。
- `输入 contract：` `plan_cta` 已经把 tile 的 CTA 合作关系定下来；dot 还是
  generic tensor compute，不带 Blackwell-specific residence 决策。
- `输出 contract：` dot 的 consumer 被收紧成 MMAv5 路径，A/B operand 的 carrier
  和 accumulator 的 residence 都不再抽象。
- `下游 consumer：` `optimize_dot_operands`、`promote_lhs_to_tmem`、
  `mma_lowering`、`tensor_memory_allocation`、`MMAv5.cpp`。

它负责把 generic `tt.dot` 变成适合目标架构的 tensor-core path。对
SM100 来说，最重要的是：

- 当 `getMMAVersionSafe(...)` 选到 version 5 时，`tt.dot` 会改写成
  `ttng.tc_gen5_mma` / `ttng.tc_gen5_mma_scaled`
- A/B operand 被 materialize 成 shared-memory memdesc
- accumulator 不再只是普通 distributed tensor，而是进入
  `ttng.tmem_alloc -> ttng.tc_gen5_mma -> ttng.tmem_load`

这一步建立的不是“最后指令”，而是 Blackwell compute-state contract：

- operand 住在 shared/TMEM 哪边
- accumulator 住在 TMEM
- dot 结果通过 async token 和 barrier 协议交接

#### `plan_cta`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/PlanCTA.cpp`

- `归类：` distributed execution mapping。
- `输入 contract：` TTGIR 已经有基本 execution encoding，但“一个 tile 由几个
  CTA 合作”这件事还没完全固定。
- `输出 contract：` `DotOp` / `ReduceOp` / StoreLike op 的 CTA tiling 和
  CGA-related distributed encoding 被定下来。
- `下游 consumer：` `accelerate_matmul`、`optimize_descriptor_encoding`、
  cluster barrier insertion、`check_matmul_two_cta`。

作用：

- 给 `DotOp` / `ReduceOp` / StoreLike op 计算 CTA tiling
- 把 CGA layout 传播到相关 distributed encoding 上

它的意义不只是多 CTA 切 tile，而是先把“一个 tile 由几个 CTA 合作”这个
contract 固定下来。后面的 shared layout、descriptor encoding、cluster
barrier、`two_ctas` lowering 都依赖这个决定。

它不负责决定 operand 应该住在 shared、descriptor 还是 TMEM；那是后面
organization pass 的责任。

#### `check_matmul_two_cta`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp`

- `归类：` 更接近 lowering-side legality / consistency check，而不是新的
  mapping 或 scheduling 决策。
- `输入 contract：` MMAv5 op 已经生成，kernel 内各处 matmul 的 CTA mode
  已经是既成事实。
- `输出 contract：` module 级别得到一个统一的 `two_ctas` invariant，后续
  lowering 不需要在多个 op 之间重复做一致性推断。
- `下游 consumer：` `convert-triton-gpu-to-llvm` 之后的 MMAv5/PTX lowering。

作用：

- 检查所有 MMAv5 op 的 `two_ctas` 设置是否一致
- 把结果写成 module attr，供后续 lowering 读取

当前仓库里 `BlockedToMMAv5` 仍然把 generic dot path 的 `useTwoCTAs`
写死为 `false`，原因是代码里明确注释了 PTX 13+ 目前要求 kernel 内
`tcgen` CTA mode 一致，暂时没有重新打开自动 2CTA。

所以要区分两件事：

- `two_ctas` 基础设施是存在的
- 但 generic `tt.dot -> MMAv5` 这条 canonical path 目前并没有自动启用它

### 4.2 保留 descriptor，并把它真的降成 TMA 协议

#### 保留 descriptor

关键文件：

- `third_party/nvidia/backend/compiler.py`

- `归类：` 这是 TMA 路径的 organization 前置条件，不是独立算子级优化。
- `输入 contract：` TTIR 里仍有 descriptor 级语义，尚未退化成 raw pointer。
- `输出 contract：` `make_ttgir` 还能继续看到 `tt.descriptor_*`，后面才可能
 生成真正的 descriptor/TMA transport path。
- `下游 consumer：` `optimize_descriptor_encoding`、`tma_lowering`。

SM100 不会在 `make_ttir` 阶段把 descriptor 提前改成 pointer。这本质上是
一条“不要过早失去信息”的优化：

- 目标不是尽快变简单
- 目标是保留足够的结构，让后续能够生成真正的 TMA transport path

#### `optimize_descriptor_encoding`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp`

- `归类：` descriptor consumer-facing organization。
- `输入 contract：` descriptor 语义仍在，`plan_cta` 已固定 CTA 组织方式，
  当前 shared layout 还不一定落在 TMA descriptor 支持子集里。
- `输出 contract：` descriptor 对应的 shared encoding 被收敛到
  hardware-compatible form，后面的 TMA lowering 不必再猜布局。
- `下游 consumer：` `tma_lowering`、后续 proxy/fence legality 分析、TMA
  相关 LLVM lowering。

作用：

- 给 tensor descriptor 选 shared-memory encoding
- 优先选 non-transposed 的 `NVMMASharedEncodingAttr`
- 要求和原先 shared-linear layout 等价，但要落在 TMA descriptor 支持的硬件子集里

它解决的是一个很具体的约束：

- TMA descriptor 不是“任何 shared layout 都支持”
- 必须把 layout 收敛到 hardware-compatible encoding

#### `tma_lowering`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp`

- `归类：` 更接近 scheduling 里的 protocol materialization，而不只是
  organization。
- `输入 contract：` descriptor encoding 已经定下来，loop/schedule 主体也已
  基本成形，但 descriptor 访问还只是描述式语义。
- `输出 contract：` TMA transport 被展开成显式的 buffer、mbarrier、wait、
  inval、fenceproxy protocol。
- `下游 consumer：` `proxy_fence_insertion`、`convert-triton-gpu-to-llvm`、
  `nvgpu_to_llvm`。

作用：

- `descriptor_load/gather` 变成：
  `local_alloc + init_barrier + barrier_expect + async_tma_copy + wait/inval`
- `descriptor_store/reduce/scatter` 变成：
  `local_alloc + fence_async_shared + async_tma_store + wait`
- `make_tensor_desc` 变成：
  scratch allocation + 设备端 descriptor object + `tensormap_fenceproxy_acquire`

这一步的核心不是“lower 成几个 op”，而是把“通过 descriptor 访问 tile”
这种描述式语义，变成 Hopper/Blackwell 真正执行 TMA 时必须显式持有的：

- staging buffer
- mbarrier 生命周期
- async copy 完成条件
- tensormap fence/proxy acquire

### 4.3 建立 TMEM compute-state 路径

#### `optimize_accumulator_init`

关键文件：

- `lib/Dialect/TritonGPU/Transforms/OptimizeAccumulatorInit.cpp`

- `归类：` 更像 compute-state cleanup，目标是在不改主计算语义的前提下缩短
  accumulator 协议。
- `输入 contract：` loop-carried accumulator 已经出现，但第一轮是否真的需要
 读取旧值还没被证明。
- `输出 contract：` 某些 MMAv5 path 可以从“无旧 accumulator”启动，减少无谓
  的 TMEM init/load。
- `下游 consumer：` `hoist_tmem_alloc`、`mma_lowering`、后续 TMEM
  allocation / lowering。

作用：

- 识别 loop 里 accumulator 的零初始化
- 如果第一次 MMA 本来就不需要读取旧 accumulator，就把
  `useAccumulator/useC` 置成 `false`
- 避免无意义的 TMEM 初始化/回读

它本质上是在缩短 “accumulator state machine” 的第一段路径。

#### `hoist_tmem_alloc`（会跑两次）

关键文件：

- `lib/Dialect/TritonGPU/Transforms/HoistTMEMAlloc.cpp`

- `归类：` cleanup，重点是 live range 和表示收敛，不重新定义 residence。
- `输入 contract：` TMEM carrier 已经出现，但 alloc/load/store 的 tokenized
  graph 还带有局部脚手架和冗余 placement。
- `输出 contract：` alloc 更靠近其真正的支配边界，部分 load/store 链被收敛，
  后续更容易做 partition、allocation 和 token 清理。
- `下游 consumer：` `warp_specialize`、`optimize_partition_warps`、
  `remove_tmem_tokens`、`tensor_memory_allocation`。

作用：

- 合并 alloc/store、load/store 等 tokenized TMEM pattern
- 把 alloc 尽量 hoist 出 `if` / loop
- sink 某些 load，缩小活跃区间

SM100 这里为什么要跑两次：

- 第一次在 warp specialization 前，先把 TMEM graph 尽量 canonicalize
- 第二次在 partition / select-if 合并之后，再利用更新后的 CFG 继续 hoist

这类变换的直接收益通常不是“减少一个 op”，而是：

- 更短的 TMEM live range
- 更少的跨 partition ownership transfer
- 更少的 aref / barrier 噪音

#### `promote_lhs_to_tmem`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/PromoteLHSToTMem.cpp`

- `归类：` compute consumer-facing organization。
- `输入 contract：` `accelerate_matmul` 已把 MMAv5 主路径立起来，accumulator
  的 TMEM residence 已知，但 LHS 还不一定进入 TMEM。
- `输出 contract：` 某些 LHS operand 被改成和 accumulator 协调的 TMEM
  carrier，shared/TMEM 分工进一步收紧。
- `下游 consumer：` `optimize_tmem_layouts`、`tensor_memory_allocation`、
  `mma_lowering`、`TensorMemoryToLLVM.cpp`。

作用：

- 把满足条件的 `TCGen5MMA` LHS operand 从 shared/local path 提升到 TMEM
- A operand 的 TMEM encoding 取“和 accumulator 同 blockM、`colStride = 1`”
  的 densely packed 形式
- 如果原 layout 不兼容，会在允许时插最小的 `convert_layout`

这一步是 Blackwell 路径的关键分叉点之一：

- Hopper 主要回答“shared operand 怎么喂 WGMMA”
- Blackwell 还要回答“operand 是否应该进 TMEM”

#### `mma_lowering`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp`

- `归类：` scheduling 里的 protocol materialization，不是最终 PTX lowering。
- `输入 contract：` MMAv5 op 已经选型完成，TMEM/shared residence 基本明确，
  但 completion side 还是较高层语义。
- `输出 contract：` MMA 自身携带 completion barrier 和 scale residence 等
  target protocol 信息，后续 lowering 不必再回头恢复这些约束。
- `下游 consumer：` `tensor_memory_allocation`、`tmem_barrier_insertion`、
  `MMAv5.cpp`、`NVGPUToLLVMPass.cpp`。

作用：

- 把同步语义的 MMAv5 op 改成带 completion barrier 的 async 语义
- 把 scaled MMA 的 scale operand 从 shared 转成 TMEM
- 尝试把后继 `tc_gen5_commit` 合并回 MMA op

它做的不是最终 PTX lowering，而是先把“完成条件”和“scale residence”
这些对后面 codegen 很关键的语义收进 op 自身。

#### `optimize_tmem_layouts`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp`

- `归类：` TMEM consumer-facing organization。
- `输入 contract：` operand / accumulator 已决定走 TMEM，但具体 load/store/
  reduce 形态还可能对后续 lowering 很别扭。
- `输出 contract：` TMEM side 的 subtile、split、join、reduce 布局被改成更
  接近真实 consumer 需求的 form。
- `下游 consumer：` `interleave_tmem`、`tensor_memory_allocation`、
  `TensorMemoryToLLVM.cpp`。

它主要做四类事：

- 把 `reshape -> trans -> split` 的 TMEM load 链改成
  `tmem_subslice + tmem_load`
- 把 join 后再 store 的链拆回多个独立 `tmem_store`
- 如果 8 warps 的 `tmem_load` 后面要沿 N 做 reduce，改成沿 M 分布的布局
- 如果 shared<->TMEM 的搬运在 `16x256b` layout 下更容易 vectorize，
  就切到那个 layout

这一步的本质不是“布局更好看”，而是让：

- subtiling 更直接
- reduction 不必跨 warp 做额外归约
- shared/TMEM 边界的 lowering 更容易生成高质量指令

#### `interleave_tmem`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/InterleaveTMem.cpp`

- `归类：` target-specific cleanup。
- `输入 contract：` TMEM carrier 和主要 barrier 关系已经成形，可以开始在不改
  ownership / carrier 的前提下压缩寄存器活跃区间。
- `输出 contract：` `tmem_load/store` 的时序更接近 use/def，后续 register
  pressure 和代码生成空间都更好。
- `下游 consumer：` `reduce_data_duplication`、`reorder_instructions`、
  `tensor_memory_allocation`、`TensorMemoryToLLVM.cpp`。

作用：

- 尽量把 `tmem_load` 下沉到更靠近 use 的位置
- 尽量把 `tmem_store` 上提
- 通过本地 alias analysis 避开真正冲突的 TMEM 访问和 barrier signal

目标非常直接：减寄存器压力。

对 Blackwell 来说，TMEM 很多时候是在“减少寄存器驻留”，但如果 `tmem_load`
太早发出，结果又会重新长时间占寄存器，所以必须再做这一步。

#### `remove_tmem_tokens`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/RemoveTMEMTokens.cpp`

- `归类：` cleanup。
- `输入 contract：` token 之前主要作为 hoist / reorder / canonicalization 的
  脚手架存在，真正该保留的顺序已基本转成 barrier / CFG / dataflow。
- `输出 contract：` TMEM IR 从“带临时协议脚手架”收束成“只保留必要顺序信息”。
- `下游 consumer：` `canonicalizer`、`loop_aware_cse`、
  `tensor_memory_allocation`、后续 LLVM lowering。

作用：

- 在 token 不再需要表达依赖时，把 `TMEMAlloc/Load/Store/MMA` 上的 token
  和 dependency operand/result 删掉

这一步不是 target feature，而是 contract 收束：

- 前面 token 用来约束重排和 hoist
- 到这个时点以后，真正该保留的顺序应该已经由 barrier / layout / CFG
  自身表达了

#### `tensor_memory_allocation`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`

- `归类：` make_llir 边界上的资源分配 pass，也属于 lowering-side legality
  前置条件。
- `输入 contract：` 逻辑 TMEM alloc、layout、barrier intent 都已存在，但
  “落在哪块物理 TMEM”还没被决定。
- `输出 contract：` 每个 TMEM slice 拿到物理 row/col offset，module 级
  `ttg.tensor_memory_size` 也被确定。
- `下游 consumer：` `tmem_barrier_insertion`、`TensorMemoryToLLVM.cpp`、
  `NVGPUToLLVMPass.cpp`。

它负责把逻辑上的 `ttng.tmem_alloc` 变成物理 TMEM offset。

关键机制：

- 用 2D bitmap 做 first-fit 分配
- 按 liveness 回收和复用 TMEM 区域
- 列对齐目前强制按 4 columns
- 对某些 MMAv5 情况施加 row constraint
  例如 64-row 的 A operand 和 accumulator 要落在同一 row group
- 最终给每个 alloc 写：
  `tensor_memory_col_offset`
  `tensor_memory_row_offset`
- 同时写 module attr：
  `ttg.tensor_memory_size`

这一步是 SM100 上最典型的“资源不是抽象无限”的证据。TMEM 是物理受限资源，
后面的 reuse hazard、alloc/dealloc、copy layout 都依赖这个分配结果。

#### `tmem_barrier_insertion`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp`

- `归类：` lowering-side legality repair。
- `输入 contract：` `tensor_memory_allocation` 已经把逻辑 TMEM slice 映射成
  真实物理区间，所以 alias / reuse hazard 终于可判定。
- `输出 contract：` 只有真的存在 TMEM reuse/alias 风险的地方才补
  `ttg.barrier local`。
- `下游 consumer：` `convert-triton-gpu-to-llvm`、`TensorMemoryToLLVM.cpp`。

作用：

- 对 aliasing TMEM slice 做精细分析
- 在 `load/store/store-store/load->mma/store->mma` 等需要时插入
  `ttg.barrier local`

重要点在于它不是“只看 value 相不相等”，而是利用
`tensor_memory_col_offset` / `row_offset` 建模 TMEM slice，尽量只为真的
reuse hazard 插 barrier。

#### 最终 TMEM / MMAv5 lowering

关键文件：

- `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp`

这一步已经不再是 TTGIR 主链里的“再做一次智能决策”，而是消费前面已经建立好
的 contract：

- `消费的输入：` 物理 TMEM offset、layout、completion barrier、CTA mode、
  以及 shared/TMEM residence。
- `输出：` 真实 `tcgen05` 指令和相关 runtime protocol。

这一层才真正把前面建立好的 contract 落成 `tcgen05` 指令：

- `ttg.tensor_memory_size` 驱动
  `tcgen05.alloc -> relinquish_alloc_permit -> dealloc`
- `TMEMCopyOp` 落成 `tcgen05.cp`
- `TMEMLoadOp` / `TMEMStoreOp` 落成 `tcgen05.ld` / `tcgen05.st`
- `TCGen5MMA*` / `TCGen5CommitOp` 落成 `tcgen05.mma` / `tcgen05.commit`

所以前面的 `TMEM` 优化不是局部美化 IR，而是在为这一步准备：

- 合法地址和 slice
- 合法 layout
- 合法 barrier / completion protocol
- 合法寄存器活跃区间

### 4.4 automatic warp specialization 和寄存器/warp 重平衡

#### `assign_latencies` + `schedule_loops`

深挖文档：

- `learn_triton/notes/2026-06-29-assign-latencies.md`
- `learn_triton/notes/2026-06-29-schedule-loops.md`

- `归类：` target-driven scheduling。
- `输入 contract：` residence / carrier 已基本定下，loop body 里的关键 op
  也已经有 target flavor，但时序关系还偏“结构上存在”而不是“显式阶段合同”。
- `输出 contract：` loop body 里哪些 op 应该前推、重叠、分 stage，开始被写成
  latency / stage 级别的 coarse schedule。
- `下游 consumer：` `warp_specialize`、`pipeline`、`fence_insertion`。

在 SM100 上，这两步尤其重要，因为 loop body 里常见的是：

- `ttng.tc_gen5_mma`
- `ttng.tmem_load/store`
- TMA / barrier / async op

它们需要显式 latency contract 才能继续做 pipeline 和 partition scheduling。

#### `warp_specialize`

关键文件：

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
- `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td`

- `归类：` scheduling pass，但它会重写 partition ownership / handoff
  protocol。
- `输入 contract：` coarse schedule 已存在，loop 也已经足够规范化，可以开始
  划分 producer / consumer / worker 角色。
- `输出 contract：` loop 不再只是“一个 block 顺序执行”，而是变成多个 warp
  partition 之间通过 aref/barrier 交接的协议图。
- `下游 consumer：` `optimize_partition_warps`、`combine_tensor_select_and_if`、
  `hoist_tmem_alloc(true)`、`warp_specialize_to_llvm`。

SM100 分支用的是 automatic warp specialization，不是 Hopper 那套
`hopper_warpspec`。

它内部会串起一组更细的 pass：

- `PartitionScheduling`
- `NVWSHoistTmemStore`
- `NVWSInsertAref`
- `NVWSInsertTmemAref`
- `NVWSLowerAref`
- `PartitionLoops`
- `NVWSLowerWarpGroup`
- `ScheduleLoops`

这组 pass 真正做的事是：

- 把 loop 切成 producer / consumer / worker partitions
- 为跨 partition 的 SMEM/TMEM ownership transfer 插入 aref
- 再把 aref lower 成真实 barrier protocol

所以 warp specialization 在 Blackwell 上不是“把 loop body 切几块”这么简单，
而是一整套数据所有权与同步协议重写。

#### `optimize_partition_warps`

关键文件：

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp`

- `归类：` scheduling / 资源平衡 pass。
- `输入 contract：` warp-specialized partition 已经存在，但每个 partition 该拿
  多少 warps / registers 还没针对当前 IR 压实。
- `输出 contract：` `partitionNumWarps`、`requestedRegisters` 和相关 relayout
  被定下来，后续可以把寄存器预算真的落到不同 warpgroup。
- `下游 consumer：` `warp_specialize_to_llvm`、最终 `nvvm.setmaxnreg` lowering。

作用：

- 估算每个 partition 的 tensor register pressure
- 尝试把 partition warp 数从 `8 -> 4 -> 2 -> 1` 逐步缩小
- 但保留硬约束：
  - `TMALoadLikeOpInterface` 至少 2 warps
  - `TMEMLoad/Store/Alloc` 至少 4 warps
- 如果 warp 数变化，还会重新给 partition relayout
- 最后把 `requestedRegisters` 和新的 `partitionNumWarps` 写回

这一步的设计意图很明确：

- 不是平均分 warps
- 而是让真正吃寄存器的 default partition 拿到更多寄存器预算

#### `warp_specialize_to_llvm`

关键文件：

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`

- `归类：` lowering boundary 上对 scheduling contract 的消费。
- `输入 contract：` `ttg.warp_specialize` region、partition warp 数和
  requested register 预算都已定下来。
- `输出 contract：` LLVM CFG、显式 barrier、`nvvm.setmaxnreg` 等 runtime
  可执行形式。
- `下游 consumer：` `nvgpu_to_llvm`、NVVM/PTX codegen。

作用：

- 把 `ttg.warp_specialize` lower 成显式控制流和 barrier
- 如果模块有 `maxnreg`，还会在默认 warpgroup / worker warp 之间做
  `nvvm.setmaxnreg` 动态寄存器重分配

这一步说明前面的 `optimize_partition_warps` 不只是改 metadata：

- 它会和 LLVM lowering 配合，真正让 worker warps 让出寄存器
- 默认 warpgroup 在运行主计算时拿到更多 register budget

#### cluster barrier allocator / initializer

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.cpp`
- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp`

- `归类：` lowering-side legality / protocol materialization。
- `输入 contract：` cluster barrier 的逻辑需求已经在 TTGIR/TTNVGPU 层显式化，
  但 slot 和 entry 初始化序列还不存在。
- `输出 contract：` cluster mbarrier slot 分配、kernel entry 初始化和 arrive/
  wait 序列都被补齐。
- `下游 consumer：` `ClusterOpsToLLVM.cpp`、最终 cluster-aware NVVM/PTX
  lowering。

作用：

- 给 warp-specialized region 里需要 cluster barrier 的地方分配 mbarrier slot
- 在 kernel entry 初始化这些 mbarrier
- 为所有相关 warps 建立 cluster arrive/wait 初始化序列

这保证了跨 CTA / cluster 的同步结构在 warp specialization 下也有可执行载体。

### 4.5 legality repair：async proxy / cluster / shared-TMEM 边界

#### `fence_insertion`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`

- `归类：` TTGIR 末段的 legality repair。
- `输入 contract：` scheduling / async consumer 基本已成形，但 generic
  register->shared producer 和 async consumer 之间的有序性还没完全显式。
- `输出 contract：` 必要的 `fence.async.shared` 被补齐，且尽量 hoist 到不破坏
  overlap 的位置。
- `下游 consumer：` `lower_mma`、`proxy_fence_insertion`、后续 LLVM lowering。

作用：

- 当 dot operand 依赖 register->shared copy 时，在 async consumer 前插
  `fence.async.shared`
- 如果依赖都在 loop 外，还会尽量把 fence hoist 出 loop

它不是重新安排 pipeline，而是在当前抽象层上把已经暴露出的 shared/async
hazard 修到合法。

#### `proxy_fence_insertion`

关键文件：

- `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp`

- `归类：` make_llir 边界上的 legality repair。
- `输入 contract：` shared-memory / async-proxy effect 已经具体到足以做 alias
  判断，但 cross-proxy ordering 还没有完全显式。
- `输出 contract：` generic proxy 和 async proxy 之间按需补齐
  `FenceAsyncSharedOp`。
- `下游 consumer：` `convert-triton-gpu-to-llvm`、目标端 async transport
  lowering。

作用：

- 分析 generic proxy 和 async proxy 间的 aliasing 依赖
- 在 `TMALoad/TMAStore/TMEMCopy/MMA/...` 前按需插
  `FenceAsyncSharedOp`

Hopper+/Blackwell shared memory 有 generic proxy / async proxy 区分，这一步
就是在修补两种 proxy 之间的内存有序性。

#### cluster barrier insertion（在 `convert-triton-gpu-to-llvm` 内部）

关键文件：

- `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`
- `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp`

- `归类：` lowering 边界上的 cluster legality repair。
- `输入 contract：` cross-CTA shared / reduce / atomic 依赖已经足够具体，能够
  判定是否真的需要 cluster barrier。
- `输出 contract：` 必要的 `ttng.cluster_barrier` 和 cross-CTA mbarrier init
  sequencing 被补齐。
- `下游 consumer：` cluster barrier allocator / initializer、
  `ClusterOpsToLLVM.cpp`。

注意这不是 Python pipeline 里单独列出来的 pass，但它确实会在
`convert-triton-gpu-to-llvm` 里运行。

作用：

- 分析 cross-CTA shared / layout / reduce / atomic 等依赖
- 按需插 `ttng.cluster_barrier`
- 处理 cross-CTA mbarrier init sequencing

所以如果只盯 `compiler.py` 的 pass 列表，会漏掉这条 SM100 很关键的
cluster legality 修复链。

---

## 5. 哪些是 SM100 分支“新增”的，哪些只是对 SM100 特别关键

### 5.1 `capability >= 100` 分支里显式新增的

- `optimize_accumulator_init`
- `hoist_tmem_alloc(false)`
- `promote_lhs_to_tmem`
- `assign_latencies`
- `schedule_loops`
- `warp_specialize`
- `pipeline`
- `optimize_partition_warps`
- `combine_tensor_select_and_if`
- `hoist_tmem_alloc(true)`
- `remove_tmem_tokens`

这组最能代表 “Blackwell path 和 Hopper path 开始显式分叉”。

### 5.2 不是 SM100-only，但在 SM100 变成主线关键点的

- 保留 descriptor，不提前 rewrite 成 pointer
- `plan_cta`
- `accelerate_matmul`
- `optimize_descriptor_encoding`
- `tma_lowering`
- `optimize_tmem_layouts`
- `interleave_tmem`
- `tensor_memory_allocation`
- `check_matmul_two_cta`
- `proxy_fence_insertion`
- `tmem_barrier_insertion`
- `warp_specialize_to_llvm`
- `nvgpu_to_llvm` / `TensorMemoryToLLVM` / `MMAv5.cpp`

它们有些也服务 Hopper+，有些甚至架构无关，但在 SM100 上会直接决定是否能形成
`tcgen05 + TMEM + TMA + WS` 这条主路径。

---

## 6. 最终会落成什么样的目标指令/协议

从当前仓库的 lowering 实现和 `learn_triton/dumps/matmul/sm100_*` dump 可以看到，
SM100 最终会落到这些目标协议：

- TMEM 基址分配：
  `tcgen05.alloc.cta_group::*`
- 分配许可释放：
  `tcgen05.relinquish_alloc_permit.cta_group::*`
- TMEM 释放：
  `tcgen05.dealloc.cta_group::*`
- shared -> TMEM copy：
  `tcgen05.cp.cta_group::*`
- TMEM load/store：
  `tcgen05.ld.*` / `tcgen05.st.*`
- MMAv5：
  `tcgen05.mma.cta_group::*`
- MMA completion：
  `tcgen05.commit.cta_group::*`
- completion / transport 同步：
  `mbarrier.init / wait / inval`
- warp specialization register rebalance：
  `nvvm.setmaxnreg`

这说明前面那些 pass 最终都不是“抽象整理 IR”，而是在为这些具体协议准备：

- residence
- layout
- issue side
- completion side
- reuse side
- cluster/CTA synchronization side

---

## 7. 如果只想记住最重要的几条

1. SM100 不是单个 `lower_mma` pass 的问题，而是一整条
   `MMAv5 + TMEM + TMA + warp specialization + legality repair` pipeline。
2. 真正的结构性分叉发生在 `capability >= 100` 的 `make_ttgir` 分支：
   `optimize_accumulator_init -> hoist/promote -> schedule -> warp_specialize -> pipeline -> optimize_partition_warps -> remove_tmem_tokens`。
3. descriptor/TMA 和 TMEM/MMAv5 是两条不同 contract，但在 Blackwell 上经常首尾相接。
4. `tensor_memory_allocation`、`tmem_barrier_insertion`、`proxy_fence_insertion`
   这三步说明 SM100 的难点不只是算得快，而是要合法地管理有限 TMEM 和 async proxy。
5. `warp_specialize_to_llvm` 不只是“lower control flow”，它还会和
   `optimize_partition_warps` 配合做动态寄存器再分配。

---

## 8. 继续深挖时，建议按这个顺序读

先看总览，再看专项笔记：

- `learn_triton/notes/2026-06-29-accelerate-matmul.md`
- `learn_triton/notes/2026-06-29-assign-latencies.md`
- `learn_triton/notes/2026-06-29-schedule-loops.md`
- `learn_triton/notes/2026-06-29-pipeline.md`
- `learn_triton/notes/2026-06-30-warp-specialize.md`
- `learn_triton/notes/2026-07-01-tma-pass-learning.md`
- `learn_triton/notes/2026-07-01-tmem-pass-learning.md`
- `learn_triton/notes/2026-07-02-barriers-and-fences.md`
- `learn_triton/notes/2026-07-09-aref-and-tmem-token.md`

如果你要把这个总览继续展开成“按 pass 看 before/after dump 的证据版”，最自然的下一步是：

- 为 `sm100_num_ctas1`
  做一份 `effective passes` 路线图
- 再把 `sm100_num_ctas2`
  里的 cluster / two-CTA / multicast 相关证据补进来
