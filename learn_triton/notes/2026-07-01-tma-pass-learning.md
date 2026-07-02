# 2026-07-01 学习笔记：TMA 相关 pass

## 1. 这篇笔记只回答什么问题

这篇只看 `TMA` 主线：

```text
tt.tensordesc
  -> shared layout contract
  -> async_tma_* + mbarrier protocol
  -> generic proxy / async proxy visibility
```

这里不讨论 Blackwell `TMEM` 本身的驻留、地址分配和 barrier，那是另一条 compute-state 路径。

这篇真正要回答的是三件事：

1. descriptor 关联的 shared tile，何时被证明成 TMA 真能编码的布局？
2. 抽象 `tt.descriptor_*` 访存，何时变成 Hopper/Blackwell 真正执行的 async protocol？
3. generic proxy 和 async proxy 之间的 shared-memory 可见性，何时补齐？

## 2. TMA 的边界先说清楚

`TMA` 是 Tensor Memory Accelerator。当前主题是：

- global tensor tile 如何异步搬到 shared memory
- shared memory 中的 tile 如何异步写回 global
- 为了发出这条 transport path，IR 需要建立哪些 descriptor、barrier 和 proxy contract

它和 `TMEM` 的关系是：

- `TMA` 主要解决 `global <-> shared`
- `TMEM` 主要解决 Blackwell tensor-core operand / accumulator 驻留在什么 memory space，以及如何布局、分配、同步
- 在 Blackwell 上，两条线可能首尾相接，但不属于同一个 memory-space contract

## 3. 先把几个术语钉住

后面 `OptimizeDescriptorEncoding` 那段如果直接读源码，很容易把 `tensor descriptor`、`tensormap`、`shared layout` 和 `Blocked layout` 混在一起。这里先把它们分层。

### 3.1 `tensor descriptor` 和 `tensormap` 不是一层东西

可以把这两者理解成：

- `tt.make_tensor_descriptor` / `tt.descriptor_*`：Triton IR 里的逻辑描述，表达“我要按这个 shape / stride / tile 去访问 tensor”
- `tensormap`：NVIDIA TMA 硬件真正使用的描述符对象，表达“这块 global tensor tile 如何被 TMA engine 搬运”

`TMALowering` 真正把前者物化成后者，入口见：

- [TMALowering.cpp:166](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:166)
- [TMAUtilities.cpp:125](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:125)

`tensormap` 里会写入的关键字段包括：

- `global_address`
- `global_dim`
- `global_stride`
- `box_dim`
- `element_strides`
- `elem_type`
- `swizzle_mode`

这些字段最终通过 `ttng.tensormap_create` 建出来，见 [TMAUtilities.cpp:185](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:185)。

所以更准确地说：

```text
tensor descriptor
  --(TMA-compatible layout contract established)-->
tensormap
  --(consumed by async_tma_*)-->
TMA transport
```

换句话说，`tensormap` 可以视为 **TMA-compatible tensor descriptor 在硬件路径上的落地表示**；不是任意 descriptor 都能无条件映射成 `tensormap`。

### 3.2 `shared layout` 是 shared memory 里的摆放规则

这里的 `shared layout`，不是泛指“数据放在 shared memory 里”，而是更具体的：

```text
logical tensor index -> shared-memory physical address
```

它描述的是一块 tile 落到 shared memory 后，元素如何排布，包括：

- `order`
- swizzle
- `perPhase` / `maxPhase`
- 是否 transpose
- cluster / CTA 相关布局信息

在 Triton 里，这类信息通常由 shared encoding attribute 表达，例如：

- `SharedLinearEncodingAttr`
- `SwizzledSharedEncodingAttr`
- `NVMMASharedEncodingAttr`

从线性布局抽象看，shared layout 的输入维度主要是 `offset` 和 `block`，见 [LinearLayoutConversions.h:30](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h:30)。

### 3.3 `Blocked layout` 是线程分工规则，不是 shared-memory 摆放规则

`BlockedEncodingAttr` 回答的是另一类问题：

```text
logical tensor index -> register / lane / warp / block ownership
```

它主要描述：

- 每个 thread 持有多少元素
- 一个 warp 覆盖多大 tile
- 一个 CTA 如何覆盖更大 tile
- 这些分块如何帮助 coalesced load/store

定义说明里明确写到，它是“each warp owns a contiguous portion of the target tensor”，见 [TritonGPUAttrDefs.td:738](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738)。从线性布局抽象看，它的输入维度是 `register` / `lane` / `warp` / `block`，见 [LinearLayoutConversions.h:23](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h:23)。

所以：

- `Blocked layout` 主要是“算的人怎么分”
- `shared layout` 主要是“存到 shared 里怎么摆”

两者有关，但不是一个层次的 layout。

### 3.4 它们在 TMA 主题里怎么串起来

在这篇笔记关心的路径里，关系最好记成：

```text
distributed / blocked tensor view
  -> shared layout
  -> TMA-compatible descriptor contract
  -> tensormap
  -> async_tma_* transport
```

这也解释了为什么 `OptimizeDescriptorEncoding` 讨论的是 `shared layout`，不是 `Blocked layout`：

- `Blocked layout` 解决的是线程侧如何拥有和处理 tensor
- `OptimizeDescriptorEncoding` 解决的是 descriptor 指向的 shared-side tile，是否已经收缩到 TMA 硬件真能编码的布局子集

所以这个 pass 的关键问题不是“哪个 thread 持有哪个元素”，而是：

```text
当前 shared encoding
能不能稳定映射成 TMA hardware tensormap？
```

这也是“再往上游看，shared layout 仍然主要服务 Triton 自己的 tensor 表达”那句话的具体含义：上游只需要这块 tile 对 Triton 的 shared-memory 读写和 layout conversion 是合法、线性等价的；到了这里，才必须进一步要求它也落在 `tensormap` 真能表达的硬件子集里。

## 4. Pipeline 位置

当前 NVIDIA backend 中，与 TMA 直接相关的 pipeline 顺序在 [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:280)：

```text
make_ttgir:
  optimize_descriptor_encoding
  ...
  if sm90+:
    tma_lowering
  ...
  fence_insertion
  lower_mma

make_llir:
  allocate_shared_memory_nv
  ...
  proxy_fence_insertion
  to_llvmir
```

对应位置：

- `optimize_descriptor_encoding`:
  [compiler.py:280](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:280)
- `tma_lowering`:
  [compiler.py:318](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:318)
- `fence_insertion` / `lower_mma`:
  [compiler.py:325](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:325)
- `allocate_shared_memory_nv`:
  [compiler.py:384](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:384)
- `proxy_fence_insertion`:
  [compiler.py:390](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:390)

这个顺序本身就是设计意图：

1. 先把 descriptor 对应的 shared layout 约束清楚。
2. 再把抽象 descriptor 访存展开成显式 async protocol。
3. 在 TTGIR 阶段先插一轮较理想位置的 fence。
4. shared memory allocation 完成后，再按 alias 结果补保守 fence。

## 5. 一句话心智模型

TMA 这条线可以压成三个 contract：

```text
descriptor layout contract
  -> async transport protocol contract
  -> proxy visibility contract
```

记住这三层，比背 pass 名字更稳。

## 6. 各 pass 在做什么

## 6.1 `OptimizeDescriptorEncoding`

源码：
[OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:29)

### Problem

进入这个 pass 时，IR 已经有 `tt.tensordesc`，也已经知道 descriptor 指向哪块 shared-side tile，但还缺一个对后续 TMA 路径至关重要的事实：这块 tile 的 shared encoding 是否落在 tensormap 真能表达的那一小块布局子集里。

这里之所以是问题边界，是因为再往上游看，shared layout 仍然主要服务 Triton 自己的 tensor 表达，编译器还可以继续保持“线性等价即可”的抽象；但从这里往下，`TMALowering` 和 `TMAUtilities` 必须读取稳定的 swizzle、block-shape 和 element-type 语义去创建真正的 tensormap。如果这个 pass 不先把“descriptor 可编码”变成显式事实，后面就不是优化质量问题，而是 TMA path 根本没有稳定 lowering 前提。

### Goal

把每个参与 TMA 的 descriptor 绑定到一个 TMA 真正可编码的 shared-memory layout。

### Constraint

source:

- 上游 shared layout 决策首先服务一般 Triton shared tensor，不会提前替每个 descriptor 选定 tensormap-friendly encoding
- Hopper/Blackwell tensormap 编码能力只覆盖共享内存布局空间里的一个硬件子集

manifestation:

- 不能默认“任意 shared layout 都能直接进入 TMA”
- 当前实现首先接受的是**非转置** `NVMMASharedEncodingAttr`
- 如果原布局不是现成兼容编码，就必须找一个线性等价、但可映射到 tensormap 的替代编码

关键代码：

- 兼容入口：
  [OptimizeDescriptorEncoding.cpp:31](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:31)
- fallback 枚举：
  [OptimizeDescriptorEncoding.cpp:61](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:61)
  [OptimizeDescriptorEncoding.cpp:70](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:70)

当前 fallback 显式枚举：

- `fp4Padded in {false, true}`
- `swizzle in {0, 32, 64, 128}`

### Design intent

不改变 descriptor 这层高层抽象，也不在这里直接展开 TMA 指令协议，而是只做一件关键的前置合法化：把 shared tile 绑定到一个 tensormap 可以稳定编码的布局上。这样后续 TMA lowering 可以直接消费这个 contract，而不需要在 lowering 阶段重新猜布局兼容性。

### Decision

这个 pass 实际回答的是：

```text
当前 descriptor 对应的 shared layout
是否已经是 TMA 可编码布局？
如果不是，能否找到一个线性等价的可编码布局？
```

### Output contract

这个 pass 之后，后续 TMA 路径可以依赖：

- descriptor 已经携带 shared layout 信息
- 该 layout 可以映射到合法的 tensormap 语义
- `TMALowering` / `TMAUtilities` 不需要再面对“抽象但未定界”的 shared encoding

### 为什么它必须先于 `TMALowering`

后面的 descriptor lowering 会直接消费 descriptor 上的编码信息去创建 tensormap 相关对象，入口见：

- [TMAUtilities.cpp:27](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:27)
- [TMAUtilities.cpp:125](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp:125)

所以这里不是“先做也行，后做也行”的 cleanup，而是 TMA lowering 的合法性前提。

## 6.2 `TMALowering`

源码：
[TMALowering.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:27)

测试：
[tma_lowering.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/test/TritonNvidiaGPU/tma_lowering.mlir:1)

### Problem

进入这个 pass 时，IR 里表达的还是“通过 descriptor 访问一个 tile”这种描述式意图；它还没有把 Hopper+/Blackwell 真正执行 TMA 所需的 staging object、mbarrier 生命周期、async copy 依赖和设备端 descriptor object 明确写出来。

这里之所以是问题边界，是因为从这个位置开始，后续 pass 和 LLVM/NVVM lowering 面对的必须是“可执行协议”，不能再是“高层想做一次 descriptor 访存”的抽象说法。`OptimizeDescriptorEncoding` 已经把 layout 合法性解决掉了，接下来如果还不把 descriptor op 展开成显式 protocol，后续同步 pass 看不到真实依赖，lowering 也无法落到 PTX TMA 指令和 barrier 语义上。

### Goal

把抽象的 `tt.descriptor_*` 和 `tt.make_tensor_descriptor` 改写成 NVIDIA dialect 中真实可执行的 TMA 协议。

### Constraint

source:

- TMA hardware 不是 register-to-register 的普通 load/store，而是围绕 shared/local staging、barrier 和 descriptor object 运行的异步 transport 协议
- store / reduce / scatter 路径会跨越 generic proxy 与 async proxy 的可见性边界
- 设备端最终需要的是可引用的 tensormap object，而不是纯逻辑 descriptor 值

manifestation:

- load / gather 不能只变成“另一个 load op”，必须显式展开成 local alloc + barrier alloc/init/expect + async_tma + wait
- store / reduce / scatter 不能直接从寄存器端发起，必须先 materialize 到 shared-side object，再显式补 `fence_async_shared`
- `make_tensor_descriptor` 不能保留抽象值形式，必须 materialize 成设备端 descriptor object

### Design intent

把 descriptor 访问统一改写成“显式对象 + 显式 barrier + 显式 async transport”的协议 IR。这样后续 pass 只需要理解普通的 async op、fence 和 wait，而不需要再理解高层 descriptor 语义。

### Decision

这个 pass 实际回答的是：

```text
每一种 descriptor 相关 op
在 NVIDIA dialect 中应该展开成哪套 async TMA 协议？
```

### 它具体改写什么

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

### Load / Gather 模板

`lowerTMALoad` 的固定骨架见 [TMALowering.cpp:27](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:27)：

1. `ttg.local_alloc`
2. barrier memdesc alloc
3. `ttng.init_barrier`
4. `ttng.barrier_expect`
5. `ttng.async_tma_*`
6. `ttng.wait_barrier`
7. `ttng.inval_barrier`
8. `ttg.local_load`

它表达的不是“descriptor load 直接产出寄存器 tensor”，而是：

```text
async copy to shared/local object
  -> barrier completion
  -> local_load to registers
```

### Store / Reduce / Scatter 模板

store 类路径的公共骨架见 [TMALowering.cpp:100](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:100) 和 [TMALowering.cpp:116](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:116)：

1. `ttg.local_alloc`
2. `ttng.fence_async_shared`
3. `ttng.async_tma_*`
4. `ttng.async_tma_store_wait`

核心语义是：

```text
generic proxy 先把 shared 写好
  -> fence 让 shared 对 async proxy 可见
  -> TMA store engine 再读取 shared
```

### `make_tensor_descriptor` 模板

`tt.make_tensor_descriptor` 的展开见 [TMALowering.cpp:166](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:166)：

1. `ttg.global_scratch_alloc`
2. `ttng.tensormap_create`
3. `ttng.tensormap_fenceproxy_acquire`
4. `ttng.reinterpret_tensor_descriptor`

### Output contract

这个 pass 之后，后续 pipeline 可以依赖：

- `tt.descriptor_*` 这层抽象已经消失
- IR 中显式存在 async copy、barrier、fence 和 descriptor-object materialization
- 后续 pass 不需要再理解“描述式 descriptor 访存”，只需要处理已经展开的执行协议

## 6.3 `FenceInsertion`

源码：
[FenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:1)

### Problem

进入这个 pass 时，`TMALowering` 等更早的变换已经把 shared-side producer 和 async-side consumer 暴露成可见的 use-def 关系，但“存在依赖”并不自动等于“generic proxy 对 async proxy 可见”。特别是 dot / mma consumer 读取的 shared operand，可能仍然来自寄存器到 shared 的写入链。

这里之所以是问题边界，是因为它位于 TTGIR 优化尾部：一方面，前面的 lowering 已经把真正的 shared producer 和 async consumer 暴露出来；另一方面，控制流和 use-def 仍然足够结构化，可以把 fence 放在更理想的位置并尝试 hoist。如果再往后拖，剩下的就只能是 allocation 后基于 alias 的保守补救。

### Goal

在 TTGIR 优化阶段，先把一批“明显需要的” async-shared fence 放到较理想的位置。

### Constraint

source:

- Hopper+ 把 generic proxy 和 async proxy 分开，shared memory 的可见性需要显式桥接
- 这个阶段还没有 shared allocation 结果，但已经有比较结构化的 use-def 链，可用于做更积极的位置选择

manifestation:

- 不能在这里做精确 alias-based 全覆盖分析
- 当前实现只围绕 `DotOpInterface` 的 A/B operand，沿 use-def 追踪“寄存器写 shared”链，然后在 consumer 前插 fence，并尽量向循环外 hoist

关键代码：

- 遍历 dot op 并插 fence：
  [FenceInsertion.cpp:39](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:39)
- 查找 reg-to-shared 依赖：
  [FenceInsertion.cpp:77](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:77)

### Design intent

把“理想位置的 fence”尽量放在仍有结构化控制流的信息阶段解决，而不是等所有情况都拖到 allocation 之后再做保守修补。这样能让常见 dot / mma 路径先拿到更干净的同步位置。

### Decision

这个 pass 实际回答的是：

```text
这个 dot/mma consumer 的 shared operand
是否依赖寄存器到 shared 的写入链？
如果依赖，fence 能否安全 hoist 到更外层？
```

### Output contract

这个 pass 之后，TMA/MMA 路径中一部分明显的 generic->async 可见性缺口已经在 TTGIR 阶段补齐；后面的 `ProxyFenceInsertion` 主要承担 allocation 后的保守兜底。

## 6.4 `ProxyFenceInsertion`

源码：
[ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:1)

### Problem

进入这个 pass 时，shared memory allocation 已经完成，因此不同 op 是否访问 aliasing buffer 终于可判定；但 generic proxy 和 async proxy 之间剩余的可见性缺口还没有自动消失。尤其在 TMA、MMA、TMEMCopy 这类 async-side op 参与时，单靠普通 shared RAW/WAR 依赖并不能说明跨 proxy 是否可见。

这里之所以是问题边界，是因为只有到了 allocation 之后，pass 才能把“同一块 shared buffer”这件事从抽象 memdesc 关系收束成真实 alias 关系；再往下游，LLVM lowering 期望看到的应该已经是 fence 补齐后的 IR，而不是自己再回头推导高层 proxy contract。

### Goal

在 allocation 之后，按真实 alias 结果保守补齐 generic proxy / async proxy 的 shared-memory 可见性。

### Constraint

source:

- Hopper+/Blackwell 的 shared memory 有 generic proxy 与 async proxy 之分
- 不同 async-side op 的读写方向并不相同，必须先做 proxy 分类
- allocation 之后才能知道哪些 shared buffer 真正 alias

manifestation:

- 不能把这个问题降格成普通 shared-memory 依赖分析
- 当前实现把 `TMALoadLikeOpInterface` 和 `CLCTryCancelOp` 视为 async proxy write
- 把 `WarpGroupDotOp`、`MMAv5OpInterface`、`TMEMCopyOp`、`TMAStoreLikeOpInterface` 视为 async proxy read
- 然后用 allocation analysis 判断这些 proxy access 是否与已有 shared access 相交，需要时在 op 前插 `ttng.fence_async_shared`

关键代码：

- async proxy write 分类：
  [ProxyFenceInsertion.cpp:33](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:33)
- async proxy read 分类：
  [ProxyFenceInsertion.cpp:48](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:48)
- allocation 后交集检查与插 fence：
  [ProxyFenceInsertion.cpp:155](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:155)

### Design intent

把 proxy 可见性的“最后一公里”集中留给 allocation 之后的统一分析来做，而不是让每个 lowering 局部猜测自己是否已经足够同步。这样同步策略统一依赖同一套 alias 和 proxy 分类结果。

### Decision

这个 pass 实际回答的是：

```text
当前 async proxy op
是否与之前的 shared/generic access 落在同一 aliasing buffer 上？
如果会相交，是否必须在这里补 fence？
```

### Output contract

这个 pass 之后，后续 LLVM/NVVM lowering 可以假设：

- 需要跨 proxy 传播可见性的地方已经有 `ttng.fence_async_shared`
- lowering 不需要重新推导高层 alias / proxy contract

## 7. Hopper 与 Blackwell 上这条线有什么差别

TMA 主线本身在 Hopper (`sm90`) 和 Blackwell (`sm100`) 都存在：

- 两代都走 `OptimizeDescriptorEncoding`
- 两代都走 `TMALowering`
- 两代都需要 `FenceInsertion` / `ProxyFenceInsertion`

真正的分叉不在 TMA transport 本身，而在 transport 之后的 compute-state path：

- Hopper 更典型的是 shared + warp-group / WGMMA 路径
- Blackwell 在保留 TMA transport 的同时，还可能继续走 TMEM operand / accumulator 路径

所以更准确的说法是：

```text
TMA 是两代共享的 transport contract
TMEM 是 Blackwell 额外增加的 compute-state contract
```

## 8. TMA 和 TMEM 在哪里相连

### 8.1 它们不是同一条 memory-space 路径

- TMA 主要负责 `global <-> shared`
- TMEM 主要负责 `tensor-core state <-> tensor memory`

所以不能把 “TMA load” 理解成 “load 到 TMEM”。

### 8.2 它们在数据流上可能首尾相接

在 Blackwell 上，常见链路更接近：

```text
global
  --(TMA)--> shared/local object
  --(PromoteLHSToTMem 等)--> TMEM operand
  --(TCGen5 MMA)--> TMEM resident state
```

TMA 更像“运输入口”，TMEM 更像“计算驻留位置”。

### 8.3 它们在同步语义上会相遇

[ProxyFenceInsertion.cpp:48](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:48)
把 `MMAv5OpInterface`、`TMEMCopyOp`、`TMAStoreLikeOpInterface` 都视为 async proxy read；而 [ProxyFenceInsertion.cpp:33](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:33) 把 `TMALoadLikeOpInterface` 视为 async proxy write。

这说明 TMA transport 和后续 TMEM / MMA 消费共享 Hopper+/Blackwell 的 proxy memory model。

## 9. 读 dump 时重点看什么

建议按下面顺序读 `sm90` 和 `sm100` dump：

1. `OptimizeDescriptorEncoding` 前后：
   看 `!tt.tensordesc<...>` 是否开始绑定 `#ttg.nvmma_shared<...>`。
2. `TMALowering` 前后：
   看 `tt.descriptor_*` 是否消失，是否变成 `ttng.async_tma_*`、`ttng.wait_barrier`、`ttg.local_alloc`。
3. `FenceInsertion` 前后：
   看 dot / mma consumer 前是否新增 `ttng.fence_async_shared`。
4. `ProxyFenceInsertion` 前后：
   看 allocation 完成后，shared / async proxy 边界附近是否又补了一轮 fence。

## 10. 最小总结

如果只记一条主线，可以记成：

```text
TMA path:
  OptimizeDescriptorEncoding
    -> TMALowering
    -> FenceInsertion
    -> ProxyFenceInsertion
```

更抽象一点：

```text
TMA 解决的是
descriptor layout
  -> async transport protocol
  -> proxy visibility
```
