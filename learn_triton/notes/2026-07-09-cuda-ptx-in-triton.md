# CUDA Programming Guide / PTX ISA 9.3 在 Triton 中的体现

日期：2026-07-09

范围：只整理当前 Triton 仓库里已经有明确落点的内容。重点看

- Python / Gluon surface
- TTIR / TTGIR / TTNG IR
- NVIDIA backend lowering / pass / test

不做两类事：

- 不把两份 PDF 全量复述一遍
- 不把 repo 里还没有明显用户入口或只有零散痕迹的主题硬凑进去

证据来源：

- `learn_triton/reference/cuda-programming-guide.pdf`
- `learn_triton/reference/ptx_isa_9.3.pdf`
- 当前源码、lowering、tests、tutorials

---

## 0. 先给总判断

如果只抓一句话：

```text
CUDA Programming Guide 负责给 Triton 提供执行模型 / 存储层级 / 同步模型；
PTX ISA 负责给 Triton 提供最终要落成的指令契约。
```

所以在 Triton 里，这两份文档最明显的体现不是“某个 API 名字一样”，而是下面这些结构真的存在：

| PDF 里的概念 | Triton 里的直接体现 | 关键证据 |
|---|---|---|
| warp / CTA / cluster | `num_warps`、`num_ctas`、cluster launch、`ttng.cluster_*` | [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:109), [driver.c](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/driver.c:947), [driver.c](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/driver.c:970), [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:81) |
| global/shared/distributed shared/tensor memory | memdesc memory space、shared/tmem 资源检查、cluster + multicast 路径 | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:43), [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/compiler/compiler.py:462) |
| TMA / TensorMap / descriptor | `tt.descriptor_*`、`TensorDescriptor`、`ttng.async_tma_*`、`ttng.tensormap_*` | [TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1226), [hopper.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/experimental/gluon/nvidia/hopper.py:10), [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:446) |
| `cp.async` async-group | `ttg.async_copy_global_to_local` + `ttg.async_commit_group` + `ttg.async_wait` | [TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47), [LowerLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp:440) |
| `st.async.shared` | `ttng.async_shared_store` + mbarrier completion | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:415), [MemoryOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp:321) |
| `mbarrier` phase / expect / wait | `ttng.init_barrier`、`barrier_expect`、`arrive_barrier`、`wait_barrier` | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:257) |
| proxy fence / tensormap fence | `ttng.fence_async_shared`、`ttng.fence_mbarrier_init_release_cluster`、`ttng.tensormap_fenceproxy_acquire`、`ProxyFenceInsertion` | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:52), [ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:7), [TMAToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp:184) |
| WGMMA | `ttng.warp_group_dot`、`ttng.warp_group_dot_wait`、WGMMA pipeline 后处理 | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:203), [WGMMAPipeline.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:133) |
| TCGen05 / TMEM | `ttng.tc_gen5_*`、`ttng.tmem_*`、TMEM barrier repair | [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:632), [TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:23) |

---

## 1. 线程层级、SIMT、warp、CTA、cluster

对应 PDF：

- CUDA Programming Guide `1.2.2.2 Warps and SIMT`
- CUDA Programming Guide `2.3.2 Thread Hierarchy`
- CUDA Programming Guide `2.3.3.8 Distributed Shared Memory`
- PTX ISA `9.7.14.3 barrier.cluster`

### 1.1 Triton 直接继承了 CUDA 的 warp / CTA 基本单位

Triton NVIDIA backend 把 `warp_size` 固定成 32，并把用户给的 `num_warps` 直接变成 launch 维度：

- `CUDAOptions` 里有 `num_warps`、`num_ctas`、`warp_size = 32`
- launch 时 `blockDimX = 32 * num_warps`

见：

- [third_party/nvidia/backend/compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:109)
- [third_party/nvidia/backend/driver.c](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/driver.c:947)

这说明 Triton 不是“抽象掉了 warp”，而是把 warp 当成 codegen 和 launch 的基本预算单位。

### 1.2 `num_ctas` 对应的不是普通 grid 扩张，而是 cluster

CUDA guide 里 cluster 是一组同时驻留、能跨 CTA 同步和访问 distributed shared memory 的线程块。

Triton 里：

- `num_ctas > 1` 被限制为 `SM90+`
- launch 时会设置 `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`
- 同时设置 cluster scheduling policy

见：

- [third_party/nvidia/backend/compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:196)
- [third_party/nvidia/backend/driver.c](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/driver.c:970)

所以在 Triton/NVIDIA backend 里，`num_ctas` 的语义更接近：

```text
一个 program instance 内部包含几个 cluster-local CTAs
```

而不是“多发几个无关 block”。

### 1.3 cluster barrier 在 TTNG IR 里是 first-class op

TTNG 直接定义了：

- `ttng.cluster_arrive`
- `ttng.cluster_wait`
- `ttng.cluster_barrier`

其中 `cluster_barrier` 的说明明确写了它 lower 成 arrive/wait 对，并且在 warp specialization 下会额外包一层 synthetic `ttg.warp_specialize`，保证 worker warps 也执行 barrier。

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:81)。

LLVM lowering 侧要分清两条路径（不要混为一谈）：

- `cluster_arrive` / `cluster_wait` 走 `ClusterSyncOpConversion`，落成 **NVVM cluster barrier**（`NVVM::ClusterArriveOp` / `ClusterWaitOp`，即 `barrier.cluster.arrive` / `barrier.cluster.wait`）。
  见 [ClusterOpsToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp:154)。
- cluster-scoped 的 `mbarrier.arrive` / `mbarrier.try_wait` 不在这条路径上，而是 `cluster_barrier` 带 mbar offset 时由 `ClusterBarrierOpConversion` 生成（helper `createMBarrierArrive` / `createMBarrierWait`）。
  见 [ClusterOpsToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp:69)、[ClusterOpsToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp:183)。

结论：

```text
CUDA/PTX 里的 cluster 不是“背景知识”，而是 Triton 当前 Hopper/Blackwell 路径里的真实 IR / launch / lowering 结构。
```

---

## 2. 内存层级：global / shared / distributed shared / tensor memory

对应 PDF：

- CUDA Programming Guide `2.3.3 GPU Device Memory Spaces`
- CUDA Programming Guide `2.3.3.2 Shared Memory`
- CUDA Programming Guide `2.3.3.8 Distributed Shared Memory`
- PTX ISA 中的 state space 与各类 async op 目标地址空间

### 2.1 Triton 把 memory space 放进了类型和 effect 系统

TTNG dialect 里直接声明了三个 resource：

- `GlobalMemory`
- `SharedMemory`
- `TensorMemory`

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:43)。

后续各个 op 的 operand/result 都带着这些 effect，例如：

- TMA load: `desc` 读 global，`barrier/result` 写 shared
- TMEM op: `tmem_alloc/load/store` 显式读写 `TensorMemory`

这意味着：

```text
CUDA memory hierarchy 在 Triton 里不是注释，而是 IR 类型系统和 side-effect 分析的一部分。
```

### 2.2 shared memory 和 tensor memory 都有显式资源上限检查

运行前会检查：

- `metadata.shared` 是否超过设备允许的 shared memory
- `metadata.tmem_size` 是否超过当前约束下的 tensor memory 上限

见 [python/triton/compiler/compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/compiler/compiler.py:462)。

这正对应 CUDA guide 里“shared memory 是受架构限制的显式片上资源”，以及 Blackwell PTX 里 TMEM 也是受容量限制的目标存储。

### 2.3 distributed shared memory 在 Triton 中通常不是单独的用户类型，而是 cluster 协议的一部分

Triton 当前更常见的表达方式不是“给你一个 DSM type”，而是把 DSM 相关约束分散到：

- cluster launch (`num_ctas`)
- CGA / multicast layout
- cluster barrier / cross-CTA mbarrier
- TMA multicast / async shared store

所以更准确的理解是：

```text
distributed shared memory 在 Triton 里主要体现为 cluster-aware protocol，
而不是一个独立的新前端抽象。
```

### 2.4 register / local memory 在 Triton 中更多是 compiler-owned，而不是 first-class 用户对象

CUDA guide 里 register / local memory 很重要；但在 Triton 里：

- register 分配主要由 lowering/backend 决定
- local memory 更多是 spill 结果
- 用户一般不会像 shared / tensor memory 那样直接构造一个“local memdesc”

这说明 Triton 不是把所有 CUDA memory space 都做成同等级前端对象；它只把对算法/协议有决定性影响的空间显式化。

---

## 3. TMA、TensorDescriptor、TensorMap

对应 PDF：

- CUDA Programming Guide `4.11.2 Using the Tensor Memory Accelerator (TMA)`
- PTX ISA `9.7.9.26.5 cp.async.bulk.tensor`
- PTX ISA `9.7.9.27 tensormap.replace`
- PTX ISA `9.7.14.17 tensormap.cp_fenceproxy`

### 3.1 `tt.descriptor_load/store` 是 Triton 对 TMA descriptor 语义的核心入口

Triton core dialect 里有：

- `tt.descriptor_load`
- `tt.descriptor_store`

定义里直接写明：

- on NVIDIA targets supporting it，会 lower 到 TMA load/store

见 [TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1226)。

这说明 descriptor 不是后端黑盒，而是从 Triton IR 开始就可见的抽象。

### 3.2 高层 Python surface 已经把 CUDA guide 里的 TMA 约束编码进来了

`TensorDescriptor` / `TensorDescriptorIm2Col` 做了很多和 CUDA guide / PTX 约束同构的检查：

- rank 必须在 `1..5`
- base 必须 `16-byte aligned`
- 除最后一维外，stride 对应字节数也要 `16-byte aligned`
- 最后一维必须 contiguous
- layout 必须是 `NVMMASharedLayout`
- `fp4_padded` 还要满足更严格的 32-byte 对齐和 Blackwell 条件

见 [hopper.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/experimental/gluon/nvidia/hopper.py:10)。

这和 CUDA guide `4.11.2` 里 TMA 对 tensor map / 对齐 / tiled vs im2col 的要求是同一层约束，只是被前移到了 Triton 对象构造阶段。

### 3.3 TMA 在 TTNG IR 里不是一条指令，而是一族 op

TTNG 里有：

- `ttng.async_tma_copy_global_to_local`
- `ttng.async_tma_copy_local_to_global`
- `ttng.async_tma_gather`
- `ttng.async_tma_scatter`
- `ttng.async_tma_reduce`
- `ttng.async_tma_store_wait`

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:446)。

这和 CUDA guide 的划分很一致：

- gmem -> smem 的 bulk tensor async copy
- smem -> gmem 的 bulk async-group completion
- gather/scatter/reduce 这些“不是普通 tiled copy”的变体

### 3.4 lowering 明确落成 `cp.async.bulk.tensor.*`

TMA load lowering 里拼出来的 PTX opcode 是：

```text
cp.async.bulk.tensor.<rank>d....mbarrier::complete_tx::bytes
```

见 [LoadStoreOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1270)。

TMA store lowering 则拼成：

```text
cp.async.bulk.tensor.<rank>d.global.shared::cta.bulk_group
```

见 [LoadStoreOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp:1431)。

这正是 CUDA guide 里讲的那两套 completion model：

- gmem -> smem: shared memory barrier / mbarrier
- smem -> gmem: bulk async-group

### 3.5 TensorMap 的设备端创建/修改也有显式表示

TTNG 里有：

- `ttng.tensormap_create`
- `ttng.tensormap_fenceproxy_acquire`

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:1076)。

lowering 里还能直接看到：

- `tensormap.replace.tile.*`
- `tensormap.cp_fenceproxy.*`
- `fence.proxy.tensormap::generic.acquire.gpu`

见 [TMAToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp:22)。

这和 CUDA guide 里“修改 tensor map 后需要 cp_fenceproxy / acquire fence 才能让后续 async copy 正确看到更新”是一一对应的。

### 3.6 `OptimizeDescriptorEncoding` 体现了“descriptor 不是任意 shared layout 都能吃”

当前文件 [OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:13) 的核心结论是：

- TMA descriptors 只接受 non-transposed 的 `NVMMASharedEncodingAttr`
- 如果当前 shared layout 不是直接兼容的，就要找一个等价的 non-transposed NVMMA layout

这件事在代码里写得很直白：

- `isCompatibleSharedEncoding` 只接受 `!nvmma.getTransposed()`
- 注释直接说 `TMA descriptors only support non-transposed layouts`

见 [OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:29)。

所以 descriptor encoding pass 的设计意图不是“调 layout 好看一点”，而是：

```text
把 Triton 的共享内存布局收束到 TMA/PTX 真正能接受的 descriptor contract 上
```

---

## 4. `cp.async`、async-group、mbarrier、wait/commit

对应 PDF：

- CUDA Programming Guide `3.2.2.3.1 Async Thread and Async Proxy`
- CUDA Programming Guide `4.11.2 Using the Tensor Memory Accelerator (TMA)`
- CUDA Programming Guide `5.4 memory fence`
- PTX ISA `9.7.9.26 Asynchronous copy`
- PTX ISA `9.7.14.16 mbarrier`

### 4.1 Triton 里其实并存两套 async copy completion 模型

#### A. Ampere 风格：`cp.async` async-group

TritonGPU dialect 里有：

- `ttg.async_copy_global_to_local`
- `ttg.async_commit_group`
- `ttg.async_wait`

见 [TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47)。

它们的 contract 正对应 PTX `cp.async` / `cp.async.commit_group` / `cp.async.wait_group`。

#### B. Hopper+ 风格：mbarrier-based completion

TTNG 里有：

- `ttng.init_barrier`
- `ttng.barrier_expect`
- `ttng.arrive_barrier`
- `ttng.wait_barrier`
- `ttng.async_copy_mbarrier_arrive`

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:257)。

这对应 PTX `mbarrier.init / expect_tx / arrive / try_wait`，以及 `cp.async.mbarrier.arrive`。

### 4.2 `ttg.async_wait` 只观察 completion，不负责 CTA rendezvous

`ttg.async_wait` 的定义里明确说：

- 它只保证 async copy groups 的完成
- 它**不提供 CTA synchronization**
- 如果需要 CTA 同步，还要额外加 `ttg.local_barrier`

见 [TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47)。

这和 PTX `cp.async.wait_group` 的本质完全一致：它不是 `__syncthreads()`。

### 4.3 pipeliner 会根据合法性决定能不能走 `cp.async`

`LowerLoops.cpp` 里会判断：

- 这个 load 是否能转成 async load
- `cp.async` 至少要求 4 bytes
- TMA load 则走另一条专门路径

见 [LowerLoops.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp:440)。

所以 Triton 并不是“看到 load 就 async 化”，而是受 PTX 指令约束控制。

### 4.4 `wait_barrier` 的 lowering 就是 `mbarrier.try_wait.parity`

`ttng.wait_barrier` 的定义里已经写了它 lower 成 wait loop；真正 lowering 里能直接看到：

```text
mbarrier.try_wait.parity.shared::cta.b64
```

见：

- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:317)
- [BarrierOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp:287)

这表明 Triton 的 barrier op 不是抽象意义的“等待某件事”，而是明确依附到 PTX mbarrier phase model。

### 4.5 TMA store wait 也保留了“只等读完 shared memory”的语义

`ttng.async_tma_store_wait` 里有 `read_only` 选项，说明只等 store 对 shared memory 的读取完成，这样 shared memory 就可以安全复用了。

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:620)。

这对应 CUDA guide 里对 `cp.async.bulk.wait_group.read` 的说明。

### 4.6 Triton 也直接建模了 `st.async.shared`

除了 TMA，TTNG 还有：

- `ttng.async_shared_store`

它的定义摘要直接写了：

- 用 PTX `st.async.shared` 把 distributed tensor 异步写入 shared memory
- 完成时递减 `mbarrier` 的 transaction count
- 这条路径要求 cluster 至少有 2 个 CTAs

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:415)。

lowering 里能直接看到：

```text
st.async.weak.shared::cluster.mbarrier::complete_tx::bytes
```

见 [MemoryOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/MemoryOpToLLVM.cpp:321)。

这条线更接近 CUDA guide `Async Thread` 小节里提到的 STAS/REDAS 一类 generic-proxy async op，而不是 TMA 那条 async-proxy bulk copy。

更细的 barrier/fence 关系，已在：

- [2026-07-02-barriers-and-fences.md](./2026-07-02-barriers-and-fences.md)

里展开。

---

## 5. proxy、fence、可见性与 sequencing

对应 PDF：

- CUDA Programming Guide `3.2.2.3.1 Async Thread and Async Proxy`
- CUDA Programming Guide `4.11.2` 里对 `fence_proxy_async` / tensormap fence 的使用说明
- PTX ISA `fence.proxy.*`
- PTX ISA `fence.mbarrier_init.release.cluster`

### 5.1 Hopper+ 上 async proxy 和 generic proxy 被 Triton 当成真实 hazard

`ProxyFenceInsertion.cpp` 开头就把设计意图写出来了：

- Hopper+ 上 async proxy 和 generic proxy 分离
- 当 shared memory 在 generic proxy 和 async proxy 间交接时，需要插 fence
- 该 pass 分析 alias/dependency 后保守插入 fence

见 [ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:7)。

这几乎是 CUDA guide `Async Thread and Async Proxy` 那一节的 Triton 直译版。

### 5.2 `ttng.fence_async_shared` 是 Triton 对 `fence.proxy.async` 的显式封装

TTNG 定义：

- `ttng.fence_async_shared`

摘要就是 `fence proxy async`，并且只在 `computeCapability >= 90` 支持。

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:52)。

### 5.3 proxy fence pass 把哪些 op 视为 async proxy 读/写

`ProxyFenceInsertion.cpp` 里明确分类：

- async-proxy write：TMA load、`clc_try_cancel`
- async-proxy read：`warp_group_dot`、MMAv5、`tmem_copy`、TMA store

见 [ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:33)。

这很重要，因为它说明：

```text
在 Triton 里，proxy fence 不只是 TMA 的问题；
WGMMA / TCGen05 / TMEM copy 也都被纳入同一套 async-proxy hazard 模型。
```

### 5.4 cross-CTA mbarrier 初始化需要额外 sequencing

`ClusterBarrierInsertion.h` 明确写了：

- 对 cross-CTA mbarrier，要插
  `fence_mbarrier_init_release_cluster + cluster_arrive/wait(relaxed=true)`

见 [ClusterBarrierInsertion.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.h:11)。

这正对应 PTX 里 `fence.mbarrier_init.release.cluster` 那条规则：初始化不是普通 store，跨 CTA 观察它之前要先建 sequencing。

### 5.5 tensormap acquire 不是“顺手一 fence”，而是完整协议的一部分

`TensormapFenceproxyAcquireOpConversion` 里会发：

- `fence.proxy.tensormap::generic.acquire.gpu`
- 之后还补 `cp.async.bulk.commit_group`
- 再补 `cp.async.bulk.wait_group.read 0`
- 最后做 CTA barrier

见 [TMAToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp:184)。

这说明 TensorMap 可见性在 Triton 里被当成完整协议处理，而不是单条 fence 了事。

---

## 6. WGMMA：warpgroup 级异步 MMA

对应 PDF：

- PTX ISA `9.7.16 Asynchronous Warpgroup Level Matrix Multiply-Accumulate`

### 6.1 Triton 用 `warp_group_dot` 表达 WGMMA，而不是直接把 PTX 暴露给前端

TTNG 定义：

- `ttng.warp_group_dot`
- `ttng.warp_group_dot_wait`

其中：

- `warp_group_dot` 带 `isAsync`
- `warp_group_dot_wait` 以 `pendings` 表示还允许多少 outstanding async dot

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:203)。

这和 PTX WGMMA 的 `mma_async -> commit_group -> wait_group` 对应，但 Triton 把它提升成“dot op + wait op”。

### 6.2 WGMMA wait 数是 pass 重新计算的

`WGMMAPipeline.cpp` 里有专门的 `updateWaits`：

- 通过数 `async_commit_group` 的数量来更新 wait 的 `num`
- 再消掉冗余 wait

见 [WGMMAPipeline.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp:133)。

所以 Triton 不只是“有个 wait op”，而是显式维护 WGMMA 的 outstanding-group bookkeeping。

### 6.3 Triton 当前更强调 WGMMA 的协议面，而不是让用户拼 descriptor 位域

PTX 里 WGMMA 有大量 matrix descriptor、warpgroup、`wgmma.fence/commit/wait` 细节。

Triton 把这层复杂度主要藏在：

- `warp_group_dot`
- layout / descriptor preparation
- WGMMA lowering
- pipeline wait fixup

这正是 Triton 的设计取舍：

```text
保留 WGMMA 的执行/完成模型，
但不把 PTX 级操作数格式直接暴露为用户接口。
```

---

## 7. TCGen05 与 Tensor Memory (TMEM)

对应 PDF：

- PTX ISA `9.7.17 TensorCore 5th Generation Family Instructions`

这是当前 Triton 里和 PTX 绑定最紧的一块之一。

### 7.1 TTNG 对 tcgen05 / TMEM 几乎是一整套专门 dialect

直接相关的 TTNG op 包括：

- `ttng.tc_gen5_mma`
- `ttng.tc_gen5_mma_scaled`
- `ttng.tc_gen5_commit`
- `ttng.tmem_alloc`
- `ttng.tmem_load`
- `ttng.tmem_store`
- `ttng.tmem_subslice`
- `ttng.tmem_copy`

见：

- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:632)
- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:820)
- [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:1004)

### 7.2 `tc_gen5_commit` 就是在做 PTX 里的“把异步完成挂到 mbarrier 上”

`ttng.tc_gen5_commit` 的定义摘要非常直接：

- 让 mbarrier 跟踪此前所有 async tcgen5 op 的完成
- 完成后在 mbarrier 上做 arrive
- 多个 commit 的完成顺序按 issue 顺序保证

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:774)。

lowering 里生成的是：

```text
tcgen05.commit.cta_group::<N>.mbarrier::arrive::one.shared::cluster
```

见 [MMAv5.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp:344)。

### 7.3 `tmem_load` / `tmem_copy` 也是直接对 PTX tcgen05 指令族建模

`tmem_load` 描述里明确写了：

- reduction 形态 lower 到 `tcgen05.ld.red`

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:820)。

lowering 里能直接看到：

- `tcgen05.ld`
- `tcgen05.cp.cta_group::*`

见：

- [TensorMemoryToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp:188)
- [TensorMemoryToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp:646)

### 7.4 Triton 还专门为 TMEM 做了一遍 hazard repair

`TMemBarrierInsertion.cpp` 的核心规则是：

- WAR / RAW / WAW 要 barrier
- `load -> mma`、`store -> mma` 要 barrier
- `mma -> load/store` **不**在这里额外 barrier，因为后面的 `mbarrier wait` 会覆盖完成性

见 [TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:54)。

这正对应 PTX `tcgen05` 那一章里最难的一点：

```text
有些顺序靠 pipelined pairing，
有些顺序靠 commit/wait/mbarrier，
还有些跨线程 handoff 需要额外 specialized sync。
```

Triton 的做法不是让用户手写这些规则，而是在 pass 里把 legality 边界补出来。

### 7.5 runtime 也把 TMEM 当成显式资源

在加载 kernel 时会检查 `tmem_size` 上限。

见 [python/triton/compiler/compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/python/triton/compiler/compiler.py:469)。

这再次说明 TMEM 在 Triton 中不是“隐藏在 tensor core 背后”，而是一个必须被分配、布局、同步、容量约束共同管理的独立资源。

更细的 TMEM / token / barrier 语义，建议继续看：

- [2026-07-01-tmem-pass-learning.md](./2026-07-01-tmem-pass-learning.md)
- [2026-07-09-aref-and-tmem-token.md](./2026-07-09-aref-and-tmem-token.md)

---

## 8. 一些更“PTX 味”的直接暴露

### 8.1 special registers：`%globaltimer`、`%smid`

Triton `triton.language.extra.cuda` 直接暴露了：

- `globaltimer()`
- `smid()`
- `num_threads()`
- `num_warps()`

其中 `globaltimer` / `smid` 直接是内联 PTX：

- `mov.u64 $0, %globaltimer;`
- `mov.u32 $0, %smid;`

见：

- [third_party/nvidia/language/cuda/__init__.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/language/cuda/__init__.py:1)
- [third_party/nvidia/language/cuda/utils.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/language/cuda/utils.py:6)

这属于 PTX ISA 里比较“裸”的那层能力，Triton 也没有完全隐藏。

### 8.2 Blackwell cluster launch control 也已经有 TTNG op

`ttng.clc_try_cancel` 的定义说明：

- 它对应 `clusterlaunchcontrol.try_cancel`
- 结果异步写回 shared buffer，并在完成时 signal mbarrier

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:107)。

这说明 Triton 已经开始把 PTX 9.x 里更偏“硬件能力接口”的内容也逐步做成 IR 节点，而不只限于 matmul / copy。

---

## 9. 一个更统一的读法

把这篇压成最短的结构，其实就是：

```text
CUDA Programming Guide:
  讲执行层级、存储层级、同步范围、程序员模型

PTX ISA:
  讲这些模型最后要落成什么指令 contract

Triton:
  在前端暴露必要的高层对象
  在 IR 中显式化必须保留的协议边界
  在 pass 中修补 legality / hazard
  在 LLVM lowering 中落成具体 PTX 指令
```

从这个角度看，当前 Triton 对两份文档的体现最集中在 4 条主线：

1. `launch / hierarchy`
   `num_warps`、`num_ctas`、cluster barrier、warp specialization
2. `memory / layout`
   shared layout、descriptor encoding、TensorMap、TMEM
3. `async protocol`
   `cp.async` group、TMA `mbarrier`、proxy fence、store wait
4. `compute engine`
   WGMMA、TCGen05、TMEM path

如果后面要继续系统化学习，建议顺序：

1. 先读这篇建立总映射
2. 再读 [2026-07-02-barriers-and-fences.md](./2026-07-02-barriers-and-fences.md)
3. 再读 [2026-07-01-tmem-pass-learning.md](./2026-07-01-tmem-pass-learning.md)
4. 然后回到你正在看的 [OptimizeDescriptorEncoding.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp:1)

这样会比较容易看出：

- 哪些约束来自 CUDA 执行/存储模型
- 哪些约束来自 PTX 指令 contract
- 哪些是 Triton 自己为保证 legality / performance 加上的编译器结构
