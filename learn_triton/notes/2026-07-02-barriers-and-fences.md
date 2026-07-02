# Triton 中的 Barrier 与 Fence：Scope、存储、可见性

日期：2026-07-02
范围：Triton 会发射的每一个 barrier / fence / 同步原语，它作用在哪种存储类
（storage class）上，执行粒度（per thread / warp / warp-group / CTA / cluster），
触发它的上下文，它守护（guard）的范围，以及它建立的内存可见性保证。
（atomic 的 `.acquire/.release` + `.cta/.gpu/.sys` scope 是内存同步语义而非 barrier，
只在 §5.4 简要说明，不在主范围内。）

证据来源：当前源码（正文内联标注 `file:line`）、PTX ISA 9.3 内存模型、
CUDA Programming Guide（`learn_triton/reference/`）。凡属于 PTX 契约而非 Triton
实现细节的断言，都会明确标注。

> 说明：本笔记融合了两轮分析。所有具体的 op 名、行号、pass 名均已对当前源码
> 核实。若源码注释与结论冲突，以源码为准。

---

## 0. 先给总判断：barrier 和 fence 不是一类东西

在 Triton 里，`barrier` 和 `fence` **至少要分成 4 组看**。这是理解全部原语最
重要的框架——按"它到底在等什么/守什么"来分，而不是按名字：

1. **execution barrier（执行会合）**
   让一组线程在控制流上会合（rendezvous），并顺带给某些存储域建立可见性。
   典型：`ttg.barrier`、cluster barrier、`__syncwarp/__syncthreads` 的对应物。

2. **completion tracking（异步完成跟踪）**
   回答"某个异步硬件动作什么时候*真的完成了*"。
   典型：`mbarrier`、`cp.async.wait_group`、`wgmma.wait_group`、
   `ttng.tc_gen5_commit + wait_barrier`。

3. **proxy fence / ordering fence（跨 proxy 排序）**
   解决"不是同一个 proxy / 不是同一个执行域"之间的排序与可见性，**不负责等
   所有线程到齐**。典型：`ttng.fence_async_shared`、
   `ttng.tensormap_fenceproxy_acquire`、`fence_mbarrier_init_release_cluster`。

4. **hazard barrier（编译器自动补的冲突隔离）**
   编译器为避免某类存储冲突（hazard）自动补的 barrier。
   典型：shared memory 上的 `MembarAnalysis`，以及 TMEM 路径上的
   `TMemBarrierInsertion` 插入的 `ttg.barrier local`。

**为什么这个分法是本质的**：背后其实只有一个问题——*一个 agent 写的数据，什么
时候保证能被另一个读同一位置的 agent 观察到？* 上面 4 组只是这个问题在不同硬件
约束下的不同答案，差别沿三条轴展开：

1. **谁必须达成一致**（scope / 参与同步的线程集合）：
   thread < warp（32 lanes）< warp-group（128 threads / 4 warps）< CTA（block）
   < cluster（CGA，同一 GPC 上一组 CTA）< GPU < system。

2. **排序覆盖哪种存储**（proxy 与地址空间）：
   registers、shared memory（`.shared::cta` / `.shared::cluster`）、tensor
   memory（TMEM，Blackwell）、global memory。PTX 进一步把访问分成
   **generic proxy**（普通 ld/st）和 **async proxy**（TMA、`cp.async.bulk`、
   tcgen05）。*同一个 proxy 内*排序便宜；*跨 proxy* 排序需要 `fence.proxy`。

3. **执行 vs. 内存**——有的原语阻塞执行（会合），有的只排序内存（fence），
   多数两者都做。`bar.sync` 两者都做；`fence.proxy.async` 是纯内存排序；
   `mbarrier` 把两者解耦（arrive = 发信号，wait = 阻塞）。

**Triton 把这 4 组的"决策权"分给了三个互相独立的机制**——这是最关键的结构性
事实：

| 关注点 | hazard 类型 | 机制 | 插入的原语 |
|---|---|---|---|
| CTA 内 shared memory RAW/WAR/WAW | 数据 hazard | `MembarAnalysis`（`lib/Analysis/Membar.cpp`） | `ttg.barrier` → `bar.sync` |
| TMEM 路径 load→mma / store→mma | 数据 hazard | `TMemBarrierInsertion`（`lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp`） | `ttg.barrier local` |
| generic-proxy ↔ async-proxy on smem | proxy hazard | `FenceInsertion` / `ProxyFenceInsertion` | `ttng.fence_async_shared` → `fence.proxy.async` |
| 异步引擎完成（cp.async, wgmma, TMA, tcgen05） | latency / liveness | 在 pipeline lowering 中显式给出，不是独立分析 | 各引擎自己的 commit/wait/mbarrier op |

这几个机制通过两个标记协作，从而没有 hazard 被重复同步：
`containsLocalBarrier`（建立了同步点的 op）与 `MemWaitOpTrait`（异步完成 wait 的
op）。详见 §7。

---

## 1. Scope 词汇表（粒度轴）

先固定每个粒度的含义。这些是 PTX/硬件定义，不是 Triton 的发明。

- **per thread** —— 单个 lane 的程序顺序。这里没有任何原语*只*作用于 per-thread；
  barrier/fence 的意义就在于关联*不同*线程。但 predication（`@$0`、elect-one）
  意味着一个原语可能由一个线程*代表*一组发起（例如 `mbarrier.init` 由一个线程
  发起，但其效果是 CTA/cluster 可见的）。

- **per warp（32 lanes）** —— SIMT 执行单元。`bar.warp.sync` / `__syncwarp()`
  让一个 warp 的 32 个 lane 重收敛并排序内存。便宜：没有跨 warp 的调度器会合。
  Triton 在单 warp 的 warp-specialize partition
  （`ConvertWarpSpecializeToLLVM.cpp:75`）以及 warp-synchronous convert-layout
  中发射它；tensormap 构造代码里也能看到它——那是为了让 warp 内 cooperative
  修改 descriptor 后再继续用。

- **per warp-group（128 threads = 4 个连续 warp）** —— Hopper `wgmma` 和
  Blackwell `tcgen05` 的执行单元。`wgmma.*.sync.aligned` 要求 warp-group 的全部
  128 个线程都已收敛。这**不是** CTA barrier——CTA 内其它 warp-group 不受影响。
  "aligned" = 编译器断言 group 内每个线程都执行该指令（无发散）。

- **per CTA（thread block）** —— `bar.sync 0` / `__syncthreads()`。block 的所有
  线程会合；此前所有 shared/global 访问变为 block 内可见。这是主力原语。命名
  barrier `bar.sync N, cnt`（0 ≤ N < 16）让不相交的线程子集独立会合——被 warp
  specialization 使用。

- **per cluster（CGA —— Cooperative Grid Array，SM90+）** —— 同一 GPC 上一组
  被协同调度的 CTA，它们共享一个 *distributed shared memory*（DSMEM）窗口，
  于是 CTA A 可以寻址 CTA B 的 shared memory（`.shared::cluster`）。
  `barrier.cluster.arrive` / `barrier.cluster.wait` 让 cluster 的所有 CTA 会合。
  mbarrier 也可以是 cluster-scoped（`mbarrier.arrive...cluster`）。

- **per GPU / per system** —— fence 和 atomic 上的 `.gpu` / `.sys` scope。
  Triton 对 TMA tensormap acquire fence 用 `.gpu`（`TMAToLLVM.cpp:205`），
  device-scope atomic 也用 `.gpu`。`.sys` 用于 host 可见 / peer-GPU 排序。

粒度升级（escalation）自始至终由 barrier 的 `MemDescType` 上的 **CGA broadcast
mask** 驱动：`getCGABroadcastMask() != 0` 会把 `.shared::cta` 翻成
`.shared::cluster`，把 `mbarrier.arrive` 翻成 `mbarrier.arrive...cluster`
（`BarrierOpToLLVM.cpp:214,349`）。

---

## 2. 存储类与"为什么每种都要各自的同步"

| 存储 | 谁写 | 谁读 | 排序原语 |
|---|---|---|---|
| **registers**（distributed） | 单个线程 | 同一线程；其它线程只能经 smem | 线程内无需；跨线程需 smem + barrier |
| **shared memory `.shared::cta`** | 任意线程（generic proxy）或 TMA/tcgen05（async proxy） | CTA 内任意线程 | `bar.sync`（generic↔generic）；`fence.proxy.async`（generic↔async）；`mbarrier`（异步完成） |
| **distributed shared `.shared::cluster`** | cluster 内任意 CTA | cluster 内任意 CTA | cluster mbarrier / `barrier.cluster.*` + `fence.proxy.async.shared::cluster` |
| **tensor memory（TMEM，SM100+）** | `tcgen05.mma`（异步）、`tcgen05.st` | `tcgen05.ld`、`tcgen05.mma` | `tc_gen5_commit` → mbarrier；accumulator RAW 靠 `AsyncToken` modref；load→mma / store→mma 靠 `TMemBarrierInsertion` 的 `ttg.barrier local` |
| **global memory** | 任意线程；TMA store | 任意线程；TMA load | 带 global 位的 `bar.sync`；`cp.async.bulk.wait_group`；带 scope 的 atomic |

有几个容易踩坑的事实：

1. **TMEM 不由 `MembarAnalysis` 覆盖。** Membar pass 只跟踪 shared memory 的
   allocation slice。TMEM 的顺序由三样东西负责：穿过
   `tc_gen5_mma`/`tmem_load`/`tmem_store` 的 `AsyncToken` 依赖、
   `tc_gen5_commit`→`wait_barrier` 的 mbarrier、以及专门的 `TMemBarrierInsertion`
   pass（见 §7.2）。这也正是 `arrive_barrier` lowering 会手动发射一个*前置*
   `ttg.barrier` 的原因——`BarrierOpToLLVM.cpp:340` 的注释指出 MemBar 覆盖不了
   TMEM。

2. **async proxy 是真正独立的排序域，但要分清方向。** generic proxy 和 async
   proxy 对同一块 shared memory 的访问之间不会自动建立顺序。但 `fence.proxy.async`
   在 Triton 里主要用于 **generic → async 方向**：generic proxy 先写 smem（如
   `local_store` / `stmatrix`），随后 async proxy 读它（TMA store、WGMMA、MMAv5、
   `TMEMCopy`）。这由独立的 pass 处理（`FenceInsertion` / `ProxyFenceInsertion`），
   不是 barrier 分析。
   - **反方向（async → generic）通常*不*需要显式 `fence_async_shared`**：TMA load
     （`cp.async.bulk`）写完 smem 后被普通 `local_load` 读取，其可见性由 mbarrier
     completion（`wait_barrier`）加上 PTX 对 `cp.async.bulk` 完成时的*隐式*
     generic-async proxy fence 覆盖。所以典型 TMA load 路径
     `init/expect → async_tma_load → wait_barrier → local_load` 里**没有**额外的
     `fence_async_shared`。证据：`ProxyFenceInsertion.cpp` 把 TMA load 的写归入
     `proxyBlockInfo.syncWriteSlices` 而非累积 generic 访问的 `blockInfo`，fence
     只插在 async op *之前*以防护*先前的 generic 写*；async-proxy read 那条分支的
     注释也明说它只是 "Safe fallback ... when the earlier FenceInsertionPass did
     not place a fence"。

---

## 3. 完整清单（一）：execution / CTA / warp / cluster barrier

每一条给出：Triton op、它变成的 PTX/NVVM、粒度、守护的存储、触发上下文、守护范围、
完成时的可见性。

### 3.1 `ttg.barrier` —— Triton 的 `__syncthreads()`

- **Op**：`TTG_BarrierOp`，`include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:734`。
- **Lowering**：→ `mlir::gpu::BarrierOp`（`MemoryOpToLLVM.cpp:255`）→ NVVM
  `barrier0` → PTX `bar.sync 0`（aligned，CTA）。
- **粒度**：**per CTA**。block 的所有线程会合。
- **守护的存储**：由强制的 `addrSpace` **位掩码（bitmask）** 描述——`none`（仅
  控制）、`local`（smem）、`global_read`、`global_write`、`tensor_read`、
  `tensor_write`、`all`。可组合，如 `local|tensor_write`（`TritonGPUOps.td:734`
  描述）。
  - **关键：这个 bitmask 是 Triton IR 层的语义标签，当前 NVIDIA lowering 并不据此
    发不同的 PTX fence。** `BarrierOpConversion` 直接 `replaceOpWithNewOp<
    mlir::gpu::BarrierOp>(op)`（`MemoryOpToLLVM.cpp:255`），完全不读 `getAddrSpace`
    ——无论 bitmask 是什么，NVIDIA 后端都发同一个 `bar.sync 0`。读者不要把这个
    bitmask 直接等同为 PTX 指令选择。（对比：AMD 后端*确实*读 `hasLocal()` /
    `hasGlobalRead()` 发不同 barrier，见 `third_party/amd/.../MemoryOpToLLVM.cpp`；
    所以这个契约对 AMD 是活的，对 NVIDIA 主路径不是。）
  - `tensor_read/tensor_write` 同理是抽象标签，并不代表 PTX 有一条"TMEM barrier
    指令"直接对应它。NVIDIA 后端里主路径几乎都是 `ttg.barrier local`。
- **上下文**：由 `MembarAnalysis` 在 shared memory RAW/WAR/WAW hazard 处自动插入，
  或由某些 lowering 显式发射来给异步区间"封边"（cluster mbarrier、tcgen05 alloc）。
- **守护范围**：任意线程在 barrier *之前*发起的所有 shared/tensor/global 操作
  （按 bitmask），在 barrier 之后对*每个*线程都已完成且可见。
- **可见性**：所选地址空间的完整 CTA 内可见性。这是双向 fence + 执行会合。
- **为什么需要**：SIMT 线程跨 warp 异步运行；没有会合，warp 1 的线程 T2 可能在
  warp 0 的线程 T1 写 smem[i] 之前就读它（RAW），或在 T1 读到旧值之前就覆盖它
  （WAR）。

### 3.2 命名 barrier `bar.sync N, cnt` 与 `bar.warp.sync` —— warp specialization

- **在 C++ 中发射**（不是独立 op）：`ConvertWarpSpecializeToLLVM.cpp:79`，经
  `NVVM::BarrierOp(handle, numThreads)`。
- **粒度**：CTA 的一个**子集**——一个 warp（`bar.warp.sync`，l.75）或一个
  warp-group / 命名 cohort（`barrier.sync N, numThreads`，l.247）。
- **守护的存储**：仅参与线程之间的 shared memory。
- **上下文**：warp-specialized kernel 把 CTA 拆成一个 default warp-group 和一或
  多个 partition warp-group（producer/consumer）。每个 cohort 需要*独立于*其它
  cohort 会合——一个完整的 `bar.sync 0` 会死锁，因为各 partition 跑不同代码。命名
  barrier 每 CTA 提供 ≤16 个独立会合点。
- **守护范围 / 可见性**：只在 barrier `N` 上注册的 `numThreads` 个线程之间。其它
  warp 自由前进。
- **为什么需要**：producer warp（发 TMA load）和 consumer warp（跑 MMA）必须在不
  强制整个 block 锁步的前提下交接 buffer。

### 3.3 Cluster barrier —— CGA 级会合（SM90+）

- **Op**：`ttng.cluster_arrive`（`TritonNvidiaGPUOps.td:81`）、
  `ttng.cluster_wait`（`:87`）、`ttng.cluster_barrier`（`:92`）。
- **Lowering**：`cluster_arrive` → NVVM `ClusterArriveOp`/`ClusterArriveRelaxedOp`
  → `barrier.cluster.arrive[.relaxed]`（`ClusterOpsToLLVM.cpp:49`）；`cluster_wait`
  → `barrier.cluster.wait`（`:55`）。`cluster_barrier` 有**两条 lowering 路径**
  （`ClusterOpsToLLVM.cpp:192-238`）：
  - **默认路径**（无 `kClusterBarrierMbarOffsetAttrName` 属性，`:196-212`）：直接
    `createClusterArrive` + `createClusterWait`，即 `barrier.cluster.arrive/wait`
    对。这是普通多 CTA 协作的常规形态。
  - **特殊路径**（带 mbar offset 属性，`:214-238`，主要出现在 warp-specialized
    kernel）：才展开成 `NVVM::BarrierOp`（CTA）+ 逐 peer-CTA 的
    `mbarrier.arrive...cluster` + `mbarrier.try_wait.parity` 轮询 + 收尾
    `NVVM::BarrierOp` 的 shared-mbarrier 方案。
  换言之，"CTA barrier + peer mbarrier 序列"不是 `cluster_barrier` 的通用形态，
  只有需要 cluster-barrier mbar 的特殊路径才走它。
- **粒度**：**per cluster**——CGA 内每个 CTA 都参与。
- **守护的存储**：distributed shared memory（`.shared::cluster`）——一个 CTA 对另
  一个 CTA 的 smem 窗口的写。
- **上下文**：多 CTA 协作 kernel（cluster 级 reduction、DSMEM producer/consumer）。
  `arrive`/`wait` 分离，让一个 CTA 先发信号表示就绪，再做独立工作，然后才阻塞。
- **为什么拆成 arrive/wait**：cluster 会合延迟高（跨 CTA）；用 arrive-到-wait 之间
  的窗口重叠有用计算可以隐藏它。`relaxed` 在只关心计数时丢掉内存排序那一半。
- **`fence.mbarrier_init.release.cluster`**（`ttng.fence_mbarrier_init_release_cluster`，
  `TritonNvidiaGPUOps.td:66` → `BarrierOpToLLVM.cpp:44`，单线程 predicated）：
  让一个 CTA 里的 `mbarrier.init` 排在任何 cluster peer 对它 arrive 之前。它建立的
  invariant 是"barrier 存储在被 cluster 范围使用之前已初始化"——没有它，peer CTA
  可能对未初始化的内存 arrive。这不是 barrier completion 问题，而是 initialization
  visibility 问题。配套逻辑见 `ClusterBarrierInsertion.h`。

---

## 4. 完整清单（二）：mbarrier 家族 —— shared memory 上解耦的 arrive/wait

mbarrier（`mbarrier.*`）是一个**驻留在 shared memory 中**的 64-bit 对象，跟踪两个
计数：一个 *arrival* 计数和（可选的）一个 *transaction/byte* 计数。它的目的是把
**发信号与阻塞解耦**：producer 直接 `arrive` 而不停下；consumer 在一个 *phase
parity*（相位奇偶）位上 `wait`，该位在预期的 arrival + bytes 到齐时翻转。这是异步
pipeline 的基础。它不是"编译器脑内的 token"，而是显式的共享内存对象。

这些 op 都带 `MBarrierOpInterface`（`TritonGPUOpInterfaces.td:45`），使分析能够
通用地找到 barrier memdesc 操作数。

### 4.1 生命周期 op（都作用在 `.shared::cta` 的一个 mbarrier 对象上）

| Op | td:行 | PTX | *效果*的粒度 | 备注 |
|---|---|---|---|---|
| `ttng.init_barrier` | `:257` | `mbarrier.init.shared::cta.b64 [addr], count` | 1 线程发起，CTA/cluster 可见 | `count` = 预期 arrival 数；predicated elect-one + leader-CTA（`BarrierOpToLLVM.cpp:138`） |
| `ttng.inval_barrier` | `:277` | `mbarrier.inval.shared::cta.b64 [addr]` | 1 线程 | 复用 smem 槽位前必须调用（PTX 契约）；退役 barrier 状态机，不是清数据 |
| `ttng.arrive_barrier` | `:365` | `mbarrier.arrive[.cluster].b64 _, [addr][,count]` | 1 线程为多方发信号 | 减少 pending；scope 经 broadcast mask 升到 cluster（`:349`）；发射前置 `ttg.barrier`（`:340`），因为 MemBar 看不到 TMEM |
| `ttng.barrier_expect` | `:293` | `mbarrier.arrive.expect_tx[.cluster].b64 _, [addr], size` | 1 线程 | 为 TMA 设置预期 *byte* 计数；前置一个 `ttg.barrier`（`:207`） |
| `ttng.wait_barrier` | `:317` | SM90+：`mbarrier.try_wait.parity.shared::cta.b64` 轮询循环；SM90 以下：`test_wait`+`nanosleep` | 谁执行谁阻塞（受 predicate 约束） | 在 `$phase` parity 上等待；可选 `$pred`；`$deps` = 在 barrier 完成前可访问的内存 |
| `ttng.async_copy_mbarrier_arrive` | `:404` | `cp.async.mbarrier.arrive` | 1 线程 | 把非 bulk 的 cp.async 完成绑到 mbarrier |

- **守护的存储**：mbarrier 跟踪其填充/排空的 shared memory（或 DSMEM）buffer——
  TMA 载入的 tile、tcgen05 accumulator（经 commit）、pipeline stage buffer。
- **上下文**：软件流水化循环和 warp-specialized kernel。phase 位在每次循环迭代
  翻转，因此同一个 mbarrier 跨 stage 复用而无需重新 init。
- **粒度 / 谁阻塞**：`wait_barrier` 是"谁执行谁阻塞"，并受 `$pred` / leader-CTA
  predicate 约束——它**不是**一个自动的 consumer 集合会合。若需要让整组 consumer
  线程都在此对齐，那是靠 kernel 结构（所有 lane 同构执行该 op）或另配的
  `ttg.barrier`，不是 `wait_barrier` 本身提供的。
- **可见性**：`wait_barrier` 返回时，这些 arrival 所代表的所有写（包括
  `expect_tx` 计数的 TMA async-proxy 字节传输）都已完成且对等待线程可见。**注意**：
  mbarrier 跟踪的是*完成*，不是任意 smem 对 generic proxy 的跨-proxy*可见性*——
  但对 TMA load 这类 `cp.async.bulk` 完成，PTX 保证了随附的隐式 generic-async proxy
  fence，故其数据可被随后的 `local_load` 直接读（见 §2 的方向说明）。
- **为什么用 phase parity 而不是普通计数**：跨 pipeline 迭代复用一个 barrier 需要
  一个在 reset 后仍单调的信号。一个每完整周期翻转一次的 1-bit phase，让 `try_wait`
  能区分"本次迭代的填充"和"上次迭代的填充"，避免 reset 竞争。

### 4.2 绑定到 mbarrier 的异步数据搬运

- `ttng.async_tma_copy_global_to_local`（`:446`）、`ttng.async_tma_gather`（`:564`）：
  TMA **load** gmem→smem；完成时硬件把 `$barrier` 的 byte 计数记账。配合
  `barrier_expect`（设 bytes）+ `wait_barrier`（阻塞）。对应 PTX
  `cp.async.bulk.tensor ... mbarrier::complete_tx::bytes`。
- `ttng.async_shared_store`（`:415`）：distributed-smem store，递减一个 mbarrier
  transaction 计数；需要 ≥2-CTA cluster。
- `ttng.tc_gen5_mma[_scaled]`（`:632`/`:696`）：异步 MMAv5 写入 TMEM；若带 barrier
  操作数，op 会对它触发 commit/arrive（见 §6）。
- `ttng.clc_try_cancel`（`:110`，Blackwell SM100+，CLC = Cluster Launch Control）：
  发 `clusterlaunchcontrol.try_cancel`，用于动态 persistent kernel 原子地取消一个
  pending cluster launch。它带 `MBarrierOpInterface`，语义和 TMA load 同构——
  **异步写 result buffer**（16-byte 对齐的 `2xi64` smem），**完成时 signal mbarrier**
  （op 描述 `:114-117`；lowering 见 `BarrierOpToLLVM.cpp:407`）；消费方同样在
  `wait_barrier` 等待，随后 `clc_load_result`（`:136`）把结果读进寄存器。归入
  completion tracking：等的是"CLC 结果写好了没"，不是线程会合。
  - 典型模式里通常还会配 `barrier_expect 16` 设置 transaction byte count——因为
    result 是 `2xi64` = 16 bytes，CLC 完成时按 `complete_tx::bytes` 递减该计数，
    `wait_barrier` 才能正确判定"16 字节的结果已写完"。测试
    `test/TritonNvidiaGPU/membar-cluster.mlir:494` 就是
    `init_barrier(count=1) → clc_try_cancel → barrier_expect(16)` 这个序列。这也
    说明 CLC 与 TMA load 共用同一套 mbarrier transaction 语义（arrival 计数 + byte
    计数），不是特例。
- TMA **store**（`async_tma_copy_local_to_global` `:511`、`reduce`、`scatter`）
  **没有** mbarrier——它们用 bulk-async **group** + `async_tma_store_wait`
  （`:620`，`cp.async.bulk.wait_group`），是类似 wgmma 的计数模型。
  - **`async_tma_store_wait` 的 guard 范围要分清**（op 定义 `:620-627`）：它有
    `pendings`（等到未完成的 store group ≤ pendings）和 `read_only` 两个属性。
    `read_only` 置位时，它**只等 TMA store 从 shared memory 的读取完成**——目的是
    让 shared buffer 可以被**重新写入**（下一轮 producer 复用该 stage buffer）。
    它**不等价于**完整的 global store 完成：此时数据可能还没落到 global memory，只
    保证 TMA engine 不再读这块 smem。若要保证 global 端可见（如后续 global load
    依赖），需要不带 `read_only` 的等待（等 store 真正完成）。这个区分正是 pipeline
    里"尽早释放 smem 以提升 buffer 复用率"的关键杠杆。

一条典型 TMA load 路径（Hopper+ 最重要的一条）：descriptor/tensormap 就绪 → smem
目标 buffer 已分配 → `barrier_expect(bytes)` 先 arm barrier → 发起
`async_tma_copy...` → consumer 在 `wait_barrier(phase)` 等待 → 完成后 local_load /
下游使用 → 最后 `inval_barrier`。这里有两层问题：**completion**（TMA engine 何时
真把 tile 写进 smem，由 mbarrier wait 解决）与 **proxy ordering**（TMA 属 async
proxy，普通 smem 读写属 generic proxy）。跨 proxy 排序**要看方向和指令族**：TMA
load 这一条 async→generic 的 completion 与可见性由 `wait_barrier` + `cp.async.bulk`
随附的隐式 proxy fence 一起覆盖，**不需要**在这条路径里额外插 `fence_async_shared`；
真正需要显式 `fence.proxy.async` 的是 generic→async 方向（见 §2、§5）。

---

## 5. 完整清单（三）：Fence —— 纯内存排序，不会合

**fence** 排序*发起线程*（或某个 scope）的内存操作，而**不**阻塞等待其它线程。
它回答"让我此前的写变得可发布"，而不是"等所有人"。

> **方向很关键**（见 §2）：显式的 `fence.proxy.async` 主要用在
> **generic → async** 方向（generic 写 smem，随后 async proxy 读它）。反方向
> **async → generic**（如 TMA load 写完 smem 再由 generic `local_load` 读）通常
> *不*需要显式 fence——它由 completion 机制随附的隐式 proxy fence 覆盖，具体看该异步
> 指令族的 PTX 语义（`cp.async.bulk` 完成即带隐式 generic-async fence）。

### 5.1 `ttng.fence_async_shared` → `fence.proxy.async.shared::{cta,cluster}`

- **Op/lowering**：`TritonNvidiaGPUOps.td:52` → NVVM `FenceProxyOp(async_shared,
  space)`（`BarrierOpToLLVM.cpp:68-73`）。`bCluster` 为真时 space = cluster。
- **粒度**：**per thread** 发起，但建立的排序被整个 CTA/cluster 依赖。不是会合。
- **守护的存储**：shared memory，跨越 **proxy 边界**。重点不是"shared 本身"，而是
  shared 被两个 proxy 访问。
- **上下文**：由 fence 插入 pass 在 generic proxy 写、async proxy 后续读同一块
  smem（或反向）的边界处插入。async-proxy 写方 = TMA load / CLC
  （`ProxyFenceInsertion.cpp:32`）；async-proxy 读方 = `WarpGroupDot`、`MMAv5`、
  `TMEMCopy`、TMA store（`:47`）。
- **可见性**：fence 之后，发起线程的 generic-proxy shared 写对 async-proxy
  consumer 可见（反之亦然）。SM90+。
- **为什么要与 `bar.sync` 分开**：`bar.sync` 在 CTA 范围排序 generic proxy，但对
  async proxy 眼中的内存视图只字不提。两个 proxy 有各自的 coherence；只有
  `fence.proxy.async` 桥接它们。缺了它 → 即便 `__syncthreads` "通过"了，TMA engine
  仍读到过期的 smem。PTX/CUDA 官方文档对 async proxy 也明确要求 proxy fence。

**两个插 fence 的 pass 及其区别**（这是很多人没分清的地方）：

- `FenceInsertion`（`lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`）：
  顶部注释说它"在所有其它 pass 之后运行，插入 fence 以保证内存操作跨 generic 和
  async proxy 正确排序"。它只在 **compute capability ≥ 9.0** 生效，由
  `DotOpInterface` 驱动——沿 use-def 链找 dot 的 operand 是否来自 generic proxy 的
  shared 写（如 convertlayout 带 `sts`/`stmatrix`），是则在 `wgmma` 前补
  `fence_async_shared`。注释里的模式：`generic(convertlayout) + fence + async(wgmma)`。
- `ProxyFenceInsertion`（`.../ProxyFenceInsertion.cpp`）：更一般、alias-aware 的
  补救版本，复用与 Membar 相同的数据流骨架，在更广的 async-proxy 读/写集合上插
  `ttng.fence_async_shared`（`:107`）。一个 fence op 出现在流中会清掉 pending 依赖
  （`:114`）。

### 5.2 `ttng.tensormap_fenceproxy_acquire` → `fence.proxy.tensormap::generic.acquire.gpu`

- **Op**：`TTNG_TensormapFenceproxyAcquireOp`，`TritonNvidiaGPUOps.td:1114`。
  Lowering 见测试 `test/Conversion/tritonnvidiagpu_to_llvm.mlir:545`
  （`fence.proxy.tensormap::generic.acquire.gpu`）。也可参见 `TMAToLLVM.cpp:205`，
  predicated 到第一个 warp，后面跟一个 `cp.async.bulk.commit_group` +
  `cp.async.bulk.wait_group.read 0` 的 workaround。
- **它经常被误读成 barrier，其实不是。** 它是对 tensormap / descriptor 对象的
  acquire fence。
- **粒度**：**`.gpu` scope**（device 级）——因为 tensormap 可能在 global memory 里
  构建，并被 TMA 单元 device-wide 消费。
- **守护的存储**：tensormap / descriptor 对象本体。
- **上下文**：先有 `tensormap_create` / `tensormap_replace`，再有
  `tensormap.cp_fenceproxy...release`，使用方在真正用 descriptor 前做 acquire。
  即 kernel 在运行时改了 TMA descriptor 时，这个 fence 让对 descriptor 的写排在
  TMA 单元读它之前。
- **为什么是 `.gpu` 而不是 `.cta`**：descriptor 由异步 copy 引擎消费，其视图是
  device-scoped，不是 block-scoped。它解决的是"元数据对象何时可被另一方合法使用"，
  不是"tile 数据到了没"。

### 5.3 `fence.mbarrier_init.release.cluster`

见 §3.3 —— 让 `mbarrier.init` 排在 cluster 范围的 arrive 之前。release 语义，使初始化
状态发布给 peer CTA。

### 5.4 说明：atomic 的 `sem`/`scope` 不在本文主范围

`tt.atomic_rmw` / `tt.atomic_cas` 上的 `.acquire` / `.release` / `.acq_rel` 与
`.cta` / `.gpu` / `.sys` 是**内存同步语义**（memory consistency 的 acquire/release
顺序 + 作用 scope），它们确实影响可见性，但**不是 barrier / fence 原语**：没有会合，
也不是独立发射的 fence op，而是附着在原子指令本身上的修饰符。本文聚焦 barrier 与
显式 fence，atomic 的 acquire/release 排序只在需要时对照 PTX ISA §8 内存模型即可。

---

## 6. 完整清单（四）：异步引擎完成 —— commit / wait group 模型

这些不是可见性意义上的"barrier"；它们保证 **liveness**：别在异步引擎产出结果之前
就消费它。模型是每引擎一个 *group* FIFO；你 commit 一个 group，然后等到未完成的
group ≤ N。

### 6.1 cp.async（SM80+，非 bulk）

- `ttg.async_copy_global_to_local` → token；`ttg.async_commit_group`
  （`TritonGPUOps.td:71` → `cp.async.commit_group`，`LoadStoreOpToLLVM.cpp:1835`）；
  `ttg.async_wait`（`:47`，`[MemWaitOpTrait]` → `cp.async.wait_group N`，`:1817`）。
- **粒度**：PTX 语义是 **per executing thread** 的 cp.async-group——每个执行线程
  维护自己的 group 序列。Triton 通常让所有 lane 同构执行，但 `wait_group` 本身
  **既不提供 warp/CTA 会合，也不排序其它内存操作**，它只保证本线程发出的 cp.async
  组完成。
- **明确不同步 CTA**（op 描述原文："does not provide any synchronization in the
  CTA"）——若其它 warp 要看到载入的 smem，仍需一个 `ttg.barrier local`。这正是
  `MemWaitOpTrait` 存在的原因：Membar pass 会在 wait *之后*插 CTA barrier（§7）。
  所以 `wait_group` 更像 completion，`barrier local` 更像 cross-thread
  execution/memory sync，两者不能混。

### 6.2 wgmma / MMAv3（Hopper，warp-group 级）

- `ttng.warp_group_dot`（`:203`，`isAsync`）→ `wgmma.fence.sync.aligned`
  （`WGMMA.cpp:261`）+ `wgmma.mma_async` + `wgmma.commit_group.sync.aligned`
  （`:360`）。pipeline 注释见 `WGMMAPipeline.cpp`。
- `ttng.warp_group_dot_wait`（`:239`）→ `nvgpu.wgmma_wait_group` →
  `wgmma.wait_group.sync.aligned <pendings>`（`NVGPUToLLVMPass.cpp:340`）。
- **粒度**：**per warp-group（128 threads），`.aligned`**——4 个 warp 都须收敛。
  `wgmma.fence` 在异步 MMA 读取 accumulator 寄存器/smem operand 之前排序它们。
- **为什么是 fence + commit + wait 三件套**：`fence` 把输入发布给 wgmma proxy；
  `commit_group` 封一批；`wait_group` 阻塞到 accumulator 寄存器可安全读。重叠方式
  = 发多个 MMA，只 wait 一次。若 shared operand 来自 generic proxy producer，前面
  还可能要先有 `fence_async_shared`。典型链条：
  `generic shared write → fence_async_shared → wgmma.fence → wgmma.mma_async →
  wgmma.commit_group → wgmma.wait_group`。

### 6.3 tcgen05 / MMAv5（Blackwell，SM100+）—— mbarrier 完成

- `ttng.tc_gen5_mma[_scaled]`（`:632`/`:696`）：异步 MMA，读 smem/TMEM，写一个
  **TMEM** accumulator。`is_async=false` → 同步执行，barrier 必须缺席。
- `ttng.tc_gen5_commit`（`:774`）→ `tcgen05.commit.cta_group::{1,2}.mbarrier::
  arrive::one.shared::cluster.b64`（`MMAv5.cpp:345`）：让一个 **mbarrier** 跟踪此前
  所有异步 tcgen05 op 的完成，完成后自动对它 arrive-one。随后一个普通
  `wait_barrier` 在它上面阻塞。
- tcgen05 alloc/dealloc 由 `NVVM::BarrierOp` 封边（`NVGPUToLLVMPass.cpp:535`）。
- **粒度**：warp-group / cta_group（`two_ctas` 时为 1 或 2 个 CTA）。单 CTA 时常看成
  CTA-local completion，但定义明确支持 cluster multicast barrier，尤其输入来自
  multicast TMA descriptor 时。
- **为什么完成经由 mbarrier 而不是像 wgmma 那样的 group 计数**：MMAv5 写 TMEM，
  而 TMEM 由 `tcgen05.ld` 在*不同*的调度上读回；一个 shared memory mbarrier 是
  pipeline 的 `wait_barrier` 和 consumer 都能观察到的公共货币。accumulator 本身的
  TMEM RAW 另由穿过这些 op 的 `AsyncToken` modref 排序，load→mma / store→mma 的
  hazard 则由 `TMemBarrierInsertion` 负责（§7.2）。

---

## 7. 自动插入 —— Triton 如何决定同步放在哪

### 7.1 `MembarAnalysis`（`lib/Analysis/Membar.cpp`，`include/triton/Analysis/Membar.h`）

模块级运行（`TritonGPUToLLVM.cpp:111`）。它是一个在 **shared memory allocation
slice** 上的数据流分析。

- **它守护什么**：两个 op 的 smem allocation slice *相交*（`Membar.h:131`
  `isIntersected`）时的 hazard：
  - **RAW**（先写后读）—— 读方会看到过期数据。
  - **WAR**（先读后写）—— 写方在读方消费前覆盖。
  - **WAW**（先写后写）—— 两个写方的顺序。
  - RAR 从来不是 hazard。
  相交是 slice 级的（物理区间 + layout 子偏移），因此不相交的 buffer 正确地*不*被
  同步。
- **它插什么**：永远是一个 `AddrSpace::Local` 的 `ttg.barrier`（`Membar.cpp:243`）
  —— 一个 CTA 级 `bar.sync`。它从不插 mbarrier、fence 或 cluster barrier；那些由
  pipeline / warp-specialize lowering 放置。
- **它怎么走**（`MembarAnalysis::update`，`Membar.cpp:281`）：
  1. 若 op *就是*一个同步点（`containsLocalBarrier`，`:247` —— 包括
     `gpu::BarrierOp`、`ClusterBarrierOp`、`ClusterWaitOp`、
     `WarpSpecializePartitionsOp`、`ArriveBarrierOp`、`BarrierExpectOp`、
     `TCGen5CommitOp`）→ 清掉 pending slice（`blockInfo->sync()`）。这就是该 pass
     *识别*由其它两个机制建立的同步、从而避免重复同步的方式。
  2. 若 op 带 `MemWaitOpTrait`（异步 wait）且在它下一个内存效应前没有同步点 →
     在 wait *之后*插 barrier（`:292`），做合并。
  3. scratch op（reduce、convert-layout）：按写→读处理；warp-synchronous
     convert-layout（`warp` 维上的 `isCvtDimSync`）内部只做 warp.sync，因此*不*清
     CTA 级依赖（`:378`）。
  4. 一般情形：若当前 op 的效应与 pending 相交 → 在 op 前插 barrier。
- **后端过滤 `canSkipBarSync`**（`TritonGPUToLLVM.cpp:296`）：抑制两个已由构造保证
  顺序的 op 之间的 barrier —— 两个单线程 mbarrier-config op，或紧跟其
  `WaitBarrierOp` 的一个 `TMALoadLike`。
- **跨函数**：后序调用图遍历；被调方的 `BlockInfo` 翻译到调用点（`Membar.h:180`），
  因此调用方无需围绕 `tt.call` 手动加 barrier。

### 7.2 `TMemBarrierInsertion`（`lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp`）

这是专门覆盖 **TMEM** 路径的 hazard pass —— 因为 §2 说过 Membar 不跟踪 TMEM。

- **它守护什么**：TMEM slice 上的 hazard。判定条件（`:75`）：
  `requiresBarrier = war || raw || waw || loadToMma || storeToMma`。
- **关键的非对称性**（源码注释 `:66-69`）：`load->mma` 和 `store->mma` 依赖需要
  barrier；但 `mma->load/store` **不需要**额外 barrier —— 因为后面会有一个
  `mbarrier wait` 保证该 op 在任何线程到达 load/store 之前已完成。也就是说
  completion 方向由 §6.3 的 mbarrier 覆盖，只有喂给 MMA 的输入方向需要这个 pass 补
  CTA 级顺序点。
- **它插什么**：`ttg.barrier local`（CTA 级顺序点）。
- **这里看上去会怪**：冲突对象明明是 TMEM，为什么插的是 `barrier local`？原因是
  Triton 在这一层用 CTA barrier 来表达"线程组执行顺序 + 相关共享状态同步点"；TMEM
  真正的异步 completion 仍由后面的 mbarrier wait 负责。这条 barrier 更像"在发下一
  类访问之前，不要让 CTA 内线程越过这个协议边界"。所以它不是"TMEM 专用 PTX
  fence"，而是编译器在 Triton IR 层插入的 CTA 级 hazard separator。

### 7.3 fence 插入 pass

见 §5.1 —— `FenceInsertion`（cc≥9.0、`DotOp` 驱动、`generic→fence→wgmma`）与
`ProxyFenceInsertion`（alias-aware 通用版）在 generic-proxy ↔ async-proxy 的 smem
边界插 `ttng.fence_async_shared`。与 Membar 分开，因为它守的是 *proxy* hazard 而非
*数据* hazard —— 两者正交，可以在同一程序点同时需要。

---

## 8. 为什么需要同步 —— 因果链

上面每一个原语都追溯到四个硬件现实之一：

1. **SIMT warp 被独立调度。** 一个 block 的 warp 0 和 warp 3 不是锁步的；没有
   `bar.sync`，它们对 shared memory 的访问之间没有保证的顺序。→ 用 CTA barrier 处理
   smem RAW/WAR/WAW。

2. **异步引擎与 SM 并发运行。** cp.async、TMA、wgmma、tcgen05 立即返回"已发起"，
   稍后才完成。在完成前读目标返回垃圾。→ commit/wait-group 和 mbarrier completion。

3. **内存有多个 proxy，各自 coherence 独立。** generic proxy（SM load/store）和
   async proxy（copy/tensor 引擎）不会自动看到彼此对同一 shared memory 的写。→
   `fence.proxy.async`。

4. **排序有 scope，scope 越宽代价越大。** warp < warp-group < CTA < cluster <
   GPU。只把写发布到读方需要的那么宽，是一个性能杠杆 —— 这就是为什么 Triton 在
   `ttg.barrier` 上暴露 `addrSpace` bitmask、只经 broadcast mask 把
   `.cta` 升到 `.cluster`、并对 sub-CTA cohort 用 `bar.warp.sync` / 命名 barrier。

任一环缺失时，反复出现的失败模式是一个**依赖调度的竞争** —— 在某个 tile 大小 / 某
块 GPU / 某个 occupancy 下通过，换一个就失败。这正是为什么插入由保守的数据流分析
（Membar、TMemBarrier、ProxyFence）完成，而不是靠手工放置：分析在安全侧过近似，再
由 `canSkipBarSync` / `containsLocalBarrier` 机制把冗余的那些抠回来。

---

## 9. 统一心智模型与速查

### 9.1 最该抓住的三个问题

看任何一处同步，问这三问最稳：

1. 我现在等的是"**线程到齐**"吗？→ `ttg.barrier` / cluster barrier / warp sync。
2. 我现在等的是"**异步硬件动作完成**"吗？→ `mbarrier.wait` / `cp.async.wait_group`
   / `wgmma.wait_group`。
3. 我现在补的是"**跨 proxy / 跨域的顺序和可见性**"吗？→ `fence_async_shared` /
   `tensormap_fenceproxy_acquire` / `fence_mbarrier_init_release_cluster`。

很多人把第 2 和第 3 混掉，这是看 TMA / Hopper / Blackwell 路径最容易乱的地方：
completion 只说明"硬件动作结束了"，proxy fence 才说明"两个访问域之间建立了
happens-before"。

### 9.2 速查表

| 原语 | 存储 | 粒度 | 阻塞? | 完成时保证 |
|---|---|---|---|---|
| `ttg.barrier` / `bar.sync 0` | IR 层依赖标签（bitmask）；NVIDIA 主路径几乎都是 `local` CTA barrier | per CTA | 是 | CTA 内可见（NVIDIA 后端不按 bitmask 发不同 fence，统一 `bar.sync 0`） |
| `bar.sync N`（命名） | smem | 线程子集 | 是 | 注册线程间可见 |
| `bar.warp.sync` / `__syncwarp` | smem/regs | per warp（32） | 是（重收敛） | warp lane 间可见 |
| `wgmma.fence`/`commit`/`wait_group` | regs/smem | per warp-group（128） | 仅 wait | 异步 MMA 输入已发布 / accumulator 就绪 |
| `mbarrier.arrive` / `arrive_barrier` | smem mbarrier | 1 发起，效果 CTA/cluster | 否 | 发信号：arrival/bytes |
| `mbarrier.try_wait` / `wait_barrier` | smem mbarrier | 谁执行谁阻塞（受 predicate 约束，非自动会合） | 是 | 被跟踪的填充完成且可见 |
| `barrier.cluster.arrive`/`wait` | DSMEM | per cluster | wait 阻塞 | cluster CTA 同步 |
| `tc_gen5_commit` → mbarrier | TMEM→smem | warp-group/cta_group | 经 wait | tcgen05 op 完成 |
| `cp.async.commit/wait_group` | smem | per executing thread（不会合、不排序其它内存） | 仅 wait | cp.async load 到位（非 CTA 可见） |
| `cp.async.bulk.wait_group` / `async_tma_store_wait` | smem/global | 发起线程 | 是 | TMA store group 完成；`read_only` 时只等 store 读完 smem（允许重写 buffer），非完整 global 落地 |
| `clc_try_cancel` → mbarrier | smem（result + mbarrier） | 1 发起 | 否（经 wait） | 异步写 CLC result，完成时 signal mbarrier（Blackwell 动态 persistent kernel） |
| `fence.proxy.async.shared` | smem | thread（scope cta/cluster） | 否 | generic↔async proxy 排序 |
| `tensormap_fenceproxy_acquire` | descriptor/global | GPU | 否 | tensormap 写对 TMA 单元可见 |
| `fence.mbarrier_init.release.cluster` | smem mbarrier | thread（cluster） | 否 | init 在 cluster arrive 前发布 |
| `TMemBarrierInsertion` 插的 `ttg.barrier local` | TMEM 使用顺序 hazard（形式上 local） | per CTA | 是 | hazard 隔离（非 TMEM completion 本体） |

### 9.2b 补充：ARef 不是新原语，而是同步抽象（NVWS 层）

读 NVWS IR 时会看到 `nvws.aref.{create,put.enter,put.exit,get.enter,get.exit}`，
容易误以为是另一类同步原语。它不是。`Aref`（Asynchronous Reference）是一个
**meta-type**：`baseType` 里包着底层 buffer（`TritonGPUOps` 的 `MemDescType`，
所以它**知道**自己包的是 SMEM 还是 TMEM，见 `LowerAref.cpp:511-517` 直接读
`getMemorySpace()` 判 TMEM），但**不携带任何硬件同步语义**——类型定义就写明
"Lowers to the underlying type, and operations that use this should insert
appropriate barriers during lowering"（`NVWSTypes.td:34`）。

它解决的是**编译期约束**而非硬件约束：warp specialization 与 software pipelining
要在多 partition / 多 stage 间正确配对 barrier，裸操作 mbarrier 时 phase 对齐、
empty/full 配对、arrival count 极易出错。ARef 把"一块 buffer + 它的 producer/consumer
协议 + stage/phase + token 配对"绑成一个 SSA 值，让 `InsertAref` / `AssignStagePhase`
/ `LowerAref` 能在高层做变换，最后统一降解到 §4 的硬件 mbarrier。

`LowerAref` 的降解要点（不是"只有那两个 mbarrier"，而是按 async kind 分派）：
- create 时统一建两组 mbarrier：`emptyMbars`（count=consumerPendingCount）与
  `fullMbars`（count=producerPendingCount）（`LowerAref.cpp:244,261-262`）。
- `put.enter` 等 `empty`（`:363`）、`get.enter` 等 `full`（`:438`）——经典 empty/full
  环形缓冲。
- exit 的 full-arrive **按 `async_ops` 分派**（`insertArriveBarrier`，`:470-495`）：
  `NONE`/`WGMMA` → 显式 `arrive_barrier`（`:479`）；`TC5MMA`/`TMEMCopy` →
  `tc_gen5_commit`（`:483`）；**`TMALoad` → 什么都不发，arrive 由硬件按
  `complete_tx::bytes` 完成**（`:486-488`）。TMA 路径还会在 `put.enter` 先建
  `barrier_expect(txCount)`（`:337-338`），即 §4 的 TMA expect/HW-arrive。
- 还会在 generic↔async 交界补 `ttng.fence_async_shared`：仅当 producer 是 generic
  proxy（`NONE`）、buffer 非 TMEM、且下游 consumer 是 MMAv5 时插（`:505-535`）；
  get.exit 侧对 TMA producer 对称判定（`:552+`）。方向与 §5 的 generic→async 一致。

一句话结论：**ARef 不是新的硬件同步原语；它是 NVWS 层把 producer/consumer、
stage/phase、token 配对封装起来的同步抽象。lowering 主要展开成 empty/full mbarrier
协议，并按 async kind 补上 TMA expect/HW-arrive、tcgen05 commit、以及必要的
proxy fence（`fence_async_shared`）。** 因此本笔记不把它列进原语表，只在此标边界。

再往上一层是 **software pipeliner 的 stage/phase 轮转与 enter/exit 配对**（pipeline
pairing）：跨迭代做 stage 轮转、绕回时 `phase ^= 1`（`AssignStagePhase.cpp:315-330`
的 `XOrIOp`+`SelectOp`），以及用 token 找 matching enter/exit（`LowerAref.cpp:380`）。
它只**决定喂给 `wait_barrier %bar, %phase` 的 `phase` 参数取哪个 parity、这次用哪个
buffer slot**，本身不产生任何新的 barrier/fence 语义——§4 里 mbarrier 的 phase-parity
不变量是硬件语义，谁去推进那个 phase 是调度层的事。故 pipeline pairing 同样不在本
笔记原语范畴，属于 software pipeliner / warp specialization 主题。

### 9.3 关键文件与源码阅读顺序

建议按此顺序读：

1. `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:734` —— `ttg.barrier` 语义。
2. `include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:257` ——
   `mbarrier` 系列（init/inval/expect/wait/arrive）。
3. 同上 `:52`（`fence_async_shared`）、`:66`
   （`fence_mbarrier_init_release_cluster`）、`:81-92`（cluster barrier）、
   `:1114`（`tensormap_fenceproxy_acquire`）。
4. 同上 `:774` —— `tc_gen5_commit`（Blackwell/TMEM completion）。
5. `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp` —— 为什么 shared
   generic write 到 async reader 前要插 fence（cc≥9.0、DotOp 驱动）。
6. `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp` —— alias-aware
   的通用 proxy fence 补救。
7. `lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:66` —— TMEM
   hazard 判定（`loadToMma`/`storeToMma` 非对称性）。
8. `lib/Analysis/Membar.cpp` + `include/triton/Analysis/Membar.h` —— CTA 级 smem
   hazard 自动插 barrier。
9. `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp` —— ARef 如何按
   async kind 降解成 §4 的 mbarrier 协议 + TMA expect/HW-arrive + tcgen05 commit +
   `fence_async_shared`（配 `NVWSTypes.td:34` 的 meta-type 定义，见 §9.2b）。
9. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/`：`BarrierOpToLLVM.cpp`、
   `ClusterOpsToLLVM.cpp`、`ConvertWarpSpecializeToLLVM.cpp`、`TMAToLLVM.cpp`、
   `LoadStoreOpToLLVM.cpp`、`DotOpToLLVM/{WGMMA,MMAv5}.cpp`、`NVGPUToLLVMPass.cpp`。
10. `test/Conversion/tritonnvidiagpu_to_llvm.mlir` —— Triton op 到 PTX/NVVM 的
    具体对照（如 `:545` 的 `fence.proxy.tensormap::generic.acquire.gpu`）。

### 9.4 需要时再对 PDF 核对的点
- `mbarrier` phase-parity 语义的精确 PTX 措辞：PTX ISA 9.3 §9.7.13（parallel
  synchronization / mbarrier）。
- `fence.proxy` 的 proxy 定义与 async-proxy coherence：PTX ISA 9.3 §8（memory
  consistency model）+ §9.7.12（proxy）。
- cluster / DSMEM scope 与 `cluster.sync()`：CUDA Programming Guide
  "Thread Block Clusters" + "Distributed Shared Memory"。







