# Triton 中的 Barrier、Fence 与 Async Protocol：Scope、存储、可见性

日期：2026-07-02
范围：Triton 会发射的每一个 barrier / fence / 同步原语，以及它们如何组合成具体的
async protocol：它作用在哪种存储类（storage class）上，执行粒度（per thread /
warp / warp-group / CTA / cluster），触发它的上下文，它守护（guard）的范围，
以及它建立的内存可见性保证。
（atomic 的 `.acquire/.release` + `.cta/.gpu/.sys` scope 是内存同步语义而非 barrier，
只在 §5.4 简要说明，不在主范围内。）

证据来源：当前源码（正文内联标注 `file:line`）、PTX ISA 9.3 内存模型、
CUDA Programming Guide（`learn_triton/reference/`）。凡属于 PTX 契约而非 Triton
实现细节的断言，都会明确标注。

> 说明：本笔记融合了两轮分析。所有具体的 op 名、行号、pass 名均已对当前源码
> 核实。若源码注释与结论冲突，以源码为准。

## 可视化入口

配套图放在 `learn_triton/docs/barrier_fence_visuals/`：

- [总览与索引](../docs/barrier_fence_visuals/README.md)
- [协议总览教学页](../docs/barrier_fence_visuals/protocol-overview.html)
- [sm80 `cp.async` 教学页](../docs/barrier_fence_visuals/sm80-cp-async.html)
- [sm90 TMA `mbarrier` 教学页](../docs/barrier_fence_visuals/sm90-tma-mbarrier.html)
- [sm90 WGMMA 教学页](../docs/barrier_fence_visuals/sm90-wgmma.html)
- [sm100 `tcgen05` / TMEM 教学页](../docs/barrier_fence_visuals/sm100-tcgen05.html)

---

## 0. 先建立 protocol 视角

这篇文档现在不只回答“有哪些 barrier / fence”，还先回答一个更上层的问题：

```text
protocol = 一套 producer / consumer 交接规则
         = 谁发起
         + 谁完成
         + 谁观察完成
         + 可见性靠什么建立
         + scope 到哪一层
```

所以这里说的 `protocol`，不是单个 `wait` / `barrier` / `fence` 指令名，而是完整的
交接机制。对 TTGIR 来说，这一层主要落在
`协议显化 + hazard repair`：

- `schedule decision` 决定要不要 overlap、分几个 stage、谁先谁后。
- 本文讨论的 `协议显化` 决定这些调度决策最终落成哪套 target 可执行的
  `async copy / async MMA / barrier object / wait / fence` 组合。
- `hazard repair` 再补齐 shared memory、proxy、TMEM reuse 这些 legality 缺口。

如果你想看“这些 protocol 为什么会在这个 target 上出现”，先读
`learn_triton/docs/TARGET_DRIVEN_SCHEDULING.md`；本文只负责把这些 protocol 本身拆开。

如果只背一句短记忆：

```text
mapping = 分工
organization = 衔接
scheduling = 时序
protocol = 交接
```

那本文就是把 `scheduling` 里“交接”这一半拆开讲清楚。

### 0.1 按机制分，Triton/CUDA/PTX 里常见的 protocol 有哪些

下面这张表先给总分类。后文的 barrier / fence / wait 清单，本质上都只是这几类协议的
具体实例。

| 协议族 | 代表机制 | 核心问题 | 在 Triton 里最常见的落点 |
|---|---|---|---|
| execution rendezvous | `__syncwarp`、`__syncthreads`、`ttg.barrier`、`barrier.cluster` | 哪些线程必须先会合 | generic shared-memory 协作 |
| memory ordering | `__threadfence*`、`membar/fence`、acquire/release | 哪些先后顺序必须成立 | atomic / global / cluster 可见性 |
| atomic consistency | `cuda::atomic`、PTX `atom/red` | 同一地址的并发更新怎么合法 | correctness，不是主调度协议 |
| `cp.async` group | `cp.async -> commit_group -> wait_group` | 异步拷贝何时可消费 | Ampere-style pipeline |
| `mbarrier` transaction | `init/expect/arrive/wait/inval` | 异步事务完成如何被共享观察 | Hopper+/Blackwell pipeline 核心 |
| TMA / TensorMap | `cp.async.bulk(.tensor)`、`tensormap.replace` | 大块/张量搬运如何交接 | Hopper+ transport protocol |
| proxy synchronization | `fence.proxy.async`、`fence.proxy.tensormap` | generic 和 async/tensormap proxy 如何对齐 | `ProxyFenceInsertion` 关注点 |
| WGMMA protocol | `wgmma.fence/mma_async/commit_group/wait_group` | warp-group async MMA 如何发射和回收 | Hopper compute protocol |
| TCGen05 / TMEM protocol | `tcgen05.mma/commit/wait/fence::*` | TMEM / TensorCore5 如何完成与交接 | Blackwell compute protocol |

其中最容易混淆的是：

- `wait` 通常只是“观察完成”的一环，不等于整个 protocol。
- `barrier` 通常只是“会合点”或“barrier object”，不等于整个 protocol。
- `fence` 通常只是“补顺序/可见性”，也不等于整个 protocol。

真正要看的，是这套链条是不是完整：

```text
issue -> completion signal -> completion observe -> visibility/order -> reuse / next stage
```

### 0.2 按 target 看，三代主协议有什么差别

同一个 matmul 主循环，在不同 target 上会长成不同协议，不是因为“指令名换了”，而是
因为执行单元、异步引擎、完成模型、proxy 模型都变了。

| target | 主执行/搬运单元 | 主要 completion model | 典型协议链 | 新增约束 |
|---|---|---|---|---|
| `sm80/sm86` | `mma.sync` + `cp.async` | per-thread async-group | `cp.async -> commit_group -> wait_group -> ttg.barrier` | 没有 TMA、cluster、TMEM |
| `sm90` | `wgmma` + TMA + cluster | `wait_group` + `mbarrier` | `barrier_expect -> async_tma_load -> wait_barrier`；`fence_async_shared -> wgmma.mma_async -> commit_group -> wait_group` | generic proxy 与 async proxy 分离 |
| `sm100` | `tcgen05` + TMEM + warp specialization | `tc_gen5_commit -> mbarrier wait` | `init_barrier -> tc_gen5_mma -> tc_gen5_commit -> wait_barrier -> phase rotate` | TMEM reuse hazard、tcgen05 专用跨线程同步 |

所以：

- `sm80` 主要是 `cp.async` group protocol。
- `sm90` 开始要同时理解 `mbarrier`、TMA、proxy fence、WGMMA。
- `sm100` 再往上叠加 `tcgen05/TMEM` 专用 protocol。

这三条链分别在 §6.1、§6.2、§6.3 展开。

### 0.3 读任何一个 wait / barrier / fence，先问哪三件事

先用 §9.1 的“三问”定位：它是在等线程到齐、等异步完成，还是在补跨 proxy / 跨 scope
的顺序与可见性。把这三问和 target 连起来看，比单独背 op 名更稳。

### 0.4 先给总判断：barrier 和 fence 不是一类东西

在 Triton 里，`barrier` 和 `fence` **至少要分成 4 组看**。这是理解全部原语最
重要的框架——按"它到底在等什么/守什么"来分，而不是按名字：

1. **execution barrier（执行会合）**
   让一组线程在控制流上会合（rendezvous），并顺带给某些存储域建立可见性。
   典型：`ttg.barrier`、cluster barrier、`__syncwarp/__syncthreads` 的对应物。

2. **completion tracking（异步完成跟踪）**
   回答"某个异步硬件动作什么时候*真的完成了*"。
   典型：`mbarrier`、`cp.async.wait_group`、`wgmma.wait_group`、
   `ttng.tc_gen5_commit + wait_barrier`。

3. **proxy fence / ordering fence（跨 proxy 的内存顺序）**
   解决"不是同一个 proxy / 不是同一个执行域"之间的内存顺序与可见性，**不负责等
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
   < cluster（thread block cluster / CGA，同一 GPC 上一组 CTA）< GPU < system。

2. **内存顺序（ordering）延伸到哪种存储 / proxy**：
   registers、shared memory（`.shared::cta` / `.shared::cluster`）、tensor
   memory（TMEM，Blackwell）、global memory。PTX 进一步把访问分成
   **generic proxy**（普通 ld/st）和 **async proxy**（TMA、`cp.async.bulk`、
   tcgen05）。这里的 `ordering` 指内存访问的先后可见性顺序，不是“给存储排序”。
   *同一个 proxy 内*建立内存顺序更便宜；*跨 proxy* 建立内存顺序需要
   `fence.proxy`。

3. **执行 vs. 内存**——有的原语阻塞执行（会合），有的只建立内存顺序（fence），
   多数两者都做。`bar.sync` 两者都做；`fence.proxy.async` 是纯内存顺序保证；
   `mbarrier` 把两者解耦（arrive = 发信号，wait = 阻塞）。

**Triton 把这 4 组的"决策权"分给了三个互相独立的机制**——这是最关键的结构性
事实：

| 关注点 | hazard 类型 | 机制 | 插入的原语 |
|---|---|---|---|
| CTA 内 shared memory RAW/WAR/WAW | 数据 hazard | `MembarAnalysis`（`lib/Analysis/Membar.cpp`） | `ttg.barrier` → `bar.sync` |
| TMEM 路径 load→mma / store→mma | 数据 hazard | `TMemBarrierInsertion`（`lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp`） | `ttg.barrier local` |
| generic-proxy ↔ async-proxy on smem | proxy hazard | `FenceInsertion` / `ProxyFenceInsertion` | `ttng.fence_async_shared` → `fence.proxy.async` |
| 异步引擎完成（cp.async, wgmma, TMA, tcgen05） | latency / liveness | 在 pipeline lowering 中显式给出，不是独立分析 | 各引擎自己的 commit/wait/mbarrier op |

这里真正的协作关系主要是：其它 lowering / pass 先把 barrier、commit、wait 这类
同步边界显式放进 IR，`MembarAnalysis` 再识别这些已有边界，从而不再重复插
`ttg.barrier`。它主要依赖两个识别手段：

- `containsLocalBarrier`：把已经建立同步点的 op 识别出来，如 `gpu::BarrierOp`、
  `ClusterBarrierOp`、`ClusterWaitOp`、`ArriveBarrierOp`、`BarrierExpectOp`、
  `TCGen5CommitOp` 等。
- `MemWaitOpTrait`：把 `async_wait`、`async_tma_store_wait` 这类“异步完成 wait”识别
  出来。`MembarAnalysis` 会尽量把 CTA barrier 推迟到这些 wait 之后合并，避免更早、
  更重复地插 barrier。

所以“没有 hazard 被重复同步”的意思不是这些机制彼此共享一套复杂状态，而是：
已有同步点先由别的机制建出来，`MembarAnalysis` 再把它们当成真实边界使用。详见 §7。

---

## 1. Scope 词汇表（粒度轴）

这一轴只回答一个问题：

```text
这条同步/可见性保证，到底对多大一组线程成立？
```

先把它看成一条从小到大的层级链：

```text
thread -> warp -> warp-group -> CTA -> cluster -> GPU -> system
```

下面先给一张总表，再补容易混淆的点。

| 层级 | 可以先粗略理解成 | 典型原语 / 场景 | 最容易混淆的点 |
|---|---|---|---|
| `thread` | 单个 lane / 单条线程 | predicate 为真时由一个线程代表执行 | 不是 barrier scope，只是最小执行单位 |
| `warp` | 32 个 lanes | `bar.warp.sync` / `__syncwarp()` | 只管一个 warp，不管整个 CTA |
| `warp-group` | 4 个连续 warp = 128 线程 | `wgmma.*.sync.aligned`、`tcgen05` | 这是 Hopper/Blackwell 执行单元，不是 CTA |
| `CTA` | 一个 thread block | `bar.sync 0` / `__syncthreads()` | 这是最常见的同步粒度 |
| `cluster` | 一组协同调度的 CTA | `barrier.cluster.*`、cluster-scoped `mbarrier` | 只在 SM90+ 出现，不是整个 GPU |
| `GPU` | 当前设备上的全部线程 | `.gpu` scope 的 fence / atomic | 没有“所有线程会合”的 barrier 含义 |
| `system` | GPU + host / peer devices | `.sys` scope 的 fence / atomic | 关注跨设备可见性，不是 kernel 内局部同步 |

### 1.1 先抓住三个最常用的层级

如果第一次读，只先抓这三个就够了：

- **warp**：32 个 lanes，一个最小的 SIMT 执行组。
- **CTA**：一个 thread block，Triton/CUDA 里最常见的共享内存协作范围。
- **cluster**：多个 CTA 组成的协作组，只在 SM90+ 才有。

很多同步问题，本质上就是在问：

- 这件事只需要一个 warp 内成立？
- 还是要整个 CTA 成立？
- 还是已经跨 CTA，需要 cluster 级别？

### 1.2 各层级到底是什么意思

- **`thread`**
  单个 lane 的程序顺序。这里几乎没有“只作用于 thread 的 barrier”，因为 barrier/fence
  的意义本来就在于关联不同线程。
  但 predication（如 `@$0`、`elect-one`）会让一个线程**代表一组线程**执行某个动作。
  例如 `mbarrier.init` 常常只让一个线程真正执行，但初始化后的效果对整个 CTA 或
  cluster 可见。

- **`warp`**
  32 个 lanes 的 SIMT 执行单元。`bar.warp.sync` / `__syncwarp()` 只让这 32 个 lanes
  重收敛，并建立 warp 内的内存顺序；它不会等待同一个 CTA 里的其它 warp。
  Triton 在单 warp 的 warp-specialize partition
  （`ConvertWarpSpecializeToLLVM.cpp:75`）以及 warp-synchronous convert-layout
  里会发它。

- **`warp-group`**
  4 个连续 warp，共 128 线程。这是 Hopper `wgmma` 和 Blackwell `tcgen05` 的执行单元。
  `wgmma.*.sync.aligned` 要求这 128 个线程都收敛。
  关键点：**warp-group 不是 CUDA Guide 标准的同步 scope 层级**，而是 PTX / 硬件执行
  单元概念；它也**不是** CTA barrier，CTA 内其它 warp-group 不受影响。

- **`CTA`**
  一个 thread block。`bar.sync 0` / `__syncthreads()` 会让 block 的所有线程会合，
  并让此前的 shared/global 访问在 CTA 内可见。
  这是主力原语。命名 barrier `bar.sync N, cnt`（0 ≤ N < 16）则允许 CTA 内一个
  子集单独会合，常见于 warp specialization。

- **`cluster`**
  一组协同调度的 CTA，也就是 thread block cluster（俗称 CGA，SM90+）。
  它们共享一个 distributed shared memory（DSMEM）窗口，所以 CTA A 可以直接访问
  CTA B 的 `.shared::cluster` 地址。
  `barrier.cluster.arrive` / `barrier.cluster.wait` 让整个 cluster 会合；
  `mbarrier.arrive...cluster` 则是 cluster-scoped 的 barrier object 协议。

- **`GPU` / `system`**
  这更多出现在 fence / atomic 的 `.gpu` / `.sys` scope 上，而不是“所有线程会合”的
  barrier 上。
  `.gpu` 表示当前设备范围内的可见性；`.sys` 则扩展到 host / peer devices。
  Triton 对 TMA tensormap acquire fence 用 `.gpu`（`TMAToLLVM.cpp:205`）。

### 1.3 三个最容易混淆的点

1. **`warp-group` 不是 `CTA`**
   `wgmma` 要求 128 线程收敛，不等于整个 block 收敛。

2. **`cluster` 不是 `GPU`**
   cluster 只是同一 GPC 上一小组 CTA，不是当前 GPU 上所有 CTA。

3. **predication 不是分叉控制流**
   `@$0`、`elect-one` 的意思通常是“同一条控制流里，只有一个线程或一部分线程真正执行
   这条 op”，不是整个程序真的分成两条路。

### 1.4 Triton 里怎么从 CTA 升到 cluster

粒度升级（escalation）主要由 barrier 对应 `MemDescType` 上的 **CGA broadcast
mask** 驱动：`getCGABroadcastMask() != 0` 会把 `.shared::cta` 翻成
`.shared::cluster`，把 `mbarrier.arrive` 翻成 `mbarrier.arrive...cluster`
（`BarrierOpToLLVM.cpp:214,349`）。

---

## 2. 存储类与"为什么每种都要各自的同步"

| 存储 | 谁写 | 谁读 | 建立内存顺序 / 可见性的原语 |
|---|---|---|---|
| **registers**（distributed） | 单个线程 | 同一线程；其它线程只能经 smem | 线程内无需；跨线程需 smem + barrier |
| **shared memory `.shared::cta`** | 任意线程（generic proxy）或 TMA/tcgen05（async proxy） | CTA 内任意线程 | `bar.sync`（generic↔generic）；`fence.proxy.async`（generic↔async）；`mbarrier`（异步完成） |
| **distributed shared `.shared::cluster`** | cluster 内任意 CTA | cluster 内任意 CTA | cluster mbarrier / `barrier.cluster.*` + `fence.proxy.async.shared::cluster` |
| **tensor memory（TMEM，SM100+）** | `tcgen05.mma`（异步）、`tcgen05.st` | `tcgen05.ld`、`tcgen05.mma` | `tc_gen5_commit` → mbarrier；accumulator RAW 靠 `AsyncToken` modref；load→mma / store→mma 靠 `TMemBarrierInsertion` 的 `ttg.barrier local` |
| **global memory** | 任意线程；TMA store | 任意线程；TMA load | 带 global 位的 `bar.sync`；`cp.async.bulk.wait_group`；带 scope 的 atomic |

有几个容易踩坑的事实：

1. **TMEM 不归 `MembarAnalysis` 管。**

   先记结论：

   - `MembarAnalysis` 只管 shared memory hazard。
   - TMEM 路径要看别的机制，不要指望 MemBar 自动补齐。

   TMEM 的顺序主要靠三样东西：

   - `AsyncToken` 依赖：穿过 `tc_gen5_mma` / `tmem_load` / `tmem_store`
   - `tc_gen5_commit -> wait_barrier`：负责异步完成
   - `TMemBarrierInsertion`：专门补 `load->mma` / `store->mma` 这类 TMEM hazard

   一个很实用的记忆法是：

   ```text
   shared memory hazard -> 先想 MembarAnalysis
   TMEM hazard          -> 先想 TMemBarrierInsertion / commit / wait
   ```

   这也解释了为什么 `arrive_barrier` lowering 会手动发一个前置 `ttg.barrier`：
   [BarrierOpToLLVM.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp:338)
   的注释直接说，这里技术上本该由 MemBar 处理，但它可能涉及 TMEM，而 MemBar 没有
   对应物。

2. **`generic proxy` 和 `async proxy` 之间要先分清方向。**

   先记结论：

   - `generic -> async`：通常要显式 `fence.proxy.async`
   - `async -> generic`：在 `cp.async.bulk` / TMA load 这条路上，通常不用再补显式
     `fence_async_shared`

   为什么？

   - `generic -> async`
     普通线程先写 smem（如 `local_store` / `stmatrix`），后面 async proxy 再读这块
     smem（如 TMA store、WGMMA、MMAv5、`TMEMCopy`）。这时候要靠
     `FenceInsertion` / `ProxyFenceInsertion` 补 `fence.proxy.async`。

   - `async -> generic`
     TMA load / `cp.async.bulk` 先把数据写进 smem，后面 generic `local_load` 再读。
     这条路通常由：
     `wait_barrier` 的 completion + `cp.async.bulk` 完成时随附的隐式 generic-async
     proxy fence
     一起保证，所以一般不用额外再插一个显式 `fence_async_shared`。

   一个实用记忆法是：

   ```text
   先问方向：
   谁先写？谁后读？

   generic 写、async 读 -> 想 fence.proxy.async
   async 写、generic 读 -> 先看 wait/completion 语义够不够
   ```

   典型 TMA load 路径
   `init/expect -> async_tma_load -> wait_barrier -> local_load`
   里没有额外 `fence_async_shared`，就是这个原因。`ProxyFenceInsertion.cpp` 里也能看
   到：TMA load 的写被单独归入 `proxyBlockInfo.syncWriteSlices`，而不是当成“普通
   generic 写后面还要补 fence”的那一路。

---

## 3. 会合类：先回答“哪些线程必须先到齐”

如果你看到的是 `ttg.barrier`、命名 barrier、`bar.warp.sync`、cluster barrier，这一组
都优先按“会合”来理解，而不是按“等待某个异步动作完成”来理解。

先用一张路由表把后面几节串起来：

| 你眼前的问题 | 先看哪一节 | 代表机制 |
|---|---|---|
| 哪些线程必须先到齐 | §3 | `ttg.barrier`、`bar.sync N`、`bar.warp.sync`、cluster barrier |
| 异步 copy / MMA 什么时候真的完成 | §4 | `mbarrier`、`wait_barrier` |
| 线程已经继续跑了，但内存顺序还没建立 | §5 | `fence.proxy.async`、tensormap acquire fence |
| 为什么这里明明没写 barrier，却被编译器补出来了 | §7 | `MembarAnalysis`、`TMemBarrierInsertion`、proxy fence insertion |

所以第 3 到第 7 节不是四份平行“百科清单”，而是同一条判断链：

```text
先判断是不是线程会合
不是的话，再判断是不是异步完成
再不是的话，看是不是在补内存顺序 / proxy 可见性
最后再追编译器为什么自动插入
```

这一节只收“会合类”。

先给一句最短对照：

| 机制 | 一句话理解 | 谁和谁到齐 |
|---|---|---|
| `ttg.barrier` | 整个 CTA 一起停一下，再一起继续 | 一个 block 的所有线程 |
| `bar.sync N, cnt` / `bar.warp.sync` | 只让 CTA 里的一个子集先对齐 | 一个 warp 或某个命名子集 |
| cluster barrier | 不同 CTA 之间也要先对齐 | 一个 cluster 里的所有 CTA |

### 3.1 `ttg.barrier`：默认的“整个 CTA 会合”

先抓一句话：

```text
ttg.barrier = “这个 block 里的所有线程先都到这里，再继续”
```

它是 Triton 里最默认、最常见的会合点，对应 `__syncthreads()` 这类 CTA 级同步。

- **它解决什么问题**
  一个 CTA 里的不同 warp 不是锁步执行的。没有 barrier，warp A 可能还没把 shared
  memory 写完，warp B 就已经开始读了。

- **什么时候会出现**
  最常见是在 CTA 内 shared memory 复用、producer/consumer 交接、或
  `MembarAnalysis` 发现 RAW/WAR/WAW hazard 的地方。

- **它的粒度是什么**
  **per CTA**。也就是一个 thread block 的所有线程都得参与。

- **它大致会降成什么**
  `TTG_BarrierOp`（`include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:734`）
  会先变成 `mlir::gpu::BarrierOp`
  （`lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp:255`），再到 NVVM/PTX 的
  `bar.sync 0`。

- **最容易误解什么**
  `ttg.barrier` 上有 `addrSpace` bitmask，看起来像“这个 barrier 会精确挑不同存储发
  不同 PTX”。但 **NVIDIA 主路径不是这样工作的**：
  `BarrierOpConversion` 直接 `replaceOpWithNewOp<mlir::gpu::BarrierOp>(op)`
  （`lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp:255`），并不读取
  `getAddrSpace`。也就是说，在 NVIDIA lowering 里它几乎都落成同一个 `bar.sync 0`。
  这些 bitmask 更像 Triton IR 层的语义标签，而不是 NVIDIA PTX 指令选择开关。
  对比 AMD 后端，才会真的读取 `hasLocal()` / `hasGlobalRead()`
  （`third_party/amd/lib/TritonAMDGPUToLLVM/MemoryOpToLLVM.cpp:720-721`）。

- **读 IR 时怎么认**
  看到 `ttg.barrier`，先不要想复杂 target protocol，先把它读成：
  “这里需要整个 CTA 在继续之前先完成一次 shared-memory 交接”。

### 3.2 `bar.sync N, cnt` / `bar.warp.sync`：不是整个 CTA，只是一个子集先对齐

先抓一句话：

```text
如果只有 CTA 里一部分线程需要互相等，就不能用整个 CTA 的 ttg.barrier。
```

这是 warp specialization 最容易看错的地方。很多人一看到 barrier，就默认“整个
block 都停住”。这里恰恰不是。

- **它解决什么问题**
  warp-specialized kernel 里，不同 warp-group / partition 可能执行不同角色：
  有的负责搬运，有的负责计算。此时如果还强行用整个 CTA 的 `bar.sync 0`，就可能把
  不同角色错误地绑死，甚至死锁。

- **什么时候会出现**
  典型就是 producer warp 和 consumer warp 只需要彼此对齐，不需要整个 block 一起停。

- **它的粒度是什么**
  不是整个 CTA，而是 CTA 的一个**子集**：
  - `bar.warp.sync`：只同步一个 warp
  - `bar.sync N, cnt`：同步注册到 barrier `N` 上的那一批线程

- **它大致从哪来**
  这不是一个单独的 Triton op 清单项，常见是在 C++ lowering 里直接发射。
  `ConvertWarpSpecializeToLLVM.cpp:75` 会发 `bar.warp.sync`，
  `ConvertWarpSpecializeToLLVM.cpp:79` 通过 `NVVM::BarrierOp(handle, numThreads)`
  发命名 barrier。

- **最该记住的区别**
  `ttg.barrier` 问的是：
  “整个 block 能不能一起继续？”

  命名 barrier / warp barrier 问的是：
  “CTA 里这一小撮线程，能不能先自己对齐继续？”

- **读 IR / lowering 时怎么认**
  只要你看到 warp specialization、partition、producer/consumer cohort 这种结构，
  就要优先怀疑这里需要的是“子集会合”，不是“全 CTA 会合”。

### 3.3 Cluster barrier：会合对象已经不是线程子集，而是多个 CTA

先抓一句话：

```text
cluster barrier = “不是一个 block 内部等，而是多个 CTA 之间等”
```

这时同步范围已经从 CTA 升到了 cluster。

- **它解决什么问题**
  在 SM90+，多个 CTA 可以组成一个 thread block cluster，共享
  `.shared::cluster` / DSMEM 视图。既然 CTA A 能直接被 CTA B 读到 shared window，
  那就需要跨 CTA 的会合点。

- **什么时候会出现**
  多 CTA 协作的 kernel，比如 cluster 级 producer/consumer、DSMEM 交接、cluster
  reduction。

- **它的粒度是什么**
  **per cluster**，不是 per CTA，也不是全 GPU。参与者是一个 cluster 里的多个 CTA。

- **最常见的表面形态**
  `ttng.cluster_arrive`（`TritonNvidiaGPUOps.td:81`）、
  `ttng.cluster_wait`（`:87`）、`ttng.cluster_barrier`（`:92`）。
  常规 lowering 是：
  `cluster_arrive -> barrier.cluster.arrive`
  `cluster_wait -> barrier.cluster.wait`
  （`ClusterOpsToLLVM.cpp:49,55`）。

- **为什么会拆成 `arrive` / `wait`**
  因为跨 CTA 会合更贵，常常希望先 `arrive` 发信号，再做一点别的事，最后才 `wait`
  真正阻塞。这样可以把一部分会合延迟藏起来。

- **最容易误解什么**
  `cluster_barrier` 并不总是那种“CTA barrier + peer mbarrier 轮询”的复杂形态。
  在 `ClusterOpsToLLVM.cpp:192-238` 里有两条 lowering 路径：
  - 默认路径是直接 `barrier.cluster.arrive/wait`
  - 只有带特殊 mbar offset 属性时，才展开成更复杂的 shared-mbarrier 方案

  所以读 dump 时，不要把那条复杂路径误当成 cluster barrier 的通用定义。

- **和 §5.3 的关系**
  `cluster barrier` 负责“大家什么时候到齐”。
  `fence.mbarrier_init.release.cluster` 负责“cluster 范围要用的 mbarrier 在别人 arrive
  之前已经初始化好”。
  一个是会合问题，一个是“别的 CTA 使用它之前，它已经初始化完成并且对它们可见”的问题。

---

## 4. 完成类：mbarrier 把“发信号”和“阻塞等待”拆开

mbarrier（`mbarrier.*`）是一个**驻留在 shared memory 中**的 64-bit 对象，跟踪两个
计数：一个 *arrival* 计数和（可选的）一个 *transaction/byte* 计数。它的目的是把
**发信号与阻塞解耦**：producer 直接 `arrive` 而不停下；consumer 在一个 *phase
parity*（相位奇偶）位上 `wait`，该位在预期的 arrival + bytes 到齐时翻转。这是异步
pipeline 的基础。它不是"编译器脑内的 token"，而是显式的共享内存对象。

这些 op 都带 `MBarrierOpInterface`（`TritonGPUOpInterfaces.td:45`），使分析能够
通用地找到 barrier memdesc 操作数。

先把这节压缩成一句话：

```text
mbarrier 只回答一件事：
“某个异步 producer 的结果，什么时候才算真的完成，consumer 什么时候才能继续？”
```

所以它首先是 **completion protocol**，不是 execution barrier。

### 4.1 先看总图：一条 mbarrier 协议总是这 5 步

不要先背 op 名，先记这条标准骨架：

```text
1. init
2. expect / arrive
 -> issue async work
3. completion 记到账到 mbarrier
4. wait_barrier(phase)
5. consume / reuse
```

按这 5 步看，系统性就出来了：

| 阶段 | 要解决的问题 | 常见 op |
|---|---|---|
| `1. init` | 这个 barrier 对象能不能开始用了 | `init_barrier`、`inval_barrier` |
| `2. expect / arrive` | 这次到底等什么 | `barrier_expect`、`arrive_barrier` |
| `3. issue + completion bookkeeping` | 异步 producer 完成后，怎么把完成记到账 | `async_tma_copy_*`、`async_copy_mbarrier_arrive`、`tc_gen5_commit`、`clc_try_cancel` |
| `4. wait` | consumer 什么时候才能继续 | `wait_barrier(phase)` |
| `5. consume / reuse` | 什么时候可以读结果、什么时候可以复用槽位 | `local_load`、下游 consumer、`inval_barrier` |

这张表里最关键的一点是：

- `expect / arrive` 是在**定义完成条件**
- `wait_barrier` 是在**观察完成条件是否已经满足**

它们不是一回事。

### 4.2 再看 op：把所有 op 塞回这 5 个槽位

| 槽位 | Op | 作用 |
|---|---|---|
| `init` | `ttng.init_barrier`（`:257`） | 把 shared-memory 里的对象初始化成 mbarrier |
| `init` | `ttng.inval_barrier`（`:277`） | 一轮协议结束后退役该 barrier，准备复用槽位 |
| `expect / arrive` | `ttng.barrier_expect`（`:293`） | 预先声明这次要等多少 transaction bytes |
| `expect / arrive` | `ttng.arrive_barrier`（`:365`） | 记录 arrival；也可带 `count` |
| `issue + bookkeeping` | `ttng.async_tma_copy_global_to_local`（`:446`）、`ttng.async_tma_gather`（`:564`） | TMA load 完成后把 bytes 记到 barrier |
| `issue + bookkeeping` | `ttng.async_shared_store`（`:415`） | distributed-smem store 完成后递减 transaction 计数 |
| `issue + bookkeeping` | `ttng.async_copy_mbarrier_arrive`（`:404`） | 把非 bulk cp.async 的完成绑到 barrier |
| `issue + bookkeeping` | `ttng.tc_gen5_commit` + `ttng.tc_gen5_mma[_scaled]`（`:632`/`:696`） | 把 tcgen05 completion 变成 mbarrier 上可观察的完成点 |
| `issue + bookkeeping` | `ttng.clc_try_cancel`（`:110`） | 异步写 result buffer，完成时 signal mbarrier |
| `wait` | `ttng.wait_barrier`（`:317`） | 在指定 `phase` 上阻塞等待完成 |

这样看，mbarrier 不是“又一堆零散 op”，而是一套很稳定的模板：

```text
对象初始化
-> 完成条件声明
-> 异步 producer 发起并把完成记账
-> consumer wait
-> 结果被消费 / 槽位被复用
```

### 4.3 标准模板：先记一条最典型路径

最值得先背的是 TMA load 这条，因为它把 mbarrier 的全部角色都用全了。

```text
init_barrier
-> barrier_expect(bytes)
-> async_tma_copy_global_to_local
-> wait_barrier(phase)
-> local_load / 下游使用
-> inval_barrier
```

这条链里每一步分别在干什么：

| 步骤 | 它在干什么 |
|---|---|
| `init_barrier` | 先把 barrier 对象建起来 |
| `barrier_expect(bytes)` | 明确这次要等多少字节写完 |
| `async_tma_copy...` | 发起真正的异步搬运 |
| `wait_barrier(phase)` | consumer 在这里等“搬运真的结束” |
| `local_load / 下游使用` | 只有 wait 过后，consumer 才能把数据当成可用 |
| `inval_barrier` | 这一轮跑完，槽位进入可复用状态 |

所以你读任何一段 mbarrier IR，都可以先套这个模板：

```text
它现在是在：
建对象？
设完成条件？
发异步动作？
等完成？
还是回收槽位？
```

只要先定位到这 5 个槽位之一，内容就不会散。

### 4.4 两个变体：CLC 和 tcgen05 只是换了 producer，不是换了协议

上面的 TMA load 是“标准版”。CLC 和 tcgen05 只是把第 3 步的 producer 换掉了。

先把这两个变体的差异压成一张表：

| 变体 | 第 3 步的 async producer 是谁 | 它完成后写到哪里 | `wait_barrier` 等的是什么 |
|---|---|---|---|
| CLC | `clc_try_cancel` | smem result buffer + mbarrier 记账 | “取消结果已经写好了” |
| tcgen05 | `tc_gen5_mma` + `tc_gen5_commit` | TMEM 结果 + mbarrier 完成信号 | “此前异步 tcgen05 指令已经完成” |

这样看就容易很多：

- **CLC** 的结果本体就是一小块 smem 数据，`wait_barrier` 之后直接把它读回来。
- **tcgen05** 的结果本体在 TMEM 里，`wait_barrier` 等的是“异步 MMA 已经做完”，然后
  后续路径再去消费 TMEM 结果。

#### A. CLC：把“取消结果是否写好”接到 mbarrier 上

先抓一句话：

```text
CLC 变体里，mbarrier 等的不是 tile 数据，而是 “try_cancel 的结果是否已经写到 smem”
```

把它套回 5 步模板：

| 步骤 | CLC 里对应什么 |
|---|---|
| `init` | `init_barrier(count=1)` |
| `expect / arrive` | `barrier_expect(16)`，因为结果是 `2xi64 = 16 bytes` |
| `issue + bookkeeping` | `clc_try_cancel` 发起异步 cancel，请硬件把结果写到 smem，并在完成时 signal mbarrier |
| `wait` | `wait_barrier` |
| `consume / reuse` | `clc_load_result`（`:136`）把结果读回寄存器 |

所以 CLC 的关键不是“又多了一种 barrier”，而是：

- 异步 producer 不再是 TMA engine
- 它写的也不是 tile，而是一份 16-byte 的结果
- 但 consumer 仍然通过同一个 `wait_barrier` 来判断“现在能不能读”

测试 `test/TritonNvidiaGPU/membar-cluster.mlir:494` 之所以重要，就是因为它把这套模式写得
很直白：

```text
init_barrier(count=1)
-> clc_try_cancel
-> barrier_expect(16)
```

这说明 CLC 和 TMA load 在 mbarrier 看来是同一种 completion protocol：

```text
先声明要等的字节数
异步 producer 完成后把字节记账
consumer 再 wait
```

#### B. tcgen05：把“异步 MMA 是否做完”接到 mbarrier 上

先抓一句话：

```text
tcgen05 变体里，mbarrier 等的不是 “smem 有没有写好”，而是 “异步 MMA 有没有做完”
```

这里最容易误解的点是：tcgen05 的结果本体在 **TMEM**，不是像 TMA load/CLC 那样直接
落在一块 smem 结果缓冲上。所以要分两层看：

1. `tc_gen5_mma[_scaled]`（`:632`/`:696`）负责真正的异步 MMA 计算
2. `tc_gen5_commit` 负责把“这些异步 MMA 完成了”这件事接到 mbarrier 上

这样后面的 consumer 就仍然能沿用统一模板：

| 步骤 | tcgen05 里对应什么 |
|---|---|
| `init` | barrier 先被建好 |
| `expect / arrive` | 由 tcgen05 配套协议准备完成条件 |
| `issue + bookkeeping` | `tc_gen5_mma` 发异步 MMA，`tc_gen5_commit` 把完成记到 mbarrier |
| `wait` | `wait_barrier` 等“此前 tcgen05 已完成” |
| `consume / reuse` | 后续路径再去消费 TMEM 结果 |

所以 tcgen05 和 TMA load 的真正区别不在于“有没有 wait”：

- TMA load 是：**异步搬运完成后，smem 数据可读**
- tcgen05 是：**异步 MMA 完成后，TMEM 结果可被后续阶段消费**

`wait_barrier` 在两者里的角色是一样的：

```text
都不是自己做计算或搬运
只是负责把“异步动作已经完成”这个事实变成 consumer 可观察到的同步点
```

所以这一节最该建立的系统感其实是：

```text
mbarrier 协议基本不变
变的是第 3 步：到底是哪种 async producer 在把完成记到账
```

### 4.5 最后单独记：mbarrier 不解决什么

前面是它“负责什么”，这里单独收口它“**不负责什么**”。这一步很重要，不然最容易误读。

1. **`wait_barrier` 不是 CTA rendezvous**
   它回答“完成了没”，不是“大家到齐了没”。

2. **mbarrier 跟踪 completion，不自动替代所有 proxy ordering**
   对 TMA load 这类 `cp.async.bulk`，async→generic 这条路能成立，是因为
   `wait_barrier` + 指令完成时随附的隐式 proxy fence 一起覆盖。
   但 generic→async 那一边是否需要显式 `fence.proxy.async`，仍然要回到 §2、§5 看。

3. **不是所有 TMA 都走 mbarrier**
   TMA **store**（`async_tma_copy_local_to_global` `:511`、`reduce`、`scatter`）
   **没有** mbarrier；它们走的是 bulk-async group +
   `async_tma_store_wait`（`:620`，`cp.async.bulk.wait_group`）这一套。
   其中 `read_only` 还要分清：它只保证 TMA engine 已经把 shared buffer 读走，
   允许下一轮重写 buffer；不等价于 global store 已经完整落地。

如果要把这节压缩成最后一句话，就是：

```text
mbarrier = 一个可复用的“异步完成记账对象”
它的固定模板是：
init -> expect/arrive -> async producer 记账 -> wait -> consume/reuse
```

---

## 5. 顺序类：fence 只补内存顺序，不负责会合

**fence** 只为*发起线程*（或某个 scope）建立内存访问的先后顺序，而**不**阻塞等待
其它线程。它回答“让我此前的写对后续使用者可见”，而不是“等所有人”。

先把这节压成一句话：

```text
fence 回答的不是“什么时候完成”，而是“前面的写，什么时候才能按某个 scope / proxy
对后面的使用者可见”
```

所以它首先是 **ordering / visibility protocol**，不是 completion protocol，也不是
execution rendezvous。

### 5.1 先看总图：读 fence 先问这三件事

比起直接背指令名，更稳的是先问这三个问题：

| 问题 | 你在定位什么 | 典型答案 |
|---|---|---|
| 谁先写，谁后用 | 这是哪一条 happens-before | generic 写 smem，async proxy 后读；host/线程改 descriptor，TMA 后读 |
| 中间隔着什么边界 | 是同一 proxy、跨 proxy，还是跨 CTA / cluster | generic ↔ async proxy；普通线程 ↔ tensormap consumer；本 CTA ↔ peer CTA |
| 需要把可见性推到多大 scope | 这条顺序要对谁成立 | `.cta`、`.cluster`、`.gpu` |

把这三问压成最短模板，就是：

```text
谁先写
-> fence
-> 谁后用
```

只不过 fence 的价值在于，它把“谁后用”之前必须成立的可见性和顺序补出来。

### 5.2 再看总分类：这节其实只有三类 fence

| 类别 | 作用对象 | 作用内存类型 | 它解决什么顺序问题 | 代表 op |
|---|---|---|---|---|
| shared async-proxy fence | shared tile / shared buffer | shared memory（`.shared::cta` / `.shared::cluster`） | generic proxy 写完后，async proxy 才能合法读 | `ttng.fence_async_shared` |
| tensormap acquire fence | descriptor / tensormap 元数据 | descriptor 所在的元数据存储（通常是 global / device-visible metadata），不是 TMEM | 对 descriptor 的写必须先于 TMA 单元读它 | `ttng.tensormap_fenceproxy_acquire` |
| mbarrier-init release fence | mbarrier 对象本身 | shared memory 中的 mbarrier 对象 | mbarrier 初始化必须先于 peer CTA 使用它 | `ttng.fence_mbarrier_init_release_cluster` |

所以这节不是三条互不相干的知识点，而是三种不同的“对象 + 使用者”组合：

- 数据对象是 **shared tile**
- 元数据对象是 **tensormap / descriptor**
- 同步对象是 **mbarrier 本身**

### 5.3 最值得先抓的主线：shared memory 的 generic ↔ async proxy 边界

这是最常见、也最容易误解的一类。

先抓一句话：

```text
`bar.sync` 只能管 generic proxy 内的顺序；
跨到 async proxy，要靠 `fence.proxy.async`
```

#### A. 这条 fence 在保护什么

它保护的不是“shared memory”这个抽象名词，而是：

```text
同一块 shared memory
先被 generic proxy 写
后被 async proxy 读
```

这正是 `ttng.fence_async_shared` / `fence.proxy.async.shared::{cta,cluster}` 的语义核心。

#### B. 什么时候最常出现

最常见的模式是：

```text
generic shared write
-> fence_async_shared
-> async consumer
```

例如：

- generic `convert_layout` / `sts` / `stmatrix` 先把数据写到 smem
- 后面 `WarpGroupDot`、`MMAv5`、`TMEMCopy`、TMA store 这类 async side 再来读

对应源码：

- op / lowering：`TritonNvidiaGPUOps.td:52` → `BarrierOpToLLVM.cpp:68-73`
- async-proxy 写方 / 读方分类：`ProxyFenceInsertion.cpp:32,47`

#### C. 为什么 `bar.sync` 不够

这是这节最关键的边界：

- `bar.sync` 会让 CTA 内线程会合，并在 **generic proxy** 里建立顺序
- 但它对 **async proxy** 的内存视图不做保证

所以：

```text
__syncthreads 通过了
!=
TMA / WGMMA / tcgen05 一定看到最新的 smem
```

要把这条边界补起来，才需要 `fence.proxy.async`。

#### D. 方向为什么重要

显式 `fence.proxy.async` 主要是：

```text
generic -> async
```

也就是：

- 普通线程先写 smem
- 异步引擎后读 smem

反方向：

```text
async -> generic
```

例如 TMA load 写完 smem、后面 generic `local_load` 再读，这条路通常不靠额外显式
`fence_async_shared`，而是靠 completion + 指令完成时随附的隐式 proxy fence。这个方向
差异要和 §4 的 completion 语义一起看。

#### E. Triton 里谁负责插它

这类 fence 不是手工到处摆，而是主要靠两个 pass：

- `FenceInsertion`
  `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`
  它只在 **compute capability ≥ 9.0** 生效，由 `DotOpInterface` 驱动，典型模式是
  `generic(convertlayout) -> fence -> async(wgmma)`。
- `ProxyFenceInsertion`
  `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp`
  它是更一般、alias-aware 的补救版；`insertFence` 在 `:109`，fence 出现后会通过
  `blockInfo->sync()`（`:115-117`）清掉 pending 依赖。

### 5.4 第二类：不是数据 tile，而是 descriptor / tensormap 元数据

这一类最容易被误读成“是不是又一种 barrier”。其实不是。它解决的是：

```text
谁先把 descriptor 改好
谁后面才能拿这个 descriptor 去驱动 TMA
```

#### A. 对象变了

这里受保护的不是 shared tile，也不是 barrier 对象，而是：

- tensormap
- descriptor

也就是“异步搬运要读的元数据”。

#### B. 为什么需要单独的 acquire fence

如果 kernel 运行时会 `tensormap_create` / `tensormap_replace`，那就会出现一条新的顺序链：

```text
先写 descriptor
-> acquire fence
-> TMA 单元读 descriptor
```

`ttng.tensormap_fenceproxy_acquire`
（`TritonNvidiaGPUOps.td:1114`，lowering 见
`test/Conversion/tritonnvidiagpu_to_llvm.mlir:545` 以及 `TMAToLLVM.cpp:205`）
就是在补这条链。

#### C. 为什么 scope 是 `.gpu`

因为 descriptor 的消费者不是某个 CTA 内的普通线程，而是 device-scoped 的异步 copy
引擎。所以这里的关键不是 CTA 内局部协作，而是：

```text
这个元数据对象对设备侧消费者什么时候可见
```

因此它是：

- acquire fence
- `.gpu` scope
- 针对 descriptor / tensormap 元数据

### 5.5 第三类：不是数据可见性，而是 mbarrier 对象自身先要可用

`ttng.fence_mbarrier_init_release_cluster`
（`TritonNvidiaGPUOps.td:66` → `BarrierOpToLLVM.cpp:44`）解决的是第三种对象：

```text
mbarrier 自己
```

这里不要再把它读成数据流问题，而要读成“同步对象的初始化顺序”问题。

#### A. 它在补哪条顺序

```text
CTA A: mbarrier.init
-> fence.mbarrier_init.release.cluster
-> CTA B: mbarrier.arrive...cluster
```

它要保证的是：

- peer CTA 真正开始 `arrive` 之前
- 这个 mbarrier 对象已经初始化完成
- 并且这件事对 peer CTA 可见

#### B. 为什么它属于 fence，不属于 barrier

因为这里没有在问：

- 大家到齐了没？
- 异步动作完成了没？

这里问的是：

- **这个同步对象能不能合法被后续使用了？**

所以它属于 initialization visibility / `release` ordering，而不是 completion 或
rendezvous。

### 5.6 最后单独记：fence 不负责什么

前面是三类 fence 各自负责什么，这里收口它们共同**不负责什么**：

1. **fence 不会合**
   它不负责“让一组线程一起停下来”。会合问题还是 `ttg.barrier`、cluster barrier、
   `bar.warp.sync` 那一侧。

2. **fence 不直接等完成**
   它不回答“异步动作是不是已经做完了”。completion 还是 `wait_barrier`、
   `cp.async.wait_group`、`wgmma.wait_group`、`tc_gen5_commit -> wait_barrier` 那一侧。

3. **fence 只补一条特定的顺序链**
   它不是“全能同步”。你始终要问清楚：
   - 哪个对象
   - 谁先写
   - 谁后用
   - 中间跨没跨 proxy / scope

### 5.7 说明：atomic 的 `sem`/`scope` 不在本文主范围

`tt.atomic_rmw` / `tt.atomic_cas` 上的 `.acquire` / `.release` / `.acq_rel` 与
`.cta` / `.gpu` / `.sys` 是**内存同步语义**（memory consistency 的 acquire/release
顺序 + 作用 scope），它们确实影响可见性，但**不是 barrier / fence 原语**：没有会合，
也不是独立发射的 fence op，而是附着在原子指令本身上的修饰符。本文聚焦 barrier 与
显式 fence，atomic 的 acquire/release 内存顺序语义只在需要时对照 PTX ISA §8 内存
模型即可。

---

## 6. 按 target 看：同一条主链在三代 GPU 上怎么换协议

如果前面 §3、§4、§5 是按职责拆开，这一节就只做一件事：

```text
把同一个“搬运 -> 计算 -> 等完成 -> 进入下一轮”的主链
放到 sm80 / sm90 / sm100 上对照
```

所以这里不要把它读成“三个孤立的 target 知识点”，而要读成：

- 哪个 target 用什么搬运引擎
- 哪个 target 用什么计算引擎
- 完成由谁来跟踪
- 顺序 / 会合缺口由谁来补

### 6.1 先看统一模板：每个 target 都要回答这 4 个槽位

对主循环来说，不管 target 怎么变，读法都可以先压成同一模板：

| 槽位 | 你在问什么 |
|---|---|
| `搬运` | 下一块数据靠谁搬进来 |
| `计算` | 这块数据靠谁消费 / 计算 |
| `完成` | 什么时候才算真的搬完 / 算完 |
| `补丁` | 还需要什么会合 / fence / barrier 才合法 |

把它写成最短主链，就是：

```text
搬运
-> 计算
-> 等完成
-> 补顺序 / 会合
-> 进入下一轮
```

真正的 target 差异，主要就体现在这 4 个槽位换了谁。

### 6.2 再看总表：三代 target 分别把 4 个槽位交给谁

| target | 搬运槽位 | 计算槽位 | 完成槽位 | 典型补丁 |
|---|---|---|---|---|
| `sm80/sm86` | `cp.async` | `mma.sync` / 普通 dot | `cp.async.commit_group -> wait_group` | `ttg.barrier local` 把 per-thread completion 升成 CTA 可消费 |
| `sm90` | TMA / `cp.async.bulk` | `wgmma` | TMA 用 `mbarrier.wait`；WGMMA 用 `commit_group -> wait_group` | `fence_async_shared`、cluster barrier、mbarrier protocol |
| `sm100` | TMA + 更强的 warp specialization | `tcgen05` / TMEM | `tc_gen5_commit -> wait_barrier` | TMEM hazard barrier、cluster / cta_group 同步、proxy fence |

先抓住这张表里的两个演化趋势：

1. **完成模型在升级**
   - `sm80` 主要是 per-thread async group
   - `sm90` 开始大量进入 `mbarrier` + group-wait 双轨
   - `sm100` 把 tcgen05 completion 也接到 `mbarrier`

2. **补丁在变复杂**
   - `sm80` 主要补 CTA barrier
   - `sm90` 要补 proxy fence、cluster、mbarrier 生命周期
   - `sm100` 还要补 TMEM hazard 和更复杂的跨 warp-group / cta-group 交接

### 6.3 `sm80/sm86`：最经典的是 “cp.async 完成，但还没自动变成 CTA 可消费”

先抓一句话：

```text
sm80 的核心矛盾是：
cp.async 的完成是 per-thread 的，
但 shared tile 往往要被整个 CTA 后续消费
```

所以它的主链最好读成：

```text
cp.async
-> cp.async.commit_group
-> cp.async.wait_group
-> ttg.barrier local
-> CTA 内其它 warp / 线程开始消费
```

把它套回 4 槽位：

| 槽位 | `sm80/sm86` 里是谁 |
|---|---|
| `搬运` | `ttg.async_copy_global_to_local` / `cp.async` |
| `计算` | 普通 dot / `mma.sync` 一侧 |
| `完成` | `ttg.async_commit_group`（`TritonGPUOps.td:71` → `cp.async.commit_group`，`LoadStoreOpToLLVM.cpp:1835`）+ `ttg.async_wait`（`:47`，`LoadStoreOpToLLVM.cpp:1817`） |
| `补丁` | `ttg.barrier local`，常由 `MemWaitOpTrait` + Membar 在 wait 后补出来 |

这里最关键的边界是：

- `cp.async.wait_group` 只保证**本线程**发出的 async group 完成
- 它**不**自动让整个 CTA 其它线程都能安全消费这块 smem

这正是 op 描述里那句
"does not provide any synchronization in the CTA"
的含义，也是为什么后面还经常要接 `ttg.barrier local`。

所以 `sm80` 的主线可以压成一句：

```text
per-thread completion
-> 再补一个 CTA barrier
-> 才变成 block 级可消费的数据
```

### 6.4 `sm90`：开始分成两条主线，搬运和计算各有自己的完成协议

到了 Hopper，最该先建立的系统感是：

```text
sm90 不是一条 completion 链
而是两条并行主线：
TMA 搬运线
+ WGMMA 计算线
```

#### A. TMA 搬运线：用 mbarrier 观察“tile 是否真的进了 smem”

主链是：

```text
barrier_expect(bytes)
-> async_tma_load
-> wait_barrier(phase)
-> local_load / 下游消费
```

把它套回 4 槽位：

| 槽位 | `sm90` TMA 线里是谁 |
|---|---|
| `搬运` | `async_tma_copy_global_to_local` / `cp.async.bulk.tensor` |
| `计算` | 后面的 generic `local_load` 或 WGMMA operand 消费 |
| `完成` | `mbarrier` + `wait_barrier` |
| `补丁` | 若前面还有 generic->async 边界，则要结合 `fence_async_shared` 看 |

这条线的核心不是 group wait，而是：

- 先 `barrier_expect(bytes)`
- 让 TMA 把完成记到 mbarrier
- consumer 再 `wait_barrier`

#### B. WGMMA 计算线：用 group 模型观察“异步 MMA 是否做完”

主链是：

```text
generic shared write
-> fence_async_shared
-> wgmma.fence
-> wgmma.mma_async
-> wgmma.commit_group
-> wgmma.wait_group
```

把它套回 4 槽位：

| 槽位 | `sm90` WGMMA 线里是谁 |
|---|---|
| `搬运` | 上游 shared operand 准备 |
| `计算` | `ttng.warp_group_dot` → `wgmma.mma_async` |
| `完成` | `ttng.warp_group_dot_wait` → `wgmma.wait_group` |
| `补丁` | `fence_async_shared`、`wgmma.fence`、必要时还有 cluster 协作 |

这一条线最容易混的地方是：它的“完成”不是 `mbarrier`，而是 **group wait**。

所以 `sm90` 真正的 target 视角不是“统一用一种 wait”，而是：

- **搬运完成**看 `mbarrier.wait`
- **计算完成**看 `wgmma.wait_group`

### 6.5 `sm100`：主链继续统一到 mbarrier，但 TMEM 成为新约束中心

先抓一句话：

```text
sm100 的关键变化不是“又有一个 wait”
而是：tcgen05 / TMEM 把计算完成也接到了 mbarrier 上，同时引入了 TMEM hazard
```

这条主链最好读成：

```text
init_barrier
-> tc_gen5_mma
-> tc_gen5_commit
-> wait_barrier
-> 后续消费 TMEM 结果 / 进入下一轮
```

把它套回 4 槽位：

| 槽位 | `sm100` 里是谁 |
|---|---|
| `搬运` | TMA / multicast TMA / 上游 shared-TMEM 准备 |
| `计算` | `ttng.tc_gen5_mma[_scaled]`（`:632`/`:696`） |
| `完成` | `ttng.tc_gen5_commit`（`:774`）把完成记到 mbarrier，再由 `wait_barrier` 观察 |
| `补丁` | `TMemBarrierInsertion`、`NVVM::BarrierOp` 封边、必要的 cluster / cta_group 约束 |

这条线和 `sm90` WGMMA 最大的差别是：

- WGMMA 主要靠 group 模型回收计算完成
- tcgen05 选择把计算完成**折回 mbarrier**

原因在于：

- tcgen05 结果写到 **TMEM**
- 后续 `tcgen05.ld` / consumer 需要一个跨阶段都能共享的 completion 观察点
- mbarrier 正好提供了这种“公共货币”

所以 `sm100` 的 target 视角可以压成：

```text
计算完成也被纳入 mbarrier 世界
+ 但 TMEM hazard 需要额外的 barrier repair
```

### 6.6 最后把三代压成一张“演化表”

如果只想记住三代差异，记这张表就够了：

| 维度 | `sm80/sm86` | `sm90` | `sm100` |
|---|---|---|---|
| 搬运引擎 | `cp.async` | TMA / `cp.async.bulk` | TMA + 更复杂 producer 协作 |
| 计算引擎 | `mma.sync` / 普通 dot | `wgmma` | `tcgen05` / TMEM |
| 主要完成模型 | per-thread async group | TMA 用 `mbarrier`，WGMMA 用 group wait | tcgen05 completion 也经 `mbarrier` |
| 最典型补丁 | CTA barrier | proxy fence + cluster + mbarrier 生命周期 | TMEM hazard barrier + cta_group / cluster 协作 |
| 最容易误读的点 | `wait_group` 不等于 CTA 同步 | `mbarrier.wait` 和 `wgmma.wait_group` 不是一回事 | `wait_barrier` 等的是 tcgen05 完成，不是 TMEM hazard 全自动消失 |

所以这节最终要建立的系统感是：

```text
同一条主循环
在不同 target 上
真正换掉的是：
搬运引擎
+ 计算引擎
+ 完成模型
+ 合法性补丁
```

---

## 7. 编译器如何自动补齐：为什么源码没写，TTGIR 里却多了同步点

前面 §3、§4、§5 讲的是“这些同步原语各自是什么意思”。这一节只回答一个更实用的问题：

```text
为什么我明明没手写 barrier / fence，
dump 里却突然多出来了？
```

先给一句总判断：

```text
编译器自动补齐同步，不是因为它“喜欢多插 barrier”，
而是因为它在修三类不同的 hazard：
shared 数据 hazard
+ TMEM 数据 hazard
+ proxy 顺序 hazard
```

所以这一节最好不要按“pass 名列表”来读，而要先按“它在修哪一类危险”来读。

### 7.1 先看总图：自动补齐只分三类责任

| 看到的现象 | 先怀疑谁 | 它在修什么 | 它通常插什么 |
|---|---|---|---|
| CTA 内 shared memory 复用前后多了 `ttg.barrier local` | `MembarAnalysis` | shared memory RAW/WAR/WAW | `ttg.barrier local` |
| TMEM 路径里多了 `ttg.barrier local` | `TMemBarrierInsertion` | `load->mma` / `store->mma` 等 TMEM hazard | `ttg.barrier local` |
| `wgmma`、TMA store、TMEMCopy 前面多了 `ttng.fence_async_shared` | `FenceInsertion` / `ProxyFenceInsertion` | generic proxy ↔ async proxy 顺序缺口 | `ttng.fence_async_shared` |

这张表里最重要的不是“都叫同步”，而是：

- 前两类修的是 **数据 hazard**
- 第三类修的是 **proxy ordering hazard**

也正因为这两种 hazard 不同，所以它们可能在同一个程序点同时都需要。

### 7.2 第一类：`MembarAnalysis` 只管 shared memory 数据 hazard

先抓一句话：

```text
MembarAnalysis 的职责很窄：
只看 shared memory slice 是否冲突，
若冲突，就补一个 CTA barrier
```

它的源码入口是：

- `lib/Analysis/Membar.cpp`
- `include/triton/Analysis/Membar.h`
- 模块级接入点：`TritonGPUToLLVM.cpp:111`

#### A. 它到底在看什么

它看的是两个 op 的 **shared memory allocation slice** 是否相交。

也就是说，它不是笼统地说“你们都碰了 shared memory 就同步”，而是更具体地问：

```text
你们碰到的是不是同一块物理 shared-memory 区间？
```

只有相交时，才会构成真正的 hazard。

对应的 hazard 类型是：

- **RAW**：先写后读，读方可能看到旧数据
- **WAR**：先读后写，写方可能把读方还没消费的数据覆盖
- **WAW**：两个写之间顺序不确定
- **RAR**：不是 hazard

相交判定入口在 `Membar.h:131` 的 `isIntersected`。

#### B. 它插什么

它永远插的是：

```text
ttg.barrier local
```

源码在 `Membar.cpp:243`。也就是说，它只会补 CTA 级 `bar.sync` 语义，不会去插：

- mbarrier
- cluster barrier
- proxy fence

这些属于别的机制的职责。

#### C. 它为什么经常“等到 wait 后面再插”

这是这个 pass 最容易看起来“很聪明但难懂”的地方。其实可以压成一句：

```text
如果前面已经有异步 wait 在帮你收口，
MembarAnalysis 会尽量把 CTA barrier 推到 wait 后面合并
```

对应逻辑在 `MembarAnalysis::update`（`Membar.cpp:281`）：

1. 如果当前 op 已经是同步点，就清掉 pending slice
   - `containsLocalBarrier`（`:247`）会识别一批已有同步边界，如
     `gpu::BarrierOp`、`ClusterBarrierOp`、`ClusterWaitOp`、
     `WarpSpecializePartitionsOp`、`ArriveBarrierOp`、`BarrierExpectOp`、
     `TCGen5CommitOp`
2. 如果当前 op 带 `MemWaitOpTrait`，就尽量把 barrier 放到 wait 后
   - 见 `:292`
3. 如果既没有现成同步点、又真的发生 slice hazard，就在当前 op 前补 barrier

所以它不是“看到 shared 就立刻插”，而是：

```text
先看有没有已有同步边界能复用
能复用就别重复插
复用不了再自己补
```

#### D. 一个容易漏掉的边界

warp-synchronous convert-layout 内部如果只做了 warp 级同步，不等于 CTA 级依赖已经清掉。

这也是为什么 `warp` 维上的 `isCvtDimSync` 不会自动清掉 CTA 级 pending，见
`Membar.cpp:378`。

### 7.3 第二类：`TMemBarrierInsertion` 只管 TMEM 数据 hazard

先抓一句话：

```text
shared memory hazard 交给 Membar；
TMEM hazard 不在它的视野里，所以要单独再来一个 pass
```

源码入口：

- `lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp`

#### A. 它到底在修什么

它看的是 **TMEM slice** 上的数据 hazard，而不是 shared memory。

判定条件在 `:75`：

```text
requiresBarrier = war || raw || waw || loadToMma || storeToMma
```

这里最该记住的不是条件枚举，而是这条非对称规则：

```text
load -> mma   要 barrier
store -> mma  要 barrier
mma -> load/store  通常不要额外 barrier
```

源码注释在 `:66-69`。

#### B. 为什么会有这种非对称性

因为后一个方向已经被 completion 协议兜住了。

也就是：

- 如果是 `mma -> load/store`
- 后面本来就会有 `mbarrier wait`
- 那么“异步 MMA 什么时候真的做完”已经由 completion 覆盖

所以这个 pass 主要修的是：

```text
别让 CTA 内线程太早把下一类访问送进 MMA 之前
就越过了 TMEM 使用边界
```

#### C. 它为什么插的还是 `ttg.barrier local`

这是最容易让人困惑的一点：

```text
明明冲突对象是 TMEM，
为什么插出来的是 CTA barrier？
```

因为 Triton 在这一层表达的是：

- CTA 内线程组执行顺序
- 以及一条“别越过这个协议边界”的同步点

真正的 TMEM 异步 completion 仍然由后面的 mbarrier wait 负责。也就是说：

- 这个 barrier 不是“TMEM 专用硬件 fence”
- 它更像编译器在 TTGIR 层插入的 **hazard separator**

### 7.4 第三类：proxy fence 插入 pass 只管顺序缺口，不管数据冲突

先抓一句话：

```text
前两类 pass 修的是“同一块数据会不会被过早读/写”；
这一类 pass 修的是“两个 proxy 之间的顺序有没有建立起来”
```

所以它和 Membar / TMemBarrierInsertion 是正交的。

#### A. 什么时候会需要它

典型模式是：

```text
generic proxy 写 smem
-> async proxy 读 smem
```

这时即便没有 shared slice 冲突意义上的 RAW/WAR/WAW，顺序也可能仍然不合法，因为：

- generic proxy 的可见性世界
- async proxy 的可见性世界

不是同一个。

#### B. Triton 里是哪两个 pass 在补

- `FenceInsertion`
  - `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp`
  - 更偏模式驱动，典型是 `generic(convertlayout) -> fence -> async(wgmma)`
  - 只在 **compute capability ≥ 9.0** 生效，由 `DotOpInterface` 驱动
- `ProxyFenceInsertion`
  - `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp`
  - 更一般、alias-aware
  - `insertFence` 在 `:109`
  - fence 出现后通过 `blockInfo->sync()`（`:115-117`）清掉 pending 依赖

#### C. 它和 Membar 的根本区别

可以压成一句很短的话：

```text
Membar 问：同一块 shared 数据会不会被过早覆盖/读取？
ProxyFence 问：前一个 proxy 的写，后一个 proxy 到底看不看得见？
```

前者是数据 hazard，后者是顺序 / 可见性 hazard。

### 7.5 最后单独记：自动补齐不是“越多越安全”，而是“先保守，再去重”

这节最后最值得抓住的是编译器态度：

```text
先在安全侧过近似
再尽量识别已有同步边界，避免重复插
```

这也是为什么你会同时看到两种现象：

1. 有时编译器会补一个你源码里没写的 barrier / fence
2. 有时它又会识别“这里其实已经同步过了”，于是选择不再重复补

具体的“去重 / 抑制”机制包括：

- `containsLocalBarrier`
- `MemWaitOpTrait`
- `canSkipBarSync`（`TritonGPUToLLVM.cpp:296`）

所以阅读 dump 时，最稳的顺序是：

```text
先看新增同步点属于哪一类 hazard repair
再看它是不是在复用已有 completion / barrier 边界
最后才问它为什么正好落在这个程序点
```

---

## 8. 为什么需要同步 —— 因果链

前面各节已经把原语分成了三种职责：

- `§3` 会合：哪些线程必须先到齐
- `§4` 完成：异步动作什么时候真的结束
- `§5` 顺序：哪些写必须先对哪些使用者可见

这一节要回答的是更根本的问题：

```text
为什么 GPU 上一定会反复长出这三类同步？
```

先给一句总判断：

```text
同步不是 Triton “额外加的仪式感”，
而是 GPU 执行模型里四类根本不确定性被显式补齐后的结果。
```

下面按“硬件现实 -> 如果不补会怎样 -> Triton/PTX 用什么补 -> 对应到前面哪类职责”
这条链来读。

### 8.1 第一类根因：线程组不是天然锁步的

先抓一句话：

```text
一个 CTA 里的不同 warp 会被独立调度，
所以“我刚写完，别人肯定马上看见”这件事默认并不成立。
```

#### A. 硬件现实

SIMT 不等于“整个 block 永远锁步”。实际情况是：

- warp 0 可能已经跑到下一段
- warp 3 还停在前一段
- 同一个 CTA 内的 shared memory 读写顺序，并不会因为它们属于同一个 block 就自动成立

#### B. 如果不补会怎样

最典型就是 CTA 内 shared memory 上的三类数据 hazard：

- **RAW**：producer 还没写完，consumer 就先读
- **WAR**：consumer 还没读完，producer 就先覆盖
- **WAW**：两个写之间顺序不确定

这些问题最可怕的点在于：它们经常不是“每次都错”，而是会表现成：

```text
这个 tile size 下没事
换一个 occupancy / GPU / 调度时机就出错
```

也就是典型的调度相关竞争。

#### C. Triton / PTX 怎么补

这类根因主要对应：

- `ttg.barrier` / `bar.sync 0`
- 命名 barrier `bar.sync N`
- `bar.warp.sync`
- cluster barrier（如果范围已经跨 CTA）

在编译器自动补齐层面，最直接的修复者是 `MembarAnalysis`：它看 shared memory slice
冲突，再补 `ttg.barrier local`。

#### D. 它对应前面的哪类职责

这是最典型的 **会合 + 数据 hazard repair** 问题：

- 在语义层是 `§3` 的会合类
- 在自动插入层是 `§7` 的 shared memory hazard repair

### 8.2 第二类根因：异步引擎和 SM 不是同一个时间轴

先抓一句话：

```text
cp.async / TMA / wgmma / tcgen05 返回“已发起”，
不等于结果已经能安全消费。
```

#### A. 硬件现实

这些异步引擎有一个共同点：

- 指令发出去后，SM 可以继续跑
- 真正的数据搬运 / 计算完成发生在稍后

所以“issue”和“complete”被硬件主动拆开了。

#### B. 如果不补会怎样

如果 consumer 把“已发起”误当成“已完成”，就会出现：

- smem tile 还没搬完就去 `local_load`
- accumulator 还没就绪就去读
- TMEM 结果还没稳定就被下一阶段消费

这类错误和前一类不同，它不主要是“谁和谁到齐”，而是：

```text
异步 producer 的结果
到底什么时候才算真的 ready
```

#### C. Triton / PTX 怎么补

这类根因主要对应 `§4` 和 `§6` 的完成协议：

- `cp.async.commit_group -> wait_group`
- `barrier_expect -> async_tma_load -> wait_barrier`
- `wgmma.commit_group -> wait_group`
- `tc_gen5_commit -> wait_barrier`

这里的关键不是 barrier 名字，而是 completion model：

- `sm80` 主要是 per-thread async group
- `sm90` 开始分成 TMA 的 `mbarrier` 和 WGMMA 的 group-wait
- `sm100` 把 tcgen05 completion 也接回 `mbarrier`

#### D. 它对应前面的哪类职责

这是最典型的 **完成类协议** 问题：

- `§4` 解释 mbarrier / wait 是怎么工作的
- `§6` 解释同一条主循环在三代 target 上怎么换完成模型

### 8.3 第三类根因：同一块内存不一定只有一个“观察世界”

先抓一句话：

```text
generic proxy 看到的 shared memory，
和 async proxy 看到的 shared memory，
默认不是同一个自动保持一致的视图。
```

#### A. 硬件现实

在 Hopper / Blackwell 这些 target 上，至少要区分：

- generic proxy：普通 SM load/store 这一侧
- async proxy：TMA、`cp.async.bulk`、tcgen05、某些 tensor/copy 引擎这一侧

也就是说，即便“物理上是同一块 shared memory”，它也可能被多个不同的访问域观察。

#### B. 如果不补会怎样

最典型的误判是：

```text
__syncthreads 都过了，
那 async engine 肯定也看到最新数据了吧？
```

不对。`bar.sync` 只说明：

- CTA 内线程在 generic proxy 这一侧会合并建立顺序

它不自动说明：

- async proxy 那一侧也已经拿到了同样的可见性顺序

所以会出现：

- generic 刚写完 smem
- TMA / WGMMA / tcgen05 读的还是旧视图

#### C. Triton / PTX 怎么补

这类根因主要对应 `§5` 的 fence：

- `fence.proxy.async.shared::{cta,cluster}`
- `tensormap_fenceproxy_acquire`
- `fence.mbarrier_init.release.cluster`

它们补的是：

- shared tile 数据的跨 proxy 顺序
- descriptor / tensormap 元数据对 TMA 单元的可见性
- mbarrier 对象自身初始化对 peer CTA 的可见性

在自动补齐层面，对应的是：

- `FenceInsertion`
- `ProxyFenceInsertion`

#### D. 它对应前面的哪类职责

这是最典型的 **顺序 / 可见性类协议** 问题：

- `§5` 讲 fence 到底在补哪条 happens-before
- `§7` 讲 proxy 顺序缺口为什么会被编译器自动修

### 8.4 第四类根因：即便顺序要补，补到多大范围也不是免费的

先抓一句话：

```text
同步范围不是越大越好；
越宽的 scope，代价越高，也越容易过同步。
```

#### A. 硬件现实

GPU 上至少有这样一条粒度链：

```text
warp < warp-group < CTA < cluster < GPU < system
```

一个只需要 warp 内成立的顺序，如果硬补成 cluster 或 GPU 范围，当然也可能“正确”，但：

- 延迟更高
- 干扰更大
- 可能把原本可并行的东西不必要地锁死

#### B. 如果不区分 scope 会怎样

你会看到两种坏结果：

1. **scope 太小**
   - 读方根本不在这个范围里，顺序不成立
2. **scope 太大**
   - 虽然正确，但严重过同步，吞掉重叠和并发

这也是为什么同步原语不仅有“有没有”，还有“作用到哪一层”的问题。

#### C. Triton / PTX 怎么补

这类根因决定了前面几节里所有“粒度选择”都不是随手写的：

- `bar.warp.sync` 只锁一个 warp
- 命名 barrier 只锁 CTA 子集
- `ttg.barrier` 主要是 CTA 范围
- `barrier.cluster.*` / cluster-scoped mbarrier 才跨 CTA
- `tensormap_fenceproxy_acquire` 用 `.gpu`
- `fence.mbarrier_init.release.cluster` 用 `.cluster`

也正因为 scope 是一条性能杠杆，Triton 才会：

- 在 `ttg.barrier` 上带 `addrSpace` bitmask
- 只在需要时把 `.cta` 升到 `.cluster`
- 对 sub-CTA 协作使用 warp barrier 或命名 barrier

#### D. 它对应前面的哪类职责

这不是某一个原语独有的问题，而是前面所有章节的共同约束：

- `§1` 讲 scope 词汇表
- `§3` 讲会合粒度
- `§5` 讲 fence scope
- `§6` 讲不同 target 为什么会选不同 completion / barrier 组织

### 8.5 最后把四类根因压成一张因果表

| 硬件现实 | 默认缺口 | Triton/PTX 补什么 | 对应章节 |
|---|---|---|---|
| CTA 内 warp 独立调度 | shared 数据顺序不自动成立 | `ttg.barrier`、命名 barrier、warp barrier、cluster barrier | `§3`、`§7` |
| 异步引擎与 SM 并发 | “已发起”不等于“已完成” | `wait_group`、`mbarrier.wait`、`tc_gen5_commit -> wait_barrier` | `§4`、`§6` |
| 多个 proxy 各自有可见性世界 | generic 视图不自动等于 async 视图 | `fence.proxy.async`、tensormap acquire fence、mbarrier-init release fence | `§5`、`§7` |
| scope 越宽代价越高 | 太小不正确，太大过同步 | warp / CTA / cluster / GPU 各级 barrier/fence 选择 | `§1`、`§3`、`§5`、`§6` |

### 8.6 为什么编译器宁可保守，也不赌“通常没事”

把前面四类根因合起来，最终就会得到一个非常现实的结论：

```text
GPU 同步 bug 往往不是“必现错误”，
而是“依赖调度、规模、occupancy、target 的竞争”
```

也就是说，少一个 barrier / wait / fence，经常不是立刻稳定崩，而是：

- 某个 tile size 下通过
- 某块 GPU 上通过
- 某个 occupancy 下通过
- 一换 target 或编排方式就失败

这正是为什么 Triton 更倾向于：

1. 先用 `MembarAnalysis`、`TMemBarrierInsertion`、`FenceInsertion`、`ProxyFenceInsertion`
   在安全侧过近似
2. 再用 `containsLocalBarrier`、`MemWaitOpTrait`、`canSkipBarSync` 之类机制把明显冗余的
   那些抠回来

所以“为什么需要同步”的最终答案不是一句“为了 correctness”就结束，而是：

```text
因为 GPU 的执行、完成、可见性、scope 这四条轴默认都不是免费自动对齐的；
Triton 只是把这些缺口显式化、协议化、再在必要时自动修补。
```

---

## 9. 统一心智模型与速查

这一节不再引入新知识，只做一件事：

```text
把前面分散的会合 / 完成 / 顺序 / 自动补齐
压成一套可以随时拿来判断 IR 的统一框架
```

### 9.1 先用一棵判断树读任何一个同步点

看见一段 `wait / barrier / fence / commit / arrive`，先不要盯名字，先走这棵树：

1. 我现在看到的是“**线程到齐**”吗？
   - 是：先去 `§3`
   - 典型原语：`ttg.barrier`、cluster barrier、`bar.warp.sync`、命名 barrier
2. 我现在看到的是“**异步动作完成**”吗？
   - 是：先去 `§4` 和 `§6`
   - 典型原语：`mbarrier.wait`、`cp.async.wait_group`、`wgmma.wait_group`、
     `tc_gen5_commit -> wait_barrier`
3. 我现在看到的是“**补顺序 / 可见性**”吗？
   - 是：先去 `§5`
   - 典型原语：`fence_async_shared`、`tensormap_fenceproxy_acquire`、
     `fence_mbarrier_init_release_cluster`
4. 这些同步点是源码里没写、后来多出来的吗？
   - 是：再去 `§7`
   - 说明你看到的是 hazard repair，不只是协议本体

最容易混掉的是第 2 和第 3：

- **completion** 回答的是“异步动作做完了没”
- **ordering / visibility** 回答的是“前后的使用者之间有没有建立 happens-before”

这两件事经常同时出现，但不是一回事。

### 9.2 再压成一个统一心智模型：先分工，再交接，再修补

如果只记一句最总的结构，可以记：

```text
mapping = 分工
organization = 衔接
scheduling = 时序
protocol = 交接
repair = 修补
```

放回这篇文档的语境里：

| 层 | 它在回答什么 | 这篇里主要落到哪 |
|---|---|---|
| distributed execution mapping | 谁负责哪部分 tensor 元素 | 不是本文主线，但决定谁会参与后续同步 |
| layout / data-movement organization | 数据以什么 form / carrier 流动 | 决定 shared / descriptor / TMEM 这些对象怎么出现 |
| target-driven scheduling | 谁先谁后、是否 overlap、分几 stage | 决定为什么需要某套 protocol |
| protocol | 这次 producer / consumer 交接怎么完成 | `§3`、`§4`、`§5` |
| hazard repair | 源码没明写时，编译器自动补什么 | `§7` |

所以真正稳定的阅读顺序不是“背 op 名”，而是：

```text
先看谁和谁在交接
再看交接问题属于：
到齐？
完成？
可见性？
最后再看编译器有没有自动修补
```

### 9.3 再压成一个对象视角：到底在保护什么对象

前面所有同步，最终都围绕几类对象打转。按对象看，结构会非常稳定：

| 对象 | 常见风险 | 典型同步 |
|---|---|---|
| shared tile / shared buffer | CTA 内数据复用、generic↔async proxy 边界 | `ttg.barrier`、`fence_async_shared`、`wait_barrier` |
| mbarrier 对象本身 | 初始化、arrival/byte 记账、phase 轮转 | `init_barrier`、`barrier_expect`、`arrive_barrier`、`wait_barrier` |
| descriptor / tensormap 元数据 | TMA 单元是否看到最新元数据 | `tensormap_fenceproxy_acquire` |
| TMEM 结果 / 使用边界 | `load->mma` / `store->mma` hazard、异步 MMA 完成 | `tc_gen5_commit -> wait_barrier`、`TMemBarrierInsertion` |
| CTA / cluster 中的线程组 | 谁必须先到齐 | `bar.sync`、cluster barrier、warp barrier |

这张表最想帮你建立的感觉是：

```text
同步原语不是围着“指令名字”组织的，
而是围着“对象 + 使用者 + 风险”组织的
```

### 9.4 最后给一张速查表

| 原语 | 它主要属于哪一类 | 作用对象 / 存储 | 粒度 | 阻塞? | 最该记住的保证 |
|---|---|---|---|---|---|
| `ttg.barrier` / `bar.sync 0` | 会合 | CTA 内 shared / 相关局部状态 | per CTA | 是 | CTA 内线程先到齐，再继续 |
| `bar.sync N`（命名） | 会合 | CTA 子集共享状态 | 线程子集 | 是 | 只同步注册到该 barrier 的线程 |
| `bar.warp.sync` / `__syncwarp` | 会合 | warp 内寄存器 / smem 使用边界 | per warp（32） | 是 | 只重收敛一个 warp |
| `barrier.cluster.arrive` / `wait` | 会合 | DSMEM / cluster 协作状态 | per cluster | wait 阻塞 | cluster 内多个 CTA 对齐 |
| `mbarrier.arrive` / `arrive_barrier` | 完成记账 | smem mbarrier 对象 | 1 发起，效果 CTA/cluster | 否 | 往 barrier 里记 arrival / bytes |
| `mbarrier.try_wait` / `wait_barrier` | 完成观察 | smem mbarrier 对象 | 谁执行谁阻塞 | 是 | 被跟踪的异步事务已完成 |
| `cp.async.commit/wait_group` | 完成观察 | cp.async group / smem tile | per executing thread | 仅 wait | 本线程发出的 cp.async 已完成 |
| `wgmma.fence` / `commit` / `wait_group` | 完成观察 + 顺序 | wgmma 输入 / accumulator | per warp-group（128） | 仅 wait | 输入已对 wgmma 可见，异步 MMA 已可回收 |
| `tc_gen5_commit -> wait_barrier` | 完成观察 | TMEM 结果 + smem mbarrier | warp-group / cta_group | 经 wait | tcgen05 完成被折回 mbarrier 观察 |
| `clc_try_cancel -> wait_barrier` | 完成观察 | smem result buffer + mbarrier | 1 发起 | 经 wait | CLC 结果已经写好 |
| `fence.proxy.async.shared` | 顺序 / 可见性 | shared tile / shared buffer | thread（scope cta/cluster） | 否 | generic 与 async proxy 之间建立顺序 |
| `tensormap_fenceproxy_acquire` | 顺序 / 可见性 | descriptor / tensormap 元数据 | GPU | 否 | descriptor 写先于 TMA 单元读 |
| `fence.mbarrier_init.release.cluster` | 顺序 / 可见性 | smem mbarrier 对象 | thread（cluster） | 否 | mbarrier init 先于 peer CTA 使用可见 |
| `TMemBarrierInsertion` 插的 `ttg.barrier local` | 自动修补 | TMEM 使用边界（形式上 local） | per CTA | 是 | 隔离 TMEM 数据 hazard，不是 completion 本体 |

### 9.5 补充：ARef 不是新原语，而是同步抽象（NVWS 层）

读 NVWS IR 时会看到 `nvws.aref.{create,put.enter,put.exit,get.enter,get.exit}`，容易
误以为它是另一类硬件同步原语。不是。`ARef` 是 NVWS 层的同步抽象：把“底层 buffer +
producer/consumer 配对 + stage/phase + token 关系”封成一个 SSA 值，方便
warp specialization / software pipelining 在高层变换，最后再统一降成 §4 的硬件
mbarrier 协议。

压缩成一句结论：

- `ARef` 本身不提供新的硬件语义，类型定义就说明 lowering 时再插合适 barrier。
- `LowerAref` 主要展开成 empty/full mbarrier 协议，并按 `async kind` 分派成
  `arrive_barrier`、`barrier_expect`、`tc_gen5_commit`、或必要的
  `fence_async_shared`。
- stage/phase 轮转只决定 `wait_barrier %bar, %phase` 的 phase / slot 选择，不是新
  barrier/fence 语义。

所以本文只把它当边界说明；若要追这层抽象，再读 §9.6 里的 NVWS 源码入口。

### 9.6 关键文件与源码阅读顺序

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
   `fence_async_shared`（配 `NVWSTypes.td:34` 的 meta-type 定义，见 §9.5）。
10. `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/`：`BarrierOpToLLVM.cpp`、
   `ClusterOpsToLLVM.cpp`、`ConvertWarpSpecializeToLLVM.cpp`、`TMAToLLVM.cpp`、
   `LoadStoreOpToLLVM.cpp`、`DotOpToLLVM/{WGMMA,MMAv5}.cpp`、`NVGPUToLLVMPass.cpp`。
11. `test/Conversion/tritonnvidiagpu_to_llvm.mlir` —— Triton op 到 PTX/NVVM 的
    具体对照（如 `:545` 的 `fence.proxy.tensormap::generic.acquire.gpu`）。

### 9.7 需要时再回看的 PDF 章节
- `mbarrier` phase-parity 语义：PTX ISA 9.3 §9.7.14.16（parallel synchronization /
  `mbarrier`）。
- `fence.proxy` 的 proxy 定义与 async-proxy coherence：PTX ISA 9.3 §8.6（proxy）
  + §9.7.14.5（`fence.proxy` 指令），以及 §8（memory consistency model）。
- cluster / DSMEM scope 与 `cluster.sync()`：CUDA Programming Guide
  "Thread Block Clusters" + "Distributed Shared Memory"。
