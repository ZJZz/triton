# CUDA Programming Guide 4.9 / 4.11 学习笔记

本文整理自 `cuda-programming-guide.pdf` 的以下两节：

- `4.9 Asynchronous Barriers`，PDF 第 304 页起
- `4.11 Asynchronous Data Copies`，PDF 第 331 页起

目标不是逐段翻译，而是从 CUDA 内核开发的角度提炼出：

- 这些机制解决什么问题
- 正确的使用语义是什么
- 最容易踩的坑在哪里
- 实战里应该如何选择和组合

## 1. 总览

这两节本质上是一组内容：

- `4.9` 讲同步原语，解决“什么时候可以继续”
- `4.11` 讲异步搬运，解决“数据怎么提前搬过来”

两者组合起来，就是现代 CUDA 内核里常见的模式：

- 异步发起数据搬运
- 在搬运进行时做独立计算
- 用 barrier 或 pipeline 在真正需要数据时再等待

和传统写法相比：

- 传统同步：`__syncthreads()` / `__syncwarp()`
- 传统搬运：`ld.global -> reg -> st.shared`
- 现代写法：`async copy + split arrive/wait + pipeline`

核心收益是 `overlap copy with compute`。

## 2. 4.9 Asynchronous Barriers

### 2.1 核心模型

异步 barrier 和 `__syncthreads()` 最大的区别是，它把同步拆成了两个动作：

- `arrive()`：声明“我这一侧已经到达同步点”
- `wait(token)`：等待这一轮 barrier 真正完成

这意味着线程可以：

1. 先 `arrive()`
2. 继续做一段独立工作
3. 在真正需要同步结果时再 `wait()`

这就是所谓的 split arrive/wait。

### 2.2 初始化

barrier 必须先初始化，之后线程才能参与。

典型模式：

1. 一个线程执行 `init(&bar, expected_count)`
2. 用 `block.sync()` 或 `__syncthreads()` 让所有参与线程都看到初始化结果

这里的 `expected_count` 不是“固定等于线程数”的概念，而是：

- 当前 phase 内预计会发生多少次 `arrive()`

如果整个 block 的所有线程都参与一次 arrive，那么通常就是 `block.size()`。

### 2.3 Phase 语义

barrier 不是一次性对象，而是按 phase 周期性工作。

每个 phase 的行为可以理解为：

1. barrier 以 `expected_count` 作为倒计数起点
2. 每次 `arrive()` 都会递减倒计数
3. 计数减到 0，当前 phase 完成
4. barrier 自动 reset，进入下一 phase

`arrive()` 会返回一个 token。这个 token 属于“当前 phase”，随后 `wait(token)` 只能用于：

- 当前 phase
- 或紧邻的下一 phase

如果 token 对应的 phase 过旧或不匹配，行为是未定义的。

### 2.4 什么时候该用 barrier

适合使用 barrier 的情况：

- 需要把“到达”和“等待”拆开
- 需要让一部分线程提早发起同步，再做别的工作
- 需要跟异步 copy 联动
- 需要 producer-consumer 结构

不适合强行替代的情况：

- 只是单纯想同步整个 thread block
- 只是单纯想同步一个完整 warp

官方建议很明确：

- 整个 block 同步优先用 `__syncthreads()`
- 整个 warp 同步优先用 `__syncwarp()`

因为这些专用原语通常更直接，也更高效。

### 2.5 Warp Entanglement

这一点非常关键，也非常容易被忽略。

当 warp 在调用 arrive 类操作之前已经发生分支发散时，barrier 更新行为会变复杂：

- warp 完全收敛时，barrier 更新一次
- warp 完全发散时，可能变成 32 次独立更新

结果就是：

- barrier 开销变大
- 时序更难推断

建议：

- 尽量在收敛状态下调用 `arrive`
- 如果前面有明显分支发散，可先 `__syncwarp()`

### 2.6 Explicit Phase Tracking

默认 barrier 常见写法是：

- `token = bar.arrive()`
- `bar.wait(std::move(token))`

但在一些更底层、更追求性能的场景里，可以不依赖 token，而是显式跟踪 phase parity：

- 初始 phase parity 为 0
- 每完成一轮就翻转一次
- 用 `mbarrier_try_wait_parity()` 等待 phase flip

这种方式在“只有少数线程 arrive，其余线程只是等待”的场景里更有价值，尤其是和异步事务跟踪结合时。

### 2.7 Early Exit

这是 barrier 使用里最容易死锁的点之一。

如果某个线程参与了一系列 barrier，但中途要退出，不能直接 `return`。必须先调用：

- `arrive_and_drop()`

它的含义是：

- 当前 phase 该线程仍完成自己的到达义务
- 从下一 phase 开始，这个线程不再计入 expected count

如果忘了 drop，剩余线程可能会永远等待一个再也不会 arrive 的线程。

### 2.8 Completion Function

`cuda::barrier` 支持 completion function。

它的语义是：

- 每个 phase 完成时执行一次
- 发生在“最后一个 arrive”之后
- 发生在“等待线程被唤醒”之前

这适合做一些每 phase 只需要执行一次的工作，例如：

- 对 shared memory 里的中间结果做归约
- 更新某个共享状态

需要记住的内存语义：

- phase 内参与线程在 barrier 前的内存写入，对 completion function 可见
- completion function 的内存写入，对 wait 返回后的线程可见

### 2.9 Tracking Asynchronous Memory Operations

从 compute capability `9.0+` 开始，shared-memory barrier 可以跟踪异步内存事务。

这时 barrier 不仅等待“线程 arrive 完”，还可以等待“绑定到当前 phase 的异步事务完成”。

可以把它理解为 barrier 增加了一条语义：

- 不光等人到齐
- 还等货到齐

这正是它和 `4.11` 里异步数据搬运配合的关键。

### 2.10 Producer-Consumer 模式

barrier 很适合表达 block 内的 producer-consumer。

典型写法通常需要：

- 一个 `ready` barrier，表示缓冲区可写
- 一个 `filled` barrier，表示缓冲区已填满
- 双缓冲，让生产和消费并行

这是后面 pipeline、warp specialization、TMA 例子背后的共同骨架。

## 3. 4.11 Asynchronous Data Copies

### 3.1 这一节在讲什么

这一节不是在重复“拷贝 API 的语法”，而是在讲：

- GPU 内部数据层级之间，如何用异步方式搬数据
- 不同异步 copy 机制分别适合什么粒度和场景
- 它们如何和 barrier / pipeline 结合

文中主要覆盖三类机制：

- `LDGSTS`
- `TMA`
- `STAS`

### 3.2 三种机制的定位

#### LDGSTS

适合：

- 小粒度
- 按元素
- `global -> shared`

典型场景：

- halo 加载
- 条件性加载
- 小块 prefetch

支持：

- CC `8.0+`
- 传输粒度 `4 / 8 / 16` 字节

#### TMA

适合：

- 大块搬运
- tile 化加载/回写
- 多维数组
- 规则 tensor 布局

典型场景：

- GEMM
- attention
- stencil tile
- Hopper 上的高带宽 shared staging

支持：

- CC `9.0+`
- 1D contiguous bulk copy
- 最多 5D 的 bulk-tensor copy

#### STAS

适合：

- cluster 内 block 间传递小数据
- 从寄存器直接异步写 distributed shared memory

支持：

- CC `9.0+`
- 传输粒度 `4 / 8 / 16` 字节

### 3.3 选型原则

可以先用下面这个粗粒度判断：

- 小块、零散、逐元素搬运：`LDGSTS`
- 大块、规则、尤其是多维 tile 搬运：`TMA`
- cluster 内 block 间的小数据传递：`STAS`

## 4. LDGSTS

### 4.1 基本语义

LDGSTS 的核心目标是：

- 把 global memory 中的小块数据异步搬到 shared memory
- 让发起拷贝的线程继续执行
- 尽可能隐藏 global memory latency

注意它的限制：

- 方向只支持 `global -> shared`
- 大小只支持 `4 / 8 / 16` 字节
- 指针对齐必须满足对应字节数要求

文档还特别指出：

- 4B 和 8B 拷贝走 L1 access 模式
- 16B 拷贝可启用 L1 bypass，减少 L1 污染

最佳性能通常要求：

- global memory 对齐良好
- shared memory 对齐良好
- 最好都接近 `128B` 对齐

### 4.2 完成机制

LDGSTS 可以和以下机制配合：

- shared memory barrier
- pipeline

但一个很容易误解的点是：

- 默认每个线程只等待它自己发起的 LDGSTS copy

所以如果某线程发起的 copy 结果要被整个 block 使用，仅仅 wait copy 完成还不够，通常还需要：

- `__syncthreads()`

否则其他线程对 shared memory 的消费时序可能并未统一。

### 4.3 三类典型模式

#### 条件分支里的批量加载

在 stencil / halo 一类代码中，分支往往导致同步加载写法变成：

- 多个 `LDG`
- 夹杂多个 `STS`

不利于隐藏内存延迟。

用 async copy 替代后有两个好处：

- 数据可直接进 shared，减少寄存器中转
- 多个加载更容易同时 in-flight

#### Prefetch Pipeline

这是最重要的实践模式之一。

基本思路：

- 当前 batch 在算
- 下一 batch 正在拷

通常会用：

- 双缓冲或多级 buffer
- `pipeline.producer_acquire/commit`
- `consumer_wait/release`

这类模式的本质就是通过 staged prefetch 把访存延迟藏到计算后面。

#### Warp Specialization

做法是：

- 一个 warp 专门负责拷贝
- 其他 warp 专门负责计算

再叠加：

- 双缓冲
- barrier 或 pipeline

就形成典型的 producer-consumer 结构。

### 4.4 cooperative_groups::memcpy_async 的注意点

文档专门指出：

- `cooperative_groups::memcpy_async` 在某些场景下比更底层 API 低效

原因是它可能对每次 copy 自动立即 commit，失去“把多个 copy 合并后统一 commit”的机会。

所以如果你在追求更极限的流水控制：

- `cuda::memcpy_async + pipeline`
- 或低层 primitive

通常更值得优先考虑。

## 5. TMA

### 5.1 为什么需要 TMA

很多高性能内核不是搬几个标量，而是搬一整块 tile。

如果还让线程自己做：

- 多维地址计算
- 边界处理
- shared 布局变换

代码复杂、容易错，而且浪费执行资源。

TMA 的价值就是把这些 bulk copy 和地址解释工作更多交给硬件。

### 5.2 1D TMA

对于一维连续数组，TMA 可用于 bulk asynchronous copy。

关键点：

- 一般由单个线程发起更好
- `global -> shared` 时通常用 shared memory barrier 做完成通知
- `shared -> global` 时通常由发起线程通过 bulk async-group 等待

必须记住的要求：

- global 地址 `16B` 对齐
- shared 地址 `16B` 对齐
- 拷贝大小是 `16B` 的倍数

还有一个非常关键的行为差异：

- `cuda::memcpy_async` 只有在满足 16B 对齐和 16B 倍数大小时才会使用 TMA
- 否则它会退化为同步 copy

相反：

- `cuda::device::memcpy_async_tx`
- `cuda::ptx::cp_async_bulk`

是强制使用 TMA 的。如果要求不满足，行为未定义。

### 5.3 为什么推荐单线程发起 TMA

官方示例强调：

- 最好由一个明确唯一的线程发起 TMA 操作

原因不只是语义清晰，更是为了帮助编译器生成高效代码。

文档甚至提醒：

- 仅仅写 `if (threadIdx.x == 0)` 也可能不足以让编译器确认“这里只有一个线程发起”
- 某些情况下可能产生不理想的串行化代码

因此更稳妥的方式是：

- `elect_sync`
- 或 `cooperative_groups::invoke_one`

### 5.4 Barrier 与 transaction count

TMA 从 `global -> shared` 的关键优势之一，是可以和 barrier 的 transaction tracking 配合。

典型语义是：

1. 某个线程发起 bulk copy
2. barrier 记录本 phase 还期待多少字节到达
3. 所有参与线程对 barrier arrive
4. 只有当线程义务和数据事务都完成，barrier 才 flip

这比“线程自己猜数据什么时候到了”可靠得多。

### 5.5 Proxy Fence

这是 TMA 最容易漏掉、也最致命的 correctness 细节之一。

如果线程们先写 shared memory，然后又让 TMA 从 shared memory 读取并写回 global memory，那么必须先建立可见性顺序：

- `fence_proxy_async(space_shared)`

原因是：

- 线程普通写 shared memory 走的是 generic proxy
- TMA 从 shared 读取走的是 async proxy

没有这个 fence，就不能保证前面的 shared 写入已经对 TMA 引擎可见。

通常模式是：

1. 各线程写 shared
2. 每线程执行 `fence_proxy_async(space_shared)`
3. 再用 `__syncthreads()` 统一时序
4. 再由单线程发起 TMA store

### 5.6 多维 TMA 与 Tensor Map

多维 TMA 的关键不是 copy 本身，而是 tensor map。

它描述：

- global tensor 的维度
- stride
- tile box 大小
- element stride
- swizzle 模式

常见用法是：

- host 端用 `cuTensorMapEncodeTiled`
- 再作为 `const __grid_constant__ CUtensorMap` 传进 kernel

这是官方推荐方式，因为：

- 语义清晰
- 访问成本低
- 避免从 global memory 再读 tensor descriptor

备选方式包括：

- 放到 device `__constant__`
- 放到 global memory

但如果 tensor map 放在 global memory 并且会被更新，就必须处理 tensormap proxy 的 release-acquire 可见性问题。

### 5.7 TMA Swizzle

这是 Hopper 上非常实用的优化点。

默认情况下，TMA 按 global memory 的原始布局写 shared memory。  
这未必适合 shared memory 的访问模式，尤其是：

- 转置
- 列访问
- 某些 tile reshape

这时很容易出现 shared memory bank conflict。

TMA 允许在写入 shared 时施加 swizzle，并在写回 global 时自动 unswizzle。

这样做的意义是：

- global 中保持自然布局
- shared 中改成更利于并发访问的布局

对矩阵转置这类场景，swizzle 往往能显著降低 bank conflict。

文档里给出的关键要求包括：

- global memory 通常需要 `128B` 对齐
- shared memory 也要满足相应对齐要求
- swizzle 的 inner dimension 必须符合模式要求
- swizzle 的粒度固定是 `16B`

工程上要注意：

- swizzle 不是“开了就快”
- 它要求你同时改变 shared memory 的访问索引方式

也就是说：

- 数据布局变了
- 用户代码的 shared 索引逻辑也必须跟着变

## 6. STAS

### 6.1 它解决什么问题

在 thread block cluster 编程里，有时需要把少量数据快速传到别的 block 的 distributed shared memory。

如果为了几个标量还绕回 global memory，代价很高。

STAS 提供了更直接的路径：

- 从寄存器异步写到 distributed shared memory

### 6.2 特性与限制

- 仅支持 CC `9.0+`
- 仅支持 `4 / 8 / 16` 字节
- 方向固定：`register -> distributed shared memory`
- 完成机制依赖 shared memory barrier

### 6.3 适用场景

适合：

- cluster 内 block 间 producer-consumer
- 传小规模中间结果
- 传控制信息或 ring buffer 元信息

如果你的算法主要是大块 tile 搬运，重点还是 TMA。  
STAS 更像 cluster 级细粒度通信工具。

## 7. 这两节串起来怎么理解

可以把这两节看成“同步协议 + 数据引擎”的组合：

- `4.9` 提供阶段化同步协议
- `4.11` 提供不同层级和粒度的数据搬运引擎

常见组合方式：

- `LDGSTS + barrier`
- `LDGSTS + pipeline`
- `TMA + barrier(transaction count)`
- `warp specialization + double buffering + barrier`
- `cluster + STAS + barrier`

它们共同服务于一个目标：

- 尽量让数据搬运和计算并行发生

## 8. 高频踩坑总结

### correctness 类

- barrier 初始化后没有做 bootstrap 同步
- token 跨了错误的 phase 继续使用
- 线程提前退出却没有 `arrive_and_drop()`
- 只等了 async copy 完成，却忘了 block 内其他线程还没同步
- shared 写后立刻发起 TMA store，却没做 `fence_proxy_async`

### performance 类

- 明明只是全 block 同步，却滥用 barrier 代替 `__syncthreads()`
- arrive 前 warp 严重发散，导致 barrier 更新膨胀
- async copy 粒度太碎，commit 太频繁
- TMA 由多个线程重复发起
- 对齐不满足，导致本以为是 async/TMA，实际退化
- swizzle 开了，但 shared 访问索引没按 swizzle 规则调整

## 9. 在 Triton 中如何体现

这一节从 Triton 的角度重新看 `4.9` 和 `4.11`。

结论先说：

- 在“标准 Triton”里，`4.9 / 4.11` 更多体现为编译器和后端自动使用的机制
- 在 `experimental.gluon` 里，这两节内容已经以相对直接的编程接口暴露出来

也就是说：

- 标准 Triton 更像“你写高层意图，编译器帮你 lower 到 async copy / barrier / TMA”
- Gluon 更像“你显式操控 mbarrier、cp.async、TMA”

### 9.1 先分清两层 Triton

#### 标准 Triton

平时常见的是：

- `import triton.language as tl`
- `tl.load`
- `tl.store`
- `tl.dot`
- `tl.range(..., num_stages=...)`
- `warp_specialize=True`
- `tl.make_tensor_descriptor(...)`

这时用户一般不会显式写：

- `mbarrier.init`
- `mbarrier.expect`
- `cp.async.commit_group`

但这并不意味着 `4.9 / 4.11` 没有发生，而是：

- 这些细节多数被编译器 pass 和后端 lowering 吃掉了

#### Gluon

Gluon 提供了更低层、更接近硬件异步编程模型的接口。

你会显式看到：

- `mbarrier.init / arrive / wait / expect`
- `async_load / commit_group / wait_group`
- `tma.async_load / tma.async_store`
- `fence_async_shared`

这时 Triton 基本已经在直接表达 CUDA Guide `4.9 / 4.11` 的语义。

### 9.2 CUDA 4.9 对应 Triton 的什么

CUDA `4.9 Asynchronous Barriers` 在 Triton 里主要对应：

- `mbarrier`
- warp specialization 下的 producer-consumer 协调
- 编译器自动插入的 barrier 生命周期管理

#### 在 Gluon 里的直接映射

Ampere 与 Hopper 路径都定义了显式 `mbarrier` API：

- `nvidia/ampere/mbarrier.py`
- `nvidia/hopper/mbarrier.py`

接口层面的对应关系很直接：

- CUDA `init(bar, count)` 对应 Triton `mbarrier.init(bar, count)`
- CUDA `arrive()` 对应 Triton `mbarrier.arrive(...)`
- CUDA `wait(token)` / parity wait 对应 Triton `mbarrier.wait(bar, phase=...)`
- CUDA 的 transaction tracking 对应 Hopper 路径的 `mbarrier.expect(bar, bytes)`

其中最关键的一点是：

- Triton 的 `mbarrier.wait(bar, phase=...)` 更接近 CUDA 文档里 `4.9.3 Explicit Phase Tracking`

也就是说 Triton 更偏向：

- 直接等 phase/parity

而不是 libcu++ 里那种：

- `arrival_token + wait(token)`

#### 在标准 Triton 里的间接体现

标准 Triton 用户通常不会手写 barrier，但会写：

- `tl.range(..., num_stages=...)`
- `warp_specialize=True`

这两个高层意图会驱动后端去建立异步流水和同步协议。

源码里可以看到：

- `tl.range(..., num_stages=...)` 会在 IR 上设置 `tt.num_stages`
- `warp_specialize=True` 会在 IR 上设置 `tt.warp_specialize`

它们不是 barrier 本身，但它们会触发 barrier 相关的后端转换和调度。

### 9.3 CUDA 4.11 对应 Triton 的什么

CUDA `4.11 Asynchronous Data Copies` 在 Triton 里主要分成两条路线：

- Ampere 路线：`cp.async` / LDGSTS 类 async copy
- Hopper 路线：`TMA` / tensor descriptor / async proxy

#### 4.11.1 LDGSTS / cp.async 对应 Triton Ampere async copy

在 Gluon 的 NVIDIA Ampere 路径中，有直接的 async copy 接口：

- `async_load(...)`
- `commit_group()`
- `wait_group(num_outstanding=0)`
- `mbarrier_arrive(...)`

它们对应 CUDA 文档 `4.11.1 Using LDGSTS` 的关系如下：

- `async_load`：发起 `global -> shared` 异步拷贝
- `commit_group`：把前面发起的 copy 归入一个 group
- `wait_group`：等待 outstanding groups 下降到指定数量
- `mbarrier_arrive`：把 async copy 完成事件和 barrier 协调起来

所以从 Triton 视角看，CUDA 的：

- `cp.async`
- `cp.async.commit_group`
- `cp.async.wait_group`

在 Gluon 里基本就是：

- `async_load`
- `commit_group`
- `wait_group`

#### 4.11.2 TMA 对应 Triton Hopper tensor descriptor

Hopper 路径里，TMA 是 Triton 对 CUDA `4.11.2 Using the Tensor Memory Accelerator` 最直接的映射。

核心抽象是：

- `TensorDescriptor`
- `tensor_descriptor`
- `tma.async_load`
- `tma.async_store`
- `store_wait`

对应关系可以这样记：

- CUDA `CUtensorMap` / tensor map
- Triton `TensorDescriptor`

- CUDA global->shared bulk async TMA load
- Triton `tma.async_load(...)`

- CUDA shared->global bulk async TMA store
- Triton `tma.async_store(...)` 或 `async_copy_shared_to_global(...)`

- CUDA bulk store wait group
- Triton `tma.store_wait(...)`

这和 CUDA 指南里的结构高度一致：

- 读：通常用 barrier 跟踪完成
- 写：通常用 store group / async-group 跟踪完成

#### 4.11.2 的 transaction tracking 在 Triton 里的体现

在 Hopper 的 Gluon 教程里，一个非常标准的模式是：

1. `mbarrier.init(bar, count=1)`
2. `mbarrier.expect(bar, nbytes)`
3. `tma.async_load(desc, coord, bar, smem)`
4. `mbarrier.wait(bar, phase=...)`

这正是 CUDA 文档中：

- barrier 不只等线程 arrive
- barrier 还跟踪异步事务字节数

的 Triton 版本。

### 9.4 Triton 里的 proxy fence

CUDA 指南 `4.11` 里最容易漏的一点，是：

- TMA 通过 async proxy 访问 shared memory
- 普通 `smem.load/store` 通过 generic proxy 访问 shared memory
- 两者之间默认不自动有序

在 Triton 中，这个概念明确体现为：

- `fence_async_shared()`

对应的使用场景与 CUDA 文档完全一致：

- 线程先往 shared memory 写数据
- 然后发起 `tma.async_store`
- 中间需要 `fence_async_shared()`

否则就可能出现：

- TMA 还没看到 generic proxy 下的 shared 写入

反过来，如果先发起 TMA load，再从 shared 读寄存器数据，也需要理解“哪些同步已经建立了跨 proxy 的可见性，哪些没有”。

### 9.5 warp specialization 在 Triton 中的意义

CUDA 文档里关于：

- producer-consumer
- double buffering
- barrier
- load/compute overlap

在 Triton 中最常见的落点之一就是：

- `warp_specialize`

它本质上是把一个 CTA 内不同 warp 分成不同角色，例如：

- producer warp 专门发起 async copy / TMA load
- consumer warp 专门做 MMA 或算子主体计算

这时：

- `4.9` 提供的 barrier/mbarrier 负责角色间接力
- `4.11` 提供的 async copy / TMA 负责数据流转

所以 `warp_specialize` 不是单独的新概念，而是：

- 把 `4.9 + 4.11` 的能力组织成 Triton 里常见的内核结构

### 9.6 编译器后端如何把它们串起来

在 NVIDIA 后端，Triton 编译流水会显式跑一系列和这两节强相关的 pass：

- `assign_latencies`
- `schedule_loops`
- `pipeline`
- `coalesce_async_copy`
- `tma_lowering`
- `fence_insertion`

这说明对标准 Triton 来说，`4.9 / 4.11` 并不总是用户手写 API，而是：

- 编译器根据 `num_stages`、`warp_specialize`、descriptor load/store 等高层结构
- 自动把代码 lower 成异步 copy、TMA、mbarrier 协调和 fence

因此你在标准 Triton 里看到的：

- `num_stages`
- `descriptor.load/store`
- `TensorDescriptor`

很多时候就是对 `4.9 / 4.11` 的“声明式触发器”。

### 9.7 一张对应表

可以用下面这张表快速建立映射：

- CUDA `4.9 init/arrive/wait`
  Triton Gluon `mbarrier.init / arrive / wait`

- CUDA `4.9 explicit phase tracking`
  Triton `mbarrier.wait(bar, phase=...)`

- CUDA `4.9 tracking async transactions`
  Triton Hopper `mbarrier.expect(bar, bytes)`

- CUDA `4.11 LDGSTS / cp.async`
  Triton Ampere `async_load + commit_group + wait_group`

- CUDA `4.11 TMA tensor map`
  Triton `TensorDescriptor` / `tl.make_tensor_descriptor`

- CUDA `4.11 TMA global->shared`
  Triton `tma.async_load(...)`

- CUDA `4.11 TMA shared->global`
  Triton `tma.async_store(...)` + `store_wait(...)`

- CUDA `4.11 proxy fence`
  Triton `fence_async_shared()`

- CUDA producer-consumer / warp specialization
  Triton `warp_specialize` + compiler WS passes + mbarrier

### 9.8 一个重要判断

如果你平时写的是标准 Triton kernel，那么你应该这样理解：

- `4.9 / 4.11` 已经在用，但多数是“编译器替你在用”

如果你开始使用：

- `TensorDescriptor`
- `warp_specialize`
- Gluon 的 `mbarrier / async_copy / tma`

那么你就已经在显式编写 CUDA Guide `4.9 / 4.11` 这一层的逻辑了。

### 9.9 Triton/NVWS 的 `aref` 和 `TmemAref`

如果你继续往 Triton/NVIDIA 后端内部看，很快会遇到：

- `nvws.aref`
- `nvws-insert-aref`
- `nvws-insert-tmem-aref`

这套东西不是 CUDA Programming Guide 里的官方术语，但它们和 guide 里的语义是强相关的。

#### 什么是 `nvws.aref`

`nvws.aref` 可以理解成：

- 一个“异步引用”
- 一个描述 producer/consumer 之间异步缓冲区交接关系的 IR 抽象

严格说，它在 IR 里同时表现为：

- 一个 type：`!nvws.aref<...>`
- 围绕这个 type 工作的一族 op，例如 `aref.create`、`aref.put.enter`、`aref.put.exit`、`aref.get.enter`、`aref.get.exit`

它不是某条硬件指令，也不是 CUDA 里的单个 API 对象，而是一个更高层的协议对象。

它的职责大致是：

- 把某个 buffer 的所有权交给 producer
- producer 填充或生成数据
- 再把这个 buffer 的可读状态交给 consumer
- consumer 用完后再把它归还

因此它最接近 CUDA guide 里的组合语义，而不是单个概念：

- `4.9 Asynchronous Barriers`
- `4.9.7 Producer-Consumer Pattern Using Barriers`
- `4.10 Pipelines`
- `4.11` 中 async load / TMA handoff 的完成通知

#### 为什么说它不是 CUDA guide 里的单个术语映射

因为 `nvws.aref` 在 lowering 时不会只降成一个东西，而是会展开成：

- empty barrier
- full barrier
- wait
- arrive / commit
- 某些情况下还会插入 proxy fence

也就是说：

- `aref` 本身是协议抽象
- 真正落地到硬件时，才会变成 guide 里那些 barrier / fence / async-copy completion 机制

#### 可以把普通 `aref` 理解成什么

最接近的直觉是：

- 一个带 empty/full 状态的异步缓冲区引用

producer 侧通常是：

1. `aref.put.enter`，先等 empty
2. 写 buffer，或者发起 async load / TMA load
3. `aref.put.exit`，发出 full

consumer 侧通常是：

1. `aref.get.enter`，先等 full
2. 读或消费 buffer
3. `aref.get.exit`，发出 empty

这和 CUDA guide 里 barrier 版 producer-consumer 图几乎是同一个模型。

#### 那 `TmemAref` 又是什么

`TmemAref` 不是一个新的基础类型，而是：

- 针对 TMEM ownership transfer 的专门 aref 插入策略

Triton 的 pass 说明写得很直接：

- 当 TMEM 的 ownership 在不同 partition 之间切换时，插入 aref
- 和普通 `InsertAref` 不同，`InsertTmemAref` 把 `ArefPut/ArefGet` 用作两个 group 之间的 ping-pong ownership transfer
- 当前限制是某个特定 TMEM buffer 的 ownership 最多在两个 group 之间来回切换

所以 `TmemAref` 的关键字不是“async copy”，而是：

- ownership transfer
- ping-pong handoff
- producer partition / consumer partition

#### 为什么 TMEM 需要单独一套 Aref 插入逻辑

因为 TMEM 的核心问题不是：

- “一块 shared memory 数据什么时候 ready”

而是：

- “哪一个 partition 当前拥有这个 TMEM buffer，并且可以合法地继续在它上面做 TMEMStore / TMEMLoad / MMA”

这和普通 SMEM producer-consumer 有相似之处，但重点略有不同：

- 普通 aref 更像 buffer ready/empty 协议
- TmemAref 更像 buffer ownership 协议

#### 它在 CUDA Programming Guide 里有直接映射吗

没有一个单独、同名、精确的一一对应概念。

更接近的语义组合是：

- `4.9` 的 barrier / arrive / wait / phase handoff
- `4.10` 的 pipeline stage ownership 切换
- producer-consumer ping-pong buffer

也就是说：

- `nvws.aref` 更像“异步缓冲区交接协议”
- `TmemAref` 更像“TMEM ownership 交接协议”

#### 它和 proxy / fence 的关系

普通 `aref` lowering 里，如果一侧是 generic proxy，另一侧是 async proxy，编译器还可能插：

- `FenceAsyncSharedOp`

这对应前面讲的：

- `fence_async_shared()`
- `fence_proxy_async(space_shared)`

但 `TmemAref` 的核心关注点不是 generic-vs-async proxy 的共享内存排序，而是：

- 不同 warp-specialized partition 之间，TMEM token / ownership 的流转是否合法

所以从理解顺序上建议这样看：

1. 先把 `aref` 理解成高层 producer-consumer handoff
2. 再把 `TmemAref` 理解成这个 handoff 在 TMEM 上的特化版本
3. 最后再去看它 lowering 到 barrier / token / fence 的细节

### 9.10 Triton 后端 IR 语义关系图

如果把前面这些 Triton/NVIDIA 后端概念放到一张图里，可以按下面这条链理解：

1. `warp_specialize`
2. producer / consumer partition
3. `aref` 或 `TmemAref`
4. lowering 成 `mbarrier` / token / `fence_async_shared`
5. 最终驱动 `cp.async`、`TMA`、`TMEMLoad/Store`、`MMA`

注意：

- 这是一条“概念关系链”，不是编译 pass 的真实执行顺序
- 从心智模型上，把 `warp_specialize` 放在最上层最容易理解
- 但实际 `nvws.warp_group -> ttg.warp_specialize` 的 lowering 发生在 `aref` lowering 之后

也就是说：

- `warp_specialize` 决定“谁干什么”
- `aref / TmemAref` 决定“怎么交接”
- `mbarrier / fence` 决定“什么时候能交、交接是否有序”
- `TMA / cp.async / TMEM` 才是真正搬数据或消费数据的硬件动作

#### 第一层：`warp_specialize`

这是最上层的结构划分。

它把一个 CTA 内的不同 warp 或 warp group 分成不同 partition，例如：

- producer partition
- consumer partition
- epilogue partition

这时 Triton 后端需要解决的问题就变成：

- 一个 partition 产出的 buffer 或 TMEM accumulator，什么时候能被另一个 partition 接手

#### 第二层：`aref`

一旦 partition 之间存在：

- SMEM buffer handoff
- async load 结果 handoff
- producer-consumer 双缓冲

后端就会引入：

- `nvws.aref`

它表达的是：

- “这个 buffer 现在归谁用”
- “下一阶段谁可以读”
- “上一阶段是否已经写完”

所以普通 `aref` 最接近：

- SMEM 上的 empty/full 协议

#### 第三层：`TmemAref`

如果交接的对象不是普通 SMEM buffer，而是：

- TMEM allocation
- TMEM token
- MMA 消费的 TMEM accumulator / operand

后端就会用：

- `TmemAref`

它和普通 `aref` 的区别不在于“也有 barrier”，而在于它表达的是：

- TMEM ownership transfer

因此：

- 普通 `aref` 关注 buffer ready/empty
- `TmemAref` 关注 TMEM 当前属于哪个 partition

#### 第四层：lowering 到 barrier / token / fence

`aref` 和 `TmemAref` 都不是最终执行对象。

lowering 后，后端会把这些高层 handoff 协议拆成：

- empty barrier
- full barrier
- wait
- arrive / commit
- async token threading
- 必要时的 `FenceAsyncSharedOp`

所以可以把 lowering 后的世界理解成：

- `aref` 被编译成一套 barrier 状态机
- `TmemAref` 被编译成一套 ownership/token 状态机

#### 第五层：真正的硬件动作

到了最底层，真正和硬件打交道的才是：

- `cp.async`
- `TMA load/store`
- `TMEMLoad / TMEMStore`
- `wgmma` / `tcgen05.mma`

这些操作本身只负责：

- 搬数据
- 消费数据
- 产出结果

它们不单独解决复杂的跨 partition handoff 问题。  
这个问题是上一层的 `aref / TmemAref + barrier/fence` 在解决。

#### 一张“谁负责什么”的速记表

- `warp_specialize`
  - 负责角色划分
  - 谁是 producer，谁是 consumer

- `nvws.aref`
  - 负责普通异步缓冲区交接
  - 更像 empty/full buffer 协议

- `TmemAref`
  - 负责 TMEM 所有权交接
  - 更像 ownership/token 协议

- `mbarrier`
  - 负责阶段完成与唤醒
  - 对应 handoff 的同步基础设施

- `fence_async_shared`
  - 负责 generic proxy 和 async proxy 的顺序
  - 不负责 completion

- `cp.async / TMA / TMEMLoad/Store / MMA`
  - 负责真正的数据流动和计算
  - 不独自承担高层 handoff 协议

#### 一个最实用的理解方式

看到后端 IR 里这些名字时，可以按下面顺序问自己：

1. 这里有没有 `warp_specialize`？
2. 这里交接的是普通 buffer 还是 TMEM？
3. 如果是普通 buffer，看 `aref`
4. 如果是 TMEM ownership，看 `TmemAref`
5. 再看 lowering 后用了哪些 `mbarrier` / `fence`
6. 最后再看底层是 `cp.async`、`TMA`、还是 `TMEM/MMA`

按这个顺序看，后端 IR 会清楚很多。

## 10. 一条 CUDA async_copy 到底是怎么执行的

学习 Triton 或 CUDA 异步内存模型时，一个特别常见的问题是：

- 一条 `async_copy` 到底是谁在执行
- 它和发起它的线程是什么关系
- `generic proxy` 和 `async proxy` 到底是什么意思

这一节专门回答这几个问题。

### 10.1 先记住一句话

`async_copy` 不是“当前 CUDA 线程自己把数据慢慢搬完再返回”，而是：

1. 当前线程发起一个异步操作
2. 硬件为这次操作关联一个 `async thread`
3. 发起线程立即继续执行
4. `async thread` 在后台完成这次内存动作
5. 程序通过 barrier / wait-group / store-wait 等机制与它重新汇合

所以异步编程模型的核心不是：

- 某条 load/store 更快

而是：

- 发起线程和完成线程在抽象上已经分离

### 10.2 CUDA Guide 的三个关键概念

CUDA Guide 在 `3.2.2.3.1 Async Thread and Async Proxy` 里引入了三个概念：

- `async thread`
- `generic proxy`
- `async proxy`

#### Async Thread

当你发起一条异步操作时，这个操作会被关联到一个：

- `async thread`

它不是发起它的那个 CUDA thread 本身，而是一个单独的异步执行实体。

可以把它理解成：

- CUDA thread 负责“提交工作”
- async thread 负责“在后台把工作做完”

#### Generic Proxy

`generic proxy` 是普通内存访问所在的通道。

典型包括：

- 普通 `ld/st`
- 常规 shared/global memory 访问
- 大多数你平时写的普通线程内内存操作

#### Async Proxy

`async proxy` 是某些专门异步硬件路径使用的通道。

典型包括：

- TMA
- 一些 tensor core 异步操作，例如 `wgmma.mma_async.*`
- Blackwell 路径上的 `tcgen05.*`

### 10.3 两类 async_copy 不要混讲

这里最容易混乱的地方是：

- 并不是所有 async copy 都属于 async proxy

要分成两类。

#### 第一类：async thread operating in generic proxy

这一类包括：

- `LDGSTS`
- `STAS/REDAS`

也就是我们常说的：

- `cp.async` 一类小粒度 `global -> shared` 异步拷贝

它们虽然是异步操作，但仍然被建模为：

- `async thread` 工作在 `generic proxy`

这里值得特别强调：

- `STAS/REDAS` 虽然也是较新的异步指令，但它们不是 `async proxy`
- 它们和 `LDGSTS` 一样，属于 `async thread in generic proxy`

这是一个很好的反例：

- 不是所有“新异步硬件路径”都天然属于 `async proxy`
- 真正要和 TMA 区分开

#### 第二类：async thread operating in async proxy

这一类包括：

- TMA 的 bulk async copy
- 一些 tensor core async 操作

它们被建模为：

- `async thread` 工作在 `async proxy`

这两类的最大差别不只是“硬件不同”，而是：

- 它们和普通 load/store 的默认内存顺序关系不同

### 10.4 一条 cp.async / LDGSTS 的执行流程

以 `cp.async` / LDGSTS 为例，可以把一条异步拷贝的执行流程概括成：

1. 某个 CUDA thread 发起一条 `global -> shared` 的 async copy。
2. 这条 copy 被关联到一个 `async thread`。
3. 发起线程立即继续执行后续指令，不等待 copy 完成。
4. 多条 async copy 可以先累计起来。
5. 通过 `commit_group()` 把这些拷贝归入一个提交组。
6. 之后通过 `wait_group()`，或者借助 barrier，等待这些 copy 真正完成。
7. 完成后，shared memory 中的数据才可安全消费。

它的 completion 机制通常是：

- `commit_group + wait_group`
- 或 `mbarrier` 配合 arrive/wait

这里要注意：

- “已经发起”不等于“已经完成”
- “已经完成”也不自动等于“其他线程已经和这份 shared 数据的使用时序对齐”

所以很多场景下，除了等待 copy 完成，还需要 block 内额外同步。

### 10.5 一条 TMA async copy 的执行流程

对 TMA 来说，流程类似，但语义更重一些：

1. 某个线程发起一个 bulk async copy。
2. 这条 copy 关联到一个 `async thread`。
3. 这个 async thread 工作在 `async proxy`。
4. 发起线程继续执行。
5. 对 `global -> shared` 的 TMA load，通常先设置 barrier 期待的事务字节数，再发起 copy。
6. 通过 `mbarrier.wait(...)` 等待数据真正到达 shared memory。
7. 对 `shared -> global` 的 TMA store，通常通过 `store_wait(...)` 等待 store group 到达所需完成点。

典型模式：

1. `mbarrier.init(bar, 1)`
2. `mbarrier.expect(bar, nbytes)`
3. `tma.async_load(desc, coord, bar, smem)`
4. `mbarrier.wait(bar, phase)`

这相当于：

- barrier 不只同步线程
- barrier 还在追踪异步 copy 的事务进度

这里的 `1` 不是普适模板，而是对应：

- 只有一个 elected 线程负责对这个 barrier arrive

如果是整个 block 的参与线程都对 barrier arrive，那么初始化计数通常不是 `1`，而是参与 arrive 的线程数。

### 10.6 Generic Proxy 下的内存顺序

对于：

- `async thread operating in generic proxy`

CUDA Guide 给出的语义是：

- 在这条异步操作之前，对同一地址的普通 load/store，保证排在它之前
- 但在这条异步操作之后，对同一地址的普通 load/store，不保证自动有序

可以简单记成：

- 对同一地址，前面的普通访问默认在它前面
- 对同一地址，后面的普通访问不会自动等它

所以对 `cp.async` 一类操作来说，风险通常在：

- 你太早读/写它正在涉及的目标地址
- 或者没有正确等待 completion 就使用结果

### 10.7 Async Proxy 下的内存顺序

对于：

- `async thread operating in async proxy`

语义更严格：

- 对同一地址，不管是发起前还是发起后，generic proxy 和 async proxy 之间都不自动有序

也就是说：

- 普通线程在 generic proxy 下写 shared memory
- TMA 在 async proxy 下读 shared memory

这两件事默认没有顺序保证。

因此必须使用：

- `proxy fence`

来显式建立顺序。

这就是为什么 TMA 比 `cp.async` 更容易踩坑。

### 10.8 Proxy Fence 在解决什么

`proxy fence` 的作用不是“等待异步操作完成”，而是：

- 给不同 proxy 之间建立内存顺序

这点必须和 completion 区分开。

#### Completion 负责“什么时候完成”

典型机制：

- `wait_group`
- `mbarrier.wait`
- `store_wait`

#### Proxy fence 负责“谁先谁后”

典型机制：

- `fence_proxy_async(space_shared)`
- 在 Triton 中对应 `fence_async_shared()`

所以一句话概括就是：

- completion 解决 done
- fence 解决 ordering

### 10.9 一个最典型的 TMA hazard

最常见的错误模式是：

1. 线程先往 shared memory 里写数据
2. 接着马上发起 `TMA store`
3. 以为 TMA 一定能看到刚写进去的数据

这是不对的。

原因是：

- shared memory 普通写走 `generic proxy`
- TMA store 从 shared memory 读走 `async proxy`

默认情况下它们不自动有序。

正确模式通常是：

1. `smem.store(...)`
2. `fence_proxy_async(space_shared)` 或 Triton `fence_async_shared()`
3. 再发起 `TMA store`
4. 必要时 `store_wait(...)`

反过来，如果是：

1. 发起 `TMA load`
2. 没等 barrier 完成
3. 就去 `smem.load(...)`

同样也可能读到错误或未就绪的数据。

### 10.10 放到 Triton 里怎么理解

这套模型在 Triton 里有很直接的映射。

#### Triton Ampere async copy

Gluon 的：

- `async_load`
- `commit_group`
- `wait_group`

对应：

- `cp.async / LDGSTS`

本质上就是：

- async thread in generic proxy

#### Triton Hopper TMA

Gluon 的：

- `tma.async_load`
- `tma.async_store`
- `mbarrier.expect`
- `mbarrier.wait`
- `store_wait`

对应：

- TMA bulk async copy

本质上就是：

- async thread in async proxy

#### Triton fence_async_shared

Gluon 的：

- `fence_async_shared()`

对应：

- CUDA proxy fence

它的职责就是：

- 在 generic proxy 和 async proxy 之间建立顺序

### 10.11 一张对比表

可以把 `cp.async` 和 `TMA` 对比记成下面这样：

- `cp.async / LDGSTS`
  - 方向：通常 `global -> shared`
  - 执行模型：`async thread`
  - 所在 proxy：`generic proxy`
  - completion：`commit_group / wait_group`，或 barrier
  - 默认顺序：之前的普通访问在它前面；之后的普通访问不自动等它
  - 主要风险：过早消费 shared 结果，或和后续普通访问竞争

- `TMA`
  - 方向：`global <-> shared`，也可更复杂
  - 执行模型：`async thread`
  - 所在 proxy：`async proxy`
  - completion：`mbarrier.wait`、`store_wait`
  - 默认顺序：对同一地址，和 generic proxy 前后都不自动有序
  - 主要风险：漏掉 proxy fence，导致 generic/async 之间错序

### 10.12 面试式总结

如果要用很短的话回答这个问题，可以直接说：

- CUDA 的 async copy 是“当前线程发起、async thread 后台完成”的模型，不是当前线程自己同步搬完。
- `cp.async / LDGSTS` 属于 `async thread in generic proxy`，所以先前普通访问默认在它前面，但后续普通访问不会自动等它。
- `TMA` 属于 `async thread in async proxy`，和普通 `ld/st` 位于不同 proxy，前后都不自动有序，必须靠 proxy fence 建立顺序。
- 因而要区分两类机制：`wait_group / mbarrier / store_wait` 解决 completion，`fence_proxy_async` 解决 ordering。

## 11. 学习顺序建议

建议按下面顺序学习，而不是完全照 PDF 目录走：

1. `4.9.1` 到 `4.9.4`
先把 barrier 的初始化、phase、token、early exit 学明白。

2. `4.11.1.1` 和 `4.11.1.2`
先看 LDGSTS 的条件加载和 prefetch，这最贴近日常 kernel。

3. `4.9.6`
再回来看 barrier 如何跟踪异步事务，这时更容易理解 barrier 和 copy 的耦合。

4. `4.11.2.1`
再学 1D TMA，重点看 transaction count 和 proxy fence。

5. `4.11.2.2.5`
最后看 TMA swizzle，把 attention 放到 shared bank conflict 和布局优化上。

6. `4.11.3`
如果当前工作涉及 thread block cluster，再进入 STAS。

如果你的目标是迁移到 Triton，可以把顺序改成：

1. 先学本文前面的 CUDA 语义
2. 再看 Triton Gluon 的 `03-async-copy` 和 `04-tma`
3. 最后回到标准 Triton，观察 `num_stages`、`warp_specialize`、`TensorDescriptor` 是如何触发这些机制的

## 12. 记忆版结论

如果只记住最重要的内容，建议记这几条：

- barrier 的本质是把同步拆成 `arrive` 和 `wait`
- barrier 是 phase 化对象，不是一次性栅栏
- 线程提前退出 barrier 体系时，必须 `arrive_and_drop()`
- LDGSTS 适合小粒度 `global -> shared`
- TMA 适合大块、规则、多维 tile 搬运
- TMA 的 `shared -> global` 前，shared 写入通常需要 `fence_proxy_async`
- 真正高性能的核心不是“用了 async API”，而是“copy 和 compute 是否真的重叠起来了”
- 在 Triton 里，标准前端更多是声明意图，Gluon 才更接近直接写 `4.9 / 4.11`
- `cp.async` 和 `TMA` 都是 async thread 模型，但前者在 `generic proxy`，后者在 `async proxy`
- `wait_group / mbarrier / store_wait` 管 completion，`proxy fence` 管 ordering

## 13. 后续实战建议

如果要把这两节真正学会，下一步最值得做的是各写一个最小实验：

- 一个 `LDGSTS + double buffer` 的 prefetch kernel
- 一个 `TMA + barrier_expect_tx` 的 tile load/store kernel

如果要把这些知识迁移到 Triton，再加两个对应实验：

- 一个 Gluon `async_load + commit_group + wait_group` 的最小 kernel
- 一个 Gluon `mbarrier.expect + tma.async_load + fence_async_shared` 的最小 kernel

每个实验都用 profile 验证两件事：

- correctness 是否稳定
- copy 和 compute 是否真的 overlap

只有到这一步，这两节才算真正学进去。

## 14. 代码与参数速查（附录）

前面几节是概念提炼，这一节补上可直接参考的代码骨架和精确参数表，方便写 kernel 时对照。

### 12.1 三种异步拷贝机制对比

| 机制 | 架构 | 用途 | 方向 | 粒度 |
|------|------|------|------|------|
| LDGSTS | CC 8.0+ (Ampere) | 小的、逐元素拷贝 | global → shared | 4/8/16 B |
| TMA | CC 9.0+ (Hopper) | 大块、多维 tile 拷贝 | global ↔ shared / cluster | bulk，多维最多 5D |
| STAS | CC 9.0+ (Hopper) | 寄存器 → 分布式 shared | register → distributed shared | 4/8/16 B |

选型一句话：小而碎用 LDGSTS，大而规则用 TMA，cluster 内 block 间小数据用 STAS。所有这些拷贝的"完成信号"都靠 4.9 的异步屏障（transaction barrier + explicit phase tracking）。

<!-- APPEND-MARK -->

### 12.2 异步屏障：四个核心代码骨架

初始化（注意 bootstrap）：

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
auto block = cooperative_groups::this_thread_block();
if (block.thread_rank() == 0) {
    init(&bar, block.size());   // 第二个参数 = expected arrival count
}
block.sync();   // 用已有同步原语引导，否则其它线程看不到初始化
```

Split arrive/wait（在 arrive 和 wait 之间藏延迟）：

```cpp
auto token = bar.arrive();   // 到了但不阻塞
/* 这里做与同步无关的独立计算 */
bar.wait(std::move(token));  // 真正需要结果时才阻塞
```

显式 phase 跟踪（不用 token，跟踪 parity，性能更好）：

```cpp
int parity = 0;   // 偶数 phase=0，奇数=1；初始化后为 0
for (int i = 0; i < iteration_count; ++i) {
    cuda::ptx::mbarrier_arrive(handle);
    compute(data, i);
    while (!cuda::ptx::mbarrier_try_wait_parity(handle, parity)) {}
    parity ^= 1;
}
```

提前退出（少了 drop 会死锁）：

```cpp
if (condition_check()) {
    bar.arrive_and_drop();   // 完成本 phase 义务 + 下个 phase 不再计入 expected count
    return;
}
```

### 12.3 LDGSTS：三种 API 风格

```cpp
// 1. cuda::memcpy_async（配合 cuda::barrier）—— 可批量后单次 commit
cuda::memcpy_async(block, buffer, left,
                   cuda::aligned_size_t<4>(8 * sizeof(float)), barrier);
barrier.arrive_and_wait();

// 2. cooperative_groups::memcpy_async + cg::wait（最省事，但效率较低）
cg::memcpy_async(block, buffer, left, 8 * sizeof(float));
cg::wait(block);

// 3. 底层 primitives（最啰嗦但最可控，保证用 LDGSTS）
__pipeline_memcpy_async(buffer + tid, left + tid, sizeof(float));
__pipeline_commit();
__pipeline_wait_prior(0);
```

两个易漏点：
- `cuda::aligned_size_t<N>()` 是在告诉编译器对齐和大小是 N 的倍数，**这是启用 LDGSTS 的关键**，不加可能退化成同步拷贝。
- prefetch 流水线里，即使没有更多 batch 要取，也要继续 `producer_commit()` 保持槽位填满；这点 cooperative groups API 做不到（拿不到内部 pipeline 对象）。

### 12.4 TMA：单线程发起的 is_elected 惯用法

不要用 `if (threadIdx.x == 0)`——编译器无法确认只有一个线程，可能插 peeling loop 导致 warp 串行化：

```cpp
__device__ inline bool is_elected() {
    unsigned warp_id = threadIdx.x / 32;
    unsigned uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0);  // lane 0 广播
    return (uniform_warp_id == 0 && ptx::elect_sync(0xFFFFFFFF));    // warp 0 里选一个
}
```

也可用 `cooperative_groups::invoke_one`。

### 12.5 多维 TMA：host 造 tensor map + device 使用

host 端（注意：最快变化的维度在前，stride 单位是字节且为 16 的倍数）：

```cpp
constexpr uint32_t rank = 2;
uint64_t size[rank]     = {GMEM_WIDTH, GMEM_HEIGHT};
uint64_t stride[rank-1] = {GMEM_WIDTH * sizeof(int)};
uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};   // shared tile 大小
uint32_t elem_stride[rank] = {1, 1};
// cuTensorMapEncodeTiled(&tensor_map, ...)
```

device 端：

```cpp
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];  // 多维要 128B 对齐
    barrier::arrival_token token;
    if (is_elected()) {
        int32_t coords[2] = {x, y};
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
            &smem_buffer, &tensor_map, coords, barrier_native_handle(bar));
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    } else {
        token = bar.arrive();   // 其它线程只 arrive
    }
    bar.wait(std::move(token));
    /* 改数据 */
    ptx::fence_proxy_async(ptx::space_shared);  // shared 写对 TMA 引擎可见
    __syncthreads();
    if (is_elected()) {
        ptx::cp_async_bulk_tensor(ptx::space_global, ptx::space_shared, ...);
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());  // 等 TMA 读完 shared
    }
}
```

越界行为：读时越界部分自动零填充，左上角坐标可为负；写回时左上角不能为负。

transaction count 谁来设：`cuda::memcpy_async` 自动设；`cuda::device::memcpy_async_tx` 和 `cuda::ptx::cp_async_bulk` 不自动，要手动 `barrier_expect_tx` / `mbarrier_expect_tx`。

### 12.6 一维 TMA 对齐要求

| 地址 / 大小 | 对齐 |
|-------------|------|
| Global 地址 | 16 字节 |
| Shared 地址 | 16 字节 |
| Shared barrier 地址 | 8 字节（cuda::barrier 已保证） |
| 传输大小 | 16 字节的倍数 |

多维 bulk-tensor 额外要求：shared 地址 128 字节对齐；global 各维 size 只需 ≥1（不必是 16 的倍数）；global stride 必须是 16 字节的倍数。

### 12.7 Swizzle 模式参数表（CC 9.0）

| 模式 | swizzle 宽度 | inner dim 上限 | 重复周期 | shared 对齐 | global 对齐 |
|------|-------------|---------------|---------|------------|------------|
| 128B | 128 字节 | ≤128 | 1024 字节 | 128 字节 | 128 字节 |
| 64B | 64 字节 | ≤64 | 512 字节 | 128 字节 | 128 字节 |
| 32B | 32 字节 | ≤32 | 256 字节 | 128 字节 | 128 字节 |
| NONE | - | - | - | 16 字节 | - |

索引关系与 offset（以 128B 为例，其余把 8 换成 4/2）：

```cpp
int offset = (reinterpret_cast<uintptr_t>(smem_ptr) / 128) % 8;
// smem[y][x] <-> smem[y][((y + offset) % 8) ^ x]
```

那个异或就是 swizzle 核心：它打散了"列访问落在同一 bank"的情况。shared 缓冲最好按重复周期对齐（128B 模式即 1024 字节），否则要用 offset 公式校正。granularity 固定 16 字节。

### 12.8 STAS：cluster 内环形通信骨架

```cpp
__global__ __cluster_dims__(8, 1, 1) void producer_consumer_kernel() {
    auto cluster = this_cluster();
    __shared__ barrier_t filled, ready;
    if (threadIdx.x == 0) {
        init(&filled, 1);            // 单线程 arrive_expect_tx，故为 1
        init(&ready, BLOCK_SIZE);    // 全体 arrive，故为线程数
    }
    cluster.sync();   // 确保远程 barrier 都初始化完

    int rk = cluster.block_rank();
    auto buffer_next = cluster.map_shared_rank(buffer, (rk+1)%8);                     // 右邻居 buffer
    auto bar_next    = cluster.map_shared_rank(barrier_native_handle(filled), (rk+1)%8);

    int phase = 0;
    for (int it = 0; it < 1000; ++it) {
        st_async(&buffer_next[threadIdx.x], rk, bar_next);   // 写到右邻居
        if (threadIdx.x == 0)
            mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared,
                                      barrier_native_handle(filled), sizeof(buffer));
        while (!mbarrier_try_wait_parity(barrier_native_handle(filled), phase)) {}
        /* 消费数据，再通知左邻居 ready */
    }
}
```

最易错点：**barrier space 要选对**——映射来的远程 barrier 用 `space_cluster`，本地 barrier 用 `space_shared`，搞混会导致同步语义错误。

## 15. 在 Triton 里 4.9 / 4.11 是怎么体现的（附录）

本节内容已对照本机 Triton 源码核实：`/LocalRun/jiangzhe.zhao/my_repo/triton`，版本 **3.8.0**。API 和 IR op 名变动很快，换版本前请重新核对。

### 13.1 心智模型：CUDA 是手写，Triton 是编译器生成

最大的区别：

- CUDA C++ 里，4.9 / 4.11 是**用户手写的原语**——自己 `init/arrive/wait`、自己 `cp.async.bulk`、自己管 phase 和 transaction count。
- Triton 里，你只写 block 级的 `tl.load / tl.store / tl.dot` 和一个 `for` 循环，**异步屏障和异步拷贝由编译器 pass 自动插入**。

把这两节"实现"出来的核心是 Triton 的 **software pipeliner**（软件流水线 pass），目录：

- `lib/Dialect/TritonGPU/Transforms/Pipeliner/`
  - `SoftwarePipeliner.cpp`：pass 入口
  - `LowerLoops.cpp`：最关键，插入 async copy / barrier / 多缓冲
  - `AssignLatencies.cpp` / `ScheduleLoops.cpp`：排延迟、建调度
  - `WGMMAPipeline.cpp` / `MMAv5PipelineUtility.cpp`：Hopper WGMMA / Blackwell MMAv5 专用
  - `TMAStoresPipeline.cpp`：TMA store 流水线
- TMA load 的 lowering 另在 `lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp`

所以在 Triton 里学 4.9/4.11，不是查 API 文档，而是理解"编译器替你做了什么"。

### 13.2 4.9 异步屏障 → 编译器生成的 mbarrier

用户层几乎不暴露。`tl.debug_barrier()` 对应的是 `__syncthreads()`，**不是** 4.9 的 async barrier。真正的 4.9 在 NVIDIA 方言（前缀 `ttng`，即 `triton::nvidia_gpu`）里，由 pipeliner 和 TMA/wgmma lowering 生成。

| 4.9 概念 | Triton IR op（3.8.0 实测） |
|---|---|
| `init(&bar, count)` | `ttng.init_barrier` |
| `bar.arrive()` | `ttng.arrive_barrier`（带 `count` 属性 + 可选 predicate） |
| `barrier_expect_tx`（4.9.6 事务计数） | `ttng.barrier_expect`（signal 期待多少字节） |
| `try_wait_parity`（4.9.3 显式 phase） | `ttng.wait_barrier`，**带 `phase` 参数**，lowering 到 `mbarrier.try_wait.parity` |
| 失效屏障 | `ttng.inval_barrier` |

> ⚠️ 纠正一个常见误记：**没有单一的 `arrive_expect_tx` op**。CUDA C++ 里 `barrier_arrive_tx`（见 §2.9 / §12.5）那一个调用，在 Triton 里被拆成 `ttng.arrive_barrier` + `ttng.barrier_expect` 两个 op。

关键印证：`ttng.wait_barrier` 带 `phase` 参数、lowering 到 `try_wait.parity`，说明 **Triton pipeliner 在 Hopper 上默认就是用 4.9.3 显式 phase 跟踪 + 4.9.6 transaction barrier** 来同步 TMA 拷贝的——和本笔记 §12.2 那个 parity 骨架完全一致，因为那种写法对编译器最省（不用存 token 对象）。

### 13.3 4.11 异步拷贝 → 三条都有对应

**LDGSTS（4.11.1）→ `cp.async`，最成熟**。TritonGPU 方言（前缀 `ttg`，即 `triton::gpu`）：

| CUDA 概念 | Triton IR op |
|---|---|
| LDGSTS / `cp.async`（global→shared） | `ttg.async_copy_global_to_local` |
| `__pipeline_commit()` | `ttg.async_commit_group` |
| `__pipeline_wait_prior(n)` | `ttg.async_wait`（带 `num` 操作数） |

你写一个带 `tl.load` 的 `for` 循环，pipeliner 自动：开 `num_stages` 份 shared 缓冲、把 load 换成 async copy、插 commit/wait。**这就是 §4.3 / §12.3 的 prefetch 流水线被自动生成出来**。

**TMA（4.11.2）→ tensor descriptor**。IR op（`ttng`）：

| CUDA 概念 | Triton IR op |
|---|---|
| `cp_async_bulk_tensor`（global→shared） | `ttng.async_tma_copy_global_to_local` |
| TMA store（shared→global） | `ttng.async_tma_copy_local_to_global` |
| `cp_async_bulk_wait_group` | `ttng.async_tma_store_wait` |
| device 端建 tensor map（4.11.2.2.1） | `ttng.tensormap_create` |
| tensor map proxy acquire fence（4.11.2.2.3） | `ttng.tensormap_fenceproxy_acquire` |

另有 `ttng.async_tma_reduce / async_tma_gather / async_tma_scatter`。

用户侧 API（`python/triton/language/core.py`）：

- 创建：`tl.make_tensor_descriptor(...)`（`core.py:2646`），返回 `tensor_descriptor`。
- 读写：`tl.load_tensor_descriptor` / `tl.store_tensor_descriptor`，也可作为 descriptor 对象的方法。
- > ⚠️ **3.8.0 已删除** 旧的 `_experimental_descriptor_load/store` 和 `_experimental_make_tensor_descriptor`；`tl.make_block_ptr` 也已废弃，转向 tensor descriptor。我之前提到的 experimental API 在这个版本不存在了。
- device 端建描述符存在（`make_tensor_descriptor` 在 kernel 内调用即映射到 `ttng.tensormap_create`），对应 4.11.2.2.1 的 on-device encoding。
- **swizzle（4.11.2.2.5）**：在 descriptor 创建时选 swizzle mode，Triton 用它消 `tl.dot` 操作数的 shared bank conflict。

**STAS（4.11.3）→ 最边缘**。寄存器→远程 distributed shared，Triton 通过 cluster（`num_ctas > 1`）间接涉及，普通 kernel 基本碰不到，主要在 Hopper cluster 高级模板里。3.8.0 无稳定用户 API。

### 13.4 你实际能调的旋钮

Triton 把整个 4.9+4.11 体系压缩成几个参数：

- **`num_stages`** —— 最重要。决定软件流水线深度 = 4.11 prefetch 级数 + 4.9 多缓冲屏障个数。调它就是调"copy 和 compute 重叠多少"。出处：autotune `Config`（`python/triton/runtime/autotuner.py:351`），经 `jit.py:663` 进 compile options 驱动 pipeliner；也可按循环覆盖 `tl.range(..., num_stages=...)`（`core.py:3722`）。
- **tensor descriptor API** —— 决定走 TMA 还是普通 cp.async。
- **`num_warps` / `num_ctas`** —— 后者开 cluster，才会触及 STAS / distributed shared 路径。
- **warp specialization（对应 4.9.7 / 4.11.1.3）** —— 3.8.0 里**不是** `tl.async_task`，而是循环上的布尔开关 `tl.range(..., warp_specialize=True)`（`core.py:3723`）。编译器侧是 pass `TritonGPUAutomaticWarpSpecialization`（`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`）+ op `ttg.warp_specialize`。即自动驱动，用户只给一个 flag，没有显式 task 划分 API。

一句话：**CUDA 里你手动搭的"async copy + transaction barrier + 多缓冲流水线"，在 Triton 里就是 pipeliner 读着你的 `num_stages` 自动生成的那套 TTGIR。**

### 13.5 想自己验证

对着真实 kernel 看生成的 IR 最扎实：

- `TRITON_KERNEL_DUMP=1` dump 各阶段 IR，在 TTGIR 里搜 `async_copy_global_to_local` / `init_barrier` / `async_tma_copy` / `wait_barrier`，能直接看到 pipeliner 插进去的 4.9/4.11 实现。
- 改 `num_stages`（如 2→4）重新 dump，对比多缓冲和 commit/wait 的变化。
