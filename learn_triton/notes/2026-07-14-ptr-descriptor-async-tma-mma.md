# 2026-07-14 学习笔记：从一块矩阵数据理解 Ptr、Descriptor、TMA 与 MMA

## 0. 这篇应该怎么读

第一次阅读时，只读第 1～7 节。它们只回答一件事：

```text
计算一块矩阵时，数据从显存出发，经过哪里，最后怎样进入矩阵计算单元？
```

先不要记 TTIR、TTGIR、memdesc、tensormap 等名字。等主数据流已经能在脑中画出来，
再读后半部分的 Triton 定义和编译器 lowering。

全文最重要的结论是：

```text
ptr 和 descriptor 都不是矩阵数据。

ptr 告诉硬件“每个元素的地址在哪里”。
descriptor 告诉硬件“整个 tensor 怎么排布，我现在要其中哪一块”。

TMA 负责搬数据。
MMA/TU 负责算数据。
```

## 1. 从一个具体问题开始

假设要计算：

```text
C[0:128, 0:256] += A[0:128, 0:64] @ B[0:64, 0:256]
```

A、B、C 都可能很大，kernel 每次只处理其中一个小矩形。这个小矩形称为 **tile**。例如这里的
`A[0:128, 0:64]` 就是一个 A tile。

A、B 一开始在显存（global memory，后文简称 GMEM）里。矩阵计算单元不能高效地直接逐元素
读取远处的 GMEM，所以通常要先把 A、B 的 tile 搬到片上存储，再执行矩阵乘法。

这里还会反复用到两个词：

- **shared memory / tile buffer**：芯片上离计算单元更近的一块临时存储，用来暂存 A、B tile。
- **accumulator**：保存矩阵乘加中间结果的地方。沿 K 维每算一块，结果就继续累加到这里。

这里一共要完成四类工作：

1. 确定 A、B tile 在 GMEM 中的位置。
2. 把 tile 从 GMEM 搬到片上 buffer。
3. 等待搬运完成，避免读到未完成的数据。
4. 让 MMA/TU 读取 tile 并更新 accumulator。

ptr 和 descriptor 的差别，首先发生在第 1、2 步：**怎样描述要访问的数据，以及由谁完成地址
生成和搬运。**

先记住下面这张图，后面所有细节都只是把它展开：

```text
ptr 路径
========
线程算出每个元素地址
          |
          v
GMEM 中的 A/B --普通 load--> 各线程寄存器 --必要时再写 shared--> TU


descriptor 路径
===============
descriptor + tile 坐标 --告诉 TMA 搬哪一块--+
                                                |
                                                v
GMEM 中的 A/B -----------------------------TMA--> shared tile --> TU
```

上半条路径里，普通线程自己展开地址并参与搬运。下半条路径里，线程只提交描述和坐标，TMA
负责大块搬运。两条路径搬的都是同一份 A/B 数据。

## 2. Ptr 路径：把每个元素的地址算出来

### 2.1 一个 ptr 表示什么

一个普通 ptr 可以先理解为：

```text
一个内存地址 + 这个地址上的元素类型
```

例如 `a_ptr` 指向 A 的第一个 `fp16` 元素。

如果只执行：

```python
x = tl.load(a_ptr)
```

就是从这个地址读取一个元素。

### 2.2 读取一个 tile 时会发生什么

要读取 `A[0:128, 0:64]`，ptr 路径需要为 tile 中的每个逻辑元素计算地址：

```python
offs_m = tl.arange(0, BLOCK_M)
offs_k = tl.arange(0, BLOCK_K)

a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
a = tl.load(a_ptrs)
```

这里的 `a_ptrs` 是 pointer tensor。它可以想成一张地址表：

```text
逻辑坐标        对应地址
A[0, 0]   ->    a_ptr + 0 * stride_am + 0 * stride_ak
A[0, 1]   ->    a_ptr + 0 * stride_am + 1 * stride_ak
A[1, 0]   ->    a_ptr + 1 * stride_am + 0 * stride_ak
...
```

注意：pointer tensor 里装的是**地址**，不是 A 的数据。执行 `tl.load` 后，才得到 A 的数据。

如果 tile 可能越界，还需要为每个元素计算 mask。mask 是一组真假值，用来表示对应地址是否
有效：

```python
mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
a = tl.load(a_ptrs, mask=mask, other=0.0)
```

因此 ptr 路径的基本模型是：

```text
线程计算每个元素的 offset
  -> 形成每个元素的 ptr
  -> 计算每个元素的 mask
  -> 发出普通 load
  -> 数据通常成为各线程寄存器中持有的一小片（fragment）
```

### 2.3 为什么这种方式仍然很有用

它最大的价值是灵活。

例如索引来自数据本身：

```python
indices = tl.load(index_ptr + offs)
x = tl.load(x_ptr + indices)
```

每个元素可能访问完全不同的位置。这种访问没有一个简单的矩形 tile 可以描述，ptr 路径反而
最自然。

ptr 适合：

- 不规则 gather/scatter，也就是每个 thread 读取或写入不同的非连续位置
- 每个元素有不同 mask 或地址
- 逐元素计算、归约计算，以及 shape/index 等少量控制数据访问
- 小块数据或标量数据
- descriptor/TMA 无法表达的 layout
- 没有 TMA 类搬运硬件的目标芯片

### 2.4 ptr 路径的可能成本

对于规则的大矩阵 tile，它可能做了很多重复工作：

- 每个 thread 都参与地址计算
- 需要保存 pointer tensor 和 mask tensor
- 数据可能先进入通用寄存器，再写入 shared memory
- 地址和搬运指令会占用普通执行流水线

这不表示 ptr 一定慢。连续访问仍能被合并成连续或向量访存，编译器也可能把合适的普通 load
改写成异步 global-to-shared copy。这里只是说：**ptr 的原始语义是逐元素地址访问。**

## 3. Descriptor 路径：描述整个 tensor，只提交 tile 坐标

### 3.1 为什么需要 descriptor

对于 A 这种规则二维矩阵，我们已经知道：

```text
起始地址 = a_ptr
完整形状 = [M, K]
每维步长 = [stride_am, stride_ak]
元素类型 = fp16
每次搬运的 tile = [BLOCK_M, BLOCK_K]
```

如果把这些信息提前打包，读取下一个 tile 时就不必再次构造整张 element address 表，只需要说：

```text
请读取从 [offs_m, offs_k] 开始的 tile。
```

这个打包后的访问说明就是 tensor descriptor。

### 3.2 descriptor 中有什么

可以先把 descriptor 理解成下面这张表：

```text
base address       A 从哪里开始
global shape       A 一共有多大
global strides     A 每一维怎样排布
element type       元素是 fp16、bf16 还是其他类型
block shape        一次搬多大的 tile
padding rule       越界位置填 0、NaN，还是忽略
layout/swizzle     搬到片上后怎样摆放或重排，以匹配访存和矩阵指令
```

descriptor 仍然不是数据。它只是硬件搬运引擎读取的元数据。

### 3.3 在 Triton 中怎样使用

```python
a_desc = tl.make_tensor_descriptor(
    a_ptr,
    shape=[M, K],
    strides=[stride_am, stride_ak],
    block_shape=[BLOCK_M, BLOCK_K],
)

a = a_desc.load([offs_m, offs_k])
```

调用 `load` 时只提交 tile 的起始坐标 `[offs_m, offs_k]`。

对比两种路径：

```text
ptr:
  base + 整个 tile 的 element offsets + 整个 tile 的 masks

descriptor:
  tensor 的固定描述 + 当前 tile coordinate
```

### 3.4 descriptor 解决的根本问题

descriptor 建立了一个规则 tile 的访问契约，使专用搬运引擎可以接管：

- 多维地址生成
- 大块搬运
- 部分越界处理
- 搬入 shared memory 时的布局变换/swizzle

这正是 TMA 能工作的前提。

### 3.5 descriptor 的限制

descriptor 能减少工作，是因为它假设访问足够规则。常见约束包括：

- base 满足对齐要求
- 最内维连续
- leading stride 满足硬件对齐要求
- rank、block shape、element type 在硬件支持范围内
- shared layout 能被 TMA 编码

如果每个元素地址都不一样，descriptor 并不能凭空把不规则访问变规则。这时仍应使用 ptr，
或者使用硬件明确支持的 descriptor gather/scatter 模式。

## 4. TMA：拿着 descriptor 搬运 tile

TMA 是 Tensor Memory Accelerator。这里先把它理解为独立于普通标量计算流水线的搬运引擎。

发出一次 TMA load，大致需要告诉它：

```text
1. descriptor 在哪里
2. 这次 tile 的坐标是什么
3. 搬到哪一块 shared/local buffer
4. 完成后通知哪个 barrier
```

然后数据实际这样移动：

```text
                descriptor + coordinate
                         |
                         v
GMEM data ------------- TMA --------------> shared tile buffer
```

需要注意两件事：

1. descriptor 只是 TMA 的输入参数，数据不在 descriptor 里面。
2. TMA 的典型职责是 `GMEM <-> shared/local buffer`，它不负责矩阵乘法。

所以 `a_desc.load(...)` 在 Python 中虽然返回一个 `tl.tensor`，支持 TMA 的 lowering 并不要求
先把整个 tile 放到通用寄存器。它可以让 TMA 直接写 shared buffer。

## 5. MMA/TU：从片上 buffer 读取 tile 并计算

MMA 是 matrix multiply-accumulate engine。operand 就是矩阵指令的输入数据。若 TU 是自研
芯片的矩阵计算单元，那么在这篇的心智模型中，TU 就处在 MMA 的位置。

当 A、B 已经由 TMA 搬到 shared buffer 后，矩阵单元需要知道：

```text
A tile 在 shared/TMEM 的哪里、怎样排布
B tile 在 shared/TMEM 的哪里、怎样排布
accumulator 在哪里
矩阵指令的 M/N/K、数据类型等参数
```

这里又会出现一种 descriptor：MMA/TU operand descriptor。

它与前面的 TMA tensor descriptor 不是同一个东西：

```text
TMA descriptor:
  描述 GMEM tensor，供搬运引擎寻找 global tile。

MMA/TU operand descriptor:
  描述片上 tile，供矩阵单元读取 shared/TMEM operand。
```

完整路径是：

```text
           TMA tensor descriptor
                    |
GMEM --TMA--------> shared tile buffer
                            |
                    TU operand descriptor
                            |
                            v
                           TU
                            |
                       accumulator
```

不能把它理解成“TMA descriptor 从 TMA 传给 TU”。真正连接两个 engine 的是 shared tile
buffer，以及“这块 buffer 是否已经可读”的同步状态。

## 6. 为什么要做全异步 TMA + MMA/TU

### 6.1 同步执行的问题

最简单的 K-loop 是：

```text
搬 tile 0 -> 等待 -> 计算 tile 0
搬 tile 1 -> 等待 -> 计算 tile 1
搬 tile 2 -> 等待 -> 计算 tile 2
```

搬运时，TU 可能空闲；TU 计算时，TMA 又可能空闲。总时间接近：

```text
所有搬运时间 + 所有计算时间
```

### 6.2 异步流水想得到什么

如果准备多块片上 buffer，就可以形成流水：

```text
时间段       TMA                    TU
------------------------------------------------
0            搬 tile 0              空闲
1            搬 tile 1              算 tile 0
2            搬 tile 2              算 tile 1
3            搬 tile 3              算 tile 2
```

进入稳态后，TMA 搬下一块时，TU 正在算当前块。理想总时间更接近：

```text
max(总搬运时间, 总计算时间) + 流水启动/排空成本
```

这就是全异步设计的首要目的：

> 让 transport engine、matrix engine、store engine 和标量控制流水尽可能同时工作，隐藏延迟，
> 缩短关键路径。

### 6.3 为什么需要多个 buffer

假设只有一块 shared buffer：

```text
TMA 正在写下一块数据
TU 同时还在读上一块数据
```

两者会互相覆盖，结果错误。

因此通常使用 ping-pong 或更多 stage：

```text
buffer 0: TU 正在消费
buffer 1: TMA 正在生产
buffer 2: 空闲或等待复用
```

### 6.4 为什么需要 barrier、phase 和 wait

异步 engine 发出指令后，普通程序不能假设它已经完成。至少要保证：

1. TU 读 buffer 前，TMA 已经写完。
2. TMA 覆盖 buffer 前，TU 已经读完。
3. buffer 循环复用时，不能把上一轮的完成信号误认为这一轮完成。
4. 普通指令读取 accumulator 前，异步 TU/MMA 已经完成更新。

barrier 记录“工作是否完成”；phase 用来区分同一个 barrier 的不同轮次；wait 表示消费者在真正
需要结果的位置等待。

这不是额外装饰，而是异步执行仍然保持正确性的基本条件。

### 6.5 异步指令不等于异步流水

下面这种代码虽然使用异步指令，效果仍接近同步：

```text
issue async TMA
立刻 wait TMA
issue async MMA
立刻 wait MMA
```

只有当 wait 之前存在其他独立工作，并且使用 multibuffer 允许多个 iteration 同时在途，才有
真正的 overlap。

因此要区分：

```text
async instruction：指令可以先发射、以后等待。

async pipeline：等待被推迟，同时有另一批有用工作在执行。
```

## 7. 对最初判断的准确表述

原来的理解可以整理成：

> 对规则矩阵计算，主数据面由 global tensor descriptor 驱动 TMA，把 tile 从 GMEM 搬到片上
> buffer；TU 再通过自己的 operand descriptor 直接读取 tile，并尽可能把 accumulator 留在
> TU 专用存储。通用寄存器主要负责 tile 坐标、descriptor handle、TMA/TU 指令参数、buffer
> index、barrier phase、循环和普通控制计算。

这个判断的方向是对的，但要加三个条件：

1. TMA descriptor 与 TU operand descriptor 是两种 descriptor。
2. 是否完全不回 register tensor，取决于 TU 的 operand 和 accumulator ISA（指令集）约定。
3. 全异步的首要目的不是消灭寄存器，而是让搬运、计算和写回真正重叠。

如果 TU 本身就是矩阵计算单元，那么简化后的主数据流是：

```text
GMEM --TMA--> on-chip tile buffer --TU--> TU accumulator storage
```

而不是：

```text
data -> descriptor -> TU
```

descriptor 始终只是描述和控制信息，不是数据经过的一层存储。

---

下面进入 Triton 和 NVIDIA 后端的具体实现。第一次阅读可以先停在这里。

后文中的 **lowering**，就是编译器把高层写法逐步改写成更接近硬件的 IR 和最终指令。

## 8. `tl.tensor` 为什么不能直接等同于 register tensor

在 Triton Python 中：

```python
a = a_desc.load([offs_m, offs_k])
acc = tl.dot(a, b, acc)
```

`a` 的类型是 `tl.tensor`。但它首先表示“一个逻辑矩阵值”，不是在声明物理存储位置。

这里说的 **register tensor** 不是一块独立的硬件内存，而是“一个逻辑 tensor 被切分后，
每个 thread 分别用若干通用寄存器保存自己那一小片”的简称。

后续编译器可能选择：

```text
情况 A：distributed tensor
  每个 thread 持有若干元素
  -> 通常落到通用寄存器 fragment

情况 B：shared memdesc
  SSA value 表示一块 shared-memory object
  -> MMA 通过地址/descriptor 直接读取

情况 C：TMEM memdesc
  SSA value 表示一块 tensor-memory object
  -> Blackwell TCGen5 直接更新或读取
```

因此判断是否回到寄存器，不能只看 Python 变量，也不能只看 TTIR 中是否出现
`tt.descriptor_load`。需要继续看 TTGIR 的 encoding 和最终 ISA operand。

### 8.1 什么情况下可以保持在 shared

编译器看到类似 use-def：

```text
descriptor_load
  -> compatible shared allocation
  -> MMA consumer
```

如果 load 只有这个用途，布局又与 MMA 兼容，就可以让 TMA 写 shared、MMA 直接读 shared，
不必把 A/B tile 物化成 register tensor。

### 8.2 什么情况下仍要进入寄存器

- load 结果还要做普通 elementwise 运算
- load 有多个不兼容的用户
- shared layout 与 MMA operand layout 不兼容
- mask/`other` 语义不能由异步路径表达
- ISA 要求某个 operand 位于 registers
- accumulator ISA 本身使用 register accumulator
- epilogue 需要线程逐元素读取和变换结果

所以“数据不回寄存器”是 compiler placement 的结果，不是 descriptor API 自动保证的属性。

## 9. Triton 中 ptr 和 descriptor 的正式定义

### 9.1 Scalar pointer 与 pointer tensor

Triton IR 的 pointer type 是：

```text
!tt.ptr<T, address-space>
```

当前 pointer 只能指向 scalar element type。

一个 pointer tensor 是：

```text
tensor<BLOCK_M x BLOCK_K x !tt.ptr<T>>
```

TTIR 中常见路径为：

```text
tt.make_range / broadcast / arith
  -> tt.addptr
  -> tt.load(pointer tensor, mask, other)
```

其 invariant 是：每个逻辑元素的最终地址和有效性必须能够显式计算。

### 9.2 Block pointer

`tl.make_block_ptr` 把下面的信息包装为一个 block-level Python 对象：

```text
base + parent shape + strides + current offsets + block shape + order
```

它简化了 block boundary check，但不等于硬件 TMA descriptor。在当前仓库中它已经 deprecated，
推荐使用 tensor descriptor。

### 9.3 Tensor descriptor

`!tt.tensordesc` 是 Triton 的可移植 tile-access abstraction。

`tt.make_tensor_descriptor` 接收：

```text
base + global shape + global strides + block shape + padding
```

访问由：

```text
tt.descriptor_load
tt.descriptor_store
```

表达。

它的语义是“按 descriptor 和 tile coordinate 访问一个完整 tile”，但没有规定一定使用 NVIDIA
TMA。支持 TMA 的 backend 可以将它落到 hardware tensormap；不支持的 backend 可以将它展开
回 pointer tensor、mask 和普通 load/store。

这说明：

```text
descriptor 是 Triton 语义。
TMA 是一种 target-specific 实现。
```

## 10. Triton 的 descriptor load 怎样变成 TMA

NVIDIA TMA lowering 会把一次抽象 descriptor load 展开成：

```text
1. 分配 shared tile buffer
2. 分配并初始化 mbarrier
3. 告诉 barrier 预期完成多少字节
4. 发出 async TMA global -> shared
5. 在第一个真实 consumer 前等待 barrier
6. 让 consumer 使用 shared object
```

循环 pipeline 会进一步把 shared allocation 扩成多个 buffer，并维护：

```text
insert index     TMA 当前写哪个 buffer
extract index    MMA 当前读哪个 buffer
phase            当前是 barrier 的哪一轮
```

设备端创建的 NVIDIA tensormap 当前占 128 bytes，并要求 128-byte alignment。它包含 global
address、dimensions、strides、box dimensions、element type、swizzle 和 fill mode 等字段。

最终 TMA 指令消费的是：

```text
tensormap address + tile coordinates + shared destination + mbarrier
```

而不是一整个 element pointer tensor。

## 11. Host-side 与 device-side descriptor

### 11.1 Host-side

```python
from triton.tools.tensor_descriptor import TensorDescriptor

a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K])
kernel[grid](a_desc, ...)
```

host 在 launch 前根据 tensor metadata 构造 descriptor，再作为 kernel 参数传入。它适合 shape、
stride 和 block shape 在 launch 时已经知道的场景，可以避免 kernel 内 descriptor setup。

### 11.2 Device-side

```python
a_desc = tl.make_tensor_descriptor(
    a_ptr,
    shape=[M, K],
    strides=[stride_am, stride_ak],
    block_shape=[BLOCK_M, BLOCK_K],
)
```

kernel IR 中创建 descriptor，backend 再物化硬件 tensormap 和所需 fence。它适合 descriptor
字段依赖 runtime/device 逻辑的场景，但要承担设备端构造、存储和可见性成本。

两者的区别是 descriptor 在哪里、何时创建，不是 tile 数据放在哪里。

## 12. 什么时候选 ptr，什么时候选 descriptor

| 问题 | 更适合 ptr | 更适合 descriptor |
| --- | --- | --- |
| 地址是否规则 | 每元素不同、data-dependent | 规则多维 tensor |
| 访问粒度 | scalar/vector/small block | 较大的矩形 tile |
| mask | 每元素任意 mask | 规则边界/padding |
| 后续计算 | 普通 elementwise/reduction | MMA/TU 直接消费 |
| 地址计算 | 希望每 lane 自由计算 | 希望 transfer engine 接管 |
| pipeline | 不一定复用 tile stage | 适合 multibuffer TMA pipeline |
| 可移植 fallback | 原生通用路径 | backend 可展开回 ptr |

可以记成：

```text
访问越不规则，越偏向 ptr。
访问越规则、tile 越大、越直接进入矩阵单元，越偏向 descriptor。
```

descriptor 不是天然更快。如果 tile 很小、创建 descriptor 成本很高，或者 load 后马上要做大量
普通 register 运算，它可能没有收益。

## 13. Hopper 与 Blackwell 的关键差别

### 13.1 Hopper

典型路径是：

```text
GMEM --TMA--> shared A/B --WGMMA--> register accumulator
```

WGMMA 的 B operand 来自 shared；A 根据指令形式可以来自 shared 或 registers。accumulator 主要
仍由 warp-group threads 的寄存器 fragment 表示。

所以 Hopper 可以让 A/B 主数据绕过通用寄存器，但不能笼统说整个计算状态都不回寄存器。

### 13.2 Blackwell

典型路径可以进一步变成：

```text
GMEM --TMA--> shared A/B --TCGen5--> TMEM accumulator
```

部分 A operand 也可以来自 TMEM。只有 epilogue 或普通线程计算真正需要结果时，才通过
`tcgen05.ld` 把 TMEM 内容读到通用寄存器。

这比 Hopper 更接近“主矩阵数据和 accumulator 都不回 register tensor”。

映射到自研 TU 时，最需要先确认的是：

```text
TU 的 A/B 从哪里读？
TU 的 accumulator 存在哪里？
TU 完成状态怎样通知软件？
```

如果这三个 ISA contract 不清楚，就不能判断“全程不回寄存器”是否成立。

## 14. 自研 TMA + TU 需要定义哪些 contract

要让编译器稳定生成全异步路径，至少要明确：

1. **Global descriptor**：支持哪些 rank、stride、tile、padding 和 swizzle。
2. **Tile buffer**：TMA 写入哪种片上 memory，怎样布局和 multibuffer。
3. **TU operand**：TU 能否直接读取 tile buffer，operand descriptor 怎样编码。
4. **Accumulator**：位于 registers、专用 memory，还是 TU 内部状态。
5. **Completion**：TMA/TU 如何 arrive、commit、wait。
6. **Buffer reuse**：producer 何时可以覆盖，consumer 何时释放。
7. **Visibility**：普通 store、TMA、TU 等不同 engine 之间需要什么 fence。
8. **Compiler placement**：哪些 tensor 保持为 memory descriptor，哪些必须 materialize 到
   register tensor。

这些 contract 决定了编译器能否把：

```text
descriptor_load -> tensor -> dot
```

稳定转换成：

```text
TMA writes tile buffer
  -> barrier establishes readiness
  -> TU reads tile buffer directly
  -> TU updates dedicated accumulator
```

## 15. 如何验证数据是否真的绕过寄存器

### 15.1 看 TTIR：确认访问语义

```text
ptr path:
  tt.addptr -> tt.load

descriptor path:
  tt.make_tensor_descriptor -> tt.descriptor_load
```

这一层只能说明程序如何描述访问，不能证明物理 placement。

### 15.2 看 TTGIR：确认存储位置

检查：

- 是否出现 shared `!ttg.memdesc` / `ttg.local_alloc`
- descriptor load 是否直接连接 compatible shared allocation
- MMA operand 是 distributed tensor，还是 shared/TMEM memdesc
- accumulator 是 distributed MMA encoding，还是 TensorMemory encoding

### 15.3 看 LLVM/PTX/ISA：确认最终 operand

检查：

- 是否出现 TMA/`cp.async.bulk.tensor`
- MMA/TU operand 是 shared descriptor、TMEM descriptor 还是 registers
- accumulator 是否产生大量 register outputs
- 何处出现普通 shared load 或 TMEM-to-register load
- wait/commit/barrier 是否允许多个 iteration 同时在途

### 15.4 看 profiler：确认真正 overlap

检查：

- TMA 与 MMA/TU active cycles 是否重叠
- matrix engine utilization
- register count 和 occupancy
- shared/TMEM bandwidth 与 bank conflict
- barrier stall 和 scoreboard stall
- K-loop 是否足够长，能进入 pipeline steady state

只有 IR、最终 ISA 和 profiler 三者一致，才能证明“没有回 register tensor”和“全异步真的
发生了”，不能只看 API 名称。

## 16. 源码索引

这部分用于建立心智模型之后回到代码验证，不建议第一次阅读时逐个点开。

### Ptr 与 descriptor 定义

- Pointer type：
  [TritonTypes.td](../../include/triton/Dialect/Triton/IR/TritonTypes.td#L53)
- `tt.addptr` / `tt.load`：
  [TritonOps.td](../../include/triton/Dialect/Triton/IR/TritonOps.td#L196)
- Tensor descriptor type：
  [TritonTypes.td](../../include/triton/Dialect/Triton/IR/TritonTypes.td#L101)
- `tt.make_tensor_descriptor`：
  [TritonOps.td](../../include/triton/Dialect/Triton/IR/TritonOps.td#L983)
- `tt.descriptor_load`：
  [TritonOps.td](../../include/triton/Dialect/Triton/IR/TritonOps.td#L1226)
- Deprecated block pointer API：
  [core.py](../../python/triton/language/core.py#L2613)

### Descriptor 的两种 lowering

- Descriptor -> ptr/mask fallback pass：
  [Passes.td](../../include/triton/Dialect/Triton/Transforms/Passes.td#L38)
- Descriptor load 展开为 pointer tensor：
  [RewriteTensorDescriptorToPointer.cpp](../../lib/Dialect/Triton/Transforms/RewriteTensorDescriptorToPointer.cpp#L324)
- NVIDIA TMA lowering：
  [TMALowering.cpp](../../lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp#L26)
- Device-side tensormap creation：
  [TMAUtilities.cpp](../../lib/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.cpp#L138)
- 128-byte tensormap object：
  [TMAUtilities.h](../../include/triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h#L11)
- TMA PTX emission：
  [LoadStoreOpToLLVM.cpp](../../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1271)

### Pipeline 与 MMA placement

- 判断 load 是否必须进入 registers：
  [LowerLoops.cpp](../../lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp#L51)
- TMA multibuffer lowering：
  [LowerLoops.cpp](../../lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp#L232)
- WGMMA 是否能真正异步：
  [WGMMAPipeline.cpp](../../lib/Dialect/TritonGPU/Transforms/Pipeliner/WGMMAPipeline.cpp#L392)
- Hopper WGMMA register/shared operand 与 accumulator：
  [WGMMA.cpp](../../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/WGMMA.cpp#L76)
- Blackwell shared/TMEM operand 与 TMEM accumulator：
  [MMAv5.cpp](../../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv5.cpp#L438)
- TMEM 读取到 registers：
  [TensorMemoryToLLVM.cpp](../../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TensorMemoryToLLVM.cpp#L520)

## 17. 相关笔记

- [TMA 相关 pass](2026-07-01-tma-pass-learning.md)
- [TMEM 相关 pass](2026-07-01-tmem-pass-learning.md)
- [Pipeline](2026-06-29-pipeline.md)
- [AccelerateMatmul](2026-06-29-accelerate-matmul.md)
- [WarpSpecialize](2026-06-30-warp-specialize.md)
- [Barrier 与 fence](2026-07-02-barriers-and-fences.md)
