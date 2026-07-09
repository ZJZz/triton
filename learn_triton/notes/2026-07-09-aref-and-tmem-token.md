# ARef 和 TMEM Token

这篇专门把两个容易混在一起的概念拆开：

- `ARef`
- `TMEM token` / `AsyncToken`

它们都**不是新的硬件同步原语**，但都和异步执行、warp specialization、TMEM 路径的正确性有关。

最短结论先放前面：

```text
ARef = 高层“异步引用 / 交接协议”抽象
TMEM token = 编译器可见的依赖边
```

所以：

- `ARef` 主要回答：**值 / buffer / ownership 要怎样在不同 partition 之间交接**
- `TMEM token` 主要回答：**编译器怎样显式保住 TMEM / accumulator 的依赖，不让它被错误重排**

它们都不是：

- `bar.sync`
- `fence`
- `mbarrier`
- `wait_group`

---

## 1. 为什么这两个东西值得单独写

它们很容易被误读成“又来了一套同步原语”，但其实不是。

真正的硬件同步原语，回答的是这几类问题：

- 线程是否到齐
- 异步任务是否完成
- 某个对象是否对后续消费者可见

而 `ARef` / `TMEM token` 更像是**编译器为了把这些关系建模清楚而引入的中间抽象**：

- `ARef` 让跨 partition / 跨阶段交接先在 IR 里变成显式对象
- `TMEM token` 让 TMEM 路径上的依赖先在 IR 里变成显式边

也就是说，它们首先服务的是：

```text
正确建模
-> 正确 lowering
-> 再落成 barrier / wait / mbarrier / repair pass
```

### 1.1 先把两层边界拆开：`partition` 不是 `warp/CTA/cluster`

这里最容易混的是把 `ARef` 和 `barrier / mbarrier / fence` 都当成“某种同步工具”，然后再把
它们粗略分到不同层级里。更稳的读法是先分清：

```text
compiler abstraction layer
vs
hardware synchronization layer
```

`ARef` 主要处在前者。

它描述的是：

- 不同 `partition` 之间怎样交接 value / buffer / ownership
- 这个交接是否跨 `stage` / `phase`
- 这个交接之后要不要在 lowering 时补 barrier 资源

这里的 `partition` 是 **compiler 组织单位**，不是硬件执行 scope。它来自 warp specialization /
pipeline 变换，用来表达“哪一类 actor 负责哪一段工作”。

而 `barrier` / `mbarrier` / `fence` 主要处在后者。

它们描述的是硬件上的同步关系：

- `barrier`：谁需要 execution rendezvous
- `mbarrier`：谁在观察 async completion
- `fence`：谁在发布 ordering / visibility

这些原语才会真正落到硬件 scope 上，例如 `warp`、`CTA`、`cluster`，或者具体的 proxy /
memory-visibility 边界。

所以更准确的分工不是：

```text
ARef 服务 partition 层
barrier / mbarrier / fence 服务 warp / CTA / cluster 层
```

而是：

```text
ARef = compiler 层，先描述 partition 之间如何交接
barrier / mbarrier / fence = hardware/runtime 层，落实 execution / completion / visibility 约束
```

一句话记忆就是：

```text
ARef 先回答“谁和谁交接”
barrier / mbarrier / fence 再回答“怎样在硬件上把这个交接做正确”
```

### 1.2 再说硬一点：`ARef` / `TMEM token` 是 IR 抽象，不是硬件动作

如果只想抓一个最短判断，可以直接用下面这张表：

| 东西 | 主要停留在哪一层 | 是否有独立硬件执行实体 | 是否直接代表一次硬件同步动作 |
|---|---|---|---|
| `ARef` | NVWS / IR 建模层 | 没有 | 不直接代表 |
| `TMEM token` | TTGIR / 依赖建模层 | 没有 | 不直接代表 |
| `barrier` | hardware/runtime sync layer | 有对应同步语义 | 是 |
| `mbarrier` | hardware/runtime sync layer | 有，对象驻留于 shared memory | 是 |
| `fence` | hardware/runtime sync layer | 有对应 ordering/visibility 语义 | 是 |
| `wait_group` | hardware/runtime sync layer | 有对应 outstanding-group wait 语义 | 是 |

所以不要把下面两类东西混成一类：

```text
ARef / token
= 先把依赖、交接、ownership 关系在 IR 里描述清楚

barrier / mbarrier / fence / wait_group
= 真正让硬件去执行同步、等待、发布可见性
```

但也不要把前者理解成“纯注释”。更准确的说法是：

```text
ARef / token 不直接执行硬件同步
但它们会决定 lowering 之后需要生成哪些硬件同步机制
```

---

## 2. ARef 是什么

先看一手定义。

`NVWS_ArefType` 的描述是：

- `Asynchronous Reference`
- 一个持有底层类型异步引用的 meta-type
- 可以同时包住多个底层值
- 用于 pipelining / warp specialization 这类变换
- lowering 时再插入合适 barrier

见 [NVWSTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSTypes.td:34)。

这句话其实已经把边界说清楚了：

```text
ARef 不是硬件对象
ARef 是“异步交接关系”的高层承载体
```

### 2.1 它解决什么问题

最典型的场景是 warp specialization。

当 loop 被 split 成 producer / consumer partitions 之后，原来“一个 SSA 值直接流到下游”的图，
往往不再合法，因为：

- 生产者和消费者不在同一个 partition
- 中间可能隔着 stage / phase
- buffer 可能是多缓冲
- ownership 可能会转移

这时编译器不能只保留“值依赖”这个抽象，而要把“谁在什么时候把哪个 buffer 交给谁”显式化。

`ARef` 就是拿来承载这件事的。

可以把它理解成：

```text
先把“未来要靠 barrier / stage / phase 才能合法交接”的关系
临时收束成一个 IR 里的异步引用对象
```

### 2.1.1 用一个最小双缓冲例子看：为什么普通 SSA 不够

先看没有 split 之前最普通的情形：

```text
for tile i:
  %a = load tile i
  %b = load tile i
  %c = mma(%a, %b)
  use %c
```

这里 `%a -> mma` 这种 SSA 依赖通常就够了，因为默认前提是：

- producer 和 consumer 在同一个执行上下文里
- 中间没有 ownership 转移
- 没有“第几个 buffer slot”的问题
- 程序走到这里时，这个值就可以直接被下游用

warp specialization 之后，问题会变成另一种形状。假设有两个 partition：

- `Producer partition`：负责把 tile 搬到 shared memory
- `Consumer partition`：负责从 shared memory 读 tile 做 `mma`

并且我们用 2-stage double buffer：

- `buffer[0]`
- `buffer[1]`

逻辑上想跑成这样：

```text
t0:
  Producer 写 buffer[0]，准备 tile0

t1:
  Consumer 读 buffer[0]，计算 tile0
  Producer 同时写 buffer[1]，准备 tile1

t2:
  Consumer 读 buffer[1]，计算 tile1
  Producer 回头重用 buffer[0]，准备 tile2
```

这时 consumer 依赖的已经不是“某个 SSA 值”，而是：

```text
某个时刻
某个 stage / phase
对应的那个 buffer slot
现在是否已经合法交给我读
```

如果只保留普通 SSA 图，形状会更像：

```text
%buf0 = alloc_shared
%buf1 = alloc_shared

producer:
  async_copy tile0 -> %buf0
  async_copy tile1 -> %buf1
  async_copy tile2 -> %buf0

consumer:
  %x0 = load %buf0
  mma %x0
  %x1 = load %buf1
  mma %x1
  %x2 = load %buf0
  mma %x2
```

这张图表面上有依赖，但它没有把最关键的信息显式化：

- consumer 读 `%buf0` 时，到底读的是 `tile0` 那次写，还是 `tile2` 的那次重写
- 当前 `buffer[0]` 归 producer 写，还是归 consumer 读
- producer 什么时候算“写完，可以交给 consumer”
- consumer 什么时候算“用完，可以交回 producer 重用”

也就是说，编译器看到的如果只是：

```text
大家都在碰同一个 buffer 对象
```

那它还不知道：

```text
这个 buffer 对象在不同轮次里的不同生命周期
```

而双缓冲 / 多缓冲协议真正关心的是每个 slot 的状态切换：

```text
empty
-> producer 可写
-> full
-> consumer 可读
-> empty
-> 下一轮再次可写
```

所以 split 之后，真正需要表达的就不再只是：

```text
哪个值流向哪个 use
```

而是：

```text
哪个 buffer slot
在第几轮 / 第几个 phase
由谁持有
什么时候从 producer 交给 consumer
什么时候再交回去复用
```

这正是 `ARef` 在补的东西。

换句话说：

```text
SSA edge 主要描述 value flow
ARef 主要描述 staged buffer handoff
```

因此 `ARef` 不只是“把原值再包一层”，而是在 IR 里把下面这些信息显式化：

- 交接对象是哪组 multi-buffer slot
- 当前进入的是哪一个 `stage/phase`
- producer 的可写区间在哪里
- consumer 的可读区间在哪里
- lowering 时该生成哪套 barrier / wait / signal 来维护 empty/full handoff

所以更准确的说法不是：

```text
consumer 依赖 producer 产生的一个 SSA 值
```

而是：

```text
consumer 依赖 producer 把某个 buffer slot 在某一轮里填满，
并按协议把它交接出来
```

### 2.2 它在 IR 里长什么样

NVWS 里有一组配套 op：

- `nvws.aref.create`
- `nvws.aref.buffer`
- `nvws.aref.get.enter`
- `nvws.aref.get.exit`
- `nvws.aref.put.enter`
- `nvws.aref.put.exit`

定义见 [NVWSOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td:44)。

这些名字已经暴露了它的设计意图：

- `create`：先创建异步引用
- `put`：producer 进入/退出“我可以把数据放进去”的区域
- `get`：consumer 进入/退出“我可以把数据取出来”的区域
- `buffer`：通过 aref 间接拿到底层 buffer

所以 `ARef` 不是“某个 barrier”，而更像：

```text
一份显式的 producer/consumer 交接合同
```

### 2.3 它最后会 lower 成什么

`NVWSLowerAref` pass 的摘要写得很直接：

- `Convert nvws.aref.* to ttng.*barrier* ops`
- 为每条 aref communication 决定 matched value / barrier set
- 决定 empty/full 语义下的 wait / signal

见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)。

所以一定要把这个顺序记住：

```text
ARef
!= barrier 本体

ARef
-> lowerAref
-> 具体 barrier / wait / signal / stage / phase 资源
```

这也是为什么说它是**同步抽象**，而不是同步原语。

---

## 3. TMEM token 是什么

TMEM token 更容易被误会成“完成 token”或“同步 token”，但它其实不是。

这里先钉住它和 `ARef` 的根本差别：

```text
ARef 出现，是因为 SSA 不够表达“跨 partition 的交接协议”

TMEM token 出现，是因为 SSA 不够显式表达“TMEM / accumulator 的读写依赖链”
```

也就是说，两者都和“只靠普通 SSA 不够”有关，但补的是两种完全不同的缺口：

- `ARef` 补的是 **handoff / ownership / stage-phase 交接**
- `TMEM token` 补的是 **TMEM memory dependence 的显式串联**

如果把两者各自最短地压成一句话：

```text
ARef = 交接合同
TMEM token = 依赖流水号
```

`ttng.tc_gen5_mma` 相关 op 定义里写得很清楚：

> This operation takes and produces an optional token to indicate TMEM read and
> write on its accumulator operand. When the tokens are present, they can be
> used to check aliasing and modref on the accumulator memory.

见 [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:651)。

把这段压成中文就是：

```text
TMEM token = 把 accumulator / TMEM 上的读写依赖显式串起来，
             让编译器能判断 alias / modref / 重排是否合法
```

所以它解决的问题不是：

```text
producer 把东西交给谁
```

而是：

```text
这些都在碰同一个 TMEM / accumulator 的 op，
编译器怎么知道它们之间不能乱换顺序？
```

### 3.1 它解决什么问题

TMEM 路径有一个特点：

- 结果不只是普通 SSA 值
- 很多真实约束落在 TMEM / accumulator 的隐式存储效果上

如果没有 token，这些依赖在 IR 里会不够显眼，编译器更难知道：

- 哪个 `tc_gen5_mma` 读了前一个 accumulator 状态
- 哪个 `tcgen05.ld` / `tcgen05.st` 和已有 TMEM 访问有关
- 哪些 op 不能乱重排

可以把它想成这样一条链：

```text
token_in
-> tc_gen5_mma / tcgen05.ld / tcgen05.st
-> token_out
-> 下一个 TMEM 相关 op
```

这条链表达的不是“完成已经发生”，而是：

```text
后一个 TMEM 相关 op
依赖前一个 op
在 accumulator / TMEM 上留下的读写效果
```

所以 token 的职责不是“同步已经发生”，而是：

```text
把本来隐蔽的依赖显式化
```

### 3.2 它不负责什么

这一点必须钉死，不然最容易和 `mbarrier` 混掉。

`TMEM token` 不负责：

- 不负责 completion observation
- 不负责 execution rendezvous
- 不负责 proxy visibility
- 不负责线程阻塞

它不回答：

- “异步 tcgen05 完成了没？”
- “现在能不能安全 wait 过去？”
- “别的 warp-group 到齐了没？”

真正回答 “tcgen05 完成了没” 的，是：

```text
tc_gen5_commit -> wait_barrier
```

所以更稳的分工应该是：

```text
ARef
  = 解决“交接怎么表达”

TMEM token
  = 解决“依赖怎么显式串起来”

tc_gen5_commit -> wait_barrier
  = 解决“完成怎么观察”
```

真正修 TMEM 数据 hazard 的，是：

```text
TMemBarrierInsertion
-> ttg.barrier local
```

所以更稳的分工是：

| 机制 | 主要职责 |
|---|---|
| `TMEM token` | 显式依赖边 |
| `tc_gen5_commit -> wait_barrier` | 异步完成观察 |
| `TMemBarrierInsertion` | TMEM data hazard repair |

### 3.3 它最后会不会留下来

不会一直留到最后。

仓库里甚至有一个专门 pass：

- `triton-nvidia-gpu-remove-tmem-tokens`
- `remove TMEM memory dependency tokens from the IR, after they are no longer needed`

见 [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:193)。

这说明 token 的定位非常明确：

```text
它是中间依赖建模资源
不是最终硬件同步资源
```

另外，`AsyncTokenType` 到 LLVM 时只是一个普通整数类型，见
[TypeConverter.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:92)。
这也侧面说明它不是某种硬件专用 barrier object。

---

## 4. ARef 和 TMEM token 到底差在哪

最容易混的地方是：两者都“不是最终硬件原语”，都出现在异步 / warp-specialized 路径里。

但它们服务的问题层次不同。

### 4.1 ARef 关注的是“交接协议”

它回答的是：

```text
producer 和 consumer 之间
这个值 / buffer / ownership
应该怎么交接？
```

它关心的是：

- producer / consumer 分区
- stage / phase
- empty / full
- buffer ownership
- 后续要 lower 成哪些 barrier 资源

### 4.2 TMEM token 关注的是“依赖保真”

它回答的是：

```text
这些 TMEM / accumulator access
在编译器眼里怎样显式保持先后依赖？
```

它关心的是：

- alias
- modref
- accumulator RAW/WAR/WAW 风险
- 不让优化和调度把依赖冲掉

### 4.3 一张对照表

| 维度 | `ARef` | `TMEM token` |
|---|---|---|
| 所在层 | NVWS / warp specialization 抽象层 | TTGIR / NVIDIA GPU 异步依赖层 |
| 核心作用 | 表达异步交接 / ownership transfer | 表达 TMEM / accumulator dependency |
| 主要问题 | producer/consumer 怎样交接 | 编译器怎样看见真实依赖 |
| 会不会直接变成硬件同步原语 | 不会，先 lower 成 barrier-centric contract | 不会，后续可被移除 |
| 是否回答“完成了没” | 否 | 否 |
| 是否回答“谁到齐了没” | 否 | 否 |
| 和 `mbarrier` 的关系 | lowering 后可能落成 barrier / wait / signal 资源 | 与 `mbarrier` 正交；completion 仍看 `tc_gen5_commit -> wait_barrier` |

---

## 5. 为什么两者都要有，不能只留一个

因为它们解决的是不同维度的问题。

只靠 `ARef` 不够，因为：

- `ARef` 主要解决跨 partition 的交接和 ownership 协议
- 它不替代 TMEM / accumulator 级别的 alias / modref 依赖表达

只靠 `TMEM token` 也不够，因为：

- token 只是一条 dependency edge
- 它并不表达“哪个 producer 在哪个 stage 把哪个 buffer 交给哪个 consumer”

所以两者并存的理由是：

```text
ARef 负责把“谁和谁交接”说清楚
TMEM token 负责把“这些 TMEM 访问不能乱动”说清楚
```

---

## 6. 它们和 barrier/fence/mbarrier 的边界

这一节只收口边界。

### 6.1 ARef 不是 barrier

虽然 `LowerAref` 最后会产出 barrier-centric contract，但：

```text
ARef != barrier
```

`ARef` 只是高层同步抽象。

### 6.2 TMEM token 不是 wait / completion object

虽然它经常和 `tc_gen5_mma`、`tcgen05.ld`、`tcgen05.st` 一起出现，但：

```text
TMEM token != completion token
TMEM token != mbarrier
TMEM token != wait_group
```

### 6.3 真正的职责分工

如果你在读 Blackwell / TMEM / warp specialization 路径，建议按下面这个顺序分层：

1. `ARef`
   先看 producer/consumer 的交接协议是不是已经显式化
2. `TMEM token`
   再看 TMEM / accumulator 依赖是不是已经被显式串起来
3. `tc_gen5_commit -> wait_barrier`
   再看异步完成是不是已经有 completion observation
4. `TMemBarrierInsertion`
   最后看 TMEM data hazard 有没有被 barrier repair

---

## 7. 最后压成一句话

如果只想记一句，记这个：

```text
ARef 让“异步交接关系”显式化，
TMEM token 让“TMEM 依赖关系”显式化；
两者都服务于正确 lowering，但都不是最终硬件同步原语。
```

---

## 8. 参考

1. `NVWS_ArefType` 定义：
   [NVWSTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSTypes.td:34)
2. `nvws.aref.*` op 定义：
   [NVWSOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td:44)
3. `NVWSInsertAref` / `NVWSInsertTmemAref` / `NVWSLowerAref`：
   [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td:63)
4. `ttng.tc_gen5_mma` token 语义：
   [TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:651)
5. `RemoveTMEMTokens`：
   [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:193)
6. `AsyncTokenType -> i32` lowering：
   [TypeConverter.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp:92)
