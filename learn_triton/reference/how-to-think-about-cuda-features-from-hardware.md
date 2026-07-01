# 如何从硬件实现视角思考 CUDA Feature

这份文档不讲某一个 CUDA API 的用法，而是讲一套更底层的思考方法：

- 一个 CUDA feature 在硬件上通常意味着什么
- 一个 loop 里哪些东西是循环相关的，哪些不是
- loop unroll、software pipeline、async copy 这类优化为什么会吃资源
- 如果不是在 NVIDIA 平台，而是要实现“类似能力”，应该怎样拆问题

目标不是背结论，而是形成一套稳定的分析框架。

## 1. 先换视角：不要只看 API，要看协议和资源

很多人学 CUDA/Triton 时，容易停留在：

- 这条指令做了什么
- 这个 intrinsic 怎么调用
- 这个优化为什么更快

但如果问题变成：

- 这个 feature 在硬件里怎么实现
- 换一个 GPU 平台，怎样设计类似机制
- loop 展开后为什么资源突然爆掉

那就不能只看 API 了，而要切到下面这个视角：

- 谁发起这件事
- 谁真正执行这件事
- 中间需要维护哪些状态
- 完成和顺序怎么定义
- 消耗哪些有限资源

一句话概括：

> 把 feature 看成一个“执行 agent + 状态机 + 顺序协议 + 资源账本”。

但只停在这一步还不够。

硬件分析最后一定要落到几个可计算的问题：

- 需要多少并发，才能打满这条路径
- 现在是 compute-bound 还是 memory-bound
- 这个 loop 展开到几倍开始不值
- 资源上限先撞到的是寄存器、shared memory，还是 in-flight state

所以这份文档后面会反复用两条定量骨架：

- `Little's Law`：为了打满某条路径，需要多少 in-flight work
- `Roofline`：当前 kernel 更接近算力瓶颈还是带宽瓶颈

## 2. 分析任何 feature 的四层框架

建议固定按四层来拆。

### 2.1 语义层

先问这个 feature 对软件承诺了什么。

重点只看四件事：

- 谁发起
- 什么时候算完成
- 对谁可见
- 和哪些访问有序

例如异步 copy：

- 发起者通常是 thread / warp
- 完成不等于数据立刻能安全消费，往往还需要 `wait_group`、`mbarrier.wait` 之类的完成确认
- 可见性取决于写到哪里，比如 shared memory、tensor memory
- 顺序不一定自动成立，可能还需要 proxy fence

这一层如果没有想清楚，后面的“硬件怎么实现”基本都会跑偏。

### 2.2 执行层

再问：到底是谁在干活。

常见有三种模式：

- 发起线程发出请求，由普通执行流水跟踪完成
- 发起线程把请求提交给某个异步执行单元
- 某个专用引擎在后台执行，线程只负责发命令和等待完成

所以要强迫自己区分三类角色：

- issuing agent：谁提交请求
- executing agent：谁真正执行
- observing agent：谁感知完成并继续推进程序

很多 CUDA feature 难就难在，这三者不是同一个东西。

例如：

- 普通 load/store：线程发起后，结果是否可继续推进往往由 scoreboard 和寄存器依赖来跟踪，它并不是“发射点同步阻塞”
- `cp.async` / `LDGSTS`：线程发起，但执行上更像交给异步 copy 路径
- TMA：线程提供描述符和同步对象，真正的数据搬运由更专门的异步路径推进
- `wgmma.mma_async`：发起和执行明显分离，完成还需要单独确认

这里要特别修正一个常见误区：

- GPU 的普通 load 早就是“延迟隐藏式”的，不是 CPU 式的发射点同步完成
- warp 发出 load 之后，可以先执行与该寄存器无关的后续指令
- 真正会 stall 的时刻，通常是后面的指令第一次消费该寄存器，而 scoreboard 发现数据还没回来

所以 `cp.async` 的真正新意，不是“第一次让访存变异步”，而是：

- 数据直接落到 shared memory，而不是先落寄存器再 `st.shared`
- 完成跟踪从“按寄存器依赖计分”转成“按 async group / barrier 协议计分”

### 2.3 状态层

然后问：硬件最少要记住什么状态。

通常离不开下面这些：

- 源地址、目标地址、步长、layout、descriptor
- 当前 outstanding 请求数
- queue depth / credit
- barrier phase / token / parity
- ownership 状态
- scoreboard bits
- 某个 warp/warpgroup 的 in-flight async group

你可以把每个 feature 都想成一个小状态机：

1. 接收请求
2. 占用资源
3. 进入执行中
4. 产生完成信号
5. 释放资源

如果一个 feature 看起来“很玄”，通常只是因为它背后的状态机没有被说清楚。

### 2.4 资源层

最后问：它到底吃什么资源。

不要只盯寄存器。

完整资源账本通常包括：

- 寄存器
- shared memory
- barrier slot
- async queue entry
- descriptor slot
- scoreboard / dependency tracking state
- issue bandwidth
- memory transaction slot
- tensor core / copy engine / load-store pipe 的占用

很多优化表面上是在“增加并行度”，本质上是在花更多状态和缓冲去换 overlap。

### 2.5 两条定量主轴：Little's Law 和 Roofline

如果前面四层是“定性框架”，那这两条就是“定量主轴”。

#### 2.5.1 Little's Law：需要多少 in-flight work

最该记住的一条式子是：

`in-flight work ~= latency * throughput`

意思是：

- 想打满一条路径，必须让足够多的工作同时在飞
- 延迟越大，需要的并发越高
- 目标吞吐越大，需要的并发也越高

对内存路径尤其重要。

例如只做数量级估算：

- 假设 HBM 带宽目标是 `2 TB/s`
- 往返延迟按 `500 ns` 量级估

那么要打满它，需要同时在飞的数据量大约是：

`2 TB/s * 500 ns ~= 1 MB`

这里要特别注意口径：

- 这个 `1 MB` 是整台 GPU、所有 SM 合起来，为了逼近该带宽目标所需维持的在飞数据量级
- 真正分析到 kernel 时，通常还要继续往下拆成“每个 SM 需要多少 in-flight bytes”，再拆成“每个 warp / 每个 CTA 大概要承担多少 outstanding work”

这不是精确常数，但它给了你一个非常重要的判断：

- 只靠几个零散请求，不可能打满高带宽内存
- 一定要么靠很多 warp 的 TLP
- 要么靠更深的 async pipeline / 更多 outstanding request
- 更常见的是两者组合

这条式子会直接影响：

- 需要多少并发 warp
- `num_stages` 取几才有意义
- queue depth / barrier / async group 为什么会成为资源
- 为什么“只有双缓冲”有时还不够

#### 2.5.2 Roofline：先判断瓶颈在哪

另一条常用判断是 Roofline。

先估一个 kernel 的算术强度：

`arithmetic_intensity = FLOPs / Bytes_moved`

然后再问：

- 如果算术强度低，更多时候会被带宽限制
- 如果算术强度高，才更可能逼近算力屋顶

这条判断很重要，因为它决定优化方向：

- memory-bound：优先看访存合并、cache、shared reuse、async pipeline
- compute-bound：优先看 tensor core 利用率、issue、依赖链、寄存器复用

不要先谈优化，再去猜瓶颈。先用 Roofline 做粗分流。

#### 2.5.3 数量级锚点

硬件分析最好带几个数量级锚点，但一定要把它们当成经验量级，而不是通用真值。

常见经验量级可以这样记：

- 寄存器访问接近最便宜的片上状态，延迟可近似看成 `0-cycle/极低`
- 每线程最多大约 `255` 个 32-bit 寄存器
- 每个 SM 的寄存器总量常见是 `64K` 个 32-bit 寄存器量级
- shared memory 延迟常是几十 cycle 量级，容量是一两百 KB 量级
- L2 延迟常是几百 cycle 量级
- global/HBM 往返延迟常是几百 cycle、几百 ns 量级

这些数不需要背得特别精确，但你心里必须有量级感。  
没有量级感，就很难判断 Little's Law、occupancy、`num_stages` 是否合理。

## 3. 一个 feature 在硬件上如何实现：固定问题模板

以后看到一个新 feature，先不要急着看 API 文档，先写这 8 个问题。

### 3.1 输入是什么

例如：

- 地址
- 大小
- mask / predicate
- stride / shape
- phase / token
- barrier 指针
- descriptor

如果输入里包含复杂的 layout、mbarrier、tensor descriptor，那通常意味着硬件不会只是“执行一条普通访存”。

### 3.2 谁提交它

可能是：

- 单线程
- 一个 warp
- 一个 warpgroup
- 一个 CTA 里的 elected thread

谁提交，直接影响：

- 控制开销
- 收敛要求
- barrier expected count
- 指令语义是不是 per-thread / per-warp / collective

### 3.3 谁执行它

这是最关键的问题之一。

常见执行路径：

- ALU
- load/store pipe
- async copy path
- tensor core path
- TMA / DMA-like engine

如果执行者不是发起线程本身，那一定要继续问：

- 请求放进哪里
- 中途怎么追踪
- 完成如何回传

### 3.4 中间经过什么队列或缓冲

这一步决定吞吐和背压。

典型对象包括：

- request FIFO
- async group
- barrier state
- descriptor cache
- transaction scoreboard
- shared memory stage buffer

任何“可以 overlap 的 feature”，几乎都意味着中间有某种 in-flight 状态容器。

这一节还要补一个经常被漏掉的点：

- global memory 侧不只有“有没有请求”，还有 transaction 粒度和 coalescing
- shared memory 侧不只有“容量够不够”，还有 bank 带宽和 bank conflict

也就是说，资源账本至少要分两层：

- 容量：能不能放下
- 带宽：同一时刻能不能高效地取/写

这就是为什么：

- global load/store 要看 cache line / transaction 合并
- shared memory tile layout 要看 bank 映射
- swizzle 往往不是为了“语义正确”，而是为了避免 bank conflict、提高实际带宽

### 3.5 完成如何定义

要明确区分：

- 请求被接受
- 数据真正搬完
- 结果对消费方可见
- 消费方已经被允许继续

这几个时刻往往不是同一个时刻。

异步机制里最常见的错误就是把：

- issued
- completed
- visible
- ordered

混成一个概念。

### 3.6 顺序如何定义

一定要分开问：

- completion 怎么保证
- ordering 怎么保证

很多人会把 `wait` 当成“顺序也有了”，这是错的。  
`wait` 更多是“完成条件”，而 fence/proxy fence 处理的是“先后顺序的可观察性”。

### 3.7 背压如何产生

一个 feature 真正的成本，经常体现在背压上。

例如：

- async queue 满了
- barrier slot 不够
- shared memory stage 不够
- outstanding transaction 太多
- scoreboard 依赖太长
- issue pipe 被地址计算和控制指令挤满

如果你能描述背压是怎么出现的，说明你已经不是在看表面 API 了。

这时可以直接套前面的 Little's Law：

- 如果想要的吞吐很高，但允许的 in-flight state 很少，就一定会早早背压
- 如果延迟很长，但 warp 数、stage 数、queue depth 都不够，也一定打不满

所以背压常常不是“某个单点变慢”，而是“你提供的并发度不够维持目标吞吐”。

### 3.8 扩展瓶颈在哪里

最后看：

- 是吞吐瓶颈
- 是延迟瓶颈
- 是容量瓶颈
- 是依赖瓶颈

同一个 feature，在不同 kernel 里，主瓶颈可能完全不同。

更稳妥的顺序是：

1. 先用 Roofline 判断更偏 compute-bound 还是 memory-bound
2. 再在对应一侧继续细分是延迟、吞吐、容量还是依赖问题

这样比一上来就枚举瓶颈更不容易跑偏。

## 4. 分析 loop：先分清哪些是循环相关，哪些不是

分析 loop 时，最有用的不是先数指令，而是先分类变量。

建议固定分成三类。

### 4.1 循环不变量

这些值在整个 loop 生命周期内不变，通常适合 hoist 到循环外。

典型包括：

- base pointer
- 常量 stride
- tile shape
- descriptor
- 某些边界常量
- 固定的 barrier 地址

这些量不是 loop 的“推进器”，只是 loop 每一轮都会引用的背景参数。

### 4.2 循环携带状态

这些值会跨迭代传递，是 loop 真正的骨架。

典型包括：

- induction variable
- pointer advance 后的地址
- stage id
- parity / phase
- accumulator
- producer/consumer ownership
- 上一轮产生、下一轮继续使用的 token

只要一个值满足下面任一条，就要把它当作 loop-carried state：

- 跨迭代 live
- 决定下一轮能不能发起
- 决定下一轮访问哪里
- 决定同步协议是否成立

### 4.3 每轮临时量

这些值只服务于当前迭代，通常用完就死。

例如：

- 当前 tile 的实际地址
- 当前迭代的 predicate
- 本轮 load 得到的 fragment
- 本轮局部计算的中间结果

这类量最容易在 unroll 后膨胀，因为它们的 live range 会被拉长。

在 SIMT 机器上，还要顺手多问一句：

- 这一轮有没有 warp divergence
- 这个分支最终会变成真实分歧，还是 predication

因为分歧会带来额外的 per-iteration 成本：

- 控制流收敛/发散开销
- 无效 lane 的吞吐浪费
- predicate/live range 的进一步拉长

## 5. loop 的本质：控制状态和数据状态

进一步说，loop 里的量还可以按职责分成两类：

- control state
- data state

### 5.1 control state

控制 loop 如何推进，例如：

- 迭代计数
- 分支条件
- barrier phase
- software pipeline 的 stage index
- buffer ownership

这类状态决定“下一步能不能做、该做什么”。

### 5.2 data state

表示当前正在被处理的数据，例如：

- 当前 tile 的 fragment
- accumulator
- 本轮从内存取出的值

硬件设计时，这两类状态要分开看：

- control state 决定调度和协议
- data state 决定存储和带宽压力

很多优化之所以难，是因为它同时拉高了两者的成本。

## 6. loop unroll 后，资源为什么会变多

展开 loop 时，不要只想“代码复制了几份”，真正关键的是：

> live range 被拉长了，更多状态同时存在。

通常重点看 5 类资源。

### 6.1 寄存器

这是最常见的第一瓶颈。

unroll 之后：

- 多份地址计算结果可能同时存活
- 多份 predicate 可能同时存活
- 多份 fragment 可能同时存活
- 某些本来能立即消费的中间值，现在要等更后面的指令

所以寄存器增长往往不是“指令数线性增加”那么简单，而是取决于：

- 哪些中间值跨更长区间 live
- 哪些值被不同 stage 同时持有

### 6.2 shared memory

如果 unroll 和 software pipeline / multi-stage buffering 绑定，shared memory 往往按 stage 数增加。

典型近似：

`shared_memory ~= per_stage_buffer * num_stages`

但这通常还只是主项，实际还应记得：

- `num_stages >= 2` 才有双缓冲的基本意义
- 除了 `per_stage_buffer * num_stages`，往往还有一块固定的静态 shared memory 开销

这就是为什么 `num_stages` 很容易直接影响 occupancy。

### 6.3 outstanding 操作数

展开后更容易出现：

- 更多 in-flight load/store
- 更多 in-flight async copy
- 更多未完成的 barrier phase
- 更深的 async group

如果硬件对这些 in-flight 状态有上限，展开收益会很快碰顶。

### 6.4 调度和 issue 压力

更多展开不只是在“增加 ILP”，也在增加：

- 地址计算指令
- predicate 指令
- move / pack / unpack
- 等待和控制指令

最后可能不是算力不够，而是前端调度和 issue 被挤满。

### 6.5 occupancy

最终一定要回到 occupancy。

展开带来的资源增长，会压缩：

- 每个 SM 可驻留的 CTA 数
- warp 数
- 并发 hiding latency 的能力

但 occupancy 不只是成本项，它本身也是最原始的藏延迟机制。

### 6.6 两种藏延迟机制：TLP 和显式流水

GPU 至少有两条主要的藏延迟路径：

- 靠 TLP / 高 occupancy：warp 多，一个 stall 了就切到另一个
- 靠显式流水：warp 不一定多，但用 async copy、多 stage buffer、barrier 把延迟显式地 pipeline 掉

这两者不是完全独立的，经常是相互替代、相互补偿。

所以不能简单把“occupancy 掉了”理解成一定变差。

更准确的说法是：

- 如果没有显式流水，occupancy 低通常很危险
- 如果已经有足够深的 async pipeline，较低 occupancy 也可能仍然高效

现代 GEMM 尤其常见后一种情况：

- 寄存器和 shared memory 压力很大，occupancy 天生不高
- 于是干脆依靠 TMA / async copy / 多 stage buffering 去补偿

很多时候，这还依赖更明确的角色分工，也就是 warp specialization：

- producer warp 负责发起 TMA / async copy、推进 barrier 或 mbarrier
- consumer warp 负责做 MMA / Tensor Core 计算

也就是说，低 occupancy 下还能维持强 overlap，往往不只是“有多 stage”，而是“谁发搬运、谁做计算”已经被显式拆开了。

所以 unroll 和 pipeline 的判断，不能只看 occupancy 下降，还要看它是不是换来了足够的 overlap。

## 7. 一个实用的资源估算框架

可以用下面这个顺序快速判断一个展开/流水方案值不值得。

### 7.1 先看想换来什么

通常目标只有几类：

- 增加 ILP
- 增加 MLP
- 增加在飞请求数
- 提前发起访存
- 增加 copy/compute overlap
- 减少 loop 控制开销

### 7.2 再看要多付出什么

代价通常是：

- 更多寄存器
- 更多 shared memory
- 更多 barrier / async state
- 更复杂的依赖图
- 更差的 occupancy

### 7.3 先用 Little's Law 算“并发够不够”

先粗算：

- 目标吞吐是多少
- 目标路径延迟是多少
- 需要多少 in-flight bytes / requests / warps

如果这一步就明显不够，那后面细节再漂亮也打不满。

这时常见的补救手段只有几类：

- 增加 warp 数
- 增加 unroll / ILP
- 增加 `num_stages`
- 增加 async queue depth / outstanding group

### 7.4 再看是哪种藏延迟机制在主导

然后问：

- 当前主要靠高 occupancy 藏延迟
- 还是主要靠显式 async pipeline 藏延迟

这一步决定你该优先保什么：

- 如果主要靠 TLP，就要非常警惕寄存器和 shared memory 继续上涨
- 如果主要靠显式流水，就要重点看 stage、barrier、in-flight copy 是否足够深

在后一种情况下，还应继续追问：

- copy 和 compute 是不是由同一批 warp 交替完成
- 还是已经通过 warp specialization 拆成了 producer / consumer 分工

因为低 occupancy 下还能不能把流水跑顺，这个分工经常就是关键前提。

### 7.5 最后看总代价是否值得

可以记一个很粗但很有用的式子：

`收益 ~= 暴露出的并行性 - live range 成本 - occupancy 损失 - 调度压力`

如果后面三项超过第一项，优化大概率就是负收益。

## 8. 用 async copy 做一个硬件视角示例

下面用一个熟悉的例子把前面的框架串起来。

### 8.1 语义

`async copy` 的表面语义是：

- 发起从某处到某处的数据搬运
- 发起后线程可以先继续
- 真正消费前要确认完成

如果再往下细化：

- “发起成功”不等于“数据可读”
- “完成”不等于“顺序已经对另一侧观察者成立”

### 8.2 执行 agent

这类操作并不是因为“普通 load 不异步”才显得特别，而是因为它把异步性放到了另一套协议里。

普通 global load 的典型路径更接近：

1. warp 发出 load
2. scoreboard 跟踪目标寄存器是否 ready
3. 与该寄存器无关的指令可以继续推进
4. 真正消费该寄存器时，如果数据未回，就在依赖点 stall

而 `async copy` 更接近：

1. 线程发出请求
2. 请求进入某种 async 执行路径
3. 后台执行搬运
4. 数据直接落到 shared memory 或别的目标存储
5. 通过 wait/barrier/fence 让软件在正确的时刻观察到结果

所以从硬件角度，它至少意味着：

- 有单独的 in-flight copy 状态
- 有完成通知机制
- 可能有独立于 normal memory path 的顺序模型
- 常常还能省掉“global -> register -> shared”的中转成本

### 8.3 状态

至少需要维护：

- 源/目标地址
- 大小
- 所属 group
- 完成状态
- 与 barrier 或 wait 对象的关联

如果还有多 stage buffering，就还要维护：

- 当前 stage
- 下一 stage
- 哪个 stage 可以被 producer 重写
- 哪个 stage 还在被 consumer 读取

### 8.4 资源

它吃的资源不只是带宽，还包括：

- async queue entry
- group bookkeeping
- barrier state
- shared memory buffer
- 额外寄存器
- shared memory bank 带宽
- global transaction 合并效率

所以 async copy “更快”不是免费午餐，而是用更多状态和缓冲换 overlap。

### 8.5 怎么估 stage 该取几

这时可以把前面的定量骨架直接用起来。

思路不是先问“别人常用几 stage”，而是先问：

- 单个 stage 能提供多少 bytes 的有用工作
- 一轮 copy 到可消费大概要多久
- 目标是让 copy latency 被多少计算覆盖掉

如果：

- 一轮计算时间明显小于一轮 copy 完成时间
- 且 in-flight stage 太少

那你就需要更深的 pipeline。  
反过来，如果 stage 再加深也不能带来更多 overlap，只会继续抬高寄存器/shared memory 压力，就该停。

## 9. 如果不是 NVIDIA 平台，要实现类似能力，该怎么想

这时就不要问“有没有 `cp.async` 对应指令”，而是问更本质的问题。

### 9.1 是否需要独立执行 agent

如果想让 copy 和 compute overlap，通常就要有某种独立于主执行流的执行路径。

否则：

- 线程一边 copy 一边算只是软件错觉
- 本质上还是串行占用同一执行资源

### 9.2 是否需要显式同步对象

只要发起和执行解耦，就需要回答：

- 完成怎么通知
- 谁来等
- 多个消费者怎么协调

这通常就会落到 barrier、token、counter、event、phase bit 这类对象上。

### 9.3 是否需要独立顺序模型

如果引入了“normal path 之外的异步路径”，往往就会引入顺序问题：

- 普通 store 之后，异步读能不能立刻看到
- 异步写完之后，普通 load 何时可见
- 是不是要专门的 fence

这就是为什么 proxy 语义非常重要。

如果拿 NVIDIA 作参照，可以把它理解成：

- generic proxy：普通 load/store 所在的观察路径
- async proxy：某些异步执行路径所在的观察路径
- 两侧是否自动有序，不能想当然，往往要靠专门的 proxy fence

而且顺序保证要特别注意限定词：

- 很多规则只对“同一地址”成立
- 不能偷换成“对所有普通访问天然全局有序”

### 9.4 硬件资源要怎么限流

你一定要设计上限：

- queue 多深
- 每个线程/warp 最多多少 outstanding
- barrier 最多多少个
- 每个 CTA 最多多少 stage buffer

因为没有上限的异步机制，在真实硬件上通常不可实现。

## 10. 以后遇到类似问题，建议固定这样思考

你可以把下面这 6 行当成一个答题模板。

```text
1. Semantic contract:
2. Executing agent:
3. State to maintain:
4. Ordering/completion model:
5. Bounded resources:
6. Bottleneck after scaling/unrolling:
```

如果一个 CUDA/Triton feature、一个 loop 优化、一个硬件设计想法，你能把这 6 行填出来，说明你已经在用“实现者”的视角看问题了。

## 11. 常见误区

### 11.1 只看 API，不看状态机

这样会知道“怎么调用”，但不知道为什么会快、为什么会卡、为什么会错。

### 11.2 把 completion 和 ordering 混为一谈

这是异步机制里最常见的概念错误之一。

### 11.3 只数算术指令，不数状态成本

很多优化失败，不是算术变多，而是：

- live range 变长
- barrier/state 变多
- occupancy 掉了

### 11.4 把 unroll 当成纯编译器问题

其实它是一个很硬的硬件资源问题，因为它直接影响：

- 寄存器占用
- in-flight state
- 调度复杂度
- occupancy

### 11.5 只算容量，不算带宽和粒度

shared memory 放得下，不代表跑得快；global memory 发得出去，也不代表合并得好。

容量、带宽、transaction 粒度、bank conflict，要分开看。

## 12. 学习建议

如果你想把这种思维真正练出来，建议按下面顺序练。

### 12.1 先拿一个具体 feature 练

例如：

- `cp.async`
- TMA
- `wgmma.mma_async`
- `mbarrier`

每个 feature 都强行回答：

- 语义是什么
- 谁执行
- 状态是什么
- completion / ordering 怎么区分
- Little's Law 下需要多少 in-flight work
- 哪些资源会先爆

### 12.2 再拿一个具体 loop 练

例如 GEMM 主循环，强行拆出：

- invariant
- loop-carried control state
- loop-carried data state
- per-iteration temporaries
- warp divergence 是真实分歧还是 predication

### 12.3 最后把 feature 和 loop 合起来看

这时你会真正明白：

- 为什么 async copy 需要多 stage buffer
- 为什么 stage 一多寄存器和 shared memory 会涨
- 为什么有时低 occupancy 仍然能跑得很好
- 为什么 barrier/fence 是协议的一部分，不只是“同步指令”

## 13. 记忆版结论

把这几句记住就够了：

- 一个 feature 的硬件本质，是执行 agent、状态机、顺序协议和资源账本。
- 真正让框架可计算的两条主轴，是 `Little's Law` 和 `Roofline`。
- 分析 loop，先分 invariant、loop-carried state、temporary。
- loop unroll 的真实代价，不是代码变长，而是 live range 变长、更多状态同时存在。
- GPU 藏延迟既可以靠高 occupancy，也可以靠显式 async pipeline。
- 异步机制一定要分 completion 和 ordering。
- 普通 GPU load 本来就不是发射点同步完成；`cp.async` 的新意在目标落点和完成跟踪协议。
- 想在别的平台实现类似能力，先想执行路径、同步对象、顺序模型、资源上限。
