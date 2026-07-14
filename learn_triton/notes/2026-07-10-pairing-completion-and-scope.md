# Pairing、Completion 与 Scope：顺序为什么是多角度的

日期：2026-07-10

这篇把前面零散讨论过的几个概念单独收拢：

- `pairing` 是什么
- `completion` 是什么
- 为什么会同时出现“同线程 pairing / 同线程非 pairing / 跨线程 handoff”
- 把范围放大到 `thread / warp / warp-group / CTA / cluster` 后，优先该检查哪一种保证
- `sm80 / sm90 / sm100` 最常见原语分别是什么

这篇现在用一个更统一的框架来组织：

```text
顺序是多角度的
+ scope 也是独立变化的

所以：
不是某一种顺序满足了就够
而是相关角度和相关 scope 上都要满足
```

这正是为什么同样是“前后有关系”，文里会同时出现：

- `pairing` 说的是：ISA 有没有直接给出同线程的顺序承诺
- `completion` 说的是：异步动作是不是已经真的做完
- `barrier` 说的是：多个执行者要不要先 rendezvous
- `fence / handoff` 说的是：结果或顺序有没有被合法发布给后续使用者

如果这几层不拆开，就很容易误读成：

```text
写在前面的指令
= 顺序有了
= 完成了
= 可见了
= 别的线程也能接着用了
```

这四件事在现代 NVIDIA async protocol 里通常不是同一件事。

---

## 1. 总框架：顺序有多个角度，scope 是另一条独立轴

最容易混的地方，是把所有“前后关系”都混成一种顺序。

更稳的读法是先问：

```text
我现在缺的，到底是哪一种顺序保证？
```

### 1.1 第一条轴：顺序有多个角度

常见至少要分这 4 类：

| 来源 | 它回答什么 | 典型代表 |
|---|---|---|
| `ISA-built-in order / pairing` | 某些同线程指令之间，ISA 是否对特定配对额外承诺 issue-order 保证 | 某些 pipelined `tcgen05` pairing `(PTX)` |
| `completion observation` | 异步动作是不是真的做完了 | `cp.async.wait_group` `(PTX)`、`wgmma.wait_group` `(PTX)`、`mbarrier.wait` `(PTX)`、`wait_barrier` `(TTGIR/TritonNvidiaGPU)` |
| `execution rendezvous` | 多个执行者是否都先到齐再继续 | `ttg.barrier` `(TTGIR/TritonGPU)`、`bar.sync` `(PTX)`、cluster barrier `(PTX family)` |
| `visibility / handoff ordering` | 前面的写是否已经对后续消费者合法发布 | `fence.proxy.async` `(PTX)`、`tcgen05.fence::before_thread_sync` `(PTX)`、`tcgen05.fence::after_thread_sync` `(PTX)` |

把这张表压成一句就是：

```text
“顺序”不是一个维度
而是一组不同层面的先后约束
```

### 1.2 第二条轴：scope 也是独立变化的

即使讨论的是同一种顺序，也还要继续问：

```text
这个顺序只在本线程内部成立，
还是要扩展到 warp / warp-group / CTA / cluster？
```

因为 scope 变大之后，原来在小范围里够用的保证，常常就不够了。

最常见的范围轴是：

| scope | 典型问题 |
|---|---|
| thread | 本线程自己后面能不能继续用 |
| warp | 同一个 warp 的 32 lanes 是否已经重新对齐 |
| warp-group | 128 线程的计算 actor 是否已经满足本地协议 |
| CTA | 别的 warp / warp-group 能不能安全接手 |
| cluster | peer CTA 是否已经看到并接受到发布结果 |

### 1.3 为什么只满足一个角度还不够

真正困难的地方就在这里：

```text
某一种顺序满足了
!= 其它顺序也满足了
```

最常见的误读链条是：

1. issue 顺序已经成立
2. 但 completion 还没成立
3. completion 成立了，但别的线程还没 rendezvous
4. rendezvous 成立了，但结果还没发布到另一个 proxy / 更大 scope

所以更准确的判断方式是：

```text
先看缺的是哪一种顺序
再看这个顺序要在哪个 scope 上成立
```

### 1.4 后文应该怎么读

从这里开始，后文所有章节都可以按同一个模板去读：

```text
1. 这里讨论的是哪一种顺序？
2. 这个顺序要在哪个 scope 上成立？
3. 这个顺序/交接靠什么中介或状态承载点落地？
4. 当前已经满足了什么？
5. 还缺什么？
6. 由哪一个原语来补？
```

### 1.5 第三条轴：很多顺序都离不开“中介”

这里的“中介”不是一个口语比喻，而是更具体的问题：

```text
后面的 observer / consumer
到底是通过什么东西知道“现在可以继续”？
```

如果不把这个问题拆开，就很容易把所有同步原语都想成：

```text
发一条指令
硬件神奇地自己保证好了
```

但现实里，很多顺序之所以能成立，是因为硬件里一定有某种状态承载点，或者有一条明确建立起来的 ordering relation。

#### A. 不是所有顺序都要显式 object

最先要拆开的，是“有中介”不等于“有一个程序员可见的 shared-memory 对象”。

至少有 4 类常见中介：

| 中介类型 | 它长什么样 | 典型落点 |
|---|---|---|
| data object | 共享的数据载体本身 | shared memory tile、descriptor / TensorMap metadata、TMEM result |
| completion object | 可共享观察的完成状态对象 | `mbarrier` `(PTX object model)`，驻留在 shared memory |
| barrier state | 隐式硬件会合状态 | warp / CTA / cluster barrier resources |
| ordering relation | 没有单独对象，但有明确建立起来的顺序关系 | ISA pairing、`fence.proxy.async` `(PTX)`、`tcgen05.fence::*` `(PTX)` |

所以“有没有中介”要更准确地问成：

```text
这个顺序/交接
到底是靠 object、memory、barrier state，
还是靠一条被 ISA / fence 明确建立起来的 ordering relation？
```

#### B. 哪些情况特别依赖中介

最典型的是 `completion` 和 `handoff`。

1. `completion`

   只要异步动作和执行流解耦，就必须把：

   ```text
   “它已经完成”
   ```

   变成某个后续点能观察到的事实。

   这时中介要么是：

   - 显式 object  
     例如 `mbarrier` `(PTX object model)`，落在 shared memory

   要么是：

   - 隐式硬件记账  
     例如 `cp.async.wait_group` `(PTX)`、`wgmma.wait_group` `(PTX)` 背后的 outstanding-group / scoreboard 状态

2. `handoff`

   只要是“从一个 actor 交给另一个 actor”，几乎一定要有中介。

   但 handoff 的中介不只有一种：

   - data handoff  
     中介是 shared tile、TMEM result、descriptor metadata 这类 data object

   - completion-fact handoff  
     中介是 `mbarrier` `(PTX object model)` 这类 completion object

   - execution-permission handoff  
     中介是 `bar.sync` `(PTX)`、cluster barrier `(PTX family)` 背后的 barrier state

   - specialized ordering handoff  
     中介不是 object，而是 `tcgen05.fence::before_thread_sync` `(PTX)` /
     `tcgen05.fence::after_thread_sync` `(PTX)` 这类明确建立起来的 ordering relation

#### C. 反过来，哪些顺序不一定需要显式观察对象

`pairing` 和很多 `fence` 类顺序，就不一定表现成“先创建一个 object，再去观察它”。

- `pairing`
  更像 ISA / pipeline contract
- `fence.proxy.async`
  更像 memory/proxy ordering machinery

也就是说，它们当然也依赖硬件内部状态和规则，但通常不是：

```text
程序员显式持有一个 completion object
然后去 poll 它
```

#### D. 最短判断法

以后再看一条同步原语，可以多问一步：

```text
这个顺序/交接
最后是靠什么中介落地的？
```

如果你得到的答案是：

- shared tile / descriptor / TMEM result  
  说明它更偏 data handoff
- `mbarrier`  
  说明它更偏 completion-fact handoff
- `bar.sync` / cluster barrier 背后的硬件状态  
  说明它更偏 execution-permission handoff
- `pairing` / `fence.proxy.async` / `tcgen05.fence::*`  
  说明它更偏 ordering relation

把这个问题加进来以后，很多“为什么这里只写了 fence、那里却要 barrier object”的困惑会更容易拆开。

---

## 2. `pairing` 到底是什么意思

这里的 `pairing` 不是普通语言里的“凑成一对”，而是：

```text
两类指令之间，ISA 是否把它们当成一组有既定衔接关系的 pipeline 序列
```

更直接一点：

- 它讨论的是“顺序从哪里来”
- 它不讨论“完成了没”
- 它也不讨论“跨线程有没有交接好”

所以 `pipelined pairing` 真正表达的是：

```text
同一个线程里，
某些特定指令组合的先后关系，
ISA 已经额外特批了 issue-order 保证
```

如果属于 pairing，含义是：

- 这类前后衔接不用再额外发一个“专门为了补这对指令先后关系”的显式 ordering 机制
- 但这不等于后续消费时一定已经 completion
- 也不等于跨线程交接自动合法

一句话记忆：

```text
pairing = ISA 认可的指令衔接关系
```

再收紧一句：

```text
pairing 保证的是特定同线程指令对的 issue/execution ordering
不是一般意义上的 memory order / visibility guarantee
```

也就是说，即使某对指令属于 pairing，也不能直接推出：

- 前一个异步动作已经 completion
- 前面的写已经对后续 observer 可见
- 别的 thread / warp / CTA 现在能安全接手

所以 `pairing` 真正解决的是：

```text
这对同线程指令的顺序要不要另补 ordering 机制
```

它不直接解决：

```text
结果是不是已经完成
结果是不是已经被合法发布到别的作用域
```

---

## 3. 用 `tcgen05.mma / cp / shift / ld / st` `(PTX)` 看“属于 pairing / 不属于 pairing”

### 3.1 属于 pipelined pairing 的例子

下面这些是“同线程、属于 pipeline pairing”的典型形式：

| 前后关系 | 是否属于 pairing | 这里靠什么顺序 |
|---|---|---|
| `tcgen05.mma -> tcgen05.mma` `(PTX)` | 是 | ISA 对该配对特批的 issue-order 保证 |
| `tcgen05.cp -> tcgen05.mma` `(PTX)` | 是 | ISA 对该配对特批的 issue-order 保证 |
| `tcgen05.shift -> tcgen05.mma` `(PTX)` | 是 | ISA 对该配对特批的 issue-order 保证 |
| `tcgen05.shift -> tcgen05.cp.4x256b` `(PTX)` | 是 | ISA 对该配对特批的 issue-order 保证 |
| `tcgen05.mma -> tcgen05.shift` `(PTX)` | 是 | ISA 对该配对特批的 issue-order 保证 |

这里强调的是：

```text
这些配对说明“这几对的 issue 顺序被 ISA 额外保证”
不是说明“结果已经完成可消费”
```

### 3.2 不属于 pipelined pairing 的例子

下面这些就不能只靠 pairing 去理解：

| 前后关系 | 是否属于 pairing | 当前主要缺什么 | 为什么会缺 |
|---|---|---|---|
| `tcgen05.st -> tcgen05.ld` `(PTX)` | 否 | `st/ld` 这一路的 completion observation；通常看 `tcgen05.wait::st` `(PTX)` / `tcgen05.wait::ld` `(PTX)` | 前面的 `tcgen05.st` `(PTX)` 是异步推进的，后面的 `tcgen05.ld` `(PTX)` 不能只凭“写在后面”就假定前面的 store side 已经完成 |
| `tcgen05.mma -> tcgen05.ld` `(PTX)` | 否 | `mma` 这一路的 completion observation；通常看 `tc_gen5_commit` `(TTGIR/TritonNvidiaGPU)` -> `wait_barrier` `(TTGIR/TritonNvidiaGPU)` | 前面的 `tcgen05.mma` `(PTX)` 是异步计算，结果落到 TMEM 后还需要先把“已完成”变成可观察事实，后面的 `tcgen05.ld` `(PTX)` 才能安全接上 |
| `tcgen05.cp -> tcgen05.ld` `(PTX)` | 否 | completion observation | `tcgen05.cp` `(PTX)` 不是 pairing 已兜住的“前后天然可直接消费”场景，后面的 `tcgen05.ld` `(PTX)` 还需要先确认前面的异步复制/准备动作真的结束 |
| `tcgen05.mma -> thread-sync -> 别的线程继续 tcgen05` `(PTX)` | 否 | 跨线程 handoff ordering；通常看 `tcgen05.fence::before_thread_sync` `(PTX)` / `tcgen05.fence::after_thread_sync` `(PTX)` | 这里问题已经不是“本线程后面能不能接着用”，而是异步 `tcgen05` 流水怎样和 thread-sync / 另一个线程的后续执行合法衔接 |

所以 `tcgen05.ld / st` `(PTX)` 出现时，通常不要优先问“它和前面是不是 pairing”，而要优先问：

```text
我现在缺的是 completion，
还是缺跨线程 handoff 的衔接？
```

### 3.3 最短判断树

```text
先问：是不是同线程的 pipeline pairing？
  是 -> 该配对的 issue-order 已由 ISA 特批保证
  否 -> 再问：我缺的是 completion，还是跨线程 handoff？

缺 completion
  -> 看 `wait::ld/st` `(PTX)` 或 `commit + mbarrier/wait_barrier`

缺跨线程 handoff
  -> 看 `tcgen05.fence::before_thread_sync` `(PTX)` / `tcgen05.fence::after_thread_sync` `(PTX)`
```

### 3.4 为什么这里会出现两套 completion 模型

这件事最容易被误解成：

```text
是不是 tcgen05 只是提供了两种风格不同、效果差不多的 wait 写法？
```

不是。

更准确的说法是：

```text
不同 tcgen05 异步指令族
本来就被 ISA 设计成走不同的 completion interface
```

先分两路看：

| 前面发的异步指令 | 后面怎么等完成 | 这一路的 completion 形态 |
|---|---|---|
| `tcgen05.ld` / `tcgen05.st` `(PTX)` | `tcgen05.wait::ld` / `tcgen05.wait::st` `(PTX)` | wait-based completion |
| `tcgen05.mma` / `tcgen05.cp` / `tcgen05.shift` `(PTX)` | `commit + mbarrier wait`；在当前文里常见为 `tc_gen5_commit` `(TTGIR/TritonNvidiaGPU)` -> `wait_barrier` `(TTGIR/TritonNvidiaGPU)` | mbarrier-based completion |

所以这两套不是“二选一的语法糖”，而是：

```text
前面的 producer 类型不同
后面的 completion interface 也不同
```

#### A. 为什么 `ld/st` 走 `wait::ld/st`

`ld/st` 这一路更像：

```text
某个具体的 load/store 类异步动作
后面由 issuing thread 直接等这类动作完成
```

它更偏向：

- action-local completion
- issuing thread 自己确认这类 `ld` / `st` 已经结束

所以 ISA 直接给了对应的：

- `tcgen05.wait::ld` `(PTX)`
- `tcgen05.wait::st` `(PTX)`

这里你关心的重点是：

```text
前面的 ld/st 这类动作本身结束了没
```

#### B. 为什么 `mma/cp/shift` 走 `commit + mbarrier wait`

`mma/cp/shift` 这一路更像：

```text
前面发起了一批异步计算/复制/流水动作
后面不仅要知道“完成了没”
还要把“已完成”这件事变成一个可共享观察的事实
```

这一路的特点是：

- 前面的 producer 更像 pipeline / compute action
- 后面的 observer 不一定非得是“刚才发起它的那个程序点”
- 完成事实更适合挂到一个 completion object 上统一观察

所以它走的是：

```text
先 commit
再把 completion 链到 mbarrier
最后由 observer wait 这个 object
```

在当前文里更常见的写法是：

- `tc_gen5_commit` `(TTGIR/TritonNvidiaGPU)`
- `wait_barrier` `(TTGIR/TritonNvidiaGPU)`

如果往 PTX 概念上压缩，就是：

```text
mma/cp/shift 的完成
被折回到 mbarrier completion object
然后再由 wait 去观察
```

#### C. 所以“为什么会有两种 completion 模型”

根因不是“文法上想多给两种写法”，而是：

```text
异步 producer 的类型不同
+ 后续观察完成的形态不同
= ISA / target 暴露出两种不同的 completion interface
```

可以把它压成一句：

```text
ld/st 这一路更像“直接等这类 load/store 动作结束”
mma/cp/shift 这一路更像“把完成事实挂到 mbarrier 上共享观察”
```

#### D. 最后一句最重要

所以它们不是可随意互换的两个写法，而是：

```text
wait::ld/st
= 给 ld/st 这一族准备的 completion interface

commit + mbarrier wait
= 给 mma/cp/shift 这一族准备的 completion interface
```

如果前面发的是哪一类 async producer，后面就要走对应的 completion 模型。

---

## 4. `completion` 是什么

`completion` 只回答一件事：

```text
一个异步动作现在是不是真的做完了
```

它不等于：

- execution rendezvous
- visibility fence
- “程序写在后面，所以前面肯定已经好了”

例如：

- `cp.async` `(PTX)` issue 了，不等于 tile 已经能安全读
- `wgmma.mma_async` `(PTX)` 发出去了，不等于结果已经 retire
- `tcgen05.mma` `(PTX)` 写在前面，不等于 `tcgen05.ld` `(PTX)` 现在就能直接接

所以 completion 关注的是：

```text
issue
-> 硬件独立推进
-> 某种完成事实出现
-> observer 去 wait / poll / observe
```

---

## 5. 确实存在多种 completion model

不是所有 target 都“统一用一种 wait”。更准确的说法是：

```text
不同异步引擎，会暴露不同形态的 completion interface
```

### 5.1 `sm80`: per-thread async-group completion

核心形态：

```text
`cp.async` `(PTX)`
-> `commit_group` `(PTX)`
-> `wait_group` `(PTX)`
```

这里的 `group` 不是线程组，而是一批 `cp.async` 组成的 completion batch。

它回答的是：

```text
本线程自己发出的这批 async copy
是不是已经完成到可以继续消费的程度
```

### 5.2 `sm90` WGMMA: queue-count completion

核心形态：

```text
`wgmma.mma_async` `(PTX)`
-> `wgmma.commit_group` `(PTX)`
-> `wgmma.wait_group N` `(PTX)`
```

这里 `wait_group` 等的不是某个 barrier object，而是：

```text
outstanding committed WGMMA groups 的数量
```

所以它是：

```text
queue-count completion
```

### 5.3 `sm90` TMA / `sm100` tcgen05: object-based completion

核心形态：

```text
异步动作
-> completion 记到账到 `mbarrier` `(PTX object model)`
-> observer 执行 `mbarrier.wait` `(PTX)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)`
```

这里等的是一个 **completion object**，不是内部 group 计数。

所以它是：

```text
object-based completion
```

### 5.4 最短对照

| completion model | 它在等什么 | 典型原语 |
|---|---|---|
| per-thread async-group | 本线程某批 async copy 是否完成 | `cp.async.commit_group -> wait_group` `(PTX)` |
| queue-count completion | outstanding groups 是否降到阈值 | `wgmma.commit_group -> wait_group` `(PTX)` |
| object-based completion | 一个 completion object 是否 ready | `mbarrier.wait` `(PTX)`、`wait_barrier` `(TTGIR/TritonNvidiaGPU)` |

---

## 6. 六个层级：分别通常要补什么

这里不要死记“每一层都有一套固定原语”，而要记：

```text
scope 变大之后，
同线程天然拥有的顺序越来越不够，
就必须引入更显式的 protocol
```

### 6.1 更细的范围表

| 层级 | 先问什么 | 为什么不够 | 通常还要补什么 | 典型机制类型 |
|---|---|---|---|---|
| 本线程继续消费 | 前一个异步动作完成了吗 | 同线程顺序最多只说明“先 issue、后 issue”，不说明前一个异步动作现在已经真的做完 | completion | `wait_group` `(PTX family)`、`wait::ld/st` `(PTX)`、`mbarrier.wait` `(PTX)`、`wait_barrier` `(TTGIR/TritonNvidiaGPU)` |
| 跨 thread | 别的 thread 现在能合法接手吗 | 同线程里已经成立的顺序，不会自动扩展成另一个 thread 的 handoff 规则 | completion + handoff ordering；必要时 thread rendezvous | `tcgen05.fence::before_thread_sync` `(PTX)` / `after_thread_sync` `(PTX)`、`bar.warp.sync` `(PTX)` |
| 跨 warp | 别的 warp 是否已看到结果，且双方是否已对齐 | 另一个 warp 既不自动共享本线程的 completion 观察结果，也不自动和当前 warp 到齐 | completion + CTA 内 rendezvous + visibility/publication | `ttg.barrier` `(TTGIR/TritonGPU)` / `bar.sync` `(PTX)`、`mbarrier` `(PTX object model)`、必要时 `fence.proxy.async` `(PTX)` |
| 跨 warp-group | 不同角色的 warp-group 怎样交接 stage / buffer / ownership | 这里常常已经不是“本地前后顺序”问题，而是不同 actor 之间的角色交接和共享观察问题 | shared completion object + handoff protocol + visibility | `mbarrier` `(PTX object model)`、CTA barrier、warp-specialized handoff |
| 跨 CTA | peer CTA 能否观察到发布结果 | CTA 边界之外不会自动继承本 CTA 内的顺序、可见性和对象生命周期状态 | publication/visibility + cross-CTA rendezvous 或 completion object | cluster barrier `(PTX family)`、cluster-scoped `mbarrier` `(PTX object model)`、release/acquire、生命周期 fence |
| 跨 cluster | 是否已经超出 cluster-local 协作边界 | cluster-local 协议本来就只对 cluster 内成立，超出这个边界后必须切换到更高 scope 的发布/同步模型 | 更高 scope 的 publication / ordering / global synchronization | `.gpu` / `.sys` scope ordering、kernel boundary、runtime-level sync |

### 6.2 怎么读这 6 层

1. 本线程继续消费  
   重点先看 `ordering` 和 `completion`。  
   `pairing` 最多只帮你回答前者的一部分。

2. 跨 thread  
   这里已经进入 handoff。  
   即使 issuing 顺序已定，也还要问 completion 和交接点前后的 ordering。

3. 跨 warp  
   常见问题已经不是“我自己后面能不能接着跑”，而是“另一个 warp 是否能安全读/复用”。

4. 跨 warp-group  
   常常还伴随角色分工变化：producer、consumer、observer 不再是同一个 actor。

5. 跨 CTA  
   本地 CTA 内的 barrier 已经不够，重点转向 publication 给 peer CTA。

6. 跨 cluster  
   很多 cluster-local 协议都不再适用，协议边界上升到 `.gpu` / `.sys` 或 kernel 级别。

### 6.3 最短判断顺序

```text
1. 这是本线程自己继续用，还是要交给别人？
2. 如果只是本线程，缺的是 pairing/order，还是 completion？
3. 如果要交给别人，交接范围到哪一层：
   thread / warp / warp-group / CTA / cluster ?
4. 这个范围内，是否已经有 rendezvous？
5. completion 是否已经被共享观察？
6. visibility / publication 是否已经建立？
```

压成一句就是：

```text
本线程先看 ordering + completion
跨作用域再加 handoff + rendezvous + visibility
```

---

## 7. 三代 target 具体化：6 个层级里最典型会出现什么组合

这一节把上面的“抽象层级表”落到三代最常见协议上：

- `sm80` 的 `cp.async`
- `sm90` 的 `TMA + WGMMA`
- `sm100` 的 `tcgen05 + TMEM`

不是说每层都会稳定出现一模一样的固定原语，而是说：

```text
如果这个 target 在这一层真的发生 handoff，
最典型会由哪套 completion / rendezvous / handoff 机制来兜底
```

### 7.1 `sm80`: `cp.async`

| 层级 | 最典型组合 | 说明 |
|---|---|---|
| 本线程继续消费 | `cp.async -> commit_group -> wait_group` `(PTX)` | 先把本线程发出的 async copy 封成 completion batch，再等它完成到可消费 |
| 跨 thread | 较少单独强调；通常直接上升为 warp/CTA 协作 | `cp.async` 的 completion 是 per-thread 的，单独 thread-to-thread handoff 不是主叙事 |
| 跨 warp | `wait_group` `(PTX)` + `ttg.barrier local` `(TTGIR/TritonGPU)` / `bar.sync` `(PTX)` | 先解决 per-thread completion，再把 tile 升成 CTA 内可共享消费 |
| 跨 warp-group | `sm80` 没有像 `sm90/sm100` 那样突出的 warp-group 异步计算协议 | 如果有协作，通常仍回到 CTA barrier 语义 |
| 跨 CTA | 不是 `cp.async` 主路径 | 更常见的是 kernel 边界或更高层同步，不是本地 async copy protocol 本体 |
| 跨 cluster | `sm80` 无 cluster 协议主角化 | 不属于这条主协议的核心场景 |

最短总结：

```text
sm80 的核心是
per-thread completion
-> 再用 CTA barrier 升成共享消费
```

### 7.2 `sm90`: `TMA + WGMMA`

`sm90` 最重要的是不要把搬运线和计算线混成一条：

- 搬运线：`TMA issue` `(TTGIR/TritonNvidiaGPU or PTX cp.async.bulk.tensor family)` -> `mbarrier.wait` `(PTX)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)`
- 计算线：`wgmma.commit_group -> wgmma.wait_group` `(PTX)`

#### A. 搬运线：`TMA`

| 层级 | 最典型组合 | 说明 |
|---|---|---|
| 本线程继续消费 | issue TMA 后继续执行，后面由 observer 做 `mbarrier.wait` `(PTX)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)` | issue 本身不阻塞，完成通过 completion object 观察 |
| 跨 thread | 一个 thread/warp 发起，另一个 observer thread/warp 等 `mbarrier.wait` `(PTX)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)` | TMA 很典型地把 issue 和 observe 解耦 |
| 跨 warp | `mbarrier.wait` `(PTX)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)` + 必要时 CTA rendezvous | 某个 warp 知道“搬完了”不等于整个 CTA 已经能安全复用 |
| 跨 warp-group | producer / consumer warp-group 通过 `mbarrier` `(PTX object model)` 共享 completion fact | 这是 `sm90` warp-specialized handoff 的常见形态 |
| 跨 CTA | cluster-scoped `mbarrier` `(PTX object model)`、cluster barrier `(PTX family)`、生命周期 fence | 如果 TMA / barrier object 生命周期已经跨 CTA，就要进 cluster 协议 |
| 跨 cluster | 不属于 TMA 本地主协议主场 | 要转向更高 scope 的 ordering / synchronization |

#### B. 计算线：`WGMMA`

| 层级 | 最典型组合 | 说明 |
|---|---|---|
| 本线程继续消费 | `wgmma.mma_async -> commit_group -> wait_group` `(PTX)` | 更准确地说是同 warp-group actor 自己发起、自己回收 |
| 跨 thread | 不是 `WGMMA` 主叙事 | 计算完成主要仍由 issuing warp-group 自己看 outstanding groups |
| 跨 warp | 通常通过 CTA 内同步边界把后续使用对齐 | `wait_group` 解决的是计算 completion，不自动替代 CTA 共享交接 |
| 跨 warp-group | 计算 actor 常是一个 warp-group；若结果/阶段交给别的 warp-group，要再接 CTA handoff | `wgmma.wait_group` `(PTX)` 不等于跨 warp-group completion object |
| 跨 CTA | 不是 `WGMMA` 主路径 | 如果跨 CTA，则要借别的 publication / rendezvous 机制 |
| 跨 cluster | 不属于这条计算 completion 链的直接能力范围 | 要转向 cluster/global scope 机制 |

最短总结：

```text
sm90 不只有一种 completion：
TMA 看 `mbarrier` `(PTX object model)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)`
WGMMA 看 outstanding-group `wait_group` `(PTX)`
```

### 7.3 `sm100`: `tcgen05 + TMEM`

`sm100` 的重点是：

- 某些同线程顺序可以靠 `pairing`
- 但 completion 不能只看顺序
- 计算完成经常要被外化到 `mbarrier`
- TMEM reuse 还会额外引入 hazard 边界

| 层级 | 最典型组合 | 说明 |
|---|---|---|
| 本线程继续消费 | pairing 或 `wait::ld/st` `(PTX)`，以及 `tc_gen5_commit -> wait_barrier` `(TTGIR/TritonNvidiaGPU)` | 先分清是 `ld/st` 这一路 wait-based completion，还是 `mma/cp/shift` 这一路 mbarrier-based completion |
| 跨 thread | `tcgen05.fence::before_thread_sync` `(PTX)` / `after_thread_sync` `(PTX)` + completion | 典型问题是怎样把异步 tcgen05 流水接到 thread-sync / execution-ordering 点 |
| 跨 warp | `wait_barrier` `(TTGIR/TritonNvidiaGPU)` + CTA 内 barrier + 必要的 TMEM hazard 隔离 | 一个 warp 看到 completion 不等于另一个 warp 现在就能安全复用同一结果/资源 |
| 跨 warp-group | producer / observer / consumer warp-group 共享 `mbarrier` `(PTX object model)` completion fact | `sm100` 更容易出现“发起计算、观察完成、消费结果”不是同一个 actor |
| 跨 CTA | cluster barrier `(PTX family)`、cluster-scoped completion、`fence.mbarrier_init.release.cluster` `(PTX)` | barrier object 初始化和发布如果跨 CTA，就要显式发布给 peer CTA |
| 跨 cluster | 不属于 tcgen05/TMEM 本地 protocol 的闭环 | 这里要上升到更高 scope 的 ordering / runtime 同步 |

再补一句和 `TMEM` 相关的特殊点：

```text
tcgen05 completion 已经满足
!= TMEM reuse hazard 自动消失
```

所以 `sm100` 常常还会叠加：

- `completion` 解决“异步计算是不是做完了”
- `TMemBarrierInsertion` `(compiler pass)` / barrier repair 解决“TMEM 使用边界是不是合法”

### 7.4 三代对照压缩版

| target | 本线程最常先看什么 | 跨 CTA 前最常靠什么把结果升成共享事实 | 最容易混淆的点 |
|---|---|---|---|
| `sm80` | `cp.async` `(PTX)` 的 per-thread completion | `ttg.barrier local` `(TTGIR/TritonGPU)` / `bar.sync` `(PTX)` | `wait_group` `(PTX)` 不等于 CTA 已同步 |
| `sm90` | TMA 看 `mbarrier` `(PTX object model)` / `wait_barrier` `(TTGIR/TritonNvidiaGPU)`，WGMMA 看 `wait_group` `(PTX)` | `mbarrier`、CTA barrier、必要时 proxy fence | 搬运完成和计算完成不是同一种 completion |
| `sm100` | pairing 只管顺序；completion 还要看 `wait::ld/st` `(PTX)` 或 `wait_barrier` `(TTGIR/TritonNvidiaGPU)` | `mbarrier`、CTA/cluster handoff、必要时 TMEM hazard barrier | completion、handoff、TMEM hazard 是三件事 |

---

## 8. 把常见同步原语按这套模型重新归类

这一节把 `2026-07-02-barriers-and-fences.md` 里出现过的主要同步原语，直接按当前这篇的模型重排：

```text
这个原语在哪个 scope 用？
它补的是什么缺口？
为什么这个缺口会出现？
它是怎么补上的？
补完之后还缺不缺别的机制？
```

这里故意不把 `async_tma_copy_*`、`wgmma.mma_async`、`tcgen05.mma` 这类
**producer issue op** 也混进来，因为它们回答的是“谁发起异步工作”，不是“同步缺口怎么被补”。

### 8.0 记号约定：这里的斜杠只表示跨层对应

这一节里经常会看到：

```text
A / B
```

从这里开始，斜杠 `/` 只保留一种含义：

| 情况 | 左边 | 右边 | 应该怎么读 |
|---|---|---|---|
| 跨层对应 | 较高层 IR / compiler 名字 | 更低层 PTX/CUDA 名字 | 这是上下层映射 |

跨层对应保留斜杠写法，例如：

- `ttg.barrier (TTGIR/TritonGPU) / bar.sync 0 (PTX)`
- `init_barrier (TTGIR/TritonNvidiaGPU) / mbarrier.init (PTX)`
- `bar.warp.sync (PTX) / __syncwarp() (CUDA)`

同层 sibling op 不再用斜杠并列，而是拆成单独列或单独行，避免和跨层对应混在一起。

### 8.1 execution rendezvous：缺的是“谁必须先到齐”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `ttg.barrier` `(TTGIR/TritonGPU)` / `bar.sync 0` `(PTX)` | CTA | 缺 CTA 内 execution rendezvous | warp 独立调度，shared memory producer/consumer 不是天然锁步 | 让整个 CTA 会合后再继续 | 如果前面还有 async completion 没观察完，仍要先配 `wait_group` / `wait_barrier` |
| `bar.sync N, cnt` `(PTX)` | CTA 子集 | 缺 sub-CTA rendezvous | 只是一部分线程协作，不想把整个 CTA 都停下 | 只同步该 barrier 上注册的线程子集 | 若这批线程还跨 proxy / async completion，仍要另配 fence 或 wait |
| `bar.warp.sync` `(PTX)` / `__syncwarp()` `(CUDA)` | warp | 缺 warp 内重收敛 | 同一 warp 内 lanes 也可能因控制流分歧失去收敛 | 只让一个 warp 的 32 lanes 对齐 | 不替代 CTA barrier，更不替代 completion |
| `barrier.cluster.arrive` `(PTX)` | cluster | 缺多个 CTA 之间的 rendezvous | 会合对象已经不是 CTA 内线程，而是 peer CTA | 先向 cluster 会合点报到 | 它不替代 `barrier.cluster.wait`；对象初始化若跨 CTA 仍要 `fence.mbarrier_init.release.cluster` |
| `barrier.cluster.wait` `(PTX)` | cluster | 缺多个 CTA 之间的 rendezvous | 只有 arrive 不足以让本 CTA 等到 peer CTA 真正到齐 | 在 cluster 会合点等待所有参与 CTA 对齐 | 不替代 mbarrier init publication；对象初始化若跨 CTA 仍要 `fence.mbarrier_init.release.cluster` |

### 8.2 shared completion object：缺的是“异步完成怎么变成共享可观察事实”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `init_barrier` `(TTGIR/TritonNvidiaGPU)` / `mbarrier.init` `(PTX)` | thread 执行，效果到 CTA/cluster object | 缺 completion object 生命周期起点 | shared memory 里一开始只是普通 bytes，不是可用 barrier object | 把该 smem 槽位初始化成 mbarrier | 若该对象要给 peer CTA 用，还要 `fence.mbarrier_init.release.cluster` |
| `inval_barrier` `(TTGIR/TritonNvidiaGPU)` / `mbarrier.inval` `(PTX)` | CTA/cluster object lifecycle | 缺对象退役 / reuse 闭环 | 不退役就可能把旧 phase / 旧状态带进下一轮 | 明确结束本轮协议，允许后续复用 | 它只是 lifecycle closure，不替代 completion wait |
| `barrier_expect(bytes)` `(TTGIR/TritonNvidiaGPU)` / `mbarrier.expect_tx` `(PTX)` | CTA/cluster object | 缺 completion condition 定义 | consumer 若不知道“等什么”，wait 就没有判定条件 | 先声明本轮预期 bytes / tx 条件 | 还要有真实 producer 去把完成记到账 |
| `arrive_barrier` `(TTGIR/TritonNvidiaGPU)` / `mbarrier.arrive` `(PTX)` | CTA/cluster object | 缺 arrival bookkeeping | 有些协议不仅等 bytes，还要等 participant arrival | 往 object 里记一次 arrival / count | 不是 wait，不负责阻塞 consumer |
| `wait_barrier` `(TTGIR/TritonNvidiaGPU)` / `mbarrier.try_wait` `(PTX)` | whoever waits | 缺 completion observation | 异步引擎完成与 SM 执行解耦；“已发起”不等于“已完成” | 在 phase/ready 条件满足前阻塞或轮询 | 只回答完成，不回答 execution rendezvous 或 proxy visibility |
| `async_copy_mbarrier_arrive` | producer thread/warp -> CTA object | 缺“非 bulk async copy 完成如何共享观察” | 非 bulk async copy 的完成原本只在引擎/本线程内部 | 把该类 async copy completion 链接到 mbarrier | 之后仍由 `wait_barrier` 去观察 |
| `tc_gen5_commit -> wait_barrier` | warp-group / cta_group + CTA object | 缺“tcgen05 完成如何被共享观察” | `tcgen05` 结果落在 TMEM，后续 observer/consumer 不一定是同一个 actor | `tc_gen5_commit` 先把 completion 折回 mbarrier，再由 `wait_barrier` 观察 | TMEM hazard 仍可能要额外 `ttg.barrier local` 修补 |
| `clc_try_cancel -> wait_barrier` | producer + CTA object | 缺“CLC result buffer 完成如何共享观察” | 结果写完这件事需要变成共享 completion fact | 先把完成挂到 mbarrier，再 wait | 只是 completion，不替代其它顺序边界 |

### 8.3 actor-local completion：缺的是“本 actor 自己发出的异步批次是否完成”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `cp.async.commit_group` | executing thread | 缺 waitable batch 边界 | 一串 `cp.async` 只是连续 issue；硬件需要知道哪一批要一起等 | 把当前已 issue 的 copy 封成一个 group | 还要 `cp.async.wait_group` 去真正观察 completion |
| `cp.async.wait_group` | executing thread | 缺 per-thread completion observation | `cp.async` 异步推进；本线程后续读 smem 前必须先确认完成 | 等到本线程 outstanding cp.async groups 满足阈值 | 若 tile 要给别的线程/warp 用，通常还要 `ttg.barrier local` `(TTGIR/TritonGPU)` / `bar.sync` `(PTX)` |
| `wgmma.fence` | warp-group | 缺进入 WGMMA 计算管线前的输入顺序边界 | shared/generic 一侧准备好的输入，不会自动以 WGMMA 需要的顺序进入 async MMA pipeline | 在 WGMMA issue 前建立该计算协议要求的输入顺序 | 若前面还是 generic->async proxy 边界，往往还要先有 `fence_async_shared` |
| `wgmma.commit_group` | warp-group | 缺 waitable MMA batch 边界 | 一串 `wgmma.mma_async` 也只是连续 issue；需要 group bookkeeping | 把当前 WGMMA issue 封成一个 completion batch | 还要 `wgmma.wait_group` 去等 outstanding groups 下降 |
| `wgmma.wait_group` | warp-group | 缺 compute completion observation | WGMMA 完成模型不是 mbarrier object，而是 outstanding-group 计数 | 等到 committed groups 数量降到目标阈值 | 这只解决计算 completion；跨 warp/CTA 交接还要 CTA barrier 或别的 handoff |
| `tcgen05.wait::ld` `(PTX)` | issuing thread | 缺 `ld` 这一路的 completion observation | `tcgen05.ld` 不属于前面那些 pipelined pairing 就绪即用的场景 | 用 wait-based completion 明确等 `ld` 完成 | 不是 mbarrier-based 那一路，不能和 `mma/cp/shift` 的 wait 随意互换 |
| `tcgen05.wait::st` `(PTX)` | issuing thread | 缺 `st` 这一路的 completion observation | `tcgen05.st` 不属于前面那些 pipelined pairing 就绪即用的场景 | 用 wait-based completion 明确等 `st` 完成 | 不是 mbarrier-based 那一路，不能和 `mma/cp/shift` 的 wait 随意互换 |

### 8.4 visibility / ordering fence：缺的是“前面的写还没有被合法发布”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `fence_async_shared` `(TTGIR/TritonNvidiaGPU)` / `fence.proxy.async.shared::{cta,cluster}` `(PTX)` | thread issue，scope = CTA/cluster | 缺 generic proxy -> async proxy ordering / visibility | shared memory 的 generic 视图和 async 视图不是同一个可见性世界 | 把前面的 generic 写发布给后续 async reader | 它不等待硬件完成；completion 仍要靠 `wait_group` / `wait_barrier` |
| `tensormap_fenceproxy_acquire` | GPU | 缺 descriptor / tensormap metadata 对 TMA 单元的可见性 | metadata 虽然也是 data，但它面向的是 TMA/tensormap consumer 的单独读取路径 | 用 acquire fence 保证 descriptor 写先于 TMA 读 | 它只补 metadata visibility，不观察 TMA completion |
| `fence.mbarrier_init.release.cluster` | cluster | 缺 mbarrier object 初始化对 peer CTA 的 publication | barrier object 在一个 CTA 初始化后，peer CTA 不会自动看到“已初始化”这个事实 | 用 release fence 把 init 后状态发布给 cluster 内其它 CTA | 它只保证对象初始化可见，不回答后续 async work 是否完成 |

### 8.4A 都叫 `fence`，但它们补的不是同一种顺序

最容易混的地方，是看到名字里都有 `fence`，就把它们都理解成同一类机制。其实不是。

更稳的读法是直接问：

```text
这个 fence 到底在补哪一种顺序？
```

| 原语 | 层级 | 它补的不是哪种顺序 | 它真正补的是哪种顺序 | 典型场景 |
|---|---|---|---|---|
| `fence.proxy.async` `(PTX)` | PTX | 不是 completion，不是 rendezvous | `generic proxy -> async proxy` 的 visibility / ordering | 普通线程先写 shared memory，后面 TMA / WGMMA / async side 要读 |
| `tensormap_fenceproxy_acquire` `(PTX)` | PTX | 不是数据搬运完成，不是 CTA 同步 | descriptor / TensorMap metadata 对 TMA consumer 的 visibility / acquire ordering | 更新 descriptor 后，让 TMA 单元看到新 metadata |
| `fence.mbarrier_init.release.cluster` `(PTX)` | PTX | 不是 async work completion，不是 cluster barrier 本身 | mbarrier object 初始化结果对 peer CTA 的 publication / release ordering | 一个 CTA 初始化 mbarrier，cluster 内别的 CTA 后面要 arrive / wait 它 |
| `tcgen05.fence::before_thread_sync` `(PTX)` | PTX | 不是 generic↔async proxy fence，不是 completion wait | 异步 `tcgen05` 流水到后续 thread-sync / execution-ordering 点之前的 handoff ordering | 先有 async tcgen05，再接线程同步点 |
| `tcgen05.fence::after_thread_sync` `(PTX)` | PTX | 不是 generic↔async proxy fence，不是 completion wait | 前序 thread-sync / execution-ordering 点到后续 async `tcgen05` 流水之间的 handoff ordering | 先线程同步，再继续发 async tcgen05 |

压成一句记忆：

- `fence.proxy.async`：管不同 proxy 的“看见没看见”
- `tensormap_fenceproxy_acquire`：管 metadata 被 TMA 看见没看见
- `fence.mbarrier_init.release.cluster`：管 barrier object 初始化结果有没有发布给别的 CTA
- `tcgen05.fence::before/after_thread_sync`：管 async `tcgen05` 流水和 thread-sync 怎么接上

再压成最短判断法：

```text
如果你在问：
“另一个视图 / 另一个引擎看不看得见？”
  -> 多半是 proxy / acquire / release fence

如果你在问：
“async tcgen05 流水怎么和 thread-sync 接起来？”
  -> 多半是 tcgen05.fence::before / after_thread_sync

如果你在问：
“任务做完了没？”
  -> 这已经不是 fence 了，要看 wait / mbarrier / wait_group
```

### 8.5 specialized inter-thread handoff：缺的是“异步 tcgen05 流水怎样接到 thread-sync 上”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `tcgen05.fence::before_thread_sync` | thread -> 后续 thread-sync / execution-ordering 点 | 缺 tcgen05 async pipeline 到 thread-sync 前的 handoff ordering | `tcgen05` 异步流水的顺序，不会自动接到后面的线程同步点上 | 让前面的异步 tcgen05 排在后续 thread-sync / execution-ordering 之前 | 它不是 completion wait；若结果还没完成，仍要 completion 机制 |
| `tcgen05.fence::after_thread_sync` | 前序 thread-sync / execution-ordering 点 -> 后续 thread | 缺 thread-sync 到后续 tcgen05 async pipeline 的 handoff ordering | 前面的线程同步点，不会自动约束后续 tcgen05 异步流水怎样接上 | 让后续异步 tcgen05 排在前面的 thread-sync / execution-ordering 之后 | 它不是 generic proxy fence，也不替代 `wait::ld/st` / `wait_barrier` |

### 8.6 compiler-inserted repair：缺的是“completion 有了，但 legality 还没闭环”

| 原语 | 典型 scope | 缺了什么 | 为什么会缺 | 它怎么补 | 补完后还缺什么 |
|---|---|---|---|---|---|
| `MembarAnalysis` 插的 `ttg.barrier local` | CTA | 缺 shared memory 数据 hazard 隔离 | CTA 内 shared 复用会出现 RAW/WAR/WAW；源码未必显式写 barrier | 自动在合适位置补 `ttg.barrier local` `(TTGIR/TritonGPU)` / `bar.sync` `(PTX)` | 它不观察异步完成；若前面是 async op，仍尽量推到 wait 后 |
| `TMemBarrierInsertion` 插的 `ttg.barrier local` | CTA | 缺 TMEM use-edge legality | `load->mma` / `store->mma` 等 TMEM 边界即使 completion 成立，也可能仍有 use hazard | 自动在这些 use-edge 前插 CTA barrier | 它隔离的是 TMEM hazard，不是 tcgen05 completion 本体 |
| `FenceInsertion` / `ProxyFenceInsertion` 插的 `fence_async_shared` | CTA/cluster | 缺 generic↔async proxy ordering | 前一个 proxy 的写，对后一个 proxy 还不可见 | 自动在 pattern / alias 分析认定需要时补 `fence.proxy.async` | 仍不替代 completion wait，也不替代 execution barrier |

### 8.7 一句话速记

```text
ttg.barrier `(TTGIR/TritonGPU)` / bar.sync `(PTX)` / cluster barrier `(PTX family)`
  = 缺 execution rendezvous

mbarrier.* / wait_barrier / tc_gen5_commit
  = 缺 shared completion observation

cp.async.wait_group / wgmma.wait_group / tcgen05.wait::ld/st
  = 缺 issuing actor 自己的 completion observation

fence.proxy.async / tensormap_fenceproxy_acquire / fence.mbarrier_init.release.cluster
  = 缺 visibility / ordering publication

tcgen05.fence::before/after_thread_sync
  = 缺 specialized inter-thread handoff ordering

Membar / TMemBarrierInsertion / ProxyFenceInsertion
  = 缺 legality repair，不是协议本体
```

---

## 9. `ISA 自带` 到底是什么意思
`ISA 自带` 的意思不是“什么都不用管”，而是：

```text
某一类前后关系，
规范已经直接承诺了它需要的那部分 ordering
```

例如某些 pipelined `tcgen05` pairing，重点是：

- 这类顺序不需要你再额外发一个普通 fence 去建立
- 但不表示 completion 自动满足
- 更不表示跨线程 handoff 自动满足

所以如果 `ISA` 不自带，不要本能地说“那就插 fence”。先分清缺口：

| 如果不自带，你缺的可能是 | 该优先看的机制 |
|---|---|
| execution rendezvous | `ttg.barrier`、`bar.sync`、cluster barrier |
| completion observation | `wait_group`、`mbarrier.wait`、`wait_barrier` |
| visibility / publication | `fence.proxy.async`、acquire/release、cluster publication fence |
| specialized inter-thread handoff | `tcgen05.fence::before_thread_sync` / `after_thread_sync` |

所以最重要的不是背哪个指令名，而是别把下面几件事混成一种：

```text
有顺序
!= 已完成
!= 已对别的线程可见
!= 已经完成跨线程交接
```

---

## 10. 最后压成一句最实用的话

看到一个“前后应该有关联”的地方，按下面顺序判断最稳：

```text
1. 这是同线程吗？
2. 如果是，同线程顺序是不是 ISA / 引擎已经自带？
3. 如果不是，或者自带顺序不够，我缺的是 completion 吗？
4. completion 有了以后，还缺 execution rendezvous 吗？
5. rendezvous 有了以后，还缺 visibility / handoff 发布吗？
```

把这 5 问分开，`pairing`、`completion`、`barrier`、`fence`、`handoff` 就不容易再混了。
