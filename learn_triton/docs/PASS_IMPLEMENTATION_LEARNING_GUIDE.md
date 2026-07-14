# TTGIR / TTNGIR Pass 实现细节学习指南

## 1. 目标和边界

这份指南面向已经对 TTGIR、TTNGIR 及其主要 pass 有概念性认识，准备进一步学习实现细节的阶段。

这里不再以“这个 pass 做什么”为主要目标，而是回答：

```text
这个 pass 把编译问题建模成了什么算法问题？
它维护了哪些状态和 invariant？
为什么选择这些数据结构？
它采用了哪种分析、重写或分层设计模式？
如何用最小测试验证自己确实读懂了实现？
```

相关文档的分工是：

- `TTGIR_GUIDE.md`：建立 TTGIR 问题域、抽象边界和 pass 主链
- `IR_PASS_DIFF_LEARNING_GUIDE.md`：从 before / after IR 解释一个具体 pass effect
- 本文：进入 pass 源码，学习算法、数据结构、设计模式和 mutation protocol

## 2. 先修正阅读顺序

不要按照 backend pipeline 从前到后依次阅读源码。

pipeline 顺序表达的是 pass 之间的编译依赖，不是实现难度。例如：

- `Coalesce.cpp` 只有约 125 行，适合学习 analysis 与 mutation 的分离
- `RemoveLayoutConversions.cpp` 超过 1600 行，同时涉及 layout propagation、slice、dominance、rematerialization、拓扑排序和 SCF 重写

因此，实现层的学习顺序应该是：

```text
Pass 骨架
  -> 局部命令式重写
  -> analysis + transform
  -> pattern rewrite / interface
  -> liveness / allocation
  -> graph scheduling / multi-pass protocol
  -> 全局数据流与控制流重写
```

另一个需要明确的边界是：在当前 NVIDIA backend 中，TTGIR 和 NVIDIA 专属 IR 不是两个完全隔离的顺序阶段。

`third_party/nvidia/backend/compiler.py::make_ttgir()` 会交错加入 `ttgpuir` 和 `ttnvgpuir` pass；`make_llir()` 开头还会继续运行 tensor-memory allocation、proxy fence insertion 和 TMEM barrier insertion 等 pass。因此阅读时应追踪“该 pass 建立什么 invariant、谁消费它”，不要只按目录或 dialect 名称判断阶段。

## 3. 推荐学习阶梯

### 3.1 Pass 框架热身：`CheckMatmulTwoCTAs`

源码：

- `lib/Dialect/TritonNvidiaGPU/Transforms/CheckMatmulTwoCTAs.cpp`
- `include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td`

这个 pass 展示了最小但完整的 MLIR pass 骨架：

- TableGen 定义与生成的 pass base class
- `runOnOperation()` 入口
- `ModuleOp::walk()`
- 使用 `MMAv5OpInterface` 遍历一类操作
- 用 `WalkResult::interrupt()` 提前终止
- 发出诊断并调用 `signalPassFailure()`
- 把 module 级一致性结论写成 attribute，供下游读取

学习目标不是复杂算法，而是明确 pass 的生命周期、失败语义和 module invariant。

### 3.2 局部命令式重写：`ReduceDataDuplication`

源码和测试：

- `lib/Dialect/TritonGPU/Transforms/ReduceDataDuplication.cpp`
- `test/TritonGPU/reduce-data-duplication.mlir`

重点观察：

- guard conditions 如何逐层筛选候选 `ConvertLayoutOp`
- 如何区分 legality check 与 profitability check
- 如何构造 `LocalAllocOp` 和 `LocalLoadOp`
- `replaceAllUsesWith()` 与 `erase()` 的更新顺序
- 为什么该变换可以单次 walk，不需要 worklist 或 fixed point

这个 pass 适合练习从 lit 测试的 `RUN`、`CHECK` 和边界样例反推实现。

### 3.3 第一个完整算法 Pass：`Coalesce`

源码和测试：

- `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`
- `lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp`
- `test/TritonGPU/coalesce.mlir`

它的核心决策链是：

```text
ModuleAxisInfoAnalysis
  -> 找出 tensor-of-pointers 内存访问
  -> 读取连续性、shape、warp 数和 target 信息
  -> 为每个访问选择 coalesced encoding
  -> 记录 Operation -> Attribute
  -> 统一重写相关 operand、result 和 layout conversion
```

重点学习：

- analysis first, mutation second
- `ModuleAxisInfoAnalysis` 提供事实，pass 本身负责策略
- `MapVector` 同时提供查找能力和稳定迭代顺序
- target/layout 决策与通用 IR mutation helper 的分工
- 为什么 descriptor load/store 使用独立的 layout 选择规则

`Coalesce` 足够短，又包含真实硬件约束、分析结果和 IR 重写，是最适合作为第一个完整研究对象的 pass。

### 3.4 通用机制与后端策略：`OptimizeDescriptorEncoding`

源码：

- `lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeDescriptorEncoding.cpp`
- `lib/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.cpp`
- `include/triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h`

这里可以观察一种典型的 Template Method / Strategy 设计：

- TTGIR 层拥有通用的 descriptor layout 传播和分配算法
- NVIDIA 子类实现 compatible encoding、candidate search 和 fallback
- `runOnOperation()` 只组装目标策略并调用通用机制

需要回答：

- 哪些规则是 descriptor 的普遍约束？
- 哪些规则来自 NVIDIA TMA / shared-memory encoding？
- 为什么 target-specific 代码不应该直接复制整套传播算法？
- candidate 不存在时，fallback 建立了什么最低保证？

### 3.5 Pattern Rewrite 与接口驱动 Lowering：`MMALowering`

源码和测试：

- `lib/Dialect/TritonNvidiaGPU/Transforms/MMALowering.cpp`
- `test/TritonNvidiaGPU/mma_lowering.mlir`

重点学习：

- `OpRewritePattern` 与 `OpInterfaceRewritePattern`
- `matchAndRewrite()` 中匹配、拒绝和提交的边界
- 为什么通过 `MMAv5OpInterface` 复用多种 MMA op 的 lowering
- `RewritePatternSet` 与 greedy rewrite driver 的职责
- rewrite 前后 token、descriptor、predicate 和依赖关系必须保持的 invariant

先学清这一层，再阅读 `AccelerateMatmul` 中的大量 pattern，会更容易区分框架代码、硬件选择和具体 rewrite。

### 3.6 Liveness 与资源分配：`TensorMemoryAllocation`

源码和测试：

- `lib/Dialect/TritonNvidiaGPU/Transforms/TensorMemoryAllocation.cpp`
- `test/TritonNvidiaGPU/test_tensor_memory_allocation.mlir`

应把它理解为一个小型物理资源分配器，而不是普通 attribute rewrite：

```text
枚举 TMEM allocations
  -> 给 operation 编号
  -> 基于 MLIR Liveness 建立 live interval
  -> 判断 allocation 是否共存
  -> 在二维 TMEM 空间放置 chunk
  -> 对不重叠 lifetime 复用物理位置
  -> 写入 row / column offset 和 module size
```

重点数据结构：

- `DenseMap<Operation *, int>`：operation 到序号
- `DenseSet<Value>`：遍历 use-def 链时去重
- worklist：沿 SSA / branch 关系寻找 allocation 来源
- live interval：把动态 lifetime 压成可比较区间
- allocation chunk 集合：表达空间占用与共存约束

阅读时要分别验证时间维度的可复用性和 TMEM 二维空间维度的合法性。

### 3.7 多 Pass 协议与调度

把下面这组作为一个整体阅读：

```text
AssignLatencies
  -> ScheduleLoops
  -> Pipeline
  -> LowerLoops / PipelineExpander
```

主要源码位于：

- `lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp`
- `lib/Dialect/TritonGPU/Transforms/Pipeliner/LowerLoops.cpp`

这组 pass 的重点不是单个 rewrite，而是 pass 间协议：

- latency 如何被计算并序列化到 IR
- schedule 如何通过依赖、backward slice 和 distance 分配 stage / cluster
- pipeline 如何消费 schedule
- prologue、steady state、epilogue 如何保持循环语义
- 临时 attribute 何时产生、何时被消费、何时应清理

核心数据结构包括 `DenseMap<Operation *, int>`、`DenseSet<Operation *>`、有序 worklist、backward slice、cluster 映射以及 loop-carried value 映射。

### 3.8 最后阅读全局复杂 Pass

最后再进入：

- `RemoveLayoutConversions`
- `PlanCTA`
- `AutomaticWarpSpecialization`
- `PartitionScheduling`
- cluster barrier、TMEM barrier 和 proxy fence insertion

以 `RemoveLayoutConversions` 为例，它同时涉及：

- layout propagation
- use-def slice
- `DominanceInfo` / `PostDominanceInfo`
- rematerialization cache
- profitability 判断
- topological sort
- SCF block argument / yield 重写
- cleanup pattern

它应该作为前述分析和重写技术的综合练习，而不应该作为源码阅读入口。

## 4. 每个 Pass 的固定阅读方法

不要从源文件第一行顺序读到最后。使用下面的顺序：

1. 在 `Passes.td` 中确认 summary、options、operation scope 和 dependent dialects。
2. 在 backend `compiler.py` 中确认它的前驱、后继和 capability 分支。
3. 先读 lit 测试的 `RUN`、正例、反例和边界样例。
4. 从 `runOnOperation()` 开始，画出顶层控制流。
5. 将 helper 分为 analysis、decision、mutation、cleanup 四类。
6. 记录核心数据结构及每个容器维护的 invariant。
7. 检查 downstream consumer，确认输出为什么必须是这种形式。
8. 修改一个最小 lit 输入，先预测结果，再运行验证。

推荐为每个 pass 维护一张算法卡片：

```text
Pass:

Goal:
  它要解决的编译问题是什么？

Input assumptions:
  进入 pass 前必须已经成立什么？

Output invariant:
  pass 成功后新建立了什么事实？

Analysis state:
  算法读取、推导和缓存哪些事实？

Traversal / worklist:
  按什么顺序访问 IR？为什么顺序重要？

Decision rule:
  legality、cost、heuristic 和 fallback 分别是什么？

Data structures:
  每个 map、set、vector、interval、slice 表达什么关系？

IR mutation:
  创建、替换、移动、删除操作的顺序是什么？

Failure / fallback:
  无法变换时是跳过、降级还是终止编译？

Complexity:
  主要复杂度来自 IR walk、图遍历、候选搜索还是 fixed point？

Downstream consumer:
  谁消费这个 pass 的 op、type、layout、attribute 或顺序？
```

## 5. 数据结构学习检查表

不要只记录“代码使用了 `DenseMap`”。要说明它表达的编译关系。

| 数据结构 | 常见语义 | 阅读时要问的问题 |
|---|---|---|
| `SmallVector<T>` | 数量通常较小的有序候选、operand 或 op 集合 | 顺序是否有语义？是否会增长到很大？ |
| `DenseMap<K, V>` | IR 对象到分析结果或重写结果的映射 | key 的 identity 是 Value、Operation 还是 Attribute？缓存何时失效？ |
| `DenseSet<T>` | visited、live、excluded 或 dedup 集合 | 去重是 correctness 要求还是性能优化？ |
| `SetVector<T>` | 同时要求唯一性和确定顺序的 slice / worklist | 插入顺序是否影响 rewrite 或测试稳定性？ |
| `MapVector<K, V>` | 保留发现顺序的分析结果 | 为什么 mutation 需要确定性顺序？ |
| worklist / queue | 图或 use-def 链遍历 | 入队条件、终止条件和重复访问策略是什么？ |
| interval | lifetime 或占用范围 | 区间是否保守？相交意味着什么资源约束？ |
| slice | 某个 root 的依赖子图 | 是 backward slice 还是 forward slice？跨 region 如何处理？ |

## 6. 设计模式学习检查表

实现层最值得积累的是可复用模式，而不是 pass 名称：

| 模式 | 代表 Pass | 核心问题 |
|---|---|---|
| walk + aggregate + validate | `CheckMatmulTwoCTAs` | 如何建立 module 级一致性 invariant |
| guarded imperative rewrite | `ReduceDataDuplication` | 如何安全地局部替换 IR |
| analysis then mutation | `Coalesce` | 如何避免分析结果被中途 mutation 污染 |
| template method / target strategy | `OptimizeDescriptorEncoding` | 如何隔离通用机制与后端约束 |
| rewrite pattern + op interface | `MMALowering` | 如何扩展一族 op 的 lowering |
| liveness + interval allocation | `TensorMemoryAllocation` | 如何把逻辑 lifetime 映射到物理资源 |
| producer / consumer pass protocol | pipeliner pass group | 如何把复杂算法拆成多个可验证阶段 |
| worklist + slice + dominance | `RemoveLayoutConversions` | 如何做跨 use-def 和控制流的全局重写 |

## 7. 四周执行计划

### 第 1 周：建立源码阅读骨架

- `CheckMatmulTwoCTAs`
- `ReduceDataDuplication`
- `Coalesce`

产出：三张算法卡片；至少为 `Coalesce` 手工画出 analysis-to-mutation 数据流。

### 第 2 周：学习可扩展重写设计

- `OptimizeDescriptorEncoding`
- `MMALowering`
- 选择 `AccelerateMatmul` 中一个 pattern 作为追加练习

产出：区分通用机制、NVIDIA 策略、pattern framework 和单条硬件规则。

### 第 3 周：学习资源分析与分配

- `TensorMemoryAllocation`
- 结合测试分析 allocation reuse、冲突和二维布局

产出：为一个测试样例手工计算 live interval、共存关系和预期 offset。

### 第 4 周：学习调度和全局重写

- 重新实现性阅读 `AssignLatencies -> ScheduleLoops -> Pipeline`
- 开始拆解 `RemoveLayoutConversions`，先只读顶层 phase，不一次追完所有 helper

产出：画出 pass 间 contract；为一个 pipelined loop 解释 stage、cluster 和 loop-carried value 的变化。

## 8. 验证标准

“读完源码”不是完成标准。满足下面条件才算真正掌握一个 pass：

1. 不看源码，可以说出 input assumptions、output invariant 和 downstream consumer。
2. 可以画出主要 analysis state 和 mutation 的数据流。
3. 可以解释核心容器为什么是 map、set、有序集合、slice 或 interval。
4. 可以指出至少一个拒绝变换或 fallback 路径。
5. 给出一个新的最小 IR 样例时，可以预测 pass 是否触发以及关键输出。
6. 修改 native/compiler 源码后，能按仓库要求先运行 `make`，再运行对应单个 lit 测试。

最终目标不是记住每个 pass 的所有分支，而是形成一套可迁移的方法：看到新的 Triton / MLIR pass 时，可以迅速恢复它的约束、算法模型、状态表示、重写协议和下游 contract。
