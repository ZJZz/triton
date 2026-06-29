# 2026-06-29 学习笔记：RemoveLayoutConversions

## 1. Pass 基本信息

本轮先学习 matmul canonical dumps 里第一轮生效的
`TritonGPURemoveLayoutConversions`，也就是 `PlanCTA` 之后、`OptimizeThreadLocality`
之前的那次调用。

相关文件：

- Ampere before: [030_Before_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/030_Before_TritonGPURemoveLayoutConversions.mlir:1)
- Ampere after: [031_After_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/031_After_TritonGPURemoveLayoutConversions.mlir:1)
- Hopper before: [028_Before_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:1)
- Hopper after: [029_After_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:1)
- Blackwell before: [028_Before_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:1)
- Blackwell after: [029_After_TritonGPURemoveLayoutConversions.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:1)
- Source: [RemoveLayoutConversions.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42)
- Pass declaration: [Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td:250)
- NVIDIA pipeline wiring: [compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:274)

Effective-pass result:

| Arch | First RemoveLayoutConversions | Second RemoveLayoutConversions | Late RemoveLayoutConversions |
| --- | --- | --- | --- |
| sm86_num_ctas1 | changed, -14 lines | changed, -5 lines | no-op |
| sm90_num_ctas1 | changed, -14 lines | changed, -3 lines | no-op |
| sm100_num_ctas1 | changed, -14 lines | changed, -3 lines | no-op |

本轮重点看 first pass。三代架构这个位置的 IR 基本同形，说明这里的机制主要是
architecture-independent layout cleanup，不是当前 pass 自己引入了架构分叉。

## 2. 一句话结论

`RemoveLayoutConversions` 不是简单删除所有 `ttg.convert_layout`。它先以 load/store/dot
等 layout anchor 为准，把 anchor 希望保留的 encoding 向后传播，然后重写 producer /
consumer 的 tensor type，让大段指针计算、elementwise、loop iter_args 直接活在目标
layout 里；只有在真正的 layout 边界，例如 global-load blocked layout 到 dot-operand
layout、dot accumulator 到 store-friendly blocked layout，才保留 `ttg.convert_layout`。

## 3. IR 变化事实

三代 first pass 的 `ttg.convert_layout` 数量相同：

| Arch | Before | After | Changed? |
| --- | --- | --- | --- |
| sm86 | 19 | 3 | yes, -14 lines |
| sm90 | 19 | 3 | yes, -14 lines |
| sm100 | 19 | 3 | yes, -14 lines |

### 3.1 指针计算里的临时 convert 被消掉

Before 里 A/B/C pointer prologue 充满 layout 跳转，例如 Hopper before：

- Before line
  [38](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:38)
  到
  [53](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:53):

```text
%a_ptrs = ttg.convert_layout %offs_m_2 : tensor<64xi32, #blocked1>
  -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
...
%a_ptrs_17 = ttg.convert_layout %a_ptrs_16
  : tensor<64x32x!tt.ptr<f16>, #blocked3>
  -> tensor<64x32x!tt.ptr<f16>, #blocked5>
```

After 里这些 `convert_layout` 被吸收到 producer 的 result encoding 中：

- After line
  [27](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:27)
  到
  [49](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:49):

```text
%offs_m_0 = tt.make_range ... : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%offs_m_1 = tt.make_range ... : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
...
%a_ptrs_20 = tt.addptr ... : tensor<64x32x!tt.ptr<f16>, #blocked1>
```

IR 层面的意思：不是先用一个 layout 做完 pointer expression，再插 convert；而是复制 /
改写一部分 producer，让它一开始就产生消费者需要的 layout。

### 3.2 loop iter_args 的 layout 被统一到 anchor-friendly layout

Before 的 loop accumulator 仍用旧 blocked layout 进循环，然后 dot 前后各转一次：

- Before line
  [70](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:70)
  到
  [86](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/028_Before_TritonGPURemoveLayoutConversions.mlir:86):

```text
iter_args(..., %acc_46 = %cst) -> (..., tensor<64x64xf32, #blocked>)
%acc_51 = ttg.convert_layout %acc_46
  : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked6>
%acc_52 = tt.dot ... -> tensor<64x64xf32, #blocked6>
%2 = ttg.convert_layout %acc_52
  : tensor<64x64xf32, #blocked6> -> tensor<64x64xf32, #blocked>
scf.yield ..., %2 : ..., tensor<64x64xf32, #blocked>
```

After 里 accumulator 的 loop-carried type 直接变成 dot parent layout：

- After line
  [62](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:62)
  到
  [74](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:74):

```text
%cst = arith.constant ... : tensor<64x64xf32, #blocked>
%acc:3 = scf.for ... iter_args(..., %acc_42 = %cst)
  -> (..., tensor<64x64xf32, #blocked>)
...
%acc_45 = tt.dot ... %acc_42 ... -> tensor<64x64xf32, #blocked>
scf.yield ..., %acc_45 : ..., tensor<64x64xf32, #blocked>
```

这里 after 的 `#blocked` 名字和 before 的 `#blocked6` 语义对应：after 文件重新编号了
attribute aliases，所以不要只按 alias 名字判断 layout 是否相同，要看 attribute 内容。

### 3.3 仍保留 3 个必要 convert

After 仍有：

- After line
  [65](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:65),
  [66](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:66),
  [86](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/029_After_TritonGPURemoveLayoutConversions.mlir:86):

```text
%a_43 = ttg.convert_layout %a
  : tensor<64x32xf16, #blocked1>
  -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>

%b_44 = ttg.convert_layout %b
  : tensor<32x64xf16, #blocked2>
  -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>

%0 = ttg.convert_layout %c
  : tensor<64x64xf16, #blocked>
  -> tensor<64x64xf16, #blocked2>
```

这三个是实际 layout 边界：

- A/B global loads 保留 blocked layout，服务 coalesced memory access。
- `tt.dot` operands 必须是 `#ttg.dot_op` layout，服务 MMA/WGMMA operand contract。
- store 的 value 要转到和 store pointer 匹配的 blocked layout。

所以这个 pass 的目标是“减少不必要的 conversion”，不是“消灭全部 conversion”。

## 4. 基础概念：encoding / anchor / boundary

先把几个容易混的词定清楚。

这里的 `encoding` 不是字符编码，也不是 dtype，而是 TTGIR tensor type 上的 layout
attribute，例如 `#blocked`、`#ttg.dot_op`、`#ttg.slice`。它描述一个 logical tensor
的元素如何分布到 CTA / warp / thread / register fragment 上。`ttg.convert_layout`
保持 shape 和 element type 不变，只改变这个物理分布。

`layout-polymorphic` 的意思是：这个 op 的语义不绑定某一个固定 layout；只要满足该 op
自己的类型规则、operand/result encoding 关系、shape/element type 不变，以及上下游
layout contract，它就可以在不同 encoding 下合法存在。它不是“layout 可以任意乱换”。
更准确地说，layout-polymorphic op 可以跟随某个强约束 anchor 的 layout。

可以用这个分工记：

```text
strong-contract op:
  决定这里必须/最好使用什么 layout
layout-polymorphic op:
  在类型规则允许时跟着换 layout
RemoveLayoutConversions:
  把能跟着换的区域统一到 anchor layout
真实 layout boundary:
  两侧强 contract 不能统一，保留 ttg.convert_layout
```

典型能被统一的区域：

- pointer prologue：`program_id`、`make_range`、offset、mask、`tt.addptr` 等地址准备逻辑。
  这些操作主要计算地址/索引/谓词，通常不关心具体 layout，只要 producer 和 consumer
  类型和 verifier 约束能满足，就可以整体改到 load/store 需要的 encoding。
- elementwise 链：`arith.addf`、`arith.mulf`、cast/broadcast/reshape-like 操作大多只要求
  operands/results encoding 一致或可推导一致，不要求某个特定 encoding。
- loop-carried accumulator：`scf.for iter_args`、body block argument、`tt.dot` result、
  `scf.yield`、loop result 构成同一个跨 iteration 传递的 accumulator 链。如果这条链能
  统一到 dot-friendly encoding，就能删除 loop 边界上的来回 convert。

典型不能被统一、需要保留 convert 的边界：

- global load result 的 blocked layout 到 `tt.dot` operand 的 `#ttg.dot_op` layout。
  前者服务 memory coalescing，后者服务 MMA/WGMMA operand contract。
- dot accumulator/result layout 到 store-friendly blocked layout。前者服务 tensor-core
  累加，后者服务 global store 的 value / pointer layout 匹配。

`loop iter_args` 不是普通循环下标 `i/j/k`。普通下标是 induction variable；`iter_args`
是 loop-carried value，例如 matmul K-loop 中每轮带下去的 accumulator：

```text
acc_init -> scf.for iter_arg %acc
              %acc_next = tt.dot ..., %acc
              scf.yield %acc_next
            -> acc_final
```

因此 `iter_args` 会被讨论，是因为它们把一个 tensor value 跨 loop iteration、yield 和
loop result 串起来。只要其中一段 encoding 没统一，就容易在 loop 入口、body、yield 或
结果处出现成对的 `ttg.convert_layout`。

## 5. 核心模型：constraint propagation

把这个 pass 理解成 layout encoding 的约束传播，比理解成普通 peephole 删除更准确。

当前 pass 的第一条主线是 forward propagation，源码里对应
`propagateLayout()` / `propagateToUsers()`：

```text
anchor 说：这个 value 最好/必须使用 layout L
        ↓
沿 def-use 方向传播 L
        ↓
遇到 layout-polymorphic op：
  如果 inferDstEncoding 能推出合法结果 encoding，
  就可以把输入/输出 tensor type 改成 L
  原来的 ttg.convert_layout 变成冗余
        ↓
遇到另一个硬约束 layout L2：
  如果 L 和 L2 不能统一，这里就是 layout boundary
  保留或重新插入 ttg.convert_layout
```

所以“删除 convert”不是把真实的数据搬运藏起来，而是发现某些 convert 只是中间 IR
暂时选错 encoding 的副产物。pass 改写 producer / consumer 的 tensor type，让一整片
计算直接活在目标 encoding 里。

注意：这张图只描述 forward propagation。源码后面还有第二条独立路径：
`backwardRematerialization()` 会针对剩下的 `ttg.convert_layout`，尝试把 convert 前面的
producer slice 在目标 layout 下重新物化，从而避免真正执行 convert。这不是简单把
encoding 沿 use-def 反向传播，而是检查 producer slice 是否可 rematerialize，再重建那段
计算。

当多个 anchor 的传播在共用区域相遇时，源码允许一个 value 暂时持有多个候选 encoding。
随后 [resolveConflicts](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:376)
为每个冲突 value 单独选一个：

```text
load/store/atomic-like result:
  prefer BlockedEncodingAttr
otherwise:
  prefer MmaEncodingTrait when available
fallback:
  keep the first candidate encoding
```

这不是完整 cost model，而是当前实现的 heuristic。选择背后的直觉是：memory op 附近优先
保护 blocked/coalesced memory layout；dot/MMA 计算链附近优先保护 MMA-friendly layout。
没有被选中的另一侧如果仍需要自己的 encoding，就通过 `getValueAs(...)` 插入或保留
`ttg.convert_layout`，这条 convert 就代表真实边界。

## 6. 源码路径对应

源码顶部注释直接给出主算法：

```text
1. 找 anchor ops，保留它们的 layout。
2. 从 anchor 向 descendants 传播 layout。
3. 如果某个 value 收到多个候选 layout，选择一个，并在冲突处保留/插入 conversion。
4. 按 dominance order 重写 IR。
```

关键代码：

- `isLayoutAnchor` 把 descriptor、expensive load/store、dot、atomic、TMEM load、部分
  gather/reshape 作为 layout anchor。
- `initAnchorLayout` 把 function args 和 anchor results 的现有 encoding 放入
  `layouts`。
- `propagateToUsers` 穿过 `scf.for`、`scf.yield`、elementwise、`expand_dims`、
  `reshape`、`trans`、`join/split`、`convert_layout` 等传播 encoding。
- `setEncoding` 遇到 `ConvertLayoutOp` 时直接尝试让 destination encoding 等于 source
  encoding，这是“把 convert 变成无用 convert，再交给 canonicalization 删除”的核心。
- `resolveConflicts` 当前 heuristic 是：load/store/atomic 倾向 blocked layout；非
  load/store 倾向 MMA-like layout。
- `rewriteRegion` / `rewriteOp` 改写 op result type 和 operand，在必要处通过
  `getValueAs` 创建新的 `ConvertLayoutOp`。

主 pass `runOnOperation` 的顺序是：

```text
forward layout propagation
cleanup convert canonicalization
do backward rematerialization until fixed point
hoist remaining converts
dead iter arg elimination
final convert + scf cleanup
```

这解释了为什么 IR 里不仅少了 conversion，还能看到 producer op 的 tensor type 直接变成
目标 encoding。

## 7. Decision Tree

可以把当前实现压缩成：

```text
find anchors:
  function args
  descriptor ops
  expensive load/store
  dot / atomic / TMEM load
  selected gather/reshape

propagate anchor layouts forward:
  through scf loop args/results/yields
  through same-encoding / elementwise / view-like ops
  through convert_layout by trying to make dst == src

resolve conflicts:
  if load/store-like:
    prefer blocked layout
  else:
    prefer MMA-like layout when available

rewrite:
  change op result encodings in place when legal
  remap operands to expected encodings
  create convert only at unresolved real boundaries

cleanup:
  canonicalize no-op/dead converts
  rematerialize or hoist remaining profitable converts
```

## 8. Compiler Decision

Compiler question:

```text
当前 TTGIR 里哪些 layout conversion 只是前面 pass 产生的临时边界，
可以通过重写 producer/consumer 的 encoding 消掉？
哪些 conversion 是真实边界，必须保留给后续 lowering？
```

Decision made here:

- 保留 load/store/dot 等 anchor 的 layout contract。
- 把 cheap / layout-polymorphic 的计算移动到 anchor 需要的 layout 中。
- 在 coalesced memory layout 与 MMA/WGMMA dot operand layout 之间保留 explicit
  conversion。

Why here:

`ConvertTritonToTritonGPU`、`Coalesce`、`PlanCTA` 会产生或改变 layout contract；这些
pass 之后 IR 中自然会出现许多桥接性的 `ttg.convert_layout`。在继续做
`OptimizeThreadLocality`、`AccelerateMatmul`、`OptimizeDotOperands` 前，先把临时
layout 噪声清掉，后续 pass 看到的 IR 更接近真实边界。

## 9. Compiler Contract

Input contract:

- TTIR 已经进入 TTGIR，tensor type 都有 encoding。
- 前序 pass 已经建立初步 memory/dot/CTA layout。
- IR 里允许存在大量 `ttg.convert_layout` 作为过渡边界。

Output contract:

- 每个被分析到的 tensor value 只有一个最终选择的 encoding。
- pointer prologue、elementwise、loop-carried values 尽量直接使用 anchor-friendly
  encoding。
- 剩下的 `ttg.convert_layout` 更接近真实 layout boundary，而不是机械中间产物。

Next pass relies on:

- `OptimizeThreadLocality`、`AccelerateMatmul`、`OptimizeDotOperands` 不需要处理一堆
  可以消掉的临时 layout conversion。
- 后续 LLVM lowering 只需要 lower 必要 conversion，减少 shared-memory round-trip 或
  register shuffle 的机会。

## 10. Invariant

- Tensor logical shape 不变：`64x32`、`32x64`、`64x64` 没变。
- Element type 不变：`f16` / `f32` / pointer element type 没变。
- Memory address 的 logical meaning 不变，只是 pointer tensor 的 layout encoding 改了。
- `tt.dot` 的数学语义不变。
- `tt.store` 写回的 logical C tile 不变。

Changed only:

```text
layout / encoding placement
where conversions occur
which producer op directly materializes which encoding
```

## 11. Hardware / Execution Reason

Triton mechanism:

- TTGIR layout management pass。
- 消除不必要的 layout boundary。
- 把 layout decision 从 explicit `ttg.convert_layout` chain 变成 tensor type 的 encoding。

Hardware reason:

- Blocked layout 对 global load/store 的 coalescing 更友好。
- Dot operand layout 是 tensor core lowering 的前置 contract。
- 不必要的 layout conversion 最终可能变成 register permutation、shared-memory
  round-trip、同步或额外数据移动。

Instruction reason:

- 当前 first pass 还没有直接选择 `mma.sync` / `wgmma` 指令。
- 但它保留 `#ttg.dot_op` 边界，让后续 `AccelerateMatmul` /
  `OptimizeDotOperands` / MMA lowering 能依赖 dot operand contract。

Execution reason:

- 当前 first pass 不改变 CTA/thread scheduling，也不改变 loop trip count。
- 它改变的是每个 SSA tensor value 在什么 encoding 下被生产和消费，从而减少执行期
  layout shuffle 的机会。

Memory reason:

- Load/store 附近优先保留 blocked layout，让 global memory access 继续服务 coalescing。
- Dot operand 前保留 conversion，说明 memory-friendly layout 和 tensor-core operand
  layout 是两个真实边界，不应被这个 pass 混成一个 layout。

## 12. Architecture Evolution

本轮 first pass 的 cross-architecture before comparison：

- sm86 / sm90 / sm100 before 基本同形。
- after 也基本同形。
- 结论：first `RemoveLayoutConversions` 在这个样本里主要是架构无关的 layout cleanup。

架构差异会在后续 pass 里变明显：

- Ampere 走 `mma.sync` baseline。
- Hopper 引入 WGMMA / warp-group / TMA 相关 pass。
- Blackwell 引入 TMEM / warp specialization / partition scheduling 等更多结构。

对 `RemoveLayoutConversions` 的影响是：后期调用可能需要处理更复杂的 layout boundary，
但当前 first pass 还没有体现这种分叉。

第二次 `RemoveLayoutConversions` 已经出现小分叉：sm86 是 `-5 lines`，sm90/sm100 是
`-3 lines`。这说明 `AccelerateMatmul` 之后，MMA/WGMMA/TMEM 前的 layout cleanup 已经
开始受到架构路线影响；只是本轮 first pass 还没有体现这种分叉。

## 13. 没有这个 pass 会怎样

Correctness:

- 不一定马上错误，因为 explicit `ttg.convert_layout` 仍然表达了 layout 边界。
- 但如果后续 pass 假设 IR 已经被 layout cleanup 过，可能会遇到更复杂或未预期的
  conversion chain。

Performance:

- pointer prologue 和 loop-carried accumulator 里会保留大量临时 conversion。
- 后续 lowering 可能产生更多数据移动、shared-memory round-trip 或 register shuffle。
- dot/load/store 的真实边界会被临时边界淹没，影响后续 layout optimization 判断。

Compiler pipeline:

- 后续 pass 要在更 noisy 的 IR 上工作。
- `OptimizeDotOperands` / lowering 更难只关注真正需要满足的 dot operand contract。

## 14. Alternative Design

Alternative:

```text
更激进地传播 layout，把 load result 直接改成 dot operand layout，
并尝试连 A/B 的 dot-operand convert 也消掉。
```

为什么当前 pass 没这么做：

- `tt.load` 的 result layout 同时服务 global memory coalescing 和后续 consumer；把 load
  直接改成 `#ttg.dot_op` 可能破坏 memory-friendly blocked layout 的 contract。
- `tt.dot` operand 的 `#ttg.dot_op` layout 是 MMA/WGMMA lowering 需要的 instruction
  contract；它和 load/store 的 blocked layout 是真实边界，不是临时噪声。
- 当前设计选择保留真实边界，只把 pointer arithmetic、elementwise、loop-carried
  accumulator 这类 cheap/layout-polymorphic producer 改到目标 encoding。

## 15. Knowledge Card

What it is:

```text
TTGIR layout cleanup pass: remove transient ttg.convert_layout by rewriting
producer/consumer tensor encodings around stable layout anchors.
```

When it runs:

```text
NVIDIA pipeline calls it after PlanCTA, after AccelerateMatmul, and later after
pipeline/TMEM-related transformations.
```

Main anchors:

```text
function args, descriptor ops, load/store/atomic, dot, TMEM load, selected
gather/reshape-like ops.
```

Remember:

```text
No-op convert removal is only the visible cleanup. The more important behavior
is changing result/operand encodings so cheap producers materialize the layout
their anchored users need.
```

Hardware reason:

```text
Keep memory-friendly blocked layouts around load/store, keep dot_op layouts for
tensor-core operands, and remove layout movement that is not a real hardware
contract boundary.
```

## 16. Open Questions / 下一步

后续应继续看几类问题：

- 第二次 `RemoveLayoutConversions`：在 `AccelerateMatmul` 之后，解释 §12 提到的
  sm86 `-5 lines` 与 sm90/sm100 `-3 lines` 分叉，具体来自哪几条 IR。
- 更后期 `RemoveLayoutConversions`：本轮 sm90/sm100 canonical matmul 是 no-op，需要找一个
  包含 conditional、warp specialization、TMEM 或复杂 pipeline 的 kernel，看
  backward rematerialization / hoisting 真正生效的 IR 形态。
- `resolveConflicts` 的 heuristic 在多 anchor 冲突时是否存在可复现实例：同一个 value
  同时被 load/store-like 和 MMA-like consumer 拉向不同 layout？
