# Triton TTGIR：Legality Repair

本文只回答一个问题：`mapping / organization / scheduling` 基本定完之后，还缺哪些合法性约束，才能继续 lower。

## 1. 核心定义

```text
legality repair
  = 在不重新定义谁来算、值怎么流动、工作何时发生的前提下，
    补齐 target / storage / lowering 需要的额外顺序、可见性、别名隔离约束
```

这里的 `legal` 不是“高层语义看起来合理”，而是：

- 后续 lowering 不会违反 target protocol
- aliasing storage 的复用顺序已经被约束
- generic path 和 async path 的可见性缺口已经补上

如果压成一句短记忆：

```text
mapping = 分工
organization = 衔接
scheduling = 时序
legality = 补约束
cleanup = 去噪音
```

## 2. 和其他四类的边界

| 类别 | 核心问题 | 典型载体 |
|---|---|---|
| distributed execution mapping | 谁拥有这些 elements | `#ttg.blocked`、`CGAEncodingAttr` |
| layout / data-movement organization | 值以什么 form / carrier 流动 | `ttg.convert_layout`、`ttg.local_alloc`、descriptor、TMEM |
| target-driven scheduling | 这些工作何时发生、如何 overlap | `loop.stage`、async op、wait、barrier protocol |
| legality repair | 还缺什么约束才能继续 lower | fence、proxy ordering、TMEM reuse barrier |
| cleanup | 哪些中间表示噪音可以删除 | 冗余 convert、死代码、临时 token、重复链 |

判断边界时最容易混淆的三点：

- 它不是 scheduling。`Pipeline` / `TMALowering` 把协议显式化；`legality repair` 是在协议已经基本出现后补缺口。
- 它不是 cleanup。`cleanup` 删除噪音；`legality repair` 新增的是“没有就不合法”的约束。
- 它经常出现在很晚的位置，因为很多 hazard 只有在 shared/TMEM allocation 之后才看得见。

## 3. 常见 legality gap

### 3.1 generic proxy 和 async proxy 不是同一条可见性链

Hopper 以后，shared memory 可能同时被：

- generic path 访问
- TMA / WGMMA / TMEM 这类 async path 访问

`wait` 或 `barrier` 只解决“完成何时可观察”，不自动解决“generic proxy 写入什么时候对 async proxy 可见”。这类缺口需要 fence repair。

### 3.2 logical lifetime 不等于 physical storage lifetime

TTGIR 里两个逻辑上不同的 TMEM value，lower 之后可能 alias 到同一段 physical tensor memory。

这时 schedule 本身可能没问题，但 physical reuse 还缺顺序约束。这个缺口要在 allocation 之后才能判断，所以属于 lowering-side legality。

### 3.3 allocation 之后才暴露的 alias hazard

shared memory / tensor memory 分配前，很多 op 还只是“逻辑上在某个 buffer 上工作”。真正的 aliasing slice 通常要等
[allocate_shared_memory_nv](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:384)
和
[allocate_tensor_memory](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:385)
之后才明确。

这也是为什么 `ProxyFenceInsertion` 和 `TMemBarrierInsertion` 不在 `make_ttgir`，而在
`make_llir`。

## 4. 由哪些 pass 负责

### 4.1 `FenceInsertion`：TTGIR 内的早期 legality repair

`FenceInsertion` 在 `make_ttgir` 里，位置是
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:325)。
pass 定义在
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:43)。

它的职责是：

- 针对 generic / async proxy ordering 缺口插 fence
- 尽量在结构化 TTGIR 还完整时选择更优位置
- 先补明显、可提前判断的 legality 缺口

实现上它只对 `computeCapability >= 90` 生效，见
[FenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:36)。

更准确的理解是：

```text
FenceInsertion
  = TTGIR 阶段的 optimized placement
  不是最终 alias-aware 兜底
```

覆盖面也要分清。当前实现不是对所有 shared access 做通用扫描，而是直接
`mod.walk(DotOpInterface)`，围绕 dot operand 的 use-def 链去找 reg-to-shared copy /
`local_store` 路径，见
[FenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:39)
和
[FenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:77)。
同一个文件的 TODO 也明确写了还要支持更一般的 pattern，见
[FenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp:31)。

所以它和 `ProxyFenceInsertion` 的差别不只是“早放 / 晚放”和“是否 alias-aware”，还包括覆盖面：

- `FenceInsertion`：定向修补 dot-operand 通往 async consumer 的 shared path
- `ProxyFenceInsertion`：allocation 之后按 aliasing slice 做全 shared-memory 的保守扫描

### 4.2 `ProxyFenceInsertion`：allocation 之后的 alias-aware 兜底

`ProxyFenceInsertion` 不在 `make_ttgir`，而在 `make_llir`：
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:390)。
pass 定义在
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:65)。

它的职责是：

- 在 shared memory allocation 之后重新看 aliasing slice
- 对 earlier `FenceInsertion` 没覆盖到的情况做保守兜底
- 保证 generic proxy 和 async proxy 在 shared memory 上的顺序合法

源码里可以直接看到两点：

- `computeCapability < 90` 直接返回，见
  [ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:195)
- 它显式构造 `ModuleAllocation` 做分析，见
  [ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:200)

同一个文件还直接写了它是 earlier pass 没放 fence 时的 safe fallback，见
[ProxyFenceInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:148)。

这也是为什么不能简单理解成“把同一个 fence pass 挪晚一点”。两道 fence 的职责本来就不同：

- 前一道利用 TTGIR 结构和 dot-context 做更定向的 optimized placement
- 后一道利用 allocation + alias analysis 做覆盖面更大的 functional fallback

### 4.3 `TMemBarrierInsertion`：TMEM physical reuse 的 legality repair

`TMemBarrierInsertion` 也在 `make_llir`：
[compiler.py](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:391)。
pass 定义在
[Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:85)。

它的职责不是“再做一次 scheduling”，而是：

- 在 tensor memory allocation 之后识别 aliasing physical TMEM storage
- 当不同 logical lifetime 复用同一段 physical storage 时，补 CTA barrier
- 保证后续 lowering 不会把 TMEM reuse 降成不合法的并发访问

实现上它同样依赖 allocation 结果，见
[TMemBarrierInsertion.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/TMemBarrierInsertion.cpp:296)。

## 5. 主链位置

按当前 NVIDIA backend 的 wiring，legality repair 相关位置是：

```text
make_ttgir
  ...
  -> SymbolDCE
  -> FenceInsertion
  -> LowerMMA
  -> SCCP / CSE / Canonicalizer

make_llir
  -> allocate_shared_memory_nv
  -> allocate_tensor_memory
  -> ProxyFenceInsertion
  -> TMemBarrierInsertion
  -> to_llvmir
```

源码位置见：

- [make_ttgir](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:261)
- [make_llir](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/nvidia/backend/compiler.py:366)

所以如果你在 TTGIR pass dump 里找不到 `ProxyFenceInsertion` / `TMemBarrierInsertion`，这是正常的。它们属于 lowering-side legality，不属于 TTGIR pass 主链本身。

## 6. 读 IR 时怎么判断它是不是 legality repair

先问三件事：

1. 这个 op 是在建立新的 schedule/protocol，还是在补已有 protocol 的缺口？
2. 没有这一步，失败点会出现在“性能变差”，还是“lowering / target contract 不合法”？
3. 这个 hazard 只有 allocation / alias analysis 之后才能看见吗？

如果答案更接近下面这条链，就按 legality repair 读：

```text
已有 contract 基本成立
  -> 还缺顺序 / 可见性 / alias 隔离
  -> 插 fence / barrier
  -> 才能继续 lower
```

具体 barrier、fence、proxy、TMA、WGMMA、TMEM protocol 细节，单独看
[2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md)。
