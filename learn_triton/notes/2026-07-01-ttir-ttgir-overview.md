# 2026-07-01 学习笔记：TTIR / TTGIR 分层、完整定义与约束

## 1. 先给结论

如果领导说“把 TTIR 和 TTGIR 的完整定义看明白”，正确读法不是把所有 `*.td` 生吞一遍，而是先建立分层模型：

```text
Python frontend
  -> TTIR (`tt`)
  -> TritonGPU / 公共 TTGIR (`ttg`)
  -> vendor GPU dialect (`ttng` for NVIDIA, `amdg` for AMD)
  -> LLVM / NVVM / ROCDL
```

这里有两个容易混淆的点：

1. 严格说，`ttg` 才是 TritonGPU dialect。
2. 但工程里口头说 “TTGIR” 时，很多人会把 `ttg + ttng + amdg` 一起算作“GPU 层 IR”。

所以学习时最好同时保留两个概念：

- 狭义 TTGIR：`ttg`
- 广义 TTGIR：`ttg + vendor dialect`

---

## 2. 代码入口

核心入口文件：

- `tt`:
  [include/triton/Dialect/Triton/IR/TritonDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonDialect.td:1)
- `tt` ops:
  [include/triton/Dialect/Triton/IR/TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1)
- `tt` types:
  [include/triton/Dialect/Triton/IR/TritonTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonTypes.td:1)
- `ttg`:
  [include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td:1)
- `ttg` ops:
  [include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:1)
- `ttg` attrs/layout:
  [include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1)
- TTIR -> TTGIR lowering:
  [include/triton/Conversion/TritonToTritonGPU/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:1)
- `ttng`:
  [include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td:1)
- `ttng` ops:
  [include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:1)
- `amdg`:
  [third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUDialect.td:1)
- `amdg` ops:
  [third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:1)

约束和 verifier 关键入口：

- `tt` tensor-layout trait:
  [include/triton/Dialect/Triton/IR/Traits.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/Traits.h:50)
- `ttg` memdesc-layout trait:
  [include/triton/Dialect/TritonGPU/IR/Traits.h](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/Traits.h:21)
- `ttg` layout / encoding 推导与验证:
  [lib/Dialect/TritonGPU/IR/Dialect.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp:1)
- `ttng` operation attr verifier:
  [lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/IR/Dialect.cpp:626)

---

## 3. 每一层的目标

### 3.1 TTIR `tt`

目标是表达 Triton kernel 的前端语义：

- 计算什么
- 访问什么内存
- tensor 形状怎么变
- 哪些地方是 SPMD 语义
- 哪些地方是 reduction / scan / dot / descriptor 访问

这一层尽量不编码：

- 线程如何分配到元素
- shared memory 具体布局
- warp / warpgroup 的硬件同步细节
- vendor 专有硬件指令

### 3.2 TritonGPU `ttg`

目标是把前端语义转成“GPU 上可执行的布局和同步语义”：

- distributed tensor layout
- shared memory descriptor
- async copy token / wait / commit
- local memory load/store/gather/scatter
- warp specialization
- barrier / warp id

### 3.3 NVIDIA vendor dialect `ttng`

目标是表达 NVIDIA 特有硬件能力：

- cluster
- mbarrier
- TMA
- warp-group MMA
- TMEM
- Blackwell CLC

### 3.4 AMD vendor dialect `amdg`

目标是表达 AMD 特有硬件能力：

- buffer memory path
- TDM descriptor path
- AMD barrier / cluster barrier
- in-thread transpose / packed local load
- AMD 异步拷贝和 wait 语义

### 3.5 这一套分层为什么这样设计

这里先把“设计意图”说清楚，但要区分两种证据强度：

- 源码明确写出的目标：
  例如 pass / op / type 的 `summary`、`description`、verifier、trait。
- 基于当前结构推断的设计意图：
  例如为什么某类能力停在 `ttg`，为什么某类能力必须下沉到 `ttng/amdg`。

后面如果我说“设计意图”，默认优先指第一类；如果是第二类，会明确写成“从当前结构看，更合理的解释是”。

先压一句总体设计意图：

- `tt` 的职责是把算法和抽象访存语义稳定下来，不提前绑定硬件执行协议。
- `ttg` 的职责是把 GPU 执行必需、而且还能公共化的 contract 显式化。
- `ttng/amdg` 的职责是承接公共层装不下的硬件专有协议、资源和 ISA 路径。

这套分层想解决的核心矛盾是：

- 前端需要保持算法表达简洁、可移植。
- 后端又必须显式知道 layout、shared memory、async dependency、barrier、硬件 feature。

如果没有这层分离，就会出现两种坏结果之一：

- 要么 `tt` 被大量硬件细节污染，前端抽象塌掉。
- 要么后端拿不到足够显式的 legality / synchronization / layout contract，无法稳定 lowering。

---

## 4. TTIR 是什么

### 4.1 dialect 边界

`tt` dialect 自己在定义里已经说明，它依赖的不只是自身 op，还依赖这些 MLIR dialect：

- `arith`
- `math`
- `scf`
- `cf`
- `ub`

见：
[include/triton/Dialect/Triton/IR/TritonDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonDialect.td:1)

这意味着真实的 TTIR module 往往是：

```text
tt + arith + math + scf + cf + ub
```

而不是“只有 `tt.*`”。

### 4.2 类型系统

`tt` 里最关键的 type：

- 标量浮点：`f16/bf16/f32/f64/f8...`
- 标量整数：`i1/i4/i8/i16/i32/i64`
- tensor：`tensor<...>`
- 指针：`!tt.ptr<T>`
- tensor descriptor：`!tt.tensordesc<...>`

关键约束：

- `!tt.ptr<T>` 只能指向标量 element type，而不是任意复合对象
- `tt.tensordesc` 是可移植 descriptor 抽象，不等同于某家硬件的真实 descriptor 格式

见：
[include/triton/Dialect/Triton/IR/TritonTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonTypes.td:1)

---

## 5. TTIR 全部 op 分组 + 每组核心约束

下面这张表按学习更有用的方式分组，而不是按文件书写顺序机械罗列。

### 5.1 TTIR 全部分组表

| 分组 | Ops | 主要作用 | 核心约束 |
| --- | --- | --- | --- |
| Cast / 类型变换 | `tt.int_to_ptr`, `tt.ptr_to_int`, `tt.bitcast`, `tt.fp_to_fp` | 标量或 tensor 元素级类型变换 | shape 一致；encoding 一致；bitcast 必须位宽匹配；`fp_to_fp` 受浮点类型和 rounding 约束 |
| 基础算术补充 | `tt.clampf`, `tt.precise_sqrt`, `tt.precise_divf`, `tt.mulhiui` | Triton 自己补的非通用 arith 语义 | 输入输出类型必须匹配；大多是 elementwise；保留语义精度 contract |
| 指针算术 | `tt.addptr` | 基于 tensor/scalar offset 做指针位移 | result type 必须回到原 pointer family；shape/encoding 需匹配 |
| 全局内存访问 | `tt.load`, `tt.store` | 从 pointer 或 pointer tensor 做 load/store | pointer pointee type 必须与 value/result 匹配；`mask` / `other` 形状类型要对齐；cache/eviction/volatile 只能在允许位置出现 |
| 原子内存访问 | `tt.atomic_rmw`, `tt.atomic_cas` | 原子读改写 / compare-exchange | pointer/value/result 类型要匹配；原子操作符、memory semantic、scope 组合必须合法 |
| 形状变换 | `tt.splat`, `tt.unsplat`, `tt.expand_dims`, `tt.reshape`, `tt.broadcast`, `tt.cat`, `tt.join`, `tt.split`, `tt.trans` | 不改算法语义地重排或组合 tensor 形状 | rank/shape 要可推导；某些 op 需要保持元素总数；有些 op 保持 encoding，有些 op 允许重排元素顺序 |
| SPMD 查询 | `tt.get_program_id`, `tt.get_num_programs` | 查询 program id / grid 规模 | 维度参数必须合法；语义属于程序实例级，不是 warp 级 |
| 矩阵乘语义 | `tt.dot`, `tt.dot_scaled` | 抽象矩阵乘加或缩放矩阵乘加 | A/B/C/D 形状和元素类型必须满足 dot contract；这里只表达语义，不承诺具体用哪家 MMA/MFMA 指令 |
| Region 聚合 | `tt.reduce`, `tt.reduce.return`, `tt.scan`, `tt.scan.return`, `tt.map_elementwise`, `tt.map_elementwise.return` | 用 region 表达组合逻辑 | region block args、return types、combine 逻辑必须满足协议；不是任意 region 都合法 |
| 外部计算 / 内联 asm | `tt.extern_elementwise`, `tt.elementwise_inline_asm` | 把外部函数或 asm 接入 tensor elementwise 路径 | 必须维持 elementwise contract；输入输出类型和 pack 规则要一致；会引入额外 ABI / side-effect 风险 |
| 生成 / 统计 / 采样 | `tt.make_range`, `tt.histogram`, `tt.gather` | 生成 index tensor、做 histogram、局部 gather | shape 和 index 轴必须合法；gather 的 index/value 对齐必须成立 |
| 调试 / 断言 | `tt.print`, `tt.assert` | 设备端调试与检查 | 有 side effect；很多 pipeline/pass 会把它们当成不可随意重排的边界 |
| Tensor descriptor 建模 | `tt.make_tensor_descriptor`, `tt.descriptor_load`, `tt.descriptor_store`, `tt.descriptor_reduce`, `tt.descriptor_gather`, `tt.descriptor_scatter` | 用可移植 descriptor 抽象 tiled tensor 访存 | descriptor shape/element type/offsets 必须匹配；这是“抽象 descriptor”，不直接等于某家硬件 descriptor |
| 函数与调用 | `tt.call`, `tt.func`, `tt.return` | kernel / helper function 边界 | 符号引用、函数签名、return 类型必须匹配；调用约定受 MLIR function/call interface 约束 |

### 5.2 TTIR 的公共不变量

除了每个分组自己的约束，`tt` 层还有几个横跨大部分 op 的共性不变量。

#### 5.2.1 类型和形状必须先自洽

TTIR 最先保证的是：

- value type 自洽
- tensor rank/shape 自洽
- pointer pointee type 自洽
- region op 的输入输出协议自洽

如果这些都不成立，后面根本没法谈 GPU 布局。

#### 5.2.2 TTIR 只弱依赖 layout，不强行承诺 GPU 执行布局

`TT_Op` 基类带有 `VerifyTensorLayoutsTrait`，但这一层的重点不是“硬件 layout 合法性”，而是“不产生明显矛盾的 tensor layout contract”。

见：
[include/triton/Dialect/Triton/IR/TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:27)

#### 5.2.3 descriptor 在 TTIR 里是抽象接口，不是硬件对象

这是理解 TMA/TDM 的关键。

`tt.make_tensor_descriptor` 的目标不是立刻产出 NVIDIA TMA descriptor 或 AMD TDM descriptor，而是先建立“有 tiled tensor 访问语义的抽象对象”。真正厂商化通常发生在更后面的 GPU / vendor 层。

#### 5.2.4 `tt.dot` 只表达算法 contract

这类 op 最容易被误解。

`tt.dot` 的设计意图是：

- 表达矩阵乘加
- 保留输入精度、累加类型、缩放语义
- 不在这一层决定用 `ttng.warp_group_dot`、`ttng.tc_gen5_mma`、`#ttg.amd_mfma` 还是软件路径

换句话说，TTIR 说的是“要算 matmul”，不是“用哪条硬件指令算 matmul”。

#### 5.2.5 为什么 TTIR 停在“算法 contract”这一层

从当前结构看，TTIR 的设计意图大致是：

- 先把“程序想做什么”表达稳定。
- 暂时不把“GPU 上如何分工、如何同步、如何布局”提前固化。

这样做主要是为了保住三件事：

- 可移植性：同一个 `tt.dot`、`tt.load`、`tt.reduce` 不需要在前端区分 NVIDIA / AMD 路径。
- 可变换性：很多优化和重写还可以在不关心最终硬件路径的情况下进行。
- 降低前端负担：Python frontend 不需要直接操心 `mbarrier`、TMA、cluster、shared swizzle。

如果把这些硬件协议太早放进 TTIR，会破坏一个关键性质：

- 前端语义还没稳定，硬件路径已经被选死。

那样后面的 pass 不再是在“从算法语义推导执行策略”，而是在“被前端过早决定的硬件细节上做修补”，这通常不是健康的 IR 分层。

---

## 6. TTGIR 是什么

### 6.1 TTIR 到 TTGIR 到底新增了什么

`convert-triton-to-tritongpu` 的描述里已经写得很直接：

- 会把 Triton Dialect 转成 TritonGPU Dialect
- 也会影响 `arith`、`math`、`scf`、`cf`
- 主要工作之一是给 tensor type 增加 layout encoding
- 这些 encoding 一般包含 `numWarps`、`threadsPerWarp`、`numCTAs`

见：
[include/triton/Conversion/TritonToTritonGPU/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)

所以 TTIR -> TTGIR 不是“换个 dialect 名字”，而是把 GPU 执行所需的布局信息显式化。

### 6.2 TTGIR 的核心载体

TTGIR 关键不只在 op，还在 type/attr：

- distributed tensor encoding
- shared layout encoding
- CTA / warp / lane 映射
- `!ttg.memdesc<...>`
- module attrs:
  `ttg.num-warps`, `ttg.threads-per-warp`, `ttg.num-ctas`, `ttg.target`

这里要和上一节区分清楚：

- 上一节提到的 `arith/math/scf/cf` 是 `convert-triton-to-tritongpu` 这个转换 pass 会处理的相关 dialect
- `ttg` dialect 自己的 dependent dialect 不是这些，而是 `triton::TritonDialect` 和 `mlir::gpu::GPUDialect`

### 6.3 `ttg` 层的关键变化

到了 `ttg`，很多 TTIR 里隐含的东西被显式化：

- shared memory 不再只是一个抽象概念，而是 `memdesc`
- async copy 不再是“语义上异步”，而是 token / commit / wait 的显式 SSA 链
- barrier 不再是通用同步点，而要带明确 address space / scope 语义
- tensor layout 成为大部分 op legality 的一等公民

### 6.4 为什么 TTGIR 要把这些东西显式化

这部分源码里其实已经给了不少直接证据。

例如：

- `convert-triton-to-tritongpu` 明说自己要给 tensor type 增加 layout encoding，见
  [include/triton/Conversion/TritonToTritonGPU/Passes.td:6](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:6)
- `ttg.async.token` 明说自己存在的意义是建立 async op 和 group / sync op 之间的 SSA link，见
  [include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:13](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:13)
- `ttg.memdesc` 明说它不是普通 tensor，而是“base pointer + descriptor”的组合，且允许多 view，见
  [include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:24](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:24)

从这些定义反推，TTGIR 的设计意图很清楚：

- 把后端真正依赖的 GPU 执行 contract 从“隐含知识”变成“显式 IR 对象”。

具体说，就是把三类原本容易隐含在实现里的关系显式化：

1. 数据怎么分配给线程  
   这由 layout encoding 显式承载。
2. shared memory 里的对象是什么、能不能重解释、能不能切 view  
   这由 `memdesc` 显式承载。
3. 异步操作和等待点之间谁依赖谁  
   这由 async token / commit / wait 的 SSA 链显式承载。

这一步的设计价值，不只是“方便 lowering”，更是为了建立可验证的不变量：

- legality 可以在 IR 上检查
- dependency 可以在 IR 上追踪
- pass 插 barrier / relayout / allocate shared memory 时有稳定抓手

否则这些关系只能散落在约定、分析器和 backend pattern 里，系统会非常脆弱。

---

## 7. TTGIR / `ttng` / `amdg` 全部 op 分组 + 公共/专有语义

### 7.1 先看分层结论

| 层 | dialect | 角色 | 是否 Triton 公共语义 |
| --- | --- | --- | --- |
| TTIR | `tt` | 前端算法语义 | 是 |
| 公共 GPU 层 | `ttg` | 布局、shared、async、warp specialization | 是 |
| NVIDIA GPU 层 | `ttng` | NVIDIA 特有硬件抽象 | 否，vendor 专有 |
| AMD GPU 层 | `amdg` | AMD 特有硬件抽象 | 否，vendor 专有 |

但还有一个细节：

- 很多 vendor-specific layout encoding 其实住在 `ttg` 的 attr 体系里
- 所以“公共 GPU 层”在 op 上比较公共，在 layout attr 上并不完全 vendor-neutral

这个现象可以直接在：
[include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1)
里看到，shared layout builder 会分 NVIDIA / AMD 路径。

### 7.2 `ttg` 全部分组表

| 分组 | Ops | 语义类别 | 备注 |
| --- | --- | --- | --- |
| Layout | `ttg.convert_layout` | Triton 公共 GPU 语义 | 把相同逻辑 tensor 重映射到不同 distributed layout |
| Async token / copy | `ttg.async_copy_global_to_local`, `ttg.async_commit_group`, `ttg.async_wait` | Triton 公共 GPU 语义 | 为异步拷贝建立 token、group、wait contract |
| Shared memory alloc | `ttg.local_alloc`, `ttg.local_dealloc` | Triton 公共 GPU 语义 | 显式 shared/local memory 分配和生命周期 |
| Memdesc view | `ttg.memdesc_index`, `ttg.memdesc_subslice`, `ttg.memdesc_trans`, `ttg.memdesc_reshape`, `ttg.memdesc_reinterpret` | Triton 公共 GPU 语义 | 对 shared descriptor 做逻辑视图变换 |
| Shared/local data movement | `ttg.local_load`, `ttg.local_store`, `ttg.local_gather`, `ttg.local_scatter`, `ttg.local_atomic_scatter_rmw` | Triton 公共 GPU 语义 | shared <-> distributed tensor 之间的数据交换 |
| Pipelining helper | `ttg.predicate_stage`, `ttg.mask`, `ttg.mask.return` | Triton 公共 GPU 语义 | 软件流水与 predication 的 IR 支撑 |
| 数值扩展 | `ttg.fp4_to_fp` | Triton 公共 GPU 语义 | 通用 GPU 层的 fp4 upcast 抽象 |
| Global scratch | `ttg.global_scratch_alloc` | Triton 公共 GPU 语义 | 为后端 scratch buffer 分配建模 |
| Warp specialization | `ttg.warp_specialize`, `ttg.warp_specialize.partitions`, `ttg.warp_yield`, `ttg.warp_return` | Triton 公共 GPU 语义 | 建模一个 CTA 内多个 warp group 的异步分工 |
| 同步 / 标识 | `ttg.barrier`, `ttg.warp_id` | Triton 公共 GPU 语义 | 提供 CTA 级 barrier 和 warp id |

### 7.3 `ttg` 层核心约束

#### 7.3.1 layout legality 是主约束，不再只是类型自洽

`TTG_Op` 基类统一带：

- `VerifyTensorLayoutsTrait`
- `VerifyMemDescLayoutsTrait`

见：
[include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:27)

这说明 `ttg` 层最重要的不变量已经变成：

- tensor layout 合法
- memdesc layout 合法
- layout 能和 module 的 GPU 参数对齐

#### 7.3.2 `memdesc` 是 TTGIR 的一等公民

`!ttg.memdesc<...>` 包含：

- shape
- elementType
- encoding
- memorySpace
- mutableMemory
- allocShape

见：
[include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:17)

这意味着很多 shared memory 相关 legality，不是检查一个裸 pointer，而是检查一个带完整布局 contract 的 descriptor。

#### 7.3.3 async 语义必须显式成图

`ttg.async_copy_global_to_local` 不是“逻辑上异步就够了”，而是必须通过：

- copy
- commit group
- wait

建立后续 pass 可以消费的 SSA 依赖链。

#### 7.3.4 `ttg` 是公共层，但不等于完全 vendor-neutral

这点很关键。

`ttg` 的 op 设计大多是公共的，但 layout attr 体系里已经包含：

- NVIDIA MMA 相关 encoding
- AMD MFMA / WMMA 相关 encoding
- shared layout builder 对不同 vendor 的分支逻辑

这一点可以直接在同一个 `ttg` attr 定义文件里看到：

- `AMDMfmaEncodingAttr` 定义在
  [TritonGPUAttrDefs.td:902](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:902)
- `AMDWmmaEncodingAttr` 定义在
  [TritonGPUAttrDefs.td:1075](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1075)
- `NvidiaMmaEncodingAttr` 定义在
  [TritonGPUAttrDefs.td:1272](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1272)

更关键的是，`SwizzledSharedEncodingAttr` 的 builder 在同一处直接按 dot operand 的 parent encoding 分三条 vendor 路径：

- 先匹配 `AMDMfmaEncodingAttr`，调用 `composeSharedLayoutForOperand`，见
  [TritonGPUAttrDefs.td:123](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:123)
- 再匹配 `AMDWmmaEncodingAttr`，调用 `composeSharedLayoutForOperand`，见
  [TritonGPUAttrDefs.td:130](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:130)
- 否则走 `NvidiaMmaEncodingAttr` 分支，见
  [TritonGPUAttrDefs.td:137](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:137)

所以不要把 `ttg` 理解成“完全抽象、和厂商无关”。

---

## 8. `ttng` 全部分组表

`ttng` 是 NVIDIA vendor dialect，定义文件里直接说它依赖 `tt` 和 `ttg`。

见：
[include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUDialect.td:1)

### 8.1 `ttng` 分组表

| 分组 | Ops | 语义类别 | 说明 |
| --- | --- | --- | --- |
| Fence / cluster 基础同步 | `ttng.fence_async_shared`, `ttng.fence_mbarrier_init_release_cluster`, `ttng.cluster_arrive`, `ttng.cluster_wait`, `ttng.cluster_barrier` | NVIDIA 专有 | cluster 和 async proxy fence 是 NVIDIA 特定语义 |
| Blackwell CLC | `ttng.clc_try_cancel`, `ttng.clc_load_result`, `ttng.clc_is_canceled`, `ttng.clc_get_program_id` | NVIDIA 专有 | cluster launch control，明显是 Blackwell 路径 |
| Warp-group MMA | `ttng.warp_group_dot`, `ttng.warp_group_dot_wait` | NVIDIA 专有 | 对应 NVIDIA warp-group 级矩阵乘 |
| MBarrier | `ttng.init_barrier`, `ttng.inval_barrier`, `ttng.barrier_expect`, `ttng.wait_barrier`, `ttng.arrive_barrier`, `ttng.async_copy_mbarrier_arrive` | NVIDIA 专有 | PTX mbarrier 语义在 IR 中显式化 |
| Async shared / TMA | `ttng.async_shared_store`, `ttng.async_tma_copy_global_to_local`, `ttng.async_tma_copy_local_to_global`, `ttng.async_tma_reduce`, `ttng.async_tma_gather`, `ttng.async_tma_scatter`, `ttng.async_tma_store_wait` | NVIDIA 专有 | Hopper/Blackwell TMA 路径 |
| TCGen5 / Blackwell MMA | `ttng.tc_gen5_mma`, `ttng.tc_gen5_mma_scaled`, `ttng.tc_gen5_commit` | NVIDIA 专有 | Blackwell tensor core generation 5 |
| TMEM | `ttng.tmem_load`, `ttng.tmem_store`, `ttng.tmem_alloc`, `ttng.tmem_subslice`, `ttng.tmem_copy` | NVIDIA 专有 | Tensor memory 是 NVIDIA 特定存储层 |
| Tensor descriptor / tensormap | `ttng.reinterpret_tensor_descriptor`, `ttng.tensormap_create`, `ttng.tensormap_fenceproxy_acquire` | NVIDIA 专有 | 明确面向 NVIDIA descriptor / tensormap 机制 |

### 8.2 `ttng` 层约束特点

`ttng` 的约束已经明显带架构代际信息：

- 有些 op 只支持 `compute capability >= 80`
- 有些 op 只支持 `>= 90`
- 有些 op 只对支持 cluster 的架构合法

也就是说，这一层的 legality 不只是：

- 类型对不对
- layout 对不对

还包括：

- 架构 feature 是否存在
- barrier / descriptor / alignment 是否满足该硬件指令要求

---

## 9. `amdg` 全部分组表

AMD vendor dialect 不在主 `include/triton/Dialect/...` 下，而是在：

- `third_party/amd/include/Dialect/TritonAMDGPU/...`

这点本身就说明它是插件式 vendor 扩展，而不是 `ttg` 主体的一部分。

### 9.1 `amdg` 分组表

| 分组 | Ops | 语义类别 | 说明 |
| --- | --- | --- | --- |
| Tensor 局部变换 | `amdg.extract_slice`, `amdg.concat` | AMD 专有 | 为 AMD 路径提供特定 slice/concat 语义 |
| 条件同步 | `amdg.cond_barrier` | AMD 专有 | block 内部分线程同步语义 |
| Buffer memory path | `amdg.buffer_load`, `amdg.buffer_load_to_local`, `amdg.buffer_atomic_rmw`, `amdg.buffer_atomic_cas`, `amdg.buffer_store` | AMD 专有 | AMD buffer 指令路径的显式建模 |
| Masked memory path | `amdg.masked_load`, `amdg.masked_store` | AMD 专有 | 为 AMD lowering 暴露 masked memory 形式 |
| 缩放上采样 | `amdg.scaled_upcast_fp4`, `amdg.scaled_upcast_fp8` | AMD 专有 | 把 upcast + scale 合并成 AMD 特定语义单元 |
| Register / LDS 变换 | `amdg.in_thread_transpose`, `amdg.local_load_packed_tranposed` | AMD 专有 | 面向 AMD register/LDS 访问模式 |
| Local/global async copy | `amdg.async_copy_local_to_global`, `amdg.async_wait`, `amdg.memory_counter_wait` | AMD 专有 | AMD 异步拷贝与计数器等待 |
| TDM descriptor path | `amdg.async_tdm_copy_global_to_local`, `amdg.async_tdm_copy_local_to_global`, `amdg.async_tdm_scatter`, `amdg.async_tdm_gather`, `amdg.async_tdm_wait`, `amdg.async_tdm_intrinsic_wait`, `amdg.tdm_prefetch`, `amdg.update_tensor_descriptor` | AMD 专有 | AMD 的 tensor descriptor memory path |
| MBarrier | `amdg.init_barrier`, `amdg.wait_barrier`, `amdg.arrive_barrier`, `amdg.async_copy_mbarrier_arrive` | AMD 专有 | AMD 路径里的 barrier 机制 |
| Cluster barrier | `amdg.cluster_barrier_arrive`, `amdg.cluster_barrier_wait` | AMD 专有 | AMD 的 cluster 同步 |

### 9.2 `amdg` 层约束特点

AMD 层最重要的设计意图，不是重复造一个 `ttg`，而是把 `ttg` 里不够表达 AMD 特有硬件路径的部分显式化：

- buffer ops 需要自己的地址和偏移语义
- TDM descriptor 路径需要自己的 descriptor 更新和 wait 规则
- 某些 register/LDS 访问模式需要单独 op 才能稳定 lower

所以 `amdg` 层的核心约束往往是：

- 和 AMD buffer/TDM ABI 对齐
- 和 AMD ISA 特性对齐
- 和 AMD backend 的 lowering pattern 一一对应

---

## 10. 哪些是 Triton 自己的，哪些是 vendor 特殊的

### 10.1 可以直接这样记

Triton 公共语义：

- `tt`
- `ttg`

vendor 专有语义：

- `ttng`
- `amdg`

### 10.2 但要补一句限定

如果问题是“哪些 op 是 Triton 自己的，哪些 op 是 vendor 专有的”，上面的答案够了。

如果问题是“哪些 GPU 布局能力是 vendor-specific 的”，那答案要更细：

- `ttg` 的 op 大体是公共的
- `ttg` 的 layout / encoding attr 并不完全公共
- 其中已经包含不少 NVIDIA / AMD 专有布局知识

所以完整说法应该是：

```text
公共层 op 主要在 tt / ttg。
vendor 专有 op 主要在 ttng / amdg。
但 vendor-specific layout knowledge 部分内嵌在 ttg attr 体系里。
```

### 10.3 为什么必须单独起 vendor dialect

从当前代码结构看，`ttng` / `amdg` 单独存在，不是为了“分类好看”，而是为了隔离两类无法继续公共化的约束：

1. 硬件对象和协议本身就是专有的  
   例如 NVIDIA 的 `mbarrier`、TMA、proxy fence、TMEM，AMD 的 buffer/TDM 路径、cluster barrier。
2. legality 直接依赖架构 feature、ABI 和目标 ISA 细节  
   这类约束如果强行塞进 `ttg`，公共层 verifier 和 pattern 会被厂商分支淹没。

换句话说，vendor dialect 想解决的不是“表达更多 op”这么简单，而是：

- 把公共层无法稳定承诺的语义，推迟到拥有具体 target knowledge 的层去表达。

这样做的收益有三个：

- `ttg` 还可以维持“公共 GPU contract”这个边界。
- `ttng/amdg` 可以自由携带架构代际约束，不污染前端和公共层。
- 新硬件能力出现时，可以先在 vendor 层落地，不必先改坏公共抽象。

这也是为什么你会看到一个看似“泄漏”的现象：

- 公共 op 主要留在 `ttg`
- 但部分 vendor-specific layout knowledge 已经提前住进了 `ttg` attr

从设计上看，这不是完全理想的边界，而更像是工程上的折中：

- op 语义尽量公共化
- layout knowledge 在必要时允许提前泄漏到公共层
- 真正离不开目标 ISA 的执行协议，再下沉到 vendor dialect

---

## 11. CUDA 同步与异步搬运语义在 Triton 各层 IR 中的位置

这里专门对照你刚看的两类 CUDA feature：

- `4.9 Asynchronous Barriers`
- `4.11 Asynchronous Data Copies`

其中 `4.11` 在 CUDA Guide 里实际覆盖三条异步搬运路径：

- `LDGSTS` / `cp.async` 风格的 global -> shared 异步搬运
- `TMA` / tensormap 驱动的 bulk async tensor copy
- `STAS` 风格的 register/distributed tensor -> shared 异步写入

本节主线仍然围绕“barrier + async copy / async store 的 IR 建模”来整理。

这部分最容易犯的错误，是拿 CUDA C++ API 名字去一一 grep Triton IR。

正确问题应该是：

- 这个 CUDA feature 的“语义核心”是什么
- Triton 有没有保留这个语义核心
- 如果有，是放在 `tt`、`ttg` 还是 vendor dialect
- 如果没有原样出现，是因为不需要、无法公共化，还是应由后续 pass 推导

### 11.1 先给判断原则

判断一个 CUDA feature 会不会在某层 IR 里出现，先看三件事：

1. 它是前端算法语义，还是硬件执行协议。
2. 它能不能抽象成 vendor-neutral contract。
3. 它是不是必须依赖具体硬件对象、地址空间模型或后端分析。

按这个标准，Triton 的分层基本可以概括成：

- `tt` 只保留算法和抽象访存 contract。
- `ttg` 保留公共 GPU 布局、shared memory、async copy、CTA barrier 语义。
- `ttng` / `amdg` 保留硬件专有同步对象和专有数据搬运引擎语义。

### 11.2 对照表：CUDA feature -> Triton IR

| CUDA feature | TTIR `tt` | 公共 TTGIR `ttg` | Vendor IR `ttng` / `amdg` | 为什么这样分层 |
| --- | --- | --- | --- | --- |
| `__syncthreads()` 一类 CTA 执行同步 | 不直接表达 | `ttg.barrier` | 后端再降到具体目标指令 | 这是 GPU 执行与可见性语义，不是前端算法语义 |
| CTA 内存可见性 / fence | 不直接表达 | `ttg.barrier` 的 `addrspace` 位掩码 | 目标后端选具体 barrier / membar 形式 | TTIR 不拥有地址空间同步协议 |
| `cp.async` 风格 async copy + commit/wait group | 不直接表达 | `ttg.async_copy_global_to_local` + `ttg.async_commit_group` + `ttg.async_wait` | 各后端映射到 NVIDIA / AMD 自己的 async path | 这部分存在公共的“异步搬运 contract” |
| `STAS` 风格 async shared store | 不直接表达 | `ttg` 没有公共 async shared store op | NVIDIA: `ttng.async_shared_store` | 这是和 mbarrier / cluster 语义耦合的硬件专有 async store 路径 |
| split arrive/wait 的 `mbarrier` 对象 | 不表达 | 公共层没有完整 `mbarrier` 对象模型 | NVIDIA: `ttng.init_barrier/arrive_barrier/wait_barrier/...`；AMD 也有对应 barrier ops | `mbarrier` 是厂商硬件对象，不适合塞进公共层 |
| barrier 跟踪 async transaction completion | 不表达 | 公共层只到 token / group / wait contract | NVIDIA: `ttng.barrier_expect`、`ttng.async_copy_mbarrier_arrive`；AMD 也有 `amdg.async_copy_mbarrier_arrive` | 这是 barrier 和异步引擎耦合后的硬件协议 |
| TMA / TDM based tensor copy | TTIR 只有抽象 `tt.tensordesc` 访问语义 | `ttg` 不直接出现 TMA / TDM op | NVIDIA: `ttng.async_tma_*`；AMD: `amdg.async_tdm_*` | descriptor 能抽象，具体搬运引擎不能公共化 |
| cluster barrier / cross-CTA sync | 不表达 | `ttg` 没有公共 cluster barrier op | NVIDIA: `ttng.cluster_arrive/wait/barrier`；AMD: `amdg.cluster_barrier_arrive/wait` | cluster scope 不是通用 GPU 抽象 |
| proxy fence / async proxy fence | 不表达 | 公共层没有 | NVIDIA: `ttng.fence_async_shared`、`ttng.tensormap_fenceproxy_acquire` | 这依赖 PTX proxy memory model，强 vendor-specific |
| `arrive_and_drop` | 没有对应 TTIR op | 没有公共 op | 当前也不是按 CUDA API 原样建模 | Triton 建模编译器需要的硬件语义，不逐项复刻 CUDA C++ API |
| completion function | 不表达 | 不表达 | 不表达 | 这是 CUDA C++ library 抽象，不是后端必须保留的 IR 对象 |
| warp entanglement / convergence 要求 | 不作为 op 表达 | 不作为 op 表达 | 只体现在 verifier、lowering 前提和硬件约束里 | 这是使用约束，不是独立 SSA 语义 |

### 11.3 哪些 feature 已经明确表达在 IR 里

#### 11.3.1 CTA barrier 和地址空间可见性

`ttg.barrier` 是公共层里最重要的同步原语。

见：
[include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:734](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:734)

它表达的不是单纯“线程都停一下”，而是：

- CTA 级执行同步
- 指定地址空间上的可见性发布

而且 `addrspace` 不是单个布尔位，而是可组合的位掩码，支持：

- `local`
- `global_read`
- `global_write`
- `tensor_read`
- `tensor_write`

所以在 Triton 公共层里，barrier 已经不是纯 control barrier，而是带 memory visibility contract 的 barrier。

#### 11.3.2 公共 async copy pipeline

`ttg` 公共层明确有：

- `ttg.async_copy_global_to_local`
- `ttg.async_commit_group`
- `ttg.async_wait`

定义见：
[include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:47)

这一组 op 表达的是 vendor-neutral contract：

- 发起 global -> local 的异步搬运
- 用 commit 把一批 copy 组成 group
- 用 wait 控制 outstanding group 数量

这和 CUDA 4.11 里 async copy pipeline 的核心思想是一致的，但公共层不承诺底层一定是 NVIDIA `cp.async`、TMA，还是 AMD 的另一套异步搬运路径。

#### 11.3.3 NVIDIA 的 `mbarrier` / TMA / proxy fence

NVIDIA vendor 层把很多 CUDA / PTX 专有语义显式建模成 op。

见：
[include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:257](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:257)

这一层里和你看的 4.9 / 4.11 最直接对应的有：

- `ttng.init_barrier`
- `ttng.inval_barrier`
- `ttng.barrier_expect`
- `ttng.wait_barrier`
- `ttng.arrive_barrier`
- `ttng.async_copy_mbarrier_arrive`
- `ttng.async_shared_store`
- `ttng.async_tma_copy_global_to_local`
- `ttng.async_tma_copy_local_to_global`
- `ttng.async_tma_reduce`
- `ttng.async_tma_gather`
- `ttng.async_tma_scatter`
- `ttng.async_tma_store_wait`
- `ttng.fence_async_shared`
- `ttng.fence_mbarrier_init_release_cluster`
- `ttng.cluster_arrive`
- `ttng.cluster_wait`
- `ttng.cluster_barrier`

这说明 Triton 不是“不懂 CUDA 这些机制”，而是把它们放在 vendor 层表达。

#### 11.3.4 AMD 对应机制

AMD vendor 层也有一套平行表达：

- `amdg.init_barrier`
- `amdg.wait_barrier`
- `amdg.arrive_barrier`
- `amdg.async_copy_mbarrier_arrive`
- `amdg.async_tdm_copy_global_to_local`
- `amdg.async_tdm_copy_local_to_global`
- `amdg.async_tdm_scatter`
- `amdg.async_tdm_gather`
- `amdg.async_tdm_wait`
- `amdg.cluster_barrier_arrive`
- `amdg.cluster_barrier_wait`

见：
[third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:1135](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:1135)

所以“异步 barrier + async copy engine + cluster sync”不是 NVIDIA 独有的概念，但它们在 Triton 里依然主要落在 vendor dialect，而不是 `tt` 或 `ttg`。

### 11.4 哪些没有在 TTIR 里表达

这点要说得非常明确：

- `tt` 层基本没有 `barrier`
- 没有 `fence`
- 没有 `mbarrier`
- 没有 `async_copy_*`
- 没有 `tma`
- 没有 `cluster_*`

也就是说，TTIR 不直接承载 CUDA 4.9 / 4.11 这些硬件同步协议对象。

这不是缺失，而是分层设计。

TTIR 的职责仍然是：

- 算法语义
- 抽象 pointer / tensor descriptor
- 抽象 load/store/reduce/dot/scan

而不是：

- 哪一代 GPU 的 barrier object
- 哪一种异步搬运硬件引擎
- 哪一种 proxy fence

### 11.5 为什么这些东西不该放在 TTIR

#### 11.5.1 因为它们不是前端算法语义

`mbarrier`、TMA、proxy fence、cluster barrier 这些东西，回答的是：

- 怎样在某种 GPU 上安全执行
- 怎样和硬件异步引擎通信
- 怎样保证跨线程或跨 CTA 的可见性

它们不是“我要算什么”的一部分。

#### 11.5.2 因为很多 feature 不能公共化

例如：

- `ttng.fence_async_shared`
- `ttng.tensormap_fenceproxy_acquire`

这种 op 依赖的是 NVIDIA / PTX 的 proxy memory model。  
如果把它们塞进 `tt` 或 `ttg`，公共层就会被某家 ISA 的细节污染。

#### 11.5.3 因为很多 barrier / fence 需要后续分析才能插入

Triton 里有不少 barrier / fence 不是前端写出来的，而是后续 pass 基于依赖分析自动补的。

直接证据有：

- NVIDIA fence insertion pass：
  [include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:43](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/Transforms/Passes.td:43)
- proxy fence insertion 实现：
  [lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:107](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp:107)
- shared-memory membar 分析：
  [include/triton/Analysis/Membar.h:208](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Analysis/Membar.h:208)

这说明很多同步点的位置依赖：

- shared memory alias
- producer / consumer 顺序
- async proxy 访问路径
- buffer 重用关系

这些信息在 TTIR 阶段通常还不完整。

#### 11.5.4 因为有些 CUDA feature 本质上只是上层 API 抽象

例如：

- completion function
- `arrive_and_drop`
- warp entanglement 的使用建议

这些内容对 CUDA 程序员很重要，但不一定需要在 Triton IR 里成为一个独立 op。  
编译器只会保留生成正确硬件协议所必需的那部分语义。

### 11.6 最实用的记忆方式

把 CUDA 4.9 / 4.11 这些 feature 分成三类记：

1. 能公共化的语义  
   例如 CTA barrier、async copy group / wait contract。它们进入 `ttg`。
2. 只能厂商化的语义  
   例如 `mbarrier`、TMA / TDM、proxy fence、cluster barrier。它们进入 `ttng` / `amdg`。
3. 只是上层 API 写法，不是 IR 必须对象  
   例如 completion function、`arrive_and_drop`、warp entanglement 描述。它们通常不会作为一条 IR op 原样出现。

所以读 IR 时，不要追“有没有同名 op”，而要追：

- 语义核心有没有被保留
- 被保留在哪一层
- 是公共 contract，还是硬件专有 contract

### 11.7 从设计意图回看这张映射表

如果只看结果，你容易得到一个表面印象：

- 有些 CUDA feature 在 `ttg`
- 有些在 `ttng/amdg`
- 有些根本不在 IR 里

但从设计意图看，它们其实对应三种不同命运：

1. 语义核心可以公共化  
   例如 CTA barrier、async copy group/wait contract，所以进入 `ttg`。
2. 语义核心离不开厂商硬件协议  
   例如 `mbarrier`、TMA、proxy fence、cluster barrier，所以进入 `ttng/amdg`。
3. 语义本质上只是 API 层组织方式，不是编译器必须长期保留的 IR 对象  
   例如 completion function、`arrive_and_drop`，所以不一定进 IR。

这样再回头看第 11.2 的表，就不应该把它理解成“有没有对应名字”，而应该理解成：

- Triton 在哪一层认为这件事已经值得成为显式 contract
- 哪一层才真正拥有足够上下文去承载它

### 11.8 如果没有这层设计，会出什么问题

前面一直在正面描述：

- `tt` 保留算法 contract
- `ttg` 显式化公共 GPU contract
- `ttng/amdg` 承载 vendor-specific contract

这一节换个角度，直接看如果没有这套分层，系统会坏在哪里。

#### 11.8.1 如果把硬件协议直接塞进 TTIR

最直接的问题是前端抽象会被硬件细节污染。

例如如果在 `tt` 里直接大量出现：

- `mbarrier`
- TMA / TDM
- proxy fence
- cluster barrier
- vendor-specific shared layout

那前端就不再只是表达“要算什么”，而是在过早决定：

- 用哪家的异步引擎
- 用哪家的 barrier object
- 用哪家的 memory model

这样会带来四个后果：

1. 可移植性下降  
   同一份前端语义很难同时服务 NVIDIA 和 AMD。
2. 优化空间收缩  
   很多本该在更高层做的重写，会被过早绑定的硬件路径卡死。
3. 前端负担失控  
   写 kernel 的人等于被迫理解后端硬件协议。
4. IR 语义混乱  
   `tt` 既像算法 IR，又像 target IR，边界失真。

换句话说，如果没有 `tt` 这层“只保留算法 contract”的克制，前端和后端会直接耦死。

#### 11.8.2 如果没有 TTGIR 这层公共 GPU contract

这是另一个极端。

假设系统只有：

- `tt`
- 直接 lowering 到 `ttng` / `amdg`

那很多 GPU 执行必需的信息就只能靠后端自己“猜”出来：

- tensor 怎么映射到线程
- shared memory 对象的 shape / view / mutability
- 哪些异步 copy 属于同一组
- wait 到底在等谁
- 哪些 barrier 需要 local 可见性，哪些还要 tensor/global 可见性

这会导致三个问题：

1. legality 难以稳定验证  
   因为关键 contract 不在 IR 上，而是散在分析器和 pattern 里。
2. backend pattern 过重  
   NVIDIA / AMD 后端都要自己补出大量公共 GPU 语义，重复劳动。
3. pass 很脆弱  
   relayout、pipelining、shared memory allocation、barrier insertion 缺少统一抓手。

所以 `ttg` 这层的价值，不只是“多一个 dialect”，而是：

- 把后端共同需要的 GPU 事实，先稳定成可检查、可传递、可重写的公共 contract。

#### 11.8.3 如果没有 layout encoding 的显式化

很多 Triton 新手一开始会低估这一点，以为 layout 只是“实现细节”。

但如果 layout 不显式化，后果非常具体：

- `convert_layout` 无从成立
- shared memory swizzle / MMA operand layout 无法系统验证
- 同一个 tensor 在不同 pass 里的线程分工关系无法可靠追踪
- warp specialization 改变 warps 数量后，relayout 几乎没有稳定入口

这也是为什么 `convert-triton-to-tritongpu` 的描述会把“给 tensor type 增加 layout encoding”写成核心工作之一，而不是附带动作。

#### 11.8.4 如果没有 `memdesc`

如果 shared memory 仍然只被看成“一个 pointer”或者“一个普通 tensor”，会立刻失去几类关键能力：

- 同一块底层存储的多 view 建模
- subview / reshape / reinterpret 的合法表达
- shared buffer 生命周期与 mutability 的显式跟踪
- barrier / async wait 对具体 shared object 的依赖建模

源码里 `ttg.memdesc` 明确就是为了解决这个问题：  
shared memory 对象不是一个单纯值，它既有底层存储，又有逻辑视图。

如果没有 `memdesc`，很多现在能在 IR 上直接表达的 shared-memory 语义，只能退化成：

- 后端约定
- pattern 特判
- 或者不可组合的 ad-hoc op

#### 11.8.5 如果没有 async token / commit / wait 的 SSA 链

如果异步操作只靠“这个 op 语义上是 async”这种隐含约定，而没有：

- `ttg.async.token`
- `ttg.async_commit_group`
- `ttg.async_wait`

那编译器就很难在 IR 层回答这些问题：

- 哪些 async copy 属于同一组
- 某个 wait 到底对应哪批异步操作
- outstanding 数量的约束在哪里
- 哪些 reorder 是合法的

结果就是：

- pipelining 很难系统化
- wait 插入和移动容易出错
- barrier / async completion 的组合关系只能写死在 lowering 里

所以 async token 的真正价值，不是“多了个类型”，而是：

- 把异步依赖从注释和约定里，提升成 SSA 图的一部分。

#### 11.8.6 如果没有 vendor dialect

如果强行把所有能力都塞进 `ttg`，公共层会很快失控。

因为 `ttg` 将不得不同时承载：

- NVIDIA 的 `mbarrier` / TMA / proxy fence / TMEM / cluster
- AMD 的 buffer path / TDM / cluster barrier / LDS 特定路径

那会造成三件事：

1. verifier 和 pattern 爆炸  
   几乎每个公共 op 都要带 target-specific 分支。
2. 公共抽象名存实亡  
   `ttg` 表面叫公共层，实际上已经是多家 ISA 细节的拼盘。
3. 新硬件演进难以收敛  
   每来一代新 feature，都得先改公共层定义。

所以 vendor dialect 的真正作用，不只是“把 op 分文件放”，而是：

- 给 target-specific legality、ABI、feature gating 一个有边界的承载位置。

#### 11.8.7 如果把本该是 API 抽象的东西也都做成 IR 对象

这同样会出问题。

例如 CUDA 文档里的：

- completion function
- `arrive_and_drop`
- 一些 convergence / entanglement 使用建议

如果全部原样做成 IR 对象，IR 会承担很多并非 lowering 必需的信息。

坏处包括：

- IR 变重，但没有带来对应的可验证性收益
- backend 仍然只消费其中一部分硬件核心语义
- 读 IR 的人会误以为这些 API 表层组织方式也是后端必须长期保留的语义边界

所以一个健康的 IR 设计，不是“把上层 API 全复刻下来”，而是：

- 只保留那些对 legality、dependency、lowering、codegen 真正必要的 contract。

#### 11.8.8 一句话总结这节

如果没有这套分层，坏结果并不是抽象上的“不优雅”，而是非常工程化的失效：

- 前端被硬件污染
- 公共 GPU contract 无法稳定验证
- shared memory 和 async dependency 难以显式追踪
- backend pattern 过重
- vendor-specific feature 把公共层冲垮

所以 `tt -> ttg -> ttng/amdg` 的设计，不只是为了“把 IR 分三层”，而是为了同时满足：

- 前端语义稳定
- 中层 contract 可验证
- 后端 target knowledge 可隔离
- 新硬件特性可演进

---

## 12. 学习时最值得抓住的约束

### 12.1 TTIR 的主约束

TTIR 最重要的是这四类一致性：

- 类型一致性
- shape 一致性
- pointer / pointee 一致性
- region 协议一致性

TTIR 还没有真正进入“硬件布局合法性”主导的世界。

### 12.2 TTGIR 的主约束

TTGIR 最重要的是这五类一致性：

- tensor encoding 合法
- memdesc encoding 合法
- module GPU 参数一致
- async token / wait 依赖一致
- barrier / shared memory 使用一致

### 12.3 Vendor 层的主约束

到了 `ttng` / `amdg`，主约束再往前走一步，变成：

- 架构 feature 是否存在
- 该家硬件 ABI 是否满足
- 该家 lowering pattern 是否能完整消费这类 op

这时“类型合法”只是最低要求，不是全部要求。

---

## 13. 最推荐的阅读顺序

如果要系统看源码，我建议按这个顺序读。

1. `tt` 的 dialect 和 types  
   先搞清楚 TTIR 想表达什么，不要一开始就陷进 GPU layout。
2. `tt` 的 op 分组  
   重点看 `load/store`、`dot`、`reduce/scan`、descriptor。
3. `convert-triton-to-tritongpu` pass 描述  
   理解 TTIR 到 TTGIR 究竟新增了什么 contract。
4. `ttg` types + attrs + ops  
   重点看 memdesc、layout、async、warp_specialize。
5. `ttng` 和 `amdg`  
   重点看哪些是公共层表达不了，为什么必须单独起 vendor op。

如果只想抓主线，可以按这组文件读：

- [include/triton/Dialect/Triton/IR/TritonTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonTypes.td:1)
- [include/triton/Dialect/Triton/IR/TritonOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/Triton/IR/TritonOps.td:1)
- [include/triton/Conversion/TritonToTritonGPU/Passes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Conversion/TritonToTritonGPU/Passes.td:1)
- [include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td:1)
- [include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:1)
- [include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:1)
- [include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:1)
- [third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td](/LocalRun/jiangzhe.zhao/my_repo/triton/third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:1)

---

## 14. 最后压成一句话

一句话区分四层：

```text
TTIR (`tt`) 负责表达“我要算什么”。
公共 TTGIR (`ttg`) 负责表达“这件事如何在 GPU 上按合法布局和同步协议执行”。
`ttng` / `amdg` 负责表达“当公共层不够时，各家硬件特有的执行机制是什么”。
```

再压一句约束主线：

```text
TTIR 的主约束是类型/shape/语义一致性。
TTGIR 的主约束是 layout/memdesc/async/synchronization 一致性。
Vendor dialect 的主约束是架构 feature 和硬件 ABI 一致性。
```
