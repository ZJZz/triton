# 2026-06-29 学习笔记：AccelerateMatmul

## 1. Pass 基本信息

本轮学习 `TritonGPUAccelerateMatmul`，重点看 canonical matmul dumps 里
`num_ctas=1` 的三架构对比：

- Ampere: `sm86_num_ctas1`
- Hopper: `sm90_num_ctas1`
- Blackwell: `sm100_num_ctas1`

相关文件：

- Ampere before: [034_Before_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/034_Before_TritonGPUAccelerateMatmul.mlir:1)
- Ampere after: [035_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:1)
- Hopper before: [032_Before_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:1)
- Hopper after: [033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:1)
- Blackwell before: [032_Before_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:1)
- Blackwell after: [033_After_TritonGPUAccelerateMatmul.mlir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:1)
- Source: [AccelerateMatmul.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:489)
- Pass driver: [AccelerateMatmul.cpp](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:1023)

一句话：

```text
AccelerateMatmul 把 generic TTGIR tt.dot 改写成目标架构可执行的 tensor-core contract：
Ampere 走 MMA v2，Hopper 走 WGMMA / MMA v3，Blackwell 走 pass-internal
MMA v5 decision，对应 IR 表征是 TCGen5 + TMEM。
```

## 2. Architecture Matrix

| Arch | Before feature | After feature | Changed? |
| --- | --- | --- | --- |
| Ampere `sm86` | generic `tt.dot`，A/B 是 `#ttg.dot_op<{parent = #blocked}>` | `#ttg.nvidia_mma<{versionMajor = 2, instrShape = [16, 8]}>`，仍是 `tt.dot`，但 A/B/accumulator 都换到 MMA layout | yes |
| Hopper `sm90` | generic `tt.dot`，形态和 Ampere before 基本相同 | `#ttg.nvidia_mma<{versionMajor = 3, instrShape = [16, 64, 16]}>`，A/B local_alloc 到 shared memory，dot 变成 `ttng.warp_group_dot` | yes |
| Blackwell `sm100` | generic `tt.dot`，形态和 Ampere/Hopper before 基本相同 | pass 内部选择 v5；IR 不出现 `#ttg.nvidia_mma<versionMajor = 5>`，而是 A/B local_alloc 到 shared memory，accumulator 进入 `#tmem`，dot 变成 `ttng.tc_gen5_mma` | yes |

结论：三代进入这个 pass 前的 dot 形态基本同构；架构分叉主要由
`AccelerateMatmul` 本身造成，而不是 before IR 已经大幅分叉。

## 3. Before IR：三代基本同形

三代 before 的核心形态相同。进入这个 pass 前，loop body 里的核心形态是：

- `tt.load` 读 A/B
- A/B 从 blocked layout 转成 `#ttg.dot_op`
- 普通 `tt.dot` 直接返回 blocked accumulator

Ampere 证据：

- [034_Before_TritonGPUAccelerateMatmul.mlir:63](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/034_Before_TritonGPUAccelerateMatmul.mlir:63)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/034_Before_TritonGPUAccelerateMatmul.mlir:67):

```text
%a = tt.load %a_ptrs_40 : tensor<64x32x!tt.ptr<f16>, #blocked1>
%b = tt.load %b_ptrs_41 : tensor<32x64x!tt.ptr<f16>, #blocked2>
%a_43 = ttg.convert_layout %a
  -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
%b_44 = ttg.convert_layout %b
  -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
%acc_45 = tt.dot %a_43, %b_44, %acc_42
  -> tensor<64x64xf32, #blocked>
```

Hopper before 证据：

- [032_Before_TritonGPUAccelerateMatmul.mlir:63](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:63)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:67):

```text
%a = tt.load %a_ptrs_40 : tensor<64x32x!tt.ptr<f16>, #blocked1>
%b = tt.load %b_ptrs_41 : tensor<32x64x!tt.ptr<f16>, #blocked2>
%a_43 = ttg.convert_layout %a
  -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
%b_44 = ttg.convert_layout %b
  -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
%acc_45 = tt.dot %a_43, %b_44, %acc_42
  -> tensor<64x64xf32, #blocked>
```

Blackwell before 证据：

- [032_Before_TritonGPUAccelerateMatmul.mlir:63](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:63)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:67):

```text
%a = tt.load %a_ptrs_40 : tensor<64x32x!tt.ptr<f16>, #blocked1>
%b = tt.load %b_ptrs_41 : tensor<32x64x!tt.ptr<f16>, #blocked2>
%a_43 = ttg.convert_layout %a
  -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
%b_44 = ttg.convert_layout %b
  -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
%acc_45 = tt.dot %a_43, %b_44, %acc_42
  -> tensor<64x64xf32, #blocked>
```

这个 before contract 还没有明确选择具体 NVIDIA tensor-core generation。它只说明：

```text
logical dot 已经存在；
A/B 是 dot operand layout；
accumulator 是 blocked layout；
target capability 写在 module attribute 里。
```

## 4. Ampere After：MMA v2 / mma.sync contract

Ampere after 新增：

- [035_After_TritonGPUAccelerateMatmul.mlir:6](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:6)

```text
#mma = #ttg.nvidia_mma<{
  versionMajor = 2,
  versionMinor = 0,
  warpsPerCTA = [2, 2],
  instrShape = [16, 8]
}>
```

核心 IR 变化：

- [035_After_TritonGPUAccelerateMatmul.mlir:68](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:68)
  到
  [72](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:72):

```text
%acc_45 = ttg.convert_layout %acc_42
  : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
%a_46 = ttg.convert_layout %a_43
  -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
%b_47 = ttg.convert_layout %b_44
  -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
%acc_48 = tt.dot %a_46, %b_47, %acc_45
  -> tensor<64x64xf32, #mma>
%acc_49 = ttg.convert_layout %acc_48
  : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
```

IR 层面的意思：

- 原来的 `tt.dot` 没有被替换成 NVIDIA dialect op。
- 但它的 result / accumulator layout 被改成 `#ttg.nvidia_mma`。
- A/B dot operands 的 parent 也从 `#blocked` 改成 `#mma`。
- 这给后续 MMA lowering 一个明确 contract：这个 dot 应该按 MMA v2 fragment layout
  lower，最终对应 Ampere-era `mma.sync` 路径。

## 5. Hopper After：MMA v3 / WGMMA contract

Hopper after 新增：

- [033_After_TritonGPUAccelerateMatmul.mlir:6](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:6)
  到
  [9](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:9):

```text
#mma = #ttg.nvidia_mma<{
  versionMajor = 3,
  versionMinor = 0,
  warpsPerCTA = [4, 1],
  instrShape = [16, 64, 16]
}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, ...}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, ...}>
#smem = #ttg.shared_memory
```

核心 IR 变化：

- [033_After_TritonGPUAccelerateMatmul.mlir:67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:67)
  到
  [73](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:73):

```text
%a = tt.load %a_ptrs_40 : tensor<64x32x!tt.ptr<f16>, #blocked1>
%a_43 = ttg.local_alloc %a
  : (tensor<64x32xf16, #blocked1>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
%b = tt.load %b_ptrs_41 : tensor<32x64x!tt.ptr<f16>, #blocked2>
%b_44 = ttg.local_alloc %b
  : (tensor<32x64xf16, #blocked2>) -> !ttg.memdesc<32x64xf16, #shared1, #smem>
%acc_45 = ttg.convert_layout %acc_42
  : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #mma>
%acc_46 = ttng.warp_group_dot %a_43, %b_44, %acc_45
  -> tensor<64x64xf32, #mma>
%acc_47 = ttg.convert_layout %acc_46
  : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
```

IR 层面的意思：

- Hopper 不再把 A/B 保持为 register-side `#ttg.dot_op` operands。
- A/B 先被 materialize 成 shared-memory memdesc。
- 原来的 `tt.dot` 被替换成 `ttng.warp_group_dot`。
- `#mma versionMajor = 3` 表达 accumulator/result 的 WGMMA fragment contract。

这就是 Hopper-era WGMMA 路径的关键差异：operand contract 从 register dot operand
变成 shared-memory operand + warp-group dot。

## 6. Blackwell After：pass-internal MMA v5 / TCGen5 + TMEM contract

注意：这里的 v5 是 `getMMAVersionSafe` 在 pass 内部选择的逻辑版本号，不是
after IR 里的 `#ttg.nvidia_mma` attribute。Blackwell after IR 没有
`#ttg.nvidia_mma<versionMajor = 5>`；它的可见 IR 表征是 `#linear`、`#tmem`、
`ttng.tmem_alloc`、`ttng.tc_gen5_mma` 和 `ttng.tmem_load`。

Blackwell after 新增：

- [033_After_TritonGPUAccelerateMatmul.mlir:5](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:5)
  到
  [10](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:10):

```text
#linear = #ttg.linear<{...}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, ...}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, ...}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
```

核心 IR 变化：

- [033_After_TritonGPUAccelerateMatmul.mlir:69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:69)
  到
  [77](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:77):

```text
%a = tt.load %a_ptrs_40 : tensor<64x32x!tt.ptr<f16>, #blocked1>
%a_43 = ttg.local_alloc %a
  : (tensor<64x32xf16, #blocked1>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
%b = tt.load %b_ptrs_41 : tensor<32x64x!tt.ptr<f16>, #blocked2>
%b_44 = ttg.local_alloc %b
  : (tensor<32x64xf16, #blocked2>) -> !ttg.memdesc<32x64xf16, #shared1, #smem>
%acc_45 = ttg.convert_layout %acc_42
  : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #linear>
%acc_46, %acc_47 = ttng.tmem_alloc %acc_45
  -> (!ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
%acc_48 = ttng.tc_gen5_mma %a_43, %b_44, %acc_46[%acc_47], %true, %true
%acc_49, %acc_50 = ttng.tmem_load %acc_46[%acc_48]
  -> tensor<64x64xf32, #linear>
%acc_51 = ttg.convert_layout %acc_49
  : tensor<64x64xf32, #linear> -> tensor<64x64xf32, #blocked>
```

IR 层面的意思：

- Blackwell operand path 和 Hopper 一样，需要 A/B in shared memory。
- accumulator 不再只是 `#ttg.nvidia_mma` tensor layout，而是被放进 Tensor Memory。
- `#linear` 是 TMEM load/store 边界使用的 distributed register layout；这里作为
  `tmem_alloc` 输入和 `tmem_load` 输出的寄存器侧布局。
- `ttng.tmem_alloc` 建立 mutable TMEM accumulator。
- `ttng.tc_gen5_mma` 对 TMEM accumulator 发起 TCGen5 MMA。
- `ttng.tc_gen5_mma ... %true, %true` 的两个 true 分别来自源码里的 `useD` 和
  `pred` 参数，表示使用已有 D accumulator 并启用该 MMA。
- `ttng.tmem_load` 把结果从 TMEM 读回 distributed tensor。

所以 Blackwell 的 output contract 明显更重：除了 tensor-core instruction contract，还建立
TMEM allocation / async token / load-back contract，给后续 TMEM、barrier、allocation、
lowering passes 消费。

## 7. Source Mapping

### 7.1 版本选择：getMMAVersionSafe

源码：

- [AccelerateMatmul.cpp:41](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:41)
  到
  [83](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:83)

简化逻辑：

```text
if computeCapability < 75:
  try MMA v1
else if computeCapability < 90:
  try MMA v2
else if computeCapability < 100:
  try MMA v3, then v2 fallback
else if computeCapability < 120:
  if computeCapability == 103 and dot is int8 x int8:
    try MMA v2
  else:
    try MMA v5, then v2 fallback
else if computeCapability < 130:
  try MMA v2
```

当前三个 dump 对应：

```text
sm86  -> IR has #ttg.nvidia_mma versionMajor = 2
sm90  -> IR has #ttg.nvidia_mma versionMajor = 3
sm100 -> pass chooses versionMajor = 5 internally; IR uses tc_gen5_mma + TMEM
```

### 7.2 MMA v2 / v3 通用 pattern：BlockedToMMA

源码：

- pattern 入口见 [AccelerateMatmul.cpp:489](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:489)
- 创建 `#ttg.nvidia_mma` encoding 见 [AccelerateMatmul.cpp:441](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441)
  到
  [472](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:472)
- v3 shared-memory / `WarpGroupDotOp` 分支见 [AccelerateMatmul.cpp:540](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:540)
  到
  [559](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:559)
- v2 register operand / new `DotOp` 分支见 [AccelerateMatmul.cpp:560](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:560)
  到
  [569](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:569)
- 用 convert 把新 dot result 转回旧 result type 见 [AccelerateMatmul.cpp:572](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:572)

对应 IR：

```text
v2:
  old tt.dot -> new tt.dot with #mma parent operands/result

v3:
  tt.load -> ttg.local_alloc -> ttng.warp_group_dot -> convert back
```

### 7.3 MMA v5 pattern：BlockedToMMAv5

源码：

- pattern 入口见 [AccelerateMatmul.cpp:629](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:629)
- v5 检查见 [AccelerateMatmul.cpp:650](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:650)
- A/B shared-memory operand 创建见 [AccelerateMatmul.cpp:671](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:671)
  到
  [673](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:673)
- TMEM encoding / memdesc 创建见 [AccelerateMatmul.cpp:679](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:679)
  到
  [687](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:687)
- accumulator convert / `TMEMAllocOp` / `TCGen5MMAOp` / `TMEMLoadOp` 见
  [AccelerateMatmul.cpp:688](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:688)
  到
  [704](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:704)

对应 IR：

```text
tt.load A/B
  -> ttg.local_alloc into #ttg.nvmma_shared
accumulator
  -> convert to #linear
  -> ttng.tmem_alloc
  -> ttng.tc_gen5_mma
  -> ttng.tmem_load
  -> convert back to #blocked
```

### 7.4 Pass driver

`runOnOperation()` 注册了多个 greedy rewrite patterns：

- [AccelerateMatmul.cpp:1045](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:1045)
  到
  [1049](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:1049)

```text
BlockedToMMA
ScaledBlockedToMMA
DecomposeScaledBlocked
BlockedToMMAv5
ScaledBlockedToMMAv5
```

当前普通 f16 matmul 主要走：

```text
sm86:  BlockedToMMA, versionMajor = 2
sm90:  BlockedToMMA, versionMajor = 3
sm100: BlockedToMMAv5, internal versionMajor = 5, visible IR = TCGen5 + TMEM
```

## 8. Compiler Decision

这个 pass 回答的 compiler question 是：

```text
当前 logical tt.dot 应该匹配哪一种 NVIDIA tensor-core execution contract？
```

具体 decision：

- Ampere `sm86`: 使用 MMA v2 layout，后续 lower 到 `mma.sync` 类路径。
- Hopper `sm90`: 使用 MMA v3 / WGMMA contract，A/B 必须是 shared-memory operands，
  dot 变成 `ttng.warp_group_dot`。
- Blackwell `sm100`: 使用 MMA v5 / TCGen5 contract，A/B 是 shared-memory operands，
  accumulator 是 TMEM memdesc。这里的 MMA v5 是 pass 内部 decision，IR 不落
  `#ttg.nvidia_mma<versionMajor = 5>`。

为什么在这里做：

- `ConvertTritonToTritonGPU` / `Coalesce` / `PlanCTA` / first
  `RemoveLayoutConversions` 已经建立了基本 TTGIR layout 和 clean dot surroundings。
- 但还没有进入 pipeline、shared/TMEM allocation、MMA lowering、LLVM lowering。
- 这个位置还能在 TTGIR/NvidiaGPU 层重写 dot，并给后续 pass 一个明确 contract。

## 9. Compiler Contract

Input contract:

- IR 已经是 TTGIR。
- `tt.dot` 的 A/B operand 已经有 `#ttg.dot_op` encoding。
- accumulator/result 有 distributed blocked layout。
- module 上有 `ttg.target = "cuda:<cc>"`。
- `ttg.num-warps` / `ttg.num-ctas` 已知。

Output contract:

- Ampere: dot result/accumulator 使用 `#ttg.nvidia_mma` v2 encoding，A/B dot operands
  的 parent 是 `#mma`。
- Hopper: A/B operands 被 materialize 成 `#ttg.nvmma_shared` memdesc，dot 变成
  `ttng.warp_group_dot`，accumulator/result 使用 MMA v3 encoding。
- Blackwell: A/B operands 被 materialize 成 shared memdesc，accumulator 被 materialize
  成 TMEM memdesc，dot 变成 `ttng.tc_gen5_mma`，并通过 token 和 `tmem_load` 把结果读回。

Deferred work:

- 不做 pipelining 或 async scheduling；这些留给后续 pipeline / schedule passes。
- 不决定 shared memory 的最终 offset；`ttg.local_alloc` 只是创建 memdesc，物理分配留给
  `AllocateSharedMemory` 类 pass。
- 不清理自己插入的所有 layout boundary；新出现的 `ttg.convert_layout` 留给后续
  `RemoveLayoutConversions` 和相关 layout cleanup。
- 不把 `tt.dot` / `ttng.warp_group_dot` / `ttng.tc_gen5_mma` 最终 lower 到 LLVM/NVVM；
  这留给 MMA lowering 和更低层 conversion。

Next pass relies on:

- `OptimizeDotOperands` 可以基于已经选好的 dot/MMA contract 优化 operand layout。
- later `RemoveLayoutConversions` 可以清理新插入的 layout boundary。
- pipeline / schedule passes 可以围绕 shared-memory operand、WGMMA、TMEM token 建立
  latency hiding 和 ordering。
- `TritonNvidiaGPUMMALoweringPass` 和后续 LLVM/NVVM lowering 可以把这些 contract 继续
  lower 到具体 NVIDIA instruction path。

## 10. Invariant

这个 pass 改了很多 IR，但不改变数学语义。

Invariant:

- Tensor logical shape unchanged:
  - A: `64x32`
  - B: `32x64`
  - accumulator/result: `64x64`
- Element type unchanged:
  - A/B: `f16`
  - accumulator: `f32`
  - final store前仍 truncate 到 `f16`
- Program semantics unchanged:
  - 仍然是同一个 matmul K-loop。
  - 每轮仍然执行 `acc += dot(A_tile, B_tile)`。
- Changed only:
  - tensor-core generation
  - operand storage contract
  - accumulator layout/storage
  - shared memory / TMEM / token-level IR expression

## 11. Triton Mechanism vs Hardware Reason

Triton mechanism:

- 在 TTGIR / TritonNvidiaGPU IR 层改写 `tt.dot`。
- 为 dot result 创建 `#ttg.nvidia_mma`、`#linear` 或 TMEM-compatible layout。
- 为 A/B operand 创建 `#ttg.dot_op` parent、shared-memory memdesc，或 TMEM-related operand。
- 用 `ttg.convert_layout` 保持进入/离开 dot contract 的边界。

Hardware / optimization reason:

- Ampere:
  - instruction reason: `mma.sync` 需要特定 operand/result fragment layout。
  - execution reason: warp-level tensor-core instruction。
- Hopper:
  - instruction reason: WGMMA 从 shared memory 读取 operands。
  - execution reason: warp-group-level dot，`warpsPerCTA = [4, 1]`。源码
    [AccelerateMatmul.cpp:135](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:135)
    到
    [147](/LocalRun/jiangzhe.zhao/my_repo/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:147)
    说明 MMAv3 最小不可分 warp shape 是 `(4, 1)`；当前 4 warp CTA 正好形成一个
    沿 M 维排布的 warp group。
  - memory reason: A/B 需要 `nvmma_shared` swizzled shared-memory layout。
- Blackwell:
  - instruction reason: TCGen5 MMA 使用 shared operands 和 TMEM accumulator。
  - hardware reason: accumulator 进入 Tensor Memory。
  - synchronization reason: IR 开始出现 async token dependency，后续 barrier/fence pass 会消费。

## 12. Decision Tree

当前普通 `tt.dot` 的高层决策树可以写成：

```text
for each tt.dot:
  read compute capability from module target
  choose highest supported MMA version:
    sm80/sm86 -> v2
    sm90      -> v3, fallback v2
    sm100     -> v5, fallback v2, except unsupported sm103 int8 dot -> v2

  if version == 2:
    create #ttg.nvidia_mma v2 result layout
    convert accumulator to #mma
    convert A/B to dot_op parent=#mma
    create new tt.dot returning #mma
    convert result back to old result layout

  else if version == 3:
    create #ttg.nvidia_mma v3 result layout
    local_alloc A/B into nvmma_shared
    convert accumulator to #mma
    create ttng.warp_group_dot
    convert result back to old result layout

  else if version == 5:
    local_alloc A/B into nvmma_shared
    create tensor-memory accumulator encoding
    convert accumulator to TMEM load/store distributed layout
    tmem_alloc accumulator
    create ttng.tc_gen5_mma
    tmem_load result
    convert result back to old result layout
```

## 13. Alternative Design

Alternative 1: 保持 generic `tt.dot`，等到更晚的 MMA lowering 再选择 v2/v3/v5。

为什么这里不这么做：

- 后续 pipeline、operand optimization、shared/TMEM/barrier passes 需要提前看到真实的
  operand/result contract。
- Hopper/Blackwell 的 shared-memory/TMEM path 不是单纯 final lowering 可以局部决定的；
  它会影响 scheduling、allocation、barrier 和 layout cleanup。

Alternative 2: Hopper 继续使用 MMA v2 fallback。

为什么不优先这么做：

- `getMMAVersionSafe` 对 `sm90` 优先选择 v3，只有 shape/type 不支持时才 fallback v2。
- WGMMA 是 Hopper-era tensor-core path，能表达 warp-group 和 shared-memory operand
  contract。

Alternative 3: Blackwell fallback 到 v2。

为什么不优先这么做：

- `sm100` 优先选择 v5。v5 能表达 TCGen5 + TMEM path，这是 Blackwell-specific
  compiler/hardware contract。

## 14. If This Pass Did Not Exist

如果没有 `AccelerateMatmul`：

- IR 中只有 generic `tt.dot` 和 `#ttg.dot_op parent=#blocked`。
- 后续 pass 无法区分这个 dot 应该走 Ampere MMA v2、Hopper WGMMA，还是 Blackwell TCGen5。
- Hopper 不会提前出现 shared-memory operand memdesc，pipeline/scheduling 无法围绕 WGMMA
  operand path 做安排。
- Blackwell 不会出现 TMEM accumulator、async token、`tc_gen5_mma` 和 `tmem_load`，后续
  TMEM allocation/barrier/lowering passes 没有可消费的 IR contract。
- 最终要么无法 lower 到目标 tensor-core path，要么只能走更保守、更慢或不合法的 fallback。

## 15. Knowledge Card

```text
Pass:
  TritonGPUAccelerateMatmul

Purpose:
  Turn generic TTGIR tt.dot into architecture-specific tensor-core-ready IR.

Compiler decision:
  Choose MMA v2 vs MMA v3 / WGMMA vs MMA v5 / TCGen5.

Main IR attribute/op:
  #ttg.nvidia_mma
  #ttg.nvmma_shared
  #ttng.tensor_memory_encoding
  ttng.warp_group_dot
  ttng.tmem_alloc
  ttng.tc_gen5_mma
  ttng.tmem_load

Input contract:
  TTGIR dot exists; operands/results have distributed encodings; target cc is known.

Output contract:
  Dot has a concrete NVIDIA tensor-core execution contract.

Invariant:
  Logical tensor shapes, element types, and matmul semantics are unchanged.

Hardware reason:
  Different NVIDIA generations expose different tensor-core operand/result/storage contracts.

Next dependencies:
  OptimizeDotOperands
  RemoveLayoutConversions
  Pipeline / ScheduleLoops
  shared memory / TMEM / barrier passes
  TritonNvidiaGPUMMALoweringPass
```

## 16. Effective Or No-op

这个 pass 在三份 canonical matmul dump 里都是 effective，不是 no-op。

Ampere:

- before: `tt.dot` 使用 `#ttg.dot_op parent = #blocked`，result 是 `#blocked`，
  见 [034_Before_TritonGPUAccelerateMatmul.mlir:65](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/034_Before_TritonGPUAccelerateMatmul.mlir:65)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/034_Before_TritonGPUAccelerateMatmul.mlir:67)。
- after: 出现 `#ttg.nvidia_mma<versionMajor = 2>`，A/B parent 变成 `#mma`，result 是
  `#mma` 后再转回 `#blocked`，见
  [035_After_TritonGPUAccelerateMatmul.mlir:6](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:6)
  和
  [68](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:68)
  到
  [72](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm86_num_ctas1/mlir-pass-dump.split/035_After_TritonGPUAccelerateMatmul.mlir:72)。

Hopper:

- before: generic `tt.dot`，见
  [032_Before_TritonGPUAccelerateMatmul.mlir:65](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:65)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:67)。
- after: 出现 `#ttg.nvidia_mma<versionMajor = 3>`、A/B `ttg.local_alloc` 和
  `ttng.warp_group_dot`，见
  [033_After_TritonGPUAccelerateMatmul.mlir:6](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:6)
  以及
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:67)
  到
  [73](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm90_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:73)。

Blackwell:

- before: generic `tt.dot`，见
  [032_Before_TritonGPUAccelerateMatmul.mlir:65](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:65)
  到
  [67](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/032_Before_TritonGPUAccelerateMatmul.mlir:67)。
- after: 出现 `#linear`、`#tmem`、A/B `ttg.local_alloc`、
  `ttng.tmem_alloc`、`ttng.tc_gen5_mma` 和 `ttng.tmem_load`，见
  [033_After_TritonGPUAccelerateMatmul.mlir:5](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:5)
  到
  [10](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:10)，以及
  [69](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:69)
  到
  [77](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/033_After_TritonGPUAccelerateMatmul.mlir:77)。

## 17. Open Questions

- Hopper/Blackwell A/B 为什么分别得到 `swizzlingByteWidth = 64` 和 `128`？
  初步判断和 operand shape、K/N 维连续性、bank conflict 规避有关；需要继续读
  `NVMMASharedEncodingAttr::get` 和 shared-memory layout selection。
- `BlockedToMMAv5` 里 `useTwoCTAs = false` 是临时禁用；后续应研究 2CTA TCGen5
  对 B operand split、TMEM encoding 和 barrier 的影响。
- `getDefaultLayoutForTmemLdSt` 如何从 `#tmem` memdesc 推出当前 after IR 的
  `#linear` layout？这是理解 Blackwell TMEM load/store 边界的下一步。
