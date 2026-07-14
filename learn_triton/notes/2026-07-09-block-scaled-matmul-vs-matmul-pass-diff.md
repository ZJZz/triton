# Block Scaled Matmul vs 普通 Matmul：SM100 `mlir-pass-dump.split` 对比

日期：2026-07-09

## 0. 这次实际对比的是什么例子

这份笔记比较的是两个已经生成好的 SM100 / `num_ctas=1` dump，它们分别对应：

- 普通 matmul：
  `learn_triton/kernels/matmul.py` 里的 `matmul_kernel`
- block-scaled matmul：
  `python/tutorials/10-block-scaled-matmul.py` 的核心 kernel
  `block_scaled_matmul_kernel`

其中 block-scaled 这边为了稳定导出 IR，实际使用的是从 tutorial 中抽出来的最小可编译版本：

- [tutorial_kernel_only.py](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/tutorial_kernel_only.py)

所以这里的对比对象不是“两个抽象概念上的 matmul”，而是：

- 一个普通 payload-only matmul kernel
- 一个面向 NVIDIA SM100 / NVFP4 block scaling 路径的 scaled matmul kernel

Block Scaled Matmul 本身并不只存在于 NVIDIA，AMD 也支持，主要是在 CDNA4 上通过 scaled MFMA 来支持。

## 1. 什么是 Block Scaled Matmul，它和普通 Matmul 有什么区别，有什么用

普通 matmul 的语义是：

```text
C = A @ B
```

block-scaled matmul 的语义则是：

```text
C = (A * scale_a) @ (B * scale_b)
```

这里的 `scale_a` / `scale_b` 不是“每个元素一个 scale”，也不是“整张矩阵一个 scale”，而是按 block 尤其按 K 方向分块广播的 scale tensor。一个 scale 值会覆盖一组连续低精度元素，例如 NVFP4 场景里一组 FP4 payload 会共享一个 scale。

因此它和普通 matmul 的差异不只是“多乘了两个 scale”，而是整个计算 contract 都变了：

- 普通 matmul 是 payload-only：编译器只需要搬运 A/B payload，核心算子是 `tt.dot`
- block-scaled matmul 是 payload + scales：scale 也是 first-class operand，核心算子变成 `tt.dot_scaled`
- 后端要解决的问题也从“把 payload 送进 MMA”变成“把 payload 和 scale 一起按硬件协议送进 scaled MMA”

它的用途本质上是：

```text
用更低精度的数据格式压缩带宽和存储成本，
同时尽量保住 matmul 的数值可用性和 Tensor Core 吞吐
```

这类路径常见于低精度 GEMM、FP4/FP8 microscaling 和大模型训练/推理。收益通常是更小的输入体积和更低的带宽压力，代价则是更复杂的 descriptor、layout、pipeline 和 lowering 协议。

## 2. Block Scaled Matrix Multiplication 相对普通 matmul 新引入了什么

后面 IR 里真正会看到的新东西，主要有六类。

### 2.1 新的数据模型

普通 matmul 只有 A/B payload；block-scaled 还显式携带 A scale 和 B scale。scale 不再是旁路 metadata，而是 MMA 语义的一部分。

### 2.2 新的计算语义

普通 matmul 对应 `tt.dot`，block-scaled 对应 `tt.dot_scaled`。这要求编译器维护的不只是 payload tile，还要维护 scale format、broadcast/regroup 关系以及 scaled MMA 的合法性。

### 2.3 新的布局问题

普通 matmul 只需要把 payload 组织成 MMA-friendly layout；block-scaled 还要把 scale 重排成 tcgen05 block-scale 需要的布局，所以会出现显式的 `reshape -> trans -> reshape` 链。

### 2.4 新的访存协议

普通 matmul 这份例子主要走 raw pointer load/store；block-scaled 走 descriptor load/store，再进一步 lower 成 TMA global-to-shared 和 local-to-global。

### 2.5 新的 pipeline 状态

普通 matmul 主要 overlap A/B payload 和 accumulator；block-scaled 还要同时 overlap `scale_a` 和 `scale_b`，因此 barrier、ring buffer、wait 点和 stage bookkeeping 都会变重。

### 2.6 新的 target-specific lowering 目标

普通 matmul 的最终目标是 `tcgen05.mma`；block-scaled 的最终目标是 `tcgen05.mma_scaled`。这要求后端同时建立 payload encoding、scale encoding、scale residency 和 payload/scale 同步关系。

## 3. 普通 matmul 和 block-scaled matmul 的 pass 差异，应该怎么理解

不要把这次对比理解成：

```text
block-scaled 多了两个 pass，所以它更复杂
```

更准确的理解是：

```text
两者走的是同一条 SM100 pipeline；
block-scaled 的复杂度主要来自它把“scale 也是 first-class operand”
这件事显式编码进了 TTIR / TTGIR / descriptor / TMA / TMEM / barrier 协议里；
pass 数量上只体现成 early canonicalization 多了两轮。
```

所以：

- `pass 数量差异` 是现象
- `IR contract 差异` 是原因
- `payload-only` vs `payload + scales + descriptor + TMA` 才是本质区别

范围：只比较下面两份已经生成好的 SM100 / `num_ctas=1` dump。

- 普通 matmul：
  `learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split`
- block-scaled matmul：
  `learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/mlir-pass-dump.split`

对比目标：

- pass 数量和顺序有什么不同
- 为什么会有这些不同

证据文件：

- 普通 matmul stage dump：
  [matmul_kernel.ttir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/stage_dump/AG5CM2SARLZNDHJPRSLAOVPYZWS2KC33PIRC5D7SWTYEXWBKVPCQ/matmul_kernel.ttir)
  [matmul_kernel.ttgir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/stage_dump/AG5CM2SARLZNDHJPRSLAOVPYZWS2KC33PIRC5D7SWTYEXWBKVPCQ/matmul_kernel.ttgir)
- block-scaled stage dump：
  [block_scaled_matmul_kernel.ttir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/stage_dump/5U46AADGTU57O7TS2KPQSEEOUL2JCTKFITUP3ZA2SDK5OOUDKBBA/block_scaled_matmul_kernel.ttir)
  [block_scaled_matmul_kernel.ttgir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/stage_dump/5U46AADGTU57O7TS2KPQSEEOUL2JCTKFITUP3ZA2SDK5OOUDKBBA/block_scaled_matmul_kernel.ttgir)
- block-scaled descriptor encoding 前后对比：
  [descriptor_encoding.diff](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/descriptor_encoding.diff)

---

## 4. 五个关键节点的 pass-by-pass 对照表

下面这张表不是在列“全部 pass”，而是抓五个最能看出 contract 分叉的检查点：

- TTIR
- `OptimizeDescriptorEncoding`
- `TritonGPUPipeline`
- `TMALowering`
- `MMALowering`

对应证据文件：

- 普通 matmul：
  [TTIR](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/stage_dump/AG5CM2SARLZNDHJPRSLAOVPYZWS2KC33PIRC5D7SWTYEXWBKVPCQ/matmul_kernel.ttir)
  [After OptimizeDescriptorEncoding](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/041_After_TritonNvidiaGPUOptimizeDescriptorEncodingPass.mlir)
  [After TritonGPUPipeline](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/085_After_TritonGPUPipeline.mlir)
  [After TMALowering](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/107_After_TritonNvidiaGPUTMALoweringPass.mlir)
  [After MMALowering](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/mlir-pass-dump.split/123_After_TritonNvidiaGPUMMALoweringPass.mlir)
- block-scaled matmul：
  [TTIR](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/stage_dump/5U46AADGTU57O7TS2KPQSEEOUL2JCTKFITUP3ZA2SDK5OOUDKBBA/block_scaled_matmul_kernel.ttir)
  [After OptimizeDescriptorEncoding](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/mlir-pass-dump.split/045_After_TritonNvidiaGPUOptimizeDescriptorEncodingPass.mlir)
  [After TritonGPUPipeline](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/mlir-pass-dump.split/089_After_TritonGPUPipeline.mlir)
  [After TMALowering](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/mlir-pass-dump.split/111_After_TritonNvidiaGPUTMALoweringPass.mlir)
  [After MMALowering](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/mlir-pass-dump.split/127_After_TritonNvidiaGPUMMALoweringPass.mlir)

| 关键节点 | 普通 matmul | block-scaled matmul |
| --- | --- | --- |
| `TTIR` | 还是最朴素的 payload-only contract：`!tt.ptr<f16>` 参数、`tt.load -> tt.dot -> tt.store`。编译器看到的只是 A/B/C pointer tile。 | 一开始就是 descriptor + scale contract：A/B/C 都是 `tt.tensordesc`，其中 scale 还是 rank-5 descriptor。loop 里是 `4x tt.descriptor_load + reshape/trans/reshape + tt.dot_scaled + tt.descriptor_store`。 |
| `OptimizeDescriptorEncoding` | 从“功能语义”角度几乎不变：仍然没有 descriptor 参数，也没有 scale path。IR 已经进入 matmul-accelerated TTGIR，出现 `ttg.local_alloc + ttng.tmem_alloc + ttng.tc_gen5_mma`，但 pass 本身没有要修的 descriptor contract。 | 这是 block-scaled 第一次把 descriptor legality 显式固定下来：5 个 descriptor 都绑定到 `nvmma_shared` 系列编码，其中 `a_desc` 和 `b_desc` 复用同一个 `#shared`，`c_desc` 绑定到 `#shared2`，两份 rank-5 scale descriptor 绑定到 `#shared1`。同时 IR 保留 `descriptor_load + memdesc_reshape/trans + tc_gen5_mma_scaled`，说明后续要走的已经不是普通 MMA 路径。 |
| `TritonGPUPipeline` | software pipeliner 把普通 payload matmul 改写成异步 producer/consumer loop：`ttng.tmem_alloc` 带 token，loop 内插入 `ttng.wait_barrier`，A/B 仍通过普通 `tt.load` 进入 shared，再喂给 `ttng.tc_gen5_mma`。本质还是 payload-only 双缓冲。 | pipeline 直接变成四路协议：A、B、`scale_a`、`scale_b` 都进入 ring buffer。可以直接看到 `ttng.barrier_expect`、多次 `ttng.async_tma_copy_global_to_local`、scale 的 `memdesc_reshape/trans`、以及仍在 shared-memory scale 上运行的 `ttng.tc_gen5_mma_scaled`。输出在这个节点还保留 `tt.descriptor_store`。 |
| `TMALowering` | 几乎没有可见语义变化。因为这个 kernel 没有 tensordesc I/O，A/B 还是 `tt.load`，C 还是 `tt.store`，不会产生 TMA global-to-local 或 local-to-global protocol。 | 这一节点把 block-scaled 的 descriptor I/O 真正固定成 TMA 协议。输入侧 `async_tma_copy_global_to_local` 仍然存在，输出侧则从 `tt.descriptor_store` 变成 `ttng.async_tma_copy_local_to_global + ttng.async_tma_store_wait`。但此时 scale 仍驻留 shared，`tc_gen5_mma_scaled` 还没有改成消费 scale-TMEM。 |
| `MMALowering` | 到这里 contract 很稳定：普通 A/B payload 从 shared 喂给 `ttng.tc_gen5_mma`，accumulator 驻留 TMEM，整个 kernel 不存在 scale buffer、`tmem_copy` 或 `tc_gen5_mma_scaled`。 | 这是 block-scaled 最关键的一步：两份 scale 先做 `memdesc_reshape/trans`，再通过 `ttng.tmem_copy` 进入专用 `#tmem_scales`，最终 `ttng.tc_gen5_mma_scaled` 消费的是“payload in shared + scales in TMEM”。也就是说，scaled MMA 的最终硬件 contract 到这里才完全成形。 |

这张表有两个值得单独强调的点：

- 对普通 matmul 来说，五个节点里真正变化大的只有 `Pipeline` 和 `MMALowering`，而且变化都围绕 payload 搬运和普通 MMA。
- 对 block-scaled 来说，五个节点每一步都在建立不同层次的合法性：先有 descriptor/scale 语义，再有 descriptor encoding，再有四路 pipeline，再有 TMA store，最后才有 scale-TMEM + `mma_scaled` 的最终执行 contract。

如果只抓主线来读，这份对比最值得重点看的其实不是 TTIR，而是 TTGIR。因为 TTIR 主要是在声明“这个 kernel 的 operand contract 是什么”，真正把这种 contract 展开成异步搬运、barrier、shared/TMEM residency 和 target-specific MMA 协议的是 TTGIR。

## 5. pass 数量和顺序有什么区别

### 5.1 数量

普通 matmul：

- `86 Before + 86 After`

block-scaled matmul：

- `88 Before + 88 After`

所以 block-scaled 比普通 matmul 多了：

- `2` 个 `Before`
- `2` 个 `After`

### 5.2 pass 名字集合

两边的 pass 名字集合相同。

也就是说：

- 没有某个 pass 只出现在普通 matmul
- 也没有某个 pass 只出现在 block-scaled matmul

### 5.3 唯一明确的 multiplicity 差异

只有一个 pass 的出现次数不同：

- `CanonicalizerPass`
  普通 matmul：`10`
  block-scaled：`12`

其余 pass 的出现次数都一致。

### 5.4 顺序分叉点

两边的 pass 主干顺序一致，但分叉点比前一版描述得更早。

更准确地说：

- 普通 matmul：
  `Before_Inliner -> 3x Canonicalizer -> After_Inliner -> 1x Canonicalizer -> TritonCombineOps`
- block-scaled matmul：
  `Before_Inliner -> 5x Canonicalizer -> After_Inliner -> 1x Canonicalizer -> TritonCombineOps`

所以差异不是“在 InlinerPass 之后、进入 `TritonCombineOps` 之前又多跑了两轮”，而是：

```text
block-scaled 在 pre-inline canonicalization 阶段多做了两轮 canonicalization
```

换句话说，后面的主干仍然一致，额外迭代发生在 early TTIR 的定点收敛阶段。

---

## 6. 为什么只多了 canonicalizer

这要从目标倒着看。

`CanonicalizerPass` 不是“功能 pass”，它不负责引入新协议。
它负责把当前 IR 收敛成更规整、更容易让后续 pass 消费的形式。

所以当某个 kernel 在前端引入了更多：

- 中间 reshape/trans
- 冗余 tensor 变换
- descriptor 相关样板 IR
- 可折叠的 index / shape 组合

你最常见看到的现象就是：

- pipeline 没变
- 但 `CanonicalizerPass` 需要多跑几轮

这正是这里发生的事。

---

## 7. 两边 TTIR 的根本差异

这一节只做简要说明，目的是交代两边从一开始就不是同一个 contract。真正值得重点看的过程差异在下一节 TTGIR。

### 7.1 普通 matmul 的 TTIR 在做什么

普通 matmul 的 TTIR 可以直接压成一行：

```text
ptr + load + dot + store
```

从 [matmul_kernel.ttir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/stage_dump/AG5CM2SARLZNDHJPRSLAOVPYZWS2KC33PIRC5D7SWTYEXWBKVPCQ/matmul_kernel.ttir) 可以直接看到：

- 参数是原始 `!tt.ptr<f16>`
- loop 体核心是 `tt.load %a_ptrs`、`tt.load %b_ptrs`、`tt.dot %a, %b, %acc`
- 最后 `tt.store`

核心意思只有一句：编译器看到的是 payload-only matmul。

### 7.2 block-scaled matmul 的 TTIR 在做什么

block-scaled 的 TTIR 也可以压成一行：

```text
descriptor_load + scale reshape/trans + dot_scaled + descriptor_store
```

从 [block_scaled_matmul_kernel.ttir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/stage_dump/5U46AADGTU57O7TS2KPQSEEOUL2JCTKFITUP3ZA2SDK5OOUDKBBA/block_scaled_matmul_kernel.ttir) 可以直接看到：

- 参数不是原始 pointer，而是：
  - `!tt.tensordesc<128x128xui8>`
  - `!tt.tensordesc<1x1x4x2x256xf8E4M3FN>`
  - `!tt.tensordesc<256x128xui8>`
  - `!tt.tensordesc<1x2x4x2x256xf8E4M3FN>`
  - `!tt.tensordesc<128x256xf16>`
- loop 体里多了四个 descriptor load：
  - A payload
  - B payload
  - A scale
  - B scale
- scale load 后不是直接进入 MMA，而是：
  - `tt.reshape`
  - `tt.trans`
  - `tt.reshape`
- 核心算子是 `tt.dot_scaled`
- 输出是 `tt.descriptor_store`

核心意思也只有一句：编译器看到的不再只是 payload，而是 payload + scales + descriptor 的 matmul contract。

### 7.3 为什么这会多两轮 canonicalize

因为 block-scaled 的 TTIR 比普通 matmul 多了几类中间结构：

- descriptor argument expansion
- descriptor_load / descriptor_store
- 5D scale tensor 的 reshape/trans/reshape 链
- `dot_scaled` 的专用 operand path

这些结构都不是本文最关心的“最终执行协议”，只是说明后面的 TTGIR 必然会比普通 matmul 展开出更多状态。

---

## 8. 两边 TTGIR 的过程差异

真正大的区别出现在 TTGIR。

### 8.1 普通 matmul 的 TTGIR

普通 matmul 的 TTGIR 见：
[matmul_kernel.ttgir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/matmul/sm100_num_ctas1/stage_dump/AG5CM2SARLZNDHJPRSLAOVPYZWS2KC33PIRC5D7SWTYEXWBKVPCQ/matmul_kernel.ttgir)

它的主要结构是：

- `tt.load`
- `ttg.local_alloc`
- `ttng.tmem_alloc`
- `ttng.tc_gen5_mma`
- `ttng.wait_barrier`
- `ttng.tmem_load`
- 普通 `tt.store`

可以把它理解成：

```text
payload 直接从 global/pointer path 进来，
进入 shared，
再喂给 tcgen05.mma，
accumulator 驻留 TMEM，
最后读回并 store。
```

### 8.2 block-scaled matmul 的 TTGIR

block-scaled 的 TTGIR 见：
[block_scaled_matmul_kernel.ttgir](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/stage_dump/5U46AADGTU57O7TS2KPQSEEOUL2JCTKFITUP3ZA2SDK5OOUDKBBA/block_scaled_matmul_kernel.ttgir)

它比普通 matmul 多了整整一套 descriptor + scale protocol：

- payload 通过 `ttng.async_tma_copy_global_to_local` 进入 shared
- A scale / B scale 也通过 `ttng.async_tma_copy_global_to_local` 进入 shared
- scale 在 shared 里继续：
  - `memdesc_reshape`
  - `memdesc_trans`
  - `memdesc_reshape`
- 重排后的 scale 还会进一步 `ttng.tmem_copy` 到 scale 专用 TMEM
- 计算核心不再是 `ttng.tc_gen5_mma`，而是 `ttng.tc_gen5_mma_scaled`
- 输出也不是普通 store，而是：
  - `ttng.async_tma_copy_local_to_global`
  - `ttng.async_tma_store_wait`

这说明 block-scaled 的 lowering 目标不只是“做 MMA”，而是：

```text
把 payload 和 scale 都组织成 tcgen05.mma_scaled 的合法输入协议
```

### 8.3 直接数关键 op

在两份 TTGIR 里统计关键 op，差异很明显：

普通 matmul：

- `ttng.async_tma_copy_global_to_local` = `0`
- `ttng.tc_gen5_mma` = `2`
- `ttng.tc_gen5_mma_scaled` = `0`
- `ttng.barrier_expect` = `0`
- `ttng.wait_barrier` = `2`
- `ttng.tmem_copy` = `0`

block-scaled matmul：

- `ttng.async_tma_copy_global_to_local` = `20`
- `ttng.tc_gen5_mma` = `0`
- `ttng.tc_gen5_mma_scaled` = `2`
- `ttng.barrier_expect` = `5`
- `ttng.wait_barrier` = `4`
- `ttng.tmem_copy` = `4`

这里最重要的不是具体数字，而是它们反映的结构：

- 普通 matmul：只有 payload pipeline
- block-scaled：同时 pipeline `A / B / scale_a / scale_b`

如果把整个对比压成一条最重要的主线，TTGIR 的区别其实就是：

- 普通 matmul：producer/consumer 只围绕 A/B payload 展开
- block-scaled：producer/consumer 同时围绕 A/B payload 和 A/B scales 展开

后面你看到的 `async_tma_copy_global_to_local`、`barrier_expect`、`wait_barrier`、`tmem_copy`、`tc_gen5_mma_scaled`，本质上都是这条主线在不同阶段的展开结果。

---

## 9. `OptimizeDescriptorEncoding` 在 block-scaled 里为什么重要

普通 matmul 没有 descriptor。

所以：

- `triton-nvidia-optimize-descriptor-encoding`
  对普通 matmul 基本没有实际工作可做

但 block-scaled 不一样。

它的输入参数本来就是 `tt.tensordesc<...>`，所以 descriptor encoding 会直接影响：

- TMA 能否合法生成
- shared layout 是否匹配 tcgen05 期望
- scale descriptor 的 rank-5 shared encoding 是否能成立

从 [descriptor_encoding.diff](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/dumps/block_scaled_matmul/sm100_nvfp4/descriptor_encoding.diff) 可以看到：

- `a_desc` / `b_desc` 被绑定到 `#ttg.nvmma_shared<...>`
- 两份 scale descriptor 被绑定到 rank-5 `#ttg.nvmma_shared<...>`
- `c_desc` 也被绑定到 `#ttg.nvmma_shared<...>`

这一步建立的不是“优化细节”，而是：

```text
descriptor 必须有 target-compatible shared encoding，
后面的 TMA / MMAv5 lowering 才有合法输入
```

---

## 10. 为什么 block-scaled 的 pipeline 过程更重

这一节只是把前面的结构差异压成定量结果。block-scaled 比普通 matmul 更重，不是因为“后端乱长代码”，而是因为它本来就在维护更大的协议面：

- descriptor legality
- TMA async copy
- barrier bookkeeping
- multi-buffer bookkeeping
- scale layout conversion
- scale residency management

对应地，产物体积也明显更大：

- 普通 matmul 的 `TTGIR` 约 `16.2K`
- block-scaled 的 `TTGIR` 约 `34.4K`

而 `LLIR` 也从约 `85.5K` 膨胀到约 `192.9K`

一句话总结就是：普通 matmul 主要是在组织 payload-only MMA；block-scaled matmul 则是在组织 payload + scales 的 `mma_scaled` 协议，所以共享同一条 pipeline，但会表现出更早的 canonicalization 需求和更重的 descriptor/TMA/TMEM 状态。
