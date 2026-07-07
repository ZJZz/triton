# Barrier / Fence Protocol Visuals

这组页面把 [2026-07-02-barriers-and-fences.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-07-02-barriers-and-fences.md) 里的主协议改成“教学页”表达：

- [总览对照图](./protocol-overview.html)
- [sm80 `cp.async` 教学页](./sm80-cp-async.html)
- [sm90 TMA + `mbarrier` 教学页](./sm90-tma-mbarrier.html)
- [sm90 WGMMA 教学页](./sm90-wgmma.html)
- [sm100 `tcgen05` / TMEM 教学页](./sm100-tcgen05.html)

推荐阅读顺序：

- 先看 `protocol-overview.html`，按 target 建立“输入内存 -> 模块 -> 输出内存 -> wait object -> sync scope -> 合法 consumer”的主链。
- 再看具体 target 页。每一页都固定分成：
  - 视角 1：数据从哪里来，经过哪个模块，最后落到哪里
  - 视角 2：thread / warp / warp-group / CTA / cluster 里，哪一层负责 issue、wait、consume
  - 视角 3：多个异步任务先发出，再在 barrier / wait 处汇合的 timeline
  - 视角 4~6：当前步骤解释、完整协议步骤、常见误解

当前主输出是手写 HTML 教学页；较早版本保留了 `*.workflow.json`、`*.sequence.json`、`*.dataflow.json` 作为历史参考。
