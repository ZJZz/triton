# Learn Triton

这个目录现在按用途分层，避免把脚本、kernel、笔记、dump 混在一起。

## 目录说明

- `docs/`
  - 长期说明文档
- `notes/`
  - 按日期记录的学习笔记
- `context/`
  - 换机器或换 AI 时用的交接上下文
- `kernels/`
  - 用来观察 backend 编译行为的示例 Triton kernels
- `tools/`
  - Python 工具脚本
- `scripts/`
  - 常用 shell 入口
- `dumps/`
  - 编译产物、pass dumps、stage dumps

## 常用入口

查看整体学习说明：

- [docs/GUIDE.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/docs/GUIDE.md)

查看今天的学习笔记：

- [notes/2026-06-26-notes.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/notes/2026-06-26-notes.md)

查看换机器继续用的交接文件：

- [context/2026-06-26.md](/LocalRun/jiangzhe.zhao/my_repo/triton/learn_triton/context/2026-06-26.md)

## 常用命令

生成 `vecadd` dump：

```bash
./learn_triton/scripts/compile_and_dump.sh \
  learn_triton/kernels/vec_add.py \
  add_kernel \
  "*fp32:16, *fp32:16, *fp32:16, i32, 1024" \
  "1024,1,1" \
  vecadd
```

生成多架构 `matmul` dump：

```bash
./learn_triton/scripts/dump_multi_chip.sh \
  learn_triton/kernels/matmul.py \
  matmul_kernel \
  "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 64, 64, 32" \
  unused \
  matmul \
  3
```
