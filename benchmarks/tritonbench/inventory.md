# TritonBench benchmark inventory

骨架占位 —— 首次 sweep 之后再填具体数字。格式参考
`benchmarks/aiter/inventory.md`：顶层总数表、skip / failed 原因分类、
race-detector 信号、三 backend 对照表。

## 来源

- Submodule: `benchmarks/tritonbench/tritonbench/` —— `https://github.com/thunlp/TritonBench`
- Submodule HEAD: 见 `git submodule status benchmarks/tritonbench/tritonbench`

## 待办（在 sweep 前）

1. 摸 TritonBench 的测试结构：是 pytest 形式还是自带 driver？测试入口在
   `EVAL/`、`LLM_generated/` 还是别的地方？
2. 决定要不要装它自己的 Python 依赖（参考 triton-viz 用 `uv sync --extra test`）。
3. 填 `benchmarks/tritonbench/pytest_files.txt`（每行一个 repo-relative 路径，
   形如 `benchmarks/tritonbench/tritonbench/...`）。
4. 验证 `python run.py --benchmark tritonbench --backend baseline <file>`
   能跑通一个 test。
5. 写 `scripts/run_tritonbench_<backend>.sh`（仿 `scripts/run_aiter_*.sh`）。
6. 跑完一轮 sweep 后回填本文档。
