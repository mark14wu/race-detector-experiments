# AITER benchmark inventory

Snapshot of the `aiter/op_tests/triton_tests/` pytest suite as seen on the
NVIDIA/B200 host this repo runs on. Numbers below come from the most recent
end-to-end GSan sweep (`runs/aiter_gsan_pytest.csv`).

## Top-line counts

| 结果   | 数量    |
|--------|--------:|
| passed |   3,626 |
| failed |  11,138 |
| skipped|  12,902 |
| errors |      23 |
| **总计**| **27,689** |

- 涉及 test 文件数：**72**（见 `pytest_files.txt`）
- 通过率（去掉 skipped）：3626 / (3626+11138+23) ≈ **24.5%**
- Skip 率：12902 / 27689 ≈ **46.6%**

## 12,902 个 SKIPPED 的原因（占总数 47%）

AITER 是 ROCm/AMD 项目。绝大多数 test 在源码层就有
`@pytest.mark.skipif(...)` 这类基于 AMD 硬件/数据类型的门控；在 NVIDIA CUDA
上这些 gate 会成立、pytest 标记 SKIPPED。**这些 skip 并非 bug，是 AITER
按设计在非 CDNA4 硬件上不跑。**

| 原因 | 数量 |
|---|---:|
| MXFP4 not supported on this architecture | 9,248 |
| Shape incompatible with gating config    | 1,572 |
| float8 × mx only supported on CDNA4      |   928 |
| FP4 kernels not supported on MI300       |   448 |
| Gluon implementation requires CDNA4      |   340 |
| MOE stack not fully implemented on non-CDNA4 | 144 |
| Numerical tolerance gate                 |   128 |
| 其他 / 未归类                            |    94 |

## 11,138 个 FAILED 的原因

绝大多数失败**不是** race 检测的问题，而是 AITER（ROCm）和 CUDA 之间预先
存在的架构不匹配。详见 `AGENTS.md` constraint #2。

- **GSan 私 pool OOM**（主要）：AITER 测试用生产 shape（如
  `vocab=128256`）。GSan 6× shadow 区在 178 GiB B200 上的私 pool 里装不下，
  `torch.OutOfMemoryError` 直接抛。
- **AMD-only autotune kwarg `waves_per_eu`** → NVIDIA Triton compile 路径
  不认 → `KeyError`。
- **数值断言失败**：测试拿 Triton kernel 结果跟 PyTorch reference 对比，
  部分 shape/dtype 超出 tolerance。

## 23 个 ERROR 的原因

收集时（collection-time）错误。`from aiter import dtypes` 这条 import 链
最终走到 ROCm-only JIT init；影响 72 个文件里的约 19 个
（AGENTS.md constraint #4）。

## Race-detector 信号

跨 27,689 个 test：

| backend     | race 行数 |
|-------------|---------:|
| gsan        | 179（4 个文件） |
| triton_viz  | 0   |
| baseline    | 0（按定义） |

### 按文件细分（GSan）

| 文件 | race 行数 |
|------|---------:|
| `aiter/op_tests/triton_tests/test_rmsnorm.py` | 103 |
| `aiter/op_tests/triton_tests/test_moe_gemm_a8w8_blockscale.py` | 32 |
| `aiter/op_tests/triton_tests/test_moe_gemm_int8_smoothquant.py` | 32 |
| `aiter/op_tests/triton_tests/test_layernorm.py` | 12 |
| **合计** | **179** |

triton_viz 在同一批 72 文件上报 0 race。两种解读：要么 triton-viz 的
happens-before 分析在这些 kernel 上是 sound 的（不太可能——GSan 是更保守
的工具，通常报的是真 race），要么它在 interpreter 模式下没探索到关键
interleaving。详见 `AGENTS.md` "Alternative backend" 章节。

## 怎么重新生成这份文档

- **顶层数字**：从 `runs/aiter_<backend>_pytest.csv` 里 `passed/failed/
  skipped/errors` 四列求和。
- **race 计数**：从 `race_count` 列求和。
- **Skip 原因分类**：`pytest -rs` 跑 skip 数多的文件，从输出里 grep
  `^SKIPPED \[N\]` 行就拿到每条 reason 和它的计数。

这文件 check-in 进 repo，每次跑完全量 sweep 后**手动**更新。
