# AITER benchmark inventory

Snapshot of the `benchmarks/aiter/aiter/op_tests/triton_tests/` pytest suite as seen on the
NVIDIA/B200 host this repo runs on. **Canonical numbers come from the
`baseline` backend** (`runs/aiter_baseline_pytest.csv`) — that's pytest's
view of the suite without any race-detector instrumentation, which is what
"how many tests does this benchmark have" should mean.

## Top-line counts

| 结果   | 数量    |
|--------|--------:|
| passed |   5,820 |
| failed |   8,944 |
| skipped|  12,902 |
| errors |      19 |
| **总计**| **27,685** |

- 涉及 test 文件数：**72**（见 `pytest_files.txt`）
- 通过率（去掉 skipped）：5820 / (5820+8944+19) ≈ **39.4%**
- Skip 率：12902 / 27685 ≈ **46.6%**

Pytest collect-only reports 27,666 collected + 19 collection-error files;
the 19 collection-error files contribute 1 "errors" each at run time,
giving 27,685.

## 12,902 个 SKIPPED 的原因（占总数 47%）

AITER 是 ROCm/AMD 项目。绝大多数 test 在源码层就有
`@pytest.mark.skipif(...)` 这类基于 AMD 硬件/数据类型的门控；在 NVIDIA CUDA
上这些 gate 会成立、pytest 标记 SKIPPED。**这些 skip 并非 bug，是 AITER
按设计在非 CDNA4 硬件上不跑。**Skipped 数字在三个 backend 上几乎完全
一致（12,902 / 12,900 / 12,902），印证 skip 是源码层决定、跟 race detector
选择无关。

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

## 8,944 个 FAILED 的原因（baseline）

按文件归类的"primary failure"分布——28 个 baseline 文件 **每个 test 都失败**，
归因如下（每文件取数量最多的 error 模式作为代表）：

| 主因 | 文件数 | 涉及测试数 | 说明 |
|---|---:|---:|---|
| 缺 Blackwell-specific autotune config（`100-<KERNEL>.json`） | **22** | ~7,500 | AITER 给每种 `<compute_capability>-<kernel>` 准备了 autotune JSON 配置文件，B200 的 `100-` 系列**只在 CDNA4 上 ship**；NVIDIA 上 import 时报 `AssertionError: Required config file doesn't exist` 或 `FileNotFoundError` |
| AMD-only autotune kwarg `waves_per_eu` | **9** | ~2,500 | NVIDIA Triton compile 路径不认这个参数 → `KeyError: 'Keyword argument waves_per_eu'` |

合计 28 / 8,579 — 跟 `8,944 baseline failed` 几乎完全对得上（剩 365 个 failed
散在 11 个 has-some-passed 文件里，常见原因是数值 tolerance 超限）。

**注意：baseline failed 里完全没有 GSan 私 pool OOM**（baseline 不创建私
pool）。GSan 那边失败数会涨到 11,138（多 2,194），增量主要来源是 **GSan
私 pool OOM** 对生产 shape kernel（如 `test_causal_conv1d` 一个文件就
贡献 4920 - 2948 = 1,972 假失败）。详见 `AGENTS.md` constraint #2。

详见 `benchmarks/aiter/passing_files.md` 看 11 个**有 test 通过**的文件清单。

## 19 个 ERROR 的原因

收集时（collection-time）错误。`from aiter import dtypes` 这条 import 链
最终走到 ROCm-only JIT init；影响 72 个文件里的 19 个（一对一映射）。
详见 `AGENTS.md` constraint #4。

## Race-detector 信号

跨 27,685 个 test：

| backend     | race 行数 | extra teardown errors |
|-------------|---------:|---:|
| gsan        | 179（4 个文件） | 4 |
| triton_viz  | 0   | 0 |
| baseline    | 0（按定义） | 0 |

### 按文件细分（GSan）

| 文件 | race 行数 | extra teardown error |
|------|---------:|:-:|
| `benchmarks/aiter/aiter/op_tests/triton_tests/test_rmsnorm.py` | 103 | ✓ |
| `benchmarks/aiter/aiter/op_tests/triton_tests/test_moe_gemm_a8w8_blockscale.py` | 32 | ✓ |
| `benchmarks/aiter/aiter/op_tests/triton_tests/test_moe_gemm_int8_smoothquant.py` | 32 | ✓ |
| `benchmarks/aiter/aiter/op_tests/triton_tests/test_layernorm.py` | 12 | ✓ |
| **合计** | **179** | **4** |

GSan 的 P+F+S+E 总数 27,689 比 baseline 多 4，**正好对应**这 4 个 race-positive
文件每个在 session-scope fixture teardown 阶段额外报 1 个 pytest error
（race-detect 触发的 CUDA device-side assert 污染 GPU 状态，导致 fixture
退出时 `torch.cuda.synchronize()` 抛异常）。这 4 个 error **不是新的 test
用例**——是 GSan 检出 race 的二级证据。

triton_viz 在同一批 72 文件上报 0 race。两种解读：要么 triton-viz 的
happens-before 分析在这些 kernel 上是 sound 的（不太可能——GSan 是更保守
的工具，通常报的是真 race），要么它在 interpreter 模式下没探索到关键
interleaving。详见 `AGENTS.md` "Alternative backend" 章节。

## 三 backend 总数对照

| metric | gsan | triton_viz | baseline (canonical) |
|---|---:|---:|---:|
| passed | 3,626 | 1,366 | **5,820** |
| failed | 11,138 | 8,207 | **8,944** |
| skipped | 12,902 | 12,900 | **12,902** |
| errors | 23 | 19 | **19** |
| **total** | 27,689 | 22,492 | **27,685** |
| race 行 | **179** | 0 | 0 |
| exit=124 hung 文件 | 0 | 3 | 0 |

triton_viz 22,492 比 baseline 少 5,193，几乎全部来自 3 个 hung 文件
（`test_unified_attention*`、`test_causal_conv1d`）：interpreter 模式下
numpy per-op，生产 shape 跑不完，pytest --timeout=180 thread-method 也
拦不住 numpy C 调用。

## 怎么重新生成这份文档

- **顶层数字**：以 `runs/aiter_baseline_pytest.csv` 为准——baseline
  没有 instrumentation 噪音。`passed/failed/skipped/errors` 四列求和。
- **race 计数**：从 `runs/aiter_gsan_pytest.csv` 的 `race_count` 列求和。
- **Skip 原因分类**：`pytest -rs` 跑 skip 数多的文件，从输出里 grep
  `^SKIPPED \[N\]` 行就拿到每条 reason 和它的计数。

这文件 check-in 进 repo，每次跑完全量 sweep 后**手动**更新。
