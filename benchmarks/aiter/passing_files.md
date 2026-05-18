# AITER files with passing tests (baseline)

Of the 72 files in `pytest_files.txt`, only **11** have at least one test that
passes on baseline (plain Triton, no race detector, CUDA/B200). These are the
kernels whose Triton code paths happen to be ROCm-portable. See `inventory.md`
for what blocks the other 61 files.

Numbers below come from `runs/aiter_baseline_pytest.csv`. The "test functions"
column lists the top-level `def test_*` names defined in each file; the
actual test count is much larger because every function has many parametrized
variants.

| # | file (stem) | passed | failed | test functions |
|---|---|---:|---:|---|
| 1 | `test_causal_conv1d` | 4,920 | 0 | `test_causal_conv1d_update`, `test_causal_conv1d_update_with_batch_gather`, `test_causal_conv1d_varlen` |
| 2 | `test_fused_add_rmsnorm_pad` | 420 | 0 | `test_mul_add` |
| 3 | `test_unified_attention_sparse_mla` | 144 | 0 | `test_triton_unified_attn` |
| 4 | `test_rmsnorm` | 108 | 252 | `test_fused_add_rmsnorm`, `test_rms_norm_dynamic_per_token_fp8_quant`, `test_rmsnorm`, `test_rmsnorm_dynamicquant`, `test_rmsnorm_fused_add_dynamicquant`, `test_rmsnorm_fused_add_smoothquant`, `test_rmsnorm_smoothquant` |
| 5 | `test_gated_delta_rule` | 82 | 0 | `test_chunk`, `test_chunk_opt`, `test_chunk_opt_varlen`, `test_chunk_opt_vk`, `test_chunk_opt_vk_varlen`, `test_chunk_varlen`, `test_fused_recurrent`, `test_fused_sigmoid_gating_delta_rule_update` |
| 6 | `test_layernorm` | 48 | 96 | `test_fused_add_layernorm`, `test_layernorm`, `test_layernorm_dynamicquant`, `test_layernorm_fused_add_dynamicquant`, `test_layernorm_fused_add_smoothquant`, `test_layernorm_smoothquant` |
| 7 | `test_topk` | 40 | 0 | `test_topk` |
| 8 | `test_prefill_attention` | 32 | 0 | `test_op_fwd` |
| 9 | `test_moe_align_block_size` | 9 | 0 | `test_correctness` |
| 10 | `test_pa_prefill` | 9 | 9 | `test_contexted_kv_attention`, `test_contexted_kv_attention_alibi` |
| 11 | `test_chunked_pa_prefill` | 8 | 8 | `test_contexted_kv_attention`, `test_contexted_kv_attention_alibi` |
| **合计** | — | **5,820** | **365** | — |

## Full repo-relative paths

```
aiter/op_tests/triton_tests/test_causal_conv1d.py
aiter/op_tests/triton_tests/normalization/test_fused_add_rmsnorm_pad.py
aiter/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py
aiter/op_tests/triton_tests/normalization/test_rmsnorm.py
aiter/op_tests/triton_tests/test_gated_delta_rule.py
aiter/op_tests/triton_tests/normalization/test_layernorm.py
aiter/op_tests/triton_tests/test_topk.py
aiter/op_tests/triton_tests/attention/test_prefill_attention.py
aiter/op_tests/triton_tests/moe/test_moe_align_block_size.py
aiter/op_tests/triton_tests/attention/test_pa_prefill.py
aiter/op_tests/triton_tests/attention/test_chunked_pa_prefill.py
```

## How to use this list

- **Race-detector signal scope**: the 4 files where GSan flagged race lines
  (`test_rmsnorm` 103, `test_layernorm` 12, `test_moe_gemm_a8w8_blockscale` 32,
  `test_moe_gemm_int8_smoothquant` 32) all involve kernels in this 11-file
  set — except the two `moe_gemm_*`, which are in the 22 missing-config
  category. The race signal is concentrated in
  `normalization/test_rmsnorm.py` and `normalization/test_layernorm.py`.

- **Fair instrumentation overhead comparison** (per
  `analysis/end_to_end_runtime.py`): the 8-file analysis set is the
  intersection of this 11-file list with "no OOM tainting under gsan" and
  "no compile-error". These 8 files are where backend timing comparisons are
  meaningful.

- **Reproducing one file end-to-end**:
  ```bash
  python run.py --backend baseline   aiter/op_tests/triton_tests/normalization/test_rmsnorm.py
  python run.py --backend gsan       aiter/op_tests/triton_tests/normalization/test_rmsnorm.py
  python run.py --backend triton_viz aiter/op_tests/triton_tests/normalization/test_rmsnorm.py
  ```
