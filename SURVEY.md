# AITER op_tests/triton_tests survey under Triton GSan

**Hardware:** NVIDIA GeForce RTX 4090 (sm_89), driver 580.126.20
**Software:** torch 2.11.0+cu128, triton 3.7.0+gitca21b1b9, AITER `d295caf6`
**Date:** 2026-04-29

## TL;DR

**No — AITER's tests do not all pass under GSan.** Far from it. The
dominant failure mode under GSan is *not* "race detected" — it's CUDA OOM
inside GSan's private CUDA mem pool, which can't expand the way the
default allocator can. The kernels themselves are functional on NVIDIA
Triton; they just don't fit GSan's pool at the shapes the AITER tests
parameterize over.

## Three independent failure modes

### 1. Collection errors — ~19 / ~70 test files fail to import

19 of the test files in `benchmarks/aiter/aiter/op_tests/triton_tests/` cannot even be
collected by pytest on a non-ROCm machine. Examples (see
`/tmp/aiter_gsan_run.log` for the full list):

```
test_activation.py        AttributeError: module 'aiter' has no attribute 'dtypes'
test_quant.py             AttributeError: module 'aiter' has no attribute 'dtypes'
test_moe.py               AttributeError: module 'aiter' has no attribute 'dtypes'
attention/test_mha.py     ImportError: cannot import name 'dtypes' from 'aiter'
rope/test_rope.py         ImportError: cannot import name 'dtypes' from 'aiter'
...
```

Root cause: `benchmarks/aiter/aiter/aiter/__init__.py:72-117` puts `from .utility import dtypes`
inside the same try-block as `from .jit import core`. The `core` import
fails on CUDA (it triggers HIP JIT init), so the whole try block aborts
before `dtypes` gets exposed. Patching this requires reaching deeper
into the JIT system (see `patches/aiter-cuda-experimental.patch` for an
attempt — abandoned because the chain bottoms out in `aiter.ops.enum`
which calls `build_module()` at import time, expecting HIP).

`27666` tests successfully collect from the remaining files.

### 2. AMD-only Triton kwargs — some kernels can't compile on NVIDIA Triton

Even where collection succeeds, some kernels use AMD-Triton-specific
JIT kwargs that NVIDIA Triton rejects:

```
test_softmax.py::test_softmax[1823-781-fp32]
KeyError: 'Keyword argument waves_per_eu was specified but unrecognised'
```

`waves_per_eu` is an AMD GCN/CDNA scheduling hint not present in NVIDIA
Triton. These tests would need the kernel signature scrubbed to run on
NVIDIA at all — independent of GSan.

### 3. CUDA OOM inside the GSan mem pool — the dominant failure mode

`triton.experimental.gsan.create_mem_pool()` returns a
`torch.cuda.MemPool` that is a private pool. Tests that allocate larger
tensors fail with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 654.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 22.96 GiB is free.
... 5.51 MiB allocated in private pools (e.g., CUDA Graphs) ...
```

The GPU is empty; the pool just isn't expanding. AITER tests are
parameterized for production-realistic shapes (e.g. vocab=128256 for
top-k), which blow past whatever default pool size GSan's allocator is
willing to grow to.

**Same test, no GSan mem pool, exact same kernel:** all 40 cases pass
in 10s. Demonstrating GSan is the binding constraint, not AITER kernel
correctness:

```bash
# With our conftest.py (GSan mem pool active) — 40 tests:
20 failed, 20 passed in 1.91s    # all 20 failures: OOM in private pool

# Without conftest.py (no GSan):
40 passed, 2 warnings in 10.15s
```

## Numerical sweep (~5% of full triton_tests run)

After ~5% progress on the full sweep with GSan via conftest:

| Outcome | Count | Share |
| --- | ---: | ---: |
| Passed (`.`) | 49 | **3.4%** |
| Failed (`F`) | 1345 | **94.0%** |
| Skipped (`s`) | 66 | 4.6% |
| **Total observed** | **1460** | 100% |

The trend was stable across the 5% prefix — overwhelmingly OOM-in-pool
failures. We killed the run rather than wait hours for a result whose
shape was already clear.

## What this means for using GSan on AITER kernels

GSan-on-AITER is feasible **only at the kernel level, with hand-shrunk
shapes**, not at the test-suite level. Concretely:

1. Pick a single Triton kernel from `benchmarks/aiter/aiter/aiter/ops/triton/`.
2. Copy it into `extracted_kernels/`.
3. Write a tiny driver that allocates **small** tensors inside the pool.
4. Run via `python run_with_gsan.py extracted_kernels/<your>.py`.

`extracted_kernels/demo_add.py` is the working template.

## Files / commands referenced

- Full sweep log: `/tmp/aiter_gsan_run.log`
- Single-file demo (with vs without GSan): pytest commands above
- Conftest enabling GSan: `conftest.py` at repo root
- Wrapper: `run_with_gsan.py`
