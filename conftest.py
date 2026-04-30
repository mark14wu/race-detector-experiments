"""Pytest conftest that enables Triton GSan for the entire session.

Mirrors the fixture pattern used by Triton's own python/test/gsan/test_gsan.py:
set instrumentation_mode = "gsan", create a CUDA mem pool, and run all tests
inside `with torch.cuda.use_mem_pool(pool):` so GSan can observe tensor
allocations.

Place this at the repo root so pytest picks it up regardless of which test
directory is invoked (rootdir = repo root).
"""
import os
import sys

# Make `import aiter` resolve to ./aiter/aiter without installing the package.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_AITER_PATH = os.path.join(_REPO_ROOT, "aiter")
if os.path.isdir(_AITER_PATH) and _AITER_PATH not in sys.path:
    sys.path.insert(0, _AITER_PATH)

import pytest
import torch
import triton
from triton.experimental.gsan import create_mem_pool


@pytest.fixture(scope="session", autouse=True)
def _gsan_session():
    triton.knobs.compilation.instrumentation_mode = "gsan"
    print(
        f"[gsan-conftest] instrumentation_mode={triton.knobs.compilation.instrumentation_mode}",
        file=sys.stderr,
    )
    pool = create_mem_pool()
    with torch.cuda.use_mem_pool(pool):
        yield
    torch.cuda.synchronize()
