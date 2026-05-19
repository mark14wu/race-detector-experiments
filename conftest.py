"""Pytest conftest. Delegates backend setup to run_common.install_backend.

Dispatches on the BACKEND env var; defaults to 'gsan'. Honors BENCHMARK env
for the per-benchmark sys.path safety net (only matters when pytest is run
directly, bypassing run.py).
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Per-benchmark sys.path safety net for `pytest <file>` invocations that
# bypass run.py. Mirrors run.py's BENCHMARK_PATHS dict.
_BENCHMARK = os.environ.get("BENCHMARK", "aiter").strip().lower()
_BENCHMARK_PATHS = {
    "aiter":       os.path.join(_REPO_ROOT, "benchmarks", "aiter", "aiter"),
    "tritonbench": os.path.join(_REPO_ROOT, "benchmarks", "tritonbench", "tritonbench"),
}
_benchmark_path = _BENCHMARK_PATHS.get(_BENCHMARK)
if _benchmark_path and os.path.isdir(_benchmark_path) and _benchmark_path not in sys.path:
    sys.path.insert(0, _benchmark_path)

# triton-viz lives in third_party/triton-viz and isn't always pip-installed.
_TRITON_VIZ_PATH = os.path.join(_REPO_ROOT, "third_party", "triton-viz")
if os.path.isdir(_TRITON_VIZ_PATH) and _TRITON_VIZ_PATH not in sys.path:
    sys.path.insert(0, _TRITON_VIZ_PATH)

# run_common.install_backend lives at repo root.
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
from run_common import install_backend  # noqa: E402

_BACKEND = os.environ.get("BACKEND", "gsan").strip().lower()


@pytest.fixture(scope="session", autouse=True)
def _race_detector_session():
    with install_backend(_BACKEND):
        yield
