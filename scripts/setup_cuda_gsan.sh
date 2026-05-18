#!/usr/bin/env bash
# Install pieces that uv.lock cannot lock:
#   - PyTorch from the CUDA 12.8 wheel index
#   - Triton main built from source (required for GSan)
#
# AITER is intentionally NOT installed — its setup.py raises on non-ROCm
# (aiter/setup.py:148-149). We expose it via PYTHONPATH instead;
# run_common.prepare_env inserts <repo>/aiter into sys.path at runtime.
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/setup_cuda_gsan.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 0. Bail early if the user cloned without --recursive.
if [ ! -d third_party/triton/.git ] && [ ! -f third_party/triton/.git ]; then
  echo "third_party/triton is missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi
if [ ! -d aiter/.git ] && [ ! -f aiter/.git ]; then
  echo "aiter submodule is missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

# 1. PyTorch (CUDA 12.8). The cu128 wheels work fine against driver CUDA >= 12.8.
echo "[setup] installing torch (cu128)..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. Triton main from source. --no-build-isolation lets the build reuse the
# torch already installed in the venv and speeds up rebuilds.
echo "[setup] building triton from source..."
pushd third_party/triton >/dev/null
uv pip install -r python/requirements.txt
uv pip install -e . --no-build-isolation -v
popd >/dev/null

# 3. AITER stays out of pip. run.py (via run_common.prepare_env) adds
# <repo>/aiter to sys.path at runtime, and `import aiter` succeeds because
# aiter/__init__.py tolerates a missing CK/HIP runtime.

echo "[setup] done. Sanity check:"
python - <<'PY'
import torch, triton
from triton.experimental.gsan import create_mem_pool  # noqa: F401
print("  torch =", torch.__version__, "cuda =", torch.version.cuda, "available =", torch.cuda.is_available())
print("  triton =", triton.__version__, "(gsan import OK)")
PY
