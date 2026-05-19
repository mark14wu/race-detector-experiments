#!/usr/bin/env bash
# Run all TritonBench_G_v1 files under the `triton_viz` backend
# (triton-viz RaceDetector via TRITON_INTERPRET=1, Z3-backed).
#
# Output:
#   - CSV row per file: runs/tritonbench_triton_viz_pytest.csv
#   - Per-file raw log: runs/logs/triton_viz_pytest_<stem>.log
#   - Progress log:     /tmp/run_tritonbench_triton_viz.log
#
# Usage:
#   bash scripts/run_tritonbench_triton_viz.sh
#
# IMPORTANT setup (one-time):
#   cd third_party/triton-viz && uv sync --extra test
# (NOT `uv pip install -e` — misses z3-solver, anytree, ... deps.)
#
# Notes:
# - Interpreter mode runs every Triton op as numpy on the original CUDA
#   tensors. Even small TritonBench kernels can be slow per call.
# - Numerical assertions may fail (interpreter doesn't write results back to
#   CUDA tensors). Compare `race_count` and timing, not numerics.

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/tritonbench/test_files.txt
PROGRESS_LOG=/tmp/run_tritonbench_triton_viz.log
PER_FILE_TIMEOUT=600

total=$(wc -l < "$LIST")
i=0
echo "=== tritonbench triton_viz: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --benchmark tritonbench --backend triton_viz "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== tritonbench triton_viz DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
