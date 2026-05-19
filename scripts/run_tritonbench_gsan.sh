#!/usr/bin/env bash
# Run all TritonBench_G_v1 files under the `gsan` backend
# (Triton GSan: per-script CUDA mem pool + race instrumentation).
#
# Output:
#   - CSV row per file: runs/tritonbench_gsan_pytest.csv (upsert by `script`)
#   - Per-file raw log: runs/logs/gsan_pytest_<stem>.log
#   - Progress log:     /tmp/run_tritonbench_gsan.log
#
# Usage:
#   bash scripts/run_tritonbench_gsan.sh
#
# Notes:
# - Each script gets its own GSan mem pool (set up by install_backend in the
#   script_runner wrapper). TritonBench shapes are small, so OOM-in-pool is
#   unlikely (unlike aiter's production shapes).
# - Race lines surface via the orchestrator's stdout/stderr sniffer.

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/tritonbench/test_files.txt
PROGRESS_LOG=/tmp/run_tritonbench_gsan.log
PER_FILE_TIMEOUT=300

total=$(wc -l < "$LIST")
i=0
echo "=== tritonbench gsan: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --benchmark tritonbench --backend gsan "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== tritonbench gsan DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
