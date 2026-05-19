#!/usr/bin/env bash
# Run all TritonBench_G_v1 files under the `baseline` backend
# (plain Triton, no race instrumentation). Each file is a self-contained
# `python <file>.py` script.
#
# Output:
#   - CSV row per file: runs/tritonbench_baseline_pytest.csv (upsert by `script`)
#   - Per-file raw log: runs/logs/baseline_pytest_<stem>.log
#   - Progress log:     /tmp/run_tritonbench_baseline.log
#
# Usage:
#   bash scripts/run_tritonbench_baseline.sh

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/tritonbench/test_files.txt
PROGRESS_LOG=/tmp/run_tritonbench_baseline.log
PER_FILE_TIMEOUT=120  # individual TritonBench files are small kernels — seconds, not minutes

total=$(wc -l < "$LIST")
i=0
echo "=== tritonbench baseline: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --benchmark tritonbench --backend baseline "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== tritonbench baseline DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
