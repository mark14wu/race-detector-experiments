#!/usr/bin/env bash
# Run all 72 AITER pytest files under the `baseline` backend
# (plain Triton, no race instrumentation).
#
# Output:
#   - CSV row per file: runs/aiter_baseline_pytest.csv (upsert by `script`)
#   - Per-file raw log: runs/logs/baseline_pytest_<stem>.log
#   - Progress log:     /tmp/run_aiter_baseline.log
#
# Usage:
#   bash scripts/run_aiter_baseline.sh
#
# Run baseline first to get the "what fails anyway" reference. Compare its
# CSV to gsan/triton_viz to isolate race-detector overhead from pre-existing
# AITER-on-CUDA breakage. See benchmarks/aiter/inventory.md.

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/aiter/pytest_files.txt
PROGRESS_LOG=/tmp/run_aiter_baseline.log
PER_FILE_TIMEOUT=3600  # seconds; matches gsan cap. test_causal_conv1d has 4920 collected tests — at ~3.6 tests/s baseline needs ~22 min on that file alone

total=$(wc -l < "$LIST")
i=0
echo "=== baseline: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --backend baseline "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== baseline DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
