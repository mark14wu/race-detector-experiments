#!/usr/bin/env bash
# Run all 72 AITER pytest files under the `gsan` backend
# (Triton GSan: per-session CUDA mem pool + race instrumentation).
#
# Output:
#   - CSV row per file: runs/aiter_gsan_pytest.csv (upsert by `script`)
#   - Per-file raw log: runs/logs/gsan_pytest_<stem>.log
#   - Progress log:     /tmp/run_aiter_gsan.log
#
# Usage:
#   bash scripts/run_aiter_gsan.sh
#
# Expectations (see benchmarks/aiter/inventory.md):
#   - 27,689 individual tests; majority skip on architectural gates.
#   - test_causal_conv1d alone runs ~2620 s (4920 tests, mostly OOM); we
#     keep a 1-hour per-file shell cap so it can complete instead of being
#     SIGTERM'd.
#   - Race lines expected in 4 files (test_rmsnorm, test_layernorm,
#     test_moe_gemm_a8w8_blockscale, test_moe_gemm_int8_smoothquant).

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/aiter/pytest_files.txt
PROGRESS_LOG=/tmp/run_aiter_gsan.log
PER_FILE_TIMEOUT=3600  # seconds; test_causal_conv1d legitimately runs ~44 min

total=$(wc -l < "$LIST")
i=0
echo "=== gsan: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --benchmark aiter --backend gsan "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== gsan DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
