#!/usr/bin/env bash
# Run all 72 AITER pytest files under the `triton_viz` backend
# (triton-viz RaceDetector via TRITON_INTERPRET=1, Z3-backed).
#
# Output:
#   - CSV row per file: runs/aiter_triton_viz_pytest.csv (upsert by `script`)
#   - Per-file raw log: runs/logs/triton_viz_pytest_<stem>.log
#   - Progress log:     /tmp/run_aiter_triton_viz.log
#
# Usage:
#   bash scripts/run_aiter_triton_viz.sh
#
# IMPORTANT setup (one-time):
#   cd third_party/triton-viz && uv sync --extra test
# (NOT `uv pip install -e` — that misses z3-solver, anytree, ... deps.)
#
# Expectations (see benchmarks/aiter/inventory.md):
#   - Interpreter mode runs every Triton op as a numpy op on the original
#     CUDA tensors. Production-shape tests (e.g. vocab=128256) take minutes
#     per individual test, and pytest's thread-method --timeout cannot
#     interrupt numpy C calls. A handful of files will hit the shell cap;
#     CSV row should be marked `notes=TIMEOUT 600` manually.
#   - Numerical assertions fail because interpreter doesn't write results
#     back to CUDA tensors. Ignore assertion failures; compare `race_count`
#     and timing only.
#   - Race lines historically: 0 across all 72 files (vs gsan's 179).

set -u
cd "$(dirname "$0")/.."

source .venv/bin/activate

LIST=benchmarks/aiter/pytest_files.txt
PROGRESS_LOG=/tmp/run_aiter_triton_viz.log
PER_FILE_TIMEOUT=600   # seconds; longer doesn't help — numpy isn't interruptible

total=$(wc -l < "$LIST")
i=0
echo "=== triton_viz: $total files, $PER_FILE_TIMEOUT s/file cap, start $(date +%H:%M:%S) ===" \
  | tee "$PROGRESS_LOG"

while IFS= read -r f; do
    i=$((i + 1))
    echo "[$i/$total] $f  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
    timeout "$PER_FILE_TIMEOUT" python run.py --backend triton_viz "$f" \
        </dev/null >/dev/null 2>&1
    rc=$?
    echo "  -> exit=$rc  $(date +%H:%M:%S)" | tee -a "$PROGRESS_LOG"
done < "$LIST"

echo "=== triton_viz DONE at $(date +%H:%M:%S) ===" | tee -a "$PROGRESS_LOG"
