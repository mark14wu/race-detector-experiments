"""Shared utilities for AITER race-detector analysis scripts.

Used by:
  - end_to_end_time_itemized.py        (per-file dump)
  - end_to_end_time_overhead_aiter.py  (aggregate overhead + pass rate)

These two scripts have to agree on a few things or their numbers
disagree (which historically caused confusion):

  * what counts as 'OOM-failed test in gsan' (`oom_failed_test_count`)
  * what counts as 'structurally unreliable target_seconds'
    (`is_unreliable_time`): TIMEOUT vs FAILED - RACE DETECTED
  * the OOM pro-rate factor (N-K)/N (`pro_rate_factor`)
  * filter predicates: collection_error / amd_only_skip /
    compile_error (`is_collection_error`, `is_amd_only_skip`,
    `is_compile_error_tainted`)

Anything pure-formatting (column widths, headers, table layout) stays
in the caller — only data/logic primitives live here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def csv_path(benchmark: str, backend: str) -> Path:
    """Standard run.py output path: `runs/<benchmark>_<backend>_pytest.csv`."""
    return REPO_ROOT / "runs" / f"{benchmark}_{backend}_pytest.csv"


def load_csv(path: Path) -> dict[str, dict]:
    """Read a `runs/*.csv` and index its rows by `script`'s basename
    (without `.py`). Numeric columns are coerced to int / float, and
    `target_seconds` becomes `None` when the cell is empty (which is
    how `run_with_reporting` records timeout-killed runs)."""
    rows: dict[str, dict] = {}
    if not path.is_file():
        print(f"[warn] missing CSV: {path}", file=sys.stderr)
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            stem = Path(r["script"]).stem
            for k in ("passed", "failed", "skipped", "errors",
                      "race_count", "exit_code"):
                r[k] = int(r.get(k) or 0)
            try:
                r["target_seconds"] = (
                    float(r["target_seconds"])
                    if r.get("target_seconds") else None
                )
            except ValueError:
                r["target_seconds"] = None
            rows[stem] = r
    return rows


# ---------------------------------------------------------------------------
# Log scanning
# ---------------------------------------------------------------------------
def log_has_any(log_path_str: str, needles: tuple) -> bool:
    """True iff the log at `log_path_str` contains any of `needles`."""
    if not log_path_str:
        return False
    p = Path(log_path_str)
    if not p.is_file():
        return False
    try:
        with open(p, errors="replace") as f:
            for line in f:
                if any(n in line for n in needles):
                    return True
    except OSError:
        return False
    return False


def count_oom_lines(log_path_str: str) -> int:
    """Count `torch.OutOfMemoryError` occurrences in a log file."""
    if not log_path_str:
        return 0
    p = Path(log_path_str)
    if not p.is_file():
        return 0
    n = 0
    try:
        with open(p, errors="replace") as f:
            for line in f:
                if "torch.OutOfMemoryError" in line:
                    n += 1
    except OSError:
        return 0
    return n


def oom_failed_test_count(g_row: dict) -> int:
    """Estimate the number of distinct OOM-failed tests in a gsan row.

    Heuristic: pytest's `--tb=line` prints ~2 OOM lines per failed test
    (the `E   torch.OutOfMemoryError` traceback line + the matching
    `--tb=line` summary line). Capped at `g_row["failed"]` so non-OOM
    failures aren't attributed to OOM."""
    lines = count_oom_lines(g_row.get("log_path", ""))
    estimated = max(0, lines // 2)
    return min(estimated, g_row.get("failed", 0))


# ---------------------------------------------------------------------------
# Pro-rating + reliability
# ---------------------------------------------------------------------------
def pro_rate_factor(b_row: dict, k: int) -> float:
    """OOM-pro-rate scalar: (N-K)/N, where N = baseline P+F+E.

    Multiply baseline / triton_viz `target_seconds` by this factor to
    model 'time spent on the N-K tests that didn't OOM in gsan.'
    gsan's own time stays unscaled — its K OOM-fails cost ~milliseconds
    and don't materially shift gsan_time. When K >= N, factor is 0.0
    and downstream code should print the ratio as `n/a`."""
    n = (b_row.get("passed", 0) + b_row.get("failed", 0)
         + b_row.get("errors", 0))
    return ((n - k) / n) if n > 0 else 0.0


def is_unreliable_time(row: dict) -> str | None:
    """Return a short label for why this row's target_seconds is
    structurally not comparable, or None if it's a normal measurement.

    Only two markers — OOM is intentionally NOT classified here; it's
    handled via `pro_rate_factor` so the OOM-failed tests get
    subtracted from baseline / tviz times.

    - TIMEOUT: shell SIGTERM at PER_FILE_TIMEOUT killed the orchestrator
      before `finally` could write target_seconds.
    - FAILED - RACE DETECTED: gsan's race-detect CUDA assert broke
      the context; recorded time mixes real kernel work with
      microsecond zombie tests after the assert."""
    if row.get("exit_code") == 124 or row.get("target_seconds") is None:
        return "TIMEOUT"
    if row.get("race_count", 0) > 0:
        return "FAILED - RACE DETECTED"
    return None


# ---------------------------------------------------------------------------
# File-level filter predicates (used by the aggregate analyzer; some
# also useful for itemized callers that want to skip noise files)
# ---------------------------------------------------------------------------
def is_collection_error(b_row: dict) -> bool:
    """File contributed only collection-time errors (no test executed)."""
    return (b_row.get("errors", 0) > 0
            and b_row.get("passed", 0) == 0
            and b_row.get("failed", 0) == 0
            and b_row.get("skipped", 0) == 0)


def is_amd_only_skip(b_row: dict) -> bool:
    """Every test in the file was skipped (AMD-only gate fired)."""
    return (b_row.get("skipped", 0) > 0
            and b_row.get("passed", 0) == 0
            and b_row.get("failed", 0) == 0
            and b_row.get("errors", 0) == 0)


# Kernel-side CUDA incompatibilities — fail in ~constant time on every
# backend (compile / setup / config discovery), so the ratio is
# meaningless. Categories:
#   - CompilationError / waves_per_eu: AMD-only autotune-kwarg / dtype.
#   - AssertionError on config-file existence / FileNotFoundError:
#     AITER ships per-arch autotune JSONs (`100-<KERNEL>.json` for
#     Blackwell SM10), CDNA4-only in the repo.
_COMPILE_ERROR_NEEDLES = (
    "triton.compiler.errors.CompilationError",
    "KeyError: 'Keyword argument waves_per_eu",
    "Required config file doesn't exist",
    "isn't an existent file.",
    "FileNotFoundError",
)


def is_compile_error_tainted(log_path_str: str) -> bool:
    """File's baseline log shows a kernel-side CUDA incompatibility."""
    return log_has_any(log_path_str, _COMPILE_ERROR_NEEDLES)
