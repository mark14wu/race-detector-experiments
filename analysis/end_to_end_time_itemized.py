"""Per-file timing dump across all three backends.

Lists every file from `benchmarks/<benchmark>/pytest_files.txt` whose
baseline run produced at least one passing test (i.e. the kernels that
actually exercise the test machinery on CUDA — see
`benchmarks/aiter/passing_files.md`). For each file:

- baseline / gsan / triton_viz `target_seconds` (OOM-pro-rated)
- backend / baseline overhead ratio
- per-backend passed / failed / errors

Where a backend's row has:
  - `exit_code == 124` (shell SIGTERM at the PER_FILE_TIMEOUT cap) or
    `target_seconds` missing → cell prints `TIMEOUT`.
  - `race_count > 0` → cell prints `FAILED - RACE DETECTED`. GSan's
    race detector fires a CUDA device-side assert; the rest of the
    pytest session runs in a permanently-broken CUDA context and the
    remaining tests fail in microseconds. `target_seconds` looks like
    a number but it's the prefix of real work + zombie time.

OOM handling (per-test time subtraction, NOT a marker):

  Let K = number of OOM-failed tests in gsan's log. Let N = baseline's
  P+F+E count for the file. Then:
    - `baseline_adjusted = baseline_time * (N - K) / N`
    - `gsan_adjusted     = gsan_time`        (OOM-fast-fails cost ~0)
    - `tviz_adjusted     = tviz_time * (N - K) / N`
  When K == N, every backend's adjusted time collapses to 0 and ratios
  become `n/a`.

All this lives in `analysis/common.py` and is shared with the
aggregator `end_to_end_time_overhead_aiter.py`.

Usage:
    python analysis/end_to_end_time_itemized.py [--benchmark NAME]

Defaults: benchmark=aiter. CSV paths follow the run.py convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import common as cm

_TIME_CELL_W = 22  # wide enough for "FAILED - RACE DETECTED"


def fmt_time_adjusted(row: dict, factor: float) -> str:
    """Print the OOM-adjusted target_seconds (or the unreliable label)."""
    label = cm.is_unreliable_time(row)
    if label:
        return f"{label:>{_TIME_CELL_W}}"
    t = row["target_seconds"] * factor
    return f"{t:>{_TIME_CELL_W - 1}.2f}s"


def _adjusted_value(row: dict, factor: float) -> float | None:
    if cm.is_unreliable_time(row) is not None:
        return None
    return row["target_seconds"] * factor


def fmt_ratio_adjusted(num_row: dict, den_row: dict,
                       num_factor: float, den_factor: float) -> str:
    n = _adjusted_value(num_row, num_factor)
    d = _adjusted_value(den_row, den_factor)
    if n is None or d is None or d <= 0:
        return "    n/a"
    return f"{n/d:6.2f}x"


def fmt_counts(row: dict) -> str:
    p, f, s, e = (row.get(k, 0) for k in ("passed", "failed", "skipped", "errors"))
    return f"P={p:>4} F={f:>4} S={s:>4} E={e}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="aiter")
    args = p.parse_args(argv)

    b = cm.load_csv(cm.csv_path(args.benchmark, "baseline"))
    g = cm.load_csv(cm.csv_path(args.benchmark, "gsan"))
    t = cm.load_csv(cm.csv_path(args.benchmark, "triton_viz"))
    if not b:
        print(f"[fatal] no baseline CSV for benchmark={args.benchmark}",
              file=sys.stderr)
        return 1

    files = sorted(
        (stem for stem, row in b.items() if row.get("passed", 0) > 0),
        key=lambda s: -b[s]["passed"],  # most passes first
    )

    print(f"Benchmark: {args.benchmark}")
    print(f"Files with baseline passed > 0: {len(files)}")
    print()

    head = (f"{'file':<40} {'K':>4} "
            f"{'baseline*':>{_TIME_CELL_W}} {'gsan':>{_TIME_CELL_W}} {'tviz*':>{_TIME_CELL_W}}  "
            f"{'gsan/b':>7} {'tviz/b':>7}  "
            f"baseline counts             gsan counts                 tviz counts")
    print(head)
    print(f"{'':<40} {'oom':>4} "
          f"{'(N-K)/N':>{_TIME_CELL_W}} {'as-is':>{_TIME_CELL_W}} {'(N-K)/N':>{_TIME_CELL_W}}")
    print("-" * len(head))

    for s in files:
        b_row = b.get(s, {})
        g_row = g.get(s, {})
        t_row = t.get(s, {})
        k = cm.oom_failed_test_count(g_row)
        factor = cm.pro_rate_factor(b_row, k)
        print(f"{s:<40} {k:>4} "
              f"{fmt_time_adjusted(b_row, factor)} "
              f"{fmt_time_adjusted(g_row, 1.0)} "
              f"{fmt_time_adjusted(t_row, factor)}  "
              f"{fmt_ratio_adjusted(g_row, b_row, 1.0, factor):>7} "
              f"{fmt_ratio_adjusted(t_row, b_row, factor, factor):>7}  "
              f"{fmt_counts(b_row):<26}  {fmt_counts(g_row):<26}  {fmt_counts(t_row)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
