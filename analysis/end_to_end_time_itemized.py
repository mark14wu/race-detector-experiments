"""Per-file timing dump across all three backends.

Lists every file from `benchmarks/<benchmark>/pytest_files.txt` whose
baseline run produced at least one passing test (i.e. the kernels that
actually exercise the test machinery on CUDA — see
`benchmarks/aiter/passing_files.md`). For each file:

- baseline / gsan / triton_viz `target_seconds`
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

  Let K = number of OOM-failed tests in gsan's log (estimated as
  `oom_lines // 2`, capped at `gsan_failed`). Let N = baseline's
  P+F+E count for the file. Assume uniform per-test time within a
  backend; then:
    - `baseline_adjusted = baseline_time * (N - K) / N`
        (baseline ran all N tests including the K that would later
        OOM in gsan — pro-rate to model "baseline restricted to the
        N-K tests gsan was able to run")
    - `gsan_adjusted     = gsan_time`
        (gsan's K OOM-fails cost ~0; the time is already
        approximately the (N-K)-test time)
    - `tviz_adjusted     = tviz_time * (N - K) / N`
        (tviz didn't OOM, ran all N; same pro-rate as baseline)
  When K == N (every gsan test OOM'd), every backend's adjusted time
  collapses to 0 and ratios become `n/a`.

Ratios involving a non-numeric or zero cell print as `n/a`.

Use this when you need to see the raw per-file picture — `end_to_end_
time_overhead.py` aggregates over a filtered analysis set and hides
exactly the files that this script shows verbatim.

Usage:
    python analysis/end_to_end_time_itemized.py [--benchmark NAME]

Defaults: benchmark=aiter. CSV paths follow the run.py convention.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_csv(path: Path) -> dict[str, dict]:
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
                    float(r["target_seconds"]) if r.get("target_seconds") else None
                )
            except ValueError:
                r["target_seconds"] = None
            rows[stem] = r
    return rows


_TIME_CELL_W = 22  # wide enough for "FAILED - RACE DETECTED"


def _is_unreliable(row: dict) -> str | None:
    """Return a short label for why this row's time is structurally
    not comparable, or None if it's a normal measurement. OOM is NOT
    handled here — it's compensated via per-test subtraction below."""
    if row.get("exit_code") == 124 or row.get("target_seconds") is None:
        return "TIMEOUT"
    if row.get("race_count", 0) > 0:
        return "FAILED - RACE DETECTED"
    return None


def _count_gsan_oom_lines(log_path_str: str) -> int:
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


def oom_test_count(g_row: dict) -> int:
    """Estimate the number of distinct OOM-failed tests in gsan's log.
    pytest prints ~2 OOM lines per failed test under --tb=line (the
    `E` traceback line + the `--tb=line` summary line). Cap at the
    gsan_failed count."""
    lines = _count_gsan_oom_lines(g_row.get("log_path", ""))
    estimated = max(0, lines // 2)
    return min(estimated, g_row.get("failed", 0))


def adjusted_time(row: dict, factor: float) -> float | None:
    """Apply OOM-subtraction pro-rate factor to a row's target_seconds.
    gsan stays at `factor=1.0` (its time already excludes OOM-fast-fails);
    baseline / tviz get factored by `(N - K) / N`. Returns None if the
    row is structurally unreliable (TIMEOUT / RACE)."""
    if _is_unreliable(row) is not None:
        return None
    return row["target_seconds"] * factor


def fmt_time_adjusted(row: dict, factor: float) -> str:
    """Print the OOM-adjusted target_seconds (or the unreliable label)."""
    label = _is_unreliable(row)
    if label:
        return f"{label:>{_TIME_CELL_W}}"
    t = row["target_seconds"] * factor
    return f"{t:>{_TIME_CELL_W - 1}.2f}s"


def fmt_ratio_adjusted(num_row: dict, den_row: dict,
                       num_factor: float, den_factor: float) -> str:
    n = adjusted_time(num_row, num_factor)
    d = adjusted_time(den_row, den_factor)
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

    csv_path = lambda backend: (REPO_ROOT / "runs"
                                / f"{args.benchmark}_{backend}_pytest.csv")
    b = load_csv(csv_path("baseline"))
    g = load_csv(csv_path("gsan"))
    t = load_csv(csv_path("triton_viz"))
    if not b:
        print(f"[fatal] no baseline CSV at {csv_path('baseline')}", file=sys.stderr)
        return 1

    # Per the user's intent: enumerate the files where baseline ran at least
    # one passing test (matches benchmarks/<benchmark>/passing_files.md).
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
        # OOM-adjustment factor: subtract K OOM-failed gsan tests from
        # the test set, pro-rate baseline / tviz times accordingly.
        k = oom_test_count(g_row)
        n = (b_row.get("passed", 0) + b_row.get("failed", 0)
             + b_row.get("errors", 0))
        factor = (n - k) / n if n > 0 else 0.0
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
