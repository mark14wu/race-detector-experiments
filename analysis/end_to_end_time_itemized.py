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

OOM-failed tests are **not** marked — they contribute negligible time
(allocation fails in milliseconds), so the recorded `target_seconds`
is approximately the time spent on the non-OOM-failed tests. The
ratio cells compare backend file-level times directly, which is fair
when OOM-failed tests cost ~0 anyway (gsan / tviz / baseline all spend
roughly the same near-zero on those data points; the OOM behavior is
gsan-specific but doesn't change the gsan_time meaningfully).

Ratios involving a non-numeric cell print as `n/a`.

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
    """Return a short label for why this row's target_seconds is not
    comparable, or None if it's a normal measurement.

    Only two markers are emitted:
      - TIMEOUT — orchestrator's `finally` didn't run, no time was written.
      - FAILED - RACE DETECTED — GSan's race-detect CUDA assert
        permanently broke the context, so the time mixes real work
        with zombie post-assert fast-fail.
    OOM-failed tests are intentionally NOT flagged here. OOM costs
    ~milliseconds per test, so a file's recorded `target_seconds` is
    approximately the time the backend spent on the non-OOM tests."""
    if row.get("exit_code") == 124 or row.get("target_seconds") is None:
        return "TIMEOUT"
    if row.get("race_count", 0) > 0:
        return "FAILED - RACE DETECTED"
    return None


def fmt_time(row: dict) -> str:
    """A nicely-padded cell for the time column."""
    label = _is_unreliable(row)
    if label:
        return f"{label:>{_TIME_CELL_W}}"
    return f"{row['target_seconds']:>{_TIME_CELL_W - 1}.2f}s"


def fmt_ratio(num_row: dict, den_row: dict) -> str:
    if _is_unreliable(num_row) or _is_unreliable(den_row):
        return "    n/a"
    n = num_row["target_seconds"]
    d = den_row["target_seconds"]
    if d <= 0:
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

    head = (f"{'file':<40} "
            f"{'baseline':>{_TIME_CELL_W}} {'gsan':>{_TIME_CELL_W}} {'tviz':>{_TIME_CELL_W}}  "
            f"{'gsan/b':>7} {'tviz/b':>7}  "
            f"baseline counts             gsan counts                 tviz counts")
    print(head)
    print("-" * len(head))

    for s in files:
        b_row = b.get(s, {})
        g_row = g.get(s, {})
        t_row = t.get(s, {})
        print(f"{s:<40} "
              f"{fmt_time(b_row)} {fmt_time(g_row)} {fmt_time(t_row)}  "
              f"{fmt_ratio(g_row, b_row):>7} {fmt_ratio(t_row, b_row):>7}  "
              f"{fmt_counts(b_row):<26}  {fmt_counts(g_row):<26}  {fmt_counts(t_row)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
