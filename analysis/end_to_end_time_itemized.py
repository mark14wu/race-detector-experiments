"""Per-file timing dump across all three backends.

Lists every file from `benchmarks/<benchmark>/pytest_files.txt` whose
baseline run produced at least one passing test (i.e. the kernels that
actually exercise the test machinery on CUDA — see
`benchmarks/aiter/passing_files.md`). For each file:

- baseline / gsan / triton_viz `target_seconds`
- backend / baseline overhead ratio
- per-backend passed / failed / errors

Where a backend's row has `exit_code == 124` (shell SIGTERM at the
PER_FILE_TIMEOUT cap) or `target_seconds` is missing, the value is
printed as `TIMEOUT` instead of a number; ratios involving it are
printed as `n/a`.

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


def fmt_time(row: dict) -> str:
    """A nicely-padded cell for the time column. TIMEOUT if SIGTERM'd."""
    if row.get("exit_code") == 124:
        return "  TIMEOUT"
    t = row.get("target_seconds")
    if t is None:
        return "  TIMEOUT"          # also covers "no row" / parse errors
    return f"{t:8.2f}s"


def fmt_ratio(num_row: dict, den_row: dict) -> str:
    if num_row.get("exit_code") == 124 or den_row.get("exit_code") == 124:
        return "    n/a"
    n = num_row.get("target_seconds")
    d = den_row.get("target_seconds")
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

    head = (f"{'file':<40} "
            f"{'baseline':>10} {'gsan':>10} {'tviz':>10}  "
            f"{'gsan/b':>7} {'tviz/b':>7}  "
            f"baseline counts             gsan counts                 tviz counts")
    print(head)
    print("-" * len(head))

    for s in files:
        b_row = b.get(s, {})
        g_row = g.get(s, {})
        t_row = t.get(s, {})
        print(f"{s:<40} "
              f"{fmt_time(b_row):>10} {fmt_time(g_row):>10} {fmt_time(t_row):>10}  "
              f"{fmt_ratio(g_row, b_row):>7} {fmt_ratio(t_row, b_row):>7}  "
              f"{fmt_counts(b_row):<26}  {fmt_counts(g_row):<26}  {fmt_counts(t_row)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
