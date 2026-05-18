"""Cross-backend comparison for a benchmark suite.

Reads three CSVs produced by `run.py`:
  - runs/<benchmark>_baseline_pytest.csv  (ground-truth, no instrumentation)
  - runs/<benchmark>_gsan_pytest.csv
  - runs/<benchmark>_triton_viz_pytest.csv

…and reports, for the gsan and triton_viz backends:
  - overhead = backend_time / baseline_time  (avg, median, min, max, sum/sum)
  - passed / failed / errors counts and pass rate (passed / (passed+failed+errors))

The comparison excludes files where the per-file shell-timeout numbers
don't reflect real instrumentation cost. Three filter classes (all
applied, intersection of survivors used for every backend so the file set
is consistent):

  1. collection_error — baseline returned errors > 0 with no passed /
     failed (file failed at pytest collection, e.g. `from aiter import
     dtypes` → ROCm-only JIT init crash; AGENTS.md constraint #4).

  2. amd_only_skip — baseline returned skipped > 0 with no passed /
     failed / errors (every test in the file is gated by a
     `@pytest.mark.skipif(...)` on an AMD-only architecture/dtype, and
     the gate fired on this NVIDIA host).

  3. oom — the file's gsan log contains `torch.OutOfMemoryError`. GSan
     reserves a private CUDA pool that overflows on production shapes,
     so the failures aren't kernel-level race signals; they distort the
     overhead ratio downward (GSan exits fast on OOM). We exclude on the
     gsan log only; baseline and triton-viz don't have a private pool.

  4. compile_error — the file's baseline log shows
     `triton.compiler.errors.CompilationError` (e.g. fp8e4b8 not
     supported on this device) or `KeyError: 'Keyword argument
     waves_per_eu'` (AMD-only autotune kwarg). These are kernel-side
     CUDA-incompatibilities, not instrumentation cost — they fail in
     ~the same time on every backend, distorting the overhead ratio
     toward 1x.

  5. race_aborted — gsan reported `race_count > 0` for the file.
     GSan's race detection fires a device-side CUDA assert, which
     puts the CUDA context into a permanent error state for the rest
     of the pytest session: all remaining tests fail in microseconds
     without actually executing the kernel. This makes gsan's total
     runtime artificially smaller than baseline (often <1x), which
     looks like an instrumentation speedup but is really "gsan
     stopped doing work earlier."

  6. timeout — any of the three backends recorded `exit_code=124`
     (shell SIGTERM at PER_FILE_TIMEOUT) for this file. The shell
     killed the orchestrator's `finally` block before it could write
     `target_seconds`, so there's no honest number to compare. The
     file is excluded from every backend so the surviving file set
     is identical and `n` matches across backends.

Usage:
    python analysis/end_to_end_time_overhead.py [--benchmark NAME] \
                                                [--include-skipped] [--verbose]

Defaults: benchmark=aiter. CSV/log paths follow the run.py convention.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_csv(path: Path) -> dict[str, dict]:
    """Index a CSV by stem (basename of `script` minus `.py`)."""
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
                r["target_seconds"] = float(r["target_seconds"]) if r.get("target_seconds") else None
            except ValueError:
                r["target_seconds"] = None
            rows[stem] = r
    return rows


def is_collection_error(b: dict) -> bool:
    """File contributed only collection-time errors (no test executed)."""
    return (b["errors"] > 0 and b["passed"] == 0 and
            b["failed"] == 0 and b["skipped"] == 0)


def is_amd_only_skip(b: dict) -> bool:
    """Every test in the file was skipped (AMD-only gate fired)."""
    return (b["skipped"] > 0 and b["passed"] == 0 and
            b["failed"] == 0 and b["errors"] == 0)


def _log_has_any(log_path_str: str, needles: tuple) -> bool:
    """True iff the file at log_path_str contains any of `needles`."""
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


def is_oom_tainted(log_path_str: str) -> bool:
    """File's gsan log mentions torch.OutOfMemoryError → exclude."""
    return _log_has_any(log_path_str, ("torch.OutOfMemoryError",))


# Kernel-side CUDA-incompatibility signatures. Same time on every backend
# (fails in compile / setup), so they bias overhead toward 1x.
_COMPILE_ERROR_NEEDLES = (
    "triton.compiler.errors.CompilationError",
    "KeyError: 'Keyword argument waves_per_eu",
)


def is_compile_error_tainted(log_path_str: str) -> bool:
    """File's baseline log shows a compile-time CUDA incompatibility."""
    return _log_has_any(log_path_str, _COMPILE_ERROR_NEEDLES)


def overhead_stats(numerator: dict[str, dict], denominator: dict[str, dict],
                   stems: list[str]) -> dict:
    """Compute target_seconds ratios across the given stems. Skip stems
    where either side's target_seconds is missing or non-positive."""
    rows = []
    for s in stems:
        num = numerator.get(s, {}).get("target_seconds")
        den = denominator.get(s, {}).get("target_seconds")
        if num is None or den is None:
            continue
        if num <= 0 or den <= 0:
            continue
        rows.append((s, num / den, num, den))
    if not rows:
        return {"n": 0}
    ratios = [r[1] for r in rows]
    sum_num = sum(r[2] for r in rows)
    sum_den = sum(r[3] for r in rows)
    rows_by_ratio = sorted(rows, key=lambda r: r[1])
    return {
        "n": len(rows),
        "avg": mean(ratios),
        "median": median(ratios),
        "min": rows_by_ratio[0][1],
        "min_file": rows_by_ratio[0][0],
        "max": rows_by_ratio[-1][1],
        "max_file": rows_by_ratio[-1][0],
        "sum_ratio": sum_num / sum_den,
        "sum_num": sum_num,
        "sum_den": sum_den,
    }


def pass_rate_stats(rows: dict[str, dict], stems: list[str],
                    include_skipped: bool) -> dict:
    p = sum(rows.get(s, {}).get("passed", 0) for s in stems)
    f = sum(rows.get(s, {}).get("failed", 0) for s in stems)
    e = sum(rows.get(s, {}).get("errors", 0) for s in stems)
    s_ = sum(rows.get(s, {}).get("skipped", 0) for s in stems)
    if include_skipped:
        denom = p + f + e + s_
    else:
        denom = p + f + e
    return {
        "passed": p, "failed": f, "errors": e, "skipped": s_,
        "denom": denom,
        "rate": (p / denom) if denom else 0.0,
    }


def print_overhead(name: str, st: dict) -> None:
    if st["n"] == 0:
        print(f"  {name}: no comparable files")
        return
    print(f"  {name}  (n={st['n']})")
    print(f"    avg     = {st['avg']:6.2f}x")
    print(f"    median  = {st['median']:6.2f}x")
    print(f"    min     = {st['min']:6.2f}x   ({st['min_file']})")
    print(f"    max     = {st['max']:6.2f}x   ({st['max_file']})")
    print(f"    sum/sum = {st['sum_ratio']:6.2f}x"
          f"  ({st['sum_num']:.1f}s / {st['sum_den']:.1f}s)")


def print_passrate(name: str, st: dict, include_skipped: bool) -> None:
    extra = f"  skipped={st['skipped']}" if include_skipped else ""
    denom_desc = "P+F+E+S" if include_skipped else "P+F+E"
    print(f"  {name:<12} passed={st['passed']:>6}  "
          f"failed={st['failed']:>6}  errors={st['errors']:>4}{extra}  "
          f"-> rate = {st['passed']}/{st['denom']} "
          f"({denom_desc}) = {100*st['rate']:.1f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="aiter",
                   help="Benchmark suite name. Default: aiter.")
    p.add_argument("--include-skipped", action="store_true",
                   help="Include skipped tests in pass-rate denominator. "
                        "Default excludes them (matches the usual notion of "
                        "pass rate over actually-runnable tests).")
    p.add_argument("--verbose", action="store_true",
                   help="Also list the per-file pre-filter outcome bucket.")
    args = p.parse_args(argv)

    csv_path = lambda backend: (REPO_ROOT / "runs"
                                / f"{args.benchmark}_{backend}_pytest.csv")
    b = load_csv(csv_path("baseline"))
    g = load_csv(csv_path("gsan"))
    t = load_csv(csv_path("triton_viz"))

    if not b:
        print(f"[fatal] no baseline CSV at {csv_path('baseline')}", file=sys.stderr)
        return 1

    all_stems = sorted(b.keys())
    print(f"Benchmark: {args.benchmark}")
    print(f"Files in baseline CSV: {len(all_stems)}")

    # ----- apply filters -----
    excluded_collection: list[str] = []
    excluded_amdskip: list[str] = []
    excluded_oom: list[str] = []
    excluded_compile: list[str] = []
    excluded_race: list[str] = []
    excluded_timeout: list[str] = []
    kept: list[str] = []
    for s in all_stems:
        if is_collection_error(b[s]):
            excluded_collection.append(s); continue
        if is_amd_only_skip(b[s]):
            excluded_amdskip.append(s); continue
        # OOM detection: look at gsan's log (OOM is a gsan-pool-only artifact)
        g_log = g.get(s, {}).get("log_path", "")
        if is_oom_tainted(g_log):
            excluded_oom.append(s); continue
        # Compile-time CUDA incompatibility: check baseline log (kernel-side,
        # not instrumentation; same in every backend).
        b_log = b.get(s, {}).get("log_path", "")
        if is_compile_error_tainted(b_log):
            excluded_compile.append(s); continue
        # Race-induced CUDA-context abort: GSan firing __assertfail kills
        # the rest of the pytest session. The remaining tests fail in µs,
        # making gsan's total target_seconds artificially small.
        if g.get(s, {}).get("race_count", 0) > 0:
            excluded_race.append(s); continue
        # Shell SIGTERM (exit_code=124) in ANY backend: the orchestrator's
        # `finally` didn't write target_seconds. Drop the file from every
        # backend so the surviving file set is the same across backends.
        if any(d.get(s, {}).get("exit_code", 0) == 124 for d in (b, g, t)):
            excluded_timeout.append(s); continue
        kept.append(s)

    print(f"  excluded (collection_error)       : {len(excluded_collection)}")
    print(f"  excluded (amd_only_skip)          : {len(excluded_amdskip)}")
    print(f"  excluded (gsan log shows OOM)     : {len(excluded_oom)}")
    print(f"  excluded (baseline compile_error) : {len(excluded_compile)}")
    print(f"  excluded (gsan race-induced abort): {len(excluded_race)}")
    print(f"  excluded (any backend TIMEOUT)    : {len(excluded_timeout)}")
    print(f"  --> analysis set                  : {len(kept)} files")
    if args.verbose:
        for label, lst in [("collection_error", excluded_collection),
                           ("amd_only_skip", excluded_amdskip),
                           ("oom", excluded_oom),
                           ("compile_error", excluded_compile),
                           ("race_aborted", excluded_race),
                           ("timeout", excluded_timeout)]:
            if lst:
                print(f"  [{label}] {lst}")

    print()
    print("=== Overhead = backend_time / baseline_time ===")
    print_overhead("gsan       ", overhead_stats(g, b, kept))
    print_overhead("triton_viz ", overhead_stats(t, b, kept))

    print()
    skipped_note = " (incl. skipped)" if args.include_skipped else " (excl. skipped)"
    print(f"=== Pass rate{skipped_note} on the analysis set ===")
    print_passrate("baseline   ", pass_rate_stats(b, kept, args.include_skipped),
                   args.include_skipped)
    print_passrate("gsan       ", pass_rate_stats(g, kept, args.include_skipped),
                   args.include_skipped)
    print_passrate("triton_viz ", pass_rate_stats(t, kept, args.include_skipped),
                   args.include_skipped)

    return 0


if __name__ == "__main__":
    sys.exit(main())
