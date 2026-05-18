"""Cross-backend comparison for the AITER benchmark suite.

Reads three CSVs produced by `run.py`:
  - runs/<benchmark>_baseline_pytest.csv  (ground-truth, no instrumentation)
  - runs/<benchmark>_gsan_pytest.csv
  - runs/<benchmark>_triton_viz_pytest.csv

…and reports, for the gsan and triton_viz backends:
  - overhead = backend_time / baseline_time  (avg, median, min, max, sum/sum)
  - passed / failed / errors counts and pass rate (passed / (passed+failed+errors))

The comparison filters out files whose per-file timing or counts don't
honestly reflect instrumentation cost. Five whole-file exclusion classes
(intersection survives for every backend so the file set is consistent),
plus a per-test OOM adjustment that keeps OOM-partial files but subtracts
the OOM-failed tests uniformly from every backend's counts.

Whole-file exclusions:

  1. collection_error — baseline returned errors > 0 with no passed /
     failed (file failed at pytest collection, e.g. `from aiter import
     dtypes` → ROCm-only JIT init crash; AGENTS.md constraint #4).

  2. amd_only_skip — baseline returned skipped > 0 with no passed /
     failed / errors (every test in the file is gated by a
     `@pytest.mark.skipif(...)` on an AMD-only architecture/dtype).

  3. compile_error — the file's baseline log shows a kernel-side
     CUDA incompatibility that fails every test in roughly constant
     time on every backend (so the ratio is meaningless). Signatures:
       - `triton.compiler.errors.CompilationError` (fp8e4b8 / MXFP4)
       - `KeyError: 'Keyword argument waves_per_eu'` (AMD-only kwarg)
       - `AssertionError: Required config file doesn't exist` /
         `isn't an existent file.` / `FileNotFoundError` — AITER's
         per-arch autotune JSONs are CDNA4-only; the Blackwell-100
         configs aren't shipped so `load_autotune_config(...)`
         asserts before the kernel runs.

  4. race_aborted — gsan reported `race_count > 0` for the file.
     GSan's race detection fires a device-side CUDA assert that
     permanently breaks the CUDA context; the rest of the file fails
     in microseconds. The recorded gsan target_seconds is unfixable
     because we can't disentangle "work before the assert" from
     "fast-failing zombie tests after." The itemized table prints
     `FAILED - RACE DETECTED` for these.

  5. timeout — any of the three backends recorded `exit_code=124`
     (shell SIGTERM at PER_FILE_TIMEOUT) for this file. No honest
     `target_seconds` was written. The file is excluded from every
     backend so the surviving file set is identical and `n` matches
     across backends.

Per-test OOM adjustment (does NOT drop the file):

  oom_subtraction — if gsan's log shows K OOM-failed tests in a file,
  apply the same K subtraction to:

    * Pass-rate counts: gsan absorbs K from `failed` first, then
      `passed`; baseline and triton_viz absorb K from `failed` first
      then `passed` too (so all three backends see N-K effective
      tests for the file).

    * Time ratios: gsan's `target_seconds` is left as-is (the K
      OOM-fast-fails cost milliseconds and don't materially shift
      its time). baseline / triton_viz `target_seconds` is
      pro-rated by (N-K)/N — those backends actually ran the K
      OOM-victim tests, so removing them means scaling the
      file-level time by the surviving fraction (assuming roughly
      uniform per-test cost within a backend).

  Files where OOM dominates (K >= N → nothing left after
  subtraction) are dropped from the analysis set.

  This matches `end_to_end_time_itemized.py`'s per-file pro-rate
  exactly, so a row's gsan/baseline ratio in this aggregator's
  numbers equals the same row's `gsan/b` cell in the itemized
  table.

Usage:
    python analysis/end_to_end_time_overhead_aiter.py [--include-skipped] [--verbose]

Defaults: benchmark=aiter (hardcoded; pass --benchmark to override).
CSV/log paths follow the run.py convention.
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
    """Estimate the number of distinct OOM-failed tests in gsan's log.

    Heuristic: pytest emits roughly 2 `torch.OutOfMemoryError` lines per
    failed test (the `E` traceback line + the `--tb=line` line). Cap at
    `gsan_failed` since a non-OOM failure shouldn't be attributed to OOM."""
    lines = count_oom_lines(g_row.get("log_path", ""))
    estimate = max(0, lines // 2)
    return min(estimate, g_row.get("failed", 0))


def apply_oom_subtraction(row: dict, oom_count: int) -> dict:
    """Return a shallow-copied row with `oom_count` tests subtracted from
    failed first (capped at 0), then from passed. Mirrors how the OOM tests
    would have shown up in each backend: gsan listed them as failed; the
    other backends would have included them in their passes / fails."""
    out = dict(row)
    f_sub = min(oom_count, out.get("failed", 0))
    out["failed"] = out.get("failed", 0) - f_sub
    remaining = oom_count - f_sub
    p_sub = min(remaining, out.get("passed", 0))
    out["passed"] = out.get("passed", 0) - p_sub
    return out


# Kernel-side CUDA-incompatibility signatures. Same time on every backend
# (fails in compile / setup / config-discovery), so they bias overhead
# toward 1x and the ratios aren't measuring instrumentation cost.
#   - CompilationError / waves_per_eu: AMD-only autotune-kwarg / dtype.
#   - AssertionError on a config-file existence check, and the
#     FileNotFoundError variant of the same: AITER ships per-arch
#     autotune JSONs (`100-<KERNEL>.json` for Blackwell SM10), but only
#     the CDNA4 ones are present in the repo, so the kernel's
#     `load_autotune_config(...)` call asserts on missing file.
_COMPILE_ERROR_NEEDLES = (
    "triton.compiler.errors.CompilationError",
    "KeyError: 'Keyword argument waves_per_eu",
    "Required config file doesn't exist",
    "isn't an existent file.",
    "FileNotFoundError",
)


def is_compile_error_tainted(log_path_str: str) -> bool:
    """File's baseline log shows a compile-time CUDA incompatibility."""
    return _log_has_any(log_path_str, _COMPILE_ERROR_NEEDLES)


def overhead_stats(numerator: dict[str, dict], denominator: dict[str, dict],
                   stems: list[str],
                   num_factor: dict[str, float] | None = None,
                   den_factor: dict[str, float] | None = None) -> dict:
    """Compute target_seconds ratios across the given stems. Skip stems
    where either side's target_seconds is missing or non-positive.

    `num_factor` / `den_factor`: optional per-stem multipliers applied to
    target_seconds before computing the ratio. Used for OOM-aware
    pro-rating: when K of N tests OOM-fail in gsan, the backends that
    actually ran those K tests (baseline / tviz) get scaled by
    (N-K)/N so the comparison is restricted to the (N-K) tests gsan
    was able to attempt. Mirrors `end_to_end_time_itemized.py`."""
    num_factor = num_factor or {}
    den_factor = den_factor or {}
    rows = []
    for s in stems:
        num = numerator.get(s, {}).get("target_seconds")
        den = denominator.get(s, {}).get("target_seconds")
        if num is None or den is None:
            continue
        if num <= 0 or den <= 0:
            continue
        num *= num_factor.get(s, 1.0)
        den *= den_factor.get(s, 1.0)
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
    excluded_compile: list[str] = []
    excluded_race: list[str] = []
    excluded_timeout: list[str] = []
    excluded_oom_dominated: list[str] = []
    # Per-file OOM-test counts: subtracted from each backend's counts so
    # the comparison stays apples-to-apples. Not used to drop the file
    # unless the subtraction would leave zero effective tests.
    oom_per_file: dict[str, int] = {}
    kept: list[str] = []
    for s in all_stems:
        if is_collection_error(b[s]):
            excluded_collection.append(s); continue
        if is_amd_only_skip(b[s]):
            excluded_amdskip.append(s); continue
        # Compile-time CUDA incompatibility: check baseline log (kernel-side,
        # not instrumentation; same in every backend).
        b_log = b.get(s, {}).get("log_path", "")
        if is_compile_error_tainted(b_log):
            excluded_compile.append(s); continue
        # Race-induced CUDA-context abort: GSan firing __assertfail kills
        # the rest of the pytest session. Timing is unfixable.
        if g.get(s, {}).get("race_count", 0) > 0:
            excluded_race.append(s); continue
        # Shell SIGTERM (exit_code=124) in ANY backend: orchestrator's
        # `finally` didn't write target_seconds. Drop so file set matches.
        if any(d.get(s, {}).get("exit_code", 0) == 124 for d in (b, g, t)):
            excluded_timeout.append(s); continue
        # OOM is now a per-test subtraction, not a whole-file exclusion.
        # Estimate K = number of gsan-OOM-failed tests. If subtracting K
        # would leave zero effective tests in baseline, the file is OOM-
        # dominated and contributes nothing meaningful — drop it.
        oom_count = oom_failed_test_count(g.get(s, {}))
        b_total = b[s]["passed"] + b[s]["failed"] + b[s]["errors"]
        if oom_count >= b_total:
            excluded_oom_dominated.append(s); continue
        oom_per_file[s] = oom_count
        kept.append(s)

    print(f"  excluded (collection_error)        : {len(excluded_collection)}")
    print(f"  excluded (amd_only_skip)           : {len(excluded_amdskip)}")
    print(f"  excluded (baseline compile_error)  : {len(excluded_compile)}")
    print(f"  excluded (gsan race-induced abort) : {len(excluded_race)}")
    print(f"  excluded (any backend TIMEOUT)     : {len(excluded_timeout)}")
    print(f"  excluded (OOM-dominated, K >= total): {len(excluded_oom_dominated)}")
    print(f"  --> analysis set                   : {len(kept)} files")
    files_with_oom_subtraction = sum(1 for v in oom_per_file.values() if v > 0)
    total_subtracted = sum(oom_per_file.values())
    print(f"  …of which {files_with_oom_subtraction} files have partial OOM "
          f"({total_subtracted} test data-points subtracted uniformly across backends)")
    if args.verbose:
        for label, lst in [("collection_error", excluded_collection),
                           ("amd_only_skip", excluded_amdskip),
                           ("compile_error", excluded_compile),
                           ("race_aborted", excluded_race),
                           ("timeout", excluded_timeout),
                           ("oom_dominated", excluded_oom_dominated)]:
            if lst:
                print(f"  [{label}] {lst}")
        oom_files = [(s, k) for s, k in oom_per_file.items() if k > 0]
        if oom_files:
            print(f"  [oom_partial_subtraction] {oom_files}")

    # Build per-file pro-rate factors for time comparisons. gsan stays
    # at 1.0 (OOM-fast-fails cost ~0; gsan_time already reflects the
    # (N-K)-test time). baseline / tviz get scaled by (N-K)/N so they
    # represent "the time those backends would spend on just the
    # non-OOM tests" — same logic as end_to_end_time_itemized.py.
    pro_rate: dict[str, float] = {}
    for s in kept:
        k = oom_per_file.get(s, 0)
        n = b[s]["passed"] + b[s]["failed"] + b[s]["errors"]
        pro_rate[s] = ((n - k) / n) if n > 0 else 0.0

    print()
    print("=== Overhead = backend_time / baseline_time  (OOM-pro-rated) ===")
    print_overhead("gsan       ",
                   overhead_stats(g, b, kept,
                                  num_factor=None,  # gsan unchanged
                                  den_factor=pro_rate))
    print_overhead("triton_viz ",
                   overhead_stats(t, b, kept,
                                  num_factor=pro_rate,
                                  den_factor=pro_rate))

    # Build OOM-adjusted view of each backend's rows so pass-rate counts
    # are comparable across backends (the OOM-failed test data-points are
    # subtracted from every backend's totals).
    b_adj = {s: apply_oom_subtraction(b[s], oom_per_file.get(s, 0)) for s in kept}
    g_adj = {s: apply_oom_subtraction(g.get(s, {}), oom_per_file.get(s, 0)) for s in kept}
    t_adj = {s: apply_oom_subtraction(t.get(s, {}), oom_per_file.get(s, 0)) for s in kept}

    print()
    skipped_note = " (incl. skipped)" if args.include_skipped else " (excl. skipped)"
    print(f"=== Pass rate{skipped_note}, OOM-adjusted, on the analysis set ===")
    print_passrate("baseline   ", pass_rate_stats(b_adj, kept, args.include_skipped),
                   args.include_skipped)
    print_passrate("gsan       ", pass_rate_stats(g_adj, kept, args.include_skipped),
                   args.include_skipped)
    print_passrate("triton_viz ", pass_rate_stats(t_adj, kept, args.include_skipped),
                   args.include_skipped)

    return 0


if __name__ == "__main__":
    sys.exit(main())
