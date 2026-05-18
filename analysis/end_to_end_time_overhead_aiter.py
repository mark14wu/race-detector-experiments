"""Cross-backend overhead aggregate for the AITER benchmark suite.

Reads the three CSVs produced by `run.py`:
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
the OOM-failed tests uniformly from every backend's counts and prorates
baseline / triton_viz time accordingly.

Whole-file exclusions:

  1. collection_error — `from aiter import dtypes` chain crashed before
     any test ran (AGENTS.md constraint #4).
  2. amd_only_skip — every test gated by `@skipif(...)` on an AMD-only
     HW/dtype.
  3. compile_error — baseline log shows a kernel-side CUDA
     incompatibility (`CompilationError` / `waves_per_eu` /
     missing `100-<KERNEL>.json` autotune config).
  4. race_aborted — gsan flagged `race_count > 0`. Race-detect CUDA
     assert permanently corrupts the CUDA context; the rest of the
     file fails in microseconds and the recorded gsan time can't be
     untangled from those zombie tests.
  5. timeout — any backend exited with `exit_code=124` (shell SIGTERM
     at PER_FILE_TIMEOUT). Drop so the file set matches across
     backends.

Per-test OOM adjustment (does NOT drop the file):

  oom_subtraction — if gsan's log shows K OOM-failed tests in a file,
  apply the same K subtraction to:

    * Pass-rate counts: gsan absorbs K from `failed` first, then
      `passed`; baseline and triton_viz absorb K the same way (so
      all three backends see N-K effective tests for the file).

    * Time ratios: gsan's `target_seconds` is left as-is (the K
      OOM-fast-fails cost milliseconds and don't materially shift
      its time). baseline / triton_viz `target_seconds` is
      pro-rated by (N-K)/N — those backends actually ran the K
      OOM-victim tests, so removing them means scaling the
      file-level time by the surviving fraction.

  Files where OOM dominates (K >= N → nothing left after
  subtraction) are dropped from the analysis set.

  This matches `end_to_end_time_itemized.py`'s per-file pro-rate
  exactly — a row's gsan/baseline ratio in this aggregator's
  numbers equals the same row's `gsan/b` cell in the itemized
  table.

All the primitives — CSV parsing, log scanning, OOM counting,
pro-rate factor, structural-unreliable detection, filter predicates
— live in `analysis/common.py`. This script only adds aggregation.

Usage:
    python analysis/end_to_end_time_overhead_aiter.py [--include-skipped] [--verbose]

Defaults: benchmark=aiter (pass --benchmark to override).
"""
from __future__ import annotations

import argparse
import sys
from statistics import mean, median

import common as cm


# ---------------------------------------------------------------------------
# Pass-rate count subtraction
# ---------------------------------------------------------------------------
def apply_oom_subtraction(row: dict, oom_count: int) -> dict:
    """Return a shallow-copied row with `oom_count` tests subtracted from
    `failed` first (capped at 0), then from `passed`. Mirrors how the
    OOM-victim tests would have shown up in each backend: gsan listed
    them as failed; the other backends would have included them in
    their passes / fails."""
    out = dict(row)
    f_sub = min(oom_count, out.get("failed", 0))
    out["failed"] = out.get("failed", 0) - f_sub
    remaining = oom_count - f_sub
    p_sub = min(remaining, out.get("passed", 0))
    out["passed"] = out.get("passed", 0) - p_sub
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def overhead_stats(numerator: dict[str, dict], denominator: dict[str, dict],
                   stems: list[str],
                   num_factor: dict[str, float] | None = None,
                   den_factor: dict[str, float] | None = None) -> dict:
    """Compute target_seconds ratios across `stems`, with optional
    per-stem pro-rate multipliers (used for OOM-aware comparison)."""
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
    denom = p + f + e + s_ if include_skipped else p + f + e
    return {"passed": p, "failed": f, "errors": e, "skipped": s_,
            "denom": denom, "rate": (p / denom) if denom else 0.0}


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
    p.add_argument("--benchmark", default="aiter")
    p.add_argument("--include-skipped", action="store_true",
                   help="Include skipped tests in pass-rate denominator.")
    p.add_argument("--verbose", action="store_true",
                   help="List per-bucket excluded files.")
    args = p.parse_args(argv)

    b = cm.load_csv(cm.csv_path(args.benchmark, "baseline"))
    g = cm.load_csv(cm.csv_path(args.benchmark, "gsan"))
    t = cm.load_csv(cm.csv_path(args.benchmark, "triton_viz"))
    if not b:
        print(f"[fatal] no baseline CSV for benchmark={args.benchmark}",
              file=sys.stderr)
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
    oom_per_file: dict[str, int] = {}
    kept: list[str] = []
    for s in all_stems:
        if cm.is_collection_error(b[s]):
            excluded_collection.append(s); continue
        if cm.is_amd_only_skip(b[s]):
            excluded_amdskip.append(s); continue
        if cm.is_compile_error_tainted(b.get(s, {}).get("log_path", "")):
            excluded_compile.append(s); continue
        if g.get(s, {}).get("race_count", 0) > 0:
            excluded_race.append(s); continue
        if any(d.get(s, {}).get("exit_code", 0) == 124 for d in (b, g, t)):
            excluded_timeout.append(s); continue
        # OOM is per-test subtraction, not whole-file exclusion. Drop
        # only when subtracting K would leave nothing comparable.
        oom_count = cm.oom_failed_test_count(g.get(s, {}))
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

    # ----- pro-rate factors for time comparison -----
    pro_rate: dict[str, float] = {
        s: cm.pro_rate_factor(b[s], oom_per_file.get(s, 0)) for s in kept
    }

    print()
    print("=== Overhead = backend_time / baseline_time  (OOM-pro-rated) ===")
    print_overhead("gsan       ",
                   overhead_stats(g, b, kept,
                                  num_factor=None,        # gsan unchanged
                                  den_factor=pro_rate))
    print_overhead("triton_viz ",
                   overhead_stats(t, b, kept,
                                  num_factor=pro_rate,
                                  den_factor=pro_rate))

    # ----- OOM-adjusted pass-rate views -----
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
