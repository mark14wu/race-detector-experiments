"""Shared scaffolding for the unified `run.py` orchestrator.

What lives here:

  * `T0`             — wall-time anchor that survives `os.execv` (re-exec into
                       `.venv`) so `total_seconds` is meaningful end-to-end.
  * `ensure_venv`    — re-exec under the repo's `.venv/bin/python` when needed.
  * `prepare_env`    — `PYTHONPATH=<benchmark>[+...]`, default env vars, `sys.path`
                       insertions; takes a backend-specific knob bag.
  * `BackendConfig`  — describes a backend's name, env-var defaults, extra
                       paths, race-output regex, and per-test pytest timeout.
  * `BACKEND_SPECS`  — registry of all known backends keyed by short name
                       (`gsan`, `triton_viz`, `baseline`).
  * `get_backend`    — look up a BackendConfig by short name.
  * `make_pytest_runner` — build the target_runner closure that
                       subprocess-spawns `python -m pytest <file>` and parses
                       its summary; shared by all backends.
  * `add_common_args` / `resolve_paths` — `--csv`, `--log`, `--log-dir`,
                       `--no-csv`, `--no-log`, `--benchmark`, `--notes`.
  * `run_with_reporting` — the actual orchestration: tee stdout+stderr to a
                       log file, watch for race lines, time the target, write
                       a CSV row at the end, return an exit code.

Future benchmarks (e.g. `torchinductor`) only need to:
  1. Drop a `benchmarks/<name>/pytest_files.txt` + `inventory.md`;
  2. Pass `--benchmark <name>` to `run.py`.
No new Python file required.

CSV schema (one row per run):

    timestamp, benchmark, backend, script, target_seconds, total_seconds,
    race_count, exit_code, passed, failed, skipped, errors, log_path, notes

A new column can be added by editing `_CSV_FIELDS` plus the row builder; the
writer will rewrite the header automatically when the CSV is first created.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import runpy
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

# ---------------------------------------------------------------------------
# Cross-exec wall-clock anchor. `time.perf_counter()` resets across exec on
# some platforms; `time.time()` (epoch seconds) is stable, which is what we
# want for `total_seconds`.
# ---------------------------------------------------------------------------
_T0_ENV = "_RUN_T0_EPOCH"
if _T0_ENV not in os.environ:
    os.environ[_T0_ENV] = repr(time.time())
T0: float = float(os.environ[_T0_ENV])


DEFAULT_CSV_DIR_REL = Path("runs")  # filename is <benchmark>_<backend>.csv
DEFAULT_LOG_DIR_REL = Path("runs") / "logs"

# Matches Triton GSan's race lines and triton-viz's RaceDetector output.
# Anchored to start-of-line (after optional whitespace/`>` markers) so we
# don't false-match `[driver] If GSan reports 'race detected' on ...` style
# commentary that race-candidate drivers print to document what they expect.
DEFAULT_RACE_PATTERN = re.compile(
    r"^\s*(?:>\s*)?"
    r"(?:(?:Read after write|Write after read|Write after write) race detected"
    r"|Race detected)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Repo root (this file lives at the repo root; orchestrator subprocess `cwd`)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent
_TRITON_VIZ_PATH: Path = REPO_ROOT / "third_party" / "triton-viz"


# ---------------------------------------------------------------------------
# Backend description
# ---------------------------------------------------------------------------
@dataclass
class BackendConfig:
    name: str                                 # used to build CSV/log filenames
    extra_env: dict = field(default_factory=dict)   # setdefault'd, user wins
    extra_paths: list = field(default_factory=list) # added to sys.path / PYTHONPATH
    race_pattern: re.Pattern = DEFAULT_RACE_PATTERN
    pytest_timeout: int = 60                  # per-test --timeout in seconds

    @property
    def tag(self) -> str:
        return f"[{self.name}-runner]"


# ---------------------------------------------------------------------------
# Registry of all supported backends
# ---------------------------------------------------------------------------
# The `name` field is what `resolve_paths` plugs into the CSV/log filename
# (`runs/<benchmark>_<name>.csv`), so we keep the `_pytest` suffix that
# was in use before the refactor (existing CSVs stay compatible).
#
# The `BACKEND` env var injected here is what the pytest subprocess's
# `conftest.py` dispatches on — short name, no suffix.
BACKEND_SPECS: dict[str, BackendConfig] = {
    "gsan": BackendConfig(
        name="gsan_pytest",
        extra_env={
            "BACKEND": "gsan",
            "TRITON_DISABLE_LINE_INFO": "0",
            "TRITON_ALWAYS_COMPILE": "1",
        },
        extra_paths=[],
        pytest_timeout=60,
    ),
    "triton_viz": BackendConfig(
        name="triton_viz_pytest",
        extra_env={
            "BACKEND": "triton_viz",
            "TRITON_INTERPRET": "1",
            "TRITON_DISABLE_LINE_INFO": "0",
        },
        extra_paths=[_TRITON_VIZ_PATH],
        # triton-viz runs in interpreter mode (numpy per op) → much slower
        # per-test; raise the per-test timeout. Numpy can't be interrupted
        # so this is a soft ceiling anyway; shell `timeout` is the hard cap.
        pytest_timeout=180,
    ),
    "baseline": BackendConfig(
        name="baseline_pytest",
        extra_env={
            "BACKEND": "baseline",  # conftest.py treats this as alias for "none"
        },
        extra_paths=[],
        pytest_timeout=60,
    ),
}


def get_backend(name: str) -> BackendConfig:
    """Return the `BackendConfig` for a short backend name.

    Raises `ValueError` on unknown name. The returned object is the same
    instance as in `BACKEND_SPECS` — do not mutate it."""
    if name not in BACKEND_SPECS:
        raise ValueError(
            f"Unknown backend: {name!r}. Choose from: "
            f"{sorted(BACKEND_SPECS.keys())}"
        )
    return BACKEND_SPECS[name]


# ---------------------------------------------------------------------------
# Venv re-exec
# ---------------------------------------------------------------------------
def ensure_venv(repo_root: Path, caller_script: str) -> None:
    """Re-exec under `<repo>/.venv/bin/python` if the active interpreter isn't
    it. No-op if the venv doesn't exist (let the caller hit the real
    `ModuleNotFoundError` for triton/torch in that case)."""
    venv_py = repo_root / ".venv" / "bin" / "python"
    if not venv_py.exists():
        return
    if Path(sys.executable).resolve() == venv_py.resolve():
        return
    os.execv(str(venv_py), [str(venv_py), caller_script, *sys.argv[1:]])


# ---------------------------------------------------------------------------
# Environment prep
# ---------------------------------------------------------------------------
def prepare_env(backend: BackendConfig, benchmark_path: Path) -> None:
    extra_paths = [Path(p) for p in backend.extra_paths]

    parts: list[str] = [str(benchmark_path), *(str(p) for p in extra_paths)]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    for k, v in backend.extra_env.items():
        os.environ.setdefault(k, v)

    for p in [benchmark_path, *extra_paths]:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Backend installation (shared by pytest conftest and script_runner)
# ---------------------------------------------------------------------------
@contextmanager
def install_backend(name: str):
    """Install the named race-detector backend for the `with` block. Same
    install steps the legacy pytest fixture in conftest.py used to do inline;
    both conftest.py and scripts/script_runner.py now go through this.

    `baseline` is aliased to `none` (no instrumentation)."""
    name = name.strip().lower()
    if name == "baseline":
        name = "none"

    if name == "gsan":
        import torch
        import triton
        from triton.experimental.gsan import create_mem_pool

        triton.knobs.compilation.instrumentation_mode = "gsan"
        print(
            f"[gsan] instrumentation_mode="
            f"{triton.knobs.compilation.instrumentation_mode}",
            file=sys.stderr,
        )
        pool = create_mem_pool()
        with torch.cuda.use_mem_pool(pool):
            yield
        torch.cuda.synchronize()

    elif name == "triton_viz":
        import triton
        import triton_viz
        from triton_viz.clients import RaceDetector
        from triton_viz.core.config import config as tv_cfg
        from triton_viz.wrapper import create_patched_jit, create_patched_autotune

        tv_cfg.cli_active = True

        def _wrap(kernel):
            tracer = triton_viz.trace(client=RaceDetector(abort_on_error=False))
            return tracer(kernel)

        patched_jit = create_patched_jit(_wrap)
        patched_autotune = create_patched_autotune(_wrap)
        triton.jit = patched_jit
        triton.language.jit = patched_jit
        import triton.runtime.interpreter as _interp
        _interp.jit = patched_jit
        triton.autotune = patched_autotune
        os.environ.setdefault("TRITON_INTERPRET", "1")

        print(
            f"[triton_viz] RaceDetector installed; "
            f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}",
            file=sys.stderr,
        )
        yield

    elif name == "none":
        print("[baseline] no race detector installed", file=sys.stderr)
        yield

    else:
        raise ValueError(
            f"unknown backend {name!r}; "
            f"expected one of gsan, triton_viz, baseline/none"
        )


# ---------------------------------------------------------------------------
# Tee'd stream that also sniffs for race lines
# ---------------------------------------------------------------------------
class _RaceTeeStream:
    """Wraps `underlying`; mirrors writes to `log_fh` (if given); records any
    line that matches `race_pattern` in `race_hits`."""

    def __init__(self, underlying, log_fh, race_pattern: Optional[re.Pattern]):
        self._under = underlying
        self._log = log_fh
        self._pat = race_pattern
        self.race_hits: list[str] = []

    def write(self, data):
        if data:
            if self._pat is not None:
                for line in data.splitlines():
                    if self._pat.search(line):
                        self.race_hits.append(line)
            if self._log is not None:
                self._log.write(data)
                self._log.flush()
        return self._under.write(data)

    def flush(self):
        if self._log is not None:
            self._log.flush()
        return self._under.flush()

    def __getattr__(self, name):
        return getattr(self._under, name)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser, default_benchmark: str) -> None:
    g = parser.add_argument_group("reporting")
    g.add_argument("--csv", default=None,
                   help="CSV results path (default: "
                        f"<repo>/{DEFAULT_CSV_DIR_REL}/<benchmark>_<backend>.csv).")
    g.add_argument("--log", default=None,
                   help="Raw log path (default: auto-generated under --log-dir).")
    g.add_argument("--log-dir", default=None,
                   help=f"Parent dir for default --log (default: <repo>/{DEFAULT_LOG_DIR_REL}).")
    g.add_argument("--no-csv", action="store_true",
                   help="Disable CSV row writing.")
    g.add_argument("--no-log", action="store_true",
                   help="Disable raw log tee.")
    g.add_argument("--benchmark", default=default_benchmark,
                   help=f"Benchmark suite tag for CSV (default: {default_benchmark!r}).")
    g.add_argument("--notes", default="",
                   help="Free-text annotation, written to the CSV row.")


def resolve_paths(
    args: argparse.Namespace,
    repo_root: Path,
    backend_name: str,
    script_path: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    csv_path: Optional[Path]
    if args.no_csv:
        csv_path = None
    elif args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = (
            repo_root / DEFAULT_CSV_DIR_REL / f"{args.benchmark}_{backend_name}.csv"
        )

    log_path: Optional[Path]
    if args.no_log:
        log_path = None
    elif args.log:
        log_path = Path(args.log)
    else:
        log_dir = Path(args.log_dir) if args.log_dir else repo_root / DEFAULT_LOG_DIR_REL
        # No timestamp in the default filename: re-running the same kernel
        # under the same backend overwrites the previous log. Use --log to
        # opt out (e.g. when you want to compare two runs side-by-side).
        log_path = log_dir / f"{backend_name}_{script_path.stem}.log"

    return csv_path, log_path


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "timestamp", "benchmark", "backend", "script",
    "target_seconds", "total_seconds",
    "race_count", "exit_code",
    # pytest-mode rows fill these in; runpy-mode rows leave them blank.
    "passed", "failed", "skipped", "errors",
    "log_path", "notes",
]


# ---------------------------------------------------------------------------
# Pytest log parsing — used by run_aiter_*_pytest.py and the offline re-parser.
# ---------------------------------------------------------------------------
# Match pytest summary lines like:
#   "2 failed, 38 passed in 15.27s"
#   "1972 failed, 2948 passed in 2616.57s (0:43:36)"
#   "no tests ran in 0.20s"
# Require at least one "N <kind>" or "no tests ran" prefix AND "in X.YZs"
# somewhere — robust to trailing "(H:MM:SS)" wall-time suffix.
_SUMMARY_LINE_RE = re.compile(
    r"(?:\d+\s+(?:passed|failed|skipped|errors?|warnings?|deselected|xfailed|xpassed)"
    r"|no tests ran)"
    r".*in \d+(?:\.\d+)?s"
)
_COUNT_RES = {
    "passed":  re.compile(r"(\d+)\s+passed"),
    "failed":  re.compile(r"(\d+)\s+failed"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
    "errors":  re.compile(r"(\d+)\s+errors?"),
}
# Progress-char lines in pytest -q output: optional leading whitespace, then
# a run of .FsExX, then ONE of: whitespace+[NN%], more whitespace, "+++"
# (timeout marker), or end-of-line. Excludes "FAILED benchmarks/...::..." lines
# because those don't fit the suffix.
_PROGRESS_LINE_RE = re.compile(
    r"^(?P<chars>[.FsExX]+)(?:\s|\[|\+|$)"
)


def parse_pytest_summary(text: str) -> dict:
    """Best-effort: pull passed/failed/skipped/errors from pytest -q output.

    Strategy:
      1. Look for the final `"… in X.YZs"` summary line and grep `\\d+ <kind>`.
      2. If that fails (pytest hung before summary, e.g. under triton-viz +
         interpreter-mode + numpy-uninterruptible-timeout), fall back to
         counting the per-test progress characters (`.`/`F`/`s`/`E`) emitted
         live by `pytest -q`.

    Returns dict with keys passed/failed/skipped/errors. All zero if nothing
    parseable was found."""
    out = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    # --- 1. Summary line ---
    summary_line = ""
    for line in reversed(text.splitlines()):
        if _SUMMARY_LINE_RE.search(line):
            summary_line = line
            break
    if summary_line:
        for key, pat in _COUNT_RES.items():
            m = pat.search(summary_line)
            if m:
                out[key] = int(m.group(1))
        if any(out.values()):
            return out

    # --- 2. Fallback: count progress chars ---
    progress = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for line in text.splitlines():
        m = _PROGRESS_LINE_RE.match(line)
        if not m:
            continue
        for c in m.group("chars"):
            if   c == ".": progress["passed"]  += 1
            elif c == "F": progress["failed"]  += 1
            elif c == "s": progress["skipped"] += 1
            elif c == "E": progress["errors"]  += 1
    if any(progress.values()):
        return progress

    return out


def _upsert_csv_row(csv_path: Path, row: dict) -> None:
    """Write `row` to `csv_path`, replacing any existing row with the same
    `script` value. Truncate-rewrite each call to keep semantics consistent
    with the log files (re-running the same target overwrites the previous
    record). Other rows for OTHER scripts in the same CSV are preserved."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r.get("script") != row.get("script")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in existing:
            w.writerow({k: r.get(k, "") for k in _CSV_FIELDS})
        w.writerow({k: row.get(k, "") for k in _CSV_FIELDS})


# ---------------------------------------------------------------------------
# Target runner factory — used by `run.py` for every backend
# ---------------------------------------------------------------------------
def make_pytest_runner(
    backend: BackendConfig,
    test_file: Path,
    extra_csv: dict,
) -> Callable[[], None]:
    """Build a closure that runs `python -m pytest <test_file>` as a
    subprocess. The subprocess inherits the orchestrator's environment
    (including `BACKEND=<short_name>` from `prepare_env`), which the
    repo-root `conftest.py` dispatches on. After the subprocess completes,
    P/F/S/E counts are parsed into `extra_csv` and `SystemExit` is raised
    with pytest's returncode so `run_with_reporting` records it as
    `exit_code`."""

    def _runner() -> None:
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-q", "--no-header",
            "--tb=line",
            f"--timeout={backend.pytest_timeout}",
            "--timeout-method=thread",
            "-p", "no:cacheprovider",
        ]
        print(f"{backend.tag} cmd: {' '.join(cmd)}", file=sys.stderr)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
        )

        buffered: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)  # tee'd by run_with_reporting
            buffered.append(line)
        proc.wait()

        counts = parse_pytest_summary("".join(buffered))
        extra_csv.update(counts)

        # Propagate pytest's exit code via SystemExit so the SystemExit
        # handler in run_with_reporting picks it up as our exit_code.
        raise SystemExit(proc.returncode)

    return _runner


# ---------------------------------------------------------------------------
# Script runner factory (TritonBench-style benchmarks — plain `python <file>`)
# ---------------------------------------------------------------------------
def make_script_runner(
    backend: BackendConfig,
    test_file: Path,
    extra_csv: dict,
) -> Callable[[], None]:
    """Runner for benchmarks that run as plain `python <file>.py` (no pytest).
    Spawns `python scripts/script_runner.py <file>`; that wrapper reads the
    BACKEND env, calls `install_backend(...)`, then `runpy.run_path`s the
    target.

    `extra_csv` is NOT populated with passed/failed/skipped/errors — those are
    pytest concepts. Treat `exit_code == 0` as the pass signal."""

    def _runner() -> None:
        wrapper = REPO_ROOT / "scripts" / "script_runner.py"
        cmd = [sys.executable, str(wrapper), str(test_file)]
        print(f"{backend.tag} cmd: {' '.join(cmd)}", file=sys.stderr)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
        proc.wait()

        raise SystemExit(proc.returncode)

    return _runner


# ---------------------------------------------------------------------------
# The main entry point used by each `run_*` script
# ---------------------------------------------------------------------------
def run_with_reporting(
    *,
    backend: BackendConfig,
    benchmark: str,
    script_path: Path,
    script_args: list[str],
    csv_path: Optional[Path],
    log_path: Optional[Path],
    notes: str,
    target_runner: Callable[[], None],
    extra_csv: Optional[dict] = None,
) -> int:
    """Run `target_runner()` with stdout+stderr tee'd to `log_path`, sniff for
    race lines, time the call, append a CSV row, return an exit code.

    `extra_csv` is a dict the runner may mutate before raising / returning;
    its keys (when they appear in `_CSV_FIELDS`) override the defaults in the
    written row. Used by the pytest runner to report `passed/failed/...`."""

    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: each run overwrites its log. CSV (which is append-only)
        # is the place that accumulates history.
        log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
        log_fh.write(
            f"===== {time.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"benchmark={benchmark} backend={backend.name} "
            f"script={script_path} =====\n"
        )
        log_fh.flush()
        print(f"{backend.tag} log -> {log_path}", file=sys.stderr)

    stderr_tee = _RaceTeeStream(sys.stderr, log_fh, backend.race_pattern)
    stdout_tee = _RaceTeeStream(sys.stdout, log_fh, backend.race_pattern)
    sys.stderr = stderr_tee
    sys.stdout = stdout_tee

    sys.argv = [str(script_path), *script_args]

    t_target_start = time.perf_counter()
    t_target_end = t_target_start
    exit_code = 0

    try:
        target_runner()
        t_target_end = time.perf_counter()
    except SystemExit as e:
        t_target_end = time.perf_counter()
        if isinstance(e.code, int):
            exit_code = e.code
        elif e.code is None:
            exit_code = 0
        else:
            print(str(e.code), file=sys.stderr)
            exit_code = 1
    except BaseException:
        t_target_end = time.perf_counter()
        traceback.print_exc(file=sys.stderr)
        exit_code = 2
    finally:
        # While streams are still tee'd, write the summary so it lands in the
        # log file too. Restore + close the log AFTER.
        race_count = len(stderr_tee.race_hits) + len(stdout_tee.race_hits)
        if race_count > 0:
            exit_code = max(exit_code, 1)
            print(f"{backend.tag} {race_count} race line(s) detected",
                  file=sys.stderr)
        elif exit_code == 0:
            print(f"{backend.tag} OK — no race lines on stderr",
                  file=sys.stderr)

        target_s = t_target_end - t_target_start
        total_s = time.time() - T0
        print(
            f"{backend.tag} timing target={target_s:.3f}s total={total_s:.3f}s",
            file=sys.stderr,
        )

        sys.stderr = stderr_tee._under
        sys.stdout = stdout_tee._under
        if log_fh is not None:
            log_fh.close()

        if csv_path is not None:
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "benchmark": benchmark,
                "backend": backend.name,
                "script": str(script_path),
                "target_seconds": f"{target_s:.6f}",
                "total_seconds": f"{total_s:.6f}",
                "race_count": race_count,
                "exit_code": exit_code,
                "log_path": str(log_path) if log_path is not None else "",
                "notes": notes,
            }
            if extra_csv:
                row.update(extra_csv)
            _upsert_csv_row(csv_path, row)
            print(f"{backend.tag} csv -> {csv_path}", file=sys.stderr)

    return exit_code
