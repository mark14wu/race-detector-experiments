"""Unified orchestrator for the benchmark × backend matrix.

Single CLI:

    python run.py --backend {gsan,triton_viz,baseline} \
                  [--benchmark NAME] [reporting flags]   \
                  <test_file> [-- pytest_args]

`<test_file>` is a pytest test module (e.g. one of the 72 files in
`benchmarks/aiter/pytest_files.txt`). The orchestrator spawns
`python -m pytest <test_file>` in a subprocess and lets the repo-root
`conftest.py` install the race-detector fixture based on the `BACKEND=<name>`
env var that `prepare_env` injects.

CSV output: `runs/<benchmark>_<backend>_pytest.csv` (upsert by `script`).
Log output: `runs/logs/<backend>_pytest_<stem>.log` (truncated each run).

Adding a new benchmark suite: drop
`benchmarks/<name>/pytest_files.txt` + `benchmarks/<name>/inventory.md`,
then pass `--benchmark <name>`.

Exit codes mirror pytest's own (0 = all passed; 1 = some failed; 5 = no
tests collected; 2 = collection error; 124 = killed by shell `timeout`).
A non-zero `race_count` in the CSV row signals that the race sniffer
caught a race report regardless of pytest's own exit code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_CHOICES = ["gsan", "triton_viz", "baseline"]

import run_common  # noqa: E402
run_common.ensure_venv(REPO_ROOT, __file__)

from run_common import (  # noqa: E402
    add_common_args,
    get_backend,
    make_pytest_runner,
    prepare_env,
    resolve_paths,
    run_with_reporting,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Unified race-detector orchestrator. "
                    "Runs one pytest test file under a chosen backend; "
                    "writes per-file CSV row + tee'd log.",
    )
    p.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        required=True,
        help="Race-detection backend.",
    )
    p.add_argument(
        "script",
        help="Path to a pytest test_*.py file (e.g. "
             "aiter/op_tests/triton_tests/test_topk.py).",
    )
    p.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to pytest. Use `--` to separate.",
    )
    add_common_args(p, default_benchmark="aiter")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    target = Path(args.script).resolve()
    if not target.is_file():
        print(f"[run] target not found: {target}", file=sys.stderr)
        return 3

    forwarded = args.script_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    backend = get_backend(args.backend)
    prepare_env(REPO_ROOT, backend)

    csv_path, log_path = resolve_paths(args, REPO_ROOT, backend.name, target)

    extra_csv: dict = {}
    runner = make_pytest_runner(backend, target, extra_csv)

    return run_with_reporting(
        backend=backend,
        benchmark=args.benchmark,
        script_path=target,
        script_args=forwarded,
        csv_path=csv_path,
        log_path=log_path,
        notes=args.notes,
        target_runner=runner,
        extra_csv=extra_csv,
    )


if __name__ == "__main__":
    sys.exit(main())
