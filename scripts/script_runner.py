"""Subprocess entry point for benchmarks that run as plain `python <file>.py`
(currently: TritonBench). The pytest equivalent is conftest.py + the test
file itself.

Reads BACKEND from env, installs the matching race detector via
run_common.install_backend(...), then runs the target with runpy.run_path.

Usage (orchestrated):
    BACKEND=gsan PYTHONPATH=... python scripts/script_runner.py <target.py>

The orchestrator (run.py + run_common.make_script_runner) sets BACKEND,
BENCHMARK, PYTHONPATH before spawning this wrapper.

Exit codes:
    0  target completed
    1  target raised (traceback printed to stderr)
    2  bad CLI args
    3  target file not found
   <N> if target called sys.exit(N)
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_common import install_backend  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <target.py>", file=sys.stderr)
        return 2

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        print(f"[script_runner] target not found: {target}", file=sys.stderr)
        return 3

    backend = os.environ.get("BACKEND", "baseline").strip().lower()
    print(
        f"[script_runner] BACKEND={backend} target={target}",
        file=sys.stderr,
    )

    with install_backend(backend):
        try:
            runpy.run_path(str(target), run_name="__main__")
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1
        except BaseException:
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
