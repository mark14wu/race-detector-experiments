"""Run a Python script with Triton's GSan (global memory race detector) enabled.

Usage:
    python run_with_gsan.py path/to/script.py [-- script args...]

Env vars worth setting alongside this wrapper:
    TRITON_DISABLE_LINE_INFO=0   # keep line info so race reports include source lines
    TRITON_ALWAYS_COMPILE=1      # force recompile so GSan instrumentation isn't skipped via cache

Notes:
    - Requires NVIDIA/CUDA. GSan is not implemented on the ROCm backend.
    - Tensors that GSan should observe must be allocated inside the
      `with torch.cuda.use_mem_pool(pool):` context.
    - This wrapper sets `triton.knobs.compilation.instrumentation_mode = "gsan"`
      directly, so the env var TRITON_INSTRUMENTATION_MODE is not required.
"""
import argparse
import os
import runpy
import sys

import torch
import triton
from triton.experimental.gsan import create_mem_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    repo_root = os.path.dirname(os.path.abspath(__file__))
    aiter_path = os.path.join(repo_root, "aiter")
    if os.path.isdir(aiter_path) and aiter_path not in sys.path:
        sys.path.insert(0, aiter_path)

    # Mimic `python path/to/script.py` so local helper imports (e.g. `from common import ...`)
    # work. runpy.run_path does not do this automatically.
    script_path = os.path.abspath(args.script)
    script_dir = os.path.dirname(script_path)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    triton.knobs.compilation.instrumentation_mode = "gsan"
    print(
        f"[gsan-wrapper] instrumentation_mode={triton.knobs.compilation.instrumentation_mode}",
        file=sys.stderr,
    )

    pool = create_mem_pool()
    sys.argv = [script_path] + script_args

    with torch.cuda.use_mem_pool(pool):
        runpy.run_path(script_path, run_name="__main__")

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
