"""Pytest conftest that enables a race-detector backend for the entire
session. Dispatches on the `BACKEND` env var:

  * `BACKEND=gsan` (default, including unset) — Triton GSan path. Mirrors
    `triton/python/test/gsan`'s pattern: set `instrumentation_mode = "gsan"`,
    create a CUDA mem pool, run all tests inside `with use_mem_pool(pool):`.

  * `BACKEND=triton_viz` — triton-viz's `RaceDetector` path. Forces
    `TRITON_INTERPRET=1` and monkey-patches `triton.jit` / `triton.autotune`
    so every kernel runs through the interpreter under `RaceDetector`.

  * `BACKEND=baseline` or `BACKEND=none` — no instrumentation; plain pytest.
    `baseline` is the user-facing name; it's normalized to `none` internally.
    Useful as a control next to gsan/triton_viz.

Place this at the repo root so pytest picks it up regardless of which test
directory is invoked.
"""
import os
import sys

# Make `import aiter` resolve to ./aiter/aiter without installing the package.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_AITER_PATH = os.path.join(_REPO_ROOT, "aiter")
if os.path.isdir(_AITER_PATH) and _AITER_PATH not in sys.path:
    sys.path.insert(0, _AITER_PATH)

# triton-viz lives in third_party/triton-viz and is not always pip-installed.
_TRITON_VIZ_PATH = os.path.join(_REPO_ROOT, "third_party", "triton-viz")
if os.path.isdir(_TRITON_VIZ_PATH) and _TRITON_VIZ_PATH not in sys.path:
    sys.path.insert(0, _TRITON_VIZ_PATH)

_BACKEND = os.environ.get("BACKEND", "gsan").strip().lower()
# `baseline` is an alias for `none` (plain Triton, no instrumentation).
if _BACKEND == "baseline":
    _BACKEND = "none"

import pytest  # noqa: E402


def _setup_gsan():
    import torch
    import triton
    from triton.experimental.gsan import create_mem_pool

    triton.knobs.compilation.instrumentation_mode = "gsan"
    print(
        f"[gsan-conftest] instrumentation_mode={triton.knobs.compilation.instrumentation_mode}",
        file=sys.stderr,
    )
    return create_mem_pool(), torch


def _install_triton_viz_patches():
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

    print(
        f"[triton_viz-conftest] RaceDetector installed; "
        f"TRITON_INTERPRET={os.environ.get('TRITON_INTERPRET')}",
        file=sys.stderr,
    )


@pytest.fixture(scope="session", autouse=True)
def _race_detector_session():
    if _BACKEND == "gsan":
        pool, torch = _setup_gsan()
        with torch.cuda.use_mem_pool(pool):
            yield
        torch.cuda.synchronize()
    elif _BACKEND == "triton_viz":
        # TRITON_INTERPRET must be set before triton is imported by tests; we
        # ensure that here. The parent orchestrator should also pre-set it
        # in env so subprocess pytest collection sees it.
        os.environ.setdefault("TRITON_INTERPRET", "1")
        _install_triton_viz_patches()
        yield
    elif _BACKEND == "none":
        print("[conftest] BACKEND=none — no race detector installed",
              file=sys.stderr)
        yield
    else:
        raise pytest.UsageError(
            f"Unknown BACKEND={_BACKEND!r}; expected one of "
            f"'gsan', 'triton_viz', 'none'."
        )
