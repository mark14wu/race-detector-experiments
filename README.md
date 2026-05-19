# race-detector-experiments

Run AITER's Triton kernels under multiple race-detector backends on NVIDIA/CUDA.
AITER's CK/HIP paths are out of scope — we only need its Triton kernels
importable so we can probe them.

Three race-detector backends are supported (`--backend ...`):

- `gsan` — Triton's GSan (global memory race detector). Hardware-accurate but
  needs a private CUDA mem pool that overflows on production shapes.
- `triton_viz` — triton-viz's Z3-backed `RaceDetector` running under
  `TRITON_INTERPRET=1`. Slow but no OOM.
- `baseline` — plain Triton, no instrumentation. Control runs.

## Layout

```
benchmarks/<name>/<name>/       benchmark source submodule (NOT pip-installed)
benchmarks/<name>/              per-benchmark test list + inventory.md
third_party/triton/             triton-lang/triton, built from source
third_party/triton-viz/         triton-viz on race-detector-z3-demo
run.py                          unified entry: --backend × pytest test_file
run_common.py                   shared scaffolding (BackendConfig registry, etc.)
conftest.py                     installs the chosen backend per session
```

## Setup

Requires Linux, NVIDIA GPU, CUDA 12.8+ driver, `uv`.

```bash
git clone --recursive <this repo> && cd race-detector-experiments
uv venv --python 3.11 && source .venv/bin/activate
uv sync --extra cuda
```

`uv sync --extra cuda` installs torch (PyTorch cu128 wheel index), builds Triton
from `third_party/triton` (source, editable), installs triton-viz from
`third_party/triton-viz` (source, editable), plus pure-Python test deps.
First-time Triton build is ~3 min. On ROCm hosts use `--extra rocm` instead.

Sanity-check the GSan path itself:

```bash
TRITON_DISABLE_LINE_INFO=0 python -m pytest -n 8 third_party/triton/python/test/gsan
```

## Running

```bash
# One test file under one backend:
python run.py --backend gsan        benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py
python run.py --backend triton_viz  benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py
python run.py --backend baseline    benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py

# Forward extra pytest args:
python run.py --backend gsan benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py -- -k small
```

Output per run:
- **CSV row** at `runs/<benchmark>_<backend>_pytest.csv` (upsert by `script`).
- **Raw log** at `runs/logs/<backend>_pytest_<stem>.log` (truncated per run).

CSV columns include `passed/failed/skipped/errors` (parsed from pytest's
summary line, with a progress-char fallback for runs where pytest got
SIGTERM'd mid-stream). `race_count` is independent of pytest's exit code:
the orchestrator sniffs stdout+stderr for race lines.

The full AITER inventory (27,689 tests, why they pass/fail/skip) is in
`benchmarks/aiter/inventory.md`. The canonical 72-file list is
`benchmarks/aiter/pytest_files.txt`.

## Known working environment

```text
triton HEAD : ed8317b20881e443aaf6c91d161cbacf6143dc53
aiter HEAD  : d295caf6b977b3b0af02a9de06722811fb529cf3
python      : 3.11.13
torch       : 2.9.0+cu128
triton      : 3.7.0+gited8317b2
GPU         : RTX 4090 (sm_89), driver 580.126.20
```

On B200 + CUDA 13 the Triton submodule needs
`patches/triton-gsan-blackwell.patch` — see `patches/triton-gsan-blackwell.md`.
