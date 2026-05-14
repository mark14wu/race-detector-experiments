# race-detector-experiments

Run AITER's Triton kernels under Triton GSan (global memory race detector) on
NVIDIA/CUDA. AITER's CK/HIP paths are out of scope — we only need its Triton
kernels importable. For real bug-hunting, extract a kernel into
`extracted_kernels/` and write a tiny driver; AITER op_tests are noisy under
GSan (see `AGENTS.md`).

## Layout

```
aiter/                          ROCm/aiter submodule (NOT pip-installed)
third_party/triton/             triton-lang/triton, built from source
third_party/triton-viz/         triton-viz on race-detector-z3-demo
run_aiter_gsan.py               main entry point (env + GSan + race sniffer)
run_with_gsan.py                low-level GSan wrapper (no sniffer)
conftest.py                     enables GSan for pytest sweeps
extracted_kernels/              minimal Triton drivers
scripts/setup_cuda_gsan.sh      one-time: torch cu128 + Triton source build
```

## Setup

Requires Linux, NVIDIA GPU, CUDA 12.8+ driver, `uv`.

```bash
git clone --recursive <this repo> && cd race-detector-experiments
uv venv --python 3.11 && source .venv/bin/activate && uv sync
bash scripts/setup_cuda_gsan.sh
```

Sanity-check the GSan path itself:

```bash
TRITON_DISABLE_LINE_INFO=0 python -m pytest -n 8 third_party/triton/python/test/gsan
```

## Running a kernel under GSan

```bash
python run_aiter_gsan.py                                    # defaults to demo_add.py
python run_aiter_gsan.py extracted_kernels/<driver>.py
python run_aiter_gsan.py extracted_kernels/<driver>.py -- --n 4096
```

The orchestrator re-execs under `.venv`, sets `PYTHONPATH=aiter` +
`TRITON_DISABLE_LINE_INFO=0` + `TRITON_ALWAYS_COMPILE=1`, configures
`instrumentation_mode="gsan"`, runs the driver inside `use_mem_pool`, and
sniffs stderr for race lines. Exit codes: `0` clean, `1` race detected,
`2` target raised, `3` env problem.

Race lines look like `Read after write race detected` /
`Write after read race detected` / `Write after write race detected`,
each followed by a source location.

For pytest sweeps or direct env control, drop down to `run_with_gsan.py`.
The full extract-a-kernel recipe and op_tests caveats live in `AGENTS.md`.

## Known working environment

```text
triton HEAD : ca21b1b95798f632c03dfaeb8ad4c9a78506860c
aiter HEAD  : d295caf6b977b3b0af02a9de06722811fb529cf3
python      : 3.11.13
torch       : 2.11.0+cu128
triton      : 3.7.0+gitca21b1b9
GPU         : RTX 4090 (sm_89), driver 580.126.20
```

On B200 + CUDA 13 the Triton submodule needs
`patches/triton-gsan-blackwell.patch` — see `patches/triton-gsan-blackwell.md`.
