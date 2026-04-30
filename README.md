# race-detector-experiments

Run AITER's Triton-only kernels under Triton's GSan (global memory race detector) on NVIDIA/CUDA.

> **Scope.** We do **not** try to make AITER fully work on CUDA. AITER is a ROCm
> project; here we only need its Triton kernels importable so we can probe them
> with GSan. CK/HIP/FlyDSL paths are out of scope.
>
> **For actual bug hunting, prefer extracted minimal kernels over full AITER op_tests.**
> AITER op_tests carry test harness, shape sweeps, autotune, ROCm helpers, and CLI
> parsing as noise. GSan is a race detector, not a validator that AITER's test
> framework runs on CUDA. Use op_tests only for smoke checks; for real
> investigation, copy the kernel into `extracted_kernels/` and write a minimal
> driver.

## Layout

```
.
├── aiter/                          # ROCm/aiter submodule (not pip-installed)
├── third_party/triton/             # triton-lang/triton submodule, built from source
├── scripts/setup_cuda_gsan.sh      # installs torch (cu128) + builds Triton main
├── run_with_gsan.py                # wrapper that runs a script with GSan enabled
├── pyproject.toml + uv.lock        # base Python deps (uv-managed)
├── .python-version                 # 3.11
└── extracted_kernels/              # (created on demand) minimal Triton kernels
```

## Setup

Requires: Linux, NVIDIA GPU, NVIDIA driver supporting CUDA 12.8+, `uv`.

```bash
git clone --recursive <this repo>
cd race-detector-experiments
uv venv --python 3.11
source .venv/bin/activate
uv sync
bash scripts/setup_cuda_gsan.sh   # installs torch cu128 + builds Triton from source
```

Verify the GSan path itself before touching AITER:

```bash
cd third_party/triton
TRITON_DISABLE_LINE_INFO=0 python -m pytest -n 8 python/test/gsan
cd ../..
```

If that doesn't pass, nothing downstream will.

## Running an AITER Triton-only file under GSan

Find candidate files (those that don't import top-level `aiter`):

```bash
find aiter/op_tests/triton_tests -maxdepth 4 -type f -name "*.py" -print0 \
  | xargs -0 grep -L -E '^[[:space:]]*(import|from)[[:space:]]+aiter(\.|[[:space:]])' \
  | head -20
```

Spot-check one for ROCm/HIP/CK/FlyDSL leaks:

```bash
grep -nE 'aiter|hip|rocm|ck|jit\.core|flydsl' <candidate_file.py>
```

Run it under GSan:

```bash
TRITON_DISABLE_LINE_INFO=0 \
TRITON_ALWAYS_COMPILE=1 \
python run_with_gsan.py aiter/op_tests/triton_tests/<file>.py
```

Race messages on stderr:

- `Read after write race detected`
- `Write after read race detected`
- `Write after write race detected`

`TRITON_DISABLE_LINE_INFO` is a *disable* switch. `=1` removes line info; we set
`=0` to **keep** it so race reports include source lines.

## Fallback: extracted minimal kernel

If the op_tests path is too tangled, copy a kernel into `extracted_kernels/` and
write a tiny driver that only imports `torch`, `triton`, `triton.language`:

```bash
mkdir -p extracted_kernels
cp aiter/aiter/ops/triton/<kernel_file>.py extracted_kernels/
$EDITOR extracted_kernels/driver_<kernel>.py
python run_with_gsan.py extracted_kernels/driver_<kernel>.py
```

This is the recommended path for actually finding races.

## Patch fallback (only if you must `pip install aiter`)

`aiter/setup.py:148-149` raises `NotImplementedError("Only ROCM is supported")`
on any non-ROCm install — including `BUILD_TARGET=cuda`, which falls into the
`else` branch and explicitly sets `IS_ROCM = False`.

If a specific op_test really needs `aiter` as an installed package (rare for
Triton-only paths), apply `patches/aiter-cuda-experimental.patch` to the
submodule:

```bash
cd aiter
git apply ../patches/aiter-cuda-experimental.patch
cd ..
BUILD_TARGET=cuda uv pip install -e ./aiter --no-build-isolation
```

This is **experimental and local-only** — it skips the CK/HIP build but does not
make AITER's full surface CUDA-correct. The patch is **not** committed into the
aiter submodule. If install fails further down (other HIP/CK assumptions in
setup.py), don't keep patching — switch to the extracted-kernel fallback above.

## Known working environment

`python/test/gsan` from Triton main passed (85 passed, 11 skipped) on this setup:

```text
triton submodule HEAD : ca21b1b95798f632c03dfaeb8ad4c9a78506860c
aiter submodule HEAD  : d295caf6b977b3b0af02a9de06722811fb529cf3
python                : 3.11.13 (uv-managed)
torch                 : 2.11.0+cu128 (cuda 12.8)
triton (built)        : 3.7.0+gitca21b1b9
GPU                   : NVIDIA GeForce RTX 4090 (sm_89)
driver                : 580.126.20 (CUDA driver-side 13.0)
OS                    : Linux 6.8.0-106-generic
```

`extracted_kernels/demo_add.py` runs cleanly under `run_with_gsan.py` on this
setup (no races reported, as expected for a race-free vector add).

Refresh after updating submodules:

```bash
git -C third_party/triton rev-parse HEAD
git -C aiter rev-parse HEAD
python -c "import torch, triton; print('torch', torch.__version__, torch.version.cuda); print('triton', triton.__version__)"
nvidia-smi | head -20
```
