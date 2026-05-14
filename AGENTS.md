# AGENTS.md

Guidance for AI agents running experiments in this repo. Read before doing anything.

## What this repo is

A workspace for running **ROCm/aiter's Triton kernels under NVIDIA Triton's GSan
(global memory race detector)** on a CUDA machine. AITER itself is ROCm-only;
we deliberately bypass its CK/HIP paths and only target the Triton subset.

Read `README.md` for a human-facing overview and `SURVEY.md` for the diagnostic
results from the first sweep.

## Bootstrap (every shell session)

```bash
cd /home/hwu27/workspace/race-detector-experiments
source .venv/bin/activate
export PYTHONPATH="$PWD/aiter:$PYTHONPATH"   # AITER is NOT pip-installed
```

That's it. `torch`, `triton` (built from source), and base deps are already
installed in `.venv/`. Do **not** run `scripts/setup_cuda_gsan.sh` — it has
already run and rebuilds Triton from source (~6 min).

## Verified working state

These have been run and confirmed; **don't redo them just to check**:

| Check | Command | Result |
| --- | --- | --- |
| Triton's own GSan tests | `cd third_party/triton && TRITON_DISABLE_LINE_INFO=0 pytest -n 4 python/test/gsan` | 85 passed, 11 skipped (~23s) |
| `import aiter` works on CUDA | `python -c "import aiter"` | warns, then OK (Triton ops available) |
| End-to-end GSan wrapper | `python run_with_gsan.py extracted_kernels/demo_add.py` | passes, no race (expected) |

## Known working environment

- GPU: NVIDIA GeForce RTX 4090 (sm_89), driver 580.126.20
- Python: 3.11.13 (uv-managed)
- torch: 2.11.0+cu128
- triton: 3.7.0+gitca21b1b9 (source build)
- aiter HEAD: `d295caf6b977b3b0af02a9de06722811fb529cf3`
- triton HEAD: `ca21b1b95798f632c03dfaeb8ad4c9a78506860c`

## Hard constraints — don't fight these

These cost real time to discover. **Trust the prior diagnosis in `SURVEY.md`.**

1. **AITER does not install on CUDA.** `aiter/setup.py:148` raises
   `NotImplementedError("Only ROCM is supported")`. `BUILD_TARGET=cuda` does
   *not* bypass it — it explicitly sets `IS_ROCM = False` and falls into the
   raise. Use PYTHONPATH; don't `pip install -e ./aiter`.

2. **GSan's mem pool is the binding constraint, not race detection.** GSan
   uses `cuMemAddressReserve`/`cuMemMap` to put both real tensors AND a 6×
   shadow region in GPU VRAM (24 B `ShadowCell` per 4 B real). On RTX 4090
   the practical ceiling for tracked tensors is ~3 GiB. AITER op_tests use
   production shapes (e.g. vocab=128256, ~654 MiB+ tensors) and OOM in the
   private pool even though the GPU has 22+ GiB free.
   → **Do not sweep `op_tests/triton_tests/` under GSan.** ~94% will fail to
   OOM, drowning out any real race signal.
   → For real bug hunting, extract one kernel + write a small-shape driver.

3. **Some AITER Triton kernels use AMD-only kwargs** like `waves_per_eu`,
   which NVIDIA Triton rejects with `KeyError`. If a kernel hits this, scrub
   the kwarg or skip that kernel.

4. **19 of ~70 `op_tests/triton_tests/` files fail collection** because they
   `from aiter import dtypes`, and `dtypes` is gated behind a ROCm-only JIT
   init in `aiter/__init__.py`. Patching that chain bottoms out in
   `aiter.ops.enum` calling `build_module()` at import. Don't try to
   completely fix this; route around it.

## Preferred entry point: `run_aiter_gsan.py`

There is now a Python orchestrator at the repo root that wraps the entire
"run one kernel driver under GSan" flow. **Prefer it over hand-rolling the
`source .venv/... && export PYTHONPATH=... && TRITON_*=... python
run_with_gsan.py ...` chain.** What it handles:

- Re-execs under `.venv/bin/python` if the caller used a different interpreter.
- Sets `PYTHONPATH=aiter`, `TRITON_DISABLE_LINE_INFO=0`,
  `TRITON_ALWAYS_COMPILE=1` (the latter two with `setdefault`, so user
  overrides win).
- Sets `triton.knobs.compilation.instrumentation_mode = "gsan"` directly
  (env-var-only won't take effect once triton is imported).
- Creates the GSan mem pool and runs the target script inside
  `with torch.cuda.use_mem_pool(pool):` via `runpy.run_path`.
- Wraps stderr with a sniffer that matches `(Read|Write) after (read|write)
  race detected` lines and sets a machine-readable exit code.

```bash
python run_aiter_gsan.py                                   # defaults to extracted_kernels/demo_add.py
python run_aiter_gsan.py extracted_kernels/my_driver.py
python run_aiter_gsan.py extracted_kernels/my_driver.py -- --n 4096
```

Exit codes: `0` clean, `1` race lines detected, `2` target raised, `3`
environment problem (no CUDA, missing script, ...). The lines themselves are
still written to stderr — the exit code is in addition, not in place of them.

**When to fall back to `run_with_gsan.py`:** debugging triton itself, custom
env you'd rather set explicitly, or any flow where you don't want a sniffer
between the target and your terminal.

**Don't** use `run_aiter_gsan.py` as the harness for the `op_tests` pytest
sweep — that path goes through `conftest.py`'s session-scope fixture, not
through this orchestrator. The orchestrator is single-script only.

## Running a new experiment (the main path)

The supported workflow is **extract one kernel + tiny driver**. This avoids
all four constraints above.

```bash
# 1. Pick a kernel from one of these dirs
ls aiter/aiter/ops/triton/                  # public kernel modules
ls aiter/aiter/ops/triton/_triton_kernels/  # internal @triton.jit kernels

# 2. Copy it (or a self-contained chunk of it) out
cp aiter/aiter/ops/triton/_triton_kernels/<name>.py extracted_kernels/

# 3. Strip aiter-internal imports (e.g. `make_kernel_repr`) — drop the @repr
#    decorator, keep only `import triton`, `import triton.language as tl`,
#    and torch.

# 4. Add a `main()` that allocates SMALL torch.cuda tensors (KB–MiB range,
#    NOT GiB) and launches the kernel. Use extracted_kernels/demo_add.py
#    as the template.

# 5. Run under GSan — preferred:
python run_aiter_gsan.py extracted_kernels/<your_driver>.py

#    or the lower-level wrapper if you need direct env control:
TRITON_DISABLE_LINE_INFO=0 TRITON_ALWAYS_COMPILE=1 \
  python run_with_gsan.py extracted_kernels/<your_driver>.py
```

What you're looking for in stderr:

- `Read after write race detected`
- `Write after read race detected`
- `Write after write race detected`

…each followed by source location (because `TRITON_DISABLE_LINE_INFO=0`).

If the kernel allocates inside the wrapper's `with torch.cuda.use_mem_pool(pool):`
(which it will, since allocations inside the `runpy.run_path` body are
captured), GSan will see them.

## Reproducing the partial sweep (~1460 tests, the SURVEY.md numbers)

The full triton_tests sweep collects ~27666 tests. Under GSan it does **not**
finish in reasonable time — the dominant failure mode is OOM in the GSan
private pool (see constraint #2), and each failed test still has to compile
under instrumentation. The numbers in `SURVEY.md` come from killing the run
at ~5% progress (~1460 tests). To reproduce that exact recipe:

### Step 1. Bootstrap (must be done in this shell)

```bash
cd /home/hwu27/workspace/race-detector-experiments
source .venv/bin/activate
export PYTHONPATH="$PWD/aiter:$PYTHONPATH"
```

`PYTHONPATH` is mandatory — AITER is not pip-installed.

### Step 2. Launch the sweep in the background, logging to a file

```bash
TRITON_DISABLE_LINE_INFO=0 \
  python -m pytest -q --no-header --tb=no \
    --timeout=300 --timeout-method=thread \
    -p no:cacheprovider \
    aiter/op_tests/triton_tests \
    > /tmp/aiter_gsan_run.log 2>&1 &
SWEEP_PID=$!
echo "sweep pid=$SWEEP_PID, log=/tmp/aiter_gsan_run.log"
```

The `conftest.py` at the repo root supplies a session-scope autouse fixture
that puts every test inside `with torch.cuda.use_mem_pool(create_mem_pool()):`,
so GSan is automatically active for the whole sweep.

### Step 3. Wait ~5 minutes and kill at ~5% progress

```bash
# Peek at progress every minute or so:
tail -3 /tmp/aiter_gsan_run.log

# When the progress marker (e.g. "[ 5%]") shows ~5%, kill the sweep:
pkill -f "pytest.*triton_tests"
# or: kill $SWEEP_PID
```

The sweep run from `SURVEY.md` was killed at the `[ 5%]` mark and yielded
**1460 outcome characters** in the log. Killing earlier or later will give
proportionally fewer/more.

### Step 4. Count outcomes (this is what SURVEY.md tabulates)

```bash
# Each test contributes one of: . F s E in the progress line
grep -oE '\.|F|s|E' /tmp/aiter_gsan_run.log | sort | uniq -c
```

Reference numbers from the original run:

```
     49 .     # passed   ~3.4%
   1345 F     # failed   ~94.0%   (overwhelmingly OOM in private pool)
     66 s     # skipped  ~4.6%
                                  total = 1460
```

The ratio is the load-bearing finding — not the absolute count. Same trend
should hold across kill points and minor Triton/AITER version drift.

### Step 5. (Optional) Confirm a sample failure is OOM, not a real race

```bash
# Pick any failing test file and re-run with verbose tracebacks (small subset, fast)
TRITON_DISABLE_LINE_INFO=0 \
  python -m pytest --no-header --tb=line -q --timeout=60 \
    aiter/op_tests/triton_tests/test_topk.py 2>&1 | tail -20
```

Expected: failures show `torch.OutOfMemoryError: CUDA out of memory ... in
private pools` even though the GPU is mostly free. This is the diagnostic
signature from `SURVEY.md` — *not* a GSan race report.

### What NOT to do here

- **Don't run without `&` in the foreground.** It will hang the shell
  for hours; the foreground `pkill` won't fire.
- **Don't omit `-p no:cacheprovider`** in a shared workspace — pytest cache
  files inside the aiter submodule will pollute its working tree.
- **Don't expect 27666 / X% / Y%-style numbers.** The progress percent stops
  being meaningful once you kill mid-run; report against the 1460 total.
- **Don't claim "AITER has races detected"** based on this sweep. The 94%
  failure is allocator-OOM, not race detection.

## Key files

| Path | Purpose |
| --- | --- |
| `run_aiter_gsan.py` | **Preferred** single-script orchestrator. Re-execs under `.venv`, sets `PYTHONPATH=aiter` + `TRITON_DISABLE_LINE_INFO=0` + `TRITON_ALWAYS_COMPILE=1`, configures `instrumentation_mode="gsan"`, runs the driver inside `use_mem_pool`, sniffs stderr for race lines and returns a machine-readable exit code. |
| `run_with_gsan.py` | Lower-level wrapper. Sets `instrumentation_mode="gsan"`, creates pool, runs script inside `use_mem_pool`. Auto-injects `aiter/` and the script's directory into `sys.path`. Use when you want direct control over env vars / no race sniffer. |
| `conftest.py` | Pytest session-scope autouse fixture that does the same setup as the wrapper. Picked up because `pyproject.toml` anchors pytest's rootdir here. |
| `scripts/setup_cuda_gsan.sh` | One-time installer for torch (cu128) and Triton from source. **Already run.** |
| `extracted_kernels/demo_add.py` | Working template for writing your own driver. |
| `patches/aiter-cuda-experimental.patch` | Local-only fallback to make AITER pip-installable on CUDA by skipping the ROCm raise. **Don't apply unless you've ruled out the extracted-kernel path.** Has known limits — chain bottoms out at `aiter.ops.enum`. |
| `aiter/aiter/ops/triton/` | Where to find kernels worth extracting. |
| `third_party/triton/python/triton/experimental/gsan/` | GSan implementation reference. `src/GSanAllocator.cc` and `src/GSan.h` if you need to reason about pool/shadow sizing. |

## Pitfalls / anti-patterns

- **Don't** `pip install -e ./aiter`. It will raise. Use PYTHONPATH.
- **Don't** rebuild Triton. `third_party/triton` is already built editable
  into the venv. If you accidentally `uv pip install triton`, you'll
  overwrite the source build with the wheel `triton==3.6.0`. To recover:
  `uv pip install -e third_party/triton --no-build-isolation -v`.
- **Don't** trust agent-summarized claims about AITER's setup.py logic
  without reading the lines yourself. The first explorer agent on this repo
  read the `BUILD_TARGET` branches backwards.
- **Don't** set `TRITON_INSTRUMENTATION_MODE=gsan` *after* `import triton`
  in your script and expect it to take effect — Triton caches knobs at
  import. The wrapper avoids this by setting the Python-level knob
  (`triton.knobs.compilation.instrumentation_mode = "gsan"`) directly.
- **Don't** allocate tensors *outside* the `with torch.cuda.use_mem_pool(pool):`
  block — GSan can't see them. The wrapper and conftest both put your code
  inside the block, so just don't manually bypass it.
- **Don't** scale up shapes hoping more allocations will surface races.
  Shapes that work in normal AITER tests will OOM under GSan. Start small.

## When in doubt

1. Read `SURVEY.md` for what was tried and what failed and why.
2. Read `README.md`'s "Known working environment" for last-known-good versions.
3. Check `git log` (currently a single root commit) for project history.
4. Look at `extracted_kernels/demo_add.py` as the canonical small example.
