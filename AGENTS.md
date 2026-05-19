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
```

`run.py` will set `PYTHONPATH=benchmarks/<bench>/<bench>[+third_party/triton-viz]` for its own
subprocess; you only need PYTHONPATH manually if you're running pytest
directly (without `run.py`).

Initial install (one-time): `uv sync --extra cuda` — installs torch
(cu128 wheel), builds Triton from `third_party/triton` (editable), installs
triton-viz from `third_party/triton-viz` (editable), plus base deps. Triton
source build is ~3 min the first time. **Don't re-run it casually — a fresh
sync rebuilds Triton.**

## Verified working state

These have been run and confirmed; **don't redo them just to check**:

| Check | Command | Result |
| --- | --- | --- |
| Triton's own GSan tests | `cd third_party/triton && TRITON_DISABLE_LINE_INFO=0 pytest -n 4 python/test/gsan` | 85 passed, 11 skipped (~23s) |
| `import aiter` works on CUDA | `python -c "import aiter"` | warns, then OK (Triton ops available) |
| End-to-end orchestrator (baseline) | `python run.py --backend baseline benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py` | 40 passed, exit 0 (no instrumentation) |
| End-to-end orchestrator (gsan) | `python run.py --backend gsan benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py` | 38 passed, 2 OOM-failed, exit 1 (private-pool OOM signature) |

## Known working environment

- GPU: NVIDIA GeForce RTX 4090 (sm_89), driver 580.126.20
- Python: 3.11.13 (uv-managed)
- torch: 2.9.0+cu128 (`cuda` extra)
- triton: 3.7.0+gited8317b2 (source build, editable via `[tool.uv.sources]`)
- triton-viz: editable via `[tool.uv.sources]` (source build)
- aiter HEAD: `d295caf6b977b3b0af02a9de06722811fb529cf3`
- triton HEAD: `ed8317b20881e443aaf6c91d161cbacf6143dc53`

## Hard constraints — don't fight these

These cost real time to discover. **Trust the prior diagnosis in `SURVEY.md`.**

1. **AITER does not install on CUDA.** `benchmarks/aiter/aiter/setup.py:148` raises
   `NotImplementedError("Only ROCM is supported")`. `BUILD_TARGET=cuda` does
   *not* bypass it — it explicitly sets `IS_ROCM = False` and falls into the
   raise. Use PYTHONPATH; don't `pip install -e ./benchmarks/aiter/aiter`.

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
   init in `benchmarks/aiter/aiter/aiter/__init__.py`. Patching that chain bottoms out in
   `aiter.ops.enum` calling `build_module()` at import. Don't try to
   completely fix this; route around it.

## Entry point: `run.py`

One Python orchestrator covers all (benchmark, backend) cells. Single CLI,
no subcommands:

```bash
python run.py --backend {gsan,triton_viz,baseline} <test_file> [-- pytest_args]
```

`<test_file>` is a pytest test module (e.g. one of the 72 files in
`benchmarks/aiter/pytest_files.txt`). The orchestrator spawns
`python -m pytest <test_file>` as a subprocess; the repo-root
`conftest.py` reads `BACKEND=<name>` env and installs the matching
race-detector fixture inside the subprocess.

What `run.py` (via `run_common.py`) handles:

- Re-execs under `.venv/bin/python` if the caller used a different interpreter.
- Sets `PYTHONPATH=benchmarks/<bench>/<bench>[+third_party/triton-viz]` plus per-backend env
  (`TRITON_DISABLE_LINE_INFO=0`, `TRITON_ALWAYS_COMPILE=1` for gsan;
  `TRITON_INTERPRET=1` for triton-viz; nothing extra for baseline).
- Tees subprocess stdout+stderr through a sniffer that matches
  `(Read|Write) after (read|write) race detected` lines (start-of-line
  anchored to avoid false matches on commentary).
- Parses pytest's summary line for `passed/failed/skipped/errors`,
  with a progress-char fallback when pytest gets SIGTERM'd mid-stream.
- Writes a CSV row + raw log per run (see "Reporting outputs" below).

### Backends

| `--backend` | What it installs | When to use |
|---|---|---|
| `gsan` | Triton GSan via `triton.knobs.compilation.instrumentation_mode = "gsan"` + private CUDA mem pool (`with torch.cuda.use_mem_pool(...):` per session) | Default. Hardware-accurate but production shapes OOM in the private pool (see constraint #2). |
| `triton_viz` | Monkey-patches `triton.jit`/`triton.autotune` to wrap kernels with `triton_viz.trace(client=RaceDetector(...))`. Forces `TRITON_INTERPRET=1`. | Z3-backed symbolic check. Slow; numerical results don't write back to CUDA tensors (assertion-failure side effect is expected). |
| `baseline` | Nothing — plain Triton. `conftest.py` treats `BACKEND=baseline` as alias for `BACKEND=none`. | Control. Same test suite, race detector OFF. Useful for comparing what fails for race-detector reasons vs what fails anyway. |

### Exit codes

- `0` pytest reported all tests passed (or no tests ran)
- `1` at least one test failed OR sniffer caught a race line (check `race_count`)
- `2` collection-time error inside pytest
- `124` shell `timeout` SIGTERM'd the orchestrator (mark CSV row `notes=TIMEOUT 600`)

## Reporting outputs: `runs/<benchmark>_<backend>_pytest.csv` + `runs/logs/*.log`

Every `run.py` invocation writes:

- **CSV row** upserted into `runs/<benchmark>_<backend>_pytest.csv`
  (one CSV per benchmark+backend combo, one row per `script`). Re-running
  the same script overwrites its row; rows for other scripts in the same
  CSV are preserved. Columns: `timestamp, benchmark, backend, script,
  target_seconds, total_seconds, race_count, exit_code, passed, failed,
  skipped, errors, log_path, notes`. Open in Excel / `pandas.read_csv`
  directly.
- **Raw log** at `runs/logs/<backend>_pytest_<script-stem>.log`. Full tee
  of subprocess stdout+stderr, prefixed with a
  `===== <ISO timestamp> ... =====` header line. **Each run truncates
  its log file** — re-running the same script overwrites the previous
  log. CSV is the place that accumulates history. Pass `--log <path>` if
  you want a separate log to compare two runs side-by-side.

Overrides: `--csv PATH`, `--log PATH`, `--log-dir DIR`, `--no-csv`,
`--no-log`, `--benchmark NAME`, `--notes TEXT`. `runs/` is in `.gitignore`.

## Adding a new benchmark suite

1. `mkdir -p benchmarks/<name>/`
2. Write `benchmarks/<name>/pytest_files.txt` — one repo-relative
   `test_*.py` path per line.
3. Write `benchmarks/<name>/inventory.md` — total tests + categorized
   skip/fail/error reasons. See `benchmarks/aiter/inventory.md` as a
   template.
4. `python run.py --benchmark <name> --backend gsan <one of the test files>`
   — no new Python file required. CSV path will be
   `runs/<name>_gsan_pytest.csv`.

## Sweeping all 72 test files for a benchmark

The canonical AITER test list is `benchmarks/aiter/pytest_files.txt`
(72 lines; reproduces the 27,689-test total documented in
`benchmarks/aiter/inventory.md`). Loop through it for any backend:

```bash
source .venv/bin/activate
while read -r f; do
  echo "=== $f ==="
  timeout 600 python run.py --benchmark aiter --backend gsan "$f"
done < benchmarks/aiter/pytest_files.txt
```

For triton-viz the per-file shell cap should be longer (interpreter mode
can take 200–400s per file before hitting pytest's per-test cap). For
baseline 600s is more than enough (most files finish in seconds).

Per-file outcomes accumulate via CSV upsert at
`runs/aiter_<backend>_pytest.csv`. A run that gets killed by shell
`timeout` writes `exit_code=124` and the `finally` block may not finish
writing the CSV row — fill those in manually as
`notes=TIMEOUT 600`. Inventory expectations:

- GSan: 27,689 tests, 3,626 passed, 11,138 failed, 12,902 skipped,
  179 race lines in 4 files (rmsnorm, layernorm, two moe_gemm files).
- Baseline: similar skip counts but the GSan-private-pool OOM failures
  pass instead → higher pass rate.
- Triton-viz: many per-test timeouts under interpreter mode; treat
  `exit_code=2`/numerical mismatches as expected (see Pitfalls).

The 94% OOM-failure signature from `SURVEY.md` only applies to the GSan
backend; baseline doesn't have a private pool. Use baseline runs to
isolate "real fail from race-detector overhead" vs "fail anyway".

## Key files

| Path | Purpose |
| --- | --- |
| `run.py` | **The orchestrator.** Single CLI: `--backend {gsan,triton_viz,baseline} <test_file>`. Sets env, spawns pytest subprocess, tees stdout/stderr through a race-line sniffer, writes CSV row + log via `run_common`. |
| `run_common.py` | Shared scaffolding. Holds `BACKEND_SPECS` registry, `get_backend`, `make_pytest_runner`, env prep, race-sniffing tee, CSV upsert, the `run_with_reporting` orchestrator. Future benchmarks reuse this — no new Python file needed. |
| `conftest.py` | Pytest session-scope autouse fixture. Reads `BACKEND` env (passed in by `run.py`) and installs the right race-detector inside the pytest subprocess. `baseline` is aliased to `none` (no-op). |
| `benchmarks/aiter/pytest_files.txt` | Canonical 72-line list of pytest test files for the AITER suite. Used by the sweep loop in "Sweeping all 72 test files" above. |
| `benchmarks/aiter/inventory.md` | Human-readable AITER inventory: 27,689 tests total, broken down by passed/failed/skipped/errors plus categorized skip-reason table. Regenerate manually after a full sweep. |
| `third_party/triton-viz/` | triton-viz submodule pinned on `race-detector-z3-demo`. Editable via `[tool.uv.sources]` in `pyproject.toml`; `uv sync --extra cuda` installs. |
| `patches/aiter-cuda-experimental.patch` | Local-only fallback to make AITER pip-installable on CUDA by skipping the ROCm raise. **Don't apply unless you have a specific need.** Has known limits — chain bottoms out at `aiter.ops.enum`. |
| `benchmarks/aiter/aiter/aiter/ops/triton/` | Reference for the actual `@triton.jit` kernels under test. |
| `third_party/triton/python/triton/experimental/gsan/` | GSan implementation reference. `src/GSanAllocator.cc` and `src/GSan.h` if you need to reason about pool/shadow sizing. |

## Pitfalls / anti-patterns

- **Don't** `pip install -e ./benchmarks/aiter/aiter`. It will raise. Use PYTHONPATH.
- **Don't** `uv pip install triton` manually. The project pins triton to
  `third_party/triton` via `[tool.uv.sources]` + `override-dependencies`
  in `pyproject.toml` (otherwise the torch wheel's bundled `triton==3.5.0`
  wins and `triton.experimental.gsan` disappears). `uv sync --extra cuda`
  is the only supported install path.
- **Don't** trust agent-summarized claims about AITER's setup.py logic
  without reading the lines yourself. The first explorer agent on this repo
  read the `BUILD_TARGET` branches backwards.
- **Don't** treat the numerical mismatch under `--backend triton_viz` as a
  regression. `TRITON_INTERPRET=1` + RaceDetector doesn't write results
  back to CUDA tensors; assertion-failure side effects are expected. Compare
  `race_count` and timing only.
- **Don't** scale up shapes hoping more allocations will surface races.
  Production-shape tests OOM under GSan's private pool. Use `--backend
  baseline` to see what happens without the private-pool overhead.

## When in doubt

1. Read `benchmarks/aiter/inventory.md` for the canonical breakdown of
   what passes / fails / skips and why.
2. Read `SURVEY.md` for what was tried before and how the constraints
   were discovered.
3. Read `README.md`'s "Known working environment" for last-known-good versions.
4. Check `git log` for project history.
5. Run one test file end-to-end:
   `python run.py --backend baseline benchmarks/aiter/aiter/op_tests/triton_tests/test_topk.py`
   then compare with `--backend gsan` and `--backend triton_viz`.
