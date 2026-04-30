"""Minimal demo driver: vector add under Triton GSan.

This is a race-free kernel — running it through run_with_gsan.py should
produce no race reports. Use as a template for extracting real kernels
out of aiter/aiter/ops/triton/.

Run:
    TRITON_DISABLE_LINE_INFO=0 TRITON_ALWAYS_COMPILE=1 \\
      python run_with_gsan.py extracted_kernels/demo_add.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    n = 4096
    BLOCK = 128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)

    torch.cuda.synchronize()
    expected = x + y
    max_err = (out - expected).abs().max().item()
    print(f"[demo_add] n={n}, max_err={max_err:.3e}")
    assert max_err < 1e-5, "numerical mismatch"
    print("[demo_add] OK (no race expected)")


if __name__ == "__main__":
    main()
