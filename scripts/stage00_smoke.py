from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.flash_attention import flash_attention_forward
from kernels.reference import naive_attention_forward
from kernels.vector_add import vector_add_reference


def main() -> None:
    print("Stage 00 smoke test")
    a = torch.randn(8)
    b = torch.randn(8)
    print("vector_add_reference:", vector_add_reference(a, b))

    if torch.cuda.is_available():
        q = torch.randn((1, 1, 32, 64), device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = flash_attention_forward(q, k, v)
        ref = naive_attention_forward(q, k, v)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print("flash_attention_forward reference check: ok")
    else:
        print("CUDA not available; skipped attention smoke")


if __name__ == "__main__":
    main()
