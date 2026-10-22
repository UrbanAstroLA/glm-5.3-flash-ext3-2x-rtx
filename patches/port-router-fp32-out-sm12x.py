#!/usr/bin/env python3
"""Use the fp32-output router GEMM on SM12x (vLLM PR #54048, commit b5707bf99).

GLM-5.3-Flash asks for fp32 router logits (``out_dtype=float32``). GateLinear
enables its fused bf16 x bf16 -> fp32 tier (``torch.mm(..., out_dtype=fp32)``)
only through the specialized-kernel gate (SM90 / SM100) or on ROCm, so on
compute capability 12.x (RTX PRO 6000 / RTX 50xx, GB10) the router runs bf16
``F.linear`` and casts the bf16 result to fp32. Upstream made the tier
available on any CUDA or ROCm device for a bias-free bf16 router with fp32
output. This file equals the vLLM base the image was built from, so the upstream
change applies unchanged.
Usage: port-router-fp32-out-sm12x.py VLLM_DIR
"""

import os
import sys
import tempfile
from pathlib import Path

vllm = Path(sys.argv[1])
PENDING: list[tuple[Path, str]] = []


def patch(rel: str, edits: list[tuple[str, str, int]], marker: str) -> None:
    """Validate every edit before anything is written (see the end of the file)."""
    path = vllm / rel
    source = path.read_text()
    if marker in source:
        print(f"{rel}: already applied")
        return
    for old, new, count in edits:
        if source.count(old) != count:
            raise SystemExit(f"{rel}: refusing, source drift ({source.count(old)} != {count}):{chr(10)}{old}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    PENDING.append((path, source))


patch('model_executor/layers/fused_moe/router/gate_linear.py', [
    (
"""
        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk.
        # cuBLAS on CUDA (SM90+, via allow_specialized_router_gemm); hipBLASLt on
        # ROCm, which supports the same out_dtype epilogue.
        self._router_gemm_no_bias = not bias
        self.allow_cublas_router_gemm = (
            (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
""",
"""
        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk. This is
        # the plain cuBLAS (CUDA) / hipBLASLt (ROCm) out_dtype epilogue, so it
        # applies on any CUDA-alike device (no bias, since torch.mm has no bias
        # term). The specialized-kernel gate above excludes family-120 Blackwell
        # (GB10 / DGX Spark), which this tier still covers. See #49921.
        self._router_gemm_no_bias = not bias
        self._router_gemm_cublas_capable = (
            current_platform.is_cuda() or current_platform.is_rocm()
        ) and self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
""",
        1,
    ),
    (
"""
        if (
            not self.allow_cublas_router_gemm
            and (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16
""",
"""
        if (
            not self.allow_cublas_router_gemm
            and self._router_gemm_cublas_capable
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16
""",
        1,
    ),
], marker='_router_gemm_cublas_capable')



for path, source in PENDING:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".port-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(source)
        os.chmod(tmp, os.stat(path).st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

print('MoE router fp32-output GEMM enabled on SM12x (vLLM #54048)')
