#!/usr/bin/env python3
"""CPU regression gate: GateLinear uses the fp32-output router GEMM on SM12x (port-router-fp32-out-sm12x.py).

Builds GateLinear(out_dtype=float32) under an emulated CUDA platform reporting compute capability 12.0, 12.1
and 9.0 (no GPU, no model, no network) and reads which GEMM tier it selects. Dispatch only: the numerics of
torch.mm(out_dtype=float32) are not exercised here.

Run inside the recipe image:
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$PWD/scripts:/s:ro" \
    --entrypoint python3 IMAGE /s/test-router-fp32-out-sm12x.py [--expect-unpatched] [--output report.json]
--expect-unpatched reproduces the omission on v0.9.1; the default checks the port.
"""
import argparse
import json
from types import SimpleNamespace as NS

import torch

import vllm.distributed.parallel_state as ps
import vllm.model_executor.layers.fused_moe.router.gate_linear as gl
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
ap.add_argument("--expect-unpatched", action="store_true")
ap.add_argument("--output", default="")
args = ap.parse_args()
PATCHED = not args.expect_unpatched
RESULTS = []


def check(name, holds, detail=""):
    met = bool(holds) == PATCHED
    RESULTS.append(dict(name=name, holds=bool(holds), met=met, detail=detail))
    print(f"{'PASS' if met else 'FAIL'} {name}: {'holds' if holds else 'does not hold'} {detail}")


ps.get_tp_group = lambda: NS(rank_in_group=0, world_size=1)
try:
    import vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 as llb
    llb.is_available = lambda: False
except Exception:
    pass


def fp32_out(cap):
    class EmulatedCuda(Platform):
        _enum = PlatformEnum.CUDA
        device_type = "cuda"

        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(*cap)

    gl.current_platform = EmulatedCuda()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return gl.GateLinear(4096, 288, out_dtype=torch.float32, prefix="gate").allow_cublas_router_gemm
    finally:
        torch.set_default_dtype(prev)


caps = {c: fp32_out(c) for c in ((12, 0), (12, 1), (9, 0))}
check("router: GateLinear(out_dtype=fp32) uses the fp32-output GEMM on SM120 and SM121",
      caps[(12, 0)] and caps[(12, 1)], f"(SM120 {caps[(12, 0)]}, SM121 {caps[(12, 1)]}, SM90 {caps[(9, 0)]})")

ok = all(r["met"] for r in RESULTS)
print(f"{'ALL MET' if ok else 'NOT MET'}: {sum(r['met'] for r in RESULTS)}/{len(RESULTS)} "
      f"({'port expected' if PATCHED else 'v0.9.1 omission expected'})")
if args.output:
    json.dump(dict(expect="patched" if PATCHED else "unpatched", results=RESULTS), open(args.output, "w"), indent=1)
raise SystemExit(0 if ok else 1)
